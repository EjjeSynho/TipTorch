import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))

import gc
import warnings
import logging
from typing import Callable, Optional
import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np
import torch.optim as optim

from tiptorch.tools.cubic_splines import natural_cubic_spline_coeffs, NaturalCubicSpline
from data_processing.MUSE_data_utils import filter_dataframe, reduce_dataframe, TELEMETRY_CACHE
from tiptorch.managers.input_manager import InputsTransformer

from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from tiptorch._config import *


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmallCalibratorNet(nn.Module):
    """
    Compact neural network for calibration with strong regularization.
    Designed for small datasets (~500 samples).
    
    Args:
        n_features: Number of input features
        n_outputs: Number of output values
        hidden_dim: Size of hidden layers (default: 32)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, n_features, n_outputs, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for small batches
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_dim, n_outputs),
        )
        
        # Initialize weights with smaller values for better regularization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


def release_gpu_memory(sync=False):
    """Best-effort VRAM cleanup without affecting model state."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()


class NFMCalibrator():
    def __init__(
        self,
        checkpoint_path,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.device = device
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Initialize NN architecture from saved state
        self.net = SmallCalibratorNet(
            n_features   = state['net_arch']['n_features'],
            n_outputs    = state['net_arch']['n_outputs'],
            hidden_dim   = state['net_arch']['hidden_dim'],
            dropout_rate = state['net_arch']['dropout_rate'],
        ).to(device)
        
        self.net.load_state_dict(state['net_state_dict'])
        self.net.eval()
        self.net.to(device)    

        # Outputs transformer — stored as a plain dict from InputsTransformer.save()
        raw = state['outputs_transformer']
        self.outputs_transformer = InputsTransformer.load(data=raw)
        
        # Inputs transformers
        with open(TELEMETRY_CACHE / 'MUSE/muse_telemetry_imputer.pickle', 'rb') as handle:
            self.telemetry_imputer = pickle.load(handle)
        self.telemetry_imputer.verbose = 0

        with open(TELEMETRY_CACHE / 'MUSE/muse_telemetry_scaler.pickle', 'rb') as handle:
            self.telemetry_scaler = pickle.load(handle)
            self.features = self.telemetry_scaler.feature_names_in_

        # Aux params
        self.LO_modes_max   = state.get('LO_modes_max', None)
        self.N_spline_nodes = state.get('N_spline_nodes', None)


    def prepare_telemetry(self, reduced_telemetry: pd.DataFrame) -> torch.Tensor:
        telemetry_pruned = reduce_dataframe(filter_dataframe(reduced_telemetry))
        # Standartize/normalize features
        telemetry_ = self.telemetry_scaler.transform(telemetry_pruned[self.features])
        # Restore feature names back to avoid warnings
        telemetry_ = pd.DataFrame(telemetry_, columns=self.features, index=telemetry_pruned.index)
        # Impute missing values
        telemetry_ = self.telemetry_imputer.transform(telemetry_)
        return torch.tensor(telemetry_, dtype=torch.float32, device=self.device)


    def forward(self, reduced_telemetry: pd.DataFrame):
        # Prepare the telemetry data: filter, reduce, scale, and impute
        telemetry_vector = self.prepare_telemetry(reduced_telemetry)
    
        # Pass through the network to get predictions for the PSF model inputs
        x_pred = self.net(telemetry_vector)
        x_dict_pred = self.outputs_transformer.unstack(x_pred)

        # Prepend a zeros column as the phase-bump placeholder (not predicted by the net)
        if 'LO_coefs' in x_dict_pred:
            coefs = x_dict_pred['LO_coefs']
            phase_bump = torch.zeros(coefs.shape[0], 1, device=coefs.device, dtype=coefs.dtype)
            x_dict_pred['LO_coefs'] = torch.cat([phase_bump, coefs], dim=-1)

        return x_dict_pred


    def __call__(self, reduced_telemetry: pd.DataFrame):
        return self.forward(reduced_telemetry)


    @torch.no_grad()
    def calibrate(self, reduced_telemetry: pd.DataFrame, PSF_model) -> None:
        """
        Method to get a reconciled parameter dict ready for PSFModelNFM.
        Checks compatibility and adapts the predicted parameters to match the PSF model's expected input structure.
        """
        # Do parameters dict prediction
        x_dict_pred = self.forward(reduced_telemetry)
        # If PSF modle has slightly different settings, adapt the predicted parameters accordingly (e.g. by cropping or padding with zeros)
        x_dict_adapted = self._adapt_to_PSF_model(x_dict_pred, PSF_model)
        # Update PSF model's internal PSF model inputs with the predicted parameters
        PSF_model.update_manager_params(x_dict_adapted)
        # Trigger PSF update with new parameters
        _ = PSF_model()


    def check_compatibility(self, PSF_model) -> dict:
        """
        Check structural compatibility between the calibrator and a PSFModelNFM instance.
        Issues warnings for each mismatch.
        """
        issues = {}

        if self.LO_modes_max is not None and hasattr(PSF_model, 'Z_mode_max'):
            n_calib = self.LO_modes_max - 2  # modes predicted (phase bump excluded)
            n_model = PSF_model.Z_mode_max - 2
            if n_calib != n_model:
                issues['LO_modes'] = (n_calib, n_model)
                action = 'cropping' if n_calib > n_model else 'padding with zeros'
                warnings.warn(
                    f"NFMCalibrator: LO modes mismatch — calibrator predicts {n_calib} "
                    f"Zernike modes, PSF model expects {n_model}. Will adapt by {action}."
                )

        if self.N_spline_nodes is not None and hasattr(PSF_model, 'N_wvl_ctrl'):
            n_calib = self.N_spline_nodes
            n_model = PSF_model.N_wvl_ctrl
            if n_calib != n_model:
                issues['N_ctrl'] = (n_calib, n_model)
                warnings.warn(
                    f"NFMCalibrator: N_spline_nodes mismatch — calibrator has {n_calib} "
                    f"control points, PSF model has {n_model}. Will re-sample via cubic spline."
                )
        
        return issues


    @staticmethod
    def _resample_spline_ctrls(values: torch.Tensor, n_to: int) -> torch.Tensor:
        """
        Re-sample spline control-point values to a different number of uniformly-spaced
        nodes in [0, 1] using natural cubic splines.
        """
        n_from = values.shape[-1]
        t_from = torch.linspace(0, 1, n_from, device=values.device, dtype=values.dtype)
        t_to   = torch.linspace(0, 1, n_to,   device=values.device, dtype=values.dtype)
        # torchcubicspline expects [n_nodes, batch] for x
        coeffs = natural_cubic_spline_coeffs(t_from, values.T)
        spline = NaturalCubicSpline(coeffs)
        return spline.evaluate(t_to).T  # [batch, n_to]


    def _adapt_to_PSF_model(self, x_dict: dict, PSF_model) -> dict:

        x_dict = dict(x_dict)  # shallow copy, tensors are not cloned

        # Spline control points
        if self.N_spline_nodes is not None and hasattr(PSF_model, 'N_wvl_ctrl'):
            n_ctrl_calib = self.N_spline_nodes
            n_ctrl_model = PSF_model.N_wvl_ctrl
            if n_ctrl_calib != n_ctrl_model:
                for key, val in x_dict.items():
                    if (
                        key.endswith('_ctrl')
                        and isinstance(val, torch.Tensor)
                        and val.dim() == 2
                        and val.shape[-1] == n_ctrl_calib
                    ):
                        x_dict[key] = self._resample_spline_ctrls(val, n_ctrl_model)

        # LO modes (first column = phase bump, already included)
        if 'LO_coefs' in x_dict and hasattr(PSF_model, 'Z_mode_max'):
            n_model = PSF_model.Z_mode_max - 1  # total modes the model needs (incl. phase bump)
            coefs   = x_dict['LO_coefs']
            n_have  = coefs.shape[-1]
            
            if n_have > n_model:
                x_dict['LO_coefs'] = coefs[:, :n_model]
                
            elif n_have < n_model:
                pad = torch.zeros(coefs.shape[0], n_model - n_have, device=coefs.device, dtype=coefs.dtype)
                x_dict['LO_coefs'] = torch.hstack((coefs, pad))

        return x_dict



class NFMCalibratorTrainer:
    """
    PSF-model-in-the-loop trainer for an arbitrary NN calibrator.

    Encapsulates the full training workflow:
      - wavelength-batched PSF forward passes
      - pretraining against fitted parameter values
      - end-to-end PSF loss training with optional per-sample difficulty weighting
      - K-fold difficulty estimation via out-of-fold losses + random-forest difficulty model
      - targeted fine-tuning of selected output dimensions
      - checkpoint save / load / calibrator bundle export

    Parameters
    ----------
    PSF_model : PSFModelNFM
        Initialised (and pruned) PSF model.
    calibrator : nn.Module
        Network mapping a [B, N_features] telemetry tensor -> [B, N_outputs] stacked params.
    dataset : NFMDataset (or compatible)
        Full labelled dataset. Split internally via train_test_split.
    outputs_transformer : InputsTransformer
        Knows how to stack / unstack the calibrator output vector.
    device : torch.device
    batch_size, lr, lambda_step : int / float / int
        Basic training hyper-parameters.
    predict_LO_NCPAs, predict_Cn2_profile : bool
    pre_init_astrometry : bool
        Seed per-sample dx/dy from dataset fitted values instead of zeros.
    optimize_astrometry : bool
        Include per-sample dx/dy in the optimizer.
    weights_dir : Path | None
        Directory for checkpoints / bundles (default: WEIGHTS_FOLDER/NFM_calibrator).
    test_size : float
        Fraction of samples held out for validation.
    random_state : int
    """

    def __init__(
        self,
        PSF_model,
        calibrator,
        dataset,
        outputs_transformer,
        device,
        *,
        batch_size           = 16,
        lr                   = 1e-2,
        lambda_step          = 2,
        predict_LO_NCPAs     = True,
        predict_Cn2_profile  = True,
        pre_init_astrometry  = True,
        optimize_astrometry  = False,
        weights_dir          = None,
        test_size            = 0.20,
        random_state         = 42,
        collate_fn: Optional[Callable] = None,
    ):
        self.PSF_model           = PSF_model
        self.calibrator          = calibrator
        self.dataset             = dataset
        self.outputs_transformer = outputs_transformer
        self.device              = device
        self.batch_size          = batch_size
        self.lr                  = lr
        self.predict_LO_NCPAs    = predict_LO_NCPAs
        self.predict_Cn2_profile = predict_Cn2_profile
        self.pre_init_astrometry = pre_init_astrometry
        self.optimize_astrometry = optimize_astrometry
        self.collate_fn          = collate_fn
        self.weights_dir         = Path(weights_dir) if weights_dir else WEIGHTS_FOLDER / 'NFM_calibrator'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.N_wvl_total = dataset.C
        self.H, self.W   = dataset.H, dataset.W
        self.N_features  = len(dataset.features)
        self.N_outputs   = outputs_transformer.get_stacked_size()

        # Train / val split and loaders
        self.train_idx, self.val_idx = train_test_split(
            np.arange(len(dataset)), test_size=test_size, random_state=random_state,
        )
        self.train_loader = self._make_loader(self.train_idx, shuffle=True)
        self.val_loader   = self._make_loader(self.val_idx,   shuffle=False)

        # Wavelength scheduling
        self.lambda_full    = PSF_model.λ_sim.clone()
        self.lambda_id_sets = [list(range(dataset.C))] if lambda_step == 1 else self._build_wavelength_sets(lambda_step, dataset.C-1)
        self.lambda_sets    = [self.lambda_full[ids] for ids in self.lambda_id_sets]

        # Astrometry
        self._init_astrometry()

        # Phase-bump (first LO coef) and optional NCPA median fallback
        LO_data = np.array([d['LO_coefs'] for d in dataset.fitted_vals], dtype=np.float32)
        self.phase_bump   = torch.tensor(LO_data[:, 0], device=device, dtype=torch.float32)
        self.NCPAs_median = (
            None if predict_LO_NCPAs else
            torch.tensor(np.median(LO_data[:, 1:], axis=0)[None, ...], device=device, dtype=torch.float32)
        )

        self.optimizer, self.scheduler = self._make_optimizer()

        logger.info(
            f"Trainer ready | dataset={len(dataset)} ({len(self.train_idx)} train / {len(self.val_idx)} val) | "
            f"N_features={self.N_features} | N_outputs={self.N_outputs} | "
            f"lambda_sets={len(self.lambda_id_sets)} x ~{len(self.lambda_id_sets[0])} wvl"
        )


    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_wavelength_sets(step, max_val):
        sets, thresh = [], step // 2
        for offset in range(step):
            ids = list(range(offset, max_val + 1, step))
            if ids[0] > thresh:            ids.insert(0, 0)
            if max_val - ids[-1] > thresh: ids.append(max_val)
            sets.append(ids)
        return sets


    def _make_loader(self, indices, shuffle):
        collate_fn = self.collate_fn
        if collate_fn is not None:
            collator = lambda b: collate_fn(b, device=self.device)
        elif hasattr(self.dataset, 'collate_batch'):
            collator = lambda b: self.dataset.collate_batch(b, device=self.device)
        else:
            raise ValueError(
                "NFMCalibratorTrainer requires a collate function. "
                "Pass collate_fn=... or define dataset.collate_batch(batch, device)."
            )

        return DataLoader(
            dataset=Subset(self.dataset, [int(i) for i in indices]),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collator,
            drop_last=False,
        )


    def _init_astrometry(self):
        n      = len(self.dataset)
        n_ctrl = len(np.atleast_1d(self.dataset.fitted_vals[0].get('dx_ctrl', [0])))
        if self.pre_init_astrometry and 'dx_ctrl' in self.dataset.fitted_vals[0]:
            dx_data = np.array([d['dx_ctrl'] for d in self.dataset.fitted_vals], dtype=np.float32)
            dy_data = np.array([d['dy_ctrl'] for d in self.dataset.fitted_vals], dtype=np.float32)
        else:
            self.optimize_astrometry = True
            dx_data = np.zeros((n, n_ctrl), dtype=np.float32)
            dy_data = np.zeros((n, n_ctrl), dtype=np.float32)
            
        self.dx = torch.tensor(dx_data, device=self.device, dtype=torch.float32, requires_grad=self.optimize_astrometry)
        self.dy = torch.tensor(dy_data, device=self.device, dtype=torch.float32, requires_grad=self.optimize_astrometry)


    def _make_optimizer(self, lr=None):
        lr     = lr or self.lr
        groups = [{'params': self.calibrator.parameters(), 'lr': lr, 'weight_decay': 5e-4}]
        if self.optimize_astrometry:
            groups.append({'params': [self.dx, self.dy], 'lr': 1e-3, 'weight_decay': 1e-5})
            
        opt = optim.AdamW(groups, lr=lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=7, min_lr=1e-6)
        return opt, sch


    # ── PSF forward ───────────────────────────────────────────────────────────
    def run_model(self, x_dict_NN, config, idx, lambda_ids):
        """Run the PSF model for a batch at the given wavelength indices."""
        # NOTE: idx are dataset ids (global sample ids), not per-batch local offsets.
        self.PSF_model.model.N_obs = len(idx)
        x_dict = self.PSF_model.inputs_manager.to_dict()
        # Update only parameters predicted by the calibrator and keep the rest from manager defaults.
        x_dict.update({k: v for k, v in x_dict_NN.items() if k in x_dict})
        # Astrometric shifts are stored as full-dataset tensors and indexed per current batch.
        x_dict['dx_ctrl'] = self.dx[idx]
        x_dict['dy_ctrl'] = self.dy[idx]

        # Re-attach the phase bump as the first LO coefficient.
        bump = self.phase_bump[idx].unsqueeze(-1)
        ncpa = x_dict['LO_coefs'] if self.predict_LO_NCPAs else self.NCPAs_median.expand(len(idx), -1)
        x_dict['LO_coefs'] = torch.hstack((bump, ncpa))

        # Predict only a subset of wavelengths at once to reduce peak GPU memory.
        wvl = self.lambda_full[lambda_ids].to(device=self.device)
        config['sources_science']['Wavelength'] = wvl.view(1, -1)
        self.PSF_model.model.config = config
        
        # Update TipTorch internal config to match the new wavelength set. It will trigger the iternal update
        # of the PSF model state. But if it's only one wavelengths set, then trigger the update manually.
        # Grids don't need to be updated since they are defined at the full wavelength resolution, and the PSF model will slice them accordingly.
        # Pupils don't depend on wavelength, so no need to update them either.
        if (len(self.lambda_id_sets) == 1):
            self.PSF_model.model.Update(grids=False, pupils=False, tomography=True)
        else:
            self.PSF_model.SetWavelengths(wvl)
        
        # Do not update inputs_manager with graph-connected tensors to avoid cross-batch graph retention.
        return self.PSF_model(x_dict, update_params=False)


    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss_per_sample(self, PSF_data, PSF_pred, x_dict_pred):
        diff = PSF_pred - PSF_data
        # Empirical scaling to keep PSF reconstruction and regularization terms numerically balanced.
        w = 2e4
        # Per-sample PSF reconstruction loss.
        PSF = w * (diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0 + diff.abs().mean(dim=(1, 2, 3)) * 1.6)
        if self.predict_LO_NCPAs and 'LO_coefs' in x_dict_pred:
            c  = x_dict_pred['LO_coefs']
            # L2 regularization of predicted LO modes with sign penalties used in legacy training.
            LO = c.pow(2).sum(-1) * 1e-7
            LO = LO + torch.clamp(-c[:, 0], min=0).pow(2) * 5e-5
            LO = LO + torch.clamp(-c[:, 2], min=0).pow(2) * 1e-7
        else:
            LO = torch.zeros(PSF_data.shape[0], device=PSF_data.device, dtype=PSF_data.dtype)
        return PSF + LO


    def loss_fn(self, PSF_data, PSF_pred, x_dict_pred, sample_weights=None, return_per_sample=False):
        per = self._loss_per_sample(PSF_data, PSF_pred, x_dict_pred)
        if return_per_sample:
            return per
        
        if sample_weights is None:
            return per.mean()
        
        w = sample_weights.to(dtype=per.dtype, device=per.device).view(-1)
        return (w * per).sum() / (w.sum() + 1e-12)


    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        data = {
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'calibrator_state_dict': self.calibrator.state_dict(),
            'optimizer_state_dict':  self.optimizer.state_dict(),
        }
        if self.optimize_astrometry:
            data['dx'] = self.dx.detach().clone()
            data['dy'] = self.dy.detach().clone()
            
        torch.save(data, path)
        logger.debug(f"Checkpoint saved -> {path}")


    def load_checkpoint(self, path, load_optimizer=True):
        path = Path(path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return None, None, None
        ckpt = torch.load(path, map_location=self.device)
        self.calibrator.load_state_dict(ckpt['calibrator_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.optimize_astrometry and 'dx' in ckpt:
            self.dx.data = ckpt['dx'].to(self.device)
            self.dy.data = ckpt['dy'].to(self.device)
        logger.info(f"Checkpoint loaded <- {path} (epoch {ckpt.get('epoch', '?')})")
        return ckpt.get('epoch'), ckpt.get('train_loss'), ckpt.get('val_loss')


    # ── Pretrain against fitted values ────────────────────────────────────────
    def _calibrator_friendly_fit_dict(self, fitted_vals):
        """Adapt fitted_vals dict to calibrator output structure (drop phase bump, reconstruct Cn2 GL).

        Keys are returned in the exact order of ``outputs_transformer.slices`` so that any
        downstream ``stack()`` call rebuilds slices in the same order as the original setup.
        """
        # Iterate in SLICE order (not fitted_vals order) to keep the layout stable.
        keys = list(self.outputs_transformer.slices.keys())
        x = {k: fitted_vals[k] for k in keys if k in fitted_vals}
        if 'LO_coefs' in x:
            x['LO_coefs'] = x['LO_coefs'][:, 1:]   # strip phase bump; not predicted
        if 'Cn2_weights' in x:
            # Dataset stores Cn2 profile without the GL layer; reconstruct it for full-profile supervision.
            cn2 = x['Cn2_weights'].clamp(min=1e-6)
            GL  = (1.0 - cn2.sum(-1, keepdim=True)).clamp(min=1e-6)
            x['Cn2_weights'] = torch.hstack((GL, cn2))
        return x


    def _pretrain_target_vector(self, fitted_vals, device):
        """Build the supervised pretraining target vector WITHOUT modifying outputs_transformer.slices.

        Unlike calling ``outputs_transformer.stack()``, this method applies each parameter's
        forward transform in the EXISTING slice order and writes the result into a pre-allocated
        output tensor.  Missing keys (params absent from fitted_vals) are left as zero, which is
        the neutral value in the normalised space used by most transforms here.

        This prevents the ``stack()`` side-effect that resets ``slices`` to whatever key order
        ``fitted_vals`` happens to have, which would corrupt subsequent ``unstack`` calls.
        """
        x = self._calibrator_friendly_fit_dict(fitted_vals)
        B = next(iter(x.values())).shape[0]
        y = torch.zeros(B, self.N_outputs, device=device, dtype=torch.float32)
        for key, sl in self.outputs_transformer.slices.items():
            if key not in x:
                continue
            val = x[key]
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            y[:, sl] = self.outputs_transformer.transforms[key](val)
        return y


    def pretrain(self, num_epochs=100, patience=15, lr=None, save_path=None):
        """
        Pretrain the calibrator to predict fitted parameter values (no PSF model involved).
        Provides a warm start before end-to-end PSF training.
        """
        lr        = lr or self.lr
        save_path = save_path or (self.weights_dir / 'pretrain_weights.pth')
        opt       = optim.AdamW(self.calibrator.parameters(), lr=lr, weight_decay=5e-4)
        sch       = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-4)
        criterion = nn.MSELoss()
        best_loss, best_state, pat_count = float('inf'), None, 0

        for epoch in range(num_epochs):
            self.calibrator.train()
            train_loss = 0.0
            for _, tel, fit, _, _ in self.train_loader:
                opt.zero_grad()
                y = self._pretrain_target_vector(fit, device=tel.device)
                loss = criterion(self.calibrator(tel), y)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            self.calibrator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, tel, fit, _, _ in self.val_loader:
                    y = self._pretrain_target_vector(fit, device=tel.device)
                    val_loss += criterion(self.calibrator(tel), y).item()

            avg_t, avg_v = train_loss / len(self.train_loader), val_loss / len(self.val_loader)
            sch.step(avg_v)
            is_best = avg_v < best_loss
            if is_best:
                best_loss, best_state, pat_count = avg_v, deepcopy(self.calibrator.state_dict()), 0
            else:
                pat_count += 1

            print(f"\rPretrain {epoch+1:3d}/{num_epochs} | "
                  f"train={avg_t:.4f} val={avg_v:.4f} lr={opt.param_groups[0]['lr']:.2e}"
                  + (" *" if is_best else ""), end='', flush=True)
            if pat_count >= patience:
                break

        print()
        if best_state:
            self.calibrator.load_state_dict(best_state)
        torch.save(best_state, save_path)
        logger.info(f"Pretrain complete. Best val={best_loss:.6f}, saved -> {save_path}")


    # ── Validate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader=None, return_cubes=False, verbose=False):
        """
        Evaluate the calibrator on a data loader.

        Returns avg_loss (scalar) or, when return_cubes=True,
        (PSFs_pred, PSFs_data, global_ids, NN_predictions, telemetry, avg_loss).
        """
        loader = loader or self.val_loader
        self.calibrator.eval()

        N = len(loader.dataset)
        if isinstance(loader.dataset, Subset):
            local_to_global = torch.tensor(loader.dataset.indices, dtype=torch.long)
        else:
            local_to_global = torch.arange(N, dtype=torch.long)
        global_to_local = {int(g): l for l, g in enumerate(local_to_global.tolist())}

        PSFs_pred = PSFs_data = NN_pred = tel_vecs = None
        if return_cubes:
            # Validation can return full cubes for diagnostics; tensors are kept on CPU by default.
            PSFs_pred = torch.zeros((N, self.N_wvl_total, self.H, self.W))
            PSFs_data = torch.zeros((N, self.N_wvl_total, self.H, self.W))
            NN_pred   = torch.zeros((N, self.N_outputs))
            tel_vecs  = torch.zeros((N, self.N_features))

        total_loss, total_batches = 0.0, 0
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            # Map global dataset ids to local positions in this loader split.
            lpos   = torch.tensor([global_to_local[int(i)] for i in idxs_b.cpu().tolist()], dtype=torch.long)
            x_pred = self.calibrator(tel_b)
            x_dict = self.outputs_transformer.unstack(x_pred)

            if return_cubes:
                NN_pred [lpos] = x_pred.cpu()
                tel_vecs[lpos] = tel_b.cpu()

            # Inner loop over wavelength subsets to limit memory use and keep lambda coverage complete.
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b   = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                total_loss  += self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict).item()
                total_batches += 1
                if return_cubes:
                    for wi, li in enumerate(lambda_ids):
                        PSFs_pred[lpos, li] = PSF_pred_b[:, wi].cpu()
                        PSFs_data[lpos, li] = PSF_data_b[:, li].cpu()

            if verbose:
                logger.info(f"Validated batch idxs={idxs_b.tolist()}")

        avg_loss = total_loss / max(total_batches, 1)
        if return_cubes:
            return PSFs_pred, PSFs_data, local_to_global, NN_pred, tel_vecs, avg_loss
        return avg_loss


    # ── Train ─────────────────────────────────────────────────────────────────
    def train(
        self,
        num_epochs         = 500,
        patience           = 20,
        nan_recovery       = True,
        max_nan_recoveries = 10,
        sample_weights     = None,
        checkpoint_path    = None,
        train_loader       = None,
        val_loader         = None,
    ):
        """
        End-to-end PSF-model training loop with early stopping, NaN recovery,
        and optional per-sample difficulty weighting.

        Returns (train_losses, val_losses) lists over completed epochs.
        """
        train_loader    = train_loader or self.train_loader
        val_loader      = val_loader   or self.val_loader
        checkpoint_path = Path(checkpoint_path or (self.weights_dir / 'best_calibrator_checkpoint.pth'))

        if sample_weights is not None:
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)

        best_val, patience_counter, nan_count = float('inf'), 0, 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.calibrator.train()
            epoch_loss, n_batches, nan_this_epoch = 0.0, 0, False

            for PSF_data_b, tel_b, _, config_b, idxs_b in train_loader:
                self.optimizer.zero_grad()
                batch_loss = 0.0

                # Backpropagate per wavelength subset to reduce peak graph memory.
                for lambda_ids in self.lambda_id_sets:
                    x_pred     = self.calibrator(tel_b)
                    x_dict     = self.outputs_transformer.unstack(x_pred)
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    sw         = sample_weights[idxs_b] if sample_weights is not None else None
                    loss       = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, sw)

                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_this_epoch = True
                        # Legacy NaN-recovery path: reload best checkpoint and decrease LR.
                        if nan_recovery and nan_count < max_nan_recoveries:
                            nan_count += 1
                            logger.error(f"NaN at epoch {epoch}, recovery {nan_count}/{max_nan_recoveries}")
                            self.load_checkpoint(checkpoint_path)
                            for pg in self.optimizer.param_groups:
                                pg['lr'] *= 0.7
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            raise ValueError(f"NaN loss after {nan_count} recovery attempt(s)")
                        break

                    loss.backward()
                    batch_loss += loss.item()
                    del PSF_pred_b, x_pred, x_dict, loss

                if nan_this_epoch:
                    break

                # Gradient clipping helps keep training stable for both NN and optional astrometry tensors.
                torch.nn.utils.clip_grad_norm_(self.calibrator.parameters(), max_norm=1.0)
                if self.optimize_astrometry:
                    torch.nn.utils.clip_grad_norm_([self.dx, self.dy], max_norm=10.0)
                self.optimizer.step()
                epoch_loss += batch_loss
                n_batches  += 1

            release_gpu_memory()
            if nan_this_epoch:
                continue #  I have become comfortably NaN
            
            avg_train = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            val_loss = self.validate(loader=val_loader)
            val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            is_best = val_loss < best_val
            if is_best:
                best_val, patience_counter = val_loss, 0
                self.save_checkpoint(checkpoint_path, epoch, avg_train, val_loss)
            else:
                patience_counter += 1

            lr_cur = self.optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch:4d} | train={avg_train:.4f} val={val_loss:.4f} "
                   f"lr={lr_cur:.2e} pat={patience_counter}/{patience}" + (" *" if is_best else ""))
            print(msg)
            logger.info(msg)

            if epoch % 5 == 0:
                self.save_checkpoint(
                    self.weights_dir / f'checkpoint_epoch_{epoch}.pth', epoch, avg_train, val_loss
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.load_checkpoint(checkpoint_path)
        logger.info(f"Training complete. Best val={best_val:.6f}")
        return train_losses, val_losses


    # ── K-fold difficulty weighting ───────────────────────────────────────────
    def run_kfold_difficulty_weighting(
        self,
        base_indices,
        initial_state_dict,
        n_splits      = 5,
        fold_epochs   = 120,
        fold_patience = 12,
        alpha         = 0.30,
        clip          = (0.5, 2.0),
        random_state  = 42,
        force_retrain = False,
        load_only     = False,
    ):
        """
        Estimate out-of-fold PSF losses, fit a random-forest difficulty model, and return dataset-length sample weights
        for the final training run. Weights help to focus subsequent training more on poorly predicted samples.

        Only samples in base_indices are weighted; all others receive weight 1.0.
        The calibrator and optimizer are reset to initial_state_dict afterwards.

        Parameters
        ----------
        force_retrain : bool
            If True, retrain all folds even if checkpoints exist.
        load_only : bool
            If True, only load existing fold checkpoints; raise error if any are missing.
        """
        base_indices = np.asarray(base_indices, dtype=int)
        kf         = KFold(n_splits=min(n_splits, len(base_indices)), shuffle=True, random_state=random_state)
        oof_losses = np.full(len(self.dataset), np.nan, dtype=np.float32)
        kfold_dir  = self.weights_dir / 'kfold_difficulty'
        kfold_dir.mkdir(exist_ok=True)

        # Determine which folds need to be trained vs. loaded
        fold_checkpoints = [kfold_dir / f'fold_{fold_id:02d}_best.pth' for fold_id in range(min(n_splits, len(base_indices)))]
        folds_exist = [ckpt.exists() for ckpt in fold_checkpoints]
        n_folds = len(fold_checkpoints)

        if load_only:
            # User explicitly requested load-only mode; verify all folds exist
            missing_folds = [fold_id for fold_id, exists in enumerate(folds_exist) if not exists]
            if missing_folds:
                raise FileNotFoundError(
                    f"load_only=True but checkpoints missing for folds: {missing_folds}. "
                    f"Set force_retrain=True to train them, or use load_only=False for default behavior."
                )
            logger.info(f"Loading {n_folds} pre-trained fold checkpoints from {kfold_dir}")
        elif not force_retrain and any(folds_exist):
            # Auto-detect: use existing checkpoints where available, train the rest
            n_existing = sum(folds_exist)
            logger.info(f"Found {n_existing}/{n_folds} pre-trained fold checkpoints. Training {n_folds - n_existing} new folds.")
        else:
            # Default or force_retrain: train all folds
            if force_retrain and any(folds_exist):
                logger.warning(f"force_retrain=True; retraining all {n_folds} folds (discarding {sum(folds_exist)} existing checkpoints).")
            else:
                logger.info(f"Training {n_folds} K-fold splits from scratch.")

        for fold_id, (tr_rel, va_rel) in enumerate(kf.split(base_indices)):
            fold_train_idx = base_indices[tr_rel]
            fold_val_idx   = base_indices[va_rel]
            fold_ckpt_path = fold_checkpoints[fold_id]

            # Decide whether to load or train this fold
            should_load = (not force_retrain) and fold_ckpt_path.exists()

            if should_load:
                logger.info(f"Fold {fold_id+1}/{n_folds}: Loading pre-trained checkpoint from {fold_ckpt_path}")
                self.calibrator.load_state_dict(deepcopy(initial_state_dict))
                self.load_checkpoint(fold_ckpt_path, load_optimizer=False)
            else:
                logger.info(f"Fold {fold_id+1}/{n_folds}: Training (train={len(fold_train_idx)}, val={len(fold_val_idx)})")
                self.calibrator.load_state_dict(deepcopy(initial_state_dict))
                self.optimizer, self.scheduler = self._make_optimizer()

                self.train(
                    num_epochs = fold_epochs,
                    patience   = fold_patience,
                    train_loader = self._make_loader(fold_train_idx, shuffle=True),
                    val_loader   = self._make_loader(fold_val_idx,   shuffle=False),
                    checkpoint_path=fold_ckpt_path,
                )

            # Evaluate per-sample losses on validation set
            val_ids, fold_l = self._evaluate_per_sample_losses(self._make_loader(fold_val_idx, shuffle=False))
            oof_losses[val_ids] = fold_l
            release_gpu_memory(sync=True)

        X  = np.array([[*self.dataset.telemetry[int(i)].values()] for i in base_indices], dtype=np.float32)
        y  = np.log10(oof_losses[base_indices] + 1e-12)
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        cv_r2 = cross_val_score(rf, X, y, cv=min(n_splits, len(base_indices)), scoring='r2')
        logger.info(f"RF difficulty CV R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
        rf.fit(X, y) # High R² means that telemetry has nfo on if calibrator struggles to predict a PSF.

        weights_train = self._difficulty_to_weights(rf.predict(X), alpha=alpha, clip=clip)
        weights_all   = np.ones(len(self.dataset), dtype=np.float32)
        weights_all[base_indices] = weights_train

        # Reset to the same starting point for the final training run
        self.calibrator.load_state_dict(deepcopy(initial_state_dict))
        self.optimizer, self.scheduler = self._make_optimizer()

        logger.info(f"Difficulty weights: min={weights_train.min():.3f} mean={weights_train.mean():.3f} max={weights_train.max():.3f}")
        return torch.as_tensor(weights_all, dtype=torch.float32, device=self.device)


    @torch.no_grad()
    def _evaluate_per_sample_losses(self, loader):
        self.calibrator.eval()
        global_ids = np.asarray(
            loader.dataset.indices if isinstance(loader.dataset, Subset) else range(len(loader.dataset)),
            dtype=int,
        )
        # Collect per-sample losses across all wavelength subsets and average per sample.
        loss_acc = {int(i): [] for i in global_ids}
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            x_pred = self.calibrator(tel_b)
            x_dict = self.outputs_transformer.unstack(x_pred)
            
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                per = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, return_per_sample=True).cpu().numpy()
                
                for gid, lv in zip(idxs_b.cpu().numpy().astype(int), per):
                    loss_acc[int(gid)].append(float(lv))
                    
                del PSF_pred_b, per
            del x_pred, x_dict
            
        return global_ids, np.array([np.mean(loss_acc[int(i)]) for i in global_ids], dtype=np.float32)


    @staticmethod
    def _difficulty_to_weights(pred_log_loss, alpha=0.30, clip=(0.5, 2.0)):
        d = np.asarray(pred_log_loss, dtype=np.float64)
        w = np.clip(np.exp(alpha * (d - np.median(d))) , None, None)
        w = w / np.mean(w)
        w = np.clip(w, clip[0], clip[1])
        return (w / np.mean(w)).astype(np.float32)


    # ── Fine-tune selected outputs ─────────────────────────────────────────────
    def tune_calibrator(self, tuned_params, num_epochs=50, lr=1e-3):
        """
        Fine-tune only specific output dimensions while anchoring the rest to a frozen reference.

        tuned_params : list[str] - keys present in outputs_transformer.slices.
        """
        ref = deepcopy(self.calibrator)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        mask = torch.zeros(self.N_outputs, dtype=torch.bool, device=self.device)
        for k in tuned_params:
            mask[self.outputs_transformer.slices[k]] = True
        mask = mask.unsqueeze(0)   # [1, N_outputs] for broadcasting

        opt_tune = optim.AdamW(self.calibrator.parameters(), lr=lr, weight_decay=5e-4)
        sch_tune = optim.lr_scheduler.ReduceLROnPlateau(opt_tune, factor=0.5, patience=5)
        best_loss, best_state = float('inf'), None

        fixed_mask = ~mask.squeeze(0)
        for epoch in range(num_epochs):
            self.calibrator.train()
            for PSF_data_b, tel_b, _, config_b, idxs_b in self.train_loader:
                opt_tune.zero_grad()

                # Anchor: computed once per batch (lambda-independent), separate forward pass
                if fixed_mask.any():
                    x_anch = self.calibrator(tel_b)
                    with torch.no_grad():
                        x_ref = ref(tel_b)
                    # Soft-fix strategy: keep untuned outputs close to the frozen reference model.
                    (x_anch[:, fixed_mask] - x_ref[:, fixed_mask]).pow(2).mean().mul(100.0).backward()
                    del x_anch
                else:
                    with torch.no_grad():
                        x_ref = ref(tel_b)

                # PSF loss: fresh forward pass per lambda set to keep peak memory low
                for lambda_ids in self.lambda_id_sets:
                    x_tuned = self.calibrator(tel_b)
                    # Use tuned outputs only on selected dimensions; other dims come from reference.
                    x_comb  = torch.where(mask, x_tuned, x_ref.detach())
                    x_dict  = self.outputs_transformer.unstack(x_comb)
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    loss       = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict)
                    loss.backward()
                    del PSF_pred_b, loss, x_dict, x_tuned, x_comb
                opt_tune.step()

            val_loss = self.validate()
            sch_tune.step(val_loss)
            
            if val_loss < best_loss:
                best_loss, best_state = val_loss, deepcopy(self.calibrator.state_dict())
                
            print(f"Tune {epoch+1}/{num_epochs} | val={val_loss:.4f}" + (" *" if val_loss == best_loss else ""))

        if best_state:
            self.calibrator.load_state_dict(best_state)
        logger.info(f"Tuning complete. Best val={best_loss:.6f}")


    # ── Save calibrator bundle ─────────────────────────────────────────────────
    def save_calibrator(self, path):
        """Export a self-contained calibrator bundle loadable by NFMCalibrator."""
        state = {
            'net_state_dict': self.calibrator.state_dict(),
            'net_arch': {
                'n_features':   self.N_features,
                'n_outputs':    self.N_outputs,
                'hidden_dim':   self.calibrator.network[0].out_features,
                'dropout_rate': self.calibrator.network[3].p,
            },
            'input_feature_names':  self.dataset.features,
            'output_feature_names': list(self.outputs_transformer.slices.keys()),
            'outputs_transformer':  self.outputs_transformer.save(),
            'predict_LO_NCPAs':     self.predict_LO_NCPAs,
            'predict_Cn2_profile':  self.predict_Cn2_profile,
            'LO_modes_max':         self.PSF_model.Z_mode_max,
            'N_spline_nodes':       getattr(self.PSF_model, 'N_wvl_ctrl', None),
            'predict_phase_bump':   False,
        }
        torch.save(state, path)
        logger.info(f"Calibrator bundle saved -> {path}")
