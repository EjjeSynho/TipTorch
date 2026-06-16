#%%
try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import gc
import os
import sys
import logging
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from tiptorch._config import *
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from calibrators.NFM_calibrator import SmallCalibratorNet


# ── Paths ─────────────────────────────────────────────────────────────────────
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER       = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE    = STD_FOLDER / 'dataset_cache'
BEST_CALIB_PATH  = WEIGHTS_FOLDER / 'NFM_calibrator/best_calibrator_checkpoint.pth'

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train NFM Calibrator")
parser.add_argument('--continue-training',    action='store_true',  help='Resume training from last checkpoint')
parser.add_argument('--run-kfold-difficulty', action='store_true',  help='Run K-fold difficulty weighting before final training')
parser.add_argument('--kfolds',               type=int,   default=5)
parser.add_argument('--kfold-epochs',         type=int,   default=120)
parser.add_argument('--kfold-patience',       type=int,   default=12)
parser.add_argument('--difficulty-alpha',     type=float, default=0.30)
parser.add_argument('--difficulty-clip-min',  type=float, default=0.50)
parser.add_argument('--difficulty-clip-max',  type=float, default=2.00)
parser.add_argument('--debug',                action='store_true')
parser.add_argument('--batch-size',           type=int,   default=16)

try:
    args = parser.parse_args()
except SystemExit:
    args = argparse.Namespace(
        continue_training=True,
        run_kfold_difficulty=True,
        kfolds=5,
        kfold_epochs=120,
        kfold_patience=12,
        difficulty_alpha=0.30,
        difficulty_clip_min=0.50,
        difficulty_clip_max=2.00,
        debug=True,
        batch_size=16,
    )

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")


def release_gpu_memory(sync=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class NFMDataset(Dataset):
    """
    Dataset of MUSE NFM PSF cubes with paired telemetry and fitted parameter values.
    Each sample: (PSF_cube [C,H,W], telemetry [D], fitted_params dict, config, dataset_idx).
    """
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.PSF_cubes   = data['PSF_cubes']
        self.telemetry   = data['telemetry']
        self.sample_ids  = data['sample_ids']
        self.configs     = data['model_configs']
        self.fitted_vals = data['fitted_param_values']
        self.features    = list(data['telemetry'][0].keys())

        _rename = {p: p + '_ctrl' for p in ['J', 'F', 'bg', 'dx', 'dy']}
        for i in range(len(self.sample_ids)):
            if isinstance(self.configs[i], list):
                self.configs[i] = self.configs[i][0]
            for old_k, new_k in _rename.items():
                if old_k in self.fitted_vals[i] and new_k not in self.fitted_vals[i]:
                    self.fitted_vals[i][new_k] = self.fitted_vals[i].pop(old_k)

        self.N, self.H, self.W, self.C = self.PSF_cubes.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        PSFs = torch.from_numpy(self.PSF_cubes[idx].astype(np.float32)).permute(2, 0, 1)
        tel  = torch.from_numpy(np.array(list(self.telemetry[idx].values()), dtype=np.float32))
        fit  = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.fitted_vals[idx].items()}
        return PSFs, tel, fit, self.configs[idx], idx


def collate_batch(batch, device):
    PSF_cubes, telemetry_vecs, fitted_vals, configs, idxs = zip(*batch)
    PSF_cubes      = torch.stack(PSF_cubes,      0).to(device=device, non_blocking=True)
    telemetry_vecs = torch.stack(telemetry_vecs, 0).to(device=device, non_blocking=True)
    idxs           = torch.tensor(idxs, dtype=torch.long, device=device)
    fitted_vals    = {k: torch.stack([fv[k] for fv in fitted_vals], 0).to(device=device, non_blocking=True)
                      for k in fitted_vals[0]}
    batch_config = MultipleTargetsInDifferentObservations(configs, device=device)
    batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
    batch_config['telescope']['PupilAngle'] = 0.0
    batch_config['DM']['NumberReconstructedLayers'] = torch.tensor(3.0, device=device)
    return PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
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
        self.lambda_id_sets = self._build_wavelength_sets(lambda_step, dataset.C - 1)
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
        return DataLoader(
            dataset=Subset(self.dataset, np.asarray(indices, dtype=int)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: collate_batch(b, device=self.device),
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
        self.dx = torch.tensor(dx_data, device=self.device, dtype=torch.float32,
                               requires_grad=self.optimize_astrometry)
        self.dy = torch.tensor(dy_data, device=self.device, dtype=torch.float32,
                               requires_grad=self.optimize_astrometry)

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
        x_dict = self.PSF_model.inputs_manager.to_dict()
        x_dict.update({k: v for k, v in x_dict_NN.items() if k in x_dict})
        x_dict['dx_ctrl'] = self.dx[idx]
        x_dict['dy_ctrl'] = self.dy[idx]
        bump = self.phase_bump[idx].unsqueeze(-1)
        ncpa = x_dict['LO_coefs'] if self.predict_LO_NCPAs else self.NCPAs_median.expand(len(idx), -1)
        x_dict['LO_coefs'] = torch.hstack((bump, ncpa))
        wvl = self.lambda_full[lambda_ids].to(device=self.device)
        config['sources_science']['Wavelength'] = wvl.view(1, -1)
        self.PSF_model.model.config = config
        self.PSF_model.SetWavelengths(wvl)
        return self.PSF_model(x_dict, update_params=False)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _loss_per_sample(self, PSF_data, PSF_pred, x_dict_pred):
        diff = PSF_pred - PSF_data
        w    = 2e4
        psf  = w * (diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0 + diff.abs().mean(dim=(1, 2, 3)) * 1.6)
        if self.predict_LO_NCPAs and 'LO_coefs' in x_dict_pred:
            c  = x_dict_pred['LO_coefs']
            lo = c.pow(2).sum(-1) * 1e-7
            lo = lo + torch.clamp(-c[:, 0], min=0).pow(2) * 5e-5
            lo = lo + torch.clamp(-c[:, 2], min=0).pow(2) * 1e-7
        else:
            lo = torch.zeros(PSF_data.shape[0], device=PSF_data.device, dtype=PSF_data.dtype)
        return psf + lo

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
        """Adapt fitted_vals dict to calibrator output structure (drop phase bump, reconstruct Cn2 GL)."""
        keys = set(self.outputs_transformer.slices.keys())
        x = {k: v for k, v in fitted_vals.items() if k in keys}
        if 'LO_coefs' in x:
            x['LO_coefs'] = x['LO_coefs'][:, 1:]   # strip phase bump; not predicted
        if 'Cn2_weights' in x:
            cn2 = x['Cn2_weights'].clamp(min=1e-6)
            GL  = (1.0 - cn2.sum(-1, keepdim=True)).clamp(min=1e-6)
            x['Cn2_weights'] = torch.hstack((GL, cn2))
        return x

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
                y = self.outputs_transformer.stack(self._calibrator_friendly_fit_dict(fit))
                loss = criterion(self.calibrator(tel), y)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            self.calibrator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, tel, fit, _, _ in self.val_loader:
                    y = self.outputs_transformer.stack(self._calibrator_friendly_fit_dict(fit))
                    val_loss += criterion(self.calibrator(tel), y).item()

            avg_t, avg_v = train_loss / len(self.train_loader), val_loss / len(self.val_loader)
            sch.step(avg_v)
            is_best = avg_v < best_loss
            if is_best:
                best_loss, best_state, pat_count = avg_v, deepcopy(self.calibrator.state_dict()), 0
            else:
                pat_count += 1

            print(f"\rPretrain {epoch+1:3d}/{num_epochs} | "
                  f"train={avg_t:.4e} val={avg_v:.4e} lr={opt.param_groups[0]['lr']:.2e}"
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
            PSFs_pred = torch.zeros((N, self.N_wvl_total, self.H, self.W))
            PSFs_data = torch.zeros((N, self.N_wvl_total, self.H, self.W))
            NN_pred   = torch.zeros((N, self.N_outputs))
            tel_vecs  = torch.zeros((N, self.N_features))

        total_loss, total_batches = 0.0, 0
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            lpos   = torch.tensor([global_to_local[int(i)] for i in idxs_b.cpu().tolist()], dtype=torch.long)
            x_pred = self.calibrator(tel_b)
            x_dict = self.outputs_transformer.unstack(x_pred)

            if return_cubes:
                NN_pred[lpos]  = x_pred.cpu()
                tel_vecs[lpos] = tel_b.cpu()

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

                for lambda_ids in self.lambda_id_sets:
                    x_pred     = self.calibrator(tel_b)
                    x_dict     = self.outputs_transformer.unstack(x_pred)
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    sw         = sample_weights[idxs_b] if sample_weights is not None else None
                    loss       = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, sw)

                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_this_epoch = True
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

                torch.nn.utils.clip_grad_norm_(self.calibrator.parameters(), max_norm=1.0)
                if self.optimize_astrometry:
                    torch.nn.utils.clip_grad_norm_([self.dx, self.dy], max_norm=10.0)
                self.optimizer.step()
                epoch_loss += batch_loss
                n_batches  += 1

            release_gpu_memory()
            if nan_this_epoch:
                continue

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
            msg = (f"Epoch {epoch:4d} | train={avg_train:.4e} val={val_loss:.4e} "
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
    ):
        """
        Estimate out-of-fold PSF losses, fit a random-forest difficulty model, and
        return dataset-length sample weights for the final training run.

        Only samples in base_indices are weighted; all others receive weight 1.0.
        The calibrator and optimizer are reset to initial_state_dict afterwards.
        """
        base_indices = np.asarray(base_indices, dtype=int)
        kf           = KFold(n_splits=min(n_splits, len(base_indices)), shuffle=True, random_state=random_state)
        kfold_dir    = self.weights_dir / 'kfold_difficulty'
        kfold_dir.mkdir(exist_ok=True)
        oof_losses   = np.full(len(self.dataset), np.nan, dtype=np.float32)

        for fold_id, (tr_rel, va_rel) in enumerate(kf.split(base_indices)):
            fold_train_idx = base_indices[tr_rel]
            fold_val_idx   = base_indices[va_rel]
            logger.info(f"Fold {fold_id+1}/{n_splits}: train={len(fold_train_idx)}, val={len(fold_val_idx)}")
            self.calibrator.load_state_dict(deepcopy(initial_state_dict))
            self.optimizer, self.scheduler = self._make_optimizer()
            self.train(
                num_epochs=fold_epochs, patience=fold_patience,
                train_loader=self._make_loader(fold_train_idx, shuffle=True),
                val_loader=self._make_loader(fold_val_idx,   shuffle=False),
                checkpoint_path=kfold_dir / f'fold_{fold_id:02d}_best.pth',
            )
            val_ids, fold_l = self._evaluate_per_sample_losses(
                self._make_loader(fold_val_idx, shuffle=False)
            )
            oof_losses[val_ids] = fold_l
            release_gpu_memory(sync=True)

        X  = np.array([[*self.dataset.telemetry[int(i)].values()] for i in base_indices], dtype=np.float32)
        y  = np.log10(oof_losses[base_indices] + 1e-12)
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        cv_r2 = cross_val_score(rf, X, y, cv=min(n_splits, len(base_indices)), scoring='r2')
        logger.info(f"RF difficulty CV R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
        rf.fit(X, y)

        weights_train = self._difficulty_to_weights(rf.predict(X), alpha=alpha, clip=clip)
        weights_all   = np.ones(len(self.dataset), dtype=np.float32)
        weights_all[base_indices] = weights_train

        # Reset to the same starting point for the final training run
        self.calibrator.load_state_dict(deepcopy(initial_state_dict))
        self.optimizer, self.scheduler = self._make_optimizer()

        logger.info(f"Difficulty weights: min={weights_train.min():.3f} "
                    f"mean={weights_train.mean():.3f} max={weights_train.max():.3f}")
        return torch.as_tensor(weights_all, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _evaluate_per_sample_losses(self, loader):
        self.calibrator.eval()
        global_ids = np.asarray(
            loader.dataset.indices if isinstance(loader.dataset, Subset) else range(len(loader.dataset)),
            dtype=int,
        )
        loss_acc = {int(i): [] for i in global_ids}
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            x_pred = self.calibrator(tel_b)
            x_dict = self.outputs_transformer.unstack(x_pred)
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                per = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict,
                                   return_per_sample=True).cpu().numpy()
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
                    (x_anch[:, fixed_mask] - x_ref[:, fixed_mask]).pow(2).mean().mul(100.0).backward()
                    del x_anch
                else:
                    with torch.no_grad():
                        x_ref = ref(tel_b)

                # PSF loss: fresh forward pass per lambda set to keep peak memory low
                for lambda_ids in self.lambda_id_sets:
                    x_tuned = self.calibrator(tel_b)
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
            print(f"Tune {epoch+1}/{num_epochs} | val={val_loss:.4e}" + (" *" if val_loss == best_loss else ""))

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


# =============================================================================
# Setup
# =============================================================================
#%%
dataset     = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_wvl_total = dataset.C
logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W})")

#%%
# Initialize the PSF model with a representative batch config
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

_tmp_batch  = tuple([dataset[i] for i in np.random.randint(0, len(dataset), size=args.batch_size)])
_, _, _, _tmp_config, _ = collate_batch(_tmp_batch, device=default_device)

if 'PSF_model' in locals():
    del PSF_model

with torch.no_grad():
    PSF_model = PSFModelNFM(
        _tmp_config,
        multiple_obs   = True,
        LO_NCPAs       = True,
        chrom_defocus  = False,
        use_Moffat     = False,
        retain_PSDs    = False,
        N_spline_nodes = 5,
        Z_mode_max     = 9,
        device         = default_device,
    )

del _tmp_batch, _tmp_config
release_gpu_memory(sync=True)

# Remove parameters that are fixed externally and do not need to be predicted
for _key in ['Jxy', 'bg_ctrl', 'dx_ctrl', 'dy_ctrl', 'L0',
             'F_norm', 'F_norm_lambda_ctrl', 'src_dirs_x', 'src_dirs_y',
             'wind_speed_single', 'wind_dir_single']:
    try:
        PSF_model.inputs_manager.delete(_key)
    except Exception:
        pass   # key may have an alternate name; delete what is present

# Handle unicode key variants
for _key in ['F_norm_λ_ctrl']:
    try:
        PSF_model.inputs_manager.delete(_key)
    except Exception:
        pass

if PSF_model.use_Moffat:
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

predict_LO_NCPAs    = True
predict_Cn2_profile = True
PSF_model.inputs_manager.set_optimizable(['LO_coefs'],    predict_LO_NCPAs)
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'], predict_Cn2_profile)
print(PSF_model.inputs_manager)

#%%
# Build the outputs transformer: maps calibrator predictions <-> PSF model inputs
outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
_buf = outputs_transformer.unstack(PSF_model.inputs_manager.stack())
if 'LO_coefs' in _buf:
    _buf['LO_coefs'] = _buf['LO_coefs'][:, 1:]   # strip phase-bump column; not predicted
_ = outputs_transformer.stack(_buf)
del _buf

N_outputs  = outputs_transformer.get_stacked_size()
N_features = len(dataset.features)
release_gpu_memory()
print(f"Calibrator: {N_features} inputs -> {N_outputs} outputs")

#%%
# Calibrator network  (can be swapped for any nn.Module with the same I/O contract)
calibrator = SmallCalibratorNet(
    n_features   = N_features,
    n_outputs    = N_outputs,
    hidden_dim   = 48,
    dropout_rate = 0.2,
).to(default_device)

#%%
# Instantiate the trainer
trainer = NFMCalibratorTrainer(
    PSF_model           = PSF_model,
    calibrator          = calibrator,
    dataset             = dataset,
    outputs_transformer = outputs_transformer,
    device              = default_device,
    batch_size          = args.batch_size,
    lr                  = 1e-2,
    lambda_step         = 2,
    predict_LO_NCPAs    = predict_LO_NCPAs,
    predict_Cn2_profile = predict_Cn2_profile,
    pre_init_astrometry = True,
    optimize_astrometry = False,
)

# =============================================================================
# Pretraining
# =============================================================================
#%%
pretrain_path = trainer.weights_dir / 'pretrain_weights.pth'
if (not args.continue_training) or not pretrain_path.exists():
    trainer.pretrain(num_epochs=100, patience=15, save_path=pretrain_path)
else:
    calibrator.load_state_dict(torch.load(pretrain_path, map_location=default_device))
    logger.info(f"Loaded pretrain weights from {pretrain_path}")

release_gpu_memory(sync=True)

val_loss_pretrain = trainer.validate()
logger.info(f"Post-pretrain validation loss: {val_loss_pretrain:.6f}")

# =============================================================================
# Main training
# =============================================================================
#%%
if args.continue_training and BEST_CALIB_PATH.exists():
    trainer.load_checkpoint(BEST_CALIB_PATH)

BASE_STATE     = deepcopy(calibrator.state_dict())
sample_weights = None

if args.run_kfold_difficulty:
    sample_weights = trainer.run_kfold_difficulty_weighting(
        base_indices       = trainer.train_idx,
        initial_state_dict = BASE_STATE,
        n_splits           = args.kfolds,
        fold_epochs        = args.kfold_epochs,
        fold_patience      = args.kfold_patience,
        alpha              = args.difficulty_alpha,
        clip               = (args.difficulty_clip_min, args.difficulty_clip_max),
    )

# Back up previous best checkpoint before overwriting
if BEST_CALIB_PATH.exists():
    os.rename(BEST_CALIB_PATH, BEST_CALIB_PATH.with_name(BEST_CALIB_PATH.stem + '_backup.pth'))

#%%
train_losses, val_losses = trainer.train(
    num_epochs         = 500,
    patience           = 20,
    nan_recovery       = True,
    max_nan_recoveries = 10,
    sample_weights     = sample_weights,
    checkpoint_path    = BEST_CALIB_PATH,
)

release_gpu_memory()
logger.info(f"Training done. Final train={train_losses[-1]:.6f} val={val_losses[-1]:.6f} best={min(val_losses):.6f}")

try:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses,   label='Val')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Training Losses')
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()
except Exception:
    pass

#%%
# Optional: fine-tune selected parameters after main training
trainer.tune_calibrator(tuned_params=['F_ctrl', 'dn'], num_epochs=20, lr=1e-3)

# =============================================================================
# Evaluation & diagnostics
# =============================================================================
#%%
trainer.load_checkpoint(BEST_CALIB_PATH)
calibrator.eval()

PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, telemetry_vecs, final_val_loss = \
    trainer.validate(return_cubes=True)

PSFs_pred_cube = PSFs_pred_cube.cpu()
PSFs_data_cube = PSFs_data_cube.cpu()
validation_ids = validation_ids.cpu()
release_gpu_memory()
print(f"Validation loss: {final_val_loss:.6f} | PSF cubes {PSFs_pred_cube.shape}")

#%%
trainer.save_calibrator(trainer.weights_dir / 'NFM_calibrator_bundle.pth')

#%%
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as fh:
    muse_df = pickle.load(fh)

lambda_full = trainer.lambda_full
id_src      = np.random.randint(0, PSFs_data_cube.shape[0])
true_id     = validation_ids[id_src].item()
wvl_select  = np.s_[0, N_wvl_total // 2, -1]

PSF_0 = PSFs_data_cube[id_src]
PSF_1 = PSFs_pred_cube[id_src]
vmin  = np.percentile(PSF_0[PSF_0 > 0].numpy(), 10)
vmax  = np.percentile(PSF_0[PSF_0 > 0].numpy(), 99.995)

fig, axes = plt.subplots(1, len(wvl_select), figsize=(15, 1.35 * len(wvl_select)))
p_errs = []
for i, lmbd in enumerate(wvl_select):
    err = plot_radial_PSF_profiles(
        PSF_0[lmbd].numpy(), PSF_1[lmbd].numpy(),
        'Data', 'TipTorch', cutoff=40, y_min=3e-2, linthresh=1e-2,
        return_profiles=True, ax=axes[i],
    )[2].squeeze().max().item()
    axes[i].set_title(f"lambda = {int((lambda_full[lmbd] * 1e9).round().item())} nm")
    p_errs.append(err)
plt.tight_layout(); plt.show()
print(f"DeltaSR per wavelength: {np.array(p_errs)}")

draw_PSF_stack(
    PSF_0.numpy()[wvl_select, ...], PSF_1.numpy()[wvl_select, ...],
    average=True, min_val=vmin, max_val=vmax, crop=80, cmap='inferno',
)
plt.title(muse_df.loc[true_id]['Filename']); plt.show()

#%%
# Per-sample loss histogram
def per_sample_losses(PSF_data, PSF_pred, x_dict_pred):
    if isinstance(x_dict_pred, torch.Tensor):
        x_dict_pred = outputs_transformer.unstack(x_dict_pred)
    diff = PSF_pred - PSF_data
    w    = 2e4
    psf  = w * (diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0 + diff.abs().mean(dim=(1, 2, 3)) * 1.6)
    if predict_LO_NCPAs and 'LO_coefs' in x_dict_pred:
        c = x_dict_pred['LO_coefs']
        lo = (c.pow(2).sum(-1) * 1e-7
              + torch.clamp(-c[:, 0], min=0).pow(2) * 5e-5
              + torch.clamp(-c[:, 2], min=0).pow(2) * 1e-7)
    else:
        lo = torch.zeros(PSF_data.shape[0])
    return psf + lo

loss_SR = per_sample_losses(PSFs_data_cube, PSFs_pred_cube, NN_predictions)
_ = plt.hist(loss_SR.numpy(), bins=50, log=True)
plt.xlabel('Per-sample loss'); plt.ylabel('Count'); plt.grid(True); plt.show()

#%%
# Spectrally-averaged radial profiles across the validation set
fig = plt.figure(figsize=(10, 6))
PSF_avg = lambda x: x.cpu().numpy().mean(axis=1)
plot_radial_PSF_profiles(
    PSF_avg(PSFs_data_cube), PSF_avg(PSFs_pred_cube),
    'Data', 'TipTorch', title='Spectrally averaged PSF', cutoff=40,
    ax=fig.add_subplot(111),
)
plt.tight_layout()
# plt.show()
# plt.savefig(STD_FOLDER / 'plots/calibrated_prediction_white.pdf', dpi=300)

#%%
# Feature-loss correlation
from scipy.stats import pearsonr, spearmanr

input_features = dataset.features
loss_vals = loss_SR.numpy()
tel_np    = telemetry_vecs.numpy()

pearson_r  = np.array([pearsonr( tel_np[:, i], loss_vals)[0] for i in range(len(input_features))])
spearman_r = np.array([spearmanr(tel_np[:, i], loss_vals)[0] for i in range(len(input_features))])
sort_idx   = np.argsort(np.abs(spearman_r))[::-1]
x_pos      = np.arange(len(input_features))

fig, ax = plt.subplots(figsize=(max(12, len(input_features) * 0.9), 5))
ax.bar(x_pos - 0.2, pearson_r[sort_idx],  width=0.4, label='Pearson r',  alpha=0.8)
ax.bar(x_pos + 0.2, spearman_r[sort_idx], width=0.4, label='Spearman rho', alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([input_features[i] for i in sort_idx], rotation=45, ha='right')
ax.set_ylabel('Correlation with loss')
ax.set_title('Feature-loss correlation (sorted by |Spearman rho|)')
ax.legend(); ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout(); plt.show()

#%%
# t-SNE diagnostic
from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2, metric="cosine", perplexity=50,
              init="pca", learning_rate="auto", random_state=0).fit_transform(telemetry_vecs)
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=12, c=loss_SR.log().numpy(), cmap='viridis')
plt.colorbar(sc, label='log(Loss)')
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
plt.title('t-SNE of telemetry coloured by log-loss')
plt.grid(True, alpha=0.3); plt.show()


#%%
# =============================================================================
# SHAP Analysis
# -----------------------------------------------------------------------------
# Uses DeepExplainer (gradient-based, exact for torch nn.Module) to compute
# SHAP values: shap_values[i] is the contribution of each input feature to
# each output neuron i.  We then aggregate across outputs to answer:
#   - Which input features matter most globally?
#   - Which outputs are controlled by which inputs?
#   - Which features are essentially irrelevant for all outputs?
# =============================================================================
#%%
import shap

# ── 1.  Prepare background and test sets ─────────────────────────────────────
# Rebuild the telemetry matrix directly from the dataset so that `tel_np` and
# `input_features` are always consistent, regardless of what is stored in the
# outer-scope `telemetry_vecs` variable (which may be stale from a previous
# interactive run or from a different analysis in the same session).
shap_input_features = dataset.features         # D feature names — SHAP-local copy
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)  # [N_dataset, D]

N_background = min(100, len(shap_tel_np))
rng          = np.random.default_rng(42)
bg_idx       = rng.choice(len(shap_tel_np), size=N_background, replace=False)
background   = torch.tensor(shap_tel_np[bg_idx], dtype=torch.float32, device=default_device)
X_explain    = torch.tensor(shap_tel_np,          dtype=torch.float32, device=default_device)

#%%
# ── 2.  Build explainer and compute SHAP values ───────────────────────────────
# DeepExplainer does not support LayerNorm (used in SmallCalibratorNet), so we
# use GradientExplainer instead — it computes integrated-gradient-based SHAP
# values and supports any differentiable PyTorch model.
# shap_values shape: [N_explain, N_features, N_outputs]
calibrator.eval()
explainer = shap.GradientExplainer(calibrator, background)
sv_list   = explainer.shap_values(X_explain)    # list of N_outputs arrays [N, N_feat]
shap_values = np.stack(sv_list)        # [N, N_feat, N_out]
print(f"SHAP values computed: {shap_values.shape}  (samples x features x outputs)") 

# Mean |SHAP| across samples: [N_feat, N_out]
mean_abs_shap = np.abs(shap_values).mean(axis=0)

#%%
# ── 3.  Build readable output names from the transformer ──────────────────────
# Each output neuron corresponds to one element of the stacked parameter vector.
# We expand the parameter names to match individual neurons for clear labelling.
output_neuron_names = []
for param_name, slc in outputs_transformer.slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_neuron_names.append(param_name)
    else:
        output_neuron_names.extend([f"{param_name}[{i}]" for i in range(width)])

assert len(output_neuron_names) == N_outputs, f"Name count mismatch: {len(output_neuron_names)} vs {N_outputs}"

#%%
# ── 4.  Global feature importance  (mean |SHAP| summed over all outputs) ─────
# Re-derive shap_input_features from mean_abs_shap's leading dimension so this
# cell is self-consistent even when run independently in an interactive session.
shap_input_features = dataset.features[:mean_abs_shap.shape[0]]
# Replace certain names in the features list:
shap_input_features[shap_input_features.index('gain')] = 'LO WFS gain'
shap_input_features[shap_input_features.index('window')] = 'LO WFS window'
shap_input_features[shap_input_features.index('theta0')] = 'Θ₀'
shap_input_features[shap_input_features.index('frequency')] = 'LO WFS frequency'
shap_input_features[shap_input_features.index('NGS mag (from ph.)')] = 'NGS mag'
shap_input_features[shap_input_features.index('LGS photons, [photons/m^2/s]')] = 'LGS flux, [ph./m²/s]'

shap_input_features = [feat.replace('_binned_', '_') for feat in shap_input_features]
shap_input_features = [feat.replace('Tau0', 'τ₀') for feat in shap_input_features]
shap_input_features = [feat.replace('Cn2', 'Cₙ²') for feat in shap_input_features]

global_importance   = mean_abs_shap.sum(axis=1)    # [N_feat]
feat_order          = np.argsort(global_importance)[::-1]

fig, ax = plt.subplots(figsize=(max(10, len(shap_input_features) * 0.55), 5))
bars = ax.bar(range(len(shap_input_features)), global_importance[feat_order], color=plt.cm.RdYlGn(global_importance[feat_order] / global_importance.max()))

ax.set_xticks(range(len(shap_input_features)))
ax.set_xticklabels([shap_input_features[i] for i in feat_order], rotation=45, ha='right')
ax.set_ylabel('Mean |SHAP| summed over all outputs')
ax.set_title('Global feature importance across all calibrator outputs')
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout(); plt.show()

# Print ranked table
print(f"\n{'Feature':<35} {'Global |SHAP|':>14}")
print("-" * 51)
for i in feat_order:
    print(f"{shap_input_features[i]:<35} {global_importance[i]:>14.4f}")

#%%
# ── 5.  Identify useless features ─────────────────────────────────────────────
# A feature is "useless" if its global importance is below a relative threshold
# (here: less than 1 % of the most important feature).
useless_threshold  = 0.01 * global_importance.max()
useless_feat_mask  = global_importance < useless_threshold
useless_features   = [shap_input_features[i] for i in range(len(shap_input_features)) if useless_feat_mask[i]]
useful_features    = [shap_input_features[i] for i in range(len(shap_input_features)) if not useless_feat_mask[i]]

print(f"\nUseless features (< 1% of max global importance): {useless_features}")
print(f"Useful features ({len(useful_features)}): {useful_features}")

# ── 6.  Heatmap: input features vs. output parameters ─────────────────────────
#%%
# Aggregate per-parameter-group (not per-neuron) for readability:
# collapse the neuron dimension by averaging within each parameter group.
param_names  = list(outputs_transformer.slices.keys())
mean_abs_per_param = np.zeros((len(shap_input_features), len(param_names)))
for j, (pname, slc) in enumerate(outputs_transformer.slices.items()):
    mean_abs_per_param[:, j] = mean_abs_shap[:, slc].mean(axis=1)

import seaborn as sns

# Normalise each output column to [0,1] so that small-magnitude outputs are still visible
norm_heatmap = mean_abs_per_param / (mean_abs_per_param.max(axis=0, keepdims=True) + 1e-12)

# Order features by global importance (most important at top)
feat_order_hm = np.argsort(global_importance)[::-1]

fig, ax = plt.subplots(figsize=(max(10, len(param_names) * 0.65),
                                max(6,  len(shap_input_features) * 0.35)))
sns.heatmap(
    norm_heatmap[feat_order_hm, :],
    xticklabels=param_names,
    yticklabels=[shap_input_features[i] for i in feat_order_hm],
    cmap='magma',
    linewidths=0.3,
    linecolor='#333',
    vmin=0, vmax=1,
    ax=ax,
    cbar_kws={'label': 'Normalised mean |SHAP| (per output column)'},
)
ax.set_title('Input feature → output parameter importance (column-normalised)')
ax.set_xlabel('Calibrator output (predicted parameter)')
ax.set_ylabel('Input feature (telemetry, sorted by global importance)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.show()

# ── 7.  SHAP beeswarm summary plot (all outputs combined) ────────────────────
#%%
# Flatten shap_values to [N, N_feat] by summing |SHAP| over outputs per sample,
# then feed to SHAP's summary_plot in "violin" style for a cleaner view.
shap_summary = shap_values.sum(axis=-1)    # [N, N_feat]  — signed sum over outputs

# Re-derive locally so this cell works even when run independently.
_n_feat = shap_summary.shape[1]
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)
assert shap_tel_np.shape[1] == _n_feat, (
    f"Dataset has {shap_tel_np.shape[1]} features but shap_values has {_n_feat}. "
    "Re-run Steps 1-2 to recompute shap_values with the current dataset."
)
shap_input_features = np.array(dataset.features[:_n_feat])

shap.summary_plot(
    shap_summary,
    features=shap_tel_np,
    feature_names=shap_input_features,
    plot_type='violin',
    max_display=len(shap_input_features),
    show=True,
)

# ── 8.  Per-output SHAP bar charts for the most important parameters ──────────
#%%
# Show the top-k most important features for each predicted parameter individually.
TOP_K_FEATURES = 10   # how many features to display per parameter

n_params  = len(param_names)
ncols     = min(4, n_params)
nrows     = int(np.ceil(n_params / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 4.5, nrows * 3.5),
                         constrained_layout=True)
axes_flat = np.array(axes).ravel()

for j, pname in enumerate(param_names):
    ax   = axes_flat[j]
    slc  = outputs_transformer.slices[pname]
    imp  = mean_abs_shap[:, slc].mean(axis=1)          # mean over neurons in this param
    top  = np.argsort(imp)[::-1][:TOP_K_FEATURES]
    ax.barh([shap_input_features[i] for i in top[::-1]], imp[top[::-1]],
            color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_title(pname, fontsize=9)
    ax.set_xlabel('Mean |SHAP|', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='x', alpha=0.3)

# Hide any unused axes
for ax in axes_flat[n_params:]:
    ax.set_visible(False)

fig.suptitle(f'Top-{TOP_K_FEATURES} input features per predicted parameter', fontsize=11)
plt.show()

# ── 9.  SHAP dependence plot for the single most predictive feature per output ─
#%%
print("\nMost predictive input feature per predicted parameter:")
print(f"{'Parameter':<30} {'Feature':<35} {'Mean |SHAP|':>12}")
print("-" * 80)
for j, pname in enumerate(param_names):
    slc  = outputs_transformer.slices[pname]
    imp  = mean_abs_shap[:, slc].mean(axis=1)
    best = int(np.argmax(imp))
    print(f"{pname:<30} {shap_input_features[best]:<35} {imp[best]:>12.4f}")

#%%
# Dependence plots: for each parameter group show the dominant feature's
# SHAP value vs. its raw value, coloured by the second-strongest feature.
# Limit to the first 6 parameter groups for visual clarity.
# Re-derive shap_tel_np locally for the same session-isolation reason as Step 7.
_n_feat = shap_values.shape[1]
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)
assert shap_tel_np.shape[1] == _n_feat, (
    f"Dataset has {shap_tel_np.shape[1]} features but shap_values has {_n_feat}. "
    "Re-run Steps 1-2 to recompute shap_values with the current dataset."
)
shap_input_features = np.array(dataset.features[:_n_feat])

shap_per_param = {
    pname: shap_values[:, :, slc].mean(axis=-1)   # [N, N_feat]  mean SHAP over neurons
    for pname, slc in outputs_transformer.slices.items()
}

N_PARAM_PLOTS = min(6, n_params)
fig, axes = plt.subplots(2, (N_PARAM_PLOTS + 1) // 2,
                         figsize=((N_PARAM_PLOTS + 1) // 2 * 5, 8),
                         constrained_layout=True)
axes_flat = np.array(axes).ravel()

for j, pname in enumerate(list(shap_per_param.keys())[:N_PARAM_PLOTS]):
    sv  = shap_per_param[pname]                     # [N, N_feat]
    imp = np.abs(sv).mean(axis=0)
    top2 = np.argsort(imp)[::-1][:2]
    feat_x   = shap_input_features[top2[0]]
    feat_col = shap_input_features[top2[1]] if len(top2) > 1 else feat_x

    ax    = axes_flat[j]
    x_val = shap_tel_np[:, top2[0]]
    y_val = sv[:, top2[0]]
    c_val = shap_tel_np[:, top2[1]]

    sc = ax.scatter(x_val, y_val, c=c_val, cmap='coolwarm', s=10, alpha=0.7)
    ax.axhline(0, color='black', lw=0.7, linestyle='--')
    ax.set_xlabel(feat_x, fontsize=8)
    ax.set_ylabel(f'SHAP for {pname}', fontsize=8)
    ax.set_title(pname, fontsize=9)
    fig.colorbar(sc, ax=ax, label=feat_col, fraction=0.03, pad=0.02)
    ax.grid(True, alpha=0.3)

for ax in axes_flat[N_PARAM_PLOTS:]:
    ax.set_visible(False)

fig.suptitle('SHAP dependence: dominant feature per parameter\n(colour = second-strongest feature)', fontsize=11)
plt.show()


#%%


