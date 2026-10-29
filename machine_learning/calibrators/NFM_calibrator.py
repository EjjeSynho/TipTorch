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
from data_processing.MUSE_data_utils import filter_dataframe, reduce_dataframe
from data_processing.NFM_bundle import telemetry_resources, BUNDLE_PATH
from data_processing.telemetry_preprocessor import TelemetryPreprocessor
from data_processing.Cn2_profile import Cn2Grid
from tiptorch.managers.input_manager import InputsTransformer
from tiptorch.tools.utils import BalancedPSFLoss, r0 as r0_from_seeing

from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

from tiptorch._config import *
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Calibrator configs (NFM_calibrator_config.json: direct, NFM_factor_calibrator_config.json: factor, ETC_calibrator_config.json)
CONFIGS_FOLDER = Path(__file__).resolve().parent / 'configs'


def load_telemetry_preprocessor(bundle_state: dict) -> TelemetryPreprocessor:
    """
    Telemetry scaler + imputer the calibrator was trained with (TelemetryPreprocessor.state_dict() copied from the dataset
    cache into the bundle by NFMCalibratorTrainer.save_calibrator).
    """
    preprocessor = TelemetryPreprocessor.from_state_dict(bundle_state['telemetry_preprocessor'])
    if list(bundle_state['input_feature_names']) != preprocessor.features:
        raise ValueError('Telemetry features of the preprocessor do not match the calibrator inputs')

    return preprocessor


def telemetry_base_params(telemetry_physical, features: list, L0_const: float | None = None, Cn2_grid=None) -> dict:
    """
    Telemetry-derived first guesses of the PSF-model parameters the factor calibrator corrects: r0 at 500 nm from the header
    seeing, the speed of the single wind layer from the header, the Cn2 profile [GL, layer 2, layer 3] from the AOF profile
    reprojected onto the fixed layers of Cn2_grid (Cn2_grid_* features, Cn2_binning 'grid') or, with the K-means Cn2 features
    (Cn2_binning 'kmeans'), from the fractions Cn2_frac_binned_2/3. L0 has no telemetry counterpart, so a dataset-level constant
    (median of the fits) is used instead.

    telemetry_physical: [N, N_features] in physical units (TelemetryPreprocessor.inverse_transform of the calibrator inputs).
    Returns float32 tensors with the shapes of the PSF-model inputs: r0, L0 [N], wind_speed_single [N, 1], Cn2_weights [N, 3].
    """
    T   = torch.as_tensor(telemetry_physical, dtype=torch.float32).reshape(-1, len(features))
    col = lambda name: T[:, features.index(name)]

    seeing = col('Seeing (header)').clamp(0.2, 3.0)               # ["]
    if Cn2_grid is not None and all(name in features for name in Cn2_grid.feature_names):
        Cn2 = torch.stack([col(name) for name in Cn2_grid.feature_names], dim=-1).clamp(min=1e-3) # imputed profiles are not normalized
    else:
        f2  = col('Cn2_frac_binned_2').clamp(1e-3, 0.98)
        f3  = col('Cn2_frac_binned_3').clamp(1e-3, 0.98)
        Cn2 = torch.stack([(1.0 - f2 - f3).clamp(min=1e-3), f2, f3], dim=-1)

    Cn2 = Cn2 / Cn2.sum(dim=-1, keepdim=True)

    base = {
        'r0':                r0_from_seeing(seeing, 500e-9),       # [m]
        'wind_speed_single': col('Wind speed (header)').clamp(0.5, 30.0)[:, None], # [m/s]
        'Cn2_weights':       Cn2,
    }
    
    if L0_const is not None:
        base['L0'] = torch.full((len(T),), float(L0_const), dtype=T.dtype, device=T.device)

    return base


def Cn2_grid_features_from_GLF(Cn2_grid, GLF) -> dict:
    """
    Cn2 telemetry features (Cn2_grid_*) implied by a ground-layer fraction GLF [N] and the median grid profile above the ground:
    the ETC regime, where only the GLF of the turbulence category is known.
    """
    GLF   = np.clip(np.asarray(GLF, dtype=float).ravel(), 0.0, 1.0)
    upper = Cn2_grid.median_fracs[1:] / Cn2_grid.median_fracs[1:].sum()
    g     = np.hstack([GLF[:, None], (1.0 - GLF)[:, None] * upper[None, :]])
    return {name: g[:, k] for k, name in enumerate(Cn2_grid.feature_names)}


def default_split(dataset):
    """ Train/val split stored in the dataset cache, where samples acquired close in time never end up in both subsets. """
    return np.asarray(dataset.train_idx, dtype=int), np.asarray(dataset.val_idx, dtype=int)


def kfold_splits(dataset, base_indices, n_splits, random_state=42):
    """
    K-fold splits of base_indices (relative positions), grouped by acquisition time (the time groups of the dataset cache), so that
    samples close in time stay in the same fold.
    """
    groups = np.asarray(dataset.time_groups)[base_indices]
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))), shuffle=True, random_state=random_state)
    return list(gkf.split(base_indices, groups=groups))


class KcGroupBatchSampler(torch.utils.data.Sampler):
    """
    Batches of positions into a Subset such that every batch holds samples of a single k_c group (TipTorch holds one k_c per model,
    see NFMDataset). The batches are formed within each group (the last batch of a group may be smaller) and, with shuffle, the samples
    within the groups and the order of all batches are shuffled with torch's global RNG (seeded by the training script).
    """
    def __init__(self, groups, batch_size, shuffle):
        self.groups     = np.asarray(groups, dtype=int) # [N_subset] k_c group of every position
        self.batch_size = int(batch_size)
        self.shuffle    = bool(shuffle)
        self.n_batches  = int(sum(int(np.ceil(np.sum(self.groups == g) / self.batch_size)) for g in np.unique(self.groups)))

    def __iter__(self):
        batches = []
        for g in np.unique(self.groups):
            pos = np.flatnonzero(self.groups == g)
            if self.shuffle:
                pos = pos[torch.randperm(len(pos)).numpy()]
            batches += [pos[i:i + self.batch_size].tolist() for i in range(0, len(pos), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return iter(batches)

    def __len__(self):
        return self.n_batches


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
        self.telemetry_preprocessor = data['telemetry_preprocessor'] # TelemetryPreprocessor.state_dict()
        self.Cn2_grid    = data['Cn2_grid'] # Cn2Grid.state_dict() of the Cn2 features / configs, None with the K-means Cn2 features
        # Derotated PSFs (configs with PupilAngle 0) or raw frames (configs with the per-sample pupil angle, rotated by the model)
        self.derotated   = bool(data['derotated'])
        # Train/val split grouped by acquisition time (positions into sample_ids)
        self.train_idx   = data['train_idx']
        self.val_idx     = data['val_idx']
        self.time_groups = data['time_groups']
        # Samples flagged 'k_c mismatch' (fewer corrected DM modes, see datasets/MUSE_STD_dataset/kc_mismatch.py): k_c relative to the
        # nominal one and the k_c group (0 = nominal). Their configs carry the scaled DM pitch. The PSF model holds a single k_c per batch, so
        # the loaders batch the samples per group (KcGroupBatchSampler) and run_model rebuilds the model grids when the k_c changes
        self.kc_fraction = np.asarray(data['kc_fraction'], dtype=float)
        self.kc_group    = np.asarray(data['kc_group'],    dtype=int)

        # The fitted table stores the spline control values without the '_ctrl' suffix: J (the isotropic jitter width) and wing_frac
        # (image-frame wing) are the chromatic parameters of STD_fitter.py's regimes, see PSFModelNFM.polychromatic_params
        _rename = {p: p + '_ctrl' for p in ['J', 'wing_frac', 'F', 'bg', 'dx', 'dy']}
        
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

    @staticmethod
    def collate_batch(batch, device):
        PSF_cubes, telemetry_vecs, fitted_vals, configs, idxs = zip(*batch)
        PSF_cubes      = torch.stack(PSF_cubes,      0).to(device=device, non_blocking=True)
        telemetry_vecs = torch.stack(telemetry_vecs, 0).to(device=device, non_blocking=True)
        idxs           = torch.tensor(idxs, dtype=torch.long, device=device)
        fitted_vals    = {k: torch.stack([fv[k] for fv in fitted_vals], 0).to(device=device, non_blocking=True) for k in fitted_vals[0]}
        # The pupil angles stay per sample ([N_obs] in the merged config): 0 for derotated caches, the header angles for raw frames
        batch_config = MultipleTargetsInDifferentObservations(configs, device=device)
        batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
        batch_config['DM']['NumberReconstructedLayers'] = torch.tensor(batch_config['atmosphere']['Cn2Heights'].shape[-1], device=device)
        return PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs


STD_DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache'


def dataset_cache_path() -> Path:
    """
    STD-star dataset cache written by create_STD_dataset.py. The same file holds either regime, derotated PSFs (configs with
    PupilAngle 0) or raw frames (per-sample pupil angles); its 'derotated' entry tells which.
    """
    return STD_DATASET_CACHE / 'muse_STD_stars_dataset.pt'


def load_STD_dataset(derotated: bool | None = None) -> NFMDataset:
    """ Loads the STD-star dataset cache; with `derotated` given, checks that the stored cache is of that regime """
    dataset = NFMDataset(dataset_cache_path())
    if derotated is not None and dataset.derotated != bool(derotated):
        raise ValueError(f"The dataset cache {dataset_cache_path().name} is {'derotated' if dataset.derotated else 'raw'} "
                         f"but derotated_PSFs = {derotated} was requested; rebuild it with create_STD_dataset.py")
    return dataset


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
    bundle_type = 'direct' # the calibrator_type of the bundles this class loads (see load_NFM_calibrator)

    def __init__(
        self,
        checkpoint_path,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self._bundle_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = self._bundle_state

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
        
        # Inputs transformer (scaler + imputer)
        self.telemetry_preprocessor = load_telemetry_preprocessor(state)
        self.features = self.telemetry_preprocessor.features

        # Cn2 profile features: the fixed altitude grid (Cn2_grid_*) or the K-means features (Cn2_binning)
        self.Cn2_grid    = Cn2Grid.from_state_dict(state['Cn2_grid']) if state['Cn2_grid'] is not None else None
        self.Cn2_binning = state['Cn2_binning']
        # Median Cn2 profile of the K-means features (None when the telemetry resources were not packed: the one of the distributed bundle is used)
        self.median_Cn2_profile = state.get('median_Cn2_profile')

        # A factor bundle loaded with the base class would return the neutral values of the normalization for every parameter (its
        # network outputs are corrections, see NFMFactorCalibrator): use load_NFM_calibrator, which picks the class from the bundle
        if state['calibrator_type'] != self.bundle_type:
            raise TypeError(f"{type(self).__name__} cannot load a '{state['calibrator_type']}' calibrator bundle ({Path(checkpoint_path).name}): "
                            "use load_NFM_calibrator(bundle_path, device), which selects NFMCalibrator / NFMFactorCalibrator from the bundle")

        # Aux params
        self.LO_modes_max    = state['LO_modes_max']
        self.N_spline_nodes  = state['N_spline_nodes'] # None: PSF model without splines
        self.model_flags     = state['model_flags']
        self.fitted_shared   = state['fitted_shared']  # {key: tensor [1, ...]}, see fitted_shared_values()
        self.training_config = state['training_config'] # the resolved training config of the run


    @staticmethod
    def _tree_to_cpu(obj):
        """Recursively detach tensors and move them to CPU for portable serialization."""
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        
        if isinstance(obj, dict):
            return {k: NFMCalibrator._tree_to_cpu(v) for k, v in obj.items()}
        
        if isinstance(obj, (list, tuple)):
            return type(obj)(NFMCalibrator._tree_to_cpu(v) for v in obj)
        
        return deepcopy(obj)


    def save(self, path, **metadata):
        """
        Save a calibrator bundle with the current network weights.

        The original loaded bundle is used as the template, so telemetry scalers,
        output transforms, and architecture metadata are preserved. This is useful
        for science-data fine-tuning, where the tuned calibrator must be stored
        separately from the baseline STD-trained bundle.
        """
        state = deepcopy(self._bundle_state)
        state['net_state_dict'] = self._tree_to_cpu(self.net.state_dict())
        state['telemetry_preprocessor'] = self.telemetry_preprocessor.state_dict()
        if metadata:
            state['fine_tuning'] = metadata
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._tree_to_cpu(state), path)
        logger.info(f"Calibrator bundle saved -> {path}")
        return path


    def prepare_features(self, telemetry_features: pd.DataFrame) -> torch.Tensor:
        """Scale and impute telemetry already reduced to the calibrator features."""
        telemetry_ = self.telemetry_preprocessor.transform(telemetry_features)
        return torch.tensor(telemetry_, dtype=torch.float32, device=self.device)


    def prepare_telemetry(self, reduced_telemetry: pd.DataFrame) -> torch.Tensor:
        telemetry_pruned = reduce_dataframe(filter_dataframe(reduced_telemetry), Cn2_binning=self.Cn2_binning, Cn2_grid=self.Cn2_grid,
                                            median_Cn2_profile=self.median_Cn2_profile)
        return self.prepare_features(telemetry_pruned)


    def forward_prepared(self, telemetry_vector: torch.Tensor):
        """Predict PSF parameters from telemetry that is already scaled and imputed."""
        telemetry_vector = torch.as_tensor(
            telemetry_vector,
            dtype=torch.float32,
            device=self.device,
        )
        if telemetry_vector.ndim == 1:
            telemetry_vector = telemetry_vector.unsqueeze(0)

        n_features = self.net.network[0].in_features
        if telemetry_vector.ndim != 2 or telemetry_vector.shape[-1] != n_features:
            raise ValueError(
                "Prepared telemetry must have shape [N, "
                f"{n_features}], got {tuple(telemetry_vector.shape)}."
            )

        # Pass through the network to get predictions for the PSF model inputs
        x_pred = self.net(telemetry_vector)
        x_dict_pred = self._params_from_pred(x_pred, telemetry_vector)

        # Prepend a zeros column as the phase-bump placeholder (not predicted by the net)
        if 'LO_coefs' in x_dict_pred:
            coefs = x_dict_pred['LO_coefs']
            phase_bump = torch.zeros(coefs.shape[0], 1, device=coefs.device, dtype=coefs.dtype)
            x_dict_pred['LO_coefs'] = torch.cat([phase_bump, coefs], dim=-1)

        return x_dict_pred


    def _params_from_pred(self, x_pred: torch.Tensor, telemetry_vector: torch.Tensor) -> dict:
        """Network outputs [N, N_outputs] -> PSF-model parameters (phase bump excluded)."""
        return self.outputs_transformer.unstack(x_pred)


    def forward(self, reduced_telemetry: pd.DataFrame):
        # Prepare the telemetry data: filter, reduce, scale, and impute
        telemetry_vector = self.prepare_telemetry(reduced_telemetry)
        return self.forward_prepared(telemetry_vector)


    def __call__(self, reduced_telemetry: pd.DataFrame):
        return self.forward(reduced_telemetry)


    @torch.no_grad()
    def _apply_calibration(self, x_dict_pred: dict, PSF_model) -> None:
        """Adapt a calibrator prediction and apply it to a PSF model."""
        x_dict_adapted = self._adapt_to_PSF_model(x_dict_pred, PSF_model)
        # The fitted shared keys (e.g. the wing shape at the medians of the fits) are dataset-wide values: set them in the PSF model as well.
        # The inputs left to the model and the fitted keys (per-sample fits, unknown for new data) are not touched
        names = PSF_model.inputs_manager.get_names(optimizable_only=False)
        
        for key, val in self.fitted_shared.items():
            if key in names:
                cur = PSF_model.inputs_manager[key]
                v   = torch.as_tensor(val, dtype=cur.dtype, device=cur.device).reshape(1, -1)
                x_dict_adapted[key] = (v.reshape(-1)[:1].expand(cur.shape[0]) if cur.dim() == 1 else v.expand(cur.shape[0], -1).reshape(cur.shape)).clone()
       
        PSF_model.update_manager_params(x_dict_adapted)
        _ = PSF_model()


    @torch.no_grad()
    def calibrate(self, reduced_telemetry: pd.DataFrame, PSF_model) -> None:
        """
        Method to get a reconciled parameter dict ready for PSFModelNFM.
        Checks compatibility and adapts the predicted parameters to match the PSF model's expected input structure.
        """
        # Do parameters dict prediction
        x_dict_pred = self.forward(reduced_telemetry)
        self._apply_calibration(x_dict_pred, PSF_model)


    @torch.no_grad()
    def calibrate_prepared(self, telemetry_vector: torch.Tensor, PSF_model) -> None:
        """
        Calibrate a PSF model from telemetry that is already scaled and imputed.

        This is the dataset/batched equivalent of :meth:`calibrate`; it avoids
        applying the telemetry preprocessing pipeline to prepared NN inputs.
        """
        x_dict_pred = self.forward_prepared(telemetry_vector)
        self._apply_calibration(x_dict_pred, PSF_model)


    def check_compatibility(self, PSF_model) -> dict:
        """
        Check structural compatibility between the calibrator and a PSFModelNFM instance.
        Issues warnings for each mismatch.
        """
        issues = {}

        if hasattr(PSF_model, 'Z_mode_max'):
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

        # Jitter / wing regime: the predictions of a regime the PSF model does not have are ignored, and the model's inputs of a regime the
        # calibrator does not predict stay at their defaults (e.g. J_aniso for a calibrator trained without the stretch, wing_frac for one trained
        # without wing). The fitted shared keys need no check: the bundle's values are set in the PSF model at calibration
        for flag in ('anisotropic_jitter', 'wing_kernel'):
            f_calib, f_model = bool(self.model_flags[flag]), bool(getattr(PSF_model, flag))
            
            if f_calib != f_model:
                issues[flag] = (f_calib, f_model)
                warnings.warn(f"NFMCalibrator: {flag} mismatch — the calibrator was trained with {flag} = {f_calib}, the PSF model has {f_model}: "
                              "the predictions of the missing regime are ignored and the model's inputs stay at their defaults.")

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



# Key groups of the "calibrator" section of a training config. They do not overlap; the inputs of the PSF model in none of them are left to
# the model, the calibrator does not touch them (see MODEL_REQUIRED_INPUTS):
#   predicted_keys         predicted per sample by the network, directly (in the normalized space of the outputs transformer)
#   factor_keys            predicted per sample by the network as a factor on a telemetry-derived baseline (factor calibrator, FACTOR_BASE_KEYS)
#   optimized_shared_keys  one value for all samples, starting from the median of the training fits and trained with the network (factor calibrator)
#   fitted_keys            per sample, the values of the fitted dataset (e.g. the astrometry dx_ctrl / dy_ctrl), not optimized; unknown at inference
#   fitted_shared_keys     one value for all samples set in the PSF model, the median of the training fits, not optimized (stored in the bundle,
#                          set in the PSF model by NFMCalibrator)
# The calibrator outputs (outputs transformer) are the predicted, factor and optimized shared keys
KEY_GROUPS       = ('predicted_keys', 'factor_keys', 'optimized_shared_keys', 'fitted_keys', 'fitted_shared_keys')
FACTOR_BASE_KEYS = ('r0', 'L0', 'wind_speed_single', 'Cn2_weights') # the inputs with a telemetry-derived baseline, see telemetry_base_params

# Inputs that PSFModelNFM.forward reads at every call (the jitter width, the stretch and angle of the anisotropic kernel, the chromatic defocus,
# the NCPAs): left to the model, they are passed at the model's own (wrapper default) values. The other inputs left to the model are not passed
# at all, and TipTorch uses its own values, re-initialized for every batch from the batch's config (r0, L0, wind, Cn2 profile, source
# directions) or to its defaults (F, bg, dx, dy, wing shape, ...)
MODEL_REQUIRED_INPUTS = ('J', 'J_ctrl', 'J_aniso', 'J_angle', 'chrom_defocus', 'chrom_defocus_ctrl', 'LO_coefs')


def calibrator_key_groups(calib_cfg: dict, input_names: list) -> dict:
    """
    The key groups (KEY_GROUPS) of the "calibrator" section `calib_cfg`, checked against the inputs `input_names` of the PSF model: every key
    in one group at most, factor keys with a telemetry baseline, no factor / optimized shared keys for the direct calibrator. Keys that are not
    inputs of the model in its regime (e.g. J_aniso without the anisotropic jitter) are dropped with a warning.
    Returns {group: [keys]}, each in the order of the model inputs.
    """
    groups = {g: list(calib_cfg[g]) for g in KEY_GROUPS}
    listed = [k for keys in groups.values() for k in keys]

    if repeated := sorted({k for k in listed if listed.count(k) > 1}):
        raise ValueError(f'{repeated} are listed more than once in the key groups {KEY_GROUPS}: the groups must not overlap')
    if no_base := [k for k in groups['factor_keys'] if k not in FACTOR_BASE_KEYS]:
        raise ValueError(f'factor_keys {no_base} have no telemetry baseline (only {FACTOR_BASE_KEYS} have one)')
    if calib_cfg['type'] == 'direct' and (groups['factor_keys'] or groups['optimized_shared_keys']):
        raise ValueError('The direct calibrator has no factor_keys / optimized_shared_keys: use the factor calibrator')
    if unknown := [k for k in listed if k not in input_names]:
        logger.warning(f'{unknown} are not inputs of the PSF model in the regime of the "model" section, ignored')

    return {g: [k for k in input_names if k in keys] for g, keys in groups.items()}


def build_PSF_model(config, model_cfg: dict, device):
    """
    PSFModelNFM from the "model" section of a training config, as the trainer and the validation build it. anisotropic_jitter (the isotropic
    J_ctrl stretched by the achromatic J_aniso along the axis at J_angle, instead of one isotropic J) and wing_kernel (image-frame wing:
    wing_frac, wing_width, wing_width_t, wing_angle) select the regime of the fits (STD_fitter.py: ANISOTROPIC_JITTER, WING_KERNEL). The model is
    built with the wrapper defaults; build_NFM_trainer sets the fitted shared keys (e.g. the wing shape) to the medians of the fits, which the
    bundles store (fitted_shared) and NFMCalibrator sets in the PSF models it calibrates.
    """
    from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

    return PSFModelNFM(
        config,
        multiple_obs       = True,
        LO_NCPAs           = model_cfg['LO_NCPAs'],
        chrom_defocus      = model_cfg['chrom_defocus'],
        anisotropic_jitter = model_cfg['anisotropic_jitter'],
        wing_kernel        = model_cfg['wing_kernel'],
        use_Moffat         = model_cfg['use_Moffat'],
        retain_PSDs        = model_cfg['retain_PSDs'],
        N_spline_nodes     = model_cfg['N_spline_nodes'],
        Z_mode_max         = model_cfg['Z_mode_max'],
        device             = device,
    )


def model_flags(PSF_model) -> dict:
    """ The jitter / wing regime of a PSFModelNFM (stored in the calibrator bundles, checked by NFMCalibrator.check_compatibility). """
    return {k: bool(getattr(PSF_model, k)) for k in ('anisotropic_jitter', 'wing_kernel', 'chrom_defocus')}


def fitted_values(dataset, key, idx=None) -> np.ndarray:
    """
    Fitted values of `key` of the samples `idx` (all by default) in the layout of the PSF-model input, [N, n_values]: the Cn2 weights with the
    ground layer (the fits store the upper layers only), LO_coefs with the phase bump. Raises if the fits carry no such key.
    """
    idx = range(len(dataset.fitted_vals)) if idx is None else idx
    try:
        values = np.stack([np.asarray(dataset.fitted_vals[int(i)][key], dtype=np.float32).reshape(-1) for i in idx])
    except KeyError:
        raise KeyError(f"The fits of the dataset carry no '{key}': refit the dataset in the regime of the config and rebuild the cache") from None

    if key == 'Cn2_weights': # as _calibrator_friendly_fit_dict
        values = values.clip(min=1e-3)
        values = np.hstack([(1.0 - values.sum(-1, keepdims=True)).clip(min=1e-3), values])

    return values


def fits_median(dataset, idx, key) -> torch.Tensor:
    """ Median over the samples `idx` of the fitted values of `key` (see fitted_values), as a flat tensor [n_values]. """
    return torch.as_tensor(fitted_values(dataset, key, idx)).median(dim=0).values


def fitted_shared_values(PSF_model, keys) -> dict:
    """ Values of the fitted shared keys `keys` in a PSFModelNFM (one value per key: they are dataset-wide), as {key: tensor [1, ...]}. Stored in
    the calibrator bundles and set by NFMCalibrator in the PSF models it calibrates (e.g. the wing shape at the fitted medians). """
    return {k: PSF_model.inputs_manager[k][:1].detach().cpu().clone() for k in keys}


def build_NFM_trainer(cfg: dict, dataset, device, weights_dir=None, seed: int = 0, **trainer_overrides):
    """
    PSF model, calibrator network, outputs transformer and trainer from a training config (the "model", "calibrator", "trainer" and
    "loss" sections of calibrators/configs/NFM_calibrator_config.json; the key groups of the "calibrator" section, see KEY_GROUPS, say what
    the calibrator does with each input of the PSF model), as train_NFM_calibrator.py builds them. Shared by the validation / analysis scripts
    so that they reproduce the training setup exactly.

    trainer_overrides (e.g. train_idx, val_idx) replace the corresponding trainer arguments.
    Returns (PSF_model, calibrator, outputs_transformer, trainer).
    """
    # The PSF model is initialized with a representative batch config, the actual configs are set per batch in run_model
    # (samples of a single k_c group, the configs of different k_c groups cannot be merged into one model)
    kc_groups = np.asarray(dataset.kc_group)
    pool      = np.flatnonzero(kc_groups == np.bincount(kc_groups).argmax())
    batch     = tuple(dataset[int(i)] for i in np.random.default_rng(seed).choice(pool, size=min(cfg['batch_size'], len(pool)), replace=False))
    _, _, _, batch_config, _ = dataset.collate_batch(batch, device=device)

    with torch.no_grad():
        PSF_model = build_PSF_model(batch_config, cfg['model'], device)
    del batch, batch_config
    release_gpu_memory()

    # The calibrator outputs (predicted, factor and optimized shared keys) are the optimizable inputs of the PSF model, the others are not
    names   = PSF_model.inputs_manager.get_names(optimizable_only=False)
    groups  = calibrator_key_groups(cfg['calibrator'], names)
    outputs = [k for k in names if k in groups['predicted_keys'] + groups['factor_keys'] + groups['optimized_shared_keys']]
    PSF_model.inputs_manager.set_optimizable(outputs, True)
    PSF_model.inputs_manager.set_optimizable([k for k in names if k not in outputs], False)

    # Fitted shared keys: one dataset-wide value in the PSF model, the training-set median of the fits (e.g. the wing shape wing_width /
    # wing_width_t / wing_angle, which the fitter sets). The values go into the bundle (fitted_shared)
    train_idx = trainer_overrides.get('train_idx')
    train_idx = default_split(dataset)[0] if train_idx is None else np.asarray(train_idx, dtype=int)

    for key in groups['fitted_shared_keys']:
        median = fits_median(dataset, train_idx, key)
        cur    = PSF_model.inputs_manager[key]

        with torch.no_grad():
            cur[:] = (median[0] if median.numel() == 1 else median.view(1, -1).expand(cur.shape[0], -1).reshape(cur.shape)).to(device=cur.device, dtype=cur.dtype)

        logger.info(f"fitted_shared_keys: {key} = {np.round(median.numpy(), 3).tolist()} (median of the training fits)")

    # Outputs transformer: calibrator predictions <-> PSF-model inputs, without the phase bump column of LO_coefs
    outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
    buf = outputs_transformer.unstack(PSF_model.inputs_manager.stack())

    if 'LO_coefs' in buf:
        buf['LO_coefs'] = buf['LO_coefs'][:, 1:]
    _ = outputs_transformer.stack(buf)

    del buf

    calib_cfg      = cfg['calibrator']
    trainer_kwargs = dict(
        device             = device,
        batch_size         = cfg['batch_size'],
        lr                 = cfg['trainer']['lr'],
        lambda_step        = cfg['trainer']['lambda_step'],
        fitted_keys        = groups['fitted_keys'],
        fitted_shared_keys = groups['fitted_shared_keys'],
        collate_fn         = dataset.collate_batch,
        loss_cfg           = cfg['loss'],
        weights_dir        = weights_dir,
    )
    trainer_kwargs.update(trainer_overrides)

    if calib_cfg['type'] == 'direct':
        calibrator = SmallCalibratorNet(
            n_features   = len(dataset.features),
            n_outputs    = outputs_transformer.get_stacked_size(),
            hidden_dim   = calib_cfg['hidden_dim'],
            dropout_rate = calib_cfg['dropout_rate'],
        ).to(device)
        trainer = NFMCalibratorTrainer(PSF_model, calibrator, dataset, outputs_transformer, **trainer_kwargs)

    elif calib_cfg['type'] == 'factor':
        # The network outputs only the predicted and factor keys; the optimized shared keys are dataset-level parameters of the trainer
        calibrator = SmallFactorNet(
            n_features   = len(dataset.features),
            n_outputs    = sum(sl.stop - sl.start for k, sl in outputs_transformer.slices.items() if k not in groups['optimized_shared_keys']),
            hidden_dim   = calib_cfg['hidden_dim'],
            dropout_rate = calib_cfg['dropout_rate'],
        ).to(device)
        trainer = NFMFactorCalibratorTrainer(
            PSF_model, calibrator, dataset, outputs_transformer,
            factor_keys           = groups['factor_keys'],
            optimized_shared_keys = groups['optimized_shared_keys'],
            Cn2_targets           = calib_cfg['Cn2_targets'],
            **trainer_kwargs,
        )

    else:
        raise ValueError(f"Unknown calibrator type '{calib_cfg['type']}', expected 'direct' or 'factor'")

    trainer.training_config = deepcopy(cfg) # the resolved training config, stored in the bundle (save_calibrator) for the validation
    return PSF_model, calibrator, outputs_transformer, trainer


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
        Initialised PSF model, its optimizable inputs the calibrator outputs and its fitted shared keys set (build_NFM_trainer).
    calibrator : nn.Module
        Network mapping a [B, N_features] telemetry tensor -> [B, N_outputs] stacked params.
    dataset : NFMDataset (or compatible)
        Full labelled dataset, split by the train / val indices stored in its cache (default_split) unless train_idx / val_idx are given.
    outputs_transformer : InputsTransformer
        Knows how to stack / unstack the calibrator output vector.
    device : torch.device
    batch_size, lr, lambda_step : int / float / int
        Basic training hyper-parameters.
    fitted_keys : list[str]
        Inputs set per sample to the values of the fitted dataset (e.g. the astrometry dx_ctrl / dy_ctrl), not optimized.
    fitted_shared_keys : list[str]
        Inputs set in the PSF model to one dataset-wide value (the medians of the fits, see build_NFM_trainer), stored in the bundle.
    weights_dir : Path | None
        Directory for checkpoints / bundles (default: WEIGHTS_FOLDER/NFM_calibrator).
    train_idx, val_idx : array | None
        Train / val split (dataset positions); default: the one stored in the dataset cache.
    """
    calibrator_type = 'direct'

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
        fitted_keys          = (),
        fitted_shared_keys   = (),
        weights_dir          = None,
        train_idx            = None,
        val_idx              = None,
        collate_fn: Optional[Callable] = None,
        loss_cfg: Optional[dict] = None,
    ):
        self.PSF_model           = PSF_model
        self.calibrator          = calibrator
        self.dataset             = dataset
        self.outputs_transformer = outputs_transformer
        self.device              = device
        self.batch_size          = batch_size
        self.lr                  = lr
        self.fitted_keys         = list(fitted_keys)
        self.fitted_shared_keys  = list(fitted_shared_keys)
        self.collate_fn          = collate_fn
        self.weights_dir         = Path(weights_dir) if weights_dir else WEIGHTS_FOLDER / 'NFM_calibrator'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.N_wvl_total = dataset.C
        self.H, self.W   = dataset.H, dataset.W
        self.N_features  = len(dataset.features)
        self.N_outputs   = outputs_transformer.get_stacked_size()

        # PSF loss: 'legacy' (linear MSE + MAE) or 'balanced' (see BalancedPSFLoss), optionally mixed with the legacy term
        self.loss_cfg = dict(loss_cfg or {}) # as given, stored in the bundle for provenance
        loss_cfg = dict(self.loss_cfg)
        self.LO_sign_penalty = loss_cfg.pop('LO_sign_penalty', True)
        loss_type = loss_cfg.pop('type', 'legacy')
        self.legacy_loss_weight = loss_cfg.pop('legacy_weight', 0.0)
        # Parameter-space regularization: standardized distance of the predictions to the per-sample fits (the pretraining
        # criterion) added to the PSF loss, and, for the factor calibrator, shrinkage of the corrections towards the prior
        self.fit_reg_weight = loss_cfg.pop('fit_reg_weight', 0.0)
        self.shrink_weight  = loss_cfg.pop('shrink_weight',  0.0)
        self._fit_inv_var   = None
        if loss_type == 'balanced':
            self.balanced_loss = BalancedPSFLoss(self.H, self.W, device, **loss_cfg)
        elif loss_type == 'legacy':
            self.balanced_loss = None
        else:
            raise ValueError(f"Unknown loss type '{loss_type}', expected 'legacy' or 'balanced'")

        # Train / val split and loaders
        if train_idx is not None and val_idx is not None:
            self.train_idx = np.asarray(train_idx, dtype=int)
            self.val_idx   = np.asarray(val_idx,   dtype=int)
        else:
            self.train_idx, self.val_idx = default_split(dataset)
        self.train_loader = self._make_loader(self.train_idx, shuffle=True)
        self.val_loader   = self._make_loader(self.val_idx,   shuffle=False)

        # Wavelength scheduling
        self.lambda_full    = PSF_model.λ_sim.clone()
        self.lambda_id_sets = [list(range(dataset.C))] if lambda_step == 1 else self._build_wavelength_sets(lambda_step, dataset.C-1)
        self.lambda_sets    = [self.lambda_full[ids] for ids in self.lambda_id_sets]

        # Inputs taken from the PSF model and broadcast to the batch at every forward: the fitted shared keys and, among the inputs left to the
        # model, the ones its forward needs (MODEL_REQUIRED_INPUTS). The other inputs left to the model are not passed (see run_model)
        names = PSF_model.inputs_manager.get_names(optimizable_only=False)
        self.model_keys = [k for k in names if k in self.fitted_shared_keys or
                           (k in MODEL_REQUIRED_INPUTS and k not in outputs_transformer.slices and k not in self.fitted_keys)]

        # Fitted keys: the per-sample values of the fitted dataset, [N_dataset, ...] in the shape of the PSF-model input
        self.fitted = {k: torch.as_tensor(fitted_values(dataset, k), device=device).reshape(len(dataset), *PSF_model.inputs_manager[k].shape[1:])
                       for k in self.fitted_keys}

        # Phase bump (first LO coefficient): always the per-sample fit, it is not predicted
        self.phase_bump = torch.as_tensor(fitted_values(dataset, 'LO_coefs')[:, 0], device=device) if 'LO_coefs' in names else None

        self.optimizer, self.scheduler = self._make_optimizer()

        logger.info(
            f"Trainer ready | dataset={len(dataset)} ({len(self.train_idx)} train / {len(self.val_idx)} val) | " + self._kc_groups_summary() + " | "
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


    def _kc_groups_summary(self):
        groups = self.dataset.kc_group
        if len(np.unique(groups)) == 1:
            return 'k_c groups: none (single k_c)'
        
        parts = []
        for g in np.unique(groups):
            frac = float(np.mean(np.asarray(self.dataset.kc_fraction)[np.asarray(groups) == g]))
            parts.append(f"{g}: {int(np.sum(np.asarray(groups)[self.train_idx] == g))}+{int(np.sum(np.asarray(groups)[self.val_idx] == g))} @ k_c {100 * frac:.1f}%")
        
        return 'k_c groups (train+val): ' + ', '.join(parts)

    def _make_loader(self, indices, shuffle):
        collate_fn = self.collate_fn
        if collate_fn is not None:
            collator = lambda b: collate_fn(b, device=self.device)
            
        elif hasattr(self.dataset, 'collate_batch'):
            collator = lambda b: self.dataset.collate_batch(b, device=self.device)
        else:
            raise ValueError("NFMCalibratorTrainer requires a collate function. Pass collate_fn=... or define dataset.collate_batch(batch, device).")

        indices = [int(i) for i in indices]
        # One k_c group per batch (the PSF model holds a single k_c): the mismatched samples form their own, possibly smaller, batches
        return DataLoader(
            dataset=Subset(self.dataset, indices),
            batch_sampler=KcGroupBatchSampler(np.asarray(self.dataset.kc_group)[indices], self.batch_size, shuffle),
            num_workers=0,
            collate_fn=collator,
        )


    def _trainable_tensors(self):
        """Network parameters (plus the dataset-level constants of the factor calibrator)."""
        return list(self.calibrator.parameters())


    def _extra_param_groups(self, lr):
        return []


    def _make_optimizer(self, lr=None):
        lr     = lr or self.lr
        
        groups = [{'params': self.calibrator.parameters(), 'lr': lr, 'weight_decay': 5e-4}]
        groups += self._extra_param_groups(lr)
        opt = optim.AdamW(groups, lr=lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=7, min_lr=1e-6)
      
        return opt, sch


    # ── PSF forward ───────────────────────────────────────────────────────────
    def run_model(self, x_dict_NN, config, idx, lambda_ids):
        """Run the PSF model for a batch at the given wavelength indices."""
        # NOTE: idx are dataset ids (global sample ids), not per-batch local offsets.
        B = len(idx)
        self.PSF_model.model.N_obs = B
        manager = self.PSF_model.inputs_manager

        # The fitted shared keys and the required inputs left to the model: dataset-wide values in the PSF model, stored with the N_obs the
        # model was built with, broadcast to the batch. The other inputs left to the model are not passed: TipTorch re-initializes them for
        # every batch from its config or to its defaults (Update below)
        x_dict = {k: manager[k][:1].expand(B, *manager[k].shape[1:]).clone() for k in self.model_keys}
        # The calibrator outputs (predicted, factor and optimized shared keys) and the per-sample fits of the fitted keys
        x_dict.update({k: v for k, v in x_dict_NN.items() if k in self.outputs_transformer.slices})
        x_dict.update({k: v[idx] for k, v in self.fitted.items()})

        # The phase bump (first LO coefficient) is the per-sample fit; the calibrator outputs carry the LO coefficients without it
        if 'LO_coefs' in x_dict:
            NCPAs = x_dict['LO_coefs'] if 'LO_coefs' in self.outputs_transformer.slices else x_dict['LO_coefs'][:, 1:]
            x_dict['LO_coefs'] = torch.hstack((self.phase_bump[idx].unsqueeze(-1), NCPAs))

        # Predict only a subset of wavelengths at once to reduce peak GPU memory.
        wvl = self.lambda_full[lambda_ids].to(device=self.device)
        config['sources_science']['Wavelength'] = wvl.view(1, -1)
        self.PSF_model.model.config = config
        
        # Update TipTorch internal config to match the new wavelength set. It will trigger the iternal update
        # of the PSF model state. But if it's only one wavelengths set, then trigger the update manually.
        # Grids don't need to be updated since they are defined at the full wavelength resolution, and the PSF model will slice them accordingly.
        # Pupils don't depend on wavelength, so no need to update them either.
        # Batches of samples flagged 'k_c mismatch' carry a scaled DM pitch: a different k_c changes the AO-corrected frequency grid and
        # the masks (InitGrids), the tomographic projector and the static OTF, so the model is rebuilt with grids=True when it changes
        pitch_b    = config['DM']['DmPitchs']
        pitch_b    = float(pitch_b.min()) if isinstance(pitch_b, torch.Tensor) else float(np.min(pitch_b))
        kc_changed = abs(pitch_b - float(self.PSF_model.model.pitch)) > 1e-9

        if (len(self.lambda_id_sets) == 1):
            self.PSF_model.model.Update(grids=kc_changed, pupils=False, tomography=True)
        else:
            if kc_changed: # SetWavelengths skips the update when the wavelengths did not change
                self.PSF_model.model.Update(grids=True, pupils=False, tomography=True)
            self.PSF_model.SetWavelengths(wvl)
        
        # Do not update inputs_manager with graph-connected tensors to avoid cross-batch graph retention.
        return self.PSF_model(x_dict, update_params=False)


    # ── Loss ──────────────────────────────────────────────────────────────────
    def _legacy_PSF_loss(self, PSF_data, PSF_pred):
        diff = PSF_pred - PSF_data
        # Empirical scaling to keep PSF reconstruction and regularization terms numerically balanced.
        w = 2e4
        return w * (diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0 + diff.abs().mean(dim=(1, 2, 3)) * 1.6)


    # ── Prediction hooks (overridden by the factor calibrator) ────────────────
    def _params_from_pred(self, x_pred, tel_b):
        """Network outputs [B, N_outputs] -> PSF-model parameters of the batch (phase bump excluded)."""
        return self.outputs_transformer.unstack(x_pred)


    def _predict(self, tel_b):
        x_pred = self.calibrator(tel_b)
        return x_pred, self._params_from_pred(x_pred, tel_b)


    def net_slices(self) -> dict:
        """{key: slice of the network output}: the stacked layout of the outputs transformer (the factor network predicts only some keys)."""
        return self.outputs_transformer.slices


    @property
    def N_net_outputs(self) -> int:
        return max((sl.stop for sl in self.net_slices().values()), default=0)


    def _params_to_normalized(self, x_dict):
        """PSF-model parameters (phase bump excluded) -> stacked vector in the normalized space of the outputs transformer,
        i.e. the direct calibrator's output space, for plots comparable between the direct and the factor calibrators."""
        B = next(iter(x_dict.values())).shape[0]
        y = torch.zeros(B, self.N_outputs, device=self.device, dtype=torch.float32)
        for key, sl in self.outputs_transformer.slices.items():
            if key in x_dict:
                val = x_dict[key]
                y[:, sl] = self.outputs_transformer.transforms[key](val.unsqueeze(-1) if val.dim() == 1 else val)
        return y


    def _train_fit_dict(self):
        """Fitted values and telemetry vectors of the whole training set on the device (no PSF cubes loaded)."""
        idx = [int(i) for i in self.train_idx]
        fit = {k: torch.stack([torch.as_tensor(self.dataset.fitted_vals[i][k], dtype=torch.float32) for i in idx]).to(self.device)
               for k in self.dataset.fitted_vals[idx[0]]}
        tel = torch.stack([self.dataset[i][1] for i in idx]).to(self.device)
        return fit, tel


    def _train_fit_targets(self):
        """Pretraining targets of the whole training set, [N_train, N_outputs]."""
        fit, tel = self._train_fit_dict()
        with torch.no_grad():
            return self._pretrain_target_vector(fit, device=self.device, tel=tel)


    def _regularization(self, x_pred, x_dict, fit_b, tel_b):
        """Per-sample parameter-space regularization [B] added to the PSF loss (zero unless configured)."""
        reg = torch.zeros(x_pred.shape[0], device=x_pred.device, dtype=x_pred.dtype)
        if self.fit_reg_weight > 0:
            if self._fit_inv_var is None:
                self._fit_inv_var = 1.0 / self._train_fit_targets().var(dim=0).clamp(min=1e-4)
            with torch.no_grad():
                y_fit = self._pretrain_target_vector(fit_b, device=x_pred.device, tel=tel_b)
            reg = reg + self.fit_reg_weight * ((x_pred - y_fit).pow(2) * self._fit_inv_var).mean(dim=-1)
        return reg


    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss_per_sample(self, PSF_data, PSF_pred, x_dict_pred):
        # Per-sample PSF reconstruction loss.
        if self.balanced_loss is None:
            PSF = self._legacy_PSF_loss(PSF_data, PSF_pred)
        else:
            PSF = self.balanced_loss(PSF_data, PSF_pred)
            if self.legacy_loss_weight > 0:
                PSF = PSF + self.legacy_loss_weight * self._legacy_PSF_loss(PSF_data, PSF_pred)
        if 'LO_coefs' in x_dict_pred: # predicted (or optimized shared) LO coefficients
            c  = x_dict_pred['LO_coefs'] # predicted modes without the phase bump: [defocus, astig., astig., coma, ...]
            # L2 regularization of predicted LO modes
            LO = c.pow(2).sum(-1) * 1e-7
            if self.LO_sign_penalty:
                # Legacy sign penalties on defocus (c[:, 0]) and the first astigmatism (c[:, 2]). Note that the phase bump makes
                # the sign of defocus identifiable, so forcing it positive biases the solution (flipping the sign of negative fitted
                # defocus values increases the PSF loss by ~20%).
                LO = LO + torch.clamp(-c[:, 0], min=0).pow(2) * 5e-5
                LO = LO + torch.clamp(-c[:, 2], min=0).pow(2) * 1e-7
        else:
            LO = torch.zeros(PSF_data.shape[0], device=PSF_data.device, dtype=PSF_data.dtype)
        return PSF + LO


    def loss_fn(self, PSF_data, PSF_pred, x_dict_pred, sample_weights=None, return_per_sample=False, reg=None):
        per = self._loss_per_sample(PSF_data, PSF_pred, x_dict_pred)
        if reg is not None:
            per = per + reg # parameter-space regularization, see _regularization
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
        data.update(self._extra_state())
        torch.save(data, path)
        logger.debug(f"Checkpoint saved -> {path}")


    def _extra_state(self) -> dict:
        """Additional tensors stored in the checkpoints and bundles (the factor calibrator's constants)."""
        return {}


    def _load_extra_state(self, state: dict) -> None:
        pass


    def load_checkpoint(self, path, load_optimizer=True):
        path = Path(path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return None, None, None
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.calibrator.load_state_dict(ckpt['calibrator_state_dict'])
        self._load_extra_state(ckpt)
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
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
            # Clamped at 1e-3: log(1e-6) targets of the inverse-softmax transform are extreme outliers for a negligible weight
            cn2 = x['Cn2_weights'].clamp(min=1e-3)
            GL  = (1.0 - cn2.sum(-1, keepdim=True)).clamp(min=1e-3)
            x['Cn2_weights'] = torch.hstack((GL, cn2))
        return x


    def _pretrain_target_vector(self, fitted_vals, device, tel=None):
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


    def pretrain(self, num_epochs=100, patience=15, lr=None, save_path=None, standardize=True):
        """
        Pretrain the calibrator to predict fitted parameter values (no PSF model involved).
        Provides a warm start before end-to-end PSF training.

        With standardize=True, the squared error of every output is divided by the variance of its target over the
        training set. Otherwise, the MSE is dominated by the outputs with the largest spread in the normalized space
        (Cn2 log-weights and LO coefficients, ~85% of the target variance, both barely predictable from telemetry),
        while r0, J and F, which matter most for the PSF, get only a few percent of the weight.
        """
        lr        = lr or self.lr
        save_path = save_path or (self.weights_dir / 'pretrain_weights.pth')
        # Predicted outputs the fits do not carry get a zero (neutral) target silently (_pretrain_target_vector): a regime mismatch between
        # the fits and the config (e.g. a wing_frac_ctrl output with fits made without the wing), say so
        missing = [k for k in self.outputs_transformer.slices if k not in self.dataset.fitted_vals[0]]
        if missing:
            warnings.warn(f"pretrain: the fits of the dataset carry no {missing}; their pretraining targets are 0 (neutral). The regime of the "
                          "fits (STD_fitter.py flags) does not match the \"model\" section of the config: refit the dataset and rebuild the cache")
        opt       = optim.AdamW(self.calibrator.parameters(), lr=lr, weight_decay=5e-4)
        sch       = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-4)
        best_loss, best_state, pat_count = float('inf'), None, 0

        if standardize:
            inv_var = 1.0 / self._train_fit_targets().var(dim=0).clamp(min=1e-4)
        else:
            inv_var = torch.ones(self.N_net_outputs, device=self.device)
        criterion = lambda y_pred, y: ((y_pred - y).pow(2) * inv_var).mean()

        for epoch in range(num_epochs):
            self.calibrator.train()
            train_loss = 0.0
            for _, tel, fit, _, _ in self.train_loader:
                opt.zero_grad()
                y = self._pretrain_target_vector(fit, device=tel.device, tel=tel)
                loss = criterion(self.calibrator(tel), y)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            self.calibrator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, tel, fit, _, _ in self.val_loader:
                    y = self._pretrain_target_vector(fit, device=tel.device, tel=tel)
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
            NN_pred   = torch.zeros((N, self.N_net_outputs))
            tel_vecs  = torch.zeros((N, self.N_features))

        total_loss, total_count = 0.0, 0
        for PSF_data_b, tel_b, fit_b, config_b, idxs_b in loader:
            # Map global dataset ids to local positions in this loader split.
            lpos   = torch.tensor([global_to_local[int(i)] for i in idxs_b.cpu().tolist()], dtype=torch.long)
            x_pred, x_dict = self._predict(tel_b)
            reg = self._regularization(x_pred, x_dict, fit_b, tel_b)

            if return_cubes:
                NN_pred [lpos] = x_pred.cpu()
                tel_vecs[lpos] = tel_b.cpu()

            # Inner loop over wavelength subsets to limit memory use and keep lambda coverage complete.
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b   = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                # Weight by the batch size so that every sample counts equally
                total_loss  += self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, reg=reg).item() * len(idxs_b)
                total_count += len(idxs_b)
                if return_cubes:
                    for wi, li in enumerate(lambda_ids):
                        PSFs_pred[lpos, li] = PSF_pred_b[:, wi].cpu()
                        PSFs_data[lpos, li] = PSF_data_b[:, li].cpu()

            if verbose:
                logger.info(f"Validated batch idxs={idxs_b.tolist()}")

        avg_loss = total_loss / max(total_count, 1)
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

        best_val, patience_counter, nan_count, bad_grad_count = float('inf'), 0, 0, 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.calibrator.train()
            epoch_loss, n_batches, nan_this_epoch = 0.0, 0, False

            for PSF_data_b, tel_b, fit_b, config_b, idxs_b in train_loader:
                self.optimizer.zero_grad()
                batch_loss = 0.0

                # Backpropagate per wavelength subset to reduce peak graph memory.
                for lambda_ids in self.lambda_id_sets:
                    x_pred, x_dict = self._predict(tel_b)
                    reg        = self._regularization(x_pred, x_dict, fit_b, tel_b)
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    sw         = sample_weights[idxs_b] if sample_weights is not None else None
                    loss       = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, sw, reg=reg)

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

                # A NaN in the backward pass of the PSF model (one bad parameter set in the batch) leaves the loss finite but the
                # gradients non-finite: the optimizer step would corrupt the weights and the next forward pass would crash
                # (torch.linalg.lstsq of the tomographic reconstructor). Skip such batches instead.
                grads = [p.grad for p in self._trainable_tensors() if p.grad is not None]
                if not all(torch.isfinite(g).all() for g in grads):
                    bad_grad_count += 1
                    logger.warning(f"Non-finite gradients at epoch {epoch} (sample ids {idxs_b.tolist()}): batch skipped, {bad_grad_count} so far")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Gradient clipping helps keep training stable
                torch.nn.utils.clip_grad_norm_(self._trainable_tensors(), max_norm=1.0)
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
        folds = kfold_splits(self.dataset, base_indices, n_splits, random_state)
        oof_losses = np.full(len(self.dataset), np.nan, dtype=np.float32)
        kfold_dir  = self.weights_dir / 'kfold_difficulty'
        kfold_dir.mkdir(exist_ok=True)

        # Determine which folds need to be trained vs. loaded
        fold_checkpoints = [kfold_dir / f'fold_{fold_id:02d}_best.pth' for fold_id in range(len(folds))]
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

        for fold_id, (tr_rel, va_rel) in enumerate(folds):
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
        cv_r2 = cross_val_score(rf, X, y, cv=folds, scoring='r2')
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
        for PSF_data_b, tel_b, fit_b, config_b, idxs_b in loader:
            x_pred, x_dict = self._predict(tel_b)
            reg = self._regularization(x_pred, x_dict, fit_b, tel_b)
            
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                per = self.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict, return_per_sample=True, reg=reg).cpu().numpy()
                
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

        tuned_params : list[str] - keys predicted by the network (net_slices); the others (constants of the factor calibrator) are skipped.
        """
        slices = self.net_slices()
        if skipped := [k for k in tuned_params if k not in slices]:
            warnings.warn(f"tune_calibrator: {skipped} are not predicted by the network, skipped")
        tuned_params = [k for k in tuned_params if k in slices]
        if not tuned_params:
            raise ValueError(f"tune_calibrator: none of the tuned parameters is predicted by the network ({list(slices)})")

        ref = deepcopy(self.calibrator)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        mask = torch.zeros(self.N_net_outputs, dtype=torch.bool, device=self.device)
        for k in tuned_params:
            mask[slices[k]] = True
        mask = mask.unsqueeze(0)   # [1, N_net_outputs] for broadcasting

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
                    x_dict  = self._params_from_pred(x_comb, tel_b)
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
        """
        Export a self-contained calibrator bundle loadable by NFMCalibrator: the telemetry resources of the distributed bundle (stages 1-2,
        see data_processing/NFM_bundle.py) with the calibrator on top (stage 3). The telemetry preprocessor and the Cn2 grid are the ones
        of the dataset cache, i.e. the ones the calibrator was trained with. install_calibrator copies the result into the distributed bundle.
        """
        resources = telemetry_resources()
        if 'archive_telemetry' not in resources or 'median_Cn2_profile' not in resources:
            logger.warning(f'{BUNDLE_PATH} has no telemetry resources (LGS / IRLOS tables, median Cn2 profile): the bundle will not be '
                           'sufficient to process new data, pack them with create_STD_dataset.py')
        state = {
            **resources,
            'net_state_dict': self.calibrator.state_dict(),
            'net_arch': {
                'n_features':   self.N_features,
                'n_outputs':    self.N_net_outputs,
                'hidden_dim':   self.calibrator.network[0].out_features,
                'dropout_rate': self.calibrator.network[3].p,
            },
            'input_feature_names':  self.dataset.features,
            'output_feature_names': list(self.outputs_transformer.slices.keys()),
            'outputs_transformer':  self.outputs_transformer.save(),
            'telemetry_preprocessor': self.dataset.telemetry_preprocessor, # TelemetryPreprocessor.state_dict()
            'Cn2_grid':             self.dataset.Cn2_grid, # Cn2Grid.state_dict() of the Cn2 features, None with the K-means features
            'Cn2_binning':          'grid' if self.dataset.Cn2_grid is not None else 'kmeans',
            'LO_modes_max':         self.PSF_model.Z_mode_max,
            'N_spline_nodes':       getattr(self.PSF_model, 'N_wvl_ctrl', None),
            'model_flags':          model_flags(self.PSF_model), # jitter / wing regime of the PSF model the calibrator was trained with
            'fitted_shared':        fitted_shared_values(self.PSF_model, self.fitted_shared_keys), # set in the PSF models NFMCalibrator calibrates
            'training_config':      getattr(self, 'training_config', None), # the resolved training config (build_NFM_trainer): validate_NFM_calibrator.py rebuilds the trainer from it
            'predict_phase_bump':   False,
            'loss_cfg':             self.loss_cfg,
            'calibrator_type':      self.calibrator_type,
        }
        state.update(self._extra_state())
        torch.save(NFMCalibrator._tree_to_cpu(state), path)
        logger.info(f"Calibrator bundle saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Factor calibrator
#
#  The network predicts corrections to prior values instead of absolute parameters, so that a zero output reproduces the prior
#  and the corrections can be shrunk towards it (shrink_weight in the "loss" section of calibrators/configs/NFM_factor_calibrator_config.json):
#    factor keys (r0, L0, wind_speed_single, Cn2_weights): value = base * exp(delta + mean log-ratio), the base derived from
#        the telemetry (telemetry_base_params), L0 from the median of the fits; the Cn2 weights are renormalized to 1.
#        The Cn2 base is the AOF profile reprojected onto the fixed layers of the grid (Cn2_grid_* features) or the K-means
#        fractions; its corrections are pretrained to zero (Cn2_targets: prior), since the fitted profiles are unreliable, or
#        to the fits like the other factor keys (Cn2_targets: fits)
#    predicted keys (dn, J_ctrl):                          value = median of the fits + delta, in the normalized space
#    optimized shared keys (F_ctrl, LO_coefs,              one value for all samples, starting from the median of the training fits and
#      wing_frac_ctrl, J_aniso, J_angle)                   trained end-to-end together with the network, not predicted
#  The network outputs only the predicted and factor keys; the parameters the calibrator returns (with the optimized shared keys), the
#  validation plots and the bundles keep the stacked layout of the outputs transformer. The fitted and fitted shared keys and the inputs
#  left to the model are handled as for the direct calibrator (see KEY_GROUPS).
# ═══════════════════════════════════════════════════════════════════════════════

def _as_float_tensor(value, device) -> torch.Tensor:
    """Tensor or array (as stored in checkpoints / bundles) -> float32 tensor on the device."""
    value = value.detach() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
    return value.to(dtype=torch.float32, device=device)


class SmallFactorNet(SmallCalibratorNet):
    """SmallCalibratorNet whose last layer starts at zero: the initial predictions are exactly the prior."""
    def _initialize_weights(self):
        super()._initialize_weights()
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


class FactorParameterization:
    """
    Maps the corrections predicted by the factor network to PSF-model parameters, and the fits to corrections (pretraining
    targets). The statistics (median of the normalized fits, mean log-ratio fit / base of the factor entries, std of the
    corrections, i.e. the widths of the shrinkage prior) are computed on the training set by fit() and stored in the bundle.

    The optimized shared keys (one dataset-level value each, trained by NFMFactorCalibratorTrainer) are not predicted: params() takes their
    values from `consts`.

    The statistics live in the stacked layout of the outputs transformer [N_outputs]; the network outputs only the predicted and factor entries
    [N_net] (pred_index: their positions in the stacked layout, net_slices: the keys' slices of the network output).
    """
    def __init__(self, slices: dict, features: list, factor_keys, tel_mean, tel_scale, optimized_shared_keys=(),
                 L0_const=None, Y_median=None, log_ratio_mean=None, delta_std=None, Cn2_grid=None, Cn2_targets='fits'):
        self.slices                = {k: (v if isinstance(v, slice) else slice(int(v[0]), int(v[1]))) for k, v in slices.items()}
        self.features              = list(features)
        self.factor_keys           = [k for k in factor_keys           if k in self.slices]
        self.optimized_shared_keys = [k for k in optimized_shared_keys if k in self.slices]
        self.direct_keys           = [k for k in self.slices if k not in self.factor_keys and k not in self.optimized_shared_keys] # predicted_keys
        self.L0_const              = None if L0_const is None else float(L0_const)
        self.N_outputs             = max(sl.stop for sl in self.slices.values())

        net_keys         = [k for k in self.slices if k not in self.optimized_shared_keys] # the predicted and factor keys
        self.pred_index  = torch.tensor([i for k in net_keys for i in range(self.slices[k].start, self.slices[k].stop)], dtype=torch.long)
        self.net_slices, n = {}, 0
        for k in net_keys:
            self.net_slices[k] = slice(n, n + self.slices[k].stop - self.slices[k].start)
            n = self.net_slices[k].stop
        self.N_net       = n

        if Cn2_targets not in ('prior', 'fits'):
            raise ValueError(f"Unknown Cn2_targets '{Cn2_targets}', expected 'prior' or 'fits'")
        self.Cn2_grid    = None if Cn2_grid is None else (Cn2_grid if isinstance(Cn2_grid, Cn2Grid) else Cn2Grid.from_state_dict(Cn2_grid))
        self.Cn2_targets = Cn2_targets
        self.prior_keys  = ['Cn2_weights'] if (Cn2_targets == 'prior' and 'Cn2_weights' in self.factor_keys) else [] # corrections pretrained to 0

        as_t = lambda x, default: default if x is None else _as_float_tensor(x, 'cpu')
        self.tel_mean       = as_t(tel_mean,  None)
        self.tel_scale      = as_t(tel_scale, None)
        self.Y_median       = as_t(Y_median,       torch.zeros(self.N_outputs))
        self.log_ratio_mean = as_t(log_ratio_mean, torch.zeros(self.N_outputs))
        self.delta_std      = as_t(delta_std,      torch.ones(self.N_outputs))


    def to(self, device):
        for name in ('tel_mean', 'tel_scale', 'Y_median', 'log_ratio_mean', 'delta_std', 'pred_index'):
            setattr(self, name, getattr(self, name).to(device))
        return self


    def to_stacked(self, delta: torch.Tensor) -> torch.Tensor:
        """Network outputs [B, N_net] -> corrections in the stacked layout [B, N_outputs], zero at the optimized shared entries."""
        return delta.new_zeros(delta.shape[0], self.N_outputs).index_copy(1, self.pred_index, delta)


    def base_values(self, tel_std: torch.Tensor) -> dict:
        """Telemetry-derived base values from the standardized telemetry vectors [B, N_features]."""
        return telemetry_base_params(tel_std * self.tel_scale + self.tel_mean, self.features, L0_const=self.L0_const, Cn2_grid=self.Cn2_grid)


    def log_ratios(self, fit_phys: dict, base: dict) -> torch.Tensor:
        """log(fit / base) of the factor entries, [B, N_outputs] (zero elsewhere)."""
        B = next(iter(base.values())).shape[0]
        r = torch.zeros(B, self.N_outputs, device=self.Y_median.device)
        for k in self.factor_keys:
            r[:, self.slices[k]] = torch.log(fit_phys[k].reshape(B, -1).clamp(min=1e-4) / base[k].reshape(B, -1).clamp(min=1e-4))
        return r


    def _stacked_targets(self, y_fit: torch.Tensor, fit_phys: dict, base: dict) -> torch.Tensor:
        """Corrections reproducing the fits [B, N_outputs]: y_fit - median (direct), log(fit / base) - mean log-ratio (factor); the entries
        of the optimized shared keys are not network outputs and are dropped by pretrain_targets."""
        delta = y_fit - self.Y_median
        r     = self.log_ratios(fit_phys, base)
        for k in self.factor_keys:
            sl = self.slices[k]
            delta[:, sl] = r[:, sl] - self.log_ratio_mean[sl]
        for k in self.prior_keys:
            delta[:, self.slices[k]] = 0.0
        return delta


    def pretrain_targets(self, y_fit: torch.Tensor, fit_phys: dict, base: dict) -> torch.Tensor:
        """Network outputs reproducing the fits, [B, N_net]."""
        return self._stacked_targets(y_fit, fit_phys, base)[:, self.pred_index]


    def params(self, delta: torch.Tensor, tel_std: torch.Tensor, transformer, consts: dict) -> dict:
        """Network outputs [B, N_net] + standardized telemetry -> PSF-model parameters of the batch (phase bump excluded); consts:
        the optimized shared keys in physical units, {key: [1, n]}."""
        delta = self.to_stacked(delta)
        B     = delta.shape[0]
        x     = transformer.unstack(self.Y_median + delta) # direct keys (the rest is overwritten below)
        base  = self.base_values(tel_std)

        for k in self.factor_keys:
            sl  = self.slices[k]
            val = base[k].reshape(B, -1) * torch.exp(delta[:, sl] + self.log_ratio_mean[sl])
            if k == 'Cn2_weights':
                val = val / val.sum(dim=-1, keepdim=True)
            x[k] = val.squeeze(-1) if sl.stop - sl.start < 2 else val

        for k in self.optimized_shared_keys:
            sl  = self.slices[k]
            val = consts[k].expand(B, -1)
            x[k] = val.squeeze(-1) if sl.stop - sl.start < 2 else val

        return x


    def fit(self, y_fit: torch.Tensor, fit_phys: dict, base: dict):
        """Training-set statistics: median of the normalized fits, mean log-ratio of the factor entries, std of the corrections."""
        self.Y_median       = y_fit.median(dim=0).values
        r                   = self.log_ratios(fit_phys, base)
        self.log_ratio_mean = r.mean(dim=0)
        for k in self.prior_keys: # the prior is the base itself
            self.log_ratio_mean[self.slices[k]] = 0.0
        self.delta_std      = self._stacked_targets(y_fit, fit_phys, base).std(dim=0)
        for k in self.prior_keys: # width of the shrinkage prior: the scatter of the fits around the base
            self.delta_std[self.slices[k]] = r[:, self.slices[k]].std(dim=0)
        self.delta_std      = self.delta_std.clamp(min=1e-3)
        return self


    def state_dict(self) -> dict:
        return {
            'slices':         {k: [sl.start, sl.stop] for k, sl in self.slices.items()},
            'features':       self.features,
            'factor_keys':    self.factor_keys,
            'optimized_shared_keys': self.optimized_shared_keys,
            'L0_const':       self.L0_const,
            'tel_mean':       self.tel_mean.cpu().numpy(),
            'tel_scale':      self.tel_scale.cpu().numpy(),
            'Y_median':       self.Y_median.cpu().numpy(),
            'log_ratio_mean': self.log_ratio_mean.cpu().numpy(),
            'delta_std':      self.delta_std.cpu().numpy(),
            'Cn2_grid':       None if self.Cn2_grid is None else self.Cn2_grid.state_dict(),
            'Cn2_targets':    self.Cn2_targets,
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(**state)


class NFMFactorCalibrator(NFMCalibrator):
    """NFMCalibrator for a factor-calibrator bundle: the network outputs are corrections, see FactorParameterization."""
    bundle_type = 'factor'
    def __init__(self, checkpoint_path, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(checkpoint_path, device)
        state = self._bundle_state
        self.parameterization = FactorParameterization.from_state_dict(state['factor_parameterization']).to(device)
        self.consts = {k: _as_float_tensor(v, device) for k, v in state['consts'].items()}

    def _params_from_pred(self, x_pred: torch.Tensor, telemetry_vector: torch.Tensor) -> dict:
        return self.parameterization.params(x_pred, telemetry_vector, self.outputs_transformer, self.consts)


def load_NFM_calibrator(bundle_path, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """NFMCalibrator or NFMFactorCalibrator, depending on the type stored in the bundle."""
    state = torch.load(bundle_path, map_location='cpu', weights_only=False)
    calibrator_class = NFMFactorCalibrator if state['calibrator_type'] == 'factor' else NFMCalibrator
    return calibrator_class(bundle_path, device=device)


class NFMFactorCalibratorTrainer(NFMCalibratorTrainer):
    """
    NFMCalibratorTrainer for the factor calibrator: the same data, PSF loss, pretraining, k-fold weighting and bundle
    machinery, but the network predicts corrections (FactorParameterization) of the predicted and factor keys only, and the optimized
    shared keys are dataset-level parameters trained together with the network (at a tenth of the learning rate).

    factor_keys, optimized_shared_keys, Cn2_targets : see the "calibrator" section of
    calibrators/configs/NFM_factor_calibrator_config.json (calibrator_key_groups, build_NFM_trainer); every output that is not an optimized
    shared key is a network output (FactorParameterization.N_net), the predicted keys as corrections to the median of the fits. The optimized
    shared keys start from the median of the training fits and are trained in the normalized space of the outputs transformer
    (optimized_params), where one learning rate suits them all (J_angle in degrees as well as wing_frac ~ 0.005); checkpoints and bundles
    store them in physical units ('consts').
    """
    calibrator_type = 'factor'

    def __init__(self, PSF_model, calibrator, dataset, outputs_transformer, *,
                 factor_keys=(), optimized_shared_keys=(), Cn2_targets='fits', **kwargs):
        preprocessor = TelemetryPreprocessor.from_state_dict(dataset.telemetry_preprocessor) # the base values are derived from the telemetry
        self.parameterization = FactorParameterization(outputs_transformer.slices, dataset.features, factor_keys, preprocessor.mean, preprocessor.scale,
                                                       optimized_shared_keys=optimized_shared_keys, Cn2_grid=dataset.Cn2_grid, Cn2_targets=Cn2_targets)
        if calibrator.network[-1].out_features != self.parameterization.N_net:
            raise ValueError(f'The factor network has {calibrator.network[-1].out_features} outputs, the predicted and factor keys '
                             f'{list(self.parameterization.net_slices)} need {self.parameterization.N_net}')
        self.optimized_params = {} # {key: nn.Parameter [1, n]}, the optimized shared keys in the normalized space of the outputs transformer
        super().__init__(PSF_model, calibrator, dataset, outputs_transformer, **kwargs)
        self.parameterization.to(self.device)

        # Training-set statistics of the parameterization and the starting point of the optimized shared keys (medians of the fits)
        fit, tel = self._train_fit_dict()
        fit_phys = self._calibrator_friendly_fit_dict(fit)
        self.parameterization.L0_const = float(fit['L0'].median()) if 'L0' in self.parameterization.factor_keys else None
        
        with torch.no_grad():
            y_fit = super()._pretrain_target_vector(fit, device=self.device)
            self.parameterization.fit(y_fit, fit_phys, self.parameterization.base_values(tel))
        
        for k in self.parameterization.optimized_shared_keys:
            if k not in fit_phys:
                raise KeyError(f"The fits of the dataset carry no '{k}', its shared value cannot start from their median: refit the dataset in "
                               "the regime of the config and rebuild the cache")
           
            median = fit_phys[k].reshape(len(tel), -1).median(dim=0).values.unsqueeze(0).detach().clone()
            self.optimized_params[k] = nn.Parameter(self.outputs_transformer.transforms[k](median))

        self.optimizer, self.scheduler = self._make_optimizer() # now including the optimized shared keys

        logger.info(f"Factor calibrator | network outputs: predicted {self.parameterization.direct_keys}, factor {self.parameterization.factor_keys} "
                    f"({self.parameterization.N_net} outputs) | optimized shared: {self.parameterization.optimized_shared_keys} | "
                    f"fitted: {self.fitted_keys} | fitted shared: {self.fitted_shared_keys} | L0 = {self.parameterization.L0_const} | "
                    f"Cn2: {self.parameterization.Cn2_grid or 'K-means features'}, targets {self.parameterization.Cn2_targets}")


    def optimized_shared_values(self) -> dict:
        """The optimized shared keys in physical units, {key: [1, n]} (differentiable w.r.t. optimized_params)."""
        return {k: self.outputs_transformer.transforms[k].inverse(p) for k, p in self.optimized_params.items()}


    # ── Hooks ──────────────────────────────────────────────────────────────────
    def net_slices(self) -> dict:
        return self.parameterization.net_slices


    def _pretrain_target_vector(self, fitted_vals, device, tel=None):
        if tel is None:
            raise ValueError('The factor calibrator needs the telemetry vectors to compute its targets (base values)')
        y_fit = super()._pretrain_target_vector(fitted_vals, device)
        return self.parameterization.pretrain_targets(y_fit, self._calibrator_friendly_fit_dict(fitted_vals), self.parameterization.base_values(tel))


    def _params_from_pred(self, x_pred, tel_b):
        return self.parameterization.params(x_pred, tel_b, self.outputs_transformer, self.optimized_shared_values())


    def _regularization(self, x_pred, x_dict, fit_b, tel_b):
        reg = super()._regularization(x_pred, x_dict, fit_b, tel_b)
        if self.shrink_weight > 0:
            # Gaussian prior on the corrections, widths = training-set scatter of the corrections reproducing the fits
            P   = self.parameterization
            reg = reg + self.shrink_weight * (x_pred / P.delta_std[P.pred_index]).pow(2).mean(dim=-1)
        return reg


    def _dataset_level_params(self):
        return list(self.optimized_params.values())


    def _trainable_tensors(self):
        return list(self.calibrator.parameters()) + self._dataset_level_params()


    def _extra_param_groups(self, lr):
        groups = []
        if self.optimized_params:
            groups.append({'params': list(self.optimized_params.values()), 'lr': lr * 0.1, 'weight_decay': 0.0})
        return groups


    def _extra_state(self) -> dict:
        with torch.no_grad():
            consts = {k: v.cpu().clone() for k, v in self.optimized_shared_values().items()}
        return {
            'consts':                 consts, # physical units, as NFMFactorCalibrator uses them
            'factor_parameterization': self.parameterization.state_dict(),
        }


    def _load_extra_state(self, state: dict) -> None:
        for k, v in state.get('consts', {}).items(): # the optimized shared keys in physical units
            if k in self.optimized_params:
                self.optimized_params[k].data = self.outputs_transformer.transforms[k](_as_float_tensor(v, self.device).reshape(1, -1))
        if 'factor_parameterization' in state:
            self.parameterization = FactorParameterization.from_state_dict(state['factor_parameterization']).to(self.device)
