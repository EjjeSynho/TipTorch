import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))

import gc
import warnings
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

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
CONFIGS_FOLDER    = Path(__file__).resolve().parent / 'configs'
STD_DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache'


# ── Helpers ───────────────────────────────────────────────────────────────────
def release_gpu_memory(sync=False):
    """Best-effort VRAM cleanup without affecting model state."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()


def _to_cpu(obj):
    """Detached CPU copies of the tensors of a nested dict / list / tuple, deep copies of the rest (portable serialization)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return deepcopy(obj)


def _as_float_tensor(value, device) -> torch.Tensor:
    """Tensor or array (as stored in checkpoints / bundles) -> float32 tensor on the device."""
    value = value.detach() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
    return value.to(dtype=torch.float32, device=device)


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

    if Cn2_grid is not None and all(name in features for name in Cn2_grid.feature_names):
        Cn2 = torch.stack([col(name) for name in Cn2_grid.feature_names], dim=-1).clamp(min=1e-3) # imputed profiles are not normalized
    else:
        f2  = col('Cn2_frac_binned_2').clamp(1e-3, 0.98)
        f3  = col('Cn2_frac_binned_3').clamp(1e-3, 0.98)
        Cn2 = torch.stack([(1.0 - f2 - f3).clamp(min=1e-3), f2, f3], dim=-1)

    base = {
        'r0':                r0_from_seeing(col('Seeing (header)').clamp(0.2, 3.0), 500e-9), # [m]
        'wind_speed_single': col('Wind speed (header)').clamp(0.5, 30.0)[:, None],         # [m/s]
        'Cn2_weights':       Cn2 / Cn2.sum(dim=-1, keepdim=True),
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


def kfold_splits(dataset, base_indices, n_splits, random_state=42):
    """
    K-fold splits of base_indices (relative positions), grouped by acquisition time (the time groups of the dataset cache), so that
    samples close in time stay in the same fold.
    """
    groups = np.asarray(dataset.time_groups)[base_indices]
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))), shuffle=True, random_state=random_state)
    return list(gkf.split(base_indices, groups=groups))


# ── Dataset ───────────────────────────────────────────────────────────────────
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
        self.n_batches  = int(sum(np.ceil(np.sum(self.groups == g) / self.batch_size) for g in np.unique(self.groups)))

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
        self.configs     = [c[0] if isinstance(c, list) else c for c in data['model_configs']]
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
        for fit in self.fitted_vals:
            for p in ('J', 'wing_frac', 'F', 'bg', 'dx', 'dy'):
                if p in fit and p + '_ctrl' not in fit:
                    fit[p + '_ctrl'] = fit.pop(p)

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


# ── Networks ──────────────────────────────────────────────────────────────────
class SmallCalibratorNet(nn.Module):
    """Compact network for small datasets (~500 samples): two hidden layers with LayerNorm (better than BatchNorm for small batches),
    ReLU and dropout, small initial weights for regularization."""
    def __init__(self, n_features, n_outputs, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_outputs),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


class SmallFactorNet(SmallCalibratorNet):
    """SmallCalibratorNet whose last layer starts at zero: the initial predictions are exactly the prior."""
    def _initialize_weights(self):
        super()._initialize_weights()
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


# ── Inference ─────────────────────────────────────────────────────────────────
class NFMCalibrator:
    """
    A trained calibrator bundle (NFMCalibratorTrainer.save_calibrator): telemetry -> PSF-model parameters, set in a PSFModelNFM by
    calibrate(). load_NFM_calibrator picks this class or NFMFactorCalibrator from the bundle.
    """
    bundle_type = 'direct' # the calibrator_type of the bundles this class loads

    def __init__(self, checkpoint_path, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device          = device
        self.checkpoint_path = Path(checkpoint_path)
        self._bundle_state   = state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # A factor bundle loaded with this class would return the neutral values of the normalization for every parameter (its network
        # outputs are corrections, see NFMFactorCalibrator)
        if state['calibrator_type'] != self.bundle_type:
            raise TypeError(f"{type(self).__name__} cannot load a '{state['calibrator_type']}' calibrator bundle ({self.checkpoint_path.name}): "
                            "use load_NFM_calibrator(bundle_path, device), which selects NFMCalibrator / NFMFactorCalibrator from the bundle")

        self.net = SmallCalibratorNet(**state['net_arch']).to(device)
        self.net.load_state_dict(state['net_state_dict'])
        self.net.eval()

        self.outputs_transformer    = InputsTransformer.load(data=state['outputs_transformer'])
        self.telemetry_preprocessor = load_telemetry_preprocessor(state) # scaler + imputer of the inputs
        self.features               = self.telemetry_preprocessor.features

        # Cn2 profile features: the fixed grid (Cn2_grid_*) or the K-means ones, whose median Cn2 profile is None when the telemetry resources
        # were not packed (the one of the distributed bundle is used then)
        self.Cn2_grid           = Cn2Grid.from_state_dict(state['Cn2_grid']) if state['Cn2_grid'] is not None else None
        self.Cn2_binning        = state['Cn2_binning']
        self.median_Cn2_profile = state.get('median_Cn2_profile')

        self.LO_modes_max    = state['LO_modes_max']
        self.N_spline_nodes  = state['N_spline_nodes'] # None: PSF model without splines
        self.model_flags     = state['model_flags']
        self.fitted_shared   = state['fitted_shared']  # {key: tensor [1, ...]}, see fitted_shared_values()
        self.training_config = state['training_config']


    def save(self, path, **metadata):
        """
        Bundle with the current network weights and telemetry preprocessor, the loaded bundle as the template (e.g. a calibrator fine-tuned
        on science data, stored apart from the STD-trained one); `metadata` goes into its 'fine_tuning' entry.
        """
        state = {**self._bundle_state, 'net_state_dict': self.net.state_dict(), 'telemetry_preprocessor': self.telemetry_preprocessor.state_dict()}
        if metadata:
            state['fine_tuning'] = metadata
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_to_cpu(state), path)
        logger.info(f"Calibrator bundle saved -> {path}")
        return path


    def prepare_features(self, telemetry_features: pd.DataFrame) -> torch.Tensor:
        """Scale and impute telemetry already reduced to the calibrator features."""
        return torch.tensor(self.telemetry_preprocessor.transform(telemetry_features), dtype=torch.float32, device=self.device)


    def prepare_telemetry(self, reduced_telemetry: pd.DataFrame) -> torch.Tensor:
        """Filter, reduce, scale and impute the telemetry."""
        telemetry_pruned = reduce_dataframe(
            filter_dataframe(reduced_telemetry),
            Cn2_binning=self.Cn2_binning, Cn2_grid=self.Cn2_grid,
            median_Cn2_profile=self.median_Cn2_profile
        )
        return self.prepare_features(telemetry_pruned)


    def forward_prepared(self, telemetry_vector) -> dict:
        """PSF-model parameters from telemetry already scaled and imputed, [N, N_features] or one vector; LO_coefs with a zero phase bump."""
        tel = torch.as_tensor(telemetry_vector, dtype=torch.float32, device=self.device)
        tel = tel.unsqueeze(0) if tel.ndim == 1 else tel
        if tel.ndim != 2 or tel.shape[-1] != self.net.network[0].in_features:
            raise ValueError(f"Prepared telemetry must have shape [N, {self.net.network[0].in_features}], got {tuple(tel.shape)}.")

        x = self._params_from_pred(self.net(tel), tel)
        if 'LO_coefs' in x: # the phase bump is not predicted
            x['LO_coefs'] = torch.cat([torch.zeros_like(x['LO_coefs'][:, :1]), x['LO_coefs']], dim=-1)
        return x


    def _params_from_pred(self, x_pred: torch.Tensor, telemetry_vector: torch.Tensor) -> dict:
        """Network outputs [N, N_outputs] -> PSF-model parameters (phase bump excluded)."""
        return self.outputs_transformer.unstack(x_pred)


    def forward(self, reduced_telemetry: pd.DataFrame) -> dict:
        """PSF-model parameters from raw telemetry."""
        return self.forward_prepared(self.prepare_telemetry(reduced_telemetry))

    __call__ = forward


    @torch.no_grad()
    def calibrate(self, reduced_telemetry: pd.DataFrame, PSF_model) -> None:
        """Predict the parameters from raw telemetry and set them in PSF_model, adapted to its structure (see _adapt_to_PSF_model)."""
        self._apply_calibration(self.forward(reduced_telemetry), PSF_model)


    @torch.no_grad()
    def calibrate_prepared(self, telemetry_vector: torch.Tensor, PSF_model) -> None:
        """calibrate() for telemetry already scaled and imputed (dataset vectors, batched)."""
        self._apply_calibration(self.forward_prepared(telemetry_vector), PSF_model)


    @torch.no_grad()
    def _apply_calibration(self, x_dict_pred: dict, PSF_model) -> None:
        x_dict = self._adapt_to_PSF_model(x_dict_pred, PSF_model)
        # The fitted shared keys (e.g. the wing shape at the medians of the fits) are dataset-wide values: set them in the PSF model as well.
        # The inputs left to the model and the fitted keys (per-sample fits, unknown for new data) are not touched
        names = PSF_model.inputs_manager.get_names(optimizable_only=False)
        for key, val in self.fitted_shared.items():
            if key in names:
                cur = PSF_model.inputs_manager[key]
                v   = torch.as_tensor(val, dtype=cur.dtype, device=cur.device).reshape(1, -1)
                x_dict[key] = (v.reshape(-1)[:1].expand(cur.shape[0]) if cur.dim() == 1 else v.expand(cur.shape[0], -1).reshape(cur.shape)).clone()
        PSF_model.update_manager_params(x_dict)
        _ = PSF_model()


    def check_compatibility(self, PSF_model) -> dict:
        """Structural mismatches between the calibrator and a PSFModelNFM, {name: (calibrator, model)}, each with a warning."""
        issues = {}
        
        if hasattr(PSF_model, 'Z_mode_max') and self.LO_modes_max != PSF_model.Z_mode_max:
            n_calib, n_model = self.LO_modes_max - 2, PSF_model.Z_mode_max - 2 # predicted modes, phase bump excluded
            issues['LO_modes'] = (n_calib, n_model)
            warnings.warn(f"NFMCalibrator: LO modes mismatch — calibrator predicts {n_calib} Zernike modes, PSF model expects {n_model}. "
                          f"Will adapt by {'cropping' if n_calib > n_model else 'padding with zeros'}.")

        if self.N_spline_nodes is not None and hasattr(PSF_model, 'N_wvl_ctrl') and self.N_spline_nodes != PSF_model.N_wvl_ctrl:
            issues['N_ctrl'] = (self.N_spline_nodes, PSF_model.N_wvl_ctrl)
            warnings.warn(f"NFMCalibrator: N_spline_nodes mismatch — calibrator has {self.N_spline_nodes} control points, PSF model has "
                          f"{PSF_model.N_wvl_ctrl}. Will re-sample via cubic spline.")

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


    def _adapt_to_PSF_model(self, x_dict: dict, PSF_model) -> dict:
        """The predictions with the spline control values re-sampled to the model's number of nodes (natural cubic spline over uniform nodes
        in [0, 1]) and LO_coefs (phase bump included) cropped or zero-padded to its number of modes."""
        x_dict = dict(x_dict) # shallow copy, tensors are not cloned
        n_from, n_to = self.N_spline_nodes, getattr(PSF_model, 'N_wvl_ctrl', None)
        if n_from is not None and n_to is not None and n_from != n_to:
            for key, val in x_dict.items():
                if key.endswith('_ctrl') and torch.is_tensor(val) and val.dim() == 2 and val.shape[-1] == n_from:
                    t_from, t_to = (torch.linspace(0, 1, n, device=val.device, dtype=val.dtype) for n in (n_from, n_to))
                    x_dict[key] = NaturalCubicSpline(natural_cubic_spline_coeffs(t_from, val.T)).evaluate(t_to).T # the spline takes [n_nodes, batch]

        if 'LO_coefs' in x_dict and hasattr(PSF_model, 'Z_mode_max'):
            coefs, n_model = x_dict['LO_coefs'], PSF_model.Z_mode_max - 1 # the modes the model needs, phase bump included
            pad = torch.zeros(coefs.shape[0], max(n_model - coefs.shape[-1], 0), device=coefs.device, dtype=coefs.dtype)
            x_dict['LO_coefs'] = torch.hstack((coefs[:, :n_model], pad))
        return x_dict


# ── Training setup ────────────────────────────────────────────────────────────
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
    return PSFModelNFM(config, multiple_obs=True, device=device, **model_cfg)


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
    kc_groups    = np.asarray(dataset.kc_group)
    pool         = np.flatnonzero(kc_groups == np.bincount(kc_groups).argmax())
    batch        = tuple(dataset[int(i)] for i in np.random.default_rng(seed).choice(pool, size=min(cfg['batch_size'], len(pool)), replace=False))
    batch_config = dataset.collate_batch(batch, device=device)[3]
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
    train_idx = np.asarray(dataset.train_idx if train_idx is None else train_idx, dtype=int)
    for key in groups['fitted_shared_keys']:
        median = torch.as_tensor(fitted_values(dataset, key, train_idx)).median(dim=0).values
        cur    = PSF_model.inputs_manager[key]
        with torch.no_grad():
            cur[:] = (median[0] if median.numel() == 1 else median.view(1, -1).expand(cur.shape[0], -1).reshape(cur.shape)).to(cur)
        logger.info(f"fitted_shared_keys: {key} = {np.round(median.numpy(), 3).tolist()} (median of the training fits)")

    # Outputs transformer: calibrator predictions <-> PSF-model inputs, without the phase bump column of LO_coefs (stack() sets the slices)
    outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
    buf = outputs_transformer.unstack(PSF_model.inputs_manager.stack())
    if 'LO_coefs' in buf:
        buf['LO_coefs'] = buf['LO_coefs'][:, 1:]
    outputs_transformer.stack(buf)
    del buf

    calib_cfg      = cfg['calibrator']
    net_kwargs     = dict(n_features=len(dataset.features), hidden_dim=calib_cfg['hidden_dim'], dropout_rate=calib_cfg['dropout_rate'])
    trainer_kwargs = dict(device=device, batch_size=cfg['batch_size'], lr=cfg['trainer']['lr'], lambda_step=cfg['trainer']['lambda_step'],
                          fitted_keys=groups['fitted_keys'], fitted_shared_keys=groups['fitted_shared_keys'], loss_cfg=cfg['loss'],
                          weights_dir=weights_dir) | trainer_overrides

    if calib_cfg['type'] == 'direct':
        calibrator = SmallCalibratorNet(n_outputs=outputs_transformer.get_stacked_size(), **net_kwargs).to(device)
        trainer    = NFMCalibratorTrainer(PSF_model, calibrator, dataset, outputs_transformer, **trainer_kwargs)

    elif calib_cfg['type'] == 'factor':
        # The network outputs only the predicted and factor keys; the optimized shared keys are dataset-level parameters of the trainer
        n_net      = sum(sl.stop - sl.start for k, sl in outputs_transformer.slices.items() if k not in groups['optimized_shared_keys'])
        calibrator = SmallFactorNet(n_outputs=n_net, **net_kwargs).to(device)
        trainer    = NFMFactorCalibratorTrainer(PSF_model, calibrator, dataset, outputs_transformer, factor_keys=groups['factor_keys'],
                                                optimized_shared_keys=groups['optimized_shared_keys'], **trainer_kwargs)
    else:
        raise ValueError(f"Unknown calibrator type '{calib_cfg['type']}', expected 'direct' or 'factor'")

    trainer.training_config = deepcopy(cfg) # the resolved training config, stored in the bundle (save_calibrator) for the validation
    return PSF_model, calibrator, outputs_transformer, trainer


# ── Training ──────────────────────────────────────────────────────────────────
class NFMCalibratorTrainer:
    """
    PSF-model-in-the-loop trainer of the direct calibrator (the network predicts the PSF-model parameters in the normalized space of the
    outputs transformer): pretraining on the fitted values, end-to-end training on the PSF loss with early stopping, NaN recovery and
    optional per-sample difficulty weights (k-fold out-of-fold losses), fine-tuning of selected outputs, checkpoints and the bundle export.

    PSF_model           PSFModelNFM, its optimizable inputs the calibrator outputs and its fitted shared keys set (build_NFM_trainer)
    calibrator          network [B, N_features] telemetry -> [B, N_net_outputs]
    dataset             NFMDataset (or compatible), split by the train / val indices of its cache unless train_idx / val_idx are given
    outputs_transformer stacks / unstacks the calibrator outputs
    lambda_step         1: every forward pass at all the wavelengths; > 1: interleaved wavelength subsets, to limit the GPU memory
    fitted_keys         inputs set per sample to the values of the fitted dataset (e.g. the astrometry dx_ctrl / dy_ctrl), not optimized
    fitted_shared_keys  inputs set in the PSF model to one dataset-wide value (the medians of the fits, see build_NFM_trainer)
    loss_cfg            the "loss" section of the training config
    weights_dir         checkpoints / bundles (default: WEIGHTS_FOLDER/NFM_calibrator)
    """
    calibrator_type = 'direct'

    def __init__(self, PSF_model, calibrator, dataset, outputs_transformer, device, *, batch_size=16, lr=1e-2, lambda_step=2, fitted_keys=(),
                 fitted_shared_keys=(), weights_dir=None, train_idx=None, val_idx=None, loss_cfg: dict | None = None):
        self.PSF_model           = PSF_model
        self.calibrator          = calibrator
        self.dataset             = dataset
        self.outputs_transformer = outputs_transformer
        self.device              = device
        self.batch_size          = batch_size
        self.lr                  = lr
        self.fitted_keys         = list(fitted_keys)
        self.fitted_shared_keys  = list(fitted_shared_keys)
        self.optimized_params    = {}   # the optimized shared keys of the factor calibrator, {key: nn.Parameter [1, n]}
        self.training_config     = None # the resolved training config (build_NFM_trainer), stored in the bundle
        self.weights_dir         = Path(weights_dir) if weights_dir else WEIGHTS_FOLDER / 'NFM_calibrator'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.N_wvl_total, self.H, self.W = dataset.C, dataset.H, dataset.W
        self.N_features = len(dataset.features)
        self.N_outputs  = outputs_transformer.get_stacked_size()

        # PSF loss: 'legacy' (linear MSE + MAE) or 'balanced' (BalancedPSFLoss with the rest of loss_cfg), optionally plus the legacy term
        loss_cfg = dict(loss_cfg or {})
        if removed := {k: v for k in ('fit_reg_weight', 'shrink_weight') if (v := loss_cfg.pop(k, 0.0))}: # still in the configs of older runs
            warnings.warn(f'The parameter-space regularization was removed on 2026-10-30, {removed} ignored')
        loss_type               = loss_cfg.pop('type', 'legacy')
        self.LO_sign_penalty    = loss_cfg.pop('LO_sign_penalty', True)
        self.legacy_loss_weight = loss_cfg.pop('legacy_weight', 0.0)
        if loss_type not in ('legacy', 'balanced'):
            raise ValueError(f"Unknown loss type '{loss_type}', expected 'legacy' or 'balanced'")
        self.balanced_loss = BalancedPSFLoss(self.H, self.W, device, **loss_cfg) if loss_type == 'balanced' else None

        # Train / val split (by default the one of the dataset cache, grouped by acquisition time) and loaders
        if train_idx is None or val_idx is None:
            train_idx, val_idx = dataset.train_idx, dataset.val_idx
        self.train_idx, self.val_idx = np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)
        self.train_loader = self._make_loader(self.train_idx, shuffle=True)
        self.val_loader   = self._make_loader(self.val_idx,   shuffle=False)

        # Wavelength sets of the forward passes: all at once, or lambda_step interleaved subsets, each with the first and the last wavelength
        self.lambda_full = PSF_model.λ_sim.clone()
        if lambda_step == 1:
            self.lambda_id_sets = [list(range(dataset.C))]
        else:
            last, self.lambda_id_sets = dataset.C - 1, []
            for offset in range(lambda_step):
                ids = list(range(offset, last + 1, lambda_step))
                if ids[0] > lambda_step // 2:         ids.insert(0, 0)
                if last - ids[-1] > lambda_step // 2: ids.append(last)
                self.lambda_id_sets.append(ids)

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

        logger.info(f"Trainer ready | dataset={len(dataset)} ({len(self.train_idx)} train / {len(self.val_idx)} val) | {self._kc_groups_summary()} | "
                    f"N_features={self.N_features} | N_outputs={self.N_outputs} | lambda_sets={len(self.lambda_id_sets)} x ~{len(self.lambda_id_sets[0])} wvl")


    def _kc_groups_summary(self):
        groups = np.asarray(self.dataset.kc_group)
        if len(np.unique(groups)) == 1:
            return 'k_c groups: none (single k_c)'
        fraction = np.asarray(self.dataset.kc_fraction)
        return 'k_c groups (train+val): ' + ', '.join(f"{g}: {np.sum(groups[self.train_idx] == g)}+{np.sum(groups[self.val_idx] == g)} @ k_c "
                                                      f"{100 * fraction[groups == g].mean():.1f}%" for g in np.unique(groups))


    def _make_loader(self, indices, shuffle):
        """Loader over the dataset positions `indices`, one k_c group per batch (the PSF model holds a single k_c): the mismatched samples
        form their own, possibly smaller, batches."""
        indices = [int(i) for i in indices]
        return DataLoader(Subset(self.dataset, indices), num_workers=0, collate_fn=lambda batch: self.dataset.collate_batch(batch, device=self.device),
                          batch_sampler=KcGroupBatchSampler(np.asarray(self.dataset.kc_group)[indices], self.batch_size, shuffle))


    def _make_optimizer(self):
        """AdamW over the network and, at a tenth of the learning rate and without weight decay, the optimized shared keys of the factor
        calibrator; the learning rate halves on plateaus of the validation loss."""
        groups = [{'params': self.calibrator.parameters(), 'lr': self.lr, 'weight_decay': 5e-4}]
        if self.optimized_params:
            groups.append({'params': list(self.optimized_params.values()), 'lr': self.lr * 0.1, 'weight_decay': 0.0})
        optimizer = optim.AdamW(groups, lr=self.lr)
        return optimizer, optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)


    # ── PSF forward ───────────────────────────────────────────────────────────
    def run_model(self, x_dict_NN, config, idx, lambda_ids):
        """Run the PSF model for a batch (idx: dataset positions, not offsets within the batch) at the given wavelength indices."""
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

        wvl = self.lambda_full[lambda_ids].to(device=self.device)
        config['sources_science']['Wavelength'] = wvl.view(1, -1)
        self.PSF_model.model.config = config

        # The new config updates the internal state of TipTorch: SetWavelengths does it for the subsets, a single set needs a manual Update
        # (the grids are defined at the full wavelength resolution and sliced, the pupils do not depend on the wavelength). Batches of samples
        # flagged 'k_c mismatch' carry a scaled DM pitch: a different k_c changes the AO-corrected frequency grid and the masks (InitGrids),
        # the tomographic projector and the static OTF, so the model is rebuilt with grids=True when it changes
        pitch      = config['DM']['DmPitchs']
        pitch      = float(pitch.min()) if isinstance(pitch, torch.Tensor) else float(np.min(pitch))
        kc_changed = abs(pitch - float(self.PSF_model.model.pitch)) > 1e-9

        if len(self.lambda_id_sets) == 1:
            self.PSF_model.model.Update(grids=kc_changed, pupils=False, tomography=True)
        else:
            if kc_changed: # SetWavelengths skips the update when the wavelengths did not change
                self.PSF_model.model.Update(grids=True, pupils=False, tomography=True)
            self.PSF_model.SetWavelengths(wvl)

        # Do not update inputs_manager with graph-connected tensors to avoid cross-batch graph retention.
        return self.PSF_model(x_dict, update_params=False)


    # ── Predictions and targets (the factor calibrator overrides these hooks) ──
    def net_slices(self) -> dict:
        """{key: slice of the network output}: the stacked layout of the outputs transformer (the factor network predicts only some keys)."""
        return self.outputs_transformer.slices


    @property
    def N_net_outputs(self) -> int:
        return max((sl.stop for sl in self.net_slices().values()), default=0)


    def _params_from_pred(self, x_pred, tel_b):
        """Network outputs [B, N_net_outputs] -> PSF-model parameters of the batch (phase bump excluded)."""
        return self.outputs_transformer.unstack(x_pred)


    def _predict(self, tel_b):
        x_pred = self.calibrator(tel_b)
        return x_pred, self._params_from_pred(x_pred, tel_b)


    def _pretrain_target_vector(self, fitted_vals, tel=None):
        """Network outputs reproducing the fits, [B, N_net_outputs]: the fits in the normalized space of the outputs transformer."""
        return self._params_to_normalized(self._calibrator_friendly_fit_dict(fitted_vals))


    def _extra_state(self) -> dict:
        """Additional entries of the checkpoints and bundles (the parameterization and the constants of the factor calibrator)."""
        return {}


    def _load_extra_state(self, state: dict) -> None:
        pass


    def _params_to_normalized(self, x_dict):
        """
        PSF-model parameters (phase bump excluded) -> stacked vector [B, N_outputs] in the normalized space of the outputs transformer, i.e. the
        direct calibrator's output space; the keys absent from x_dict get 0, the neutral value of the normalization. The vector is filled in the
        existing slices: stack() would reset them to the key order of x_dict and corrupt the later unstack() calls.
        """
        y = torch.zeros(next(iter(x_dict.values())).shape[0], self.N_outputs, device=self.device, dtype=torch.float32)
        for key, sl in self.outputs_transformer.slices.items():
            if key in x_dict:
                val = x_dict[key]
                y[:, sl] = self.outputs_transformer.transforms[key](val.unsqueeze(-1) if val.dim() == 1 else val)
        return y


    def _calibrator_friendly_fit_dict(self, fitted_vals):
        """The fits in the layout of the calibrator outputs, in the order of outputs_transformer.slices: LO_coefs without the phase bump (not
        predicted), the Cn2 weights with the ground layer (the dataset stores the upper layers only)."""
        x = {k: fitted_vals[k] for k in self.outputs_transformer.slices if k in fitted_vals}
        if 'LO_coefs' in x:
            x['LO_coefs'] = x['LO_coefs'][:, 1:]
        if 'Cn2_weights' in x:
            # Clamped at 1e-3: log(1e-6) targets of the inverse-softmax transform are extreme outliers for a negligible weight
            cn2 = x['Cn2_weights'].clamp(min=1e-3)
            x['Cn2_weights'] = torch.hstack(((1.0 - cn2.sum(-1, keepdim=True)).clamp(min=1e-3), cn2))
        return x


    def _train_fit_dict(self):
        """Fitted values and telemetry vectors of the whole training set on the device (no PSF cubes loaded)."""
        idx = [int(i) for i in self.train_idx]
        fit = {k: torch.stack([torch.as_tensor(self.dataset.fitted_vals[i][k], dtype=torch.float32) for i in idx]).to(self.device)
               for k in self.dataset.fitted_vals[idx[0]]}
        tel = torch.stack([self.dataset[i][1] for i in idx]).to(self.device)
        return fit, tel


    # ── Loss ──────────────────────────────────────────────────────────────────
    def _legacy_PSF_loss(self, PSF_data, PSF_pred):
        diff = PSF_pred - PSF_data
        # Empirical scaling to keep PSF reconstruction and regularization terms numerically balanced.
        return 2e4 * (diff.pow(2).mean(dim=(1, 2, 3)) * 1200.0 + diff.abs().mean(dim=(1, 2, 3)) * 1.6)


    def _loss_per_sample(self, PSF_data, PSF_pred, x_dict_pred):
        """PSF loss [B] plus the L2 regularization of the predicted (or optimized shared) LO coefficients."""
        if self.balanced_loss is None:
            loss = self._legacy_PSF_loss(PSF_data, PSF_pred)
        else:
            loss = self.balanced_loss(PSF_data, PSF_pred)
            if self.legacy_loss_weight > 0:
                loss = loss + self.legacy_loss_weight * self._legacy_PSF_loss(PSF_data, PSF_pred)

        if 'LO_coefs' in x_dict_pred:
            c  = x_dict_pred['LO_coefs'] # the modes without the phase bump: [defocus, astig., astig., coma, ...]
            LO = c.pow(2).sum(-1) * 1e-7
            if self.LO_sign_penalty:
                # Legacy sign penalties on defocus (c[:, 0]) and the first astigmatism (c[:, 2]). Note that the phase bump makes the sign of
                # defocus identifiable, so forcing it positive biases the solution (flipping the sign of negative fitted defocus values
                # increases the PSF loss by ~20%).
                LO = LO + torch.clamp(-c[:, 0], min=0).pow(2) * 5e-5 + torch.clamp(-c[:, 2], min=0).pow(2) * 1e-7
            loss = loss + LO
        return loss


    def loss_fn(self, PSF_data, PSF_pred, x_dict_pred, sample_weights=None, return_per_sample=False):
        """Mean loss of the batch (weighted by sample_weights [B]) or, with return_per_sample, the per-sample losses [B]."""
        per = self._loss_per_sample(PSF_data, PSF_pred, x_dict_pred)
        if return_per_sample:
            return per
        if sample_weights is None:
            return per.mean()
        w = sample_weights.to(dtype=per.dtype, device=per.device).view(-1)
        return (w * per).sum() / (w.sum() + 1e-12)


    # ── Checkpoints ───────────────────────────────────────────────────────────
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        torch.save({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'calibrator_state_dict': self.calibrator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), **self._extra_state()}, path)
        logger.debug(f"Checkpoint saved -> {path}")


    def load_checkpoint(self, path, load_optimizer=True):
        """The network (and the extra state) of a checkpoint; a missing file is logged and skipped."""
        path = Path(path)
        
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.calibrator.load_state_dict(ckpt['calibrator_state_dict'])
        
        self._load_extra_state(ckpt)
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
        logger.info(f"Checkpoint loaded <- {path} (epoch {ckpt.get('epoch', '?')})")


    # ── Pretraining on the fitted values ──────────────────────────────────────
    def pretrain(self, num_epochs=100, patience=15, lr=None, save_path=None):
        """
        Pretrain the network on the fitted values (no PSF model involved): a warm start for the end-to-end training. The outputs are compared
        standardized, the squared error of every output divided by the variance of its target over the training set. Without it, the MSE is
        dominated by the outputs with the largest spread in the normalized space (Cn2 log-weights and LO coefficients, ~85% of the target
        variance, both barely predictable from telemetry), while r0, J and F, which matter most for the PSF, get only a few percent of the weight.
        """
        save_path = save_path or self.weights_dir / 'pretrain_weights.pth'
        # Outputs the fits do not carry get a zero (neutral) target silently (_params_to_normalized): a regime mismatch between the fits and the
        # config (e.g. a wing_frac_ctrl output with fits made without the wing), say so
        if missing := [k for k in self.outputs_transformer.slices if k not in self.dataset.fitted_vals[0]]:
            warnings.warn(f"pretrain: the fits of the dataset carry no {missing}; their pretraining targets are 0 (neutral). The regime of the "
                          "fits (STD_fitter.py flags) does not match the \"model\" section of the config: refit the dataset and rebuild the cache")

        fit, tel = self._train_fit_dict()
        with torch.no_grad():
            inv_var = 1.0 / self._pretrain_target_vector(fit, tel).var(dim=0).clamp(min=1e-4)
        criterion = lambda tel_b, fit_b: ((self.calibrator(tel_b) - self._pretrain_target_vector(fit_b, tel_b)).pow(2) * inv_var).mean()

        opt = optim.AdamW(self.calibrator.parameters(), lr=lr or self.lr, weight_decay=5e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-4)
        best_loss, best_state, pat_count = float('inf'), None, 0

        for epoch in range(num_epochs):
            self.calibrator.train()
            train_loss = 0.0
            for _, tel_b, fit_b, _, _ in self.train_loader:
                opt.zero_grad()
                loss = criterion(tel_b, fit_b)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            self.calibrator.eval()
            with torch.no_grad():
                val_loss = sum(criterion(tel_b, fit_b).item() for _, tel_b, fit_b, _, _ in self.val_loader)

            avg_t, avg_v = train_loss / len(self.train_loader), val_loss / len(self.val_loader)
            sch.step(avg_v)
            if is_best := avg_v < best_loss:
                best_loss, best_state, pat_count = avg_v, deepcopy(self.calibrator.state_dict()), 0
            else:
                pat_count += 1

            print(f"\rPretrain {epoch+1:3d}/{num_epochs} | train={avg_t:.4f} val={avg_v:.4f} lr={opt.param_groups[0]['lr']:.2e}" + (" *" if is_best else ""),
                  end='', flush=True)
            if pat_count >= patience:
                break

        print()
        if best_state:
            self.calibrator.load_state_dict(best_state)
        torch.save(best_state, save_path)
        logger.info(f"Pretrain complete. Best val={best_loss:.6f}, saved -> {save_path}")


    # ── Validation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader=None, return_cubes=False):
        """
        Mean loss of the calibrator over a loader (the validation one by default) or, with return_cubes, (PSFs_pred, PSFs_data, global_ids,
        NN_predictions, telemetry, mean loss), the arrays in the order of the loader's dataset positions (global_ids).
        """
        loader = loader or self.val_loader
        self.calibrator.eval()
        global_ids = torch.tensor(loader.dataset.indices if isinstance(loader.dataset, Subset) else range(len(loader.dataset)), dtype=torch.long)
        local      = {int(g): l for l, g in enumerate(global_ids.tolist())}

        N = len(global_ids)
        if return_cubes: # full cubes for the diagnostics, kept on the CPU
            PSFs_pred, PSFs_data = torch.zeros((N, self.N_wvl_total, self.H, self.W)), torch.zeros((N, self.N_wvl_total, self.H, self.W))
            NN_pred,   tel_vecs  = torch.zeros((N, self.N_net_outputs)), torch.zeros((N, self.N_features))

        total_loss, total_count = 0.0, 0
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            pos = torch.tensor([local[int(i)] for i in idxs_b.tolist()], dtype=torch.long)
            x_pred, x_dict = self._predict(tel_b)
            if return_cubes:
                NN_pred[pos], tel_vecs[pos] = x_pred.cpu(), tel_b.cpu()

            for lambda_ids in self.lambda_id_sets: # every wavelength, in subsets to limit the memory
                PSF_pred_b   = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                total_loss  += self.loss_fn(PSF_data_b[:, lambda_ids], PSF_pred_b, x_dict).item() * len(idxs_b) # every sample counts equally
                total_count += len(idxs_b)
                if return_cubes:
                    PSFs_pred[pos[:, None], lambda_ids] = PSF_pred_b.cpu()
                    PSFs_data[pos[:, None], lambda_ids] = PSF_data_b[:, lambda_ids].cpu()

        avg_loss = total_loss / max(total_count, 1)
        return (PSFs_pred, PSFs_data, global_ids, NN_pred, tel_vecs, avg_loss) if return_cubes else avg_loss


    # ── End-to-end training ───────────────────────────────────────────────────
    def train(self, num_epochs=500, patience=20, nan_recovery=True, max_nan_recoveries=10, sample_weights=None, checkpoint_path=None,
              train_loader=None, val_loader=None):
        """
        End-to-end training through the PSF model with early stopping on the validation loss, optional dataset-length sample weights (see
        run_kfold_difficulty_weighting) and NaN recovery (the best checkpoint reloaded at a lower learning rate). Saves the best checkpoint
        and one every 5 epochs, and ends with the best one loaded. Returns (train_losses, val_losses) of the completed epochs.
        """
        train_loader    = train_loader or self.train_loader
        val_loader      = val_loader   or self.val_loader
        checkpoint_path = Path(checkpoint_path or self.weights_dir / 'best_calibrator_checkpoint.pth')
        sample_weights  = None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device)
        trainable       = [*self.calibrator.parameters(), *self.optimized_params.values()]

        best_val, patience_counter, nan_count, bad_grad_count = float('inf'), 0, 0, 0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.calibrator.train()
            epoch_loss, n_batches, nan_epoch = 0.0, 0, False

            for PSF_data_b, tel_b, _, config_b, idxs_b in train_loader:
                self.optimizer.zero_grad()
                batch_loss = 0.0

                for lambda_ids in self.lambda_id_sets: # backpropagated per wavelength subset to limit the graph memory
                    x_pred, x_dict = self._predict(tel_b)
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    loss = self.loss_fn(PSF_data_b[:, lambda_ids], PSF_pred_b, x_dict, None if sample_weights is None else sample_weights[idxs_b])

                    if not torch.isfinite(loss): # reload the best checkpoint at a lower learning rate
                        if not nan_recovery or nan_count >= max_nan_recoveries:
                            raise ValueError(f"NaN loss after {nan_count} recovery attempt(s)")
                        nan_count, nan_epoch = nan_count + 1, True
                        logger.error(f"NaN at epoch {epoch}, recovery {nan_count}/{max_nan_recoveries}")
                        self.load_checkpoint(checkpoint_path)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] *= 0.7
                        self.optimizer.zero_grad(set_to_none=True)
                        break

                    loss.backward()
                    batch_loss += loss.item()
                    del PSF_pred_b, x_pred, x_dict, loss

                if nan_epoch:
                    break

                # A NaN in the backward pass of the PSF model (one bad parameter set in the batch) leaves the loss finite but the gradients
                # non-finite: the optimizer step would corrupt the weights and the next forward pass would crash (torch.linalg.lstsq of the
                # tomographic reconstructor). Skip such batches instead.
                if not all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None):
                    bad_grad_count += 1
                    logger.warning(f"Non-finite gradients at epoch {epoch} (sample ids {idxs_b.tolist()}): batch skipped, {bad_grad_count} so far")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                self.optimizer.step()
                epoch_loss += batch_loss
                n_batches  += 1

            release_gpu_memory()
            if nan_epoch:
                continue # I have become comfortably NaN

            avg_train = epoch_loss / max(n_batches, 1)
            val_loss  = self.validate(loader=val_loader)
            train_losses.append(avg_train)
            val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if is_best := val_loss < best_val:
                best_val, patience_counter = val_loss, 0
                self.save_checkpoint(checkpoint_path, epoch, avg_train, val_loss)
            else:
                patience_counter += 1

            msg = (f"Epoch {epoch:4d} | train={avg_train:.4f} val={val_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e} "
                   f"pat={patience_counter}/{patience}" + (" *" if is_best else ""))
            print(msg)
            logger.info(msg)

            if epoch % 5 == 0:
                self.save_checkpoint(self.weights_dir / f'checkpoint_epoch_{epoch}.pth', epoch, avg_train, val_loss)
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.load_checkpoint(checkpoint_path)
        logger.info(f"Training complete. Best val={best_val:.6f}")
        return train_losses, val_losses


    # ── K-fold difficulty weighting ───────────────────────────────────────────
    def run_kfold_difficulty_weighting(self, base_indices, initial_state_dict, n_splits=5, fold_epochs=120, fold_patience=12, alpha=0.30,
                                       clip=(0.5, 2.0), random_state=42, force_retrain=False, load_only=False):
        """
        Dataset-length sample weights for the final training run, larger for the samples the calibrator predicts poorly: out-of-fold PSF
        losses of k-fold runs over base_indices (grouped by acquisition time), a random forest predicting their log from the telemetry, and
        weights exp(alpha (prediction - median)) normalized to a mean of 1 and clipped to `clip`; the other samples get 1. The fold
        checkpoints of weights_dir/kfold_difficulty are reused unless force_retrain (with load_only they must all exist). The network and the
        optimizer are reset to initial_state_dict afterwards.
        """
        base_indices = np.asarray(base_indices, dtype=int)
        folds        = kfold_splits(self.dataset, base_indices, n_splits, random_state)
        kfold_dir    = self.weights_dir / 'kfold_difficulty'
        kfold_dir.mkdir(exist_ok=True)
        checkpoints  = [kfold_dir / f'fold_{i:02d}_best.pth' for i in range(len(folds))]
        if load_only and (missing := [i for i, ckpt in enumerate(checkpoints) if not ckpt.exists()]):
            raise FileNotFoundError(f"load_only=True but checkpoints missing for folds: {missing}. Set force_retrain=True to train them, "
                                    "or use load_only=False for default behavior.")

        oof_losses = np.full(len(self.dataset), np.nan, dtype=np.float32)
        for fold_id, ((tr_rel, va_rel), ckpt) in enumerate(zip(folds, checkpoints)):
            fold_train_idx, fold_val_idx = base_indices[tr_rel], base_indices[va_rel]
            self.calibrator.load_state_dict(deepcopy(initial_state_dict))
            if ckpt.exists() and not force_retrain:
                logger.info(f"Fold {fold_id+1}/{len(folds)}: Loading pre-trained checkpoint from {ckpt}")
                self.load_checkpoint(ckpt, load_optimizer=False)
            else:
                logger.info(f"Fold {fold_id+1}/{len(folds)}: Training (train={len(fold_train_idx)}, val={len(fold_val_idx)})")
                self.optimizer, self.scheduler = self._make_optimizer()
                self.train(num_epochs=fold_epochs, patience=fold_patience, checkpoint_path=ckpt,
                           train_loader=self._make_loader(fold_train_idx, shuffle=True), val_loader=self._make_loader(fold_val_idx, shuffle=False))

            val_ids, fold_losses = self._evaluate_per_sample_losses(self._make_loader(fold_val_idx, shuffle=False))
            oof_losses[val_ids] = fold_losses
            release_gpu_memory(sync=True)

        X  = np.array([[*self.dataset.telemetry[int(i)].values()] for i in base_indices], dtype=np.float32)
        y  = np.log10(oof_losses[base_indices] + 1e-12)
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        cv_r2 = cross_val_score(rf, X, y, cv=folds, scoring='r2')
        logger.info(f"RF difficulty CV R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}") # high: the telemetry tells where the calibrator struggles

        difficulty = rf.fit(X, y).predict(X)
        w = np.exp(alpha * (difficulty - np.median(difficulty)))
        w = np.clip(w / np.mean(w), *clip)
        weights = np.ones(len(self.dataset), dtype=np.float32)
        weights[base_indices] = w / np.mean(w)

        # Reset to the same starting point for the final training run
        self.calibrator.load_state_dict(deepcopy(initial_state_dict))
        self.optimizer, self.scheduler = self._make_optimizer()

        logger.info(f"Difficulty weights: min={weights[base_indices].min():.3f} mean={weights[base_indices].mean():.3f} max={weights[base_indices].max():.3f}")
        return torch.as_tensor(weights, dtype=torch.float32, device=self.device)


    @torch.no_grad()
    def _evaluate_per_sample_losses(self, loader):
        """(dataset positions, the PSF loss of every sample averaged over the wavelength sets) of a loader."""
        self.calibrator.eval()
        global_ids = np.asarray(loader.dataset.indices if isinstance(loader.dataset, Subset) else range(len(loader.dataset)), dtype=int)
        losses     = {int(i): [] for i in global_ids}
        for PSF_data_b, tel_b, _, config_b, idxs_b in loader:
            x_pred, x_dict = self._predict(tel_b)
            for lambda_ids in self.lambda_id_sets:
                PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                per = self.loss_fn(PSF_data_b[:, lambda_ids], PSF_pred_b, x_dict, return_per_sample=True)
                for i, loss in zip(idxs_b.tolist(), per.tolist()):
                    losses[i].append(loss)
        return global_ids, np.array([np.mean(losses[int(i)]) for i in global_ids], dtype=np.float32)


    # ── Fine-tuning of selected outputs ───────────────────────────────────────
    def tune_calibrator(self, tuned_params, num_epochs=50, lr=1e-3):
        """
        Fine-tune the network outputs of `tuned_params` (keys of net_slices; the others, e.g. the optimized shared keys of the factor calibrator,
        are skipped with a warning) on the PSF loss, the other outputs anchored to a frozen copy of the network. Ends with the best state loaded.
        """
        slices = self.net_slices()
        if skipped := [k for k in tuned_params if k not in slices]:
            warnings.warn(f"tune_calibrator: {skipped} are not predicted by the network, skipped")
        if not (tuned_params := [k for k in tuned_params if k in slices]):
            raise ValueError(f"tune_calibrator: none of the tuned parameters is predicted by the network ({list(slices)})")

        ref   = deepcopy(self.calibrator).eval().requires_grad_(False)
        tuned = torch.zeros(self.N_net_outputs, dtype=torch.bool, device=self.device)
        for k in tuned_params:
            tuned[slices[k]] = True

        opt = optim.AdamW(self.calibrator.parameters(), lr=lr, weight_decay=5e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        best_loss, best_state = float('inf'), None

        for epoch in range(num_epochs):
            self.calibrator.train()
            for PSF_data_b, tel_b, _, config_b, idxs_b in self.train_loader:
                opt.zero_grad()
                with torch.no_grad():
                    x_ref = ref(tel_b)
                if not tuned.all(): # soft fix: the untuned outputs are kept close to the frozen reference
                    (self.calibrator(tel_b)[:, ~tuned] - x_ref[:, ~tuned]).pow(2).mean().mul(100.0).backward()

                for lambda_ids in self.lambda_id_sets: # a fresh forward pass per wavelength set to limit the memory
                    x_dict     = self._params_from_pred(torch.where(tuned, self.calibrator(tel_b), x_ref), tel_b) # the reference outside the tuned outputs
                    PSF_pred_b = self.run_model(x_dict, config_b, idxs_b, lambda_ids)
                    self.loss_fn(PSF_data_b[:, lambda_ids], PSF_pred_b, x_dict).backward()
                    del PSF_pred_b, x_dict
                opt.step()

            val_loss = self.validate()
            sch.step(val_loss)
            if is_best := val_loss < best_loss:
                best_loss, best_state = val_loss, deepcopy(self.calibrator.state_dict())
            print(f"Tune {epoch+1}/{num_epochs} | val={val_loss:.4f}" + (" *" if is_best else ""))

        if best_state:
            self.calibrator.load_state_dict(best_state)
        logger.info(f"Tuning complete. Best val={best_loss:.6f}")


    # ── Calibrator bundle ─────────────────────────────────────────────────────
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
            'calibrator_type':        self.calibrator_type,
            'net_state_dict':         self.calibrator.state_dict(),
            'net_arch':               {'n_features': self.N_features, 'n_outputs': self.N_net_outputs,
                                       'hidden_dim': self.calibrator.network[0].out_features, 'dropout_rate': self.calibrator.network[3].p},
            'input_feature_names':    self.dataset.features,
            'output_feature_names':   list(self.outputs_transformer.slices),
            'outputs_transformer':    self.outputs_transformer.save(),
            'telemetry_preprocessor': self.dataset.telemetry_preprocessor, # TelemetryPreprocessor.state_dict()
            'Cn2_grid':               self.dataset.Cn2_grid, # Cn2Grid.state_dict() of the Cn2 features, None with the K-means features
            'Cn2_binning':            'grid' if self.dataset.Cn2_grid is not None else 'kmeans',
            'LO_modes_max':           self.PSF_model.Z_mode_max,
            'N_spline_nodes':         getattr(self.PSF_model, 'N_wvl_ctrl', None),
            'model_flags':            model_flags(self.PSF_model), # jitter / wing regime of the PSF model the calibrator was trained with
            'fitted_shared':          fitted_shared_values(self.PSF_model, self.fitted_shared_keys), # set in the PSF models NFMCalibrator calibrates
            'training_config':        self.training_config, # validate_NFM_calibrator.py rebuilds the trainer from it
            **self._extra_state(),
        }
        torch.save(_to_cpu(state), path)
        logger.info(f"Calibrator bundle saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Factor calibrator
#
#  The network predicts corrections to prior values instead of absolute parameters, so that a zero output reproduces the prior:
#    factor keys (r0, L0, wind_speed_single, Cn2_weights): value = base * exp(delta + mean log-ratio), the base derived from
#        the telemetry (telemetry_base_params), L0 from the median of the fits; the Cn2 weights are renormalized to 1.
#        The Cn2 base is the AOF profile reprojected onto the fixed layers of the grid (Cn2_grid_* features) or the K-means fractions
#    predicted keys (dn, J_ctrl):                          value = median of the fits + delta, in the normalized space
#    optimized shared keys (F_ctrl, LO_coefs,              one value for all samples, starting from the median of the training fits and
#      wing_frac_ctrl, J_aniso)                            trained end-to-end together with the network, not predicted
#  The network outputs only the predicted and factor keys; the parameters the calibrator returns (with the optimized shared keys), the
#  validation plots and the bundles keep the stacked layout of the outputs transformer. The fitted and fitted shared keys and the inputs
#  left to the model are handled as for the direct calibrator (see KEY_GROUPS).
# ═══════════════════════════════════════════════════════════════════════════════

class FactorParameterization:
    """
    Maps the corrections predicted by the factor network to PSF-model parameters, and the fits to corrections (pretraining targets). The
    statistics (median of the normalized fits, mean log-ratio fit / base of the factor entries) are computed on the training set by fit() and
    stored in the bundle. They live in the stacked layout of the outputs transformer [N_outputs]; the network outputs only the predicted and
    factor entries [N_net] (pred_index: their positions in the stacked layout, net_slices: the keys' slices of the network output). The
    optimized shared keys (one dataset-level value each, trained by NFMFactorCalibratorTrainer) are not predicted: params() takes them from `consts`.
    """
    def __init__(self, slices: dict, features: list, factor_keys, tel_mean, tel_scale, optimized_shared_keys=(), L0_const=None,
                 Y_median=None, log_ratio_mean=None, Cn2_grid=None):
        self.slices                = {k: (v if isinstance(v, slice) else slice(int(v[0]), int(v[1]))) for k, v in slices.items()}
        self.features              = list(features)
        self.factor_keys           = [k for k in factor_keys           if k in self.slices]
        self.optimized_shared_keys = [k for k in optimized_shared_keys if k in self.slices]
        self.direct_keys           = [k for k in self.slices if k not in self.factor_keys + self.optimized_shared_keys] # predicted_keys
        self.L0_const              = None if L0_const is None else float(L0_const)
        self.Cn2_grid              = Cn2_grid if Cn2_grid is None or isinstance(Cn2_grid, Cn2Grid) else Cn2Grid.from_state_dict(Cn2_grid)
        self.N_outputs             = max(sl.stop for sl in self.slices.values())

        net_keys        = [k for k in self.slices if k not in self.optimized_shared_keys] # the predicted and factor keys
        self.pred_index = torch.tensor([i for k in net_keys for i in range(self.slices[k].start, self.slices[k].stop)], dtype=torch.long)
        self.net_slices, n = {}, 0
        for k in net_keys:
            width = self.slices[k].stop - self.slices[k].start
            self.net_slices[k] = slice(n, n + width)
            n += width
        self.N_net = n

        self.tel_mean       = _as_float_tensor(tel_mean,  'cpu')
        self.tel_scale      = _as_float_tensor(tel_scale, 'cpu')
        self.Y_median       = torch.zeros(self.N_outputs) if Y_median       is None else _as_float_tensor(Y_median,       'cpu')
        self.log_ratio_mean = torch.zeros(self.N_outputs) if log_ratio_mean is None else _as_float_tensor(log_ratio_mean, 'cpu')


    def to(self, device):
        for name in ('tel_mean', 'tel_scale', 'Y_median', 'log_ratio_mean', 'pred_index'):
            setattr(self, name, getattr(self, name).to(device))
        return self


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


    def targets(self, y_fit: torch.Tensor, fit_phys: dict, base: dict) -> torch.Tensor:
        """Network outputs reproducing the fits [B, N_net]: y_fit - median (predicted keys), log(fit / base) - mean log-ratio (factor keys)."""
        delta = y_fit - self.Y_median
        r     = self.log_ratios(fit_phys, base)
        for k in self.factor_keys:
            sl = self.slices[k]
            delta[:, sl] = r[:, sl] - self.log_ratio_mean[sl]
        return delta[:, self.pred_index]


    def params(self, delta: torch.Tensor, tel_std: torch.Tensor, transformer, consts: dict) -> dict:
        """Network outputs [B, N_net] + standardized telemetry -> PSF-model parameters of the batch (phase bump excluded); consts:
        the optimized shared keys in physical units, {key: [1, n]}."""
        B     = delta.shape[0]
        delta = delta.new_zeros(B, self.N_outputs).index_copy(1, self.pred_index, delta) # stacked layout, zero at the optimized shared entries
        x     = transformer.unstack(self.Y_median + delta) # the predicted keys (the others are overwritten below)
        base  = self.base_values(tel_std)
        flat  = lambda k, val: val.squeeze(-1) if self.slices[k].stop - self.slices[k].start < 2 else val # single values as [B]

        for k in self.factor_keys:
            sl  = self.slices[k]
            val = base[k].reshape(B, -1) * torch.exp(delta[:, sl] + self.log_ratio_mean[sl])
            x[k] = flat(k, val / val.sum(dim=-1, keepdim=True) if k == 'Cn2_weights' else val)
        for k in self.optimized_shared_keys:
            x[k] = flat(k, consts[k].expand(B, -1))
        return x


    def fit(self, y_fit: torch.Tensor, fit_phys: dict, base: dict):
        """Training-set statistics: median of the normalized fits, mean log-ratio of the factor entries."""
        self.Y_median       = y_fit.median(dim=0).values
        self.log_ratio_mean = self.log_ratios(fit_phys, base).mean(dim=0)
        return self


    def state_dict(self) -> dict:
        return {
            'slices':                {k: [sl.start, sl.stop] for k, sl in self.slices.items()},
            'features':              self.features,
            'factor_keys':           self.factor_keys,
            'optimized_shared_keys': self.optimized_shared_keys,
            'L0_const':              self.L0_const,
            'Cn2_grid':              None if self.Cn2_grid is None else self.Cn2_grid.state_dict(),
            **{name: getattr(self, name).cpu().numpy() for name in ('tel_mean', 'tel_scale', 'Y_median', 'log_ratio_mean')},
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        # Bundles before 2026-10-30 also carry delta_std (the widths of the shrinkage prior) and Cn2_targets: training-only, ignored
        return cls(**{k: v for k, v in state.items() if k not in ('delta_std', 'Cn2_targets')})


class NFMFactorCalibrator(NFMCalibrator):
    """NFMCalibrator for a factor-calibrator bundle: the network outputs are corrections, see FactorParameterization."""
    bundle_type = 'factor'

    def __init__(self, checkpoint_path, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(checkpoint_path, device)
        self.parameterization = FactorParameterization.from_state_dict(self._bundle_state['factor_parameterization']).to(device)
        self.consts = {k: _as_float_tensor(v, device) for k, v in self._bundle_state['consts'].items()}

    def _params_from_pred(self, x_pred: torch.Tensor, telemetry_vector: torch.Tensor) -> dict:
        return self.parameterization.params(x_pred, telemetry_vector, self.outputs_transformer, self.consts)


def load_NFM_calibrator(bundle_path, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """NFMCalibrator or NFMFactorCalibrator, depending on the type stored in the bundle."""
    state = torch.load(bundle_path, map_location='cpu', weights_only=False)
    return (NFMFactorCalibrator if state['calibrator_type'] == 'factor' else NFMCalibrator)(bundle_path, device=device)


class NFMFactorCalibratorTrainer(NFMCalibratorTrainer):
    """
    NFMCalibratorTrainer for the factor calibrator: the network predicts the corrections (FactorParameterization) of the predicted and
    factor keys only, and the optimized shared keys are dataset-level parameters trained together with the network (at a tenth of the
    learning rate). They start from the medians of the training fits and are trained in the normalized space of the outputs transformer
    (optimized_params), where one learning rate suits them all (J_angle in degrees as well as wing_frac ~ 0.005); checkpoints and bundles
    store them in physical units ('consts'). factor_keys, optimized_shared_keys: the key groups of the "calibrator" section of the
    training config (calibrator_key_groups, build_NFM_trainer).
    """
    calibrator_type = 'factor'

    def __init__(self, PSF_model, calibrator, dataset, outputs_transformer, *, factor_keys=(), optimized_shared_keys=(), **kwargs):
        preprocessor = TelemetryPreprocessor.from_state_dict(dataset.telemetry_preprocessor) # the base values are derived from the telemetry
        self.parameterization = P = FactorParameterization(outputs_transformer.slices, dataset.features, factor_keys, preprocessor.mean,
                                                           preprocessor.scale, optimized_shared_keys=optimized_shared_keys, Cn2_grid=dataset.Cn2_grid)
        if calibrator.network[-1].out_features != P.N_net:
            raise ValueError(f'The factor network has {calibrator.network[-1].out_features} outputs, the predicted and factor keys '
                             f'{list(P.net_slices)} need {P.N_net}')
        super().__init__(PSF_model, calibrator, dataset, outputs_transformer, **kwargs)
        P.to(self.device)

        # Training-set statistics of the parameterization and the starting point of the optimized shared keys (medians of the fits)
        fit, tel = self._train_fit_dict()
        fit_phys = self._calibrator_friendly_fit_dict(fit)
        P.L0_const = float(fit['L0'].median()) if 'L0' in P.factor_keys else None
        with torch.no_grad():
            P.fit(super()._pretrain_target_vector(fit), fit_phys, P.base_values(tel))

        for k in P.optimized_shared_keys:
            if k not in fit_phys:
                raise KeyError(f"The fits of the dataset carry no '{k}', its shared value cannot start from their median: refit the dataset in "
                               "the regime of the config and rebuild the cache")
            
            median = fit_phys[k].reshape(len(tel), -1).median(dim=0).values.unsqueeze(0)
            self.optimized_params[k] = nn.Parameter(self.outputs_transformer.transforms[k](median))
            
        self.optimizer, self.scheduler = self._make_optimizer() # now including the optimized shared keys

        logger.info(f"Factor calibrator | network outputs: predicted {P.direct_keys}, factor {P.factor_keys} ({P.N_net} outputs) | "
                    f"optimized shared: {P.optimized_shared_keys} | fitted: {self.fitted_keys} | fitted shared: {self.fitted_shared_keys} | "
                    f"L0 = {P.L0_const} | Cn2: {P.Cn2_grid or 'K-means features'}")


    def optimized_shared_values(self) -> dict:
        """The optimized shared keys in physical units, {key: [1, n]} (differentiable w.r.t. optimized_params)."""
        return {k: self.outputs_transformer.transforms[k].inverse(p) for k, p in self.optimized_params.items()}


    def net_slices(self) -> dict:
        return self.parameterization.net_slices


    def _params_from_pred(self, x_pred, tel_b):
        return self.parameterization.params(x_pred, tel_b, self.outputs_transformer, self.optimized_shared_values())


    def _pretrain_target_vector(self, fitted_vals, tel=None):
        if tel is None:
            raise ValueError('The factor calibrator needs the telemetry vectors to compute its targets (base values)')
        P = self.parameterization
        return P.targets(super()._pretrain_target_vector(fitted_vals), self._calibrator_friendly_fit_dict(fitted_vals), P.base_values(tel))


    def _extra_state(self) -> dict:
        with torch.no_grad():
            consts = {k: v.cpu().clone() for k, v in self.optimized_shared_values().items()}
        return {'consts': consts, 'factor_parameterization': self.parameterization.state_dict()} # consts in physical units, as NFMFactorCalibrator uses them


    def _load_extra_state(self, state: dict) -> None:
        for k, v in state.get('consts', {}).items(): # the optimized shared keys in physical units
            if k in self.optimized_params:
                self.optimized_params[k].data = self.outputs_transformer.transforms[k](_as_float_tensor(v, self.device).reshape(1, -1))

        if 'factor_parameterization' in state:
            self.parameterization = FactorParameterization.from_state_dict(state['factor_parameterization']).to(self.device)
