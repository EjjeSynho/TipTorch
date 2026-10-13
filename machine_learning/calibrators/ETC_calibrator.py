"""
ETC-regime PSF prediction for MUSE NFM.

In the exposure-time-calculator (ETC) regime only a handful of inputs is known per observation: seeing, coherence time tau0,
airmass, NGS magnitude (hence the IRLOS regime) and the turbulence category (TC) with its median Cn2 profile. The PSF-R calibrator
(`NFMCalibrator`) instead consumes 31 telemetry features, 20 of which are either closed-loop AO diagnostics or observation
nuisances that an ETC cannot know. This module provides:

- the ETC feature groups of the calibrator inputs and the ETC conditions table (`etc_table*`);
- `ETCTelemetryBuilder`: 31-D standardized telemetry for ETC conditions, with the unknown features filled either by the trained
  imputer (`fill='impute'`, deterministic), copied from a nearest-neighbour real observation (`fill='donor'`, stochastic
  hot-deck, preserves the joint distribution of the unknowns) or drawn from a conditional Gaussian fitted on the training
  telemetry (`fill='gaussian'`);
- PSF-parameter predictors sharing one interface (`predict(etc, rng) -> (x_dict, info)`):
    * `CalibratorPredictor`  the PSF-R calibrator on built telemetry (strategy B of the ExecPlan: a model in front of the calibrator),
    * `DonorParamsPredictor` the fitted PSF-model parameters of a nearest-neighbour training observation, r0 and wind rescaled to
                             the requested seeing / tau0 (strategy A, non-parametric),
    * `DirectNetPredictor`   a small network from the ETC inputs to the PSF-model parameters, trained end-to-end with the PSF model
                             (strategy A, parametric);
- `ETCPSFRenderer`: batched rendering of the predicted parameters with `PSFModelNFM` at given wavelengths;
- `ETCCalibrator`: the bundle (pool of training observations, direct network, references) saved by `train_ETC_calibrator.py`.

Tensor conventions: ETC tables are pandas DataFrames with one row per observation; built telemetry is [N, 31] float32 in the
standardized units of the calibrator's `TelemetryPreprocessor`; `x_dict` holds PSF-model parameters as tensors [N, ...] on the
device; rendered PSFs are [N, N_wvl, H, W] on the CPU.
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiptorch._config import DATA_FOLDER, default_device
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from tiptorch.managers.input_manager import InputsTransformer
from data_processing.telemetry_preprocessor import TelemetryPreprocessor
from calibrators.NFM_calibrator import NFMCalibrator, SmallCalibratorNet, release_gpu_memory

PUPIL_PATH = DATA_FOLDER / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'

# ------------------------------------------------- Feature groups -------------------------------------------------
# The ETC knows these directly (the IRLOS regime follows from the NGS magnitude, see `IRLOS_frequency` / `IRLOS_gain`)
ETC_DIRECT_FEATURES = ['Seeing (header)', 'Tau0 (header)', 'Tel. altitude', 'NGS mag (from ph.)', 'IRLOS frequency', 'IRLOS gain']
# Site / turbulence-profile features the ETC replaces by TC constants (median profile, ASM wind) or derives from seeing and tau0
ETC_PROFILE_FEATURES = [
    'Wind dir (header)', 'Wind speed (header)', 'Free Atmosphere Seeing ["]', 'MASS Tau0 [s]', 'MASS-DIMM Turb Velocity [m/s]',
    'MASS_FRACGL', 'IA_FWHMLINOBS', 'LGS_TUR_ALT', 'LGS photons, [photons/m^2/s]', 'MASS_TURB total', 'Cn2 fraction below 2000m',
    'Cn2_alt_binned_2', 'Cn2_frac_binned_2', 'Cn2_alt_binned_3', 'Cn2_frac_binned_3',
]
# Closed-loop AO performance diagnostics: measured during the observation, unknown to the ETC
AO_DIAGNOSTIC_FEATURES = ['LGS_STREHL', 'LGS_TURVAR_RES', 'LGS_FWHM_GAIN', 'LGS_SLOPERMS', 'theta0']
# Observation-specific nuisances (pointing, exposure, temperature)
NUISANCE_FEATURES = ['Exp. time', 'Tel. azimuth', 'Par. angle', 'Derot. angle', 'Temperature (header)']

FEATURE_GROUPS = {
    'ETC inputs':      ETC_DIRECT_FEATURES,
    'Site / profile':  ETC_PROFILE_FEATURES,
    'AO diagnostics':  AO_DIAGNOSTIC_FEATURES,
    'Nuisance':        NUISANCE_FEATURES,
}

# ------------------------------------------- ETC conditions (Enrico's TC constants) -------------------------------------------
TC_LIST = [10, 20, 30, 50, 100]
TC_WIND_SPEED = {10: 10.4,  20: 10.5,  30: 9.9,   50: 10.1,  100: 14.5}   # ASM wind speed [m/s]
TC_WIND_DIR   = {10: 12.5,  20: 28.8,  30: 40.2,  50: 37.2,  100: 37.0}   # ASM wind direction [deg]
TC_L0         = {10: 12.48, 20: 17.1,  30: 22.1,  50: 23.5,  100: 25.5}   # outer scale [m]
TC_GLF        = {10: 0.971, 20: 0.973, 30: 0.980, 50: 0.972, 100: 0.977}  # ground-layer fraction of the median profile
ETC_HO_PHOTONS             = 1740.2396627862904  # [photons/subaperture/frame], LGS WFS flux used by the ETC
ETC_LGS_PHOTON_RATE_SCALE  = 1240e3              # subapertures x HO frames per second -> 'LGS photons, [photons/m^2/s]'
ETC_REFERENCE_WAVELENGTH   = 500e-9              # seeing / r0 reference wavelength [m]
ETC_WHITE_PROFILE_HEIGHTS  = [0.0, 2000.0, 20000.0]  # [m], TIPTOP's two-layer ETC profile plus a zero-weight placeholder


def turbulence_categories(seeing, tau0_ms):
    """ ESO turbulence category (percentile) from seeing ["] and coherence time [ms] """
    seeing, tau = np.atleast_1d(np.asarray(seeing, float)), np.atleast_1d(np.asarray(tau0_ms, float))
    tc = np.full(len(seeing), 100, dtype=int)
    mask10  = (seeing <= 0.6) & (tau >= 5.2)
    mask20  = (seeing <= 0.7) & (tau >= 4.4) & ~mask10
    mask30  = (seeing <= 0.8) & (tau >= 4.1) & ~(mask10 | mask20)
    mask50  = (seeing <= 1.0) & (tau >= 3.2) & ~(mask10 | mask20 | mask30)
    tc[mask50], tc[mask30], tc[mask20], tc[mask10] = 50, 30, 20, 10
    return tc


def IRLOS_frequency(ngs_mag):
    """ IRLOS loop frequency [Hz] as a function of the NGS J magnitude (Enrico's regime switch) """
    return np.where(np.asarray(ngs_mag, float) <= 16, 500.0, 200.0)


def IRLOS_gain(ngs_mag):
    """ IRLOS detector gain as a function of the NGS J magnitude (Enrico's regime switch) """
    m = np.asarray(ngs_mag, float)
    return np.where(m <= 10.5, 1.0, np.where(m <= 16, 68.0, 100.0))


def NGS_mag_to_photons(Jmag, frequency, airmass, k_J=0.09, delta_mag=0.4):
    """ IRLOS photons per subaperture per frame from the NGS J magnitude (Enrico / Arseniy's conversion) """
    IRLOS_mag = np.asarray(Jmag, float) + delta_mag + k_J * np.asarray(airmass, float)
    N_ph = 1.9e12 / 368 * 10 ** (-0.4 * IRLOS_mag)
    return N_ph / (4 * np.asarray(frequency, float))


def etc_table(seeing, tau0, airmass, ngs_mag, irlos_freq=None, irlos_gain=None, tc=None):
    """
    ETC conditions table: `seeing` ["], `tau0` [s], `airmass`, `ngs_mag` [J mag]; the TC and the IRLOS regime are derived unless
    given (`irlos_freq` [Hz], `irlos_gain`). Adds `zenith_deg` and `ngs_flux` [photons/subaperture/frame].
    """
    seeing, tau0, airmass, ngs_mag = (np.atleast_1d(np.asarray(v, float)) for v in (seeing, tau0, airmass, ngs_mag))
    airmass = np.clip(airmass, 1.0, None)
    df = pd.DataFrame({'seeing': seeing, 'tau0': tau0, 'airmass': airmass, 'ngs_mag': ngs_mag})
    df['tc']         = turbulence_categories(seeing, tau0 * 1e3) if tc is None else np.asarray(tc, int)
    df['irlos_freq'] = IRLOS_frequency(ngs_mag) if irlos_freq is None else np.asarray(irlos_freq, float)
    df['irlos_gain'] = IRLOS_gain(ngs_mag) if irlos_gain is None else np.asarray(irlos_gain, float)
    df['zenith_deg'] = np.degrees(np.arccos(1.0 / airmass))
    df['ngs_flux']   = NGS_mag_to_photons(ngs_mag, df['irlos_freq'].to_numpy(), airmass)
    return df


def etc_table_from_stars(stars):
    """ ETC table from the Monte Carlo observation table of `run_tiptop_random_carlo.py` (`tau` in ms, `ngs` in J mag) """
    df = etc_table(stars['seeing'], np.asarray(stars['tau'], float) * 1e-3, stars['airmass'], stars['ngs'], tc=stars['tc'])
    if 'index' in stars.colnames:
        df.insert(0, 'index', np.asarray(stars['index']))
    return df


def etc_table_from_telemetry(telemetry_physical: pd.DataFrame, real_irlos=True):
    """
    ETC table replaying real observations: the ETC inputs are read from the physical-unit telemetry (header seeing and tau0,
    altitude, NGS magnitude). With `real_irlos` the recorded IRLOS frequency / gain are kept instead of the magnitude switch.
    """
    t = telemetry_physical
    airmass = 1.0 / np.cos(np.deg2rad(90.0 - t['Tel. altitude'].to_numpy(float)))
    kw = {'irlos_freq': t['IRLOS frequency'].to_numpy(float), 'irlos_gain': t['IRLOS gain'].to_numpy(float)} if real_irlos else {}
    return etc_table(t['Seeing (header)'].to_numpy(float), t['Tau0 (header)'].to_numpy(float), airmass, t['NGS mag (from ph.)'].to_numpy(float), **kw)


def tc_profile_telemetry(etc: pd.DataFrame, heights=ETC_WHITE_PROFILE_HEIGHTS):
    """
    Physical-unit values of the site / profile telemetry features implied by the ETC conditions and the TC median profile
    (same recipe as `build_ETC_telemetry` in `validate_NFM_calibrator.py`). Returns a DataFrame [N, len(ETC_PROFILE_FEATURES)].
    """
    tc      = etc['tc'].to_numpy(int)
    seeing  = etc['seeing'].to_numpy(float)
    tau0    = etc['tau0'].to_numpy(float)
    glf     = np.array([TC_GLF[t] for t in tc])
    r0      = 0.98 * ETC_REFERENCE_WAVELENGTH / np.deg2rad(seeing / 3600.0)
    k       = 2.0 * np.pi / ETC_REFERENCE_WAVELENGTH
    return pd.DataFrame({
        'Wind dir (header)':             np.array([TC_WIND_DIR[t] for t in tc]),
        'Wind speed (header)':           np.array([TC_WIND_SPEED[t] for t in tc]),
        'Free Atmosphere Seeing ["]':    seeing * (1.0 - glf) ** 0.6,
        'MASS Tau0 [s]':                 tau0,
        'MASS-DIMM Turb Velocity [m/s]': 0.314 * r0 / tau0,
        'MASS_FRACGL':                   glf,
        'IA_FWHMLINOBS':                 seeing,
        'LGS_TUR_ALT':                   1.0 - glf,
        'LGS photons, [photons/m^2/s]':  np.full_like(seeing, ETC_HO_PHOTONS * ETC_LGS_PHOTON_RATE_SCALE),
        'MASS_TURB total':               r0 ** (-5.0 / 3.0) / (0.423 * k ** 2),
        'Cn2 fraction below 2000m':      glf,
        'Cn2_alt_binned_2':              np.full_like(seeing, heights[1] / 1e3),
        'Cn2_frac_binned_2':             1.0 - glf,
        'Cn2_alt_binned_3':              np.full_like(seeing, heights[2] / 1e3),
        'Cn2_frac_binned_3':             np.zeros_like(seeing),
    })[ETC_PROFILE_FEATURES]


def etc_direct_telemetry(etc: pd.DataFrame):
    """ Physical-unit values of the ETC input features [N, len(ETC_DIRECT_FEATURES)] """
    return pd.DataFrame({
        'Seeing (header)':    etc['seeing'].to_numpy(float),
        'Tau0 (header)':      etc['tau0'].to_numpy(float),
        'Tel. altitude':      90.0 - etc['zenith_deg'].to_numpy(float),
        'NGS mag (from ph.)': etc['ngs_mag'].to_numpy(float),
        'IRLOS frequency':    etc['irlos_freq'].to_numpy(float),
        'IRLOS gain':         etc['irlos_gain'].to_numpy(float),
    })[ETC_DIRECT_FEATURES]


# ------------------------------------------------- Donor selection -------------------------------------------------
class DonorSelector:
    """
    Nearest-neighbour selection of real training observations ('donors') for ETC conditions, in the standardized space of the
    conditioning features (`ETC_DIRECT_FEATURES` by default, optionally weighted). `draw` returns one of the `k` nearest donors
    per query at random (`k=1`: the nearest), `neighbours` returns all `k`.
    """
    def __init__(self, pool_cond: np.ndarray, k=10, weights=None):
        self.pool_cond = np.asarray(pool_cond, float)   # [N_pool, N_cond]
        self.k         = int(k)
        self.weights   = np.ones(self.pool_cond.shape[1]) if weights is None else np.asarray(weights, float)

    def neighbours(self, query_cond: np.ndarray):
        """ Indices [N, k] of the k nearest donors and their distances [N, k] """
        q = np.asarray(query_cond, float)
        d = np.sqrt((((q[:, None, :] - self.pool_cond[None, :, :]) * self.weights) ** 2).sum(-1))  # [N, N_pool]
        k = min(self.k, self.pool_cond.shape[0])
        idx = np.argsort(d, axis=1)[:, :k]
        return idx, np.take_along_axis(d, idx, axis=1)

    def draw(self, query_cond: np.ndarray, rng: np.random.Generator):
        idx, _ = self.neighbours(query_cond)
        pick = rng.integers(0, idx.shape[1], size=idx.shape[0])
        return idx[np.arange(idx.shape[0]), pick]


# ------------------------------------------------- Telemetry builders -------------------------------------------------
class ETCTelemetryBuilder:
    """
    Standardized 31-D calibrator telemetry for ETC conditions.

    The ETC inputs (`ETC_DIRECT_FEATURES`) are always set from the ETC table. The site / profile features come from the TC
    constants (`profile='TC'`) or from the donor observation (`profile='donor'`, requires `fill='donor'`). The remaining features
    (AO diagnostics, nuisances) are filled by
      - 'impute':   the calibrator's trained linear imputer (conditional mean, deterministic),
      - 'donor':    copied from a nearest-neighbour real observation (`DonorSelector`), `deterministic=True` averages the k donors,
      - 'gaussian': a linear-Gaussian model of the unknown features given the ETC inputs, fitted on the pool (`deterministic=True`
                    returns the conditional mean).
    """
    def __init__(self, preprocessor: TelemetryPreprocessor, fill='impute', profile='TC', pool_scaled=None, pool_heights=None,
                 pool_phase_bump=None, k=10, deterministic=False, cond_weights=None, heights=ETC_WHITE_PROFILE_HEIGHTS):
        
        if fill not in ('impute', 'donor', 'gaussian'):
            raise ValueError(f"Unknown fill '{fill}'")
        
        if profile not in ('TC', 'donor'):
            raise ValueError(f"Unknown profile policy '{profile}'")
        
        if profile == 'donor' and fill != 'donor':
            raise ValueError("profile='donor' requires fill='donor'")
        
        self.pre           = preprocessor
        self.features      = list(preprocessor.features)
        self.fill          = fill
        self.profile       = profile
        self.k             = k
        self.deterministic = deterministic
        self.heights       = list(heights)
        self.direct_idx    = np.array([self.features.index(f) for f in ETC_DIRECT_FEATURES])
        self.profile_idx   = np.array([self.features.index(f) for f in ETC_PROFILE_FEATURES])
        self.rest_idx      = np.array([i for i in range(len(self.features)) if i not in set(self.direct_idx) | set(self.profile_idx)])
        self.unknown_idx   = np.concatenate([self.profile_idx, self.rest_idx]) if profile == 'donor' else self.rest_idx
        self.pool_scaled   = None if pool_scaled is None else np.asarray(pool_scaled, np.float64)
        self.pool_heights  = None if pool_heights is None else np.asarray(pool_heights, float)
        self.pool_phase_bump = None if pool_phase_bump is None else np.asarray(pool_phase_bump, float)
        self.selector      = None
        self.gauss         = None
        
        if fill in ('donor', 'gaussian'):
            if self.pool_scaled is None:
                raise ValueError(f"fill='{fill}' needs the pool of training telemetry")
            
            self.selector = DonorSelector(self.pool_scaled[:, self.direct_idx], k=k, weights=cond_weights)
            
            if fill == 'gaussian':
                self._fit_gaussian()

    @property
    def name(self):
        tag = self.fill + ('-mean' if self.deterministic and self.fill != 'impute' else '')
        return f'{tag}/{self.profile}'

    def _fit_gaussian(self, ridge=1e-2, shrink=0.1):
        """ Linear regression of the unknown features on the ETC inputs plus the (shrunk) residual covariance, in standardized units """
        X = np.hstack([self.pool_scaled[:, self.direct_idx], np.ones((self.pool_scaled.shape[0], 1))])
        Y = self.pool_scaled[:, self.unknown_idx]
        
        beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
        R = Y - X @ beta
        cov = np.cov(R, rowvar=False)
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        
        self.gauss = {'beta': beta, 'chol': np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0])),
                      'lo': Y.min(0), 'hi': Y.max(0)}

    def scale(self, physical: pd.DataFrame, idx):
        """ Physical -> standardized units for the features at `idx` (DataFrame columns in that order) """
        return (physical.to_numpy(float) - self.pre.mean[idx]) / self.pre.scale[idx]

    def build(self, etc: pd.DataFrame, rng=None):
        """
        Standardized telemetry [N, 31] (float32) for the ETC rows, plus a dict with `donor` (pool indices or -1), `heights`
        ([N, 3] Cn2 heights of the donors' configs when available, else None) and `phase_bump` (donors' fitted phase bump or None).
        """
        rng = np.random.default_rng() if rng is None else rng
        N = len(etc)
        Z = np.full((N, len(self.features)), np.nan)
        Z[:, self.direct_idx] = self.scale(etc_direct_telemetry(etc), self.direct_idx)
        if self.profile == 'TC':
            Z[:, self.profile_idx] = self.scale(tc_profile_telemetry(etc, self.heights), self.profile_idx)

        donors, heights, bump = np.full(N, -1), None, None
        if self.fill == 'impute':
            Z = self.pre.impute(Z)
        elif self.fill == 'donor':
            if self.deterministic:
                nb, _ = self.selector.neighbours(Z[:, self.direct_idx])
                Z[:, self.unknown_idx] = self.pool_scaled[nb][:, :, self.unknown_idx].mean(1)
                donors = nb[:, 0]
            else:
                donors = self.selector.draw(Z[:, self.direct_idx], rng)
                Z[:, self.unknown_idx] = self.pool_scaled[donors][:, self.unknown_idx]
            if self.pool_heights is not None:
                heights = self.pool_heights[donors]
            if self.pool_phase_bump is not None:
                bump = self.pool_phase_bump[donors]
        else:
            X = np.hstack([Z[:, self.direct_idx], np.ones((N, 1))])
            mean = X @ self.gauss['beta']
            if not self.deterministic:
                mean = mean + rng.standard_normal(mean.shape) @ self.gauss['chol'].T
            Z[:, self.unknown_idx] = np.clip(mean, self.gauss['lo'], self.gauss['hi'])
        return Z.astype(np.float32), {'donor': donors, 'heights': heights, 'phase_bump': bump}


# ------------------------------------------------- Predictors -------------------------------------------------
def _fitted_to_x_dict(fitted: dict, idx, device):
    """
    PSF-model parameters of the pool observations `idx` from the stored fitted values: F_ctrl, J_ctrl, r0, L0, dn,
    wind_speed_single, Cn2_weights (ground layer reconstructed, 3 layers) and LO_coefs (8, phase bump first).
    """
    t = lambda a: torch.as_tensor(np.asarray(a)[idx], dtype=torch.float32, device=device)
    cn2 = np.clip(np.asarray(fitted['Cn2_weights'])[idx], 0.0, None)
    gl  = np.clip(1.0 - cn2.sum(-1, keepdims=True), 1e-3, None)
    return {
        'F_ctrl':            t(fitted['F']),
        'J_ctrl':            t(fitted['J']),
        'r0':                t(fitted['r0']),
        'L0':                t(fitted['L0']),
        'dn':                t(fitted['dn']),
        'wind_speed_single': t(fitted['wind_speed_single']).view(-1, 1),
        'Cn2_weights':       torch.as_tensor(np.hstack([gl, cn2]), dtype=torch.float32, device=device),
        'LO_coefs':          t(fitted['LO_coefs']),
    }


class ETCPredictor:
    """ Common interface: `predict(etc, rng) -> (x_dict, info)`; `info` has `heights` ([N, 3] or None), `donor`, `phase_bump` """
    name       = 'predictor'
    stochastic = False

    def predict(self, etc: pd.DataFrame, rng=None):
        raise NotImplementedError


class CalibratorPredictor(ETCPredictor):
    """
    Strategy B: the PSF-R calibrator (`NFMCalibrator`) applied to telemetry built by an `ETCTelemetryBuilder`. The phase bump
    (first LO coefficient, not predicted by the calibrator) is set to `phase_bump` (a value, or 'donor' to take the donor's).
    """
    def __init__(self, calibrator: NFMCalibrator, builder: ETCTelemetryBuilder, phase_bump=0.0):
        self.calibrator = calibrator
        self.builder    = builder
        self.phase_bump = phase_bump
        self.stochastic = builder.fill != 'impute' and not builder.deterministic
        self.name       = f'calibrator[{builder.name}]'

    @torch.no_grad()
    def predict(self, etc, rng=None):
        Z, info = self.builder.build(etc, rng)
        x = self.calibrator.forward_prepared(torch.as_tensor(Z, device=self.calibrator.device))
        if 'LO_coefs' in x:
            if isinstance(self.phase_bump, str) and self.phase_bump == 'donor' and info['phase_bump'] is not None:
                bump = torch.as_tensor(info['phase_bump'], dtype=x['LO_coefs'].dtype, device=x['LO_coefs'].device)
            else:
                bump = torch.full((len(etc),), 0.0 if isinstance(self.phase_bump, str) else float(self.phase_bump),
                                  dtype=x['LO_coefs'].dtype, device=x['LO_coefs'].device)
            x['LO_coefs'] = torch.cat([bump.view(-1, 1), x['LO_coefs'][:, 1:]], dim=1)
        info['telemetry'] = Z
        return x, info


class DonorParamsPredictor(ETCPredictor):
    """
    Strategy A (non-parametric): the fitted PSF-model parameters of a nearest-neighbour training observation (same donor
    selection as the telemetry builder). r0 is rescaled to the requested seeing (r0 ~ 1/seeing at 500 nm, both at zenith) and the
    ground-layer wind speed to the requested tau0 (v ~ r0 / tau0), when enabled. The donor's Cn2 heights go with its weights.
    """
    def __init__(self, pool_scaled, pool_fitted: dict, pool_direct_physical: pd.DataFrame, preprocessor, pool_heights=None,
                 k=10, deterministic=False, rescale_r0=True, rescale_wind=True, wind_ratio_clip=(0.5, 2.0), cond_weights=None, device=default_device):
        self.features     = list(preprocessor.features)
        self.direct_idx   = np.array([self.features.index(f) for f in ETC_DIRECT_FEATURES])
        self.pre          = preprocessor
        self.pool_scaled  = np.asarray(pool_scaled, np.float64)
        self.pool_fitted  = pool_fitted
        self.pool_seeing  = pool_direct_physical['Seeing (header)'].to_numpy(float)
        self.pool_tau0    = pool_direct_physical['Tau0 (header)'].to_numpy(float)
        self.pool_heights = None if pool_heights is None else np.asarray(pool_heights, float)
        self.selector     = DonorSelector(self.pool_scaled[:, self.direct_idx], k=k, weights=cond_weights)
        self.deterministic, self.rescale_r0, self.rescale_wind, self.wind_ratio_clip = deterministic, rescale_r0, rescale_wind, wind_ratio_clip
        self.device       = device
        self.stochastic   = not deterministic
        self.name         = 'donor-params' + ('-nearest' if deterministic else '') + ('' if rescale_r0 else '-noscale')

    @torch.no_grad()
    def predict(self, etc, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        q = (etc_direct_telemetry(etc).to_numpy(float) - self.pre.mean[self.direct_idx]) / self.pre.scale[self.direct_idx]
        donors = self.selector.neighbours(q)[0][:, 0] if self.deterministic else self.selector.draw(q, rng)
        x = _fitted_to_x_dict(self.pool_fitted, donors, self.device)
        seeing, tau0 = etc['seeing'].to_numpy(float), etc['tau0'].to_numpy(float)
        r0_ratio = self.pool_seeing[donors] / seeing if self.rescale_r0 else np.ones(len(etc))
       
        if self.rescale_r0:
            x['r0'] = x['r0'] * torch.as_tensor(r0_ratio, dtype=torch.float32, device=self.device)
        
        if self.rescale_wind:
            v_ratio = np.clip(r0_ratio * self.pool_tau0[donors] / tau0, *self.wind_ratio_clip)
            x['wind_speed_single'] = x['wind_speed_single'] * torch.as_tensor(v_ratio, dtype=torch.float32, device=self.device).view(-1, 1)
       
        heights = None if self.pool_heights is None else self.pool_heights[donors]
        return x, {'donor': donors, 'heights': heights, 'phase_bump': np.asarray(self.pool_fitted['LO_coefs'])[donors, 0]}


class DirectNetPredictor(ETCPredictor):
    """
    Strategy A (parametric): `SmallCalibratorNet` from the standardized ETC inputs to the stacked PSF-model parameters, trained
    end-to-end with `NFMCalibratorTrainer` on `ETCDataset`. The phase bump is a constant (`phase_bump`, the training median).
    """
    def __init__(self, net: nn.Module, outputs_transformer: InputsTransformer, feature_mean, feature_scale, phase_bump=0.0, device=default_device):
        self.net = net.to(device).eval()
        self.outputs_transformer = outputs_transformer
        self.mean, self.scale = np.asarray(feature_mean, float), np.asarray(feature_scale, float)
        self.phase_bump = float(phase_bump)
        self.device = device
        self.name = 'direct-net'

    @torch.no_grad()
    def predict(self, etc, rng=None):
        Z = (etc_direct_telemetry(etc).to_numpy(float) - self.mean) / self.scale
        x = self.outputs_transformer.unstack(self.net(torch.as_tensor(Z, dtype=torch.float32, device=self.device)))
        if 'LO_coefs' in x:
            bump = torch.full((len(etc), 1), self.phase_bump, dtype=x['LO_coefs'].dtype, device=self.device)
            x['LO_coefs'] = torch.cat([bump, x['LO_coefs']], dim=1)
        return x, {'donor': np.full(len(etc), -1), 'heights': None, 'phase_bump': None}


# ------------------------------------------------- Rendering -------------------------------------------------
def build_etc_config(template: dict, row, wavelengths_m, heights, ho_photons=ETC_HO_PHOTONS):
    """
    Per-observation TipTorch config (lists layout) for one ETC row: template config with the ETC atmosphere (seeing, TC outer
    scale / ground-layer fraction / ASM wind, `heights` [3] in m), zenith angle, LGS flux, NGS flux and IRLOS regime, wavelengths.
    The atmosphere entries only matter where the predictors do not override them (layer heights, wind direction).
    """
    tc  = int(row['tc'])
    glf = TC_GLF[tc]
    cfg = deepcopy(template)
    cfg['telescope']['ZenithAngle']        = [[float(row['zenith_deg'])]]
    cfg['telescope']['PathPupil']          = str(PUPIL_PATH)
    cfg['telescope']['PupilAngle']         = 0.0
    cfg['atmosphere']['Seeing']            = [float(row['seeing'])]
    cfg['atmosphere']['L0']                = [TC_L0[tc]]
    cfg['atmosphere']['Cn2Weights']        = [[glf, 1.0 - glf, 0.0]]
    cfg['atmosphere']['Cn2Heights']        = [[float(h) for h in heights]]
    cfg['atmosphere']['WindSpeed']         = [[TC_WIND_SPEED[tc]] * 3]
    cfg['atmosphere']['WindDirection']     = [[TC_WIND_DIR[tc]] * 3]
    cfg['sources_science']['Wavelength']   = [float(w) for w in wavelengths_m]
    cfg['sources_HO']['Height']            = [[90e3]]  # TipTorch stretches the LGS height with the airmass itself
    cfg['sensor_HO']['NumberPhotons']      = [[ho_photons] * 4]
    cfg['sensor_LO']['NumberPhotons']      = [int(max(float(row['ngs_flux']), 1.0))]
    cfg['sensor_LO']['Gain']               = [int(row['irlos_gain'])]
    cfg['RTC']['SensorFrameRate_LO']       = [int(row['irlos_freq'])]
    cfg['DM']['NumberReconstructedLayers'] = [3]
    return cfg


def flatten_x_dict(x_dict):
    """ dict of tensors [N, ...] -> dict of 1-D numpy columns (`name` or `name[j]`) """
    cols = {}
    for key, val in x_dict.items():
        v = val.detach().cpu().numpy()
        if v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1):
            cols[key] = v.reshape(-1)
        else:
            for j in range(v.shape[1]):
                cols[f'{key}[{j}]'] = v[:, j]
    return cols


class ETCPSFRenderer:
    """
    Renders the PSFs of an ETC table with `PSFModelNFM` for any `ETCPredictor`: configs are built from `template_config`,
    the predicted parameters applied, PSFs returned as [N, N_wvl, H, W] (CPU) together with a parameter table.
    `heights_policy`: 'median' (dataset median Cn2 heights), 'white' (TIPTOP ETC profile heights) or 'donor' (the predictor's
    donor heights, falling back to the median).
    """
    def __init__(self, template_config: dict, model_cfg: dict, wavelengths_nm, median_heights, heights_policy='median',
                 batch_size=16, device=default_device):
        self.template       = template_config
        self.model_cfg      = dict(model_cfg)
        self.wavelengths_nm = [float(w) for w in wavelengths_nm]
        self.median_heights = [float(h) for h in median_heights]
        self.heights_policy = heights_policy
        self.batch_size     = batch_size
        self.device         = device

    def _heights(self, info, i):
        if self.heights_policy == 'white':
            return ETC_WHITE_PROFILE_HEIGHTS
        if self.heights_policy == 'donor' and info.get('heights') is not None:
            return info['heights'][i]
        return self.median_heights

    def make_model(self, configs):
        config = MultipleTargetsInDifferentObservations(configs, device=self.device)
        config['DM']['NumberReconstructedLayers'] = torch.tensor(config['atmosphere']['Cn2Heights'].shape[-1], device=self.device)
        return PSFModelNFM(
            config, multiple_obs=True, device=self.device,
            LO_NCPAs=self.model_cfg['LO_NCPAs'], chrom_defocus=self.model_cfg['chrom_defocus'], use_Moffat=self.model_cfg['use_Moffat'],
            retain_PSDs=self.model_cfg['retain_PSDs'], N_spline_nodes=self.model_cfg['N_spline_nodes'], Z_mode_max=self.model_cfg['Z_mode_max'],
        )

    @torch.no_grad()
    def render(self, predictor: ETCPredictor, etc: pd.DataFrame, rng=None, n_draws=1, return_DL=False, verbose=False):
        """
        PSFs [N * n_draws, N_wvl, H, W] and a table with the ETC columns, `draw`, `donor` and the predicted parameters
        (one block of N rows per draw; a deterministic predictor is rendered once regardless of `n_draws`).
        """
        rng = np.random.default_rng() if rng is None else rng
        n_draws = n_draws if predictor.stochastic else 1
        wvl_m = [w * 1e-9 for w in self.wavelengths_nm]
        PSFs, DLs, tables = [], [], []
        for draw in range(n_draws):
            x_all, info = predictor.predict(etc, rng)
            for start in range(0, len(etc), self.batch_size):
                ids = list(range(start, min(start + self.batch_size, len(etc))))
                configs = [build_etc_config(self.template, etc.iloc[i], wvl_m, self._heights(info, i)) for i in ids]
                model = self.make_model(configs)
                x = {k: v[ids] for k, v in x_all.items()}
                model.update_manager_params(x)
                PSFs.append(model().detach().cpu())
                if return_DL:
                    dl = model.model.DLPSF().detach().cpu()
                    DLs.append(dl.expand(len(ids), -1, -1, -1) if dl.shape[0] == 1 else dl)
                model.cleanup()
                del model
                release_gpu_memory()
                if verbose:
                    print(f'\r  {predictor.name}: draw {draw + 1}/{n_draws}, {ids[-1] + 1}/{len(etc)} rendered', end='', flush=True)
            table = etc.reset_index(drop=True).copy()
            table['draw']  = draw
            table['donor'] = info['donor']
            for k, v in flatten_x_dict(x_all).items():
                table[k] = v
            tables.append(table)
        if verbose:
            print()
        out = (torch.cat(PSFs, 0), pd.concat(tables, ignore_index=True))
        return out + (torch.cat(DLs, 0),) if return_DL else out


# ------------------------------------------------- Dataset view for the direct network -------------------------------------------------
class ETCDataset(torch.utils.data.Dataset):
    """
    View of an `NFMDataset` exposing only the ETC input features as telemetry (standardized like the full vectors) and,
    optionally, configs whose Cn2 layer heights are replaced by `heights` (the inference-time profile). Everything else
    (`fitted_vals`, split, time groups, shapes) is forwarded so that `NFMCalibratorTrainer` can train on it.
    """
    def __init__(self, base, features=ETC_DIRECT_FEATURES, heights=None):
        self.base         = base
        self.features     = list(features)
        self.feature_idx  = torch.tensor([base.features.index(f) for f in self.features])
        self.fitted_vals  = base.fitted_vals
        self.sample_ids   = base.sample_ids
        self.train_idx, self.val_idx, self.time_groups = base.train_idx, base.val_idx, base.time_groups
        self.N, self.H, self.W, self.C = base.N, base.H, base.W, base.C
        self.telemetry_preprocessor = None
        self.configs = base.configs
        if heights is not None:
            self.configs = []
            for cfg in base.configs:
                cfg = deepcopy(cfg)
                cfg['atmosphere']['Cn2Heights'] = [[float(h) for h in heights]]
                self.configs.append(cfg)
        self.collate_batch = base.collate_batch

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        PSFs, tel, fit, _, i = self.base[idx]
        return PSFs, tel[self.feature_idx], fit, self.configs[idx], i


# ------------------------------------------------- Predictor presets -------------------------------------------------
# name -> (predictor kind, predictor kwargs, Cn2 heights policy of the renderer, stochastic)
PREDICTOR_PRESETS = {
    'impute':               ('impute',               {},                   'median', False),
    'impute-white':         ('impute',               {},                   'white',  False),  # the previous Monte Carlo recipe
    'gaussian-mean':        ('gaussian-mean',        {},                   'median', False),
    'gaussian':             ('gaussian',             {},                   'median', True),
    'donor-mean':           ('donor-mean',           {},                   'median', False),
    'donor':                ('donor',                {},                   'median', True),
    'donor-profile':        ('donor',                {'profile': 'donor'}, 'donor',  True),
    'donor-params':         ('donor-params',         {},                   'donor',  True),
    'donor-params-nearest': ('donor-params-nearest', {},                   'donor',  False),
    'direct-net':           ('direct-net',           {},                   'median', False),
}


# ------------------------------------------------- Bundle -------------------------------------------------
class ETCCalibrator:
    """
    ETC calibrator bundle (`train_ETC_calibrator.py`): the pool of training observations (standardized telemetry, fitted
    PSF-model parameters, Cn2 heights), the direct network, the dataset median heights, the template config and a reference to
    the PSF-R calibrator bundle. `predictor(kind, ...)` builds the predictors, `renderer(...)` the PSF renderer.
    """
    def __init__(self, bundle_path, device=default_device, psfr_bundle=None):
        self.device = device
        self.path   = Path(bundle_path)
        self.state  = torch.load(self.path, map_location='cpu', weights_only=False)
        self.pre    = TelemetryPreprocessor.from_state_dict(self.state['telemetry_preprocessor'])
        self.pool   = self.state['pool']
        self.median_heights = list(self.state['median_heights'])
        self.phase_bump_median = float(self.state['phase_bump_median'])
        self.template_config = self.state['template_config']
        self.model_cfg = self.state['model_cfg']
        self.psfr_bundle_path = Path(psfr_bundle) if psfr_bundle else Path(self.state['psfr_calibrator_bundle'])
        self._psfr = None
        self._direct = None

    @property
    def psfr_calibrator(self):
        if self._psfr is None:
            self._psfr = NFMCalibrator(self.psfr_bundle_path, device=self.device)
            if list(self._psfr.features) != list(self.pre.features):
                raise ValueError('The PSF-R calibrator features differ from the ETC bundle telemetry features')
        return self._psfr

    @property
    def pool_direct_physical(self):
        idx = [self.pre.features.index(f) for f in ETC_DIRECT_FEATURES]
        return pd.DataFrame(self.pre.inverse_transform(self.pool['telemetry_scaled'])[:, idx], columns=ETC_DIRECT_FEATURES)

    def builder(self, fill='impute', profile='TC', **kw):
        return ETCTelemetryBuilder(self.pre, fill=fill, profile=profile, pool_scaled=self.pool['telemetry_scaled'],
                                   pool_heights=self.pool['heights'], pool_phase_bump=np.asarray(self.pool['fitted']['LO_coefs'])[:, 0], **kw)

    def direct_net_predictor(self):
        if self._direct is None:
            d = self.state.get('direct_net')
            if d is None:
                raise ValueError('The ETC bundle has no direct network')
            net = SmallCalibratorNet(**d['arch'])
            net.load_state_dict(d['state_dict'])
            self._direct = DirectNetPredictor(net, InputsTransformer.load(data=d['outputs_transformer']), d['feature_mean'], d['feature_scale'],
                                              phase_bump=self.phase_bump_median, device=self.device)
        return self._direct

    def predictor(self, kind, **kw):
        """
        kind: 'impute' | 'gaussian' | 'gaussian-mean' | 'donor' | 'donor-mean' (PSF-R calibrator on built telemetry),
              'donor-params' | 'donor-params-nearest' (fitted parameters of donors), 'direct-net'.
        kw: `profile`, `k`, `phase_bump`, `cond_weights`, `rescale_r0`, `rescale_wind`.
        """
        phase_bump = kw.pop('phase_bump', self.phase_bump_median)
        if kind == 'direct-net':
            return self.direct_net_predictor()
        if kind.startswith('donor-params'):
            return DonorParamsPredictor(self.pool['telemetry_scaled'], self.pool['fitted'], self.pool_direct_physical, self.pre,
                                        pool_heights=self.pool['heights'], deterministic=kind.endswith('nearest'), device=self.device, **kw)
        fill, deterministic = kind.replace('-mean', ''), kind.endswith('-mean')
        builder = self.builder(fill=fill, deterministic=deterministic, **kw)
        return CalibratorPredictor(self.psfr_calibrator, builder, phase_bump=phase_bump)

    def renderer(self, wavelengths_nm, heights_policy='median', batch_size=16):
        return ETCPSFRenderer(self.template_config, self.model_cfg, wavelengths_nm, self.median_heights, heights_policy, batch_size, self.device)
