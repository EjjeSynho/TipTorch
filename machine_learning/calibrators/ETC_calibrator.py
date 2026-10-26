"""
ETC-regime PSF prediction for MUSE NFM.

An exposure time calculator (ETC) knows only a few things about an observation: seeing, coherence time tau0, airmass, NGS
magnitude (hence the IRLOS regime) and the turbulence category (TC). This module predicts the TipTorch PSF-model parameters from
these ETC inputs alone, with two predictors sharing the interface `predict(etc, rng) -> (x_dict, info)`:

- `DonorPredictor`    non-parametric sampler: among the training observations closest to the query in the standardized space of
                      the ETC inputs, one ('donor') is drawn and its *fitted* PSF-model parameters are reused, r0 rescaled to the
                      requested seeing and the ground-layer wind to the requested tau0. Repeated draws give the distribution of
                      possible PSFs; `nearest=True` makes it deterministic.
- `DirectNetPredictor` a small network (`SmallCalibratorNet`) from the ETC inputs to the PSF-model parameters, trained end-to-end
                      with the PSF model (`train_ETC_calibrator.py`); the 'expected' PSF of the conditions.

`ETCPSFRenderer` renders the predicted parameters with `PSFModelNFM`; `ETCCalibrator` loads the bundle (donor pool, network,
template config) written by `train_ETC_calibrator.py`; `ETCDataset` is the dataset view used to train the network.

The PSF model is the one of the STD-star fits and of the NFM calibrator (`build_PSF_model`, the "model" section of the training config):
anisotropic jitter (Jx, Jy, J_angle) and the image-frame wing (wing_frac, fixed shape) since 2026-10-21. The kept fixed inputs of the
training (J_angle and the wing shape at the training medians of the fits) are stored in the bundle (`fixed_inputs`) and set by the renderer.

Conventions: ETC tables are pandas DataFrames with one row per observation (`etc_table`); `x_dict` holds the PSF-model parameters
as tensors [N, ...] on the device; rendered PSFs are [N, N_wvl, H, W] on the CPU.
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiptorch._config import DATA_FOLDER, default_device, project_settings
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from tiptorch.managers.input_manager import InputsTransformer
from calibrators.NFM_calibrator import SmallCalibratorNet, build_PSF_model, release_gpu_memory

PUPIL_PATH = DATA_FOLDER / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'
STD_DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache'


def dataset_cache_path(derotated: bool) -> Path:
    """ STD-star dataset cache of the calibrators: derotated PSFs (configs with PupilAngle 0) or raw frames (per-sample pupil angles) """
    return STD_DATASET_CACHE / ('muse_STD_stars_dataset.pt' if derotated else 'muse_STD_stars_dataset_raw.pt')


def donor_keys(model_cfg: dict) -> list:
    """ PSF-model inputs a donor passes on (the keys of the fits), in the jitter / wing regime of the "model" section `model_cfg` """
    jitter = ['Jx_ctrl', 'Jy_ctrl', 'J_angle'] if model_cfg.get('anisotropic_jitter', False) else ['J_ctrl']
    wing   = ['wing_frac_ctrl'] if model_cfg.get('wing_kernel', False) else []
    return ['F_ctrl', *jitter, *wing, 'r0', 'L0', 'dn', 'wind_speed_single', 'Cn2_weights', 'LO_coefs']

# The calibrator telemetry features an ETC can provide (the IRLOS regime follows from the NGS magnitude, see `IRLOS_frequency`)
ETC_INPUTS = ['Seeing (header)', 'Tau0 (header)', 'Tel. altitude', 'NGS mag (from ph.)', 'IRLOS frequency', 'IRLOS gain']
PREDICTORS = ['donor', 'donor-nearest', 'direct-net']

# ------------------------------------------- ETC conditions (Enrico's TC constants) -------------------------------------------
TC_LIST       = [10, 20, 30, 50, 100]
TC_WIND_SPEED = {10: 10.4,  20: 10.5,  30: 9.9,   50: 10.1,  100: 14.5}   # ASM wind speed [m/s]
TC_WIND_DIR   = {10: 12.5,  20: 28.8,  30: 40.2,  50: 37.2,  100: 37.0}   # ASM wind direction [deg]
TC_L0         = {10: 12.48, 20: 17.1,  30: 22.1,  50: 23.5,  100: 25.5}   # outer scale [m]
TC_GLF        = {10: 0.971, 20: 0.973, 30: 0.980, 50: 0.972, 100: 0.977}  # ground-layer fraction of the median profile
ETC_HO_PHOTONS = 1740.2396627862904  # [photons/subaperture/frame], LGS WFS flux used by the ETC


def turbulence_categories(seeing, tau0_ms):
    """ ESO turbulence category (percentile) from seeing ["] and coherence time [ms] """
    seeing, tau = np.atleast_1d(np.asarray(seeing, float)), np.atleast_1d(np.asarray(tau0_ms, float))
    tc = np.full(len(seeing), 100, dtype=int)
    tc[(seeing <= 1.0) & (tau >= 3.2)] = 50
    tc[(seeing <= 0.8) & (tau >= 4.1)] = 30
    tc[(seeing <= 0.7) & (tau >= 4.4)] = 20
    tc[(seeing <= 0.6) & (tau >= 5.2)] = 10
    return tc


def IRLOS_frequency(ngs_mag):
    """ IRLOS loop frequency [Hz] as a function of the NGS J magnitude (Enrico's regime switch) """
    return np.where(np.asarray(ngs_mag, float) <= 16, 500.0, 200.0)


def IRLOS_gain(ngs_mag):
    """ IRLOS detector gain as a function of the NGS J magnitude (Enrico's regime switch) """
    m = np.asarray(ngs_mag, float)
    return np.where(m <= 10.5, 1.0, np.where(m <= 16, 68.0, 100.0))


def NGS_mag_to_photons(Jmag, frequency, airmass, k_J=0.09, delta_mag=0.4):
    """ IRLOS photons per subaperture per frame from the NGS J magnitude (Enrico's conversion) """
    IRLOS_mag = np.asarray(Jmag, float) + delta_mag + k_J * np.asarray(airmass, float)
    return 1.9e12 / 368 * 10 ** (-0.4 * IRLOS_mag) / (4 * np.asarray(frequency, float))


def ETC_table(seeing, tau0, airmass, ngs_mag, irlos_freq=None, irlos_gain=None, tc=None):
    """
    ETC conditions table: `seeing` ["], `tau0` [s], `airmass`, `ngs_mag` [J mag]; the TC and the IRLOS regime are derived unless
    given. Adds `zenith_deg` and `ngs_flux` [photons/subaperture/frame].
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


def ETC_table_from_stars(stars):
    """ ETC table from the Monte Carlo observation table of `run_tiptop_random_carlo.py` (`tau` in ms, `ngs` in J mag) """
    df = ETC_table(stars['seeing'], np.asarray(stars['tau'], float) * 1e-3, stars['airmass'], stars['ngs'], tc=stars['tc'])
    if 'index' in stars.colnames:
        df.insert(0, 'index', np.asarray(stars['index']))
    return df


def ETC_table_from_telemetry(telemetry_physical: pd.DataFrame, real_irlos=True):
    """ ETC table replaying real observations from their physical-unit telemetry (`ETC_INPUTS` columns) """
    t = telemetry_physical
    airmass = 1.0 / np.cos(np.deg2rad(90.0 - t['Tel. altitude'].to_numpy(float)))
    kw = {'irlos_freq': t['IRLOS frequency'].to_numpy(float), 'irlos_gain': t['IRLOS gain'].to_numpy(float)} if real_irlos else {}
    return ETC_table(t['Seeing (header)'].to_numpy(float), t['Tau0 (header)'].to_numpy(float), airmass, t['NGS mag (from ph.)'].to_numpy(float), **kw)


def ETC_inputs(etc: pd.DataFrame):
    """ The ETC inputs of an ETC table as the calibrator telemetry features, physical units [N, 6] """
    return np.column_stack([etc['seeing'], etc['tau0'], 90.0 - etc['zenith_deg'], etc['ngs_mag'], etc['irlos_freq'], etc['irlos_gain']]).astype(float)


# ------------------------------------------------- Predictors -------------------------------------------------
class DonorPredictor:
    """
    Fitted PSF-model parameters of a nearest-neighbour training observation. `pool` holds, per training observation, the ETC
    inputs (`etc_inputs` [N, 6], physical units), the fitted parameters (`fitted`, keyed by the PSF-model inputs, see `donor_keys`:
    F_ctrl, Jx_ctrl / Jy_ctrl / J_angle (or J_ctrl), wing_frac_ctrl, r0, L0, dn, wind_speed_single, Cn2_weights [N, 2] without the
    ground layer, LO_coefs [N, 8] with the phase bump first) and the Cn2 layer heights (`heights` [N, 3]). The donor's own jitter
    angle and wing fraction are passed on with the rest of its fit. Queries are matched in the standardized input space (`mean`,
    `scale`); one of the `k` nearest donors is drawn at random (`nearest=True`: always the nearest). With `rescale`, r0 follows the
    requested seeing (r0 ~ 1/seeing, both at zenith) and the ground-layer wind the requested tau0 (v ~ r0/tau0, ratio clipped to [0.5, 2]).
    """
    def __init__(self, pool: dict, mean, scale, k=10, nearest=False, rescale=True, device=default_device):
        self.pool, self.k, self.nearest, self.rescale, self.device = pool, int(k), nearest, rescale, device
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.pool_z = (np.asarray(pool['etc_inputs'], float) - self.mean) / self.scale
        self.stochastic = not nearest
        self.name = 'donor-nearest' if nearest else 'donor'

    def donors(self, etc: pd.DataFrame, rng):
        """ Pool index of the donor of every query row """
        z = (ETC_inputs(etc) - self.mean) / self.scale
        order = np.argsort(np.linalg.norm(z[:, None, :] - self.pool_z[None, :, :], axis=-1), axis=1)[:, :self.k]
        pick = np.zeros(len(etc), int) if self.nearest else rng.integers(0, order.shape[1], size=len(etc))
        return order[np.arange(len(etc)), pick]

    @torch.no_grad()
    def predict(self, etc: pd.DataFrame, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        d = self.donors(etc, rng)
        f = self.pool['fitted']
        t = lambda key: torch.as_tensor(np.asarray(f[key])[d], dtype=torch.float32, device=self.device)
        cn2 = np.clip(np.asarray(f['Cn2_weights'])[d], 0.0, None)
        gl  = np.clip(1.0 - cn2.sum(-1, keepdims=True), 1e-3, None)
        x = {key: t(key) for key in f if key != 'Cn2_weights'}
        x['wind_speed_single'] = x['wind_speed_single'].view(-1, 1)
        x['Cn2_weights'] = torch.as_tensor(np.hstack([gl, cn2]), dtype=torch.float32, device=self.device)
        if self.rescale:
            donor_inputs = np.asarray(self.pool['etc_inputs'])[d]
            r0_ratio = donor_inputs[:, 0] / etc['seeing'].to_numpy(float)
            v_ratio  = np.clip(r0_ratio * donor_inputs[:, 1] / etc['tau0'].to_numpy(float), 0.5, 2.0)
            x['r0'] = x['r0'] * torch.as_tensor(r0_ratio, dtype=torch.float32, device=self.device)
            x['wind_speed_single'] = x['wind_speed_single'] * torch.as_tensor(v_ratio, dtype=torch.float32, device=self.device).view(-1, 1)
        return x, {'donor': d, 'heights': np.asarray(self.pool['heights'])[d]}


class DirectNetPredictor:
    """ `SmallCalibratorNet` from the standardized ETC inputs to the stacked PSF-model parameters; the phase bump is a constant """
    def __init__(self, net, outputs_transformer: InputsTransformer, mean, scale, phase_bump=0.0, device=default_device):
        self.net = net.to(device).eval()
        self.outputs_transformer = outputs_transformer
        self.mean, self.scale, self.phase_bump, self.device = np.asarray(mean, float), np.asarray(scale, float), float(phase_bump), device
        self.stochastic = False
        self.name = 'direct-net'

    @torch.no_grad()
    def predict(self, etc: pd.DataFrame, rng=None):
        z = torch.as_tensor((ETC_inputs(etc) - self.mean) / self.scale, dtype=torch.float32, device=self.device)
        x = self.outputs_transformer.unstack(self.net(z))
        bump = torch.full((len(etc), 1), self.phase_bump, dtype=x['LO_coefs'].dtype, device=self.device)
        x['LO_coefs'] = torch.cat([bump, x['LO_coefs']], dim=1)
        return x, {'donor': np.full(len(etc), -1), 'heights': None}


# ------------------------------------------------- Rendering -------------------------------------------------
def build_ETC_config(template: dict, row, wavelengths_m, heights):
    """
    TipTorch config (lists layout) of one ETC row: the template with the ETC atmosphere (seeing, TC outer scale / ground-layer
    fraction / ASM wind, `heights` [3] in m), zenith angle, LGS flux, NGS flux and IRLOS regime, science wavelengths [m].
    The atmosphere entries only matter where the predictors do not override them (layer heights, wind direction). The pupil is
    not rotated: the ETC metrics are radial, and the jitter and the wing are defined in the image frame whatever the pupil angle.
    """
    tc, glf = int(row['tc']), TC_GLF[int(row['tc'])]
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
    cfg['sensor_HO']['NumberPhotons']      = [[ETC_HO_PHOTONS] * 4]
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
        if v.ndim == 1 or v.shape[1] == 1:
            cols[key] = v.reshape(-1)
        else:
            cols.update({f'{key}[{j}]': v[:, j] for j in range(v.shape[1])})
            
    return cols


class ETCPSFRenderer:
    """
    Renders the PSFs of an ETC table for a predictor with `PSFModelNFM`: configs built from `template_config` (layer heights of
    the donor when the predictor provides them, `median_heights` otherwise), the model of the "model" section `model_cfg`
    (`build_PSF_model`) with the kept fixed inputs `fixed_inputs` ({key: tensor [1, ...]}, e.g. J_angle; a prediction of the same
    input overrides them), predicted parameters applied, PSFs [N, N_wvl, H, W].
    """
    def __init__(self, template_config: dict, model_cfg: dict, wavelengths_nm, median_heights, batch_size=16, device=default_device, fixed_inputs=None):
        self.template, self.model_cfg, self.batch_size, self.device = template_config, dict(model_cfg), batch_size, device
        self.fixed_inputs   = dict(fixed_inputs or {})
        self.wavelengths_nm = [float(w) for w in wavelengths_nm]
        self.median_heights = [float(h) for h in median_heights]


    def make_model(self, configs):
        config = MultipleTargetsInDifferentObservations(configs, device=self.device)
        config['DM']['NumberReconstructedLayers'] = torch.tensor(config['atmosphere']['Cn2Heights'].shape[-1], device=self.device)
        model = build_PSF_model(config, self.model_cfg, self.device)
        names = model.inputs_manager.get_names(optimizable_only=False)

        with torch.no_grad():
            for key, val in self.fixed_inputs.items():
                if key in names:
                    cur = model.inputs_manager[key]
                    cur[:] = torch.as_tensor(val, dtype=cur.dtype, device=cur.device).reshape(1, *cur.shape[1:])

        return model


    @torch.no_grad()
    def render(self, predictor, etc: pd.DataFrame, rng=None, n_draws=1, return_DL=False, verbose=False):
        """
        PSFs [N * n_draws, N_wvl, H, W] and a table with the ETC columns, `draw`, `donor` and the predicted parameters (one block of
        N rows per draw; a deterministic predictor is rendered once).
        """
        rng = np.random.default_rng() if rng is None else rng
        n_draws = n_draws if predictor.stochastic else 1
        wvl_m = [w * 1e-9 for w in self.wavelengths_nm]
        PSFs, DLs, tables = [], [], []
        
        for draw in range(n_draws):
            x_all, info = predictor.predict(etc, rng)
            
            for start in range(0, len(etc), self.batch_size):
                ids = list(range(start, min(start + self.batch_size, len(etc))))
                heights = lambda i: self.median_heights if info['heights'] is None else info['heights'][i]
                model = self.make_model([build_ETC_config(self.template, etc.iloc[i], wvl_m, heights(i)) for i in ids])
                model.update_manager_params({k: v[ids] for k, v in x_all.items()})
                PSFs.append(model().detach().cpu())
                
                if return_DL:
                    dl = model.model.DLPSF().detach().cpu()
                    DLs.append(dl.expand(len(ids), -1, -1, -1) if dl.shape[0] == 1 else dl)
                    
                model.cleanup()
                del model
                release_gpu_memory()
                if verbose:
                    print(f'\r  {predictor.name}: draw {draw + 1}/{n_draws}, {ids[-1] + 1}/{len(etc)} rendered', end='', flush=True)
           
            table = etc.reset_index(drop=True).assign(draw=draw, donor=info['donor'], **flatten_x_dict(x_all))
            tables.append(table)
            
        if verbose:
            print()
            
        out = (torch.cat(PSFs, 0), pd.concat(tables, ignore_index=True))
        return out + (torch.cat(DLs, 0),) if return_DL else out


# ------------------------------------------------- Dataset view for the direct network -------------------------------------------------
class ETCDataset(torch.utils.data.Dataset):
    """
    View of an `NFMDataset` exposing only the ETC inputs as telemetry (standardized like the full vectors) and configs whose Cn2
    layer heights are replaced by `heights` (the inference-time profile); everything else is forwarded for `NFMCalibratorTrainer`,
    including the k_c groups (the trainer batches the samples per group: configs of different DM pitches cannot share a model).
    """
    def __init__(self, base, heights):
        self.base, self.features = base, list(ETC_INPUTS)
        self.feature_idx = torch.tensor([base.features.index(f) for f in ETC_INPUTS])
        self.fitted_vals, self.sample_ids = base.fitted_vals, base.sample_ids
        self.train_idx, self.val_idx, self.time_groups = base.train_idx, base.val_idx, base.time_groups
        self.kc_group, self.kc_fraction, self.derotated = base.kc_group, base.kc_fraction, base.derotated
        self.N, self.H, self.W, self.C = base.N, base.H, base.W, base.C
        self.telemetry_preprocessor = None
        self.collate_batch = base.collate_batch
        self.configs = []
        
        for cfg in base.configs:
            cfg = deepcopy(cfg)
            cfg['atmosphere']['Cn2Heights'] = [[float(h) for h in heights]]
            self.configs.append(cfg)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        PSFs, tel, fit, _, i = self.base[idx]
        return PSFs, tel[self.feature_idx], fit, self.configs[idx], i


# ------------------------------------------------- Bundle -------------------------------------------------
class ETCCalibrator:
    """ ETC calibrator bundle of `train_ETC_calibrator.py`: donor pool, direct network, template config, standardization """
    def __init__(self, bundle_path, device=default_device):
        self.device = device
        self.path   = Path(bundle_path)
        self.state  = torch.load(self.path, map_location='cpu', weights_only=False)
        self.pool, self.mean, self.scale = self.state['pool'], self.state['etc_mean'], self.state['etc_scale']
        self.median_heights, self.phase_bump = list(self.state['median_heights']), float(self.state['phase_bump_median'])
        self.template_config, self.model_cfg = self.state['template_config'], self.state['model_cfg']
        self.fixed_inputs  = self.state.get('fixed_inputs', {})  # bundles before 2026-10-21: none (isotropic jitter, no wing)
        self.dataset_cache = Path(self.state['dataset_cache'])  # the dataset the bundle was built from (its validation split)
        # Pools before 2026-10-21 are keyed F / J instead of the PSF-model inputs F_ctrl / J_ctrl
        self.pool['fitted'] = {({'F': 'F_ctrl', 'J': 'J_ctrl'}).get(k, k): v for k, v in self.pool['fitted'].items()}


    def predictor(self, name, **kw):
        """ 'donor' | 'donor-nearest' (kw: k, rescale) | 'direct-net' """
        if name in ('donor', 'donor-nearest'):
            return DonorPredictor(self.pool, self.mean, self.scale, nearest=name.endswith('nearest'), device=self.device, **kw)
        if name == 'direct-net':
            d = self.state.get('direct_net')
            if d is None:
                raise ValueError('The ETC bundle has no direct network')
            net = SmallCalibratorNet(**d['arch'])
            net.load_state_dict(d['state_dict'])
            return DirectNetPredictor(net, InputsTransformer.load(data=d['outputs_transformer']), d['feature_mean'], d['feature_scale'],
                                      phase_bump=self.phase_bump, device=self.device)
        raise ValueError(f"Unknown predictor '{name}', expected one of {PREDICTORS}")


    def renderer(self, wavelengths_nm, batch_size=16):
        return ETCPSFRenderer(self.template_config, self.model_cfg, wavelengths_nm, self.median_heights, batch_size, self.device, self.fixed_inputs)
