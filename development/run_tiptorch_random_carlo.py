"""
Monte Carlo generation of MUSE NFM PSFs with the calibrated TipTorch model (`PSFModelNFM` + `NFMCalibrator`), the counterpart of
`run_tiptop_random_carlo.py` for the ETC comparison.

The same random observation table (`data/Enrico_ETC/generated_random_PSFs/stars.fits`: seeing, tau0, NGS J magnitude, airmass,
TC) is replayed through the calibrated model, following the ETC recipe of `machine_learning/validate_NFM_calibrator.py`:

- every sample borrows the config and telemetry of a random real observation of the calibrator dataset (template) for the
  quantities the ETC does not provide (pointing, exposure time, temperature, ...);
- the atmosphere / LGS / NGS entries of the config and of the telemetry are replaced by the sampled conditions and by the
  TC-dependent ETC profile used for TIPTOP (`L0`, ground-layer fraction, ASM wind), with the same IRLOS regime switches
  (`IRLOS_frequency`, `IRLOS_gain`, `NGS_mag_to_photons`);
- the AO closed-loop diagnostics of the template (LGS Strehl, residuals, slopes RMS, theta0) are set to NaN and imputed;
- the calibrator predicts the PSF model parameters (r0, L0, Cn2 weights, wind speed, F, J, dn, LO NCPAs) and the model renders the
  PSFs at the TIPTOP wavelengths 500, 600, 700, 800 and 900 nm.

PSFs are cropped to TIPTOP's 100 x 100 px field (25 mas/px, peak at pixel 50) and stored one file per sample and wavelength in
`data/Enrico_ETC/tiptorch_random_PSFs/{wave}nm/<name>.fits` in TIPTOP's FITS layout, with TIPTOP's header metrics (SR, FWHM, EE50)
computed by `ETC_psf_metrics.tiptop_metrics`. The predicted model parameters go to `tiptorch_random_PSFs/predicted_params.fits`
and the telemetry to `tiptorch_random_PSFs/telemetry.csv`.

Requirements: the trained calibrator bundle referenced by `machine_learning/NFM_calibrator_config.json` (`*_meta.json` in
`data/weights/NFM_calibrator/`), the dataset cache `muse_STD_stars_dataset.pt` (loaded with mmap, only configs and telemetry are
read), a CUDA device (~1 min for 466 samples) and the astro-tiptop checkouts for the P3 metric functions (`tiptop_env.py`).
Run from the repository root in the `TipTorch` conda env:

    python development/run_tiptorch_random_carlo.py [--batch-size 16] [--seed 0] [--overwrite]
"""

import sys
import json
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    sys.path.insert(0, str(p))

from tiptorch._config import DATA_FOLDER, WEIGHTS_FOLDER, default_device, project_settings
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from calibrators.NFM_calibrator import NFMCalibrator, release_gpu_memory

import tiptop_env  # noqa: F401
import run_tiptop_random_carlo as rc
from ETC_psf_metrics import load_pupil, tiptop_metrics, write_psf_file, crop_center, normalize_psf, PIXEL_SCALE_MAS, PSF_SIZE

ETC_FOLDER    = REPO_ROOT / 'data' / 'Enrico_ETC'
OUTPUT_DIR    = ETC_FOLDER / 'tiptorch_random_PSFs'
STARS_TABLE   = ETC_FOLDER / 'generated_random_PSFs' / 'stars.fits'
CONFIG_PATH   = REPO_ROOT / 'machine_learning' / 'NFM_calibrator_config.json'
DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache' / 'muse_STD_stars_dataset.pt'
PUPIL_PATH    = DATA_FOLDER / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'

WAVELENGTHS_M = [w * 1e-9 for w in rc.WAVELENGTHS]
ETC_LGS_PHOTON_RATE_SCALE = 1240e3   # subapertures * HO frames per second (validate_NFM_calibrator.py)
ETC_REFERENCE_WAVELENGTH  = 500e-9   # seeing / r0 reference wavelength [m]
CN2_HEIGHTS = [0.0, 2000.0, 20000.0] # [m], ETC two-layer profile plus a zero-weight placeholder for the 3-layer calibrator
AO_DIAGNOSTICS = ('LGS_STREHL', 'LGS_TURVAR_RES', 'LGS_FWHM_GAIN', 'LGS_SLOPERMS', 'theta0')


def load_calibrator_setup(config_path=CONFIG_PATH):
    with open(config_path) as f:
        cfg = json.load(f)
    meta_path = WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json'
    with open(meta_path) as f:
        meta = json.load(f)
    bundle_path = Path(meta.get('bundle_path', WEIGHTS_FOLDER / 'NFM_calibrator' / 'NFM_calibrator_bundle.pth'))
    return cfg, meta, bundle_path


def load_dataset_templates(path=DATASET_CACHE):
    """ Configs, scaled telemetry rows and sample ids of the calibrator dataset (PSF cubes are not read) """
    data = torch.load(path, weights_only=False, mmap=True)
    configs = [c[0] if isinstance(c, list) else c for c in data['model_configs']]
    telemetry = pd.DataFrame(list(data['telemetry']))  # object array of dicts
    return configs, telemetry, np.asarray(data['sample_ids'])


def build_config(template, row, zenith_deg):
    """ Template dataset config with the ETC atmosphere, LGS flux and the sampled NGS / IRLOS settings (lists, TipTorch layout) """
    tc  = int(row['tc'])
    glf = rc.GLF[tc]
    cfg = deepcopy(template)
    cfg['telescope']['ZenithAngle']        = [[float(zenith_deg)]]
    cfg['atmosphere']['Seeing']            = [float(row['seeing'])]
    cfg['atmosphere']['L0']                = [rc.L0[tc]]
    cfg['atmosphere']['Cn2Weights']        = [[glf, 1.0 - glf, 0.0]]
    cfg['atmosphere']['Cn2Heights']        = [CN2_HEIGHTS]
    cfg['atmosphere']['WindSpeed']         = [[rc.WIND_SPEED[tc]] * 3]
    cfg['atmosphere']['WindDirection']     = [[rc.WIND_DIR[tc]] * 3]
    cfg['sources_science']['Wavelength']   = list(WAVELENGTHS_M)
    cfg['sources_HO']['Height']            = [[90e3]]  # TipTorch stretches the LGS height with the airmass itself
    cfg['sensor_HO']['NumberPhotons']      = [[rc.HO_PHOTONS] * 4]
    cfg['sensor_LO']['NumberPhotons']      = [int(max(float(row['ngs_flux']), 1.0))]
    cfg['sensor_LO']['Gain']               = [rc.IRLOS_gain(row['ngs'])]
    cfg['RTC']['SensorFrameRate_LO']       = [int(rc.IRLOS_frequency(row['ngs']))]
    cfg['DM']['NumberReconstructedLayers'] = [len(CN2_HEIGHTS)]
    return cfg


def build_telemetry(template_physical, stars, zenith_deg):
    """
    Physical-unit telemetry of the Monte Carlo samples: the template rows (DataFrame, one per sample, physical units) with the
    atmospheric, LGS, NGS and pointing features replaced by the sampled conditions (same recipe as `build_ETC_telemetry` in
    `validate_NFM_calibrator.py`, with the sampled tau0 and NGS magnitude instead of TC constants).
    """
    tel = template_physical.copy().reset_index(drop=True)
    tc  = np.asarray(stars['tc'], dtype=int)
    seeing  = np.asarray(stars['seeing'], dtype=float)
    tau0    = np.asarray(stars['tau'], dtype=float) * 1e-3  # [ms] -> [s]
    ngs     = np.asarray(stars['ngs'], dtype=float)
    glf     = np.array([rc.GLF[t] for t in tc])
    wind    = np.array([rc.WIND_SPEED[t] for t in tc])
    wdir    = np.array([rc.WIND_DIR[t] for t in tc])

    seeing_rad = np.deg2rad(seeing / 3600.0)
    r0 = 0.98 * ETC_REFERENCE_WAVELENGTH / seeing_rad
    k  = 2.0 * np.pi / ETC_REFERENCE_WAVELENGTH
    Cn2_integral  = r0 ** (-5.0 / 3.0) / (0.423 * k**2)
    turb_velocity = 0.314 * r0 / tau0  # MASS-DIMM style effective wind from the sampled coherence time
    free_atmosphere_seeing = seeing * (1.0 - glf) ** (3.0 / 5.0)

    replacements = {
        'IRLOS frequency':               rc.IRLOS_frequency(ngs).astype(float),
        'IRLOS gain':                    np.array([rc.IRLOS_gain(m) for m in ngs], dtype=float),
        'NGS mag (from ph.)':            ngs,
        'Tel. altitude':                 90.0 - zenith_deg,
        'Seeing (header)':               seeing,
        'Tau0 (header)':                 tau0,
        'Wind dir (header)':             wdir,
        'Wind speed (header)':           wind,
        'Free Atmosphere Seeing ["]':    free_atmosphere_seeing,
        'MASS Tau0 [s]':                 tau0,
        'MASS-DIMM Turb Velocity [m/s]': turb_velocity,
        'MASS_FRACGL':                   glf,
        'IA_FWHMLINOBS':                 seeing,
        'LGS_TUR_ALT':                   1.0 - glf,
        'LGS photons, [photons/m^2/s]':  np.full_like(seeing, rc.HO_PHOTONS * ETC_LGS_PHOTON_RATE_SCALE),
        'MASS_TURB total':               Cn2_integral,
        'Cn2 fraction below 2000m':      glf,
        'Cn2_alt_binned_2':              np.full_like(seeing, CN2_HEIGHTS[1] / 1e3),
        'Cn2_frac_binned_2':             1.0 - glf,
        'Cn2_alt_binned_3':              np.full_like(seeing, CN2_HEIGHTS[2] / 1e3),
        'Cn2_frac_binned_3':             np.zeros_like(seeing),
    }
    for feature, values in replacements.items():
        if feature in tel:
            tel[feature] = values
    for feature in AO_DIAGNOSTICS:
        if feature in tel:
            tel[feature] = np.nan
    return tel


def flatten_predictions(x_dict):
    """ Calibrator output dict of tensors [B, ...] -> dict of 1D numpy columns """
    cols = {}
    for key, val in x_dict.items():
        val = val.detach().cpu().numpy()
        if val.ndim == 1:
            cols[key] = val
        else:
            for j in range(val.shape[1]):
                cols[f'{key}[{j}]'] = val[:, j]
    return cols


@torch.no_grad()
def simulate_chunk(chunk_configs, chunk_telemetry, calibrator, cfg):
    """ PSFs [B, N_wvl, H, W] and diffraction-limited PSFs [B, N_wvl, H, W] of one batch of configs, plus the predicted parameters """
    chunk_config = MultipleTargetsInDifferentObservations(chunk_configs, device=default_device)
    chunk_config['PathPupil'] = str(PUPIL_PATH)  # as NFMDataset.collate_batch
    chunk_config['telescope']['PupilAngle'] = 0.0
    chunk_config['DM']['NumberReconstructedLayers'] = torch.tensor(chunk_config['atmosphere']['Cn2Heights'].shape[-1], device=default_device)

    PSF_model = PSFModelNFM(
        chunk_config,
        multiple_obs   = True,
        LO_NCPAs       = cfg['model']['LO_NCPAs'],
        chrom_defocus  = cfg['model']['chrom_defocus'],
        use_Moffat     = cfg['model']['use_Moffat'],
        retain_PSDs    = cfg['model']['retain_PSDs'],
        N_spline_nodes = cfg['model']['N_spline_nodes'],
        Z_mode_max     = cfg['model']['Z_mode_max'],
        device         = default_device,
    )
    _ = calibrator.check_compatibility(PSF_model)
    x_dict_pred = calibrator.forward_prepared(chunk_telemetry)
    calibrator._apply_calibration(x_dict_pred, PSF_model)

    PSFs    = PSF_model().detach().cpu()
    PSFs_DL = PSF_model.model.DLPSF().detach().cpu()
    PSF_model.cleanup()
    release_gpu_memory()
    return PSFs, PSFs_DL, flatten_predictions(x_dict_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stars', type=Path, default=STARS_TABLE, help='Observation table shared with run_tiptop_random_carlo.py')
    parser.add_argument('--out-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='Calibrator training config (JSON)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0, help='Seed for the choice of the template observations')
    parser.add_argument('--overwrite', action='store_true', help='Re-simulate samples whose PSF files already exist')
    args = parser.parse_args()

    if not args.stars.exists():
        rc.load_or_generate_stars(args.stars)
    stars = Table.read(args.stars)
    print(f'{len(stars)} observations from {args.stars}')

    cfg, meta, bundle_path = load_calibrator_setup(args.config)
    calibrator = NFMCalibrator(bundle_path, device=default_device)
    configs, telemetry_scaled, sample_ids = load_dataset_templates()
    if list(telemetry_scaled.columns) != list(calibrator.features):
        raise ValueError('Dataset telemetry features do not match the calibrator bundle.')
    print(f'Calibrator {bundle_path.name}: {len(calibrator.features)} features; {len(configs)} template observations')

    rng = np.random.default_rng(args.seed)
    template_idx = rng.integers(0, len(configs), size=len(stars))
    zenith_deg   = np.degrees(np.arccos(1.0 / np.asarray(stars['airmass'], dtype=float)))

    template_physical = pd.DataFrame(
        calibrator.telemetry_scaler.inverse_transform(telemetry_scaled.iloc[template_idx].to_numpy()),
        columns=calibrator.features,
    )
    telemetry_physical = build_telemetry(template_physical, stars, zenith_deg)
    telemetry_vecs     = calibrator.prepare_features(telemetry_physical)
    sample_configs     = [build_config(configs[i], row, z) for i, row, z in zip(template_idx, stars, zenith_deg)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_physical.insert(0, 'template_sample_id', sample_ids[template_idx])
    telemetry_physical.insert(0, 'index', np.asarray(stars['index']))
    telemetry_physical.to_csv(args.out_dir / 'telemetry.csv', index=False)
    for wave in rc.WAVELENGTHS:
        (args.out_dir / f'{wave}nm').mkdir(exist_ok=True)

    names = [rc.sample_name(row, row['index']) for row in stars]
    todo  = [i for i, n in enumerate(names) if args.overwrite or not all((args.out_dir / f'{w}nm' / f'{n}.fits').exists() for w in rc.WAVELENGTHS)]
    print(f'{len(todo)} samples to simulate ({len(names) - len(todo)} already present) -> {args.out_dir}')

    pupil = load_pupil(PUPIL_PATH)
    predictions = []
    for start in range(0, len(todo), args.batch_size):
        ids = todo[start:start + args.batch_size]
        PSFs, PSFs_DL, pred = simulate_chunk([sample_configs[i] for i in ids], telemetry_vecs[ids], calibrator, cfg)
        if PSFs_DL.shape[0] == 1 and PSFs.shape[0] > 1:
            PSFs_DL = PSFs_DL.expand(PSFs.shape[0], -1, -1, -1)
        PSFs, PSFs_DL = crop_center(PSFs.numpy(), PSF_SIZE), crop_center(PSFs_DL.numpy(), PSF_SIZE)
        pred = pd.DataFrame(pred)
        pred.insert(0, 'index', np.asarray(stars['index'])[ids])
        pred.insert(1, 'template_sample_id', sample_ids[template_idx[ids]])
        predictions.append(pred)

        for b, i in enumerate(ids):
            row = stars[i]
            header_items = {f'sample {k}': row[k] for k in stars.colnames}
            header_items['sample template_sample_id'] = int(sample_ids[template_idx[i]])
            header_items.update({f'telescope ZenithAngle': zenith_deg[i]})
            for wi, wave in enumerate(rc.WAVELENGTHS):
                psf = normalize_psf(PSFs[b, wi])
                sr, fwhm, ee = tiptop_metrics(psf, wave, pupil, PIXEL_SCALE_MAS)
                write_psf_file(args.out_dir / f'{wave}nm' / f'{names[i]}.fits', psf, normalize_psf(PSFs_DL[b, wi]), wave, PIXEL_SCALE_MAS,
                               {'sr': sr, 'fwhm': fwhm, 'ee': ee}, header_items)
        print(f'  simulated {min(start + args.batch_size, len(todo))}/{len(todo)}  (last SR@900nm {sr:.3f}, FWHM {fwhm:.1f} mas)')

    if predictions:
        pred_table = Table.from_pandas(pd.concat(predictions, ignore_index=True))
        pred_path  = args.out_dir / 'predicted_params.fits'
        if pred_path.exists() and not args.overwrite:  # merge with the parameters of previously simulated samples
            old = Table.read(pred_path)
            old = old[~np.isin(old['index'], pred_table['index'])]
            from astropy.table import vstack
            pred_table = vstack([old, pred_table]) if len(old) else pred_table
        pred_table.sort('index')
        pred_table.write(pred_path, overwrite=True)
        print(f'Predicted model parameters -> {pred_path}')
