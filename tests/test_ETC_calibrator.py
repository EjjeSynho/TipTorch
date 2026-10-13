"""
Checks of the ETC calibrator module (`machine_learning/calibrators/ETC_calibrator.py`): the TC constants and regime switches
agree with Enrico's Monte Carlo script, the ETC tables are consistent, the telemetry builders produce finite standardized vectors
of the right shape, and the donor / calibrator predictors render PSFs through `ETCPSFRenderer`.

Requires the ETC bundle (`train_ETC_calibrator.py`) with its PSF-R bundle, the dataset cache, the astro-tiptop checkouts
(`development/tiptop_env.py`) and a CUDA device (~30 s). Run from the repository root: python tests/test_ETC_calibrator.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    sys.path.insert(0, str(p))

from tiptorch._config import WEIGHTS_FOLDER
from calibrators import ETC_calibrator as E


def test_constants_match_enrico():
    import run_tiptop_random_carlo as rc
    assert E.TC_WIND_SPEED == rc.WIND_SPEED and E.TC_WIND_DIR == rc.WIND_DIR and E.TC_L0 == rc.L0 and E.TC_GLF == rc.GLF
    assert E.ETC_HO_PHOTONS == rc.HO_PHOTONS
    seeing, tau = np.array([0.5, 0.65, 0.75, 0.9, 1.3, 0.5]), np.array([6.0, 5.0, 4.2, 3.5, 2.0, 3.0])
    assert (E.turbulence_categories(seeing, tau) == rc.turbulence_categories(seeing, tau)).all()
    mags = np.array([9.0, 10.5, 12.0, 16.0, 16.5])
    assert (E.IRLOS_frequency(mags) == rc.IRLOS_frequency(mags)).all()
    assert (E.IRLOS_gain(mags) == np.array([rc.IRLOS_gain(m) for m in mags])).all()
    assert np.allclose(E.NGS_mag_to_photons(mags, 500, 1.2), rc.NGS_mag_to_photons(mags, 500, 1.2))


def test_etc_table():
    t = E.etc_table([0.5, 1.2], [0.006, 0.002], [1.0, 1.6], [9.0, 13.0])
    assert list(t['tc']) == [10, 100] and t['zenith_deg'].iloc[0] == 0.0 and (t['ngs_flux'] > 0).all()
    assert list(E.etc_direct_telemetry(t).columns) == E.ETC_DIRECT_FEATURES
    assert list(E.tc_profile_telemetry(t).columns) == E.ETC_PROFILE_FEATURES


def load_bundle():
    with open(REPO_ROOT / 'machine_learning' / 'ETC_calibrator_config.json') as f:
        cfg = json.load(f)
    with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
        return E.ETCCalibrator(json.load(f)['bundle_path'])


def test_builders_and_predictors():
    etc_cal = load_bundle()
    etc = E.etc_table([0.5, 0.9, 1.3], [0.006, 0.004, 0.002], [1.1, 1.4, 1.8], [9.5, 12.0, 15.0])
    rng = np.random.default_rng(0)
    for fill, profile in (('impute', 'TC'), ('donor', 'TC'), ('donor', 'donor'), ('gaussian', 'TC')):
        Z, info = etc_cal.builder(fill=fill, profile=profile).build(etc, rng)
        assert Z.shape == (3, len(etc_cal.pre.features)) and np.isfinite(Z).all()
        if fill == 'donor':
            assert (info['donor'] >= 0).all() and info['heights'].shape == (3, 3)
    renderer = etc_cal.renderer([500.0, 700.0, 900.0], batch_size=3)
    for kind in ('impute', 'donor-params', 'donor') + (('direct-net',) if etc_cal.state.get('direct_net') else ()):
        pred = etc_cal.predictor(kind)
        x, info = pred.predict(etc, rng)
        assert x['r0'].shape[0] == 3 and x['LO_coefs'].shape == (3, 8) and torch.isfinite(x['r0']).all()
        renderer.heights_policy = E.PREDICTOR_PRESETS[kind][2]
        PSFs, table = renderer.render(pred, etc, rng, n_draws=1)
        assert PSFs.shape[:2] == (3, 3) and torch.isfinite(PSFs).all() and (PSFs.amax(dim=(-2, -1)) > 0).all()
        assert len(table) == 3 and 'r0' in table
    print('ETC calibrator checks passed')


if __name__ == '__main__':
    test_constants_match_enrico()
    test_etc_table()
    test_builders_and_predictors()
