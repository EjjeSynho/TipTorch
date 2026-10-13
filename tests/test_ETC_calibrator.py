"""
Checks of `machine_learning/calibrators/ETC_calibrator.py`: TC constants and regime switches agree with Enrico's Monte Carlo
script, ETC tables are consistent, and both predictors render finite PSFs through `ETCPSFRenderer`.

Requires the ETC bundle (`train_ETC_calibrator.py`), the astro-tiptop checkouts (`tools/tiptop_env.py`) and a CUDA device.
Run from the repository root: python tests/test_ETC_calibrator.py
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
    assert (E.TC_WIND_SPEED, E.TC_WIND_DIR, E.TC_L0, E.TC_GLF, E.ETC_HO_PHOTONS) == (rc.WIND_SPEED, rc.WIND_DIR, rc.L0, rc.GLF, rc.HO_PHOTONS)
    seeing, tau = np.array([0.5, 0.65, 0.75, 0.9, 1.3, 0.5]), np.array([6.0, 5.0, 4.2, 3.5, 2.0, 3.0])
    assert (E.turbulence_categories(seeing, tau) == rc.turbulence_categories(seeing, tau)).all()
    mags = np.array([9.0, 10.5, 12.0, 16.0, 16.5])
    assert (E.IRLOS_frequency(mags) == rc.IRLOS_frequency(mags)).all()
    assert (E.IRLOS_gain(mags) == [rc.IRLOS_gain(m) for m in mags]).all()
    assert np.allclose(E.NGS_mag_to_photons(mags, 500, 1.2), rc.NGS_mag_to_photons(mags, 500, 1.2))


def test_etc_table():
    t = E.etc_table([0.5, 1.2], [0.006, 0.002], [1.0, 1.6], [9.0, 13.0])
    assert list(t['tc']) == [10, 100] and t['zenith_deg'].iloc[0] == 0.0 and (t['ngs_flux'] > 0).all()
    assert E.etc_inputs(t).shape == (2, len(E.ETC_INPUTS))


def test_predictors():
    with open(REPO_ROOT / 'machine_learning' / 'calibrators' / 'configs' / 'ETC_calibrator_config.json') as f:
        cfg = json.load(f)
    with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
        etc_cal = E.ETCCalibrator(json.load(f)['bundle_path'])
    etc = E.etc_table([0.5, 0.9, 1.3], [0.006, 0.004, 0.002], [1.1, 1.4, 1.8], [9.5, 12.0, 15.0])
    rng = np.random.default_rng(0)
    renderer = etc_cal.renderer([500.0, 700.0, 900.0], batch_size=3)
    for name in E.PREDICTORS:
        pred = etc_cal.predictor(name)
        x, info = pred.predict(etc, rng)
        assert x['r0'].shape == (3,) and x['LO_coefs'].shape == (3, 8) and all(torch.isfinite(v).all() for v in x.values())
        PSFs, table = renderer.render(pred, etc, rng, n_draws=2)
        n = 6 if pred.stochastic else 3
        assert PSFs.shape[:2] == (n, 3) and torch.isfinite(PSFs).all() and (PSFs.amax(dim=(-2, -1)) > 0).all() and len(table) == n
    nearest = etc_cal.predictor('donor-nearest')
    assert (nearest.donors(etc, rng) == nearest.donors(etc, rng)).all()
    print('ETC calibrator checks passed')


if __name__ == '__main__':
    test_constants_match_enrico()
    test_etc_table()
    test_predictors()
