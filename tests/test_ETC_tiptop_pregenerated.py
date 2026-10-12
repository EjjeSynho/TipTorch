"""
Numerical test of the adapted `development/run_tiptop_random_carlo.py` against Enrico's pregenerated TIPTOP PSFs in
`data/Enrico_ETC/pregenerated_random_PSFs/{wave}nm/` (five observations x five wavelengths, TIPTOP 1.4.3).

1. Config reconstruction: the observation (airmass from `ZenithAngle`, seeing, NGS magnitude from the file name, TC from `L0`,
   IRLOS flux) is read back from every pregenerated `.ini` and passed through `build_tiptop_config`; every key must match
   (`PathPupil` excepted, `NumberPhotons` of the LO sensor within 3 % because the magnitude is rounded in the file name).
2. PSF reproduction: every pregenerated `.ini` is re-simulated with the astro-tiptop checkouts on `sys.path` (pupil path
   localized) and compared with the stored PSF cube (relative L2 < 1e-6) and header metrics (SR, FWHM, EE50).
3. Metrics: `ETC_psf_metrics.tiptop_metrics` reproduces the stored header SR / FWHM / EE50 from the PSF arrays.

Needs the astro-tiptop checkouts pinned at TIPTOP v1.4.3, P3 v1.6.0, MASTSEL v1.5.0 (see `development/tiptop_env.py`), the
local VLT pupil in `data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits` and ~1 min on a GPU.
Run from the repository root in the `TipTorch` conda env:

    python tests/test_ETC_tiptop_pregenerated.py [--scratch DIR] [--skip-simulations]
"""

import re
import sys
import ast
import argparse
import tempfile
import configparser
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'development'))

import tiptop_env  # noqa: F401
from tiptop.tiptop import overallSimulation
import run_tiptop_random_carlo as rc
from ETC_psf_metrics import load_pupil, tiptop_metrics, read_psf_file, normalize_psf

PREGENERATED = REPO_ROOT / 'data' / 'Enrico_ETC' / 'pregenerated_random_PSFs'
L0_TO_TC = {v: k for k, v in rc.L0.items()}


def read_ini(path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    return cfg


def observation_from_ini(path):
    """ Reconstruct the observation row that produced a pregenerated `.ini` """
    cfg = read_ini(path)
    zenith  = float(cfg['telescope']['ZenithAngle'])
    airmass = 1.0 / np.cos(np.deg2rad(zenith))
    seeing  = float(cfg['atmosphere']['Seeing'])
    ngs     = float(re.search(r'_n(\d+\.\d+)', Path(path).stem).group(1))
    tc      = L0_TO_TC[float(cfg['atmosphere']['L0'])]
    flux    = ast.literal_eval(cfg['sensor_LO']['NumberPhotons'].split('#')[0].strip())[0]
    return {'airmass': airmass, 'seeing': seeing, 'ngs': ngs, 'tc': tc, 'ngs_flux': float(flux), 'tau': np.nan}


def _values_equal(a, b, rtol=1e-9):
    a, b = a.split('#')[0].strip(), b.split('#')[0].strip()
    if a == b:
        return True
    try:
        va, vb = ast.literal_eval(a), ast.literal_eval(b)
        return np.allclose(np.asarray(va, dtype=float), np.asarray(vb, dtype=float), rtol=rtol, atol=0)
    except Exception:
        return False


def check_config_reconstruction(ini_path, wave):
    row  = observation_from_ini(ini_path)
    ref  = read_ini(ini_path)
    mine = rc.build_tiptop_config(rc.read_base_config(), row, wave)
    problems = []
    for section in set(ref.sections()) | set(mine.sections()):
        if section not in ref or section not in mine:
            problems.append(f'section {section} missing'); continue
        for key in set(ref[section]) | set(mine[section]):
            if key not in ref[section] or key not in mine[section]:
                problems.append(f'{section}.{key} missing'); continue
            if key == 'PathPupil':
                continue
            if section == 'sensor_LO' and key == 'NumberPhotons':
                n_ref = ast.literal_eval(ref[section][key])[0]
                n_me  = ast.literal_eval(mine[section][key])[0]
                if abs(n_ref - n_me) > 0.03 * max(n_ref, 1):
                    problems.append(f'{section}.{key}: {n_ref} vs {n_me}')
                continue
            if not _values_equal(ref[section][key], mine[section][key]):
                problems.append(f'{section}.{key}: {ref[section][key]!r} vs {mine[section][key]!r}')
    return problems


def check_psf_reproduction(ini_path, scratch, pupil):
    """ Re-simulate one pregenerated config; returns (rel. L2, header diffs dict, metrics diffs dict) """
    cfg = read_ini(ini_path)
    cfg['telescope']['PathPupil'] = f"'{rc.PUPIL_PATH}'"
    name = Path(ini_path).stem
    with open(scratch / f'{name}.ini', 'w') as f:
        cfg.write(f)
    overallSimulation(str(scratch), name, str(scratch), name)

    ref = read_psf_file(Path(ini_path).with_suffix('.fits'))
    new = read_psf_file(scratch / f'{name}.fits')
    rel_L2 = np.linalg.norm(new['psf'] - ref['psf']) / np.linalg.norm(ref['psf'])
    hdr_diff = {k: abs(new[k] - ref[k]) for k in ('sr_hdr', 'fwhm_hdr', 'ee_hdr')}

    sr, fwhm, ee = tiptop_metrics(normalize_psf(ref['psf']), ref['wave_nm'], pupil, ref['pix_mas'])
    metric_diff = {'sr': abs(sr - ref['sr_hdr']), 'fwhm': abs(fwhm - ref['fwhm_hdr']), 'ee': abs(ee - ref['ee_hdr'])}
    return rel_L2, hdr_diff, metric_diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, default=None, help='Working folder for the re-simulated PSFs (default: temp dir)')
    parser.add_argument('--skip-simulations', action='store_true', help='Only check the config reconstruction and the metrics')
    args = parser.parse_args()
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix='ETC_pregenerated_'))
    scratch.mkdir(parents=True, exist_ok=True)

    import tiptop, p3, mastsel
    print(f'TIPTOP {tiptop.__version__}, MASTSEL {mastsel.__version__}, P3 at {Path(p3.__file__).parents[1]}')

    pupil = load_pupil()
    failures = 0
    for wave in rc.WAVELENGTHS:
        for ini_path in sorted((PREGENERATED / f'{wave}nm').glob('*.ini')):
            problems = check_config_reconstruction(ini_path, wave)
            status = 'OK ' if not problems else 'FAIL'
            failures += bool(problems)
            print(f'[config {status}] {wave} nm {ini_path.stem}' + (f': {problems}' if problems else ''))

            if args.skip_simulations:
                ref = read_psf_file(ini_path.with_suffix('.fits'))
                sr, fwhm, ee = tiptop_metrics(normalize_psf(ref['psf']), ref['wave_nm'], pupil, ref['pix_mas'])
                d = {'sr': abs(sr - ref['sr_hdr']), 'fwhm': abs(fwhm - ref['fwhm_hdr']), 'ee': abs(ee - ref['ee_hdr'])}
                ok = d['sr'] < 1e-4 and d['fwhm'] < 1e-2 and d['ee'] < 1e-4
                failures += not ok
                print(f"[metrics {'OK ' if ok else 'FAIL'}] dSR {d['sr']:.1e} dFWHM {d['fwhm']:.1e} dEE {d['ee']:.1e}")
                continue

            rel_L2, hdr_diff, metric_diff = check_psf_reproduction(ini_path, scratch, pupil)
            ok_psf = rel_L2 < 1e-6 and hdr_diff['sr_hdr'] < 1e-4 and hdr_diff['fwhm_hdr'] < 1e-2 and hdr_diff['ee_hdr'] < 1e-4
            ok_met = metric_diff['sr'] < 1e-4 and metric_diff['fwhm'] < 1e-2 and metric_diff['ee'] < 1e-4
            failures += (not ok_psf) + (not ok_met)
            print(f"[PSF {'OK ' if ok_psf else 'FAIL'}] rel L2 {rel_L2:.1e}, header dSR {hdr_diff['sr_hdr']:.1e} dFWHM {hdr_diff['fwhm_hdr']:.1e} dEE {hdr_diff['ee_hdr']:.1e}"
                  f" | [metrics {'OK ' if ok_met else 'FAIL'}] dSR {metric_diff['sr']:.1e} dFWHM {metric_diff['fwhm']:.1e} dEE {metric_diff['ee']:.1e}")

    print(f'\n{"ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"}')
    sys.exit(int(failures > 0))
