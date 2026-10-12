"""
PSF parameter tables of the Monte Carlo PSFs (TIPTOP or calibrated TipTorch), the simulated counterpart of the on-sky tables
`data/Enrico_ETC/onsky_PSFs_params/All_PSF_{wave}.fits`.

For every row of the shared observation table (`stars.fits`) and every wavelength the PSF file `<folder>/{wave}nm/<name>.fits`
is measured with `ETC_psf_metrics`: `fwhm` (Enrico's oversampled radial profile, [mas]), `strehl` (MAOPPY PSFAO fit at the PSF
wavelength) and the TIPTOP-style header metrics `strehl_tt` (SR), `fwhm_tt` and `ensquared_tt` (EE within 50 mas), which both
generators write with the same code. The result is `<folder>/psf_params_{wave}.fits` with the observation columns
(`index, airmass, seeing, tau, ngs, tc, ngs_flux`) and the parameters; missing PSFs give NaN rows.

Run from the repository root in the `TipTorch` conda env (~2 min per wavelength with 16 processes, dominated by the PSFAO fits):

    python development/ETC_compute_PSF_params.py --source tiptop
    python development/ETC_compute_PSF_params.py --source tiptorch --wave 700 900 --processes 8
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

# One BLAS thread per worker: the PSFAO fits are small and oversubscription makes the pool several times slower
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiptop_env  # noqa: F401  (MAOPPY on sys.path; the workers only need numpy / scipy / maoppy)
from ETC_psf_metrics import read_psf_file, normalize_psf, compute_fwhm, compute_strehl_psfao

REPO_ROOT  = Path(__file__).resolve().parents[1]
ETC_FOLDER = REPO_ROOT / 'data' / 'Enrico_ETC'
SOURCES    = {'tiptop': ETC_FOLDER / 'generated_random_PSFs', 'tiptorch': ETC_FOLDER / 'tiptorch_random_PSFs'}
WAVELENGTHS = [500, 600, 700, 800, 900]
PARAM_COLUMNS = ['fwhm', 'strehl', 'strehl_tt', 'fwhm_tt', 'ensquared_tt']


def sample_name(row):
    return f'muse_z{row["airmass"]:0.2f}_s{row["seeing"]:0.2f}_n{row["ngs"]:0.2f}_{int(row["index"]):04d}'


def measure_psf_file(args):
    """ (path, do_psfao) -> dict of PSF parameters; NaNs when the file is missing """
    path, do_psfao = args
    out = {k: np.nan for k in PARAM_COLUMNS}
    if not Path(path).exists():
        return out
    try:
        f = read_psf_file(path)
    except Exception as e:
        print(f'{path}: {type(e).__name__}: {e}')
        return out
    psf = normalize_psf(f['psf'])
    out['fwhm']         = compute_fwhm(psf, pixel_scale_mas=f['pix_mas'])
    out['strehl_tt']    = f['sr_hdr']
    out['fwhm_tt']      = f['fwhm_hdr']
    out['ensquared_tt'] = f['ee_hdr']
    if do_psfao:
        try:
            out['strehl'], _ = compute_strehl_psfao(psf, f['wave_nm'])
        except Exception as e:
            print(f'{path}: PSFAO fit failed: {type(e).__name__}: {e}')
    return out


def compute_params_table(folder, stars, wave, processes=16, do_psfao=True):
    """ Parameter table of the PSFs of `stars` in `<folder>/{wave}nm/`; returns an astropy Table (one row per observation) """
    paths = [(str(Path(folder) / f'{wave}nm' / f'{sample_name(row)}.fits'), do_psfao) for row in stars]
    if processes > 1:
        with mp.get_context('spawn').Pool(processes) as pool:
            results = pool.map(measure_psf_file, paths, chunksize=4)
    else:
        results = [measure_psf_file(p) for p in paths]

    table = Table()
    for col in ('index', 'airmass', 'seeing', 'tau', 'ngs', 'tc', 'ngs_flux'):
        if col in stars.colnames:
            table[col] = np.asarray(stars[col])
    table['name'] = [sample_name(row) for row in stars]
    for col in PARAM_COLUMNS:
        table[col] = np.array([r[col] for r in results], dtype=float)
    table['wave'] = np.full(len(stars), wave, dtype=int)
    table.meta['PSF_FOLDER'] = str(Path(folder) / f'{wave}nm')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=list(SOURCES), required=True)
    parser.add_argument('--folder', type=Path, default=None, help='PSF folder (default: the standard folder of --source)')
    parser.add_argument('--stars', type=Path, default=SOURCES['tiptop'] / 'stars.fits')
    parser.add_argument('--wave', type=int, nargs='*', default=WAVELENGTHS, choices=WAVELENGTHS)
    parser.add_argument('--processes', type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument('--no-psfao', action='store_true', help='Skip the (slow) PSFAO Strehl fit')
    args = parser.parse_args()

    folder = args.folder or SOURCES[args.source]
    stars  = Table.read(args.stars)
    for wave in args.wave:
        table = compute_params_table(folder, stars, wave, args.processes, not args.no_psfao)
        n_ok  = int(np.isfinite(table['fwhm']).sum())
        out   = folder / f'psf_params_{wave}.fits'
        table.write(out, overwrite=True)
        print(f'{args.source} {wave} nm: {n_ok}/{len(table)} PSFs measured -> {out}')
        for col in PARAM_COLUMNS:
            v = table[col][np.isfinite(table[col])]
            if len(v):
                print(f'   {col:13s} median {np.median(v):8.3f}   IQR [{np.percentile(v, 25):8.3f}, {np.percentile(v, 75):8.3f}]')
