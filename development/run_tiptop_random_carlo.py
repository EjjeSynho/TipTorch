# Original code developed by Enrico Congiu (ESO). Adapted for the TipTorch repository: comments translated from Italian,
# paths made repository-relative, random sample table shared with the TipTorch Monte Carlo, command-line options added.
"""
Monte Carlo generation of MUSE NFM PSFs with vanilla TIPTOP for the ETC comparison.

A table of random observations is drawn per turbulence category (TC) with `generate_random_observations` (seeing, coherence
time, NGS J magnitude, airmass) and stored in `data/Enrico_ETC/generated_random_PSFs/stars.fits`; the same table is used by
`run_tiptorch_random_carlo.py`, so the TIPTOP and TipTorch samples are paired. For every row and every science wavelength a
TIPTOP `.ini` is derived from `data/Enrico_ETC/muse_ltao_Guido.ini` (`build_tiptop_config`) and simulated with
`tiptop.overallSimulation`; the PSFs land in `data/Enrico_ETC/generated_random_PSFs/{wave}nm/`.

Requirements: the astro-tiptop checkouts on `sys.path` through `tiptop_env.py` (TIPTOP v1.4.3, P3 v1.6.0, MASTSEL v1.5.0 reproduce
the pregenerated PSFs exactly) and a GPU for cupy (optional). One simulation takes about one second on a GPU.

Run from the repository root in the `TipTorch` conda env:

    python development/run_tiptop_random_carlo.py                   # all five wavelengths, per-TC counts of the on-sky table
    python development/run_tiptop_random_carlo.py --wave 700 900    # subset of wavelengths
    python development/run_tiptop_random_carlo.py --sampling onsky  # replay the on-sky (seeing, tau0, NGS mag, airmass) rows
"""

import os
import sys
import time
import argparse
import configparser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.table import Table, vstack

for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]): # this folder and the repository root (`tools`)
    sys.path.insert(0, str(_p))
from tools import tiptop_env  # noqa: F401  (puts the astro-tiptop checkouts on sys.path before importing tiptop)
from tiptop.tiptop import overallSimulation

REPO_ROOT   = Path(__file__).resolve().parents[1]
ETC_FOLDER  = REPO_ROOT / 'data' / 'Enrico_ETC'
BASE_INI    = ETC_FOLDER / 'muse_ltao_Guido.ini'
PUPIL_PATH  = REPO_ROOT / 'data' / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'
ONSKY_TABLE = ETC_FOLDER / 'onsky_PSFs_params' / 'All_PSF_{wave}.fits'
OUTPUT_DIR  = ETC_FOLDER / 'generated_random_PSFs'
STARS_TABLE = OUTPUT_DIR / 'stars.fits'

WAVELENGTHS = [500, 600, 700, 800, 900] # [nm], the wavelengths of the on-sky parameter tables
TC_LIST     = [10, 20, 30, 50, 100]


def NGS_mag_to_photons(Jmag, frequency, airmass, k_J=0.09, delta_mag=0.4):
    """
    Convert NGS J band magnitude to the number of photons required for the
    ['sensor_LO']['NumberPhotons'] field in the TipTop configuration file.
    This function calculates the number of photons based on the J band magnitude
    and adjusts it for the TipTop system using a specified magnitude offset.
    The calculation is based on a formula provided by Arseniy.

    The default value for delta_mag has been computed comparing the J band
    magnitude of the NGS of a sample of Standard star and their magnitude recovered
    from the IRLOS counts in several observations. The IRLOS magnitudes have been
    corrected for Atmospheric extinction.

    Parameters:
    -----------
    Jmag : float
        The J band magnitude of the natural guide star (NGS).
    frequency : float
        IRLOS frequency in Hz.
    airmass : float
        airmass of the target
    k_J : float, optional
        The extinction coefficient for J band in Paranal. Default is 0.09
        (from 2008eic..work..439M).
    delta_mag : float, optional
        The magnitude offset to convert J band magnitude to IRLOS magnitude.
        Default is 0.4.
    Returns:
    --------
    float
        The number of photons as seen by the IRLOS per subaperture.
    """

    IRLOS_mag = Jmag + delta_mag + k_J * airmass # applying extinction and converting to IRLOS mag

    # Convert magnitude to flux using Arseniy formula
    J_zero_point = 1.9e12
    const = J_zero_point/368
    N_ph = const * 10**(-0.4 * IRLOS_mag)

    # N_ph is the total number of photons. We need to transform this
    # in what IRLOS would see

    N_ph_IRLOS = N_ph / (4*frequency) # 4 is the number of subapertures

    return N_ph_IRLOS


def turbulence_categories(seeing, tau):
    """ ESO turbulence category (percentile) from seeing ["] and coherence time [ms] """
    seeing, tau = np.atleast_1d(seeing), np.atleast_1d(tau)
    tcats = np.zeros(len(seeing), dtype=int)
    mask10 = (seeing <= 0.6) * (tau >= 5.2)
    mask20 = (seeing <= 0.7) * (tau >= 4.4) * ~mask10
    mask30 = (seeing <= 0.8) * (tau >= 4.1) * ~(mask10 + mask20)
    mask50 = (seeing <= 1.0) * (tau >= 3.2) * ~(mask10 + mask20 + mask30)
    mask100 = (seeing > 1) + (tau < 3.2)

    tcats[mask10] = 10
    tcats[mask20] = 20
    tcats[mask30] = 30
    tcats[mask50] = 50
    tcats[mask100] = 100

    return tcats


def IRLOS_frequency(ngs_mag):
    """ IRLOS loop frequency [Hz] as a function of the NGS J magnitude (Enrico's regime switch) """
    return np.where(np.asarray(ngs_mag) <= 16, 500, 200)


def IRLOS_gain(ngs_mag):
    """ IRLOS detector gain as a function of the NGS J magnitude (Enrico's regime switch) """
    return 1 if ngs_mag <= 10.5 else 68 if ngs_mag <= 16 else 100


def generate_random_observations(n, tc, ngs0=[9, 16], seeing_min=0.4, tau_max=10.0, rng=None):
    """
    Generate n random observations for a given turbulence category.

    Seeing is sampled from [seeing_min, tc_seeing_max].
    Coherence time (tau) is sampled from [tc_tau_min, tau_max].
    Points that would fall into a stricter TC are rejected via oversampling.
    The returned table always contains exactly n rows.
    """
    rng = np.random.default_rng() if rng is None else rng

    # TC boundaries: (max_seeing, min_tau)
    tc_bounds = {
        10:  (0.6, 5.2),
        20:  (0.7, 4.4),
        30:  (0.8, 4.1),
        50:  (1.0, 3.2),
        100: (1.4, 1.0),
    }

    # The boundary of the next-stricter TC to exclude.
    # A point is rejected if  seeing <= exc_seeing  AND  tau >= exc_tau.
    # Since stricter categories are nested, excluding the boundary of the
    # immediately-superior TC is sufficient to remove all better-condition points.
    tc_exclude = {
        10:  None,            # most stringent — nothing to exclude
        20:  (0.6, 5.2),     # exclude TC10
        30:  (0.7, 4.4),     # exclude TC20 (TC10 ⊂ TC20 region)
        50:  (0.8, 4.1),     # exclude TC30 (and TC20, TC10)
        100: (1.0, 3.2),     # exclude TC50 (and all stricter)
    }

    seeing_max, tau_min = tc_bounds[tc]
    exclude = tc_exclude[tc]

    collected_seeing = []
    collected_tau = []
    oversample = 10

    while len(collected_seeing) < n:
        batch_size = (n - len(collected_seeing)) * oversample

        s = rng.uniform(seeing_min, seeing_max, batch_size)
        t = rng.uniform(tau_min, tau_max, batch_size)

        if exclude is not None:
            exc_seeing, exc_tau = exclude
            valid = ~((s <= exc_seeing) & (t >= exc_tau))
            s = s[valid]
            t = t[valid]

        need = n - len(collected_seeing)
        collected_seeing.extend(s[:need].tolist())
        collected_tau.extend(t[:need].tolist())

    seeing = np.array(collected_seeing[:n])
    tau = np.array(collected_tau[:n])

    ngs = rng.uniform(ngs0[0], ngs0[1], n)
    airmass = rng.uniform(1, 2, n)

    return make_observations_table(seeing, tau, ngs, airmass)


def make_observations_table(seeing, tau, ngs, airmass):
    """ Observation table with the derived TC and IRLOS flux, as consumed by `build_tiptop_config` """
    seeing, tau, ngs, airmass = (np.asarray(v, dtype=float) for v in (seeing, tau, ngs, airmass))
    tc_arr = turbulence_categories(seeing, tau)
    frequency = IRLOS_frequency(ngs)
    ngs_flux = NGS_mag_to_photons(ngs, frequency, airmass)

    return Table([seeing, tau, ngs, airmass, tc_arr, ngs_flux],
                 names=['seeing', 'tau', 'ngs', 'airmass', 'tc', 'ngs_flux'])


def onsky_observations(n, tc, rng, wave=700):
    """ Replay the on-sky conditions of TC `tc` (n rows drawn from `All_PSF_{wave}.fits`, without replacement when possible) """
    onsky = Table.read(str(ONSKY_TABLE).format(wave=wave))
    onsky = onsky[(onsky['tc'] == tc) & np.isfinite(onsky['tau0'])]
    picks = rng.choice(len(onsky), size=n, replace=n > len(onsky))
    rows  = onsky[picks]
    return make_observations_table(rows['seeing'], rows['tau0'], rows['ngs_mag'], rows['airmass'])


def onsky_counts_per_TC(wave=700):
    """ Number of on-sky PSFs per TC, used as the default sample size of the Monte Carlo """
    onsky = Table.read(str(ONSKY_TABLE).format(wave=wave))
    tcs, counts = np.unique(np.asarray(onsky['tc']), return_counts=True)
    return {int(tc): int(c) for tc, c in zip(tcs, counts)}


def generate_stars_table(counts, sampling='random', ngs_range=(9.0, 16.0), seed=0):
    """ Random observation table over all TCs; `counts` maps TC -> number of rows """
    rng = np.random.default_rng(seed)
    stars = []
    for tc in TC_LIST:
        if counts.get(tc, 0) == 0:
            continue
        if sampling == 'random':
            stars.append(generate_random_observations(counts[tc], tc=tc, ngs0=list(ngs_range), rng=rng))
        elif sampling == 'onsky':
            stars.append(onsky_observations(counts[tc], tc=tc, rng=rng))
        else:
            raise ValueError(f"Unknown sampling '{sampling}'")
    stars = vstack(stars)
    stars.add_column(np.arange(len(stars)), name='index', index=0)
    stars.meta.update({'sampling': sampling, 'seed': seed, 'ngs_min': ngs_range[0], 'ngs_max': ngs_range[1]})
    return stars


# ---------------------------------------------------------------------------------------------------------------------------
# SPIE tuning of TIPTOP per wavelength (Enrico): wavelength-dependent extra jitter FWHM [mas] as a linear function of airmass,
# quadratically combined with the base jitter of 25 mas already included in the fit.
def get_extra_jitter_700(tc, airmass, orig=25):

    fit_info = {10: [20.426, 32.664],
                20: [26.564, 28.254],
                30: [21.542, 32.005],
                50: [10.211, 46.572],
                100: [32.607, 18.274]}

    simul = fit_info[tc][0]*airmass + fit_info[tc][1]

    return np.sqrt(simul**2 - orig**2)

def get_extra_jitter_500(tc, airmass, orig=25):

    fit_info = {10: [40.855,15],
                20: [60.032,-10],
                30: [60,-10],
                50: [17.579,62.779-20],
                100: [28.879,20],
                }

    simul = fit_info[tc][0]*airmass + fit_info[tc][1]

    out = np.sqrt(simul**2 - orig**2)
    if np.isnan(out):
        print(simul, orig)
        out = 0

    return out

def get_extra_jitter_600(tc, airmass, orig=25):

    fit_info = {10: [38.605,19.523-5],
                20: [59.498,-1.383-5],
                30: [46.047,17.002-10],
                50: [18.853,46.776-10],
                100: [67.210,-3.548-20],
                }

    simul = fit_info[tc][0]*airmass + fit_info[tc][1]

    out = np.sqrt(simul**2 - orig**2)
    if np.isnan(out):
        print(simul, orig)
        out = 0

    return out

def get_extra_jitter_800(tc, airmass, orig=25):

    fit_info = {10: [20.801,33.098],
                20: [13.125,42.508],
                30: [9.980,48.258],
                50: [4.435,54.736],
                100: [2.845,52.131],
            }

    simul = fit_info[tc][0]*airmass + fit_info[tc][1]

    out = np.sqrt(simul**2 - orig**2)
    if np.isnan(out):
        print(simul, orig)
        out = 0

    return out


def get_extra_jitter_900(tc, airmass, orig=25):

    fit_info = {10: [20.009,33.688],
                20: [17.862,36.748],
                30: [11.387,46.547],
                50: [1.152,56.389],
                100: [17.207,38.960],
            }

    simul = fit_info[tc][0]*airmass + fit_info[tc][1]

    out = np.sqrt(simul**2 - orig**2)
    if np.isnan(out):
        print(simul, orig)
        out = 0

    return out


# ASM wind speed values
WIND_SPEED = {
    10: 10.4,
    20: 10.5,
    30: 9.9,
    50: 10.1,
    100: 14.5,
}

# ASM wind direction values
WIND_DIR = {
    10: 12.5,
    20: 28.8,
    30: 40.2,
    50: 37.2,
    100: 37.0,
}

L0 = {
    10: 12.48,
    20: 17.1,
    30: 22.1,
    50: 23.5,
    100: 25.5
}

GLF = {
    10: 0.971,
    20: 0.973,
    30: 0.980,
    50: 0.972,
    100: 0.977
}

# extraerr = {
#     500: 140,
#     600: 120,
#     700: 150,
#     800: 150,
#     900: 170,
# }

EXTRA_ERR = {
    500: 150,
    600: 150,
    700: 130,
    800: 100,
    900: 100,
}

EXTRA_EXP = {
    500: -2.0,
    600: -2.0,
    700: -2.0,
    800: -2.0,
    900: -2.0,
}

Z_COEFF = {
    500: [0, 0, 0],
    600: [0, 0, 0],
    700: [0, 0, 0],
    800: [0, 0, 0],
    900: [0, 0, 0],
}

# zcoeff = {
#     500: [0, 0, 60],
#     600: [0, 0, 60],
#     700: [0, 0, 100],
#     800: [0, 0, 30],
#     900: [0, 0, 30],
# }

EXTRA_JITTER = {
    500: get_extra_jitter_500,
    600: get_extra_jitter_600,
    700: get_extra_jitter_700,
    800: get_extra_jitter_800,
    900: get_extra_jitter_900,
}

HO_PHOTONS = 1740.2396627862904 # [photons/subaperture/frame], LGS WFS flux used by the ETC


def read_base_config(path=BASE_INI):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)
    return config


def sample_name(row, index=None):
    """ File stem of a sample; Enrico's `muse_z{airmass}_s{seeing}_n{ngs}` plus an optional unique row index """
    name = f'muse_z{row["airmass"]:0.2f}_s{row["seeing"]:0.2f}_n{row["ngs"]:0.2f}'
    return name if index is None else f'{name}_{int(index):04d}'


def build_tiptop_config(config, row, wave, pupil_path=PUPIL_PATH):
    """
    Fill the base TIPTOP config (in place) for one observation `row` (seeing, tau, ngs, airmass, tc, ngs_flux) and one science
    wavelength `wave` [nm]; `wave=None` reproduces Enrico's multi-wavelength mode (five wavelengths with the 700 nm tuning).
    """
    tc = int(row['tc'])
    config['telescope']['PathPupil'] = f"'{pupil_path}'"
    config['telescope']['ZenithAngle'] = str(np.arccos(1/row['airmass']) * 180/np.pi)
    config['atmosphere']['Seeing'] = str(row['seeing'])
    config['atmosphere']['L0'] = str(L0[tc])
    config['atmosphere']['Cn2Weights'] = str([GLF[tc], 1-GLF[tc]])
    config['atmosphere']['Cn2Heights'] = str([30, 2000])
    config['atmosphere']['WindSpeed'] = str([WIND_SPEED[tc]]*2)
    config['atmosphere']['WindDirection'] = str([WIND_DIR[tc]]*2)

    ngs_flux = max(float(row['ngs_flux']), 1.0)

    config['sensor_LO']['NumberPhotons'] = str([int(ngs_flux)])
    config['sensor_LO']['Gain'] = str(IRLOS_gain(row['ngs']))

    tuning_wave = 700 if wave is None else wave
    config['telescope']['jitter_FWHM'] = str(EXTRA_JITTER[tuning_wave](tc, row['airmass']))
    config['telescope']['extraErrorNm'] = str(EXTRA_ERR[tuning_wave])
    config['telescope']['extraErrorExp'] = str(EXTRA_EXP[tuning_wave])
    config['telescope']['zCoefStaticOn'] = str(Z_COEFF[tuning_wave])
    config['sources_science']['Wavelength'] = str([5e-7, 6e-7, 7e-7, 8e-7, 9e-7]) if wave is None else str([wave*1e-9])

    config['sources_HO']['Height'] = str(90e3*row['airmass'])
    config['sensor_HO']['NumberPhotons'] = str([HO_PHOTONS]*4)
    config['sensor_science']['FieldOfView'] = str(100)
    config['RTC']['SensorFrameRate_LO'] = str(int(IRLOS_frequency(row['ngs'])))
    return config


def write_configs(stars, wave, config_dir, base_ini=BASE_INI, pupil_path=PUPIL_PATH):
    """ One `.ini` per row of `stars` in `config_dir`; returns the file stems """
    config_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for row in stars:
        config = build_tiptop_config(read_base_config(base_ini), row, wave, pupil_path)
        name = sample_name(row, row['index'])
        with open(config_dir / f'{name}.ini', 'w') as configfile:
            config.write(configfile)
        names.append(name)
    return names


def run_simulations(names, config_dir, psf_dir, max_workers=4, overwrite=False):
    """ Run `overallSimulation` for every config (thread pool, as in Enrico's script); existing PSFs are skipped unless `overwrite` """
    psf_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if overwrite or not (psf_dir / f'{n}.fits').exists()]
    print(f'{len(todo)} simulations to run ({len(names) - len(todo)} already present) -> {psf_dir}')

    def run_simulation(name):
        try:
            overallSimulation(str(config_dir), name, str(psf_dir), name)
        except SyntaxError:
            print(name)
        except Exception as e:  # keep the pool alive; the missing PSF is reported at the end
            print(f'{name}: {type(e).__name__}: {e}')

    start = time.time()
    # Run the simulations in parallel with a ThreadPoolExecutor (4 workers, or fewer if there are fewer files)
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(todo)))) as executor:
        list(executor.map(run_simulation, todo))
    print(f'Execution time: {time.time() - start:.2f} s')

    missing = [n for n in names if not (psf_dir / f'{n}.fits').exists()]
    if missing:
        print(f'WARNING: {len(missing)} PSFs missing, e.g. {missing[:3]}')
    return missing


def load_or_generate_stars(path=STARS_TABLE, counts=None, sampling='random', ngs_range=(9.0, 16.0), seed=0, regenerate=False):
    path = Path(path)
    if path.exists() and not regenerate:
        stars = Table.read(path)
        print(f'Loaded {len(stars)} random observations from {path}')
        return stars
    counts = onsky_counts_per_TC() if counts is None else counts
    stars = generate_stars_table(counts, sampling=sampling, ngs_range=ngs_range, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    stars.write(path, overwrite=True)
    print(f'Generated {len(stars)} random observations ({sampling}, seed {seed}) -> {path}')
    return stars


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wave', type=int, nargs='*', default=WAVELENGTHS, choices=WAVELENGTHS, help='Science wavelengths in nm (default: all)')
    parser.add_argument('--n-per-tc', type=int, default=None, help='Samples per TC (default: the on-sky counts per TC)')
    parser.add_argument('--sampling', choices=['random', 'onsky'], default='random', help='Uniform random conditions within each TC (Enrico) or replay of the on-sky conditions')
    parser.add_argument('--ngs-range', type=float, nargs=2, default=(9.0, 16.0), metavar=('MIN', 'MAX'), help='NGS J magnitude range for random sampling')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--stars', type=Path, default=None, help='Observation table (default: <out-dir>/stars.fits)')
    parser.add_argument('--regenerate-stars', action='store_true', help='Draw a new observation table even if one exists')
    parser.add_argument('--overwrite', action='store_true', help='Re-simulate PSFs that already exist')
    args = parser.parse_args()

    counts = None if args.n_per_tc is None else {tc: args.n_per_tc for tc in TC_LIST}
    stars  = load_or_generate_stars(args.stars or args.out_dir / 'stars.fits', counts, args.sampling, tuple(args.ngs_range), args.seed, args.regenerate_stars)

    for wave in args.wave:
        wave_dir   = args.out_dir / f'{wave}nm'
        config_dir = wave_dir / 'configs'
        names = write_configs(stars, wave, config_dir)
        run_simulations(names, config_dir, wave_dir, max_workers=args.workers, overwrite=args.overwrite)
