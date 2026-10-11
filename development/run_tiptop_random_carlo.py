# Original code developed by Enrico Congui


import pickle
import numpy as np
import pandas as pd
from tiptop.tiptop import overallSimulation

from astropy.table import Table, vstack
from astropy.time import Time

import configparser

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import time

import sys
import argparse


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


def generate_random_observations(n, tc, ngs0=[9, 16], seeing_min=0.4, tau_max=10.0):
    """
    Generate n random observations for a given turbulence category.

    Seeing is sampled from [seeing_min, tc_seeing_max].
    Coherence time (tau) is sampled from [tc_tau_min, tau_max].
    Points that would fall into a stricter TC are rejected via oversampling.
    The returned table always contains exactly n rows.
    """

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

        s = np.random.uniform(seeing_min, seeing_max, batch_size)
        t = np.random.uniform(tau_min, tau_max, batch_size)

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

    ngs = np.random.uniform(ngs0[0], ngs0[1], n)
    airmass = np.random.uniform(1, 2, n)

    tc_arr = turbulence_categories(seeing, tau)

    frequency = np.where(ngs <= 16, 500, 200)
    ngs_flux = NGS_mag_to_photons(ngs, frequency, airmass)

    table = Table([seeing, tau, ngs, airmass, tc_arr, ngs_flux],
                  names=['seeing', 'tau', 'ngs', 'airmass', 'tc', 'ngs_flux'])

    return table

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=int, default=900, choices=[500, 600, 700, 800, 900],
                        help='Wavelength in nm (default: 900)')
    args = parser.parse_args()
    wave = args.wave

    # opening the base ini file
    config = configparser.ConfigParser()
    config.optionxform = str
    # config.read('../May2025/ETC_calibrator/files/muse_base.ini')
    config.read('muse_ltao_Guido.ini')
    suffix = f'SPIE_fix_{wave}'


    if not os.path.isdir(f'./config_files_{suffix}'):
        os.mkdir(f'./config_files_{suffix}')
        subprocess.run(f'cp serviceDescription.json ./config_files_{suffix}', shell=True, capture_output=True, text=True)
        subprocess.run(f'cp tiptop.py ./config_files_{suffix}', shell=True, capture_output=True, text=True)
    else:
        subprocess.run([f'rm ./config_files_{suffix}/*.ini'], capture_output=True, text=True, shell=True)

    if not os.path.isdir(f'./PSF_files_{suffix}'):
        os.mkdir(f'./PSF_files_{suffix}')
    else:
        subprocess.run(f'rm ./PSF_files_{suffix}/*.fits', shell=True, capture_output=True, text=True)

    new = False

    if new:
        stars = []
        stars.append(generate_random_observations(30, tc=10,  ngs0=[0.0, 18.0]))  # TC10
        stars.append(generate_random_observations(30, tc=20,  ngs0=[0.0, 18.0]))  # TC20, excludes TC10
        stars.append(generate_random_observations(30, tc=30,  ngs0=[0.0, 18.0]))  # TC30, excludes TC20+
        stars.append(generate_random_observations(30, tc=50,  ngs0=[0.0, 18.0]))  # TC50, excludes TC30+
        stars.append(generate_random_observations(30, tc=100, ngs0=[0.0, 18.0]))  # TC100, excludes TC50+
        stars = vstack(stars)
        stars.write('stars_30.fits', overwrite=True)
    else:
        stars = Table.read('stars_30.fits')


    # ASM wind speed values
    wind_speed = {
        10:10.4,
        20:10.5,
        30:9.9,
        50:10.1,
        100: 14.5,
    }

     # ASM wind direction values
    wind_dir = {
        10: 12.5,
        20: 28.8,
        30: 40.2,
        50: 37.2,
        100: 37.0,
    }

    L0 = {
        10:12.48,
        20:17.1,
        30:22.1,
        50:23.5,
        100:25.5
    }

    GLF = {
        10:0.971,
        20:0.973,
        30:0.980,
        50:0.972,
        100:0.977
    }

    # extraerr = {
    #     500: 140,
    #     600: 120,
    #     700: 150,
    #     800: 150,
    #     900: 170,
    # }

    extraerr = {
        500: 150,
        600: 150,
        700: 130,
        800: 100,
        900: 100,
    }


    extraexp = {
        500: -2.0,
        600: -2.0,
        700: -2.0,
        800: -2.0,
        900: -2.0,
    }

    zcoeff = {
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

    extrajitt = {
        500: get_extra_jitter_500,
        600: get_extra_jitter_600,
        700: get_extra_jitter_700,
        800: get_extra_jitter_800,
        900: get_extra_jitter_900,
    }


    for row in stars:
        config['telescope']['PathPupil'] = str("'/home/econgiu/Data/ESO/MUSE_PSF/TipTop/SPIE_proceedings/ut4pupil320.fits'")
        config['telescope']['ZenithAngle'] = str(np.arccos(1/row['airmass']) * 180/np.pi)
        config['atmosphere']['Seeing'] = str(row['seeing'])
        config['atmosphere']['L0'] = str(L0[row['tc']])
        config['atmosphere']['Cn2Weights'] = str([GLF[row['tc']], 1-GLF[row['tc']]])
        config['atmosphere']['Cn2Heights'] = str([30, 2000])
        config['atmosphere']['WindSpeed'] = str([wind_speed[row['tc']]]*2)
        config['atmosphere']['WindDirection'] = str([wind_dir[row['tc']]]*2)
        
        if row['ngs_flux'] < 1:
            row['ngs_flux'] = 1
            
        config['sensor_LO']['NumberPhotons'] = str([int(row['ngs_flux'])])
        config['sensor_LO']['Gain'] = '1' if row['ngs'] <= 10.5 else '68' if row['ngs'] <= 16 else '100'
        
        if wave is None:
            config['telescope']['jitter_FWHM'] = str(extrajitt[700](row['tc'], row['airmass']))
            config['telescope']['extraErrorNm'] = str(extraerr[700])
            config['telescope']['extraErrorExp'] = str(extraexp[700])
            config['telescope']['zCoefStaticOn'] = str(zcoeff[700])
            config['sources_science']['Wavelength'] = str([5e-7, 6e-7, 7e-7, 8e-7, 9e-7])
        else:
            config['telescope']['jitter_FWHM'] = str(extrajitt[wave](row['tc'], row['airmass']))
            config['telescope']['extraErrorNm'] = str(extraerr[wave])
            config['telescope']['extraErrorExp'] = str(extraexp[wave])
            config['telescope']['zCoefStaticOn'] = str(zcoeff[wave])
            config['sources_science']['Wavelength'] = str([wave*1e-9])

        config['sources_HO']['Height'] = str(90e3*row['airmass'])
        config['sensor_HO']['NumberPhotons'] = str([1740.2396627862904, 1740.2396627862904, 1740.2396627862904, 1740.2396627862904])
        config['sensor_science']['FieldOfView'] = str(100)
        config['RTC']['SensorFrameRate_LO'] = str(500 if row['ngs'] <= 16 else 200)
        # save to a file
        with open(f'config_files_{suffix}/muse_z{row["airmass"]:0.2f}_s{row["seeing"]:0.2f}_n{row["ngs"]:0.2f}.ini', 'w') as configfile:
            config.write(configfile)

    start = time.time()

    # Raccogli tutti i file .ini da processare
    config_files = [f for f in os.listdir(f'config_files_{suffix}') if f.endswith('.ini')]

    # Esegui le simulazioni in parallelo con ThreadPoolExecutor
    max_workers = min(4, len(config_files))  # Usa 4 worker o meno se ci sono meno file

    def run_simulation(file):
        config_dir = f'config_files_{suffix}'
        psf_dir = f'PSF_files_{suffix}'
        config_name = file.replace('.ini', '')
        
        try:
            overallSimulation(config_dir, config_name, psf_dir, config_name)
            
        except SyntaxError:
            print(config_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_simulation, config_files))

    print(f"Tempo di esecuzione: {time.time() - start:.2f} secondi")


