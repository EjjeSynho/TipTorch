#%%
import sys, os
from project_settings import device, xp, use_cupy, DATA_FOLDER, WEIGHTS_FOLDER

import pickle
import re
import os
import datetime
import requests
import torch
import pandas as pd
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation
from tqdm import tqdm
from astropy.io import fits
from io import StringIO
from skimage.restoration import inpaint
from photutils.background import Background2D, MedianBackground
from astropy.io import fits
from scipy.ndimage import binary_dilation

from MUSE_preproc_utils import GetConfig, LoadImages
from tools.utils import GetROIaroundMax, GetJmag, DownloadFromRemote
from tools.plotting import wavelength_to_rgb

find_closest_λ = lambda λ, λs: xp.argmin(xp.abs(λs-λ)).astype('int')

def check_framework(x):
    # Determine whether an array is NumPy or CuPy.
    
    # Get the module from the array's class
    if hasattr(x, '__module__'):
        module_name = x.__module__.split('.')[0]
        if   module_name == 'numpy': return np
        elif module_name == 'cupy':  return xp

    # Default to NumPy if not using GPU, otherwise CuPy
    return np if not use_cupy else xp

UT4_coords = ('24d37min37.36s', '-70d24m14.25s', 2635.43)
UT4_location = EarthLocation.from_geodetic(lat=UT4_coords[0], lon=UT4_coords[1], height=UT4_coords[2]*u.m)

h = 'HIERARCH ESO '
IRLOS_upgrade_date = pd.to_datetime('2021-03-20T00:00:00').tz_localize('UTC')

# To get MUSE cubes:
# http://archive.eso.org/wdb/wdb/cal/data_types/form --> MUSE --> DATACUBE_STD --> num setups --> NFM-AO-N_Blue-IR_SCI1.0

# To get ASM data:
# http://archive.eso.org/cms/eso-data/ambient-conditions/paranal-ambient-query-forms.html


def DownloadMUSEcalibData(verbose=False):
    import logging
    logger = logging.getLogger(__name__)

    data_folder_path = str(DATA_FOLDER).replace('\\', '/')
    weights_folder_path = str(WEIGHTS_FOLDER).replace('\\', '/')
    tmp1, tmp2, tmp3 = 'https://drive.google.com/file/d/', '/view?usp=drive_link', f'{data_folder_path}/reduced_telemetry/MUSE/'

    data_to_download = [
        (
            "Downloading MUSE NFM calibrator...",
            f'{tmp1}1NdfkmVYxdXgkJbHIlxDv1XTO-ABj6ox8{tmp2}',
            f'{weights_folder_path}/MUSE_calibrator.dict'
        ),
        (
            "Downloading MUSE NFM reduced telemetry data...",
            f'{tmp1}1KJDiLgX9XeXjvskOhYLLAhlDR0UOH4p_{tmp2}',
            f'{tmp3}muse_df_fitted_transforms.pickle'
        ),
        (None, f'{tmp1}1pc7a8H4v_XzF9IT_LGrvM-OkV7jrw0D5{tmp2}', f'{tmp3}muse_df_norm_imputed.pickle'),
        (None, f'{tmp1}1BR9WtPVODV8R7oYaZSxn9ox_jfWJr9nF{tmp2}', f'{tmp3}muse_df_norm_transforms.pickle'),
        (None, f'{tmp1}1iFnB30JEsKKy14282dJ95xuWtHEfXxBN{tmp2}', f'{tmp3}muse_df.pickle'),
        (None, f'{tmp1}1LMgmTSVBhvGOOUW0PZ1Zim5OPcP8L8NB{tmp2}', f'{tmp3}MUSE_fitted_df.pkl'),
        (
            "Downloading related NFM data...",
            f'{tmp1}1YwiJLlSj_pBlGYhTfpDIBulzpxGZTewG{tmp2}',
            f'{tmp3}IRLOS_commands.csv'
        ),
        (None, f'{tmp1}1tz2RAxUMe_axaBjP9VHL3NqN8oJxlAgE{tmp2}', f'{tmp3}LGS_flux.csv'),
        (None, f'{tmp1}1zk4m9y82Movp0Ytn2z4_hsnyL8AoW58x{tmp2}', f'{tmp3}LGS_slopes.csv')
    ]
    
    for message, url, output_path in data_to_download:
        if message:
            if verbose:
                logger.info(message)
            else:
                logger.debug(message)
        
        DownloadFromRemote(share_url=url, output_path=output_path, overwrite=False, verbose=verbose)

DownloadMUSEcalibData(verbose=True)


#%%
def MatchRawWithReduced(include_dirs_raw, include_dirs_cubes, verbose=False):
    ''' Create/read a matching table between the raw exposure files and reduced cubes '''
    cubes_obs_date_table, raw_obs_date_table = {}, {}

    for folder_cubes, folder_raw in zip(include_dirs_cubes, include_dirs_raw):
        cube_files = [f for f in os.listdir(folder_cubes) if f.endswith('.fits')]
        raw_files  = [f for f in os.listdir(folder_raw) if f.endswith('.fits') or f.endswith('.fits.fz')]

        if verbose: print(f'Scanning the cubes files in \"{folder_cubes}\"...')

        for file in tqdm(cube_files):
            with fits.open(os.path.join(folder_cubes, file)) as hdul_cube:
                cubes_obs_date_table[file] = hdul_cube[0].header['DATE-OBS']
        
        if verbose: print(f'Scanning the raw files in \"{folder_raw}\"...')

        for file in tqdm(raw_files):
            with fits.open(os.path.join(folder_raw, file)) as hdul_cube:
                raw_obs_date_table[file] = hdul_cube[0].header['DATE-OBS']

        if verbose: print('-'*15)

    df1 = pd.DataFrame(cubes_obs_date_table.items(), columns=['cube', 'date'])
    df2 = pd.DataFrame(raw_obs_date_table.items(),   columns=['raw',  'date'])

    # Convert date to pd format
    df1['date'] = pd.to_datetime(df1['date']).dt.tz_localize('UTC')
    df2['date'] = pd.to_datetime(df2['date']).dt.tz_localize('UTC')

    df1.set_index('date', inplace=True)
    df2.set_index('date', inplace=True)

    # Get the matching entries
    files_matches = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

    # Get entries that are in df2 (raw files) but not in df1 (cube files)
    in_raw_only = df2[~df2.index.isin(df1.index)].copy()
    in_raw_only.reset_index(inplace=True)

    if verbose:
        print(f"Found {len(files_matches)} matching entries between raw and reduced files")
        print(f"Found {len(in_raw_only)} raw files without corresponding reduced files")
    return files_matches, in_raw_only


def GetExposureTimes(include_dirs_cubes, verbose=False):
    # ------------------ Create/read the dataset with exposure times ---------------------------------------
    exposure_times = {}
    for folder_cubes in include_dirs_cubes:
        cube_files = [f for f in os.listdir(folder_cubes) if f.endswith('.fits')]

        if verbose: print(f'Scanning the cubes files in \"{folder_cubes}\"...')

        for file in tqdm(cube_files):
            with fits.open(os.path.join(folder_cubes, file)) as hdul_cube:
                exposure_times[file] = (hdul_cube[0].header['DATE-OBS'], hdul_cube[0].header['EXPTIME'])

        df_exptime = pd.DataFrame(exposure_times.items(), columns=['filename', 'obs_time'])

        df_exptime['exposure_start'] = pd.to_datetime(df_exptime['obs_time'].apply(lambda x: x[0]))
        df_exptime['exposure_time'] = df_exptime['obs_time'].apply(lambda x: x[1])
        df_exptime.drop(columns=['obs_time'], inplace=True)
        df_exptime['exposure_end'] = df_exptime['exposure_start'] + pd.to_timedelta(df_exptime['exposure_time'], unit='s')
        df_exptime.drop(columns=['exposure_time'], inplace=True)
        df_exptime.set_index('filename', inplace=True)
        df_exptime['exposure_start'] = df_exptime['exposure_start'].dt.tz_localize('UTC')
        df_exptime['exposure_end'] = df_exptime['exposure_end'].dt.tz_localize('UTC')

    return df_exptime


#%%
# ----------------- Read the dataset with IRLOS data ---------------------------------------
def GetIRLOSInfo(IRLOS_cms_folder):

    IRLOS_cmds = pd.read_csv(IRLOS_cms_folder)
    IRLOS_cmds.drop(columns=['Unnamed: 0'], inplace=True)

    IRLOS_cmds['timestamp'] = IRLOS_cmds['timestamp'].apply(lambda x: x[:-1] if x[-1] == 'Z' else x)
    IRLOS_cmds['timestamp'] = pd.to_datetime(IRLOS_cmds['timestamp']).dt.tz_localize('UTC')
    IRLOS_cmds.set_index('timestamp', inplace=True)
    IRLOS_cmds = IRLOS_cmds.sort_values(by='timestamp')

    IRLOS_cmds['command'] = IRLOS_cmds['command'].apply(lambda x: x.split(' ')[-1])

    # Filter etries that don't look like IRLOS regimes
    pattern = r'(\d+x\d+_SmallScale_\d+Hz_\w+Gain)|(\d+x\d+_\d+Hz_\w+Gain)|(\d+x\d+_SmallScale)'
    IRLOS_cmds = IRLOS_cmds[IRLOS_cmds['command'].str.contains(pattern, regex=True)]

    # Read the pre-fetched IRLOS commands file
    IRLOS_df = GetIRLOSInfo(DATA_FOLDER / 'reduced_telemetry/MUSE/IRLOS_commands.csv')

    pattern = r'(?P<window>\d+x\d+)(?:_(?P<scale>SmallScale))?(?:_(?P<frequency>\d+Hz))?(?:_(?P<gain>\w+Gain))?'
    IRLOS_df = IRLOS_cmds['command'].str.extract(pattern)

    IRLOS_df['window']    = IRLOS_df['window'].apply(lambda x: int(x.split('x')[0]))
    IRLOS_df['frequency'] = IRLOS_df['frequency'].apply(lambda x: int(x[:-2]) if pd.notna(x) else x)
    IRLOS_df['gain']      = IRLOS_df['gain'].apply(lambda x: 68 if x == 'HighGain' else 1 if x == 'LowGain' else x)
    IRLOS_df['scale']     = IRLOS_df['scale'].apply(lambda x: x.replace('Scale','') if pd.notna(x) else x)
    IRLOS_df['scale']     = IRLOS_df['scale'].fillna('Small')

    IRLOS_df['plate scale, [mas/pix]'] = IRLOS_df.apply(lambda row: 60  if row.name > IRLOS_upgrade_date and row['scale'] == 'Small' else 242 if row.name > IRLOS_upgrade_date and row['scale'] == 'Large' else 78 if row['scale'] == 'Small' else 324, axis=1)
    IRLOS_df['conversion, [e-/ADU]']   = IRLOS_df.apply(lambda row: 9.8 if row.name > IRLOS_upgrade_date else 3, axis=1)

    IRLOS_df['gain'] = IRLOS_df['gain'].fillna(1).astype('int')
    IRLOS_df['frequency'] = IRLOS_df['frequency'].fillna(200).astype('int')

    return IRLOS_df


#%% ------------------ Read the raw data from the FITS files and from ESO archive ---------------------------------------
def time_from_sec_usec(sec, usec):
    return [datetime.datetime.fromtimestamp(ts, tz=datetime.UTC) + datetime.timedelta(microseconds=int(us)) for ts, us in zip(sec, usec)]


def read_secs_usecs(hdul, table_name):
    return hdul[table_name].data['Sec'].astype(np.int32), hdul[table_name].data['USec'].astype(np.int32)


def time_from_str(timestamp_strs):
    if isinstance(timestamp_strs, str):
        return datetime.datetime.strptime(timestamp_strs, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        return [datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f') for time_str in timestamp_strs]

#-------------------------------- IRLOS realm ------------------------------------------------
def get_background(img, sigmas=1):
    bkg_estimator = MedianBackground()
    bkg = Background2D(img, (5,)*2, filter_size=(3,), bkg_estimator=bkg_estimator)

    background = bkg.background
    background_rms = bkg.background_rms

    threshold = sigmas * background_rms
    mask = img > (background + threshold)

    return background, background_rms, mask


# Get 2x2 images from the IRLOS cube
def GetIRLOScube(hdul_raw):
    if 'SPARTA_TT_CUBE' in hdul_raw:
        IRLOS_cube = hdul_raw['SPARTA_TT_CUBE'].data.transpose(1,2,0)
        win_size = IRLOS_cube.shape[0] // 2

        # Removing the frame of zeros around the cube
        quadrant_1 = np.s_[1:win_size-1,  1:win_size-1, ...]
        quadrant_2 = np.s_[1:win_size-1,  win_size+1:win_size*2-1, ...]
        quadrant_3 = np.s_[win_size+1:win_size*2-1, 1:win_size-1, ...]
        quadrant_4 = np.s_[win_size+1:win_size*2-1, win_size+1:win_size*2-1, ...]

        IRLOS_cube_ = np.vstack([
            np.hstack([IRLOS_cube[quadrant_1], IRLOS_cube[quadrant_2]]), # 1 2
            np.hstack([IRLOS_cube[quadrant_3], IRLOS_cube[quadrant_4]]), # 3 4
        ])

        IRLOS_cube = IRLOS_cube_ - 200 # Minus constant ADU shift

        background, background_rms, mask = get_background(IRLOS_cube.mean(axis=-1), 1)
        IRLOS_cube *= mask[..., None] # Remove noisy background pixels
        return IRLOS_cube, win_size # [ADU], [pix]

    else:
        print('No IRLOS cubes found')
        return None, None


# Get IR loop parameters
def GetIRLoopData(hdul, start_timestamp, verbose=False):
    try:
        LO_regime = hdul[0].header[h+'AOS MAIN MODES IRLOS']
        if verbose: print( hdul[0].header[h+'AOS MAIN MODES IRLOS'] )

        pattern = r"(\d+x\d+)_(\w+)_(\d+Hz)_(\w+)"
        match   = re.match(pattern, LO_regime)

        if match:
            read_window, regime, LO_freq, LO_gain = match.groups()
            read_window = tuple(map(int, read_window.split('x')))[0]

            if   LO_gain == 'HighGain': LO_gain = 68
            elif LO_gain == 'LowGain':  LO_gain = 1

            LO_freq = int(LO_freq.replace('Hz', ''))
            regime = regime.replace('Scale', '')

            IRLOS_data_df = {
                'window': read_window,
                'scale': regime,
                'frequency': LO_freq,
                'gain': LO_gain,
            }
            if verbose: print('Retrieving IRLOS data from the header...')
        else:
            raise ValueError('No matching IRLOS regime found')

    except:
        def get_default_IRLOS_config():
            return {
                'window': 20,
                'scale': 'Small',
                'frequency': 200,
                'gain': 1,
            }

        try:
            selected_IRLOS_entry = IRLOS_df[IRLOS_df.index < start_timestamp].iloc[-1]
            timedelta = pd.to_timedelta(start_timestamp - selected_IRLOS_entry.name)

            if timedelta < pd.to_timedelta('32m'):
                IRLOS_data_df = selected_IRLOS_entry.to_dict()
                if verbose: print('Retrieving IRLOS data from logs...')
            else:
                IRLOS_data_df = get_default_IRLOS_config()
                if verbose: print('Closest log is too far. Using default IRLOS config...')

        except IndexError:
            IRLOS_data_df = get_default_IRLOS_config()
            if verbose: print('No IRLOS-related information found. Using default IRLOS config...')

    if IRLOS_data_df['scale'] == 'Small':
        IRLOS_data_df['plate scale, [mas/pix]'] = 78  if start_timestamp > IRLOS_upgrade_date else 60
    else:
        IRLOS_data_df['plate scale, [mas/pix]'] = 314 if start_timestamp > IRLOS_upgrade_date else 242

    IRLOS_data_df['conversion, [e-/ADU]'] = 9.8 if start_timestamp > IRLOS_upgrade_date else 3
    IRLOS_data_df['RON, [e-]'] = 1 if start_timestamp > IRLOS_upgrade_date else 11

    return pd.DataFrame([IRLOS_data_df], index=[start_timestamp])


# Computes IRLOS photons at M1 level
def GetIRLOSphotons(flux_ADU, LO_gain, LO_freq, convert_factor): #, [ADU], [1], [Hz], [e-/ADU]
    QE = 1.0 # [photons/e-]
    transmission = 0.4 # [1]
    M1_area = (8-1.12)**2 * np.pi / 4 # [m^2]
    return flux_ADU / QE * convert_factor / LO_gain * LO_freq / M1_area * transmission # [photons/s/m^2]


def GetIRLOSdata(hdul_raw, hdul_cube, start_time, IRLOS_cube):
    # Read NGS loop parameters
    IRLOS_data_df = GetIRLoopData(hdul_raw, start_time, verbose=False)

    # Read NGS flux values
    try:
        #[ADU/frame/sub aperture] * 4 sub apertures if 2x2 mode
        IRLOS_flux_ADU_h = sum([hdul_cube[0].header[f'{h}AOS NGS{i+1} FLUX'] for i in range(4)]) * 4 # [ADU/frame]
        IRLOS_data_df['ADUs (header)'] = IRLOS_flux_ADU_h

        LO_gain    = IRLOS_data_df['gain'].item()
        LO_freq    = IRLOS_data_df['frequency'].item()
        conversion = IRLOS_data_df['conversion, [e-/ADU]'].item()

        IRLOS_flux = GetIRLOSphotons(IRLOS_flux_ADU_h, LO_gain, LO_freq, conversion) # [photons/s/m^2]
        IRLOS_data_df['IRLOS photons, [photons/s/m^2]'] = np.round(IRLOS_flux).astype('uint32')
        IRLOS_data_df['NGS mag (from ph.)'] = GetJmag(IRLOS_data_df.iloc[0]['IRLOS photons, [photons/s/m^2]'])

    except KeyError:
        IRLOS_data_df['ADUs (header)'] = None
        IRLOS_data_df['IRLOS photons, [photons/s/m^2]'] = None
        IRLOS_data_df['NGS mag (from ph.)'] = None

    if IRLOS_cube is not None:
        IRLOS_flux_ADU_cube = IRLOS_cube.mean(axis=0).sum() # [ADU/frame]
        IRLOS_data_df['ADUs (from 2x2)'] = IRLOS_flux_ADU_cube
        IRLOS_flux_cube = GetIRLOSphotons(IRLOS_flux_ADU_cube, LO_gain, LO_freq, conversion) # [photons/s/m^2]
        IRLOS_data_df['IRLOS photons (cube), [photons/s/m^2]'] = IRLOS_flux_cube
        IRLOS_data_df['NGS mag (from 2x2)'] = GetJmag(IRLOS_flux_cube)
    else:
        IRLOS_data_df['ADUs (from 2x2)'] = None
        IRLOS_data_df['NGS mag (from 2x2)'] = None
        IRLOS_data_df['IRLOS photons (cube), [photons/s/m^2]'] = None

    if f'{h}SEQ NGS MAG' in hdul_cube[0].header:
        IRLOS_data_df['NGS mag (header)'] = hdul_cube[0].header[f'{h}SEQ NGS MAG']
    else:
        IRLOS_data_df['NGS mag (header)'] = None

    return IRLOS_data_df


# ------------------------ LGS realm ------------------------------------------------
def GetLGSphotons(flux_LGS, HO_gain):
    conversion_factor = 18 #[e-/ADU]
    GALACSI_transmission = 0.31 #(with VLT) / 0.46 (without VLT), different from IRLOS!
    HO_rate = 1000 #[Hz]
    detector_DIT = (1-0.01982)*1e-3 # [s], 0.01982 [ms] for the frame transfer
    QE = 0.9 # [e-/photons]
    # gain should always = 100 # Laser WFS gain
    num_subapertures = 1240
    M1_area = (8**2 - 1.12**2) * np.pi / 4 # [m^2]

    return flux_LGS * conversion_factor * num_subapertures  \
        / HO_gain / detector_DIT / QE / GALACSI_transmission \
        * HO_rate / M1_area # [photons/m^2/s]


def GetLGSdata(hdul_cube, cube_name, start_time, fill_missing=False, verbose=False):
    # AOS LGSi FLUX: [ADU] Median flux in subapertures /
    # Median of the LGSi flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.
    
    # Read the pre-fetched dataset with the LGS WFSs data
    LGS_flux_df   = pd.read_csv(DATA_FOLDER / 'reduced_telemetry/MUSE/LGS_flux.csv',   index_col=0)
    LGS_slopes_df = pd.read_csv(DATA_FOLDER / 'reduced_telemetry/MUSE/LGS_slopes.csv', index_col=0)
        
    # Select the relevant LGS data
    LGS_data_df = LGS_flux_df[LGS_flux_df['filename'] == cube_name].copy()
    LGS_data_df.rename(columns = { f'LGS{i} flux': f'LGS{i} flux, [ADU/frame]' for i in range(1,5) }, inplace=True)

    if LGS_data_df.empty:
        if verbose: print('No LGS flux data found. ', end='')
        # Create at least something if there is no data
        LGS_data_df = {'filename': cube_name, 'time': start_time}
  
        if fill_missing:
            if verbose: print('Filling with median values...')
            flux_filler  = 880.0
            works_filler = True
        else:
            if verbose: print('Skipping...')
            flux_filler  = None
            works_filler = None
    
        for i in range(1,5):
            LGS_data_df[f'LGS{i} flux, [ADU/frame]'] = flux_filler  
            LGS_data_df[f'LGS{i} works'] =works_filler
            
        LGS_data_df = pd.DataFrame(LGS_data_df, index=[-1])
        LGS_data_df.set_index('time', inplace=True)
    else:  
        LGS_data_df.index = pd.to_datetime(LGS_data_df.index).tz_localize('UTC')

    # Compute LGS photons
    HO_gains = np.array([hdul_cube[0].header[h+'AOS LGS'+str(i+1)+' DET GAIN'] for i in range(4)]) # must be the same value for all LGSs
    for i in range(1,5):
        LGS_data_df.loc[:,f'LGS{i} photons, [photons/m^2/s]'] = GetLGSphotons(LGS_data_df[f'LGS{i} flux, [ADU/frame]'], HO_gains[i-1]).round().astype('uint32')

    # Correct laser shutter state based on the data from the cube header
    for i in range(1,5):
        if f'{h}LGS{i} LASR{i} SHUT STATE' in hdul_cube[0].header:
            LGS_data_df.loc[:,f'LGS{i} works'] = hdul_cube[0].header[f'{h}LGS{i} LASR{i} SHUT STATE'] == 0
        else:
            LGS_data_df.loc[:,f'LGS{i} works'] = False

    LGS_data_df.drop(columns=['filename'], inplace=True)

    # [pix] slopes RMS / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
    LGS_slopes_data_df = LGS_slopes_df[LGS_slopes_df['filename'] == cube_name].copy()
    
    if LGS_slopes_data_df.empty:
        if verbose: print('No LGS WFS slopes data found. ', end='')
        # Create at least something if there is no data
        LGS_slopes_data_df = {'filename': cube_name, 'time': start_time}
        
        if fill_missing:
            if verbose: print('Filling with median values...')
            slopes_filler = 0.22
        else:
            if verbose: print('Skipping...')
            slopes_filler = None
            
        for i in range(1,5):
            LGS_slopes_data_df[f'AOS.LGS{i}.SLOPERMSY'] = slopes_filler
            LGS_slopes_data_df[f'AOS.LGS{i}.SLOPERMSX'] = slopes_filler
                
        LGS_slopes_data_df = pd.DataFrame(LGS_slopes_data_df, index=[-1])
        LGS_slopes_data_df.set_index('time', inplace=True)
    else:
        LGS_slopes_data_df.index = pd.to_datetime(LGS_slopes_data_df.index).tz_localize('UTC')
    
    LGS_slopes_data_df.drop(columns=['filename'], inplace=True)

    return pd.concat([LGS_data_df, LGS_slopes_data_df], axis=1)


#% ------------------------ Polychromatic cube realm-------------------------------------
# def GetSpectrum(white, data, radius=5):
#     _, ids, _ = GetROIaroundMax(white, radius*2)
#     return check_framework(data).nansum(data[:, ids[0], ids[1]], axis=(1,2))


# def GetSpectrum(data, ids):
#     return check_framework(data).nansum(data[:, ids[0], ids[1]], axis=(1,2))

# Extract the spectrum per target near the PSF peak
def GetSpectrum(data, point, radius=1, debug_show_ROI=False):
    """
    Extract spectral data from a 3D data cube at a specified point.

    Parameters:
    -----------
    data : ndarray, torch.Tensor
        The 3D data cube with shape (wavelength, y, x)
    point : list, ndarray, pd.Series, torch.Tensor
        The (x, y) coordinates of the point
    radius : int, default=1
        Radius of the region around the point to compute the spectrum from. Despite the naming, it's a square region
    debug_show_ROI : bool, default=False
        If True, displays the ROI used for calculation

    Returns:
    --------
    1D array of spectral data
    """
    # Determine the input type
    if   type(point) == list:         point = np.array(point)
    elif type(point) == pd.Series:    point = point.to_numpy()
    elif type(point) == torch.Tensor: point = point.cpu().numpy()
    else:
        raise ValueError('Point must be a list, ndarray, pandas.Series, torch.Tensor, or CuPy array')

    x, y = point[:2].astype('int')
    
    if radius == 0: # No averaging over a radius
        return data[:, y, x]
    
    y_min = max(0, y - radius)
    y_max = min(data.shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(data.shape[2], x + radius + 1)

    # Handle different array types
    if torch.is_tensor(data):
        if debug_show_ROI:
            plt.imshow(torch.nansum(data[:, y_min:y_max, x_min:x_max], dim=0).cpu().numpy(), origin='lower')
            plt.show()

        return torch.nanmean(data[:, y_min:y_max, x_min:x_max], dim=(-2,-1))
    else:
        array_module = check_framework(data) # Check if data is a CuPy array

        if debug_show_ROI:
            plt.imshow(array_module.nansum(data[:, y_min:y_max, x_min:x_max], axis=0), origin='lower')
            plt.show()

        return np.nanmean(data[:, y_min:y_max, x_min:x_max], axis=(-2,-1))


def range_overlap(v_min, v_max, h_min, h_max):
    start_of_overlap, end_of_overlap = max(v_min, h_min), min(v_max, h_max)
    return 0.0 if start_of_overlap > end_of_overlap else (end_of_overlap-start_of_overlap) / (v_max-v_min)


def GetImageSpectrumHeader(
    hdul_cube,
    show_plots=False,
    crop_cube=False,
    extract_spectrum=False,
    impaint=False,
    verbose=False):

    # Extract spectral range information
    start_spaxel = hdul_cube[1].header['CRPIX3']
    num_λs = int(hdul_cube[1].header['NAXIS3']-start_spaxel+1)
    Δλ = hdul_cube[1].header['CD3_3' ] / 10.0
    λ_min = hdul_cube[1].header['CRVAL3'] / 10.0
    λs = xp.arange(num_λs)*Δλ + λ_min
    λ_max = λs.max()

    # 1 [erg] = 10^−7 [J]
    # 1 [Angstrom] = 10 [nm]
    # 1 [cm^2] = 0.0001 [m^2]
    # [10^-20 * erg / s / cm^2 / Angstrom] = [10^-22 * W/m^2 / nm] = [10^-22 * J/s / m^2 / nm]
    units = hdul_cube[1].header['BUNIT'] # flux units

    # Polychromatic image
    if use_cupy:
        if verbose: print('Transferring the MUSE cube to GPU...')
        cube_data = xp.array(hdul_cube[1].data)
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
    else:
        cube_data = hdul_cube[1].data

    white = xp.nan_to_num(cube_data).sum(axis=0)

    if extract_spectrum:
        if verbose: print('Extracting the target\'s spectrum...')
        _, _, max_id = GetROIaroundMax(white, 10) # TODO: this function is redundant
        spectrum = GetSpectrum(cube_data, max_id, radius=5)
    else:
        spectrum = xp.ones_like(λs)

    wvl_bins = None #np.array([478, 511, 544, 577, 606, 639, 672, 705, 738, 771, 804, 837, 870, 903, 935], dtype='float32')

    # Pre-defined bad wavelengths ranges
    bad_wvls = xp.array([[450, 478], [577, 606]])
    bad_ids  = xp.array([find_closest_λ(wvl, λs) for wvl in bad_wvls.flatten()], dtype='int').reshape(bad_wvls.shape)
    valid_λs = xp.ones_like(λs)

    for i in range(len(bad_ids)):
        valid_λs[bad_ids[i,0]:bad_ids[i,1]+1] = 0
        bad_wvls[i,:] = xp.array([λs[bad_ids[i,0]], λs[bad_ids[i,1]+1]])

    if wvl_bins is not None:
        if verbose: print('Using pre-defined wavelength bins...')
        λ_bins_smart = wvl_bins
    else:
        # Bin data cubes
        # Before the sodium filter
        if verbose: print('Generating smart wavelength bins...')
        λ_bin = (bad_wvls[1][0]-bad_wvls[0][1])/3.0
        λ_bins_before = bad_wvls[0][1] + xp.arange(4)*λ_bin
        bin_ids_before = [find_closest_λ(wvl, λs) for wvl in λ_bins_before]

        # After the sodium filter
        λ_bins_num    = (λ_max-bad_wvls[1][1]) / xp.diff(λ_bins_before).mean()
        λ_bins_after  = bad_wvls[1][1] + xp.arange(λ_bins_num+1)*λ_bin
        bin_ids_after = [find_closest_λ(wvl, λs) for wvl in λ_bins_after]
        bins_smart    = bin_ids_before + bin_ids_after
        λ_bins_smart  = λs[bins_smart]

    if verbose:
        print('Wavelength bins, [nm]:')
        print(*λ_bins_smart.astype('int'))

    if isinstance(crop_cube, tuple):
        ROI = crop_cube
    elif crop_cube:
        _, ROI, _ = GetROIaroundMax(white, win=200)
    else:
        ROI = (slice(0, white.shape[0]), slice(0, white.shape[1]))

    progress_bar = tqdm if verbose else lambda x: x

    # Generate reduced cubes
    if verbose: print('Generating binned data cubes...')
    data_reduced = xp.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    std_reduced  = xp.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    # Central λs at each spectral bin
    wavelengths  = xp.zeros(len(λ_bins_smart)-1)
    flux         = xp.zeros(len(λ_bins_smart)-1)

    bad_layers = [] # list of spectral layers that are corrupted (excluding the sodium filter)
    bins_to_ignore = [] # list of bins that are corrupted (including the sodium filter)

    # Loop over spectral bins
    if verbose:
        processing_unit = "GPU" if use_cupy else "CPU"
        print(f'Processing spectral bins on {processing_unit}...')

    for bin in progress_bar( range(len(bins_smart)-1) ):
        chunk = cube_data[ bins_smart[bin]:bins_smart[bin+1], ROI[0], ROI[1] ]
        wvl_chunck = λs[bins_smart[bin]:bins_smart[bin+1]]
        flux_chunck = spectrum[bins_smart[bin]:bins_smart[bin+1]]

        # Check if it is a sodium filter range
        if range_overlap(wvl_chunck.min(), wvl_chunck.max(), bad_wvls[1][0], bad_wvls[1][1]) > 0.9:
            bins_to_ignore.append(bin)
            continue

        for i in range(chunk.shape[0]):
            layer = chunk[i,:,:]
            if xp.isnan(layer).sum() > layer.size//2: # if more than 50% of the pixela on image are NaN
                bad_layers.append((bin, i))
                wvl_chunck[i]  = xp.nan
                flux_chunck[i] = xp.nan
                continue
            else:
                if impaint:
                    # Inpaint bad pixels per spectral slice
                    # l_std = layer.flatten().std()
                    mask_inpaint = xp.zeros_like(layer, dtype=xp.int8)
                    mask_inpaint[xp.where(xp.isnan(layer))] = 1
                    # mask_inpaint[xp.where(layer < -l_std)] = 1
                    mask_inpaint[xp.where(layer < -1e3)] = 1
                    # mask_inpaint[xp.where(xp.abs(layer) < 1e-9)] = 1
                    # mask_inpaint[xp.where(layer >  5*l_std)] = 1
                    if use_cupy:
                        chunk[i,:,:] = xp.array(
                            inpaint.inpaint_biharmonic(
                                image = xp.asnumpy(layer),
                                mask  = xp.asnumpy(mask_inpaint)
                            ),
                            dtype=xp.float32
                        )
                    else:
                        chunk[i,:,:] = inpaint.inpaint_biharmonic(layer, mask_inpaint)

                else:
                    chunk[i,:,:] = xp.nan_to_num(layer, nan=0.0)

        data_reduced[bin,:,:] = xp.nansum(chunk, axis=0)
        std_reduced [bin,:,:] = xp.nanstd(chunk, axis=0)
        wavelengths[bin] = xp.nanmean(xp.array(wvl_chunck)) # central wavelength for this bin
        flux[bin] = xp.nanmean(xp.array(flux_chunck))

    # Send back to RAM
    if use_cupy:
        λs           = xp.asnumpy(λs)
        valid_λs     = xp.asnumpy(valid_λs)
        spectrum     = xp.asnumpy(spectrum)
        λ_bins_smart = xp.asnumpy(λ_bins_smart)
        data_reduced = xp.asnumpy(data_reduced)
        std_reduced  = xp.asnumpy(std_reduced)
        wavelengths  = xp.asnumpy(wavelengths)
        flux         = xp.asnumpy(flux)
        white        = xp.asnumpy(white)

    # Generate spectral plot (if applies)
    fig_handler = None
    if show_plots and crop_cube and extract_spectrum:
        if verbose: print("Generating spectral plot...")
        # Initialize arrays using np.zeros_like to match λs shape
        Rs, Gs, Bs = np.zeros_like(λs), np.zeros_like(λs), np.zeros_like(λs)

        for i, λ in enumerate(λs):
            Rs[i], Gs[i], Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

        # Scale the RGB arrays by valid_λs and spectrum
        Rs = Rs * valid_λs * spectrum / np.median(spectrum)
        Gs = Gs * valid_λs * spectrum / np.median(spectrum)
        Bs = Bs * valid_λs * spectrum / np.median(spectrum)

        # Create a color array by stacking and transposing appropriately
        colors = np.dstack([np.vstack([Rs, Gs, Bs])]*600).transpose(2,1,0)

        # Plotting using matplotlib
        fig_handler = plt.figure(dpi=200)
        plt.imshow(colors, extent=[λs.min(), λs.max(), 0, 120])
        plt.ylim(0, 120)
        plt.vlines(λ_bins_smart, 0, 120, color='white')  # draw bins borders
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=2.0, color='white')
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=0.5, color='blue')
        plt.xlabel(r"$\lambda$, [nm]")
        plt.ylabel(r"$\left[ 10^{-20} \frac{erg}{s \cdot cm^2 \cdot \AA} \right]$")
        ax = plt.gca()


    data_reduced = np.delete(data_reduced, bins_to_ignore, axis=0)
    std_reduced  = np.delete(std_reduced,  bins_to_ignore, axis=0)
    wavelengths  = np.delete(wavelengths,  bins_to_ignore)
    flux         = np.delete(flux,         bins_to_ignore)

    if verbose:
        print(str(len(bad_layers))+'/'+str(cube_data.shape[0]), '('+str(xp.round(len(bad_layers)/cube_data.shape[0],2))+'%)', 'slices are corrupted')

    del cube_data

    if use_cupy:
        if verbose: print("Freeing GPU memory...")
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


    # Collect the telemetry from the header
    if verbose: print("Collecting data...")
    spectral_info = {
        'spectrum (full)': (λs, spectrum),
        'wvl range':       [λ_min, λ_max, Δλ],  # From header
        'filtered wvls':   bad_wvls,
        'flux units':      units,
        'wvl bins':        λ_bins_smart,  # Wavelength bins
        'wvls binned':     wavelengths,
        'flux binned':     flux,
        'spectrum fig':    fig_handler
    }


    def convert_to_dms(angle):
        is_negative = angle < 0
        angle = abs(angle)
        degrees = int(angle // 10000)
        minutes = int((angle % 10000) // 100)
        seconds = angle % 100
        if is_negative:
            degrees = -degrees
        return degrees, minutes, seconds

    def format_dms(degrees, minutes, seconds):
        return f"{degrees:+03}d{minutes:02}m{seconds:06.3f}s"

    # Try reading NGS coordinate
    try:
        alpha = hdul_cube[0].header[h+'AOS NGS ALPHA'] # Alpha coordinate for the NGS, [hms]
        delta = hdul_cube[0].header[h+'AOS NGS DELTA'] # Delta coordinate for the NGS, [dms]

        # Convert numerical format to [hms] and [dms] string format
        alpha_hms = f"{int(alpha // 10000):02}h{int((alpha % 10000) // 100):02}m{alpha % 100:06.3f}s"
        delta_dms = format_dms(*convert_to_dms(delta))

        coord_NGS = SkyCoord(alpha_hms, delta_dms, frame='icrs')

        ra_NGS, dec_NGS = (coord_NGS.ra.deg, coord_NGS.dec.deg)
        
    except:
        ra_NGS, dec_NGS = (None, None)


    '''
    ra_sci  = hdul_cube[0].header['RA']
    dec_sci = hdul_cube[0].header['DEC']

    coord_science = SkyCoord(ra=ra_sci*u.deg, dec=dec_sci*u.deg, frame='icrs')

    tel_alt = hdul_cube[0].header[h+'TEL ALT']
    tel_az  = hdul_cube[0].header[h+'TEL AZ']

    altaz = AltAz(alt=tel_alt*u.deg, az=tel_az*u.deg, location=UT4_location, obstime=Time(start_time))

    coord_VLT = SkyCoord(altaz, frame='altaz', obstime=start_time)

    # Extract RA and Dec in degrees
    ra_VLT  = coord_VLT.icrs.ra.deg
    dec_VLT = coord_VLT.icrs.dec.deg

    print( coord_VLT.separation(coord_science).degree )
    print( coord_VLT.separation(coord_NGS).degree )
    print( coord_science.separation(coord_NGS).degree )
    '''

    parang = (hdul_cube[0].header[h+'TEL PARANG END' ] + hdul_cube[0].header[h+'TEL PARANG START']) * 0.5
    alt    = hdul_cube[0].header[h+'TEL ALT']

    try:
        if h+'INS DROT START' in hdul_cube[0].header:
            derot_ang = (hdul_cube[0].header[h+'INS DROT START'] + hdul_cube[0].header[h+'INS DROT END']) * 0.5
        else:
            derot_ang = hdul_cube[0].header[h+'INS DROT END']
    except:
        derot_ang = -0.5 * (parang + alt)


    if h+'SEQ NGS MAG' in hdul_cube[0].header:
        NGS_mag = hdul_cube[0].header[h+'SEQ NGS MAG']
    else:
        NGS_mag = None

    observation = {
        'Date-obs':      hdul_cube[0].header['DATE-OBS'], # UTC
        'Exp. time':     hdul_cube[0].header['EXPTIME'],
        'Tel. altitude': alt,
        'Tel. azimuth':  hdul_cube[0].header[h+'TEL AZ'],
        'Par. angle':    parang,
        'Derot. angle':  derot_ang, # [deg], = -0.5*(par + alt)
    }

    science_target = {
        'Science target': hdul_cube[0].header[h+'OBS TARG NAME'],
        'RA (science)':   hdul_cube[0].header['RA'],
        'DEC (science)':  hdul_cube[0].header['DEC'],
    }

    from_header = {
        'Pixel scale (science)': hdul_cube[0].header[h+'OCS IPS PIXSCALE']*1000, #[mas/pixel]
        'Airmass':               (hdul_cube[0].header[h+'TEL AIRM START'] + hdul_cube[0].header[h+'TEL AIRM END']) / 2.0,
        'Seeing (header)':       (hdul_cube[0].header[h+'TEL AMBI FWHM START'] + hdul_cube[0].header[h+'TEL AMBI FWHM END']) / 2.0,
        'Tau0 (header)':         hdul_cube[0].header[h+'TEL AMBI TAU0'],
        'Temperature (header)':  hdul_cube[0].header[h+'TEL AMBI TEMP'],
        'Wind dir (header)':     hdul_cube[0].header[h+'TEL AMBI WINDDIR'],
        'Wind speed (header)':   hdul_cube[0].header[h+'TEL AMBI WINDSP']
    }

    NGS_target = {
        'NGS RA':  ra_NGS,
        'NGS DEC': dec_NGS,
        'NGS mag': NGS_mag
    }

    try:
        from_header['Strehl (header)'] = hdul_cube[0].header[h+'OBS STREHLRATIO']
    except KeyError:
        try:
            from_header['Strehl (header)'] = hdul_cube[0].header[h+'DRS MUSE RTC STREHL'] #+ hdul_cube[0].header[h+'DRS MUSE RTC STREHERR']*1j
        except KeyError:
            from_header['Strehl (header)'] = np.nan
    else:
        from_header['Strehl (header)'] = np.nan

    images = {
        'cube':  data_reduced,
        'std':   std_reduced,
        'white': white[ROI],
    }

    data = science_target | observation | NGS_target | from_header

    data_df = pd.DataFrame(data, index=[0])
    data_df.rename(columns={'Date-obs': 'time'}, inplace=True)
    data_df.set_index('time', inplace=True)
    data_df.index = pd.to_datetime(data_df.index).tz_localize('UTC')
    if verbose: print('Finished processing spectral cube!')

    return images, data_df, spectral_info


# ------------------------------ Raw header realm ------------------------------------------
def extract_countables_df(hdul, table_name, entry_pattern):
    keys_list = [x for x in hdul[table_name].header.values()]

    if 'Sec' in keys_list:
        timestamps = pd.to_datetime(time_from_sec_usec(*read_secs_usecs(hdul, table_name)))
    else:
        timestamps = pd.to_datetime(time_from_str(hdul[table_name].data['TIME_STAMP'])).tz_localize('UTC')

    values_dict = {}

    for i in range (0, 50):
        entry = entry_pattern(i)
        if entry in keys_list:
            values_dict[entry] = hdul[table_name].data[entry]

    return pd.DataFrame(values_dict, index=timestamps)


def convert_to_default_byte_order(df):
    for col in df.columns:
        # 'byteswap' and 'newbyteorder' should be applied to the underlying numpy arrays
        if df[col].dtype.byteorder not in ('<', '=', '|'):
            df[col] = df[col].apply(lambda x: x.byteswap().newbyteorder() if hasattr(x, 'byteswap') else x)
    return df


def GetRawHeaderData(hdul_raw):

    Cn2_data_df, atm_data_df, asm_data_df = None, None, None

    def concatenate_non_empty_dfs(df_list):
        non_empty_dfs = [df for df in df_list if not df.empty]
        if non_empty_dfs:
            return convert_to_default_byte_order(pd.concat(non_empty_dfs, axis=1))

        else:
            return None


    if 'SPARTA_CN2_DATA' in hdul_raw:
        Cn2_data_entries    = ['L0Tot', 'r0Tot', 'seeingTot']
        Cn2_data_countables = ['CN2_ALT', 'CN2_FRAC_ALT', 'ALT', 'L0_ALT']

        Cn2_data_df = [
            extract_countables_df(hdul_raw, 'SPARTA_CN2_DATA', lambda i: f'{x}{i}') for x in Cn2_data_countables
        ] + [
            pd.DataFrame(
                { entry: hdul_raw['SPARTA_CN2_DATA'].data[entry] for entry in Cn2_data_entries if entry in hdul_raw['SPARTA_CN2_DATA'].header.values()},
                index = time_from_sec_usec(*read_secs_usecs(hdul_raw, 'SPARTA_CN2_DATA'))
            )
        ]
        Cn2_data_df = concatenate_non_empty_dfs(Cn2_data_df)


    if 'SPARTA_ATM_DATA' in hdul_raw:
        atm_countables_list = [
            'R0', 'L0', 'SEEING', 'FWHM_GAIN', 'STREHL', 'TURVAR_RES',
            'TURVAR_TOT', 'TUR_ALT', 'TUR_GND', 'SLOPERMSX', 'SLOPERMSY'
        ]

        atm_data_df = [
            extract_countables_df(hdul_raw, 'SPARTA_ATM_DATA', lambda i: f'LGS{i}_{x}') for x in atm_countables_list
        ]
        atm_data_df = concatenate_non_empty_dfs(atm_data_df)


    if 'ASM_DATA' in hdul_raw:
        asm_countables_list =  ['MASS_TURB', 'SLODAR_CNSQ']
        asm_entries = [
            'R0',
            'MASS_FRACGL',
            'DIMM_SEEING',
            'AIRMASS',
            'ASM_RFLRMS',
            'SLODAR_TOTAL_CN2',
            'IA_FWHMLIN', 'IA_FWHMLINOBS', 'IA_FWHM',
            'SLODAR_FRACGL_300', 'SLODAR_FRACGL_500',
            'ASM_WINDDIR_10', 'ASM_WINDDIR_30',
            'ASM_WINDSPEED_10', 'ASM_WINDSPEED_30',
        ]

        asm_data_df = [
            extract_countables_df(hdul_raw, 'ASM_DATA', lambda i: f'{x}{i}') for x in asm_countables_list
        ] + [
            pd.DataFrame(
                { entry: hdul_raw['ASM_DATA'].data[entry] for entry in asm_entries if entry in hdul_raw['ASM_DATA'].header.values()},
                index = pd.to_datetime(time_from_str(hdul_raw['ASM_DATA'].data['TIME_STAMP'])).tz_localize('UTC')
            )
        ]
        asm_data_df = concatenate_non_empty_dfs(asm_data_df)

    return Cn2_data_df, atm_data_df, asm_data_df


def FetchFromESOarchive(start_time, end_time, minutes_delta=1, verbose=False):
    start_time_str = (start_time - pd.Timedelta(f'{minutes_delta}m')).tz_localize(None).isoformat()
    end_time_str   = (end_time   + pd.Timedelta(f'{minutes_delta}m')).tz_localize(None).isoformat()

    req_str = lambda system: f'http://archive.eso.org/wdb/wdb/asm/{system}_paranal/query?wdbo=csv&start_date={start_time_str}..{end_time_str}'

    def GetFromURL(request_URL):
        response = requests.get(request_URL)

        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)

            if df.columns[0][0] == '#':
                if verbose: print('- No data retrieved')
                return None

            df = df[~df['Date time'].str.contains('#')]
            df.rename(columns={'Date time': 'time'}, inplace=True)
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
            df.set_index('time', inplace=True)
            return df

        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
            return None


    if verbose: print('Querying ASM database...')
    request_asm = req_str('meteo')

    asm_entries = [
        ('tab_press', 0),
        ('tab_presqnh', 0),
        ('tab_temp1', 1), ('tab_temp2', 0), ('tab_temp3', 0), ('tab_temp4', 0),
        ('tab_tempdew1', 0), ('tab_tempdew2', 0), ('tab_tempdew4', 0),
        ('tab_dustl1', 0), ('tab_dustl2', 0), ('tab_dusts1', 0), ('tab_dusts2', 0), ('tab_rain', 0),
        ('tab_rhum1', 0), ('tab_rhum2', 0), ('tab_rhum4', 0),
        ('tab_wind_dir1', 1), ('tab_wind_dir1_180', 0), ('tab_wind_dir2', 0), ('tab_wind_dir2_180', 0),
        ('tab_wind_speed1', 1), ('tab_wind_speed2', 0), ('tab_wind_speedu', 0), ('tab_wind_speedv', 0), ('tab_wind_speedw', 0)
    ]
    request_asm += ''.join([f'&{entry[0]}={entry[1]}' for entry in asm_entries])
    asm_df = GetFromURL(request_asm)


    if verbose: print('Querying MASS-DIMM data...')
    request_massdimm = req_str('mass')

    mass_dimm_entries = [
        ('tab_fwhm', 1),
        ('tab_fwhmerr', 0),
        ('tab_tau', 1),
        ('tab_tauerr', 0),
        ('tab_tet', 0),
        ('tab_teterr', 0),
        ('tab_alt', 0),
        ('tab_alterr', 0),
        ('tab_fracgl', 1),
        ('tab_turbfwhm', 1),
        ('tab_tau0', 1),
        ('tab_tet0', 0),
        ('tab_turb_alt', 0),
        ('tab_turb_speed', 1)
    ]
    request_massdimm += ''.join([f'&{entry[0]}={entry[1]}' for entry in mass_dimm_entries])
    massdimm_df = GetFromURL(request_massdimm)


    if verbose: print('Querying DIMM data...')
    request_dimm = req_str('dimm')
    dimm_entries = [
        ('tab_fwhm', 1),
        ('tab_rfl', 1),
        ('tab_rfl_time', 0)
    ]
    request_dimm += ''.join([f'&{entry[0]}={entry[1]}' for entry in dimm_entries])
    dimm_df = GetFromURL(request_dimm)

    if verbose: print('Querying SLODAR data...')
    request_slodar = req_str('slodar')
    slodar_entries = [
        ('tab_cnsqs_uts', 1),
        ('tab_fracgl300', 1),
        ('tab_fracgl500', 1),
        ('tab_hrsfit', 1),
        ('tab_fwhm', 1)
    ]
    request_slodar += ''.join([f'&{entry[0]}={entry[1]}' for entry in slodar_entries])

    slodar_df = GetFromURL(request_slodar)
    return asm_df, massdimm_df, dimm_df, slodar_df

# Function to create the flat DataFrame
def create_flat_dataframe(df):
    flat_data = {}

    for col in df.columns:
        # print(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if there is at least one non-NaN value
            if df[col].dropna().empty:
                flat_data[col] = np.nan
            else:
                # Compute the median for numeric columns, ignoring NaNs
                median_val = df[col].median(skipna=True)
                flat_data[col] = median_val
        else:
            # Check if all non-nan values are the same for non-numeric columns
            non_nan_values = df[col].dropna().unique()
            if len(non_nan_values) == 1:
                flat_data[col] = non_nan_values[0]
            else:
                flat_data[col] = np.nan

    return pd.DataFrame([flat_data])


def get_PSF_rotation(MUSE_images, derot_angle):
    from scipy.ndimage import rotate
    from tools.utils import cropper
    from skimage.restoration import inpaint_biharmonic
    from scipy.ndimage import gaussian_filter
    from tools.utils import mask_circle
    from photutils.profiles import RadialProfile
    from tools.utils import safe_centroid
    from scipy.interpolate import interp1d
    from PIL import Image, ImageDraw

    PSF_0 = np.copy(MUSE_images['cube'])
    PSF_0 = PSF_0 / PSF_0.sum(axis=(-2,-1))[:,None,None]
    PSF_0 = PSF_0.mean(axis=0)[1:,1:]

    PSF_1 = np.load(DATA_FOLDER / 'reduced_telemetry/MUSE/NFM_PSF_default.npy').mean(axis=0)
    PSF_1 = PSF_1[cropper(PSF_1, PSF_0.shape[-1])]

    # Get streaks mask
    line_wid = 15
    width, height = PSF_1.shape
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.line([(0, 20), (90, 90)], fill=1, width=line_wid)
    draw.line([(width, 20), (width-90, 90)], fill=1, width=line_wid)
    draw.line([(width, height-20), (width-90, height-90)], fill=1, width=line_wid)
    draw.line([(0, height-20), (85, height-90)], fill=1, width=line_wid)
    mask_streaks = np.array(image)

    angle_init = derot_angle * 2 + 165 - 45 # The magic formula to get the initial angle
    angle_init += 360 if angle_init <= 180 else -360

    mask_rot  = (1-mask_circle(PSF_0.shape[0], 25)) * mask_circle(PSF_0.shape[0], 80)

    def subtract_radial_avg(img):
        xycen = safe_centroid(img)
        edge_radii = np.arange(img.shape[-1]//2)
        rp = RadialProfile(img, xycen, edge_radii)

        size  = img.shape[0]
        radii = np.arange(0, len(rp.profile))

        x = np.linspace(-size//2, size//2, size)+1
        y = np.linspace(-size//2, size//2, size)+1
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)

        interp_func = interp1d(radii, rp.profile, bounds_error=False, fill_value=0)
        return interp_func(R)


    PSF_0_rad = subtract_radial_avg(PSF_0)
    PSF_1_rad = subtract_radial_avg(PSF_1)

    PSF_0_diff = mask_rot*(PSF_0-PSF_0_rad)
    PSF_0_diff = PSF_0_diff / PSF_0_diff.sum()

    PSF_1_diff = mask_rot*(PSF_1-PSF_1_rad)
    PSF_1_diff = PSF_1_diff / PSF_1_diff.sum() / 3


    search_limits = 15

    def angle_search(PSF_0, PSF_1, angle_0):

        angles = np.linspace(-search_limits, search_limits, search_limits*2+1)

        losses = []
        # for angle in tqdm(angles):
        for angle in angles:
            rot_0 = gaussian_filter(np.copy(PSF_0), 2)
            rot_1 = gaussian_filter(np.copy(PSF_1), 2)

            mask_rot = rotate(mask_streaks, angle+angle_0, reshape=False)
            rot_1_ = rotate(rot_1, angle+angle_0, reshape=False)

            loss = np.abs(rot_1_-rot_0*mask_rot).sum()

            # plt.imshow(np.abs(rot_1_-rot_0)*mask_rot)
            # plt.title(f'Angle: {angle}')
            # plt.show()

            losses.append(loss.copy())

        # plt.plot(angles, losses)
        # plt.show()

        optimal_angle = angles[np.argmin(losses)]

        return 0.0 if np.abs(optimal_angle) == search_limits else optimal_angle

    corrective_ang = angle_search(PSF_0_diff, PSF_1_diff, angle_init)
    return angle_init, corrective_ang


def get_IRLOS_phase(cube):
    try:
        from project_settings import LIFT_PATH
        # Add dependencies to the path
        for path in ['', 'DIP', 'LIFT', 'LIFT/modules', 'experimental']:
            path_new = f'{LIFT_PATH}{path}/..'
            if path_new not in sys.path:
                sys.path.append(path_new)

        from LIFT_full.experimental.IRLOS_2x2_function import estimate_2x2

        return estimate_2x2(cube)

    except Exception as e:
        print(f'Cannot import LIFT module: {e}')
        return None


#%%
def ProcessMUSEcube(
        path_raw,
        path_cube,
        crop=False,
        get_IRLOS_phase=False,
        derotate=False,
        impaint_bad_pixels=False,
        extract_spectrum=False,
        plot_spectrum=False,
        fill_missing_values=False,
        verbose=False
    ):

    cube_name = os.path.basename(path_cube)

    hdul_raw  = fits.open(path_raw)
    hdul_cube = fits.open(path_cube)

    start_time = pd.to_datetime(time_from_str(hdul_cube[0].header['DATE-OBS'])).tz_localize('UTC')
    end_time   = pd.to_datetime(start_time + datetime.timedelta(seconds=hdul_cube[0].header['EXPTIME']))

    if verbose: print(f'\n>>>>> Getting IRLOS data...')
    IRLOS_cube, win = GetIRLOScube(hdul_raw)
    IRLOS_data_df   = GetIRLOSdata(hdul_raw, hdul_cube, start_time, IRLOS_cube)
    LGS_data_df     = GetLGSdata(hdul_cube, cube_name, start_time, fill_missing_values, verbose)

    if win is not None:
        IRLOS_data_df['window'] = win

    if verbose: print(f'\n>>>>> Reading data from reduced MUSE spectral cube...')
    MUSE_images, MUSE_data_df, spectral_info = GetImageSpectrumHeader(
        hdul_cube,
        show_plots = plot_spectrum,
        crop_cube  = crop,
        extract_spectrum = extract_spectrum,
        impaint = impaint_bad_pixels,
        verbose = verbose
    )
    if verbose: print(f'\n>>>>> Reading data from raw MUSE file ...')
    Cn2_data_df, atm_data_df, asm_data_df   = GetRawHeaderData(hdul_raw)
    if verbose: print(f'\n>>>>> Getting data from ESO archive...')
    asm_df, massdimm_df, dimm_df, slodar_df = FetchFromESOarchive(start_time, end_time, minutes_delta=1)

    hdul_cube.close()
    hdul_raw.close()

    # Try to detect the rotation of the PSF with diffracive features, work only for high-SNR oon-axis PSFs
    derot_ang = MUSE_data_df['Derot. angle'].item()
    derot_ang += 360 if derot_ang < -180 else -360

    if derotate:
        pupil_angle, angle_delta = get_PSF_rotation(MUSE_images, derot_ang)
        MUSE_data_df['Pupil angle'] = pupil_angle + angle_delta
    else:
        MUSE_data_df['Pupil angle'] = derot_ang * 2 + 165 - 45

    # Estimate the IRLOS phasecube from 2x2 PSFs
    OPD_subap, OPD_aperture, PSF_estim = (None,)*3
    if IRLOS_cube is not None:
        if get_IRLOS_phase:
            if verbose: print(f'\n>>>>> Reconstructing IRLOS phase via phase diversity...')
            res = get_IRLOS_phase(IRLOS_cube)
            if res is not None:
                OPD_subap, OPD_aperture, PSF_estim = res
                OPD_subap = OPD_subap.astype(np.float32)
                OPD_aperture = OPD_aperture.astype(np.float32)
                PSF_estim = PSF_estim.astype(np.float32)

    IRLOS_phase_data = {
        'OPD per subap': OPD_subap,
        'OPD aperture':  OPD_aperture,
        'PSF estimated': PSF_estim
    }

    if verbose: print(f'\n>>>>> Packing data...')

    # Create a flat DataFrame with temporal dimension compressed
    all_df = pd.concat([
        IRLOS_data_df, LGS_data_df, MUSE_data_df, Cn2_data_df, atm_data_df,
        asm_data_df, asm_df, massdimm_df, dimm_df, slodar_df
    ])
    all_df.sort_index(inplace=True)

    flat_df = create_flat_dataframe(all_df)
    flat_df.insert(0, 'time', start_time)
    flat_df.insert(0, 'name', cube_name)

    MUSE_images['IRLOS cube'] = IRLOS_cube

    data_store = {
        'images': MUSE_images,
        'IRLOS data': IRLOS_data_df,
        'IRLOS phases': IRLOS_phase_data,
        'LGS data': LGS_data_df,
        'MUSE header data': MUSE_data_df,
        'Raw Cn2 data': Cn2_data_df,
        'Raw atm data': atm_data_df,
        'Raw ASM data': asm_data_df,
        'ASM data': asm_df,
        'MASS-DIMM data': massdimm_df,
        'DIMM data': dimm_df,
        'SLODAR data': slodar_df,
        'All data': flat_df,
        'spectral data': spectral_info,
    }
    
    print(f'Success!')
    
    return data_store, cube_name


def RenderDataSample(data_dict, file_name):

    white =  np.log10(1+np.abs(data_dict['images']['white']))
    white -= white.min()
    white /= white.max()

    # from scipy.ndimage import rotate
    # pupil_ang = data_dict['MUSE header data']['Pupil angle'].item() - 45
    # if pupil_ang < -180: pupil_ang += 360
    # if pupil_ang > 180:  pupil_ang -= 360
    # white_rot = rotate(white, pupil_ang, reshape=False)

    if data_dict['images']['IRLOS cube'] is not None:
        IRLOS_img = np.log10(1+np.abs(data_dict['images']['IRLOS cube'].mean(axis=-1)))
        IRLOS_img = data_dict['images']['IRLOS cube'].mean(axis=-1)
    else:
        IRLOS_img = np.zeros_like(white)

    title = file_name.replace('.pickle', '')
    fig, ax = plt.subplots(1,2, figsize=(14, 7.5))

    ax[0].set_title(title)
    ax[1].set_title('IRLOS (2x2)')
    ax[0].imshow(white, cmap='gray')
    ax[1].imshow(IRLOS_img, cmap='hot')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()


def LoadCachedDataMUSE(raw_path, cube_path, cache_path, save_cache=True, device=device, verbose=False):
    """
    This function prepares the data to be understandable by the PSF model. It bins the NFM cube and
    associates the necessary reduced telemetry data.
    """
    with fits.open(cube_path) as cube_fits: 
        cube_full = cube_fits[1].data # Full spectral cube taken directly from the MUSE ITS

    # Compute the mask of valid pixels
    nan_mask = np.abs(np.nansum(cube_full, axis=0)) < 1e-12
    nan_mask = binary_dilation(nan_mask, iterations=2, )
    valid_mask = ~nan_mask
    # Filter out NaN values
    cube_full = np.nan_to_num(cube_full, nan=0.0) * valid_mask[np.newaxis, :, :]

    valid_mask = torch.tensor(valid_mask, device=device).float().unsqueeze(0)

    if os.path.exists(cache_path):
        if verbose: print('Loading existing cached data cube...')
        with open(cache_path, 'rb') as f:
            data_cached = pickle.load(f)
    else:
        if verbose: print('Generating new cached data cube...')

        # Check if the folder exists, if not, create it
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
            if verbose:
                print(f"Folder '{os.path.dirname(cache_path)}' created.")

        data_cached, _ = ProcessMUSEcube(
            path_raw  = raw_path,
            path_cube = cube_path,
            crop = False,
            get_IRLOS_phase = False,
            derotate = False,
            impaint_bad_pixels = False,
            extract_spectrum = False,
            plot_spectrum = False,
            fill_missing_values = True,
            verbose = verbose
        )
    
    # Compute wavelength data
    #TODO: fix uneven Δλ division!
    λ_min, λ_max, Δλ_full = data_cached['spectral data']['wvl range']
    λ_bins = data_cached['spectral data']['wvl bins']
    Δλ_binned = np.median(np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])]))

    if hasattr(λ_max, 'item'): # To compensate for a small error in the data reduction routine
        λ_max = λ_max.item()

    λ_full = np.linspace(λ_min, λ_max, np.round((λ_max-λ_min)/Δλ_full+1).astype('int'))
    assert len(λ_full) == cube_full.shape[0]

    cube_binned, _, _, _ = LoadImages(data_cached, device=device, subtract_background=False, normalize=False, convert_images=True)
    cube_binned = cube_binned.squeeze() * valid_mask

    # Correct the flux to match MUSE cube
    cube_binned = cube_binned * (cube_full.sum(axis=0).max() /  cube_binned.sum(axis=0).max())

    # Extract config file and update it
    model_config, cube_binned = GetConfig(data_cached, cube_binned)
    cube_binned = cube_binned.squeeze()

    model_config['NumberSources'] = 1
    # The bigger size of initialized PSF is needed to extract the flux loss due to cropping to the box_size later
    model_config['sensor_science']['FieldOfView'] = 111
    # Select only a subset of predicted wavelengths and modify the config file accordingly
    λ_binned = model_config['sources_science']['Wavelength'].clone()
    # Assumes that we are not in the pupil tracking mode
    model_config['telescope']['PupilAngle'] = torch.zeros(1, device=device)

    # Select sparse wavelength set
    # TODO: a code to select the specified number of spectral slices
    ids_λ_sparse = np.arange(0, λ_binned.shape[-1], 2)
    λ_sparse = λ_binned[..., ids_λ_sparse]
    model_config['sources_science']['Wavelength'] = λ_sparse

    cube_sparse = cube_binned.clone()[ids_λ_sparse, ...] # Select the subset of λs

    spectral_info = {
        "λ_sparse":  λ_sparse.flatten() * 1e9, # [nm]
        "λ_full":    λ_full, # [nm]
        "Δλ_binned": Δλ_binned,
        "Δλ_full":   Δλ_full
    }

    if save_cache:
        try:
            with open(cache_path, 'wb') as handle:
                pickle.dump(data_cached, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print(f'Error: {e}')

    spectral_cubes = {
        "cube_full":   cube_full,   # this one contains all spectral slices
        "cube_sparse": cube_sparse, # this one contains 7 avg. spectral slices out of 14 pre-computed spectral bins (will be changed)
        "mask":        valid_mask   # this one contains the mask of the data cube
    }

    return spectral_cubes, spectral_info, data_cached, model_config
