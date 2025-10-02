#%%
import sys, os

from project_settings import device, xp, use_cupy, default_torch_type
from project_settings import WEIGHTS_FOLDER, PROJECT_PATH, TELEMETRY_CACHE, DATA_FOLDER
import pickle
import re
import os
import datetime
import requests
import torch
import json
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
from pathlib import Path

from tools.utils import GetROIaroundMax, GetJmag, check_framework
from tools.network import DownloadFromRemote
from tools.plotting import wavelength_to_rgb
from managers.parameter_parser import ParameterParser
from managers.config_manager import ConfigManager, MultipleTargetsInDifferentObservations
from scipy.ndimage import rotate
import logging

from typing import Optional, Tuple, Callable

datalab_available = False
try:
    import dlt
    from elasticsearch import Elasticsearch
    from elasticsearch_dsl import Search
    
    es = Elasticsearch(hosts=['datalab.pl.eso.org'])

    es.info() # Test the connection to the Elasticearch server
    datalab_available = True
    logging.info('Succesfully connected to ESO datalab')
    
except ImportError:
    logging.warning('Elasticsearch packages are not available: cannot read from datalab')
    
except Exception as e:
    logging.warning(f'Cannot connect to datalab: {e}')
    datalab_available = False


# Wavelength bins used to bin MUSE NFM multispectral cubes
wvl_bins = np.array([
    478.   , 492.125, 506.25 , 520.375, 534.625, 548.75 , 562.875, 
    577.   , 606.   , 620.25 , 634.625, 648.875, 663.25 , 677.5  ,
    691.875, 706.125, 720.375, 734.75 , 749.   , 763.375, 777.625,
    792.   , 806.25 , 820.625, 834.875, 849.125, 863.5  , 877.75 ,
    892.125, 906.375, 920.75 , 935.
], dtype='float32')

wvl_bins_old = np.array([
    478, 511, 544, 577, 606,
    639, 672, 705, 738, 771,
    804, 837, 870, 903, 935
], dtype='float32')


#%%
find_closest_λ   = lambda λ, λs: check_framework(λs).argmin(check_framework(λs).abs(λs-λ)).astype('int')
pupil_angle_func = lambda par_ang: (45.0 - par_ang) % 360 # where par_ang is the parallactic angle

with open(PROJECT_PATH / Path("project_config.json"), "r") as f:
    project_settings = json.load(f)

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

default_IRLOS_config = '20x20_SmallScale_500Hz_HighGain'

def time_from_sec_usec(sec, usec):
    return [datetime.datetime.fromtimestamp(ts, tz=datetime.UTC) + datetime.timedelta(microseconds=int(us)) for ts, us in zip(sec, usec)]


def read_secs_usecs(hdul, table_name):
    return hdul[table_name].data['Sec'].astype(np.int32), hdul[table_name].data['USec'].astype(np.int32)


def time_from_str(timestamp_strs):
    if isinstance(timestamp_strs, str):
        return datetime.datetime.strptime(timestamp_strs, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        return [datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f') for time_str in timestamp_strs]


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

    tmp1 = 'https://drive.google.com/file/d/'
    tmp2 = '/view?usp=drive_link'
    tmp3 = f'{TELEMETRY_CACHE.absolute().as_posix()}/MUSE/'

    data_to_download = [
        (
            "Downloading MUSE NFM calibrator...",
            f'{tmp1}1NdfkmVYxdXgkJbHIlxDv1XTO-ABj6ox8{tmp2}',
            f'{WEIGHTS_FOLDER.absolute().as_posix()}/MUSE_calibrator.dict'
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


# DownloadMUSEcalibData(verbose=True)

#%%
def MatchRawWithCubes(
        raw_folder:   str | os.PathLike,
        cubes_folder: str | os.PathLike,
        verbose: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Creates a matching table between specified raw MUSE files and reduced multispectral cubes.
    Matching happens based on the date of observation.
    '''
    
    raw_folder, cubes_folder = Path(raw_folder), Path(cubes_folder)
    cubes_obs_date_table, raw_obs_date_table = {}, {}

    if verbose: print(f'Scanning the cubes...')
    for file in tqdm(os.listdir(cubes_folder)):
        with fits.open(cubes_folder / file) as hdul_cube:
            cubes_obs_date_table[file] = hdul_cube[0].header['DATE-OBS']
    
    if verbose: print(f'Scanning the raw files...')
    for file in tqdm(os.listdir(raw_folder)):
        with fits.open(raw_folder / file) as hdul_cube:
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


def GetExposureTimesList(cubes_folder: str | os.PathLike, verbose: bool = False) -> pd.DataFrame:
    """ Creates a table with observation start and exposure times for all cubes in the folder """
    exposure_times = {}
    cubes_folder = Path(cubes_folder)

    if verbose: print(f'Processing {cubes_folder}...')

    for file in tqdm(os.listdir(cubes_folder)):
        if not file.endswith('.fits'):
            continue
        
        with fits.open(cubes_folder / file) as hdul_cube:
            exposure_times[os.path.basename(file)] = (hdul_cube[0].header['DATE-OBS'], hdul_cube[0].header['EXPTIME'])

    df_exptime = pd.DataFrame(exposure_times.items(), columns=['filename', 'obs_time'])

    df_exptime['exposure_start'] = pd.to_datetime(df_exptime['obs_time'].apply(lambda x: x[0]))
    df_exptime['exposure_time'] = df_exptime['obs_time'].apply(lambda x: x[1])
    df_exptime.drop(columns=['obs_time'], inplace=True)
    df_exptime['exposure_end'] = df_exptime['exposure_start'] + pd.to_timedelta(df_exptime['exposure_time'], unit='s')
    df_exptime.drop(columns=['exposure_time'], inplace=True)
    df_exptime.set_index('filename', inplace=True)
    df_exptime['exposure_start'] = df_exptime['exposure_start'].dt.tz_localize('UTC')
    df_exptime['exposure_end']   = df_exptime['exposure_end'].dt.tz_localize('UTC')

    return df_exptime


def check_cached_data(
    cache_path: Path,
    start_time: datetime.datetime,
    end_time:   datetime.datetime,
    fetch_function: Optional[Callable[[datetime.datetime, datetime.datetime], pd.DataFrame]] = None,
    parse_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = lambda x: x,
    time_buffer_before: datetime.timedelta = datetime.timedelta(seconds=30),
    time_buffer_after:  datetime.timedelta = datetime.timedelta(seconds=30),
    verbose: bool = False
) -> pd.DataFrame:
    """
    Get data from a cached CSV if available; otherwise fetch, parse, cache, and return.

    Parameters
    ----------
    cache_path : Path
        Path to the CSV cache file.
    start_time, end_time : datetime.datetime
        Requested time range.
    fetch_function : Optional[Callable]
        Function to fetch data if not in cache. Accepts (start_time, end_time) and returns a DataFrame.
    parse_function : Optional[Callable]
        Function to parse the fetched data before caching. Defaults to identity.
    time_buffer_before, time_buffer_after : datetime.timedelta
        Buffers added to the requested range when *querying the cache* (not when fetching).
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows within [start_time - buffer_before, end_time + buffer_after].
        Returns empty DataFrame if nothing available and no fetching done/available.
    """

    def _load_cache(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            # Corrupt or unreadable? Treat as empty.
            return pd.DataFrame()
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index, format="mixed", utc=True)
        df = df.sort_index()
        # Drop duplicate index entries, keep the first (same as your original)
        df = df[~df.index.duplicated(keep='first')]
        return df

    def _save_cache(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True)

    # 1) Load cache (or empty)
    cache_df = _load_cache(cache_path)

    # 2) Try to satisfy from cache using buffer window
    query_start = start_time - time_buffer_before
    query_end   = end_time   + time_buffer_after

    def _slice(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.loc[(df.index >= query_start) & (df.index <= query_end)]

    result_df = _slice(cache_df)

    if not result_df.empty:
        if verbose:
            print(f"Serving {len(result_df)} rows from cache: {cache_path.name}")
        return result_df

    # 3) If cache didn’t cover it, fetch (if possible)
    if fetch_function is None:
        if verbose:
            if cache_df.empty:
                print("Cache file not found or empty, and no fetch function provided.")
            else:
                print("Requested range not in cache, and no fetch function provided.")
        return pd.DataFrame()

    if verbose:
        if cache_df.empty and not cache_path.exists():
            print(f"Cache file not found. Creating new cache: {cache_path.name}")
        elif cache_df.empty:
            print("Cache is empty. Fetching data...")
        else:
            print("Requested range not in cache. Fetching new data...")

    fetched_df = fetch_function(query_start, query_end)

    if fetched_df.empty:
        if verbose:
            print(f"No data found for requested time range: {start_time} to {end_time}")
        return pd.DataFrame()

    # 4) Parse, merge into cache, dedupe, save
    parsed_df = parse_function(fetched_df) if parse_function else fetched_df

    if not parsed_df.empty:
        merged = pd.concat([cache_df, parsed_df])
        merged.index = pd.to_datetime(merged.index, utc=True)
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep='first')]
        _save_cache(cache_path, merged)

        if verbose:
            first, last = merged.index[0], merged.index[-1]
            print(f"Successfully fetched and updated cache: {cache_path.name} ({first} to {last})")

        # 5) Return slice from merged cache
        return _slice(merged)

    # Parsed empty (unlikely)
    if verbose:
        print("Fetched data parsed to empty; nothing to cache/return.")
    return pd.DataFrame()


#%% -------------------------------- IRLOS realm ------------------------------------------------
def FetchIRLOSlogsFromDatalab(timestamp_start, timestamp_end):
    query = (Search(using=es, index='vltlog*')
                .filter('range', **{'@timestamp': {
                    'lt': timestamp_end.strftime('%Y-%m-%dT%H:%M:%S'),
                    'gt': timestamp_start.strftime('%Y-%m-%dT%H:%M:%S'),
                }})
                .filter('term', envname__keyword='wmfsgw')
                .query('match', logtext='IRLOS')
                .sort('-@timestamp')
                .params(preserve_order=True))

    timestamps, logtexts = [], []
    for result in query.scan():
        timestamps.append(result['@timestamp'])
        logtexts.append(result.logtext)

    IRLOS_logs_dict = { timestamp: cmd for timestamp, cmd in zip(timestamps, logtexts) }
    IRLOS_logs_df = pd.DataFrame(IRLOS_logs_dict.items(), columns=['timestamp', 'command'])
    
    IRLOS_logs_df['timestamp'] = IRLOS_logs_df['timestamp'].apply(lambda x: x[:-1] if x[-1] == 'Z' else x)
    IRLOS_logs_df['timestamp'] = pd.to_datetime(IRLOS_logs_df['timestamp']).dt.tz_localize('UTC')
    IRLOS_logs_df.set_index('timestamp', inplace=True)
    IRLOS_logs_df = IRLOS_logs_df.sort_values(by='timestamp')
    IRLOS_logs_df['command'] = IRLOS_logs_df['command'].apply(lambda x: x.split(' ')[-1])
    
    return IRLOS_logs_df


def ParseIRLOSregimes(IRLOS_regimes_df: pd.DataFrame) -> pd.DataFrame:
    '''Reads the timestamped IRLOS regimes list and returns a dataframe with the IRLOS regimes and their timestamps'''
    # Then extract the regimes parameters using the command pattern
    cmd_pattern = r'(?P<window>\d+x\d+)(?:_(?P<scale>SmallScale))?(?:_(?P<frequency>\d+Hz))?(?:_(?P<gain>\w+Gain))?'
    IRLOS_df = IRLOS_regimes_df['command'].str.extract(cmd_pattern)

    IRLOS_df['window']    = IRLOS_df['window'].apply(lambda x: int(x.split('x')[0]))
    IRLOS_df['frequency'] = IRLOS_df['frequency'].apply(lambda x: int(x[:-2]) if pd.notna(x) else x)
    IRLOS_df['frequency'] = IRLOS_df['frequency'].fillna(200).astype('int')
    IRLOS_df['gain']      = IRLOS_df['gain'].apply(lambda x: 68 if x == 'HighGain' else 1 if x == 'LowGain' else x)
    IRLOS_df['gain']      = IRLOS_df['gain'].fillna(1).astype('int')
    IRLOS_df['scale']     = IRLOS_df['scale'].apply(lambda x: x.replace('Scale','') if pd.notna(x) else x)
    IRLOS_df['scale']     = IRLOS_df['scale'].fillna('Small') # Large scale is not used for obs, so it's safe to assume it

    # Set plate scale accounting for the upgrade date
    IRLOS_df['plate scale, [mas/pix]'] = IRLOS_df.apply(
        lambda row: 78  if row.name > IRLOS_upgrade_date and row['scale'] == 'Small' else
                    60  if row['scale'] == 'Small' else
                    314 if row.name > IRLOS_upgrade_date and row['scale'] == 'Large' else
                    242,
        axis=1
    )
    # Set conversion factor based on upgrade date
    IRLOS_df['conversion, [e-/ADU]'] = IRLOS_df.apply(lambda row: 9.8 if row.name > IRLOS_upgrade_date else 3, axis=1)
    # Set read-out noise based on upgrade date
    IRLOS_df['RON, [e-]'] = IRLOS_df.apply(lambda row: 1 if row.name > IRLOS_upgrade_date else 11, axis=1)
    
    return IRLOS_df


def ParseIRLOSlogs(IRLOS_logs_df: pd.DataFrame) -> pd.DataFrame:
    '''Reads the timestamped IRLOS commands list and returns a dataframe with the IRLOS regimes and their timestamps'''
    # Filter entries that match IRLOS regimes patterns and extract components
    regime_pattern = r'(\d+x\d+_SmallScale_\d+Hz_\w+Gain)|(\d+x\d+_\d+Hz_\w+Gain)|(\d+x\d+_SmallScale)'
    # First extract only rows that match the IRLOS regime pattern
    filtered_regimes_df = IRLOS_logs_df[IRLOS_logs_df['command'].str.match(regime_pattern)]
    return ParseIRLOSregimes(filtered_regimes_df)


def FetchLGSslopesFromDatalab(timestamp_start, timestamp_end):
    HO_slopes_RMS_keywords = [f'AOS.LGS{i}.SLOPERMS{coord}' for i,coord in zip([1,1,2,2,3,3,4,4], ['X','Y']*4)]
    try:
        slopes_RMS = []
        for entry in HO_slopes_RMS_keywords:
            slopes_RMS.append(dlt.query_ts('wmfsgw', entry, timestamp_start, timestamp_end))
            
        for j, df in enumerate(slopes_RMS):
            df.rename(columns={'value': HO_slopes_RMS_keywords[j], 'time': 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)
            
        return pd.concat(slopes_RMS, axis=1)
        
    except Exception as e:
        print(f'No LGS slopes data {e}')
        return pd.DataFrame()


def FetchLGSfluxFromDatalab(timestamp_start, timestamp_end):
    try:
        # [ADU / frame / subaperture] Median flux per subaperture with 0.17 photons/ADU when the camera gain is 100
        LGS_flux = [ dlt.query_ts('wmfsgw', f'AOS.LGS{j}.FLUX', timestamp_start, timestamp_end) for j in range(1,5) ]

        for j, df in enumerate(LGS_flux):
            df.rename(columns={'value': f'LGS{j+1} flux', 'time': 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)
            
        return pd.concat(LGS_flux, axis=1)
    
    except Exception as e:
        print(f'No LGS flux data {e}')
        return pd.DataFrame()


# Get 2x2 images from the IRLOS cube
def GetIRLOScube(hdul_raw, verbose=False):
    def _get_background_(img, sigma=1):
        bkg = Background2D(img, (5,)*2, filter_size=(3,), bkg_estimator=MedianBackground())

        background = bkg.background
        background_rms = bkg.background_rms

        threshold = sigma * background_rms
        mask = img > (background + threshold)
        
        return background, background_rms, mask
    
    if 'SPARTA_TT_CUBE' in hdul_raw:
        if verbose: print('Found IRLOS cube in the FITS file...')
        IRLOS_cube = hdul_raw['SPARTA_TT_CUBE'].data.transpose(1,2,0)
        
        win_size = IRLOS_cube.shape[0] // 2

        # Removing the frame of zeros around the cube
        quadrant_1 = np.s_[1:win_size-1,  1:win_size-1, ...]
        quadrant_2 = np.s_[1:win_size-1,  win_size+1:win_size*2-1, ...]
        quadrant_3 = np.s_[win_size+1:win_size*2-1, 1:win_size-1, ...]
        quadrant_4 = np.s_[win_size+1:win_size*2-1, win_size+1:win_size*2-1, ...]

        IRLOS_cube_ = np.vstack([
            np.hstack([IRLOS_cube[quadrant_1], IRLOS_cube[quadrant_2]]),
            np.hstack([IRLOS_cube[quadrant_3], IRLOS_cube[quadrant_4]]),
        ])

        background, background_rms, mask = _get_background_(IRLOS_cube_.mean(axis=-1), 1.5)

        mask_edges = np.ones_like(IRLOS_cube_[...,0])
        mask_edges[mask_edges.shape[0]//2,:] = 0
        mask_edges[:,mask_edges.shape[0]//2] = 0
        mask_edges[mask_edges.shape[0]//2+1,:] = 0
        mask_edges[:,mask_edges.shape[0]//2+1] = 0
        mask_edges[0,:]  = 0
        mask_edges[:,0]  = 0
        mask_edges[-1,:] = 0
        mask_edges[:,-1] = 0

        IRLOS_cube_ -= background[..., None]
        IRLOS_cube_ *= mask[..., None] * mask_edges[..., None] # Remove noisy background pixels

        return IRLOS_cube_, win_size # [ADU], [pix]

    else:
        if verbose: print('No IRLOS cubes found')
        return None, None


# Get IR loop parameters
def GetIRLoopData(hdul_raw, exposure_start, exposure_end, verbose=False):
    # Try reading IRLOS info from the header
    if h+'AOS MAIN MODES IRLOS' in hdul_raw[0].header:
        LO_regime = hdul_raw[0].header[h+'AOS MAIN MODES IRLOS']
        if verbose: print(f'Found IRLOS info in FITS header: {LO_regime}')
        regime_df_ = pd.DataFrame({'command': [LO_regime]}, index=[exposure_start])
        IRLOS_record = ParseIRLOSregimes(regime_df_)
    else:
        if verbose: print(f'Not Found IRLOS regime in FITS header, scanning archive')
        IRLOS_df = check_cached_data(
            cache_path = TELEMETRY_CACHE / 'MUSE/IRLOS_regimes_df.csv',
            start_time = exposure_start,
            end_time   = exposure_end,
            fetch_function = FetchIRLOSlogsFromDatalab if datalab_available else None,
            parse_function = ParseIRLOSlogs,
            time_buffer_after  = pd.Timedelta(seconds=10),
            time_buffer_before = pd.Timedelta(minutes=120), # Search 2 hours before the exposure to see if IRLOS config was changed before OB
            verbose=verbose
        )
        if IRLOS_df.empty:
            IRLOS_record = None
        else:
            # Selecting the reguime which was set the last before the exposure started
            IRLOS_df_before_exposure = IRLOS_df[IRLOS_df.index <= exposure_start]

            if IRLOS_df_before_exposure.empty:
                print("Warning: No IRLOS configuration found before the exposure start time")
                IRLOS_record = None
            else:
                closest_timestamp = IRLOS_df_before_exposure.index[np.argmin(np.abs(IRLOS_df_before_exposure.index - exposure_start))]
                IRLOS_record = pd.DataFrame(IRLOS_df.loc[closest_timestamp]).T
              
    if IRLOS_record is None:
        print('No IRLOS information found. Setting to a default IRLOS regime')
        IRLOS_record = ParseIRLOSregimes(pd.DataFrame({'command': default_IRLOS_config}, index=[exposure_start]))

    return IRLOS_record


def GetIRLOSphotons(flux_ADU, LO_gain, LO_freq, convert_factor): #, [ADU], [1], [Hz], [e-/ADU]
    """ Computes the number of photons corresponding to mesured ADUs at M1 level"""
    QE = 1.0 # [photons/e-]
    transmission = 0.4 # [1]
    M1_area = (8-1.12)**2 * np.pi / 4 # [m^2]
    return flux_ADU / QE * convert_factor / LO_gain * LO_freq / M1_area * transmission # [photons/s/m^2]


def GetIRLOSdata(hdul_raw, hdul_cube, exposure_start, exposure_end, IRLOS_cube, verbose=False):
    # Read NGS loop parameters
    IRLOS_data_df = GetIRLoopData(hdul_raw, exposure_start, exposure_end, verbose)
    # Read NGS flux values
    try:
        # [ADU/frame/sub aperture] * 4 sub apertures if 2x2 mode
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
def ComputeLGSphotons(flux_LGS, HO_gain):
    conversion_factor = 18 # [e-/ADU]
    GALACSI_transmission = 0.31 #(with VLT) / 0.46 (without VLT), different from IRLOS!
    HO_rate = 1000 # [Hz]
    detector_DIT = (1-0.01982)*1e-3 # [s], 0.01982 [ms] for the frame transfer
    QE = 0.9 # [e-/photons]
    # Gain should always = 100 # Laser WFS gain
    num_subapertures = 1240
    M1_area = (8**2 - 1.12**2) * np.pi / 4 # [m^2]

    return flux_LGS * conversion_factor * num_subapertures  \
        / HO_gain / detector_DIT / QE / GALACSI_transmission \
        * HO_rate / M1_area # [photons/m^2/s]


def GetLGSdata(hdul_cube, exposure_start, exposure_end, fill_missing=False, verbose=False):
    # AOS LGSi FLUX: [ADU] Median flux in subapertures /
    # Median of the LGS{i} flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.
    
    # Read/fetch dataset with the LGS WFSs data
    LGS_flux_df = check_cached_data(
        cache_path = TELEMETRY_CACHE / 'MUSE/LGS_flux_df.csv',
        start_time = exposure_start,
        end_time   = exposure_end,
        fetch_function = FetchLGSfluxFromDatalab if datalab_available else None,
        time_buffer_after  = pd.Timedelta(minutes=1),
        time_buffer_before = pd.Timedelta(minutes=1),
        verbose=verbose
    )
    
    LGS_slopes_df = check_cached_data(
        cache_path = TELEMETRY_CACHE / 'MUSE/LGS_slopes_df.csv',
        start_time = exposure_start,
        end_time   = exposure_end,
        fetch_function = FetchLGSslopesFromDatalab if datalab_available else None,
        time_buffer_after  = pd.Timedelta(minutes=1),
        time_buffer_before = pd.Timedelta(minutes=1),
        verbose=verbose
    )

    # Select the relevant LGS data
    LGS_flux_df.rename(columns = { f'LGS{i} flux': f'LGS{i} flux, [ADU/frame]' for i in range(1,5) }, inplace=True)

    if LGS_flux_df.empty:
        if verbose: print('No LGS flux data found. ', end='')
        # Create at least something if there is no data
        LGS_flux_df = {'time': exposure_start}
  
        if fill_missing:
            if verbose: print('Filling with median guess...')
            flux_median  = 880.0
            works_default_flag = True
        else:
            if verbose: print('Skipping...')
            flux_median  = None
            works_default_flag = None
    
        for i in range(1,5):
            LGS_flux_df[f'LGS{i} flux, [ADU/frame]'] = flux_median  
            LGS_flux_df[f'LGS{i} works'] = works_default_flag
            
        LGS_flux_df = pd.DataFrame(LGS_flux_df, index=[-1])
        LGS_flux_df.set_index('time', inplace=True)
    # else:  
    #     LGS_flux_df.index = pd.to_datetime(LGS_flux_df.index).tz_convert('UTC')

    # Compute LGS photons
    HO_gains = np.array([hdul_cube[0].header[h+'AOS LGS'+str(i+1)+' DET GAIN'] for i in range(4)]) # must be the same value for all LGSs
    for i in range(1,5):
        LGS_flux_df.loc[:,f'LGS{i} photons, [photons/m^2/s]'] = ComputeLGSphotons(LGS_flux_df[f'LGS{i} flux, [ADU/frame]'], HO_gains[i-1]).round().astype('uint32')

    # Correct laser shutter state based on the data from the cube header
    for i in range(1,5):
        if f'{h}LGS{i} LASR{i} SHUT STATE' in hdul_cube[0].header:
            LGS_flux_df.loc[:,f'LGS{i} works'] = hdul_cube[0].header[f'{h}LGS{i} LASR{i} SHUT STATE'] == 0
        else:
            LGS_flux_df.loc[:,f'LGS{i} works'] = False

    # [pix] slopes RMS / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
    if LGS_slopes_df.empty:
        if verbose: print('No LGS WFS slopes data found. ', end='')
        # Create at least something if there is no data
        LGS_slopes_df = {'time': exposure_start}
        
        if fill_missing:
            if verbose: print('Filling with median guess...')
            slopes_filler = 0.22
        else:
            if verbose: print('Skipping...')
            slopes_filler = None
        
        for i in range(1,5):
            LGS_slopes_df[f'AOS.LGS{i}.SLOPERMSY'] = slopes_filler
            LGS_slopes_df[f'AOS.LGS{i}.SLOPERMSX'] = slopes_filler

        LGS_slopes_df = pd.DataFrame(LGS_slopes_df, index=[-1])
        LGS_slopes_df.set_index('time', inplace=True)
    # else:
    #     LGS_slopes_df.index = pd.to_datetime(LGS_slopes_df.index).tz_convert('UTC')

    return pd.concat([LGS_flux_df, LGS_slopes_df], axis=1)


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
    
    if   type(point) == tuple:        point = list(point)
    if   type(point) == list:         point = check_framework(point[0]).array(point)
    elif type(point) == pd.Series:    point = point.to_numpy()
    elif type(point) == torch.Tensor: point = point.cpu().numpy()
    else:
        raise ValueError('Point must be a list, tuple, ndarray, pandas.Series, torch.Tensor, or CuPy array')

    if hasattr(point, 'device'): # Detect CuPy array
        point = point.get()

    x, y = point[:2].astype('int')
    
    if radius == 0: # No averaging over a radius
        return data[:, y, x]
    
    y_min = max(0, y - radius)
    x_min = max(0, x - radius)
    y_max = min(data.shape[1], y + radius + 1)
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


def compute_wavelength_bins_old(λs, bad_wvls):
    λ_max  = λs.max()
    # Before the sodium filter
    # if verbose: print('Generating smart wavelength bins...')
    λ_bin = (bad_wvls[1][0]-bad_wvls[0][1])/3.0
    λ_bins_before = bad_wvls[0][1] + np.arange(4)*λ_bin
    bin_ids_before = [find_closest_λ(wvl, λs) for wvl in λ_bins_before]

    # After the sodium filter
    λ_bins_num    = (λ_max-bad_wvls[1][1]) / np.diff(λ_bins_before).mean()
    λ_bins_after  = bad_wvls[1][1] + np.arange(λ_bins_num+1)*λ_bin
    bin_ids_after = [find_closest_λ(wvl, λs) for wvl in λ_bins_after]
    bins_smart    = bin_ids_before + bin_ids_after
    λ_bins_smart  = λs[bins_smart]
    return λ_bins_smart, bins_smart


def compute_wavelength_bins(λs, bad_ranges, n_bins=None, target_width=None, min_ratio=0.1):
    """
    Build ~equal-width spectral bins over λs while excluding bad wavelength ranges.

    Parameters
    ----------
    λs : 1D array (monotonic increasing), sampling grid
    bad_ranges : array-like of shape (m, 2), [[λ_lo, λ_hi], ...], inclusive in wavelength units
    n_bins : int, optional
        Desired total number of bins over all *good* wavelengths.
    target_width : float, optional
        Target bin width in wavelength units (use this instead of n_bins).
    min_ratio : float,
        After snapping edges to λs, any bin narrower than (min_ratio * median_width)
        will be repaired (merged with a neighbor).
    xp : numpy-like module (e.g. numpy or cupy). If None, inferred from λs.__array_namespace__.

    Returns
    -------
    λ_bins : 1D array of bin edges in wavelength units (length = nbins+1)
    bin_ids : 1D array of indices into λs for those edges
    """

    if hasattr(bad_ranges, "device"):  bad_ranges = bad_ranges.get()
    if hasattr(λs, "device"):  λs = bad_ranges.get()
               
    # --- sort & normalize bad ranges and clip to [λmin, λmax] ---
    λmin, λmax = float(λs[0]), float(λs[-1])
    if bad_ranges.size:
        bad = bad_ranges.copy()
        bad = bad[np.argsort(bad[:, 0])]
        # merge overlapping/adjacent bad intervals
        merged = []
        cur_lo, cur_hi = float(bad[0,0]), float(bad[0,1])
        for lo, hi in bad[1:]:
            lo, hi = float(lo), float(hi)
            if lo <= cur_hi:  # overlap/adjacent
                cur_hi = max(cur_hi, hi)
            else:
                merged.append([cur_lo, cur_hi])
                cur_lo, cur_hi = lo, hi
        merged.append([cur_lo, cur_hi])
        bad = np.asarray(merged, dtype=float)
        # clip to global bounds
        bad[:,0] = np.clip(bad[:,0], λmin, λmax)
        bad[:,1] = np.clip(bad[:,1], λmin, λmax)
    else:
        bad = np.empty((0,2), dtype=float)

    # --- compute good intervals by subtracting bad from [λmin, λmax] ---
    good = []
    cur = λmin
    for lo, hi in bad:
        lo, hi = float(lo), float(hi)
        if cur < lo:
            good.append([cur, lo])
        cur = max(cur, hi)
    if cur < λmax:
        good.append([cur, λmax])
    good = np.asarray(good, dtype=float)

    # If everything is bad, return no bins
    if good.size == 0:
        return np.asarray([λmin, λmax]), np.asarray([0, λs.size-1])

    # --- decide target width / number of bins ---
    lengths = good[:,1] - good[:,0]
    total_len = float(lengths.sum())

    if target_width is not None and n_bins is not None:
        # trust n_bins, treat target_width as a hint
        target_width = total_len / int(n_bins)
    elif target_width is not None:
        n_bins = max(int(round(total_len / float(target_width))), 1)
        target_width = total_len / n_bins
    else:
        if n_bins is None:
            # heuristic: aim for ≈ one bin per ~50 samples of λs (tune as you like)
            samples = λs.size
            n_bins = max(int(round(samples / 50.0)), 1)
        target_width = total_len / int(n_bins)

    n_bins = int(max(n_bins, 1))

    # --- allocate bin counts per good interval proportionally to length ---
    # Start with floor allocation and distribute the remainder by largest fractional parts.
    ideal = lengths / target_width
    base = np.floor(ideal).astype(int)
    base = np.maximum(base, 1)  # at least 1 bin per good segment
    deficit = int(n_bins - int(base.sum()))

    if deficit != 0:
        frac = (ideal - base).astype(float)
        order = np.argsort(-frac) if deficit > 0 else np.argsort(frac)  # add to largest frac, remove from smallest
        i = 0
        while deficit != 0 and i < order.size:
            idx = int(order[i])
            new_val = base[idx] + (1 if deficit > 0 else -1)
            # keep at least 1 bin in each segment
            if new_val >= 1:
                base[idx] = new_val
                deficit += (-1 if deficit > 0 else 1)
            i += 1
    per_seg_bins = base  # bins per good segment; sum equals n_bins

    # --- make continuous edges in wavelength space, then snap to λs ---
    edges = []
    for (g0, g1), k in zip(good.tolist(), per_seg_bins.tolist()):
        # k bins -> k+1 edges
        seg_edges = np.linspace(g0, g1, k + 1)
        if edges:
            # drop the first edge to avoid duplication at segment joins
            seg_edges = seg_edges[1:]
        edges.append(seg_edges)
    λ_bins = np.concatenate(edges)

    # snap to nearest available λs and ensure strictly increasing indices
    def closest_idx(vals, grid):
        # vectorized nearest index
        # assumes grid is sorted
        import numpy as _np  # safe to use numpy ops on host for indexing math
        grid_np = _np.asarray(grid)
        vals_np = _np.asarray(vals)
        idx = _np.searchsorted(grid_np, vals_np)
        idx = _np.clip(idx, 1, grid_np.size - 1)
        left = grid_np[idx - 1]
        right = grid_np[idx]
        choose_right = (vals_np - left) > (right - vals_np)
        idx = idx + choose_right.astype(_np.int64) - 1
        return idx

    bin_ids = closest_idx(λ_bins, λs)
    # enforce strictly increasing and unique edges
    bin_ids = np.asarray(bin_ids, dtype=int)
    bin_ids = np.maximum.accumulate(bin_ids)
    # remove duplicates (zero-width after snapping)
    keep = np.concatenate([np.asarray([True]), np.diff(bin_ids) > 0])
    bin_ids = bin_ids[keep]
    λ_bins = λs[bin_ids]

    # --- repair: avoid any significantly thin bins after snapping ---
    widths = np.diff(λ_bins)
    if widths.size:
        med_w = float(np.median(widths))
        min_w = med_w * float(min_ratio)

        # merge any bin that falls below min_w with its thinner neighbor side
        # iterate until stable or single pass (single pass is usually enough)
        changed = True
        while changed and widths.size:
            changed = False
            too_thin = np.where(widths < min_w)[0]
            if too_thin.size:
                changed = True
                i = int(too_thin[0])
                # merge bin i with the neighbor that yields less distortion
                if i == 0:
                    # merge with right neighbor
                    λ_bins = np.delete(λ_bins, i + 1)
                elif i == widths.size - 1:
                    # merge with left neighbor
                    λ_bins = np.delete(λ_bins, i)
                else:
                    left_w = float(widths[i - 1])
                    right_w = float(widths[i + 1])
                    # prefer merging across the smaller adjacent width to smooth variation
                    if right_w < left_w:
                        λ_bins = np.delete(λ_bins, i + 1)  # remove right edge, merge with right
                    else:
                        λ_bins = np.delete(λ_bins, i)      # remove left edge, merge with left
                # recompute
                bin_ids = closest_idx(λ_bins, λs)
                bin_ids = np.asarray(bin_ids, dtype=int)
                bin_ids = np.maximum.accumulate(bin_ids)
                keep = np.concatenate([np.asarray([True]), np.diff(bin_ids) > 0])
                bin_ids = bin_ids[keep]
                λ_bins = λs[bin_ids]
                widths = np.diff(λ_bins)

    # Ensure that the upper limit of Na-filter is covered
    id_insert_binned = find_closest_λ(bad_ranges[-1][-1], λ_bins)
    id_insert_full   = find_closest_λ(bad_ranges[-1][-1], λs)

    λ_bins  = np.insert(λ_bins,  id_insert_binned, bad_ranges[-1][-1])
    bin_ids = np.insert(bin_ids, id_insert_binned, id_insert_full)

    return λ_bins, bin_ids


def GetSpectralCubeAndHeaderData(
    hdul_cube: fits.HDUList,
    show_plots: bool = False,
    crop_cube: bool | tuple[slice, slice] = False,
    extract_spectrum: bool = False,
    impaint: bool = False,
    wvl_bins: np.ndarray | None = None,
    verbose: bool = False
) -> tuple[dict, pd.DataFrame, dict]:

    # Extract spectral range information
    start_spaxel = hdul_cube[1].header['CRPIX3']
    num_λs = int(hdul_cube[1].header['NAXIS3']-start_spaxel+1)
    Δλ     = hdul_cube[1].header['CD3_3' ] / 10.0 # [nm]
    λ_min  = hdul_cube[1].header['CRVAL3'] / 10.0 # [nm]
    λs     = np.arange(num_λs) * Δλ + λ_min
    λ_max  = λs.max()

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

    # Pre-defined bad wavelengths ranges
    bad_wvls = np.array([[450, 478], [577, 606]])
    bad_ids  = np.array([find_closest_λ(wvl, λs) for wvl in bad_wvls.flatten()], dtype='int').reshape(bad_wvls.shape)
    valid_λs = np.ones_like(λs)

    for i in range(len(bad_ids)):
        valid_λs[bad_ids[i,0]:bad_ids[i,1]+1] = 0
        bad_wvls[i,:] = np.array([λs[bad_ids[i,0]], λs[bad_ids[i,1]+1]])

    if wvl_bins is not None:
        if verbose: print('Using pre-defined wavelength bins...')
        λ_bins_smart = wvl_bins
        bins_smart = np.array([find_closest_λ(wvl, λs) for wvl in λ_bins_smart.flatten()], dtype='int')
        
    else:
        if verbose: print('Computing wavelength bins...')
        λ_bins_smart, bins_smart = compute_wavelength_bins(λs, bad_wvls, n_bins=30, min_ratio=0.1)

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

    # Generate binned cubes
    if verbose: print('Generating binned data cubes...')
    data_binned = xp.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    std_binned  = xp.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    # Central λs at each spectral bin
    wavelengths = xp.zeros(len(λ_bins_smart)-1)
    flux        = xp.zeros(len(λ_bins_smart)-1)

    bad_layers     = [] # list of corrupted spectral slices (excluding the Na-filter)
    bins_to_ignore = [] # list of corrupted spectral slices (including Na-filter)

    # Loop over spectral bins
    if verbose:
        print(f'Processing spectral bins on {"GPU" if use_cupy else "CPU"}...')

    for bin in progress_bar( range(len(bins_smart)-1) ):
        chunk       = cube_data[ bins_smart[bin]:bins_smart[bin+1], ROI[0], ROI[1] ]
        wvl_chunck  = λs[bins_smart[bin]:bins_smart[bin+1]]
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

        data_binned[bin,:,:] = xp.nansum(chunk, axis=0)
        std_binned [bin,:,:] = xp.nanstd(chunk, axis=0)
        wavelengths[bin]     = xp.nanmean(xp.array(wvl_chunck)) # central wavelength for this bin
        flux[bin]            = xp.nanmean(xp.array(flux_chunck))

    # Send back to RAM
    if use_cupy:
        spectrum    = xp.asnumpy(spectrum)
        data_binned = xp.asnumpy(data_binned)
        std_binned  = xp.asnumpy(std_binned)
        wavelengths = xp.asnumpy(wavelengths)
        flux        = xp.asnumpy(flux)
        white       = xp.asnumpy(white)

    # Generate spectral plot (if applies)
    fig_handler = None
    if show_plots and extract_spectrum:
        if verbose: print("Generating spectral plot...")
        # Initialize arrays using np.zeros_like to match λs shape
        Rs, Gs, Bs = np.zeros_like(λs), np.zeros_like(λs), np.zeros_like(λs)

        for i, λ in enumerate(λs):
            Rs[i], Gs[i], Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

        # Mask out the bad wavelength ranges
        spectrum_valid = valid_λs * spectrum.copy()

        # Scale the RGB arrays by valid_λs and spectrum
        Rs = Rs * spectrum_valid / np.median(spectrum_valid)
        Gs = Gs * spectrum_valid / np.median(spectrum_valid)
        Bs = Bs * spectrum_valid / np.median(spectrum_valid)

        # Create a color array by stacking and transposing appropriately
        colors = np.dstack([np.vstack([Rs, Gs, Bs])]*2).transpose(2,1,0)

        # Plotting using matplotlib
        fig_handler = plt.figure(dpi=200)
        plt.imshow(colors, extent=[λs.min(), λs.max(), 0, spectrum_valid.max()])
        plt.ylim(0, spectrum_valid.max())
        plt.vlines(λ_bins_smart, 0, spectrum_valid.max(), color='white')  # draw bins borders
        plt.plot(λs, spectrum_valid, linewidth=2.0, color='white')
        plt.plot(λs, spectrum_valid, linewidth=0.5, color='blue')
        plt.xlabel(r"$\lambda$, [nm]")
        plt.ylabel(r"$\left[ 10^{-20} \frac{erg}{s \cdot cm^2 \cdot \AA} \right]$")
        ax = plt.gca()
        ax.set_aspect(np.ptp(λs)/np.max(spectrum_valid)/2.5) # 1:2.5 aspect ratio


    data_binned = np.delete(data_binned, bins_to_ignore, axis=0)
    std_binned  = np.delete(std_binned,  bins_to_ignore, axis=0)
    wavelengths = np.delete(wavelengths, bins_to_ignore)
    flux        = np.delete(flux,        bins_to_ignore)

    if verbose:
        print(str(len(bad_layers))+'/'+str(cube_data.shape[0]), '('+str(xp.round(len(bad_layers)/cube_data.shape[0],2))+'%)', 'slices are corrupted')

    del cube_data

    if use_cupy:
        if verbose: print("Deallocating GPU memory...")
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # Collect the telemetry from the header
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

    derot_ang += 360 if derot_ang < -180 else 0
    derot_ang -= 360 if derot_ang >  180 else 0

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
        'Pupil angle':   pupil_angle_func(parang),
        'Derot. angle':  derot_ang
    }

    science_target = {
        'Science target': hdul_cube[0].header[h+'OBS TARG NAME'],
        'RA (science)':   hdul_cube[0].header['RA'],
        'DEC (science)':  hdul_cube[0].header['DEC'],
    }

    from_header = {
        'Pixel scale (science)': hdul_cube[0].header[h+'OCS IPS PIXSCALE']*1000, #[mas/pixel]
        'Airmass':              (hdul_cube[0].header[h+'TEL AIRM START'] + hdul_cube[0].header[h+'TEL AIRM END']) / 2.0,
        'Seeing (header)':      (hdul_cube[0].header[h+'TEL AMBI FWHM START'] + hdul_cube[0].header[h+'TEL AMBI FWHM END']) / 2.0,
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

    multispectral_cubes = {
        'cube':  data_binned,
        'std':   std_binned,
        'white': white[ROI],
    }

    data = science_target | observation | NGS_target | from_header

    data_df = pd.DataFrame(data, index=[0])
    data_df.rename(columns={'Date-obs': 'time'}, inplace=True)
    data_df.set_index('time', inplace=True)
    data_df.index = pd.to_datetime(data_df.index).tz_localize('UTC')
    if verbose: print('Finished processing spectral cube!')

    return multispectral_cubes, data_df, spectral_info


# ------------------------------ Raw header realm ------------------------------------------
def extract_countables_df(hdul, table_name, entry_pattern):
    """ Function to work with header entries which are like SOMETHING_1, SOMETHING_2, ...SOMETHING_N"""
    keys_list = [x for x in hdul[table_name].header.values()]
    max_countables_num = 50
    
    if 'Sec' in keys_list:
        timestamps = pd.to_datetime(time_from_sec_usec(*read_secs_usecs(hdul, table_name)))
    else:
        timestamps = pd.to_datetime(time_from_str(hdul[table_name].data['TIME_STAMP'])).tz_localize('UTC')

    values_dict = {}

    for i in range (0, max_countables_num):
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
    '''Extracts data from the header of raw MUSE fits file'''
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
            'IA_FWHMLIN',
            'IA_FWHMLINOBS',
            'IA_FWHM',
            'SLODAR_FRACGL_300',
            'SLODAR_FRACGL_500',
            'ASM_WINDDIR_10',
            'ASM_WINDDIR_30',
            'ASM_WINDSPEED_10',
            'ASM_WINDSPEED_30',
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


def create_flat_dataframe(df):
    """" Computes singular values for a specific MUSE NFM sample if there are several values per dataset entry. """
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


def EstimateIRLOSphase(cube: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimates the IRLOS phase using the external LIFT module that implements phase diversity.

    Parameters
    ----------
    cube : np.ndarray
        Input cube containing IRLOS data

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        If successful, returns (OPD_subap, OPD_aperture, PSF_estim)
        If unsuccessful, returns None
    """
    try:
        # Add LIFT dependencies to the path
        LIFT_PATH = Path(project_settings["LIFT_path"])
        dependency_paths = [
            '',
            'DIP',
            'LIFT',
            'LIFT/modules',
            'experimental'
        ]

        for path in dependency_paths:
            full_path = f'{LIFT_PATH}/{path}/..'
            if full_path not in sys.path:
                sys.path.append(full_path)

        # Import the LIFT estimator function
        from LIFT_full.experimental.IRLOS_2x2_function import estimate_2x2
        return estimate_2x2(cube) # Run the estimation results

    except Exception as e:
        logging.error(f'Cannot import or use LIFT module: {e}')
        return None


#%%
def ProcessMUSEcube(
        path_raw: str | os.PathLike,
        path_cube: str | os.PathLike,
        crop: bool | tuple[slice, slice] = False,
        estimate_IRLOS_phase: bool = False,
        impaint_bad_pixels: bool = False,
        extract_spectrum: bool = False,
        wavelength_bins: int | np.ndarray | list | None = None,
        plot_spectrum: bool = False,
        fill_missing_values: bool = False,
        verbose: bool = False
    ) -> tuple[dict, str]:

    cube_name = os.path.basename(path_cube)

    hdul_raw  = fits.open(path_raw)
    hdul_cube = fits.open(path_cube)

    exposure_start = pd.to_datetime(time_from_str(hdul_cube[0].header['DATE-OBS'])).tz_localize('UTC')
    exposure_end   = pd.to_datetime(exposure_start + datetime.timedelta(seconds=hdul_cube[0].header['EXPTIME']))

    if verbose: print(f'>>>>> Getting IRLOS data...')
    IRLOS_cube, win = GetIRLOScube(hdul_raw, verbose)
    IRLOS_data_df   = GetIRLOSdata(hdul_raw,  hdul_cube, exposure_start, exposure_end, IRLOS_cube, verbose)
    LGS_data_df     = GetLGSdata  (hdul_cube, exposure_start, exposure_end, fill_missing_values, verbose)

    if win is not None:
        IRLOS_data_df['window'] = win

    if verbose: print(f'>>>>> Reading data from reduced MUSE spectral cube...')
    MUSE_images, MUSE_data_df, spectral_info = GetSpectralCubeAndHeaderData(
        hdul_cube,
        show_plots = plot_spectrum,
        crop_cube  = crop,
        extract_spectrum = extract_spectrum,
        impaint = impaint_bad_pixels,
        wvl_bins = wavelength_bins,
        verbose = verbose
    )
    if verbose: print(f'>>>>> Reading data from raw MUSE file ...')
    Cn2_data_df, atm_data_df, asm_data_df   = GetRawHeaderData(hdul_raw)
    if verbose: print(f'>>>>> Getting data from ESO archive...')
    asm_df, massdimm_df, dimm_df, slodar_df = FetchFromESOarchive(exposure_start, exposure_end, minutes_delta=1)

    hdul_cube.close()
    hdul_raw.close()

    # Estimate the IRLOS phasecube from 2x2 PSFs
    OPD_subap, OPD_aperture, PSF_estim = (None,)*3
    if IRLOS_cube is not None and estimate_IRLOS_phase:
        if verbose: print(f'>>>>> Estimating IRLOS phase via phase diversity...')
        res = EstimateIRLOSphase(IRLOS_cube)
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

    if verbose: print(f'>>>>> Packing data...')

    # Create a flat DataFrame with temporal dimension compressed
    all_df = pd.concat([
        IRLOS_data_df, LGS_data_df, MUSE_data_df, Cn2_data_df, atm_data_df,
        asm_data_df, asm_df, massdimm_df, dimm_df, slodar_df
    ])
    all_df.sort_index(inplace=True)

    flat_df = create_flat_dataframe(all_df)
    flat_df.insert(0, 'time', exposure_start)
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

    valid_mask = torch.tensor(valid_mask, dtype=default_torch_type, device=device).unsqueeze(0)

    if os.path.exists(cache_path):
        if verbose: print('Loading existing cached data cube...')
        with open(cache_path, 'rb') as f:
            data_cached, _ = pickle.load(f)
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
            estimate_IRLOS_phase = False,
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
    # Δλ_binned = np.median(np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])]))
    Δλ_binned = np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])])

    if hasattr(λ_max, 'item'): # To compensate for a small error in the data reduction routine
        λ_max = λ_max.item()

    λ_full = np.linspace(λ_min, λ_max, np.round((λ_max-λ_min)/Δλ_full+1).astype('int'))
    assert len(λ_full) == cube_full.shape[0]

    cube_binned = torch.tensor(data_cached['images']['cube'], dtype=default_torch_type, device=device)
    cube_binned = cube_binned.squeeze() * valid_mask

    # Correct the flux to match MUSE cube
    cube_binned = cube_binned * (cube_full.sum(axis=0).max() /  cube_binned.sum(axis=0).max())

    # Extract config file and update it
    model_config, cube_binned = InitNFMConfig(data_cached, cube_binned, device=device, convert_config=True)
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
    λ_sparse  =  λ_binned[..., ids_λ_sparse]
    Δλ_binned = Δλ_binned[..., ids_λ_sparse]
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


def InitNFMConfig(sample, PSF_data=None, wvl_ids=None, device=device, convert_config=True):
    if PSF_data is None:
        PSF_data = sample['images']['cube']
        
    FoV_science = int(PSF_data.shape[-1])
    # N_wvl_PSF   = int(PSF_data.shape[0] if PSF_data.ndim == 3 else PSF_data.shape[1])

    wvls_ = (sample['spectral data']['wvls binned']*1e-9).tolist()
    wvls  = wvls_ if wvl_ids is None else np.array([wvls_[i] for i in wvl_ids])
        
    config_manager = ConfigManager()
    config_file    = ParameterParser(DATA_FOLDER / 'parameter_files/muse_ltao.ini').params

    h_GL = 2000 # Assumed separation altitude betwee ground and high-altitude layers

    try:
        # For NFM it's save to assume gound layer to be below 2 km, for WFM it's lower than that
        Cn2_weights = np.array([sample['All data'][f'CN2_FRAC_ALT{i}'].item() for i in range(1, 9)])
        altitudes   = np.array([sample['All data'][f'ALT{i}'].item() for i in range(1, 9)])*100 # in meters

        Cn2_weights[Cn2_weights > 1] = 0.0
        Cn2_weights[Cn2_weights < 0] = 0.0

        Cn2_weights_GL = Cn2_weights[altitudes < h_GL]
        altitudes_GL   = altitudes  [altitudes < h_GL]

        GL_frac  = Cn2_weights_GL.sum()  # Ground layer fraction   
        Cn2_w_GL = np.interp(h_GL, altitudes, Cn2_weights)
        
    except:
        GL_frac = 0.9

    config_file['NumberSources'] = 1

    config_file['telescope']['TelescopeDiameter'] = 8.0
    config_file['telescope']['ZenithAngle'] = [90.0 - sample['MUSE header data']['Tel. altitude'].item()]
    config_file['telescope']['Azimuth']     = [sample['MUSE header data']['Tel. azimuth'].item()]
    
    if 'PupilAngle' not in config_file['telescope']:
        config_file['telescope']['PupilAngle'] = 0.0
    else:
        try:
            config_file['telescope']['PupilAngle'] = sample['All data']['Pupil angle'].item()
        except (KeyError, TypeError):
            config_file['telescope']['PupilAngle'] = 0.0
    
    
    if sample['Raw Cn2 data'] is not None:
        config_file['atmosphere']['L0']  = [sample['All data']['L0Tot'].item()]
    else:
        config_file['atmosphere']['L0']  = [config_file['atmosphere']['L0']]
                
    config_file['atmosphere']['Seeing'] = [sample['MUSE header data']['Seeing (header)'].item()]
    config_file['atmosphere']['Cn2Weights'] = [[GL_frac, 1-GL_frac]]
    config_file['atmosphere']['Cn2Heights'] = [[0, h_GL]]

    config_file['atmosphere']['WindSpeed']     = [[sample['MUSE header data']['Wind speed (header)'].item(),]*2]
    config_file['atmosphere']['WindDirection'] = [[sample['MUSE header data']['Wind dir (header)'].item(),]*2]
    config_file['sources_science']['Wavelength'] = wvls

    config_file['sources_LO']['Wavelength'] = (1215 + 1625)/2.0 * 1e-9 # Mean  approx. wavelengths between J and H

    config_file['sensor_science']['PixelScale'] = sample['MUSE header data']['Pixel scale (science)'].item()
    config_file['sensor_science']['FieldOfView'] = FoV_science

    try:
        LGS_ph = np.array([sample['All data'][f'LGS{i} photons, [photons/m^2/s]'].item() / 1240e3 for i in range(1,5)])
        LGS_ph[LGS_ph < 1] = np.mean(LGS_ph)
        LGS_ph = [LGS_ph.tolist()]
    except:
        LGS_ph = [[2000,]*4]
        
    config_file['sensor_HO']['NumberPhotons']  = LGS_ph
    config_file['sensor_HO']['SizeLenslets']   = config_file['sensor_HO']['SizeLenslets'][0]
    config_file['sensor_HO']['NumberLenslets'] = config_file['sensor_HO']['NumberLenslets'][0]
    # config_file['sensor_HO']['NoiseVariance'] = 4.5

    IRLOS_ph_per_subap_per_frame = sample['IRLOS data']['IRLOS photons (cube), [photons/s/m^2]'].item()
    if IRLOS_ph_per_subap_per_frame is not None:
        IRLOS_ph_per_subap_per_frame /= sample['IRLOS data']['frequency'].item() / 4

    config_file['sensor_LO']['PixelScale']    = sample['IRLOS data']['plate scale, [mas/pix]'].item()
    config_file['sensor_LO']['NumberPhotons'] = [IRLOS_ph_per_subap_per_frame]
    config_file['sensor_LO']['SigmaRON']      = sample['IRLOS data']['RON, [e-]'].item()
    config_file['sensor_LO']['Gain']          = [sample['IRLOS data']['gain'].item()]

    config_file['RTC']['SensorFrameRate_LO'] = [sample['IRLOS data']['frequency'].item()]
    config_file['RTC']['SensorFrameRate_HO'] = [config_file['RTC']['SensorFrameRate_HO']]

    config_file['RTC']['LoopDelaySteps_HO']  = [config_file['RTC']['LoopDelaySteps_HO']]
    config_file['RTC']['LoopGain_HO']        = [config_file['RTC']['LoopGain_HO']]

    # config_file['DM']['DmPitchs'] = [config_file['DM']['DmPitchs'][0]*1.25]
    config_file['DM']['DmPitchs'][0] = 0.22

    config_file['sensor_HO']['ClockRate'] = np.mean([config_file['sensor_HO']['ClockRate']])
    
    if convert_config:
        config_manager.Convert(config_file, framework='pytorch', device=device)

    return config_file


def RotatePSF(PSF_data, angle):
    if isinstance(PSF_data, torch.Tensor):
        PSF_data_ = PSF_data.cpu().numpy()
        torch_flag = True
    else:
        PSF_data_ = PSF_data
        torch_flag = False

    PSF_data_rotated = np.zeros_like(PSF_data_)
    
    for i in range(PSF_data_.shape[0]):
        PSF_data_rotated[i,...] = rotate(PSF_data_[i,...], angle, reshape=False)
    
    if torch_flag:
        PSF_data_rotated = torch.tensor(PSF_data_rotated, dtype=default_torch_type).unsqueeze(0).to(PSF_data.device)
        
    return PSF_data_rotated


# %%
def filter_values(df: pd.DataFrame) -> pd.DataFrame:
     
    df.loc[df['SLODAR_FRACGL_300'] < 1e-12, 'SLODAR_FRACGL_300'] = np.nan
    df.loc[df['SLODAR_FRACGL_500'] < 1e-12, 'SLODAR_FRACGL_500'] = np.nan
    df.loc[df['SLODAR_TOTAL_CN2']  < 1e-12, 'SLODAR_TOTAL_CN2'] = np.nan
    df.loc[df['IA_FWHM'] > 5, 'IA_FWHM'] = np.nan
    df.loc[df['ASM_RFLRMS'] > 0.25, 'ASM_RFLRMS'] = np.nan
    df.loc[df['Strehl (header)'] < 0.01, 'Strehl (header)'] = np.nan
    df.loc[df['LGS3 flux, [ADU/frame]'] < 1e-12, 'LGS3 flux, [ADU/frame]'] = np.nan
    df.loc[df['LGS3 photons, [photons/m^2/s]'] < 1e-12, 'LGS3 photons, [photons/m^2/s]'] = np.nan
    df.loc[df['IRLOS photons, [photons/s/m^2]'] < 1e-12, 'IRLOS photons, [photons/s/m^2]'] = np.nan

    for i in range(1, 9):
        entry = f'ALT{i}'
        df.loc[df[entry] > 100, entry] = np.nan
        df.loc[df[entry] < 1e-16, entry] = np.nan
        

        entry = f'L0_ALT{i}'
        df.loc[df[entry] > 100, entry] = np.nan
        df.loc[df[entry] < 0, entry] = np.nan

        entry = f'CN2_FRAC_ALT{i}'
        df.loc[df[entry] >= 1, entry] = np.nan
        # df.loc[df[entry] < 1e-10, entry] = np.nan
        df.loc[df[entry] < 0, entry] = np.nan

        entry = f'CN2_ALT{i}'
        # df.loc[df[entry] > 1e-11, entry] = np.nan
        df.loc[df[entry] > 1e-10, entry] = np.nan
        df.loc[df[entry] < 1e-16, entry] = np.nan


    for i in range(0, 7):
        entry = f'MASS_TURB{i}'
        df.loc[df[entry] < 1e-16, entry] = np.nan

    for i in range(1, 5):   
        entry = f'LGS{i}_STREHL'
        df.loc[df[entry] < 0.01, entry] = np.nan
        
        entry = f'LGS{i}_L0'
        df.loc[df[entry] > 500, entry] = np.nan
        
        df.loc[df[entry:=f'AOS.LGS{i}.SLOPERMSX'] < 1e-9, entry] = np.nan
        df.loc[df[entry:=f'AOS.LGS{i}.SLOPERMSY'] < 1e-9, entry] = np.nan
        df.loc[df[entry:=f'LGS{i}_TURVAR_TOT'] < 1e-9, entry] = np.nan
        df.loc[df[entry:=f'LGS{i}_TUR_GND'] < 1e-9, entry] = np.nan
    
    df = df.replace([np.inf, -np.inf], np.nan)

    # numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # # Handle each numeric column
    # for col in numeric_cols:
    #     # Skip columns that are mostly NaN
    #     if df[col].isna().sum() > 0.8 * len(df):
    #         continue

    #     # Calculate mean and std, ignoring NaN values
    #     mean_val = df[col].mean(skipna=True)
    #     std_val = df[col].std(skipna=True)

    #     # Skip if std is zero or NaN
    #     if std_val == 0 or pd.isna(std_val):
    #         continue

    #     # Identify outliers (5-sigma threshold)
    #     upper_bound = mean_val + 3 * std_val
    #     lower_bound = mean_val - 3 * std_val

    #     # Replace outliers with NaN
    #     df.loc[(df[col] > upper_bound) | (df[col] < lower_bound), col] = np.nan

    return df

# Dataset cleaning
def prune_columns(df):
    
    df_ = df.copy()
    
    df_['LGS_R0']         = df_[[f'LGS{i}_R0'         for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_SEEING']     = df_[[f'LGS{i}_SEEING'     for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_STREHL']     = df_[[f'LGS{i}_SEEING'     for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_TURVAR_RES'] = df_[[f'LGS{i}_TURVAR_RES' for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_TURVAR_TOT'] = df_[[f'LGS{i}_TURVAR_TOT' for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_TUR_ALT']    = df_[[f'LGS{i}_TUR_ALT'    for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_TUR_GND']    = df_[[f'LGS{i}_TUR_GND'    for i in range(1,5)]].median(axis=1, skipna=True)
    df_['LGS_FWHM_GAIN']  = df_[[f'LGS{i}_FWHM_GAIN'  for i in range(1,5)]].median(axis=1, skipna=True)
    
    df_['LGS photons, [photons/m^2/s]'] = \
        df_[[f'LGS{i} photons, [photons/m^2/s]' for i in range(1,5)]].median(axis=1, skipna=True)
    
    # df['LGS_R0']         = df[['LGS1_R0', 'LGS2_R0', 'LGS3_R0', 'LGS4_R0']].mean(axis=1, skipna=True)
    # df['LGS_SEEING']     = df[['LGS1_SEEING', 'LGS2_SEEING', 'LGS3_SEEING', 'LGS4_SEEING']].mean(axis=1, skipna=True)
    # df['LGS_STREHL']     = df[['LGS1_SEEING', 'LGS2_SEEING', 'LGS3_SEEING', 'LGS4_SEEING']].mean(axis=1, skipna=True)
    # df['LGS_TURVAR_RES'] = df[['LGS1_TURVAR_RES', 'LGS2_TURVAR_RES', 'LGS3_TURVAR_RES', 'LGS4_TURVAR_RES']].mean(axis=1, skipna=True)
    # df['LGS_TURVAR_TOT'] = df[['LGS1_TURVAR_TOT', 'LGS2_TURVAR_TOT', 'LGS3_TURVAR_TOT', 'LGS4_TURVAR_TOT']].mean(axis=1, skipna=True)
    # df['LGS_TUR_ALT']    = df[['LGS1_TUR_ALT', 'LGS2_TUR_ALT', 'LGS3_TUR_ALT', 'LGS4_TUR_ALT']].mean(axis=1, skipna=True)
    # df['LGS_TUR_GND']    = df[['LGS1_TUR_GND', 'LGS2_TUR_GND', 'LGS3_TUR_GND', 'LGS4_TUR_GND']].mean(axis=1, skipna=True)
    # df['LGS_FWHM_GAIN']  = df[['LGS1_FWHM_GAIN', 'LGS2_FWHM_GAIN', 'LGS3_FWHM_GAIN', 'LGS4_FWHM_GAIN']].mean(axis=1, skipna=True)
    
    for i in range(1, 5):
        df_[f'AOS.LGS{i}.SLOPERMS'] = (df_[f'AOS.LGS{i}.SLOPERMSX']**2 + df_[f'AOS.LGS{i}.SLOPERMSY']**2)**0.5
        df_[f'LGS{i}_SLOPERMS']     = (df_[f'LGS{i}_SLOPERMSX']**2 + df_[f'LGS{i}_SLOPERMSY']**2)**0.5

    df_['AOS.LGS.SLOPERMS'] = df_[[f'AOS.LGS{i}.SLOPERMS' for i in range(1, 5)]].median(axis=1, skipna=True)
    df_['LGS_SLOPERMS']     = df_[[f'LGS{i}_SLOPERMS' for i in range(1, 5)]].median(axis=1, skipna=True)
    
    # df['MASS_TURB']        = df[[f'MASS_TURB{i}' for i in range(0, 7)]].sum(axis=1, skipna=True)
    turb_df = df_[[f'MASS_TURB{i}' for i in range(0,7)]].copy()
    turb_df['ground_layer'] = turb_df['MASS_TURB0']
    turb_df['upper_layers'] = turb_df[[f'MASS_TURB{i}' for i in range(1,7)]].sum(axis=1, skipna=True)
    turb_df['MASS_TURB total'] = turb_df['ground_layer'] + turb_df['upper_layers']
    turb_df['MASS_TURB ratio'] = turb_df['ground_layer'] / turb_df['MASS_TURB total']
    df_['MASS_TURB total'] = turb_df['MASS_TURB total'].copy()
    df_.loc[df_['MASS_TURB total'] < 1e-16, 'MASS_TURB total'] = np.nan
    df_['MASS_TURB ratio'] = turb_df['MASS_TURB ratio'].copy()
    
    df_['Pupil angle'] = df_['Pupil angle'].map(lambda x: x % 360)
    df_[[f'AOS.LGS{i}.SLOPERMS' for i in range(1, 5)]].median(axis=1)


    def compute_ground_layer_fraction(df: pd.DataFrame, h_GL: float = 2000) -> pd.DataFrame:
        """
        Compute Cn2 fraction below a given altitude (ground layer).

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with CN2_FRAC_ALT{i} and ALT{i} columns (i = 1..8).
        h_GL : float, optional
            Ground layer altitude threshold in meters. Default = 2000.

        Returns
        -------
        df : pandas.DataFrame
            Copy of input dataframe with a new column: 'Cn2 fraction below {h_GL}m'.
        """
        GL_fracs = []

        for j in range(len(df)):
            Cn2_weights = np.array([df.iloc[j][f'CN2_FRAC_ALT{i}'] for i in range(1, 9)])

            if not np.isnan(Cn2_weights).all():
                altitudes = np.array([df.iloc[j][f'ALT{i}'] for i in range(1, 9)]) * 100 # Convert altitudes to meters
                Cn2_weights_GL = Cn2_weights[altitudes < h_GL] # Select values below threshold
                GL_frac = np.nansum(Cn2_weights_GL)
                
                if GL_frac > 1.0+1e-10 or GL_frac < 0.0:  # Sanity check
                    GL_frac = np.nan

                GL_fracs.append(GL_frac)
            else:
                GL_fracs.append(np.nan)

        df = df.copy()
        df[f'Cn2 fraction below {h_GL}m'] = GL_fracs
        return df

    # Compute Cn² fractions for 2 and 10 kms
    df_ = compute_ground_layer_fraction(df_, h_GL=2000)
    # df_ = compute_ground_layer_fraction(df_, h_GL=10000)

    entries_to_drop = [
        # These features are just collapsed to a median one instead of FETURE1, FEATURE2...
        *[f'LGS{i}_R0' for i in range(1, 5)],
        *[f'LGS{i}_L0' for i in range(1, 5)],
        *[f'LGS{i}_SEEING' for i in range(1, 5)],
        *[f'LGS{i}_STREHL' for i in range(1, 5)],
        *[f'LGS{i} works'  for i in range(1, 5)],
        *[f'LGS{i} flux, [ADU/frame]' for i in range(1, 5)],
        *[f'LGS{i} photons, [photons/m^2/s]' for i in range(1, 5)],
        *[f'AOS.LGS{i}.SLOPERMSX' for i in range(1, 5)],
        *[f'AOS.LGS{i}.SLOPERMSY' for i in range(1, 5)],
        *[f'AOS.LGS{i}.SLOPERMS'  for i in range(1, 5)],
        *[f'LGS{i}_SLOPERMSX'  for i in range(1, 5)],
        *[f'LGS{i}_SLOPERMSY'  for i in range(1, 5)],
        *[f'LGS{i}_SLOPERMS'   for i in range(1, 5)],
        *[f'LGS{i}_TURVAR_RES' for i in range(1, 5)],
        *[f'LGS{i}_TURVAR_TOT' for i in range(1, 5)],
        *[f'LGS{i}_TUR_ALT'    for i in range(1, 5)],
        *[f'LGS{i}_TUR_GND'    for i in range(1, 5)],
        *[f'LGS{i}_FWHM_GAIN'  for i in range(1, 5)],
        
        *[f'CN2_FRAC_ALT{i}' for i in range(1, 9)],
        *[f'CN2_ALT{i}' for i in range(1, 9)],
        *[f'ALT{i}' for i in range(1, 9)],
        *[f'L0_ALT{i}' for i in range(1, 9)],
        *[f'SLODAR_CNSQ{i}' for i in range(2, 9)],
        *[f'MASS_TURB{i}' for i in range(0, 7)],
        
        # These features miss most of the values
        'SLODAR_TOTAL_CN2',
        'SLODAR_FRACGL_300',
        'SLODAR_FRACGL_500',
        'ADUs (from 2x2)',
        'NGS mag',
        'NGS mag (from 2x2)',
        'Cn2 above UTs [10**(-15)m**(1/3)]',
        'Cn2 fraction below 300m',
        'Cn2 fraction below 500m',
        'Surface layer profile [10**(-15)m**(1/3)]',
        'Air Temperature at 30m [C]',
        'Seeing ["]',
        'scale',
        'AIRMASS',
        'seeingTot',
        'L0Tot',
        'r0Tot',
        'ADUs (header)',
        'NGS mag (header)',
        'Strehl (header)',
        'IRLOS photons (cube), [photons/s/m^2]',
        'IRLOS photons, [photons/s/m^2]',
        'Pixel scale (science)',
        
        # These features has strong correlaions with other ones
        'DIMM_SEEING',
        'Wind Speed at 30m [m/s]',
        'R0',
        'LGS_R0',
        'DIMM Seeing ["]',
        'MASS_TURB ratio',
        'ASM_WINDDIR_10',
        'ASM_WINDDIR_30',
        'ASM_WINDSPEED_10',
        'ASM_WINDSPEED_30',
        'ASM_RFLRMS',
        'Wind Direction at 30m (0/360) [deg]',
        'MASS-DIMM Cn2 fraction at ground',
        'AOS.LGS.SLOPERMS',
        'MASS-DIMM Seeing ["]',
        'MASS-DIMM Tau0 [s]',
        'Airmass',
        'IA_FWHM',
        'IA_FWHMLIN',
        'LGS_SEEING',
        'LGS_TURVAR_TOT',
        'LGS_TUR_GND',
        'plate scale, [mas/pix]',
        'conversion, [e-/ADU]',
        'RON, [e-]',
        'RA (science)',
        'DEC (science)',

        # These features are non-numeric
        'Filename',
        'Science target',
        'name',
        'Bad quality',
        'Corrupted',
        'Good quality',
        'Has streaks',
        'Low AO errs',
        'Medium quality',
        'Non-point',
        'time'
        
    ]

    # Remove columns with specific names
    entries_to_drop_in_df = [entry for entry in entries_to_drop if entry in df_.columns]
    df_.drop(columns=entries_to_drop_in_df, inplace=True)
    
    return df_


def create_normalizing_transforms(df):
    from data_processing.normalizers import Invert, Gauss, Logify, YeoJohnson, Uniform, CreateTransformSequence, TransformSequence

    entry_data = [
        ('name', []),
        ('time', []),
        ('Filename',     []),
        ('Bad quality',  []),
        ('Corrupted',    []),
        ('Good quality', []),
        ('Has streaks',  []),
        ('Low AO errs',  []),
        ('Non-point',    []),
        ('Science target', []),
        ('Medium quality', []),

        ('MASS_TURB',        [Logify, Gauss]),
        ('LGS_FWHM_GAIN',    [YeoJohnson, Gauss]),
        ('AOS.LGS.SLOPERMS', [YeoJohnson, Gauss]),
        ('LGS_SLOPERMS',     [YeoJohnson, Gauss]),
        ('LGS_R0',           [YeoJohnson, Gauss]),
        ('LGS_SEEING',       [YeoJohnson, Gauss]),
        ('LGS_STREHL',       [YeoJohnson, Gauss]),
        ('LGS_TURVAR_RES',   [YeoJohnson, Gauss]),
        ('LGS_TURVAR_TOT',   [YeoJohnson, Gauss]),
        ('LGS_TUR_ALT',      [YeoJohnson, Gauss]),
        ('DIMM Seeing ["]',  [YeoJohnson, Gauss]),
        ('MASS-DIMM Seeing ["]', [YeoJohnson, Gauss]),
        ('MASS-DIMM Tau0 [s]',   [YeoJohnson, Gauss]),
        ('MASS-DIMM Turb Velocity [m/s]', [YeoJohnson, Gauss]),
        ('Wind Speed at 30m [m/s]',       [YeoJohnson, Gauss]),
        ('Free Atmosphere Seeing ["]',    [YeoJohnson, Gauss]),
        ('MASS Tau0 [s]',       [YeoJohnson, Gauss]),
        ('ASM_WINDSPEED_10',    [YeoJohnson, Gauss]),
        ('ASM_WINDSPEED_30',    [YeoJohnson, Gauss]),
        ('Seeing (header)',     [YeoJohnson, Gauss]),
        ('Tau0 (header)',       [YeoJohnson, Gauss]),
        ('Wind dir (header)',   [Uniform]),
        ('Pupil angle',         [YeoJohnson, Gauss]),
        ('R0',                  [YeoJohnson, Gauss]),
        ('DIMM_SEEING',         [YeoJohnson, Gauss]),
        ('IA_FWHMLIN',          [YeoJohnson, Gauss]),
        ('IA_FWHMLINOBS',       [YeoJohnson, Gauss]),
        ('Wind speed (header)', [YeoJohnson, Gauss]),

        ('LGS_TUR_GND', [Invert, YeoJohnson, Gauss]),
        ('MASS-DIMM Cn2 fraction at ground', [Invert, YeoJohnson, Gauss]),
        ('Temperature (header)', [Invert, YeoJohnson, Gauss]),
        ('Relative Flux RMS',    [Logify, Gauss]),
        ('ASM_RFLRMS',  [Logify, Gauss]),
        ('Airmass',     [YeoJohnson, Uniform]),
        ('IRLOS photons, [photons/s/m^2]',        [Logify, Gauss]),
        # ('IRLOS photons (cube), [photons/s/m^2]', [Logify, Gauss]),

        ('Cn2 fraction below 2000m', [YeoJohnson, Uniform]),
        ('MASS_FRACGL',              [YeoJohnson, Gauss]),

        ('IA_FWHM', [Gauss]),

        ('ASM_WINDDIR_10', [Uniform]),
        ('ASM_WINDDIR_30', [Uniform]),
        ('Wind Direction at 30m (0/360) [deg]', [Uniform]),
        ('LGS1 photons, [photons/m^2/s]', [Uniform]),
        ('LGS2 photons, [photons/m^2/s]', [Uniform]),
        ('LGS3 photons, [photons/m^2/s]', [Uniform]),
        ('LGS4 photons, [photons/m^2/s]', [Uniform]),
        ('NGS mag (from ph.)', [Uniform]),
        ('RA (science)',  [Uniform]),
        ('DEC (science)', [Uniform]),
        ('Exp. time',     [Gauss]),
        ('Tel. altitude', [Uniform]),
        ('Tel. azimuth',  [Uniform]),
        ('Par. angle',    [Uniform]),
        ('Derot. angle',  [Uniform]),
        ('NGS RA',        [Uniform]),
        ('NGS DEC',       [Uniform]),
        # ('Pixel scale (science)', [Uniform]),
        ('window',    [Uniform]),
        ('frequency', [Uniform]),
        ('gain',      [Uniform]),
        ('plate scale, [mas/pix]', [Uniform]),
        ('conversion, [e-/ADU]',   [Uniform]),
        ('RON, [e-]', [Uniform])
    ]

    # Find transforms parameters for each entry
    df_transforms = {}
    for entry, transforms_list in entry_data:
        if entry in df.columns.values and len(transforms_list) > 0:
            print(f' -- Processing \"{entry}\"')
            df_transforms[entry] = CreateTransformSequence(entry, df, transforms_list)
            
    return df_transforms


def normalize_df(df, df_transforms, backward=False):
    df_normalized = {}

    for entry in df.columns.values:
        if entry in df_transforms:
            series = df[entry].replace([np.inf, -np.inf], np.nan)
            if backward:
                transformed_values = df_transforms[entry].backward(series.dropna().values)
            else:
                transformed_values = df_transforms[entry].forward(series.dropna().values)
                    
            full_length_values = np.full_like(series, np.nan, dtype=np.float64)
            full_length_values[~series.isna()] = transformed_values 
            df_normalized[entry] = full_length_values   

    df_normalized = pd.DataFrame(df_normalized)
    df_normalized['ID'] = df.index
    df_normalized.set_index('ID', inplace=True)
    
    return df_normalized


# Data imputation
def impute_df(df):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor

    # Clone dataframe
    df_imputed = df.copy()

    # Delete non-numeric string columns
    # df_imputed = df_imputed.select_dtypes(exclude=['object'])
    # df_imputed = df_imputed.select_dtypes(exclude=['bool'])

    droppies_for_imputter = [
        # 'RA (science)',
        # 'DEC (science)',
        # 'Tel. altitude',
        # 'Tel. azimuth',
        # 'NGS RA',
        # 'NGS DEC',
        # 'NGS mag (from ph.)' # delete to restore later
    ]

    # Change scaling
    # for i in range(1, 5):
    #     df_imputed[f'LGS{i} photons, [photons/m^2/s]'] /= 1e9
        
    # df_imputed['MASS_TURB'] *= 1e13
    # df_imputed['IRLOS photons, [photons/s/m^2]'] /= 1e4
    # df_imputed['IRLOS photons (cube), [photons/s/m^2]'] /= 1e4

    if len(droppies_for_imputter) > 0:
        df_imputed.drop(columns=droppies_for_imputter, inplace=True)

    df_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)

    # scaler_imputer = StandardScaler()
    # df_imputed_data = scaler_imputer.fit_transform(df_imputed)
    # df_imputed = pd.DataFrame(df_imputed_data, columns=df_imputed.columns)

    # plot_data_filling(df_imputed)

    #%
    # imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=200, verbose=2)
    # imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, n_jobs=16), max_iter=15, random_state=0, verbose=2)

    imputer = IterativeImputer(
        estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=16),
        random_state = 42, 
        max_iter = 7,
        verbose = 2)

    imputed_data = imputer.fit_transform(df_imputed)
    df_imputed = pd.DataFrame(imputed_data, columns=df_imputed.columns)

    #%
    df_out = df.copy()

    # Copy missing values back to the original DataFrame
    for col in df_imputed.columns:
        df_out[col] = df_imputed[col]

    # df_out['NGS mag (from ph.)'] = GetJmag(df_out['IRLOS photons, [photons/s/m^2]'])
    return imputer, df_out


# %%
