#%%
import sys
sys.path.insert(0, '..')

import pickle
import re
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.restoration import inpaint
from tools.utils import GetROIaroundMax, wavelength_to_rgb, GetJmag
from project_globals import MUSE_CUBES_FOLDER, MUSE_RAW_FOLDER, MUSE_DATA_FOLDER, LIFT_PATH
from datetime import datetime
import datetime
import pandas as pd
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
import requests
from io import StringIO
from photutils.background import Background2D, MedianBackground


UT4_coords = ('24d37min37.36s', '-70d24m14.25s', 2635.43)
UT4_location = EarthLocation.from_geodetic(lat=UT4_coords[0], lon=UT4_coords[1], height=UT4_coords[2]*u.m)

h = 'HIERARCH ESO '
IRLOS_upgrade_date = pd.to_datetime('2021-03-20T00:00:00').tz_localize('UTC')

# To get MUSE cubes:
# http://archive.eso.org/wdb/wdb/cal/data_types/form --> MUSE --> DATACUBE_STD --> num setups --> NFM-AO-N_Blue-IR_SCI1.0

# To get ASM data:
# http://archive.eso.org/cms/eso-data/ambient-conditions/paranal-ambient-query-forms.html


#%% ------------------ Create/read a matching table between the raw files and reduced cubes -------------------------
if not os.path.exists(MUSE_RAW_FOLDER+'../files_matches.csv'):

    cubes_files = os.listdir(MUSE_CUBES_FOLDER)
    raw_files   = os.listdir(MUSE_RAW_FOLDER)

    cubes_obs_date_table = {}
    raw_obs_date_table   = {}
    exposure_times       = {}

    print('Scanning the cubes files...')
    for file in tqdm(cubes_files):
        with fits.open(os.path.join(MUSE_CUBES_FOLDER, file)) as hdul_cube:
            cubes_obs_date_table[file] = hdul_cube[0].header['DATE-OBS']
            exposure_times[file] = (hdul_cube[0].header['DATE-OBS'], hdul_cube[0].header['EXPTIME'])

    print('Scanning the raw files...')
    for file in tqdm(raw_files):
        with fits.open(os.path.join(MUSE_RAW_FOLDER, file)) as hdul_cube:
            raw_obs_date_table[file] = hdul_cube[0].header['DATE-OBS']


    df1 = pd.DataFrame(cubes_obs_date_table.items(), columns=['cube', 'date'])
    df2 = pd.DataFrame(raw_obs_date_table.items(),   columns=['raw',  'date'])

    # Convert date to pd format
    df1['date'] = pd.to_datetime(df1['date']).dt.tz_localize('UTC')
    df2['date'] = pd.to_datetime(df2['date']).dt.tz_localize('UTC')

    df1.set_index('date', inplace=True)
    df2.set_index('date', inplace=True)

    files_matches = pd.concat([df1, df2], axis=1).dropna()

    #% Save files_matches
    files_matches.to_csv(MUSE_RAW_FOLDER+'../files_matches.csv')

else:
    files_matches = pd.read_csv(MUSE_RAW_FOLDER+'../files_matches.csv')
    files_matches.set_index('date', inplace=True)


# ------------------ Create/read the dataset with exposure times ---------------------------------------
if not os.path.exists(MUSE_RAW_FOLDER+'../exposure_times.csv'):
    df3 = pd.DataFrame(exposure_times.items(), columns=['filename', 'obs_time'])

    df3['exposure_start'] = pd.to_datetime(df3['obs_time'].apply(lambda x: x[0]))
    df3['exposure_time'] = df3['obs_time'].apply(lambda x: x[1])
    df3.drop(columns=['obs_time'], inplace=True)
    df3['exposure_end'] = df3['exposure_start'] + pd.to_timedelta(df3['exposure_time'], unit='s')
    df3.drop(columns=['exposure_time'], inplace=True)
    df3.set_index('filename', inplace=True)
    df3['exposure_start'] = df3['exposure_start'].dt.tz_localize('UTC')
    df3['exposure_end'] = df3['exposure_end'].dt.tz_localize('UTC')
    df3.to_csv(MUSE_RAW_FOLDER+'../exposure_times.csv')
    
else:
    df3 = pd.read_csv(MUSE_RAW_FOLDER+'../exposure_times.csv')
    df3.set_index('filename', inplace=True)


# ----------------- Read the dataset with IRLOS data ---------------------------------------
IRLOS_cmds = pd.read_csv(MUSE_RAW_FOLDER+'../IRLOS_commands.csv')
IRLOS_cmds.drop(columns=['Unnamed: 0'], inplace=True)

IRLOS_cmds['timestamp'] = IRLOS_cmds['timestamp'].apply(lambda x: x[:-1] if x[-1] == 'Z' else x)
IRLOS_cmds['timestamp'] = pd.to_datetime(IRLOS_cmds['timestamp']).dt.tz_localize('UTC')
IRLOS_cmds.set_index('timestamp', inplace=True)
IRLOS_cmds = IRLOS_cmds.sort_values(by='timestamp')

IRLOS_cmds['command'] = IRLOS_cmds['command'].apply(lambda x: x.split(' ')[-1])

# Filter etries that don't look like IRLOS regimes
pattern = r'(\d+x\d+_SmallScale_\d+Hz_\w+Gain)|(\d+x\d+_\d+Hz_\w+Gain)|(\d+x\d+_SmallScale)'
IRLOS_cmds = IRLOS_cmds[IRLOS_cmds['command'].str.contains(pattern, regex=True)]

pattern = r'(?P<window>\d+x\d+)(?:_(?P<scale>SmallScale))?(?:_(?P<frequency>\d+Hz))?(?:_(?P<gain>\w+Gain))?'
IRLOS_df = IRLOS_cmds['command'].str.extract(pattern)

IRLOS_df['window'] = IRLOS_df['window'].apply(lambda x: int(x.split('x')[0]))
IRLOS_df['frequency'] = IRLOS_df['frequency'].apply(lambda x: int(x[:-2]) if pd.notna(x) else x)
IRLOS_df['gain'] = IRLOS_df['gain'].apply(lambda x: 68 if x == 'HighGain' else 1 if x == 'LowGain' else x)
IRLOS_df['scale'] = IRLOS_df['scale'].apply(lambda x: x.replace('Scale','') if pd.notna(x) else x)  
IRLOS_df['scale'] = IRLOS_df['scale'].fillna('Small')

IRLOS_df['plate scale, [mas/pix]'] = IRLOS_df.apply(lambda row: 60  if row.name > IRLOS_upgrade_date and row['scale'] == 'Small' else 242 if row.name > IRLOS_upgrade_date and row['scale'] == 'Large' else 78 if row['scale'] == 'Small' else 324, axis=1)
IRLOS_df['conversion, [e-/ADU]']   = IRLOS_df.apply(lambda row: 9.8 if row.name > IRLOS_upgrade_date else 3, axis=1)

IRLOS_df['gain'] = IRLOS_df['gain'].fillna(1).astype('int')
IRLOS_df['frequency'] = IRLOS_df['frequency'].fillna(200).astype('int')


# ----------------- Read the dataset with the LGS WFSs data ---------------------------------------
LGS_flux_df   = pd.read_csv(MUSE_RAW_FOLDER + '../LGS_flux.csv',   index_col=0)
LGS_slopes_df = pd.read_csv(MUSE_RAW_FOLDER + '../LGS_slopes.csv', index_col=0)

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
        IRLOS_cube *= mask[..., None] # Remove background pixels
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


def GetIRLOSdata(hdul_raw, start_time, IRLOS_cube):
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
    M1_area = (8-1.12)**2 * np.pi / 4 # [m^2]

    return flux_LGS * conversion_factor * num_subapertures  \
        / HO_gain / detector_DIT / QE / GALACSI_transmission \
        * HO_rate / M1_area # [photons/m^2/s]
        

def GetLGSdata(hdul_cube, cube_name, start_time):
    # AOS LGSi FLUX: [ADU] Median flux in subapertures / 
    # Median of the LGSi flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.
    LGS_data_df = LGS_flux_df[LGS_flux_df['filename'] == cube_name].copy()
    LGS_data_df.rename(columns = { f'LGS{i} flux': f'LGS{i} flux, [ADU/frame]' for i in range(1,5) }, inplace=True)
    LGS_data_df.index = pd.to_datetime(LGS_data_df.index).tz_localize('UTC')

    HO_gains = np.array([hdul_cube[0].header[h+'AOS LGS'+str(i+1)+' DET GAIN'] for i in range(4)]) # samevalue for all LGSs

    for i in range(1,5):
        LGS_data_df.loc[:,f'LGS{i} photons, [photons/m^2/s]'] = GetLGSphotons(LGS_data_df[f'LGS{i} flux, [ADU/frame]'], HO_gains[i-1]).round().astype('uint32')
    
    if LGS_data_df.empty:
        # Create at least something if there is no data
        LGS_data_df = { f'LGS{i} works': None for i in range(1,5) }
        LGS_data_df['filename'] = cube_name
        LGS_data_df['time'] = start_time
        LGS_data_df = pd.DataFrame(LGS_data_df, index=[-1])
        LGS_data_df.set_index('time', inplace=True)
    
    for i in range(1,5):
        if f'{h}LGS{i} LASR{i} SHUT STATE' in hdul_cube[0].header:
            LGS_data_df.loc[:,f'LGS{i} works'] = hdul_cube[0].header[f'{h}LGS{i} LASR{i} SHUT STATE'] == 0
        else:
            LGS_data_df.loc[:,f'LGS{i} works'] = False

    LGS_data_df.drop(columns=['filename'], inplace=True)
    
    # [pix] slopes RMS / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
    LGS_slopes_data_df = LGS_slopes_df[LGS_slopes_df['filename'] == cube_name].copy()
    LGS_slopes_data_df.index = pd.to_datetime(LGS_slopes_data_df.index).tz_localize('UTC')
    LGS_slopes_data_df.drop(columns=['filename'], inplace=True)
    
    return pd.concat([LGS_data_df, LGS_slopes_data_df], axis=1)


#% --------------------------------------------------------------------------------------
def GetSpectrum(white, radius=5):
    _, ids, _ = GetROIaroundMax(white, radius*2)
    return np.nansum(hdul_cube[1].data[:, ids[0], ids[1]], axis=(1,2))


def range_overlap(v_min, v_max, h_min, h_max):
    start_of_overlap, end_of_overlap = max(v_min, h_min), min(v_max, h_max)
    
    if start_of_overlap > end_of_overlap:
        return 0.0
    else:
        return (end_of_overlap-start_of_overlap) / (v_max-v_min)


def GetImageSpectrumHeader(hdul_cube, show_plots=True):
    find_closest = lambda λ, λs: np.argmin(np.abs(λs-λ)).astype('int')

    # Extract spectral range information
    start_spaxel = hdul_cube[1].header['CRPIX3']
    num_λs = int(hdul_cube[1].header['NAXIS3']-start_spaxel+1)
    Δλ = hdul_cube[1].header['CD3_3' ] / 10.0
    λ_min = hdul_cube[1].header['CRVAL3'] / 10.0
    λs = np.arange(num_λs)*Δλ + λ_min
    λ_max = λs.max()

    # 1 [erg] = 10^−7 [J]
    # 1 [Angstrom] = 10 [nm]
    # 1 [cm^2] = 0.0001 [m^2]
    # [10^-20 * erg / s / cm^2 / Angstrom] = [10^-22 * W/m^2 / nm] = [10^-22 * J/s / m^2 / nm]
    
    units = hdul_cube[1].header['BUNIT'] # flux units
    #%
    # Polychromatic image
    white = np.nan_to_num(hdul_cube[1].data).sum(axis=0)
    spectrum = GetSpectrum(white)

    wvl_bins = None #np.array([478, 511, 544, 577, 606, 639, 672, 705, 738, 771, 804, 837, 870, 903, 935], dtype='float32')

    # Pre-defined bad wavelengths ranges
    bad_wvls = np.array([[450, 478], [577, 606]])
    bad_ids  = np.array([find_closest(wvl, λs) for wvl in bad_wvls.flatten()], dtype='int').reshape(bad_wvls.shape)
    valid_λs = np.ones_like(λs)

    for i in range(len(bad_ids)):
        valid_λs[bad_ids[i,0]:bad_ids[i,1]+1] = 0
        bad_wvls[i,:] = np.array([λs[bad_ids[i,0]], λs[bad_ids[i,1]+1]])
        
    if wvl_bins is not None:
        λ_bins_smart = wvl_bins
    else:
        # Bin data cubes
        # Before the sodium filter
        λ_bin = (bad_wvls[1][0]-bad_wvls[0][1])/3.0
        λ_bins_before = bad_wvls[0][1] + np.arange(4)*λ_bin
        bin_ids_before = [find_closest(wvl, λs) for wvl in λ_bins_before]

        # After the sodium filter
        λ_bins_num    = (λ_max-bad_wvls[1][1]) / np.diff(λ_bins_before).mean()
        λ_bins_after  = bad_wvls[1][1] + np.arange(λ_bins_num+1)*λ_bin
        bin_ids_after = [find_closest(wvl, λs) for wvl in λ_bins_after]
        bins_smart    = bin_ids_before + bin_ids_after
        λ_bins_smart  = λs[bins_smart]

    print('Wavelength bins, [nm]:', *λ_bins_smart.astype('int'))

    Rs, Gs, Bs = np.zeros_like(λs), np.zeros_like(λs), np.zeros_like(λs)

    for i,λ in enumerate(λs):
        Rs[i], Gs[i], Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

    Rs = Rs * valid_λs * spectrum / np.median(spectrum)
    Gs = Gs * valid_λs * spectrum / np.median(spectrum)
    Bs = Bs * valid_λs * spectrum / np.median(spectrum)
    colors = np.dstack([np.vstack([Rs, Gs, Bs])]*600).transpose(2,1,0)

    fig_handler = None
    if show_plots:
        fig_handler = plt.figure(dpi=200)
        plt.imshow(colors, extent=[λs.min(), λs.max(), 0, 120])
        plt.vlines(λ_bins_smart, 0, 120, color='white') #draw bins borders
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=2.0, color='white')
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=0.5, color='blue')
        plt.xlabel(r"$\lambda$, [nm]")
        plt.ylabel(r"$\left[ 10^{-20} \frac{erg}{s \cdot cm^2 \cdot \AA} \right]$")
        ax = plt.gca()
        # ax.get_yaxis().set_visible(False)
        # plt.show()

    _, ROI, _ = GetROIaroundMax(white, win=200)

    # Generate reduced cubes
    data_reduced = np.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    std_reduced  = np.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    # Central λs at each spectral bin
    wavelengths  = np.zeros(len(λ_bins_smart)-1)
    flux         = np.zeros(len(λ_bins_smart)-1)

    bad_layers = [] # list of spectral layers that are corrupted (excluding the sodium filter)

    bins_to_ignore = []

    for bin in tqdm( range(len(bins_smart)-1) ):
        chunk = hdul_cube[1].data[ bins_smart[bin]:bins_smart[bin+1], ROI[0], ROI[1] ]
        wvl_chunck = λs[bins_smart[bin]:bins_smart[bin+1]]
        flux_chunck = spectrum[bins_smart[bin]:bins_smart[bin+1]]

        # Check if it is a sodium filter range
        if range_overlap(wvl_chunck.min(), wvl_chunck.max(), bad_wvls[1][0], bad_wvls[1][1]) > 0.9:
            bins_to_ignore.append(bin)
            continue

        for i in range(chunk.shape[0]):
            layer = chunk[i,:,:]
            if np.isnan(layer).sum() > layer.size//2: # if more than 50% of the pixela on image are NaN
                bad_layers.append((bin, i))
                wvl_chunck[i]  = np.nan
                flux_chunck[i] = np.nan
                continue
            else:
                mask_inpaint = np.zeros_like(layer, dtype=np.int8)
                mask_inpaint[np.where(np.isnan(layer))] = 1
                chunk[i,:,:] = inpaint.inpaint_biharmonic(layer, mask_inpaint)  # Fix bad pixels per spectral slice

        data_reduced[bin,:,:] = np.nansum(chunk, axis=0)
        std_reduced [bin,:,:] = np.nanstd(chunk, axis=0)
        wavelengths[bin] = np.nanmean(np.array(wvl_chunck)) # central wavelength for this bin
        flux[bin] = np.nanmean(np.array(flux_chunck))

    data_reduced = np.delete(data_reduced, bins_to_ignore, axis=0)
    std_reduced  = np.delete(std_reduced,  bins_to_ignore, axis=0)
    wavelengths  = np.delete(wavelengths,  bins_to_ignore)
    flux         = np.delete(flux,         bins_to_ignore)

    print(str(len(bad_layers))+'/'+str(hdul_cube[1].data.shape[0]), '('+str(np.round(len(bad_layers)/hdul_cube[1].data.shape[0],2))+'%)', 'slices are corrupted')

    # Collect the telemetry from the header
    spectral_info = {
        'spectrum (full)': (λs, spectrum),
        'wvl range':     [λ_min, λ_max, Δλ], # From header
        'filtered wvls': bad_wvls,
        'flux units':    units,
        'wvl bins':      λ_bins_smart, # Wavelength bins
        'wvls binned':   wavelengths,
        'flux binned':   flux,
        'spectrum fig':  fig_handler
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

    alpha = hdul_cube[0].header[h+'AOS NGS ALPHA'] # Alpha coordinate for the NGS, [hms]
    delta = hdul_cube[0].header[h+'AOS NGS DELTA'] # Delta coordinate for the NGS, [dms]

    # Convert numerical format to [hms] and [dms] string format
    alpha_hms = f"{int(alpha // 10000):02}h{int((alpha % 10000) // 100):02}m{alpha % 100:06.3f}s"
    delta_dms = format_dms(*convert_to_dms(delta))

    coord_NGS = SkyCoord(alpha_hms, delta_dms, frame='icrs')

    ra_NGS, dec_NGS = (coord_NGS.ra.deg, coord_NGS.dec.deg)

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

    return images, data_df, spectral_info


# with open('C:/Users/akuznets/Data/MUSE/DATA_raw_binned/'+files[file_id].split('.fits')[0]+'.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# plt.imshow(np.log(images['cube'].mean(axis=0)), cmap='gray')
# plt.axis('off')

# ------------------------------ Raw header reals ------------------------------------------
'''
def to_dist(x):
    median_val = np.median(x)
    std_val = np.std(x)
    
    if isinstance(x, np.ndarray):
        if std_val == 0.0:
            return median_val
        else:
            return median_val + 1j*std_val if np.abs(median_val/std_val) > 1e-6 else median_val
    else:
        return x
    '''

def extract_countables(hdul, table_name, entry_pattern):
    keys_list = [x for x in hdul[table_name].header.values()]
    values = []

    for i in range (0, 50):
        entry = entry_pattern(i)
        if entry in keys_list:
            values.append(hdul[table_name].data[entry])
            
    return values


def extract_countables_df(hdul, table_name, entry_pattern):  
    keys_list = [x for x in hdul[table_name].header.values()]

    if 'Sec' in keys_list:
        timestamps = pd.to_datetime(time_from_sec_usec(*read_secs_usecs(hdul_raw, table_name)))
    else:
        timestamps = pd.to_datetime(time_from_str(hdul_raw[table_name].data['TIME_STAMP'])).tz_localize('UTC')
    
    values_dict = {}

    for i in range (0, 50):
        entry = entry_pattern(i)
        if entry in keys_list:
            values_dict[entry] = hdul[table_name].data[entry]
            
    return pd.DataFrame(values_dict, index=timestamps)


def convert_to_default_byte_order(df):
    for col in df.columns:
        if df[col].dtype.byteorder not in ('<', '=', '|'):
            df[col] = df[col].astype(df[col].dtype.newbyteorder('<'))
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


def FetchFromESOarchive(start_time, end_time, minutes_delta=1):
    start_time_str = (start_time - pd.Timedelta(f'{minutes_delta}m')).tz_localize(None).isoformat()
    end_time_str   = (end_time   + pd.Timedelta(f'{minutes_delta}m')).tz_localize(None).isoformat()

    req_str = lambda system: f'http://archive.eso.org/wdb/wdb/asm/{system}_paranal/query?wdbo=csv&start_date={start_time_str}..{end_time_str}'

    def GetFromURL(request_URL):
        response = requests.get(request_URL)

        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            if df.columns[0][0] == '#':
                print('No data retrieved')
                return None
            
            df = df[~df['Date time'].str.contains('#')]
            df.rename(columns={'Date time': 'time'}, inplace=True)
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
            df.set_index('time', inplace=True)
            return df

        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
            return None


    print('Quring ASM database...')
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


    print('Querying MASS-DIMM data')
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


    print('Querying DIMM data')
    request_dimm = req_str('dimm')
    dimm_entries = [
        ('tab_fwhm', 1),
        ('tab_rfl', 1),
        ('tab_rfl_time', 0)
    ]
    request_dimm += ''.join([f'&{entry[0]}={entry[1]}' for entry in dimm_entries])
    dimm_df = GetFromURL(request_dimm)

    #%
    print('Querying SLODAR data')
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

    PSF_1 = np.load(MUSE_DATA_FOLDER + 'PSF_default.npy').mean(axis=0)
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
        # Add dependencies to the path
        for path in ['', 'DIP', 'LIFT', 'LIFT/modules', 'experimental']:
            path_new = f'{LIFT_PATH}{path}/..'
            if path_new not in sys.path:
                sys.path.append(path_new)
                
        from LIFT_full.experimental.IRLOS_2x2_function import estimate_2x2    

        return estimate_2x2(cube)      
        
    except Exception as e:
        print(f'Error: {e}')
        return None


#%%
bad_ids = []

# read list of files form .txt file:
# with open(MUSE_RAW_FOLDER+'../bad_files.txt', 'r') as f:
#     files_bad = f.read().splitlines()

# for i in range(len(files_bad)):
#     files_bad[i] = int(files_bad[i])

# for file_id in tqdm(files_bad):

# bad_IRLOS_ids = [68, 242, 316, 390]

# list_ids = [411, 410, 409, 405, 146, 296, 276, 395, 254, 281, 343, 335]

for file_id in tqdm(range(0, len(files_matches))):
# for file_id in tqdm(list_ids):
    print(f'>>>>>>>>>>>>>> Processing file {file_id}...')
    try:
        hdul_raw  = fits.open(os.path.join(MUSE_RAW_FOLDER,   files_matches.iloc[file_id]['raw' ]))
        hdul_cube = fits.open(os.path.join(MUSE_CUBES_FOLDER, files_matches.iloc[file_id]['cube']))

        cube_name = files_matches.iloc[file_id]['cube']

        start_time = pd.to_datetime(time_from_str(hdul_cube[0].header['DATE-OBS'])).tz_localize('UTC')
        end_time   = pd.to_datetime(start_time + datetime.timedelta(seconds=hdul_cube[0].header['EXPTIME']))

        IRLOS_cube, win = GetIRLOScube(hdul_raw)
        IRLOS_data_df   = GetIRLOSdata(hdul_raw, start_time, IRLOS_cube)
        LGS_data_df     = GetLGSdata(hdul_cube, cube_name, start_time)
        
        if win is not None:
            IRLOS_data_df['window'] = win
        
        MUSE_images, MUSE_data_df, spectral_info = GetImageSpectrumHeader(hdul_cube, show_plots=False)
        Cn2_data_df, atm_data_df, asm_data_df    = GetRawHeaderData(hdul_raw)
        asm_df, massdimm_df, dimm_df, slodar_df  = FetchFromESOarchive(start_time, end_time, minutes_delta=1)

        hdul_cube.close()
        hdul_raw.close()

        # Try to detect the rotation of the PSF with diffracive features, work only for good PSFs
        derot_ang = MUSE_data_df['Derot. angle'].item()
        derot_ang += 360 if derot_ang < -180 else -360
        pupil_angle, angle_delta = get_PSF_rotation(MUSE_images, derot_ang)
        MUSE_data_df['Pupil angle'] = pupil_angle + angle_delta

        # Estimate the IRLOS phasecube from 2x2 PSFs
        OPD_subap, OPD_aperture, PSF_estim = (None,)*3
        if IRLOS_cube is not None:
            if (res := get_IRLOS_phase(IRLOS_cube)) is not None:
                OPD_subap, OPD_aperture, PSF_estim = res
                OPD_subap = OPD_subap.astype(np.float32)
                OPD_aperture = OPD_aperture.astype(np.float32)
                PSF_estim = PSF_estim.astype(np.float32)

        IRLOS_phase_data = {
            'OPD per subap': OPD_subap,
            'OPD aperture':  OPD_aperture,
            'PSF estimated': PSF_estim
        }

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

        path_new = cube_name.split('.fits')[0] + '.pickle'
        path_new = path_new.replace(':','-')
        path_new = MUSE_RAW_FOLDER + '../DATA_reduced/' + str(file_id) + '_' + path_new

        with open(path_new, 'wb') as handle:
            pickle.dump(data_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
    except Exception as e:
        print(f'Error with file {file_id}: {e}')
        bad_ids.append(file_id)
        continue


#%% Render the dataset
for file in tqdm(os.listdir(MUSE_RAW_FOLDER+'../DATA_reduced/')):

    with open(MUSE_RAW_FOLDER+'../DATA_reduced/'+file, 'rb') as f:
        data = pickle.load(f)

    white = np.log10(1+np.abs(data['images']['white']))
    white -= white.min()
    white /= white.max()

    # from scipy.ndimage import rotate
    # pupil_ang = data['MUSE header data']['Pupil angle'].item() - 45
    # if pupil_ang < -180: pupil_ang += 360
    # if pupil_ang > 180:  pupil_ang -= 360
    # white_rot = rotate(white, pupil_ang, reshape=False)

    if data['images']['IRLOS cube'] is not None:
        IRLOS_img = np.log10(1+np.abs(data['images']['IRLOS cube'].mean(axis=-1)))
        IRLOS_img = data['images']['IRLOS cube'].mean(axis=-1)
    else:
        IRLOS_img = np.zeros_like(white)

    title = file.replace('.pickle', '')
    fig, ax = plt.subplots(1,2, figsize=(14, 7.5))

    ax[0].set_title(title)
    ax[1].set_title('IRLOS (2x2)')
    ax[0].imshow(white, cmap='gray')
    ax[1].imshow(IRLOS_img, cmap='hot')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()
    
    plt.show()
    plt.savefig(MUSE_RAW_FOLDER+'../MUSE_images/' + title + '.png')


#%%
# file_id = 173 #103#320 #327# 343
# file_id = 3 #103#320 #327# 343
# file_id = 173

file_id = 411
file_id = 405
file_id = 146
file_id = 296
# file_id = 382
# file_id = 395
# file_id = 254

# file_id = 344
# file_id = 276
# file_id = 281
# file_id = 311
# file_id = 345
file_id = 170


hdul_raw  = fits.open(os.path.join(MUSE_RAW_FOLDER,   files_matches.iloc[file_id]['raw' ]))
hdul_cube = fits.open(os.path.join(MUSE_CUBES_FOLDER, files_matches.iloc[file_id]['cube']))

cube_name = files_matches.iloc[file_id]['cube']

start_time = pd.to_datetime(time_from_str(hdul_cube[0].header['DATE-OBS'])).tz_localize('UTC')
end_time   = pd.to_datetime(start_time + datetime.timedelta(seconds=hdul_cube[0].header['EXPTIME']))

IRLOS_cube, win = GetIRLOScube(hdul_raw)
IRLOS_data_df   = GetIRLOSdata(hdul_raw, start_time, IRLOS_cube)
LGS_data_df     = GetLGSdata(hdul_cube, cube_name, start_time)

if win is not None:
    IRLOS_data_df['window'] = win

MUSE_images, MUSE_data_df, misc_info    = GetImageSpectrumHeader(hdul_cube, show_plots=True)
Cn2_data_df, atm_data_df, asm_data_df   = GetRawHeaderData(hdul_raw)
asm_df, massdimm_df, dimm_df, slodar_df = FetchFromESOarchive(start_time, end_time, minutes_delta=1)

hdul_cube.close()
hdul_raw.close()

# Get the angle of PSF rotation if streaks are visible
derot_ang = MUSE_data_df['Derot. angle'].item()
derot_ang += 360 if derot_ang < -180 else -360
pupil_angle, angle_delta = get_PSF_rotation(MUSE_images, derot_ang)
MUSE_data_df['Pupil angle'] = pupil_angle + angle_delta

# Estimate the IRLOS phasecube from 2x2 PSFs
OPD_subap, OPD_aperture, PSF_estim = (None,)*3
if IRLOS_cube is not None:
    if (res := get_IRLOS_phase(IRLOS_cube)) is not None:
        OPD_subap, OPD_aperture, PSF_estim = res

IRLOS_phase_data = {
    'OPD per subap': OPD_subap,
    'OPD aperture':  OPD_aperture,
    'PSF estimated': PSF_estim
}

#%%
# Create a flat DataFrame with temporal dimension compressed
all_df = pd.concat([
    IRLOS_data_df, LGS_data_df, MUSE_data_df, Cn2_data_df, atm_data_df,
    asm_data_df, asm_df, massdimm_df, dimm_df, slodar_df
])
all_df.sort_index(inplace=True)

flat_df = create_flat_dataframe(all_df)
flat_df.insert(0, 'time', start_time)
flat_df.insert(0, 'name', cube_name)

#%%
df = raw_df
# df = all_df
fig, ax = plt.subplots(figsize=(20, 7))

# Loop through the DataFrame to plot non-NaN values
for col in df.columns:
    non_nan_indices = df.index[~df[col].isna()]
    ax.scatter([col] * len(non_nan_indices), non_nan_indices, label=col)

ax.fill_between(df.columns, start_time, end_time, color='green', alpha=0.15, label='Observation Period')

# Set labels and title
ax.set_xlabel('Columns')
ax.set_ylabel('Timestamps')
ax.set_title('Presence of Values in DataFrame')
plt.xticks(rotation=90, ha='right')

ax.set_xlim(-0.5, len(df.columns)-0.5)

plt.grid(True)
plt.show()


#%%
%matplotlib qt

import matplotlib.pyplot as plt
import numpy as np

data = np.log(data['cube'].mean(axis=0))
# Get the extent of the data
extent = np.array([-data.shape[0]/2, data.shape[0]/2, -data.shape[1]/2, data.shape[1]/2]) * data['telemetry']['pixel scale']

# Display the data with the extent centered at zero
plt.imshow(data, extent=extent.tolist(), origin='lower')
plt.colorbar()
plt.show()

#%%
# wmfsgw
#AOS ATM TIMESTAMPSEC: [UTC] Timestamp of the data in seconds(o).
#AOS CNSQ CNSQ_i: [m^-2/3] Absolute Cn2 at altitude i.
#AOS CNSQ FRACCNSQ_i: [%] Fraction of Cn2 at altitude i.
#AOS CNSQ Hi: [km] Altitude of turbulent layer i.
#AOS CNSQ L0TOT: [m] Total outerscale.
#AOS CNSQ L0_i: [m] Outerscale at altitude i.
#AOS CNSQ R0TOT: [m] Total fried parameter.
#AOS CNSQ SEEINGTOT: [acrsec] Total seeing.
#AOS CNSQ TIMESTAMPSEC: [UTC] Timestamp of the data in seconds(o).
#AOS DSM MODE1: [nm] Amount of focus in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODE2: [nm] Amount of coma 1 in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODE3: [nm] Amount of coma 2 in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODEi: [nm] Amount of the elastic mode i in the DSM commands offloaded to the M1
#AOS IR SAi FLUX: [ADU] IRLOS subaperture flux
#AOS IR SAi SPOTSIZEi: [pix] Fwhm of the major axis of the spot in the IRLOS subaperture.
#AOS IR TTRMS X: [pix] IRLOS residual tip/tilt RMS - x axis
#AOS IR TTRMS Y: [pix] IRLOS residual tip/tilt RMS - y axis
#AOS IRCTR SUBSTATE: LO/TTS Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend
#AOS JITCTR SUBSTATE: LGS Jitter Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend
#AOS LGSi L0: [m] Outer scale of the turbulence, as measured by the WFSi
#AOS LGSi R0: [m] Fried parameter along the line of sight and at 0.5 microns, as measured by the WFSi.
#AOS LGSi SPOTSIZE1: [pix] Fwhm of the small axis of the LGSi (0.83 arcsec/pixel)
#AOS LGSi SPOTSIZE2: [pix] Elongation of LGS image vs distance between the WFSi subaperture and the launch telescope pupil of the LGSUi.
#AOS LGSi TAU0: [s] Coherence time of the turbulence (tau0) along the line of sight and at 0.5 microns.
#AOS LGSCTR SUBSTATE: HO Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#AOS LGSi DARKi: [ADU] Quadrant i dark / Dark value, as measured in the quadrant i of detector i median value (0.17 photons/ADU at gain 100).
#AOS LGSi EEIMP: Encircled Energy Improvement.
#AOS LGSi FLUX: [ADU] Median flux in subapertures / Median of the LGSi flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.
#AOS LGSi FSM X: [SPARTA unit] FSM X-axis / LGSUi Field Steering Mirror command - X axis.
#AOS LGSi FSM Y: [SPARTA unit] FSM Y-axis / LGSUi Field Steering Mirror command - Y axis.
#AOS LGSi FWHM GAIN: Ratio between the fwhm of the uncorrected PSF and of the corrected one as measured by LGS WFSi.
#AOS LGSi FWHMIMP: FWHM improvement.
#AOS LGSi ROTATION: [deg] DSM clocking / Clocking of the DSM w.r.t. the WFSi.
#AOS LGSi SEEING: [arcsec] Seeing / Seeing along the line of sight and at 0.5 microns, as measured by the WFSi.
#AOS LGSi SLOPERMSX: [pix] X slopes rms / Median of the X slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
#AOS LGSi SLOPERMSY: [pix] Y slopes rms / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
#AOS LGSi STREHL: [%] Strehl ratio / Strehl Ratio as computed from LGS WFSi measurements at science wavelength.
#AOS LGSi TUR ALT: Altitude turb fraction / Fraction of turbulence not corrected by AO loop - LGS WFSi, (high altitude layer)
#AOS LGSi TUR GND: Ground turb fraction / Fraction of turbulence corrected by AO loop - LGS WFSi, (ground layer).
#AOS LGSi TURVAR RES: [rad^2] Residual variance / Residual variance of aberrations along the line of sight and at 0.5 microns - LGS WFSi.
#AOS LGSi TURVAR TOT: [rad^2] Turb total variance / Total variance of the aberrations along the line of sight and at 0.5 microns - LGS WFSi.
#AOS LGSi VWIND: [m/s] Turb wind speed / Average wind speed along the line of sight, integrated on the turbulence layers.
#AOS LGSi XPUP: X Pupil Shift
#AOS LGSi XSHIFT: [subap %] DSM X shift / Lateral motion of the DSM w.r.t. the WFSi along the X axis - unit is % of one subaperture.
#AOS LGSi XSTRETCH: [subap %] DSM X stretch
#AOS LGSi YPUP: Y Pupil Shift
#AOS LGSi YSHIFT: [subap %] DSM Y shift / Lateral motion of the DSM w.r.t. the WFSi along the Y axis - unit is % of one subaperture.
#AOS LGSi YSTRETCH: [subap %] DSM Y stretch
#AOS MCMCTR SUBSTATE: MCM Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#AOS NA HEIGHT: [m] Na layer altitude.
#AOS NGCIR TEMP: [deg] NGC Head temp.
#AOS NGCLGSi TEMP: [deg] NGC Head temp.
#AOS NGCVIS TEMP: [deg] NGC Head temp.
#AOS NGS FLUX: [ADU] NGS flux / NGS flux (0.17 photons/ADU at gain 100).
#AOS NGS SPOTANGLE: [rad] NGS spot orientation angle
#AOS NGS SPOTSIZE1: [pix] NGS minor axis fwhm / Fwhm of the minor axis of the NGS image as measured in the tip-tilt sensor (0.167 arcsec/pixel).
#AOS NGS SPOTSIZE2: [pix] NGS major axis fwhm / Fwhm of the major axis of the NGS image as measured in the tip-tilt sensor (0.167 arcsec/pixel).
#AOS NGS TTRMS X: [pix] NGS residual tip/tilt RMS - x axis
#AOS NGS TTRMS Y: [pix] NGS residual tip/tilt RMS - y axis
#AOS VISCTR SUBSTATE: LO/TTS Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#
#•	wasm:
#INS DIMM AG DIT: [s] Autoguider base exposure time. This value is logged once at startup and on any change.
#INS DIMM AG ENABLE: Autoguider enable status. This value is logged once at startup and on any change.
#INS DIMM AG KIX: Autoguider integral gain X. This value is logged once at startup and on any change.
#INS DIMM AG KIY: Autoguider integral gain Y. This value is logged once at startup and on any change.
#INS DIMM AG KPX: Autoguider proportional gain X. This value is logged once at startup and on any change.
#INS DIMM AG KPY: Autoguider proportional gain Y. This value is logged once at startup and on any change.
#INS DIMM AG LOCKED: Locked on target. This action is logged when the target star first enters the field (asmdim).
#INS DIMM AG RADIUS: [arcsec] Autoguider in-target radius. Autoguider in-target radius for centering.
# This value is logged once at startup and on any change.
#INS DIMM AG RATE: Autoguider update rate. Autoguider update rate for averaging exposures.
# This value is logged once at startup and on any change.
#INS DIMM AG REFX: [pixel] Autoguider reference pixel X.Autoguider TCCD reference pixel for centroiding.
# This value is logged once at startup and on any change.
#INS DIMM AG REFY: [pixel] Autoguider reference pixel Y.Autoguider TCCD reference pixel for centroiding.
# This value is logged once at startup and on any change.
#INS DIMM AG START: Initiating setup to center target star. This action is logged at start of centering (asmdim).
#INS DIMM AG STOP: Star is inside target radius. This action is logged at end of centering (asmdim).
#INS DIMM CONST AF: CCD noise floor (explicit ad-count if >0, implicit n-sigma if <0). This value is logged once at startup and on any change.
#INS DIMM CONST CCDX: [pixel] X CCD total size. This value is logged once at startup and on any change.
#INS DIMM CONST CCDY: [pixel] Y CCD total size. This value is logged once at startup and on any change.
#INS DIMM CONST FL: [m] Objective Focal Length. This value is logged once at startup and on any change.
#INS DIMM CONST FR: Telescope F Ratio. This value is logged once at startup and on any change.
#INS DIMM CONST INX: [pixel^2] X Instrumental Noise (measured). This value is logged once at startup and on any change.
#INS DIMM CONST INY: [pixel^2] Y Instrumental Noise (measured). This value is logged once at startup and on any change.
#INS DIMM CONST LAMBDA: [m] Wavelength. This value is logged once at startup and on any change.
#INS DIMM CONST MD: [m] Mask Hole Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST MS: [m] Mask Hole Separation. This value is logged once at startup and on any change.
#INS DIMM CONST PA: [deg] Pupil Mask Prism Angle. This value is logged once at startup and on any change.
#INS DIMM CONST PD: [m] Pupils Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST PHI: [m] Telescope Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST PPX: [arcsec] X Pixel Angular Pitch. This value is logged once at startup and on any change.
#INS DIMM CONST PPY: [arcsec] Y Pixel Angular Pitch. This value is logged once at startup and on any change.
#INS DIMM CONST PS: [m] Pupils Separation. This value is logged once at startup and on any change.
#INS DIMM CONST PSX: [m] X Pixel Size. This value is logged once at startup and on any change.
#INS DIMM CONST PSY: [m] Y Pixel Size. This value is logged once at startup and on any change.
#INS DIMM CONST W0: [pixel] Spot separation (nominal). This value is logged once at startup and on any change.
#INS DIMM LIMIT KC: Gaussian Clipping Limit (rms of distribution). This value is logged once at startup and on any change.
#INS DIMM LIMIT KE: [%] Spot Elongation Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KF: Spot Decentering Limit (times of FWHM). This value is logged once at startup and on any change.
#INS DIMM LIMIT KN1: [%] Rejected Exposures Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KN2: [%] Clipped Exposures Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KO: Polynomial order for error-curve fitting (0=off). This value is logged once at startup and on any change.
#INS DIMM LIMIT KP: Valid Pixels Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KR: Saturation Limit (ad count). This value is logged once at startup and on any change.
#INS DIMM LIMIT KS: Statistical Error Limit 1/sqrt(N-1). This value is logged once at startup and on any change.
#INS DIMM LIMIT KS1: Max number of consecutive rejected sequences for a star (0=off). If the number is reached then a change of star is requested.
# This value is logged once at startup and on any change.
#INS DIMM LIMIT KS2: Max total number of rejected sequences for a star (0=off). If the number is reached then a change of star is requested.
# This value is logged once at startup and on any change.
#INS DIMM LIMIT SNMIN: S/N Ratio Limit. This value is logged once at startup and on any change.
#INS DIMM MODE N: Frames per sequence. This value is logged once at startup and on any change.
#INS DIMM MODE NAME: Executing mode name (DIMM or WFCE). This value is logged once at startup and on any change.
#INS DIMM MODE ND: Exposures per frame. This value is logged once at startup and on any change.
#INS DIMM MODE NX: [pixel] X CCD sub-array size. This value is logged once at startup and on any change.
#INS DIMM MODE NY: [pixel] Y CCD sub-array size. This value is logged once at startup and on any change.
#INS DIMM MODE TD: [s] Exposure time. his value is logged once at startup and on any change.
#INS DIMM SEQ START: Seeing measurement sequence started. This action is logged when a measurement sequence is started.
#INS DIMM SEQ STOP: Seeing measurement sequence completed. This action is logged when a measurement sequence is completed.
#INS DIMM SEQIN AIRMASS: Airmass of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN CCDTEMP: CCD chip temperature before sequence. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN DEC: [deg] Position in DEC of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN RA: [deg] Position in RA of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN MAG: [mag] Magnitude of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN N: Number of requested exposures for sequence. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN SKYBG: Sky background input. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN ZD: [deg] Zenith distance of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQOUT ALPHAG: [deg] Spot Axis Rotation vs CCD-x (measured). This value is logged at the end of each sequence.
#INS DIMM SEQOUT CCDTEMP: CCD chip temperature after sequence. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DIT: [s] Average Exposure Time (measured). This value is logged at the end of each sequence.
#INS DIMM SEQOUT DITVAR: [s^2] Variance on Exposure Time. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DN1: Number of exposures rejected in the frame validation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DN2: Number of exposures rejected in the statistical validation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FLUXi: Flux of window 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHM: [arcsec] Seeing average. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHMPARA: [arcsec] Seeing parallel. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHMPERP: [arcsec] Seeing perpendicular. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FXi: [pixel] Average FWHM(x) of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FYi: [pixel] Average FWHM(y) of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT NL: Long exposure validated exposures. This value is logged at the end of each sequence.
#INS DIMM SEQOUT NVAL: Number of validated exposures for sequence. This value is logged at the end of each sequence.
#INS DIMM SEQOUT RFLRMS: [ratio] Relative rms flux variation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SI: [%] Scintillation Index avg. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SKYBGVAR: Variance on Sky background. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SKYBGi: Sky background of window 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SNR: Signal-to-Noise ratio. This value is logged at the end of each sequence.
#INS DIMM SEQOUT Si: [%] Scintillation Index of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT TAU0: [s] Average coherence time. This value is logged at the end of each sequence.
#INS DIMM SEQOUT THETA0: [arcsec] Isoplanatic angle. This value is logged at the end of each sequence
#INS DIMM SEQOUT TS: [s] Long exposure time. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPARA: [arcsec^2] Variance of differential motion parallel. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPARAN: [arcsec^2] Variance of differential motion parallel (without fitting). This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPERP: [arcsec^2] Variance of differential motion perpendicular. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPERPN: [arcsec^2] Variance of differential motion perpendicular (without fitting). This value is logged at the end of each sequence
#INS DIMM SEQOUT WG: [pixel] Spot Separation (measured). This value is logged at the end of each sequence
#INS DIMM SEQOUT XG: [pixel] Image cdg X (measured). This value is logged at the end of each sequence
#INS DIMM SEQOUT YG: [pixel] Image cdg Y (measured). This value is logged at the end of each sequence
#INS METEO START: Start recording of meteorological data. This action is logged when the interface to Meteo is started (asmmet).
#INS METEO STOP: Start recording of meteorological data. This action is logged when the interface to Meteo is stopped (asmmet).
#TEL POS DEC: [deg] %HOURANG: actual telescope position DEC J2000 from backward calculation. This position is logged at regular intervals.
#TEL POS RA: [deg] %HOURANG: actual telescope position RA J2000 from backward calculation. This position is logged at regular intervals.
#TEL POS THETA: [deg] actual telescope position in THETA E/W. This position is logged at intervals of typically 1 minute.
#TEL PRESET NAME: Name of the target star. This action is logged when presetting to a new target star (asmws).


Filename: F:/ESO/Data/MUSE/DATA_raw/MUSE.2024-01-13T00-43-55.650.fits.fz
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU    1800   (0, 0)      
  1  CHAN23        1 CompImageHDU     99   (4224, 4240)   int16   
  2  CHAN17        1 CompImageHDU     99   (4224, 4240)   int16   
  3  CHAN15        1 CompImageHDU     99   (4224, 4240)   int16   
  4  CHAN11        1 CompImageHDU     99   (4224, 4240)   int16   
  5  CHAN05        1 CompImageHDU     99   (4224, 4240)   int16   
  6  CHAN03        1 CompImageHDU     99   (4224, 4240)   int16   
  7  CHAN22        1 CompImageHDU     99   (4224, 4240)   int16   
  8  CHAN20        1 CompImageHDU     99   (4224, 4240)   int16   
  9  CHAN14        1 CompImageHDU     99   (4224, 4240)   int16   
 10  CHAN10        1 CompImageHDU     99   (4224, 4240)   int16   
 11  CHAN08        1 CompImageHDU     99   (4224, 4240)   int16   
 12  CHAN02        1 CompImageHDU     99   (4224, 4240)   int16   
 13  CHAN21        1 CompImageHDU     99   (4224, 4240)   int16   
 14  CHAN19        1 CompImageHDU     99   (4224, 4240)   int16   
 15  CHAN13        1 CompImageHDU     99   (4224, 4240)   int16   
 16  CHAN09        1 CompImageHDU     99   (4224, 4240)   int16   
 17  CHAN07        1 CompImageHDU     99   (4224, 4240)   int16   
 18  CHAN01        1 CompImageHDU     99   (4224, 4240)   int16   
 19  CHAN24        1 CompImageHDU     99   (4224, 4240)   int16   
 20  CHAN18        1 CompImageHDU     99   (4224, 4240)   int16   
 21  CHAN16        1 CompImageHDU     99   (4224, 4240)   int16   
 22  CHAN12        1 CompImageHDU     99   (4224, 4240)   int16   
 23  CHAN06        1 CompImageHDU     99   (4224, 4240)   int16   
 24  CHAN04        1 CompImageHDU     99   (4224, 4240)   int16   
 25  ASM_DATA      1 BinTableHDU     75   5R x 32C   ['26A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']   
 26  SPARTA_TT_ACQ_IMG    1 CompImageHDU     14   (256, 256)   float32   
 27  SPARTA_ATM_DATA    1 BinTableHDU    103   5R x 46C   [1J, 1J, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E]   
 28  SPARTA_CN2_DATA    1 BinTableHDU     85   1R x 37C   [1J, 1J, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1E]   
 29  SPARTA_TT_CUBE    1 CompImageHDU     15   (40, 40, 100)   float32


SPARTA_ATM_DATA:
'Sec'
'USec'
'LGS1_R0 '
'LGS1_L0 '
'LGS1_SEEING'
'LGS1_FWHM_GAIN'
'LGS1_STREHL'
'LGS1_TURVAR_RES
'LGS1_TURVAR_TOT
'LGS1_TUR_ALT'
'LGS1_TUR_GND'
'LGS2_SLOPERMSX'
'LGS2_SLOPERMSY'
'LGS2_R0'
'LGS2_L0'
'LGS2_SEEING'
'LGS2_FWHM_GAIN'
'LGS2_STREHL'
'LGS2_TURVAR_RES'
'LGS2_TURVAR_TOT'
'LGS2_TUR_ALT'
'LGS2_TUR_GND'
'LGS3_SLOPERMSX'
'LGS3_SLOPERMSY'
'LGS3_R0'
'LGS3_L0'
'LGS3_SEEING'
'LGS1_SLOPERMSX'
'LGS3_FWHM_GAIN'
'LGS3_STREHL'
'LGS3_TURVAR_RES'
'LGS3_TURVAR_TOT'
'LGS3_TUR_ALT'
'LGS3_TUR_GND'
'LGS4_SLOPERMSX'
'LGS4_SLOPERMSY'
'LGS4_R0 '
'LGS4_L0 '
'LGS1_SLOPERMSY'
'LGS4_SEEING'
'LGS4_FWHM_GAIN'
'LGS4_STREHL'
'LGS4_TURVAR_RES'
'LGS4_TURVAR_TOT'
'LGS4_TUR_ALT'
'LGS4_TUR_GND'

f'LGS{}_'
['R0', 'L0', 'SEEING', 'FWHM_GAIN', 'STREHL', 'TURVAR_RES, 'TURVAR_TOT, 'TUR_ALT', 'TUR_GND']

SPARTA_TT_CUBE:

SPARTA_CN2_DATA:
'Sec'
'USec'
'L0Tot'
'r0Tot'
'seeingTot'
'CN2_ALT1'
'CN2_ALT2'
'CN2_ALT3'
'CN2_ALT4'
'CN2_ALT5'
'CN2_ALT6'
'CN2_ALT7'
'CN2_ALT8'
'CN2_FRAC_ALT1'
'CN2_FRAC_ALT2'
'CN2_FRAC_ALT3'
'CN2_FRAC_ALT4'
'CN2_FRAC_ALT5'
'CN2_FRAC_ALT6'
'CN2_FRAC_ALT7'
'CN2_FRAC_ALT8'
'L0_ALT1'
'L0_ALT2'
'L0_ALT3'
'L0_ALT4'
'L0_ALT5'
'L0_ALT6'
'L0_ALT7'
'L0_ALT8'
'ALT1'
'ALT2'
'ALT3'
'ALT4'
'ALT5'
'ALT6'
'ALT7'
'ALT8'

'Sec'
'USec'
'L0Tot'
'r0Tot'
'seeingTot'
f'CN2_ALT{i}'
f'CN2_FRAC_ALT{i}'
f'L0_ALT{i}'
f'ALT{i}'


ASM_DATA:
'TIME_STAMP'
'ASM_WINDDIR_10'
'ASM_WINDDIR_30'
'ASM_WINDSPEED_10'
'ASM_WINDSPEED_30'
'ASM_RFLRMS'
'MASS_FRACGL'
'SLODAR_FRACGL_300'
'SLODAR_FRACGL_500'
'ASM_TAU0'
'IA_FWHMLIN'
'MASS_TURB0'
'MASS_TURB1'
'MASS_TURB2'
'MASS_TURB3'
'MASS_TURB4'
'MASS_TURB5'
'MASS_TURB6'
'SLODAR_CNSQ2'
'SLODAR_CNSQ3'
'SLODAR_CNSQ4'
'SLODAR_CNSQ5'
'SLODAR_CNSQ6'
'SLODAR_CNSQ7'
'SLODAR_CNSQ8'
'IA_FWHMLINOBS'
'SLODAR_TOTAL_CN2'
'R0'
'IA_FWHM'
'AZ'
'AIRMASS'
'DIMM_SEEING'

#%%
parang = hdul_cube[0].header[h+'TEL PARANG END' ]*0.5 + hdul_cube[0].header[h+'TEL PARANG START']*0.5
ALT    = hdul_cube[0].header[h+'TEL ALT']

try:
    if h+'INS DROT START' in hdul_cube[0].header:
        derot_ang = hdul_cube[0].header[h+'INS DROT START']*0.5 + hdul_cube[0].header[h+'INS DROT END']*0.5
    else:
        derot_ang = hdul_cube[0].header[h+'INS DROT END']
except:
    derot_ang = -2*(parang + ALT)

print(f'Parallactic angle: {parang}, altitude: {ALT}, derotator: {derot_ang}')


coumns = ['ID', 'Par', 'Alt', 'Derot', 'Image']

datas = [
    [411,  61.269,  74.639,  -67.835,  27.5],
    [405, -64.6311, 63.126,    0.687, 154.5],
    [146, -33.0085, 72.976, -20.3815, 122.0],
    [296, -60.226,  74.602,   -7.276, 152.0],
    [382, 149.9915, 67.234, -108.553, -58.0],
    [395, -63.0235, 73.528, -5.33955, 153.0],
    [254, -46.5385,  81.99, -17.784,  135.0],
    # [276,  164.417,  72.376,-298.37+180, 120.0-180],
    [281, -42.8125,  65.352, -11.2668, 133.0]
]

ang_df = pd.DataFrame(datas, columns=coumns)
ang_df.set_index('ID', inplace=True)
# ang_df.sort_index(inplace=True)
# -par + 45

testos = ang_df['Image'] - (ang_df['Derot']*2 + 161)
testos = ang_df['Image'] - (161-ang_df['Par']-ang_df['Alt'])
print(testos)

V1 = np.array(ang_df['Par'])
V2 = np.array(ang_df['Alt'])
one = np.ones_like(V1)

V3 = np.array(ang_df['Derot'])
V4 = np.array(ang_df['Image'])

ranger = np.arange(-80, 160, 1)

plt.scatter(V4, V3)
plt.plot(ranger, (ranger-162)/2, 'r')
plt.xlabel('Image')
plt.ylabel('Derotator')

# A = np.vstack([V3, one]).T
A = np.vstack([V1, V2]).T

# Solve the least squares problem
result = np.linalg.lstsq(A, V3, rcond=None)

# Extract the solution (a, b) from the result
B = result[0]

# result = np.linalg.lstsq(A, V3, rcond=None)
print(A@B-V3)


#%% ------------------- Sausage predictor --------------------

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.inspection import plot_partial_dependence


spot_df = df_new_new.copy()

spot_df['Non-point'] = (spot_df['Non-point'] > 0.5).astype(int)  # Binary target

spot_df = spot_df.select_dtypes(exclude=['object'])
spot_df = spot_df.select_dtypes(exclude=['bool'])

spot_df.drop(columns='time', inplace=True)
spot_df.drop(columns='Strehl (header)', inplace=True)

spot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
spot_df = spot_df.map(lambda x: np.nan if pd.isna(x) or abs(x) > np.finfo(np.float64).max else x)
spot_df.dropna(inplace=True)


X = spot_df.drop('Non-point', axis=1)
y = spot_df['Non-point']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose a model
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print("\nPrecision-Recall AUC Score:")
print(pr_auc)

#%%

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Apply SMOTE
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Choose a model and perform hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

model = RandomForestClassifier(class_weight='balanced', random_state=0)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_resampled, y_resampled)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print("\nPrecision-Recall AUC Score:")
print(pr_auc)

#%%
from sklearn.inspection import partial_dependence
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#%%
from sklearn.inspection import permutation_importance

result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=0)

# Plot the results
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Feature Importance")
plt.show()

#%%
from sklearn.inspection import partial_dependence
import seaborn as sns

most_important_feature = 'frequency'

X_train_df = pd.DataFrame(X_train, columns=X.columns)

# Calculate partial dependence
pdp_results = partial_dependence(best_model, X_train_df, [most_important_feature], grid_resolution=50)

# Extract the partial dependence values and axes
pdp_values = pdp_results['average']
pdp_values = pdp_values[0]  # For single feature
pdp_axis   = pdp_results['values'][0]  # For single feature

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(pdp_axis, pdp_values)
plt.xlabel(most_important_feature)

plt.ylabel('Partial Dependence')
plt.title(f'Partial Dependence of {most_important_feature}')
plt.show()


#%%
import shap

X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)

# SHAP dependence plot
shap.dependence_plot(most_important_feature, shap_values[1], X_test_df)

#%%
# Check columns with all NaN values
nan_cols = dfs.columns[dfs.isna().all()]
