#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
# sys.path.insert(0, '.')
sys.path.insert(0, '..')

import pickle
import os
# from os import path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from astropy.coordinates import SkyCoord, AltAz
from astropy.io import fits
import astropy.units as u
from MUSE_data_utils import *

from project_settings import xp


STD_FOLDER   = MUSE_DATA_FOLDER / 'standart_stars/'
CUBES_FOLDER = STD_FOLDER / 'cubes/'
RAW_FOLDER   = STD_FOLDER / 'raw/'
CUBES_CACHE  = STD_FOLDER / 'cached_cubes/'

#%%
def LoadSTDStarCacheByID(id):
    ''' Searches a specific STD star by its ID in the list of cached cubes. '''
    with open(TELEMETRY_CACHE / 'MUSE/muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)

    file = muse_df.loc[muse_df.index == id]['Filename'].values[0]
    full_filename = CUBES_CACHE / f'{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample = pickle.load(handle)
    
    data_sample['All data']['Pupil angle'] = muse_df.loc[id]['Pupil angle']
    return data_sample


def LoadSTDStarData(ids, derotate_PSF=False, normalize=True, subtract_background=False, device=device):
    def get_radial_backround(img):
        ''' Computed STD star background as a minimum of radial profile '''
        from tools.utils import safe_centroid
        from photutils.profiles import RadialProfile
        
        xycen = safe_centroid(img)
        edge_radii = np.arange(img.shape[-1]//2)
        rp = RadialProfile(img, xycen, edge_radii)
        
        return rp.profile.min()
    
    
    def load_sample(id):
        sample = LoadSTDStarCacheByID(id)
  
        PSF_data = np.copy(sample['images']['cube']) 
        PSF_STD  = np.copy(sample['images']['std'])
        
        if subtract_background:
            backgrounds = np.array([ get_radial_backround(PSF_data[i,:,:]) for i in range(PSF_data.shape[0]) ])[:,None,None]
            PSF_data -= backgrounds
            
        else:
            backgrounds = np.zeros(PSF_data.shape[0])[:,None,None]

        if normalize:
            norms = PSF_data.sum(axis=(-1,-2), keepdims=True)
            PSF_data /= norms
            PSF_STD  /= norms
        else:
            norms = np.ones(PSF_data.shape[0])[:,None,None]
    
        config_file, PSF_0 = GetConfig(sample, PSF_0, convert_config=False)
        return PSF_0, PSF_STD, norms, backgrounds, config_file, sample


    PSF_0, configs, norms, bgs = [], [], [], []
    
    for id in ids:
        PSF_0_, _, norm, bg, config_dict_, sample_ = load_sample(id)
        configs.append(config_dict_)
        if derotate_PSF:
            PSF_0_rot = RotatePSF(PSF_0_, -sample_['All data']['Pupil angle'].item())
            PSF_0.append(PSF_0_rot)
        else:
            PSF_0.append(PSF_0_)
            
        norms.append(norm)
        bgs.append(bg)

    PSF_0 = torch.tensor(np.vstack(PSF_0), dtype=default_torch_type, device=device)
    norms = torch.tensor(norms, dtype=default_torch_type, device=device)
    bgs   = torch.tensor(bgs, dtype=default_torch_type, device=device)

    config_manager = ConfigManager()
    merged_config  = config_manager.Merge(configs)

    config_manager.Convert(merged_config, framework='pytorch', device=device)

    merged_config['sources_science']['Wavelength'] = merged_config['sources_science']['Wavelength'][0]
    merged_config['sources_HO']['Height']          = merged_config['sources_HO']['Height'].unsqueeze(-1)
    merged_config['sources_HO']['Wavelength']      = merged_config['sources_HO']['Wavelength'].squeeze()
    merged_config['NumberSources'] = len(ids)
    
    if derotate_PSF:
        merged_config['telescope']['PupilAngle'] = 0.0 # Meaning, that the PSF is already derotated

    return PSF_0, norms, bgs, merged_config


def RenameMUSECubes(folder_cubes_old, folder_cubes_new):
    '''Renames MUSE reduced cubes .fits files according to their exposure date and time'''
    original_cubes_exposure, new_cubes_exposure = [], []
    original_filename, new_filename = [], []

    print(f'Reding cubes in {folder_cubes_new}')
    for file in tqdm(os.listdir(folder_cubes_new)):
        if file == 'renamed':
            continue
        
        with fits.open(os.path.join(folder_cubes_new, file)) as hdul_cube:
            new_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            new_filename.append(file)

    print(f'Reading cubes in {folder_cubes_old}')
    for file in tqdm(os.listdir(folder_cubes_old)):
        with fits.open(os.path.join(folder_cubes_old, file)) as hdul_cube:
            original_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            original_filename.append(file)

    intersection = list(set(original_cubes_exposure).intersection(set(new_cubes_exposure)))

    # Remove files which intersect
    if len(intersection) > 0:
        for exposure in intersection:
            file = new_filename[new_cubes_exposure.index(exposure)]
            file_2_rm = os.path.normpath(os.path.join(folder_cubes_new, file))
            print(f'Removed duplicate: {file_2_rm}')
            os.remove(file_2_rm)

    # Rename files according to the their exposure timestamps (just for convenience)
    renamed_dir = os.path.join(folder_cubes_new, 'renamed')
    if not os.path.exists(renamed_dir):
        os.makedirs(renamed_dir)

    for file in tqdm(os.listdir(folder_cubes_new)):
        # Skip the 'renamed' directory
        if file == 'renamed':
            continue

        with fits.open(os.path.join(folder_cubes_new, file)) as hdul_cube:
            exposure = hdul_cube[0].header['DATE-OBS']

        new_name = 'M.MUSE.' + exposure.replace(':', '-') + '.fits'
        file_2_rm = os.path.normpath(os.path.join(folder_cubes_new, file))
        file_2_mv = os.path.normpath(os.path.join(renamed_dir, new_name))

        # Check if destination file already exists
        if os.path.exists(file_2_mv):
            print(f"Warning: Duplicate file found for {exposure}. Removing {file_2_rm}")
            os.remove(file_2_rm)
        else:
            os.rename(file_2_rm, file_2_mv)

    return renamed_dir


# _ = RenameMUSECubes(CUBES_FOLDER, STD_FOLDER / 'NFM_cubes_temp/')

#%%
if not os.path.exists(match_path := STD_FOLDER / 'files_matches.csv') or not os.path.exists(STD_FOLDER / 'file_mismatches.csv'):
    try:
        files_matches, file_mismatches = MatchRawWithCubes(RAW_FOLDER, CUBES_FOLDER, verbose=True)
        files_matches.to_csv(STD_FOLDER / 'files_matches.csv')
        file_mismatches.to_csv(STD_FOLDER / 'file_mismatches.csv')
    except Exception as e:
        print(f'Error: {e}')
    else:
        print(f'Raw and cubes mathes table is saved at: {match_path}')
else:
    files_matches = pd.read_csv(match_path)
    files_matches.set_index('date', inplace=True)
    file_mismatches = pd.read_csv(STD_FOLDER / 'file_mismatches.csv')
    file_mismatches.set_index('date', inplace=True)
    print(f'Read file matches file from {match_path}')

#%%
# import 
# from tqdm import tqdm

# # Move all files whoch are in file_mismatches raw column to a specified folder:
# for file in tqdm(file_mismatches['raw'].values):
#     shutil.move(RAW_FOLDER+file, STD_FOLDER / 'file_mismatches/')

#%%
if not os.path.exists( exposures_file := STD_FOLDER / 'exposure_times.csv'):
    try:
        exposures_df = GetExposureTimesList(CUBES_FOLDER, verbose=True)
        exposures_df.to_csv(exposures_file)
    except Exception as e:
        print(f'Error: {e}')
    else:
        print(f'Exposure times table is saved at: {exposures_file}')

else:
    exposures_df = pd.read_csv(exposures_file)
    exposures_df.set_index('filename', inplace=True)


#%%
bad_ids = []
# read list of files form .txt file:
# with open(MUSE_RAW_FOLDER / '../bad_files.txt', 'r') as f:
#     files_bad = f.read().splitlines()
# for i in range(len(files_bad)):
#     files_bad[i] = int(files_bad[i])
# for file_id in tqdm(files_bad):
# bad_IRLOS_ids = [68, 242, 316, 390]
# list_ids = [411, 410, 409, 405, 146, 296, 276, 395, 254, 281, 343, 335]
# cube_name = files_matches.iloc[file_id]['cube']

# wvl_bins = None #np.array([478, 511, 544, 577, 606, 639, 672, 705, 738, 771, 804, 837, 870, 903, 935], dtype='float32')
wvl_bins = np.array([
    478.   , 492.125, 506.25 , 520.375, 534.625, 548.75 , 562.875,
    577.   , 606.   , 620.25 , 634.625, 648.875, 663.25 , 677.5  ,
    691.875, 706.125, 720.375, 734.75 , 749.   , 763.375, 777.625,
    792.   , 806.25 , 820.625, 834.875, 849.125, 863.5  , 877.75 ,
    892.125, 906.375, 920.75 , 935.
], dtype='float32')


for file_id in tqdm(range(0, len(files_matches))):
    fname_new = files_matches.iloc[file_id]['cube'].replace('.fits','.pickle').replace(':','-')
    fname_new = f'{file_id}_{fname_new}'

    if fname_new in os.listdir(CUBES_CACHE):
        print(f'File {fname_new} already exists. Skipping...')
        continue

    print(f'\n\n>>>>>>>>>>>>>> Processing file {file_id}...')
    try:
        sample = ProcessMUSEcube(
            path_raw  = os.path.join(RAW_FOLDER,   files_matches.iloc[file_id]['raw' ]),
            path_cube = os.path.join(CUBES_FOLDER, files_matches.iloc[file_id]['cube']),
            crop = True,
            estimate_IRLOS_phase = False,
            impaint_bad_pixels = False,
            extract_spectrum = True,
            wavelength_bins = wvl_bins,
            plot_spectrum = False,
            fill_missing_values = True,
            verbose = True
        )

        with open(CUBES_CACHE / fname_new, 'wb') as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f'Error with file {file_id}: {e}')
        bad_ids.append(file_id)
        continue

print(f'Bad ids: {bad_ids}')


#%% Render the STD stars dataset
for file in tqdm(os.listdir(CUBES_CACHE)):
    try:
        with open(CUBES_CACHE / file, 'rb') as f:
            RenderDataSample(pickle.load(f))
            
        plt.show()
        plt.savefig(STD_FOLDER / f'MUSE_images/{file.replace(".pickle",".png")}')
        
    except:
        continue

#%%
# These two have the same target: CD-38 10980
# file_id = 344
# file_id = 405

# from astroquery.eso import Eso
# eso = Eso()
# eso.login(username='akuznets')

# result_table = eso.query_instrument('muse', column_fil ters={'target': 'NGC 6754'})


# stime = night + 'T19:00:00.00'
# etime = (Time(night).to_datetime()+timedelta(days=1)).date().isoformat() + 'T14:00:00.00'
# eso_query_dict = {'stime':stime,'etime':etime,'dp_cat':'SCIENCE','dp_type':'OBJECT,AO'}

# print(eso_query_dict)


# plt.imshow(np.log(1+np.abs(hdul_cube['DATA'].data[0,...])), cmap='gray')
# plt.axis('off')
# plt.savefig('C:/Users/akuznets/Desktop/thesis_results/MUSE/colored_PSFs_example/unprocessed_PSF_example.pdf', dpi=400)

#%%
data_offaxis = 'F:/ESO/Data/MUSE/wide_field/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits'
hdul_cube = fits.open(data_offaxis)

#%%
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

# Convert numerical format to [hms] and [dms] string formats
alpha_hms = lambda alpha: f"{int(alpha // 10000):02}h{int((alpha % 10000) // 100):02}m{alpha % 100:06.3f}s"
delta_dms = lambda delta: format_dms(*convert_to_dms(delta))

#%%
NGS_alpha = hdul_cube[0].header[h+'AOS NGS ALPHA'] # Alpha coordinate for the NGS, [hms]
NGS_delta = hdul_cube[0].header[h+'AOS NGS DELTA'] # Delta coordinate for the NGS, [dms]

coord_NGS = SkyCoord(alpha_hms(NGS_alpha), delta_dms(NGS_delta), frame='icrs')

ra_NGS, dec_NGS = (coord_NGS.ra.deg, coord_NGS.dec.deg)

#%
targ_alpha = hdul_cube[0].header[h+'TEL TARG ALPHA']
# = 162333.83 / Alpha coordinate for the target
# 162333.83

# 162509.77717
targ_delta = hdul_cube[0].header[h+'TEL TARG DELTA']
# = -391346.1 / Delta coordinate for the target
# -391346.1

# -391700.69955

coord_targ = SkyCoord(alpha_hms(targ_alpha), delta_dms(targ_delta), frame='fk4', equinox='J2000', obstime=start_time, location=UT4_location)
coord_targ = SkyCoord(alpha_hms(targ_alpha), delta_dms(targ_delta), frame='gcrs', obstime=start_time, location=UT4_location)

# coord_targ_dummy = SkyCoord(alpha_hms(targ_alpha), '-40d13m46.100s', frame='icrs')


# print( coord_targ.separation(coord_targ_dummy).degree )

#%%
# ra_targ  = hdul_cube[0].header['RA']
# dec_targ = hdul_cube[0].header['DEC']

'''
RA  = 245.892843 / [deg]  16:23:34.2 RA  (J2000) pointing
DEC =  -39.23000 / [deg] -39:13:48.0 DEC (J2000) pointing
'''
# coord_targ = SkyCoord(ra=ra_targ*u.deg, dec=dec_targ*u.deg, frame='icrs')

# #%
tel_alt = hdul_cube[0].header[h+'TEL ALT']
tel_az  = hdul_cube[0].header[h+'TEL AZ']

altaz = AltAz(alt=tel_alt*u.deg, az=tel_az*u.deg, location=UT4_location, obstime=start_time)

# coord_VLT = SkyCoord(altaz, frame='altaz', obstime=start_time)
coord_VLT = SkyCoord(altaz, frame='altaz', obstime=start_time)

#%

tel_delta = hdul_cube[0].header[h+'INS ADC1 DEC']
# = -391700.69955 / [deg] Telescope declination
tel_alpha = hdul_cube[0].header[h+'INS ADC1 RA']
# = 162509.77717 / [deg] Telescope right ascension
coord_tel = SkyCoord(alpha_hms(tel_alpha), delta_dms(tel_delta), frame='icrs')

# coord_VLT = coord_tel


#%
# Extract RA and Dec in degrees
ra_VLT  = coord_VLT.icrs.ra.deg
dec_VLT = coord_VLT.icrs.dec.deg

print( coord_VLT.separation(coord_targ).degree )
print( coord_VLT.separation(coord_NGS).degree )
print( coord_targ.separation(coord_NGS).degree )



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
