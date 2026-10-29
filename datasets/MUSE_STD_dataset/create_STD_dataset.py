#%%
try:
    ipy = get_ipython() # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess

import pandas as pd
from STD_dataset_utils import *
# The distributed calibrator bundle is built up here: telemetry resources (stage 1), telemetry preprocessor (stage 2), see NFM_bundle
from data_processing.NFM_bundle import STD_RESOURCES_FOLDER, BUNDLE_PATH, pack_telemetry_resources, pack_telemetry_preprocessor

#%%
# with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
#     muse_df = pickle.load(handle)
    
# #%%
# with open(STD_FOLDER / 'muse_STD_stars_telemetry.pickle', 'rb') as handle:
#     # reduced_telemetry = pickle.load(handle)['telemetry imputed df']
#     reduced_telemetry = pickle.load(handle)['telemetry normalized imputed df']
   
# #%%
# cn2_1 = 1 - reduced_telemetry['Cn2_frac_binned_2'] - reduced_telemetry['Cn2_frac_binned_3']
# cn2_2 = reduced_telemetry['Cn2_frac_binned_2']
# cn2_3 = reduced_telemetry['Cn2_frac_binned_3']

# fig, ax = plt.subplots(figsize=(20, 15))
# ax.hist(cn2_1, bins=100, alpha=0.5, label='Cn2_frac_1')
# ax.hist(cn2_2, bins=100, alpha=0.5, label='Cn2_frac_2')
# ax.hist(cn2_3, bins=100, alpha=0.5, label='Cn2_frac_3')
# ax.legend()
# plt.show()

#%%
# Check the existence of all necessary folders and create them if they don't exist
# STD_RESOURCES_FOLDER: working copies of the LGS / IRLOS archive tables (the datalab fetches are appended to them) and of the Cn2 grid
STD_stars_folders = [CUBES_FOLDER, RAW_FOLDER, STD_FOLDER, CUBES_CACHE, STD_RESOURCES_FOLDER, STD_FOLDER / 'NFM_cubes_temp/']
for folder in STD_stars_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Created folder: {folder}')
    else:
        print(f'Folder already exists: {folder}')

#%%
_ = RenameMUSECubes(CUBES_FOLDER, STD_FOLDER / 'NFM_cubes_temp/')

#%%
if not os.path.exists(match_path := STD_FOLDER / 'files_matches.csv') or not os.path.exists(STD_FOLDER / 'files_mismatches.csv'):
    try:
        files_matches, files_mismatches = MatchRawWithCubes(RAW_FOLDER, CUBES_FOLDER, verbose=True)
        files_matches = files_matches.sort_index()
        files_mismatches = files_mismatches.sort_index()
        files_matches.to_csv(STD_FOLDER / 'files_matches.csv')
        files_mismatches.to_csv(STD_FOLDER / 'files_mismatches.csv')
    except Exception as e:
        print(f'Error: {e}')
    else:
        print(f'Raw and cubes mathes table is saved at: {match_path}')
else:
    files_matches = pd.read_csv(match_path)
    files_matches.set_index('date', inplace=True)
    files_matches = files_matches.sort_index()
    files_mismatches = pd.read_csv(STD_FOLDER / 'files_mismatches.csv')
    files_mismatches.set_index('date', inplace=True)
    files_mismatches = files_mismatches.sort_index()
    print(f'Read the existing raw and cubes matches file from {match_path}')

#%
# # Move all files which are in file_mismatches raw column to a specified folder:
# for file in tqdm(file_mismatches['raw'].values):
#     shutil.move(RAW_FOLDER+file, STD_FOLDER / 'file_mismatches/')

#%
# if not os.path.exists( exposures_file := STD_FOLDER / 'exposure_times.csv'):
#     try:
#         exposures_df = GetExposureTimesList(CUBES_FOLDER, verbose=True)
#         exposures_df.to_csv(exposures_file)
#     except Exception as e:
#         print(f'Error: {e}')
#     else:
#         print(f'Exposure times table is saved at: {exposures_file}')

# else:
#     exposures_df = pd.read_csv(exposures_file)
#     exposures_df.set_index('filename', inplace=True)

#%% ================================ Cache MUSE NFM STD stars cubes ================================
bad_files = []

rewrite = False
# rewrite = True

verbose = False

ids_process = range(0, len(files_matches))

# Cached files are named '{ID}_{cube_name}.pickle', where ID may change between runs,
# so existing cubes are identified by the name part only
cached_names = { f.split('_', 1)[-1] for f in os.listdir(CUBES_CACHE) }

for file_id in tqdm(ids_process):
    cube_name = files_matches.iloc[file_id]['cube'].replace('.fits','.pickle').replace(':','-')
    fname_new = f'{file_id}_{cube_name}'

    if cube_name in cached_names and not rewrite:
        if verbose: print(f'Cube {cube_name} already cached. Skipping...')
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
        cached_names.add(cube_name)

    except Exception as e:
        if verbose: print(f'Error with file {fname_new}: {e}')
        bad_files.append(fname_new)
        continue

print(f'Failed files: {bad_files}')

#%%
# Ensure unique sample IDs, assign IDs based on observation date
RemoveDuplicatedCubes(CUBES_CACHE, verbose=True)
ReasignIDsByDate(CUBES_CACHE, verbose=True)

#%% ================================ Render the STD stars dataset ================================
# IDs can be reassigned by date, so existing images are matched by the name part only
IMAGES_FOLDER = STD_FOLDER / 'MUSE_images'
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
rendered_names = { f.split('_', 1)[-1] for f in os.listdir(IMAGES_FOLDER) if f.endswith('.png') }

for file in tqdm(os.listdir(CUBES_CACHE)):
    fx = file.replace(".pickle",".png")
    if fx.split('_', 1)[-1] in rendered_names and not rewrite:
        if verbose: print(f'File {fx} already exists. Skipping...')
        continue
    try:
        with open(CUBES_CACHE / file, 'rb') as f:
            RenderDataSample(pickle.load(f)[0], file)
            
        plt.savefig(STD_FOLDER / f'MUSE_images/{file.replace(".pickle",".png")}')
        plt.close()
        rendered_names.add(fx.split('_', 1)[-1])
        
    except Exception as e:
        if verbose: print(f'{e}')
        continue

#%% ================================ Label PSFs ================================
# Re-sync labels.txt ID prefixes to match the current state of the cache after the previous renaming
if (STD_FOLDER / 'labels.txt').exists():
    SyncLabelsToCubes(STD_FOLDER / 'labels.txt', CUBES_CACHE, verbose=True)

# Execute the data labeler script
result = subprocess.run([sys.executable, 'STD_stars_labeler.py'],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__)))

if result.returncode != 0:
    print(f"Error executing data_labeler.py: {result.stderr}")
else:
    print(f"data_labeler.py executed successfully: {result.stdout}")


#%% ================================ Assemble STD stars reduced telemetry dataset ================================

# Compose dataset of MUSE NFM reduced telemetry values based on the data associated with cahed data cubes
if not os.path.exists(STD_FOLDER / 'muse_df.pickle'):
    # Load labels information
    all_labels = []
    labels_df  = { 'ID': [], 'Filename': [] }

    if os.path.exists(labels_path := STD_FOLDER / 'labels.txt'):
        with open(labels_path, 'r') as f:
            for line in f:
                filename, labels = line.strip().split(': ')

                ID = filename.split('_')[0]
                pure_filename = filename.replace(ID+'_', '').replace('.png', '')

                labels_df['ID'].append(int(ID))
                labels_df['Filename'].append(pure_filename)
                all_labels.append(labels.split(', '))
    else:
        raise ValueError('Labels file does not exist!')

    labels_list = list(set( [x for xs in all_labels for x in xs] ))
    labels_list.sort()

    for i in range(len(labels_list)):
        labels_df[labels_list[i]] = []

    for i in range(len(all_labels)):
        for label in labels_list:
            labels_df[label].append(label in all_labels[i])

    labels_df = pd.DataFrame(labels_df)
    labels_df.set_index('ID', inplace=True)
    labels_df.sort_index(inplace=True)

    # Read reduced telemetry data
    muse_df = []
    files = os.listdir(processed_path := STD_FOLDER / 'cached_cubes/')

    for file in tqdm(files):
        with open(processed_path / file, 'rb') as f:
            data_sample, _ = pickle.load(f)
            df_ = data_sample['All data']
            df_['name'] = df_['name'].apply(lambda x: x.replace('.fits',''))
            df_['ID'] = int(file.split('_')[0])
            muse_df.append(df_)
            
    muse_df = pd.concat(muse_df)
    muse_df.set_index('ID', inplace=True)
    muse_df.sort_index(inplace=True)
    muse_df = muse_df.join(labels_df)
    muse_df = add_kc_mismatch_flag(muse_df) # samples recorded with a different DM correction radius (k_c)

    with open(STD_FOLDER / 'muse_df.pickle', 'wb') as handle:
        pickle.dump(muse_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print(f'Loading cached MUSE dataframe from {STD_FOLDER / "muse_df.pickle"}')
    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)


(median_Cn2_alts, median_Cn2_fracs), _, _ = AOF_Cn2_profiles_stats(muse_df) # median AOF Cn2 profile
# Fixed Cn2 grid of the 'grid' binning: 3 layers at the dataset medians of the per-sample K-means altitudes, the same for all samples
Cn2_grid = store_Cn2_grid(muse_df, median_Cn2_profile=(median_Cn2_alts, median_Cn2_fracs))

#%% ================================= Calibrator bundle, stage 1: telemetry resources ==================================
from data_processing.MUSE_data_utils import CN2_BINNING

# The LGS / IRLOS archive tables (working copies in STD_RESOURCES_FOLDER, filled while caching the cubes), the median Cn2 profile and
# the Cn2 binning (with its grid) go into the distributed bundle: from here on, InitNFMConfig / reduce_dataframe read them from it,
# as the users do

pack_telemetry_resources(
    median_Cn2_profile = (median_Cn2_alts, median_Cn2_fracs),
    Cn2_binning        = CN2_BINNING,
    Cn2_grid           = Cn2_grid if CN2_BINNING == 'grid' else None, # the grid is stored only when the features / configs use it
)

print(f'Telemetry resources packed -> {BUNDLE_PATH}')

#%% ================================= MUSE NFM STD stars dataset cleaning, imputation and scaling ==================================
from data_processing.data_analysis_utils import *

verbose = True

# Filter and reduce the telemetry dataframe, fit the scaler and the iterative imputer (see fit_telemetry_preprocessor). The scaler and
# the imputer are merged into a single lightweight TelemetryPreprocessor to process new (e.g. science) telemetry the same way;
# it goes into the distributed bundle (stage 2), into the dataset cache and from there into every trained calibrator bundle
muse_df_pruned, muse_df_pruned_scaled, muse_df_pruned_scaled_imputed, muse_df_pruned_imputed, telemetry_preprocessor = \
    fit_telemetry_preprocessor(muse_df, Cn2_binning=CN2_BINNING, verbose=verbose) # same reduction as the configs (MUSE_data_utils.CN2_BINNING)

# Calibrator bundle, stage 2: the telemetry preprocessor (stage 3, the calibrator itself, comes from train_NFM_calibrator.py --install)
pack_telemetry_preprocessor(telemetry_preprocessor.state_dict())
print(f'Telemetry preprocessor packed -> {BUNDLE_PATH}')

numeric_cols = muse_df_pruned.select_dtypes(include="number").columns
nan_mask     = muse_df_pruned[numeric_cols].isna().values  # (N_samples, N_numeric_cols)

#%%
if verbose:
    _ = plt.hist(muse_df_pruned_scaled[numeric_cols].values.flatten(), bins=100)
    plt.title('Distribution of standartized features')
    plt.show()
    
    plot_data_filling(muse_df_pruned)
    print(f'Total number of remaining features in pruned telemetry dataset: {len(muse_df_pruned.columns)}')

# if verbose:
    # for entry in muse_df_pruned.columns.values:
    #     sns.displot(data=muse_df_pruned, x=entry, kde=True, bins=100)
    #     plt.show()

#%%
# Imputed values beyond ±3 std are NOT replaced with medians anymore: the benchmark shows that they are mostly
# correct and the replacement increases the imputation error. Besides, it can't be reproduced at inference time
if verbose:
    _ = analyze_outliers(muse_df_pruned_scaled_imputed, outlier_threshold=3, nan_mask=nan_mask, verbose=verbose)

    muse_df_pruned_scaled_imputed.plot.hist(bins=100, alpha=0.5, figsize=(20, 15))
    plt.legend(ncol=4, fontsize=8, loc='upper right')
    plt.title('Distribution of normalized features per feature')
    plt.show()
    
    corr = plot_correlation_heatmap(muse_df_pruned_scaled, verbose=verbose)
    columns_to_drop = filter_by_correlation(corr, threshold=0.9, verbose=verbose)
    
    nan_percentages = analyze_NaN_distribution(muse_df_pruned)
    
    vif_results = calculate_VIF(muse_df_pruned_scaled)
    print(vif_results)
    # VIF_contributors(muse_df_pruned_scaled, 'NGS mag (from ph.)')
    

#%%
# Pack all data into a single dictionary and store it (for debugging, the training uses the dataset cache below)
reduced_telemetry = {
    'telemetry normalized imputed df': muse_df_pruned_scaled_imputed,
    'telemetry imputed df': muse_df_pruned_imputed,
    'telemetry preprocessor': telemetry_preprocessor.state_dict(),
    'Cn2 grid': Cn2_grid.state_dict() if CN2_BINNING == 'grid' else None, # the grid is stored only when the features use it
}

with open(STD_FOLDER / 'muse_STD_stars_telemetry.pickle', 'wb') as handle:
    pickle.dump(reduced_telemetry, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%
# Now, one must run the fitting routine to get the fitted PSF parameters values for each PSF sample
# 1. launch 'launch_fitting.py' to run the fitting jobs in parallel on multiple GPUs
# 2. Run 'STD_fitted_df.py' to read the fitted parameters, analyze the fitting results and store the final fitted parameters dataframe

#%%
# ================================ Load MUSE-NFW STD stars dataset cache ================================
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

with open(STD_FOLDER / 'muse_STD_stars_telemetry.pickle', 'rb') as handle:
    _telemetry_data = pickle.load(handle)
    reduced_telemetry = _telemetry_data['telemetry normalized imputed df']
    telemetry_preprocessor_state = _telemetry_data['telemetry preprocessor'] # scaler + imputer to process new telemetry
    Cn2_grid_state = _telemetry_data.get('Cn2 grid') # the Cn2 grid of the telemetry features (and of the configs, via InitNFMConfig)

with open(STD_FOLDER / 'muse_fitted_df.pickle', 'rb') as handle:
    muse_fitted_df = pickle.load(handle)['fitted_values']

# Detect the samples recorded with a different DM correction radius (k_c): scans only the fits not scored yet, appends the new
# detections (with their k_c percentage) to STD_FOLDER / 'kc_mismatch_labels.txt' and refreshes the 'k_c mismatch' / 'k_c fraction'
# columns of muse_df. See kc_mismatch.py; --force re-scores everything after re-fitting, --scan refines the k_c of the labeled samples by fits.

from kc_mismatch import update_kc_mismatch_labels

_ = update_kc_mismatch_labels(muse_df)

muse_df = add_kc_mismatch_flag(muse_df)

with open(STD_FOLDER / 'muse_df.pickle', 'wb') as handle:
    pickle.dump(muse_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

DATASET_CACHE = STD_FOLDER / 'dataset_cache'

# Derotated PSFs or raw frames with the per-sample pupil angle in the configs, rotated by the PSF model (see build_dataset_cache). Must match
# the regime of the fits (STD_fitter.py DEROTATE_PSF, raw frames since 2026-10-20) and of the training (derotated_PSFs in the calibrator
# configs). Both regimes are written to 'muse_STD_stars_dataset.pt', the cache stores its regime in the 'derotated' entry (see
# dataset_cache_path / load_STD_dataset in machine_learning/calibrators/NFM_calibrator.py)
DEROTATE_PSF = False

#%%
# Pack all data into the coherent dataset, where reduced telemetry is associated with spectral cubes and fitted PSF model parameters
# for each sample (see build_dataset_cache: good samples only, normalized PSFs, time-grouped train / val split)
good_ids, bad_ids = build_dataset_cache(
    muse_df, reduced_telemetry, muse_fitted_df, telemetry_preprocessor_state,
    out_path       = DATASET_CACHE / 'muse_STD_stars_dataset.pt',
    Cn2_grid_state = Cn2_grid_state,
    verbose        = True,
    derotate_PSF   = DEROTATE_PSF
)

if len(bad_ids) > 0:
    print('Bad ids:')
    for bad_id in bad_ids:
        print(bad_id)


#%%
'''
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

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

NGS_alpha = hdul_cube[0].header[h+'AOS NGS ALPHA'] # Alpha coordinate for the NGS, [hms]
NGS_delta = hdul_cube[0].header[h+'AOS NGS DELTA'] # Delta coordinate for the NGS, [dms]

coord_NGS = SkyCoord(alpha_hms(NGS_alpha), delta_dms(NGS_delta), frame='icrs')

ra_NGS, dec_NGS = (coord_NGS.ra.deg, coord_NGS.dec.deg)

targ_alpha = hdul_cube[0].header[h+'TEL TARG ALPHA']
targ_delta = hdul_cube[0].header[h+'TEL TARG DELTA']

coord_targ = SkyCoord(alpha_hms(targ_alpha), delta_dms(targ_delta), frame='fk4', equinox='J2000', obstime=start_time, location=UT4_location)
coord_targ = SkyCoord(alpha_hms(targ_alpha), delta_dms(targ_delta), frame='gcrs', obstime=start_time, location=UT4_location)

# coord_targ_dummy = SkyCoord(alpha_hms(targ_alpha), '-40d13m46.100s', frame='icrs')
# print( coord_targ.separation(coord_targ_dummy).degree )

# ra_targ  = hdul_cube[0].header['RA']
# dec_targ = hdul_cube[0].header['DEC']

tel_alt = hdul_cube[0].header[h+'TEL ALT']
tel_az  = hdul_cube[0].header[h+'TEL AZ']

altaz = AltAz(alt=tel_alt*u.deg, az=tel_az*u.deg, location=UT4_location, obstime=start_time)

coord_VLT = SkyCoord(altaz, frame='altaz', obstime=start_time)

tel_delta = hdul_cube[0].header[h+'INS ADC1 DEC']
tel_alpha = hdul_cube[0].header[h+'INS ADC1 RA']
coord_tel = SkyCoord(alpha_hms(tel_alpha), delta_dms(tel_delta), frame='icrs')

# Extract RA and Dec in degrees
ra_VLT  = coord_VLT.icrs.ra.deg
dec_VLT = coord_VLT.icrs.dec.deg

print( coord_VLT.separation(coord_targ).degree )
print( coord_VLT.separation(coord_NGS).degree )
print( coord_targ.separation(coord_NGS).degree )
'''

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
