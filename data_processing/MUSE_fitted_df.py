#%%
import sys
sys.path.append('..')

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from tools.normalizers import Uniform0_1

from data_processing.MUSE_STD_dataset_utils import STD_FOLDER, CUBES_CACHE
from data_processing.MUSE_data_utils import wvl_bins

# Select every 2nd wvl
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 2), a_min=0, a_max=N_wvl_max-1)
wavelength = np.round(wvl_bins[wvl_ids]).astype(int)
norm_wvl = Uniform0_1(a=475.e-9, b=935.e-9) # MUSE NFM wavelength range

#%%
IRLOS_phases_dict = {}

for file in tqdm(os.listdir(CUBES_CACHE)):
    with open(CUBES_CACHE / file, 'rb') as handle:
        data,_ = pickle.load(handle)

    if 'IRLOS phases' in data.keys():
        if data['IRLOS phases']['OPD aperture'] is not None:
            IRLOS_phases_dict[int(file.split('_')[0])] = data['IRLOS phases']['OPD aperture'].mean(axis=0)

# Save phases to disk
if len(IRLOS_phases_dict) > 0:
    with open(STD_FOLDER / 'IRLOS_phases_dict.pkl', 'wb') as handle:
        pickle.dump(IRLOS_phases_dict, handle)
else:
    print("No IRLOS phases found in the dataset.")


#%%
# experiment = '~old/fitted_λw_0,5_1,0'
# experiment = '~old/fitted_λw_1,0_0,5' #-> seems to be better
# experiment = '~old/fitted_λw_all_1,0'
experiment = 'fitted'

fitted_samples_folder = STD_FOLDER / experiment


files = os.listdir(fitted_samples_folder)

with open(fitted_samples_folder / files[0], 'rb') as handle:
    data = pickle.load(handle)
    for x in data.keys():
        print(x)


df_relevant_entries = [
    'F', 'J', 'dx', 'dy', 'bg', 'chrom_defocus',
    'bg_ctrl', 'F_ctrl', 'dx_ctrl', 'dy_ctrl', 'J_ctrl', 'chrom_defocus_ctrl',
    'LO_coefs',
    'Cn2_weights',
    'r0', 'dn', 'wind_speed_single', 'L0',
    'amp', 'b', 'alpha', 'beta', 'ratio', 'theta',
    'SR data', 'SR fit', 'FWHM fit', 'FWHM data',
    'input_vec'
]

df_relevant_entries = list(set(df_relevant_entries).intersection(set(data.keys())))
df_relevant_entries.sort()

#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in df_relevant_entries}

ids, images_data, images_fitted = [], [], []

for file in tqdm(files):
    id = int(file.split('.')[0])

    with open(fitted_samples_folder / file, 'rb') as handle:
        data = pickle.load(handle)
    
    images_data.append(   data['Img. data'] )
    images_fitted.append( data['Img. fit']  )
    
    for key in fitted_dict_raw.keys():
        fitted_dict_raw[key].append(data[key])
    ids.append(id)

#%%
xs = []
for i in range(len(ids)):
    if fitted_dict_raw['input_vec'][i].shape[-1] == fitted_dict_raw['input_vec'][0].shape[-1]:
        xs.append(fitted_dict_raw['input_vec'][i][0])
xs = np.stack(xs)

x_median = np.median(xs, axis=0)

for i in range(xs.shape[1]):
    data_ = xs[:, i]
    
    if np.abs(data_).sum() < 1e-12:
        continue
    
    plt.figure(figsize=(10, 6))
    _ = plt.hist(data_.flatten(), bins=30)
    plt.axvline(x_median[i], color='r', linestyle='dashed', linewidth=1)
    plt.title(f'Input vector component {i}')
    plt.show()

#%%
# Store images
with open(fsave := (STD_FOLDER / 'MUSE_fitted_images_data.pickle'), 'wb') as handle:
    images_dict = {
        'data': images_data,
        'fitted': images_fitted,
        'ID': ids
    }
    pickle.dump(images_dict, handle)
    print(f"Saved {len(images_data)} images to {fsave}")

#%%
singular_dict = {}
singular_entries = list( set([
    'r0', 'dn', 'Jxy', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta', 'L0', 'wind_speed_single'
]).intersection( df_relevant_entries) )

for key in singular_entries:
    singular_dict[key] = np.squeeze(np.array(fitted_dict_raw[key])).tolist()    

spline_entries = [entry for entry in df_relevant_entries if entry.endswith('_ctrl')]

N_wvl_ctrl = fitted_dict_raw['J_ctrl'][0].shape[-1] # number of spline points
λ_ctrl = np.linspace(0, 1, N_wvl_ctrl) # normalized wavelenths space ∈ [0, 1]

wavelengths_nodes = norm_wvl.inverse(λ_ctrl)*1e9 # wavelengths at the spline control points in [nm]

#%%
def evaluate_splines(y_points, λ_grid):
    spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t=λ_ctrl, x=y_points.T))
    return spline.evaluate(λ_grid).T


def make_singulars_dataset(dict_):
    dict_['ID'] = np.array(ids)
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)
    return df


def make_polychrome_dataset(data):
    dict_ = {}
    for i, wvl in enumerate(wavelength):
        dict_[wvl] = np.squeeze(np.stack(data))[:,i].tolist()

    dict_['ID'] = np.array(ids).tolist()
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)
    return df


def make_polychrome_dataset_spline(inp_dict, entry):
    if entry+'_ctrl' in spline_entries:
        data = inp_dict[entry+'_ctrl']
        wavelength_ = np.round(wavelengths_nodes).astype(int)
    else:
        wavelength_ = wavelength
        data = inp_dict[entry]
        
    dict_ = {}
    for i, wvl in enumerate(wavelength_):
        dict_[wvl] = np.squeeze(np.stack(data))[:,i].tolist()

    dict_['ID'] = np.array(ids).tolist()
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)
    return df


def make_LO_dataset(data):
    data_ = np.squeeze(np.stack(data))
    dict_ = {}
    for i in range(data_.shape[1]):
        dict_[f'Z{i+1}'] = data_[:,i].tolist()

    dict_['ID'] = np.array(ids).tolist()
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)

    df.rename(columns={'Z1': 'Phase bump'}, inplace=True)
    return df


def make_Cn2_weights_dataset(data):
    data_ = np.squeeze(np.stack(data))
    dict_ = {}
    for i in range(data_.shape[1]):
        dict_[f'Cn2_w{i+1}'] = data_[:,i].tolist()

    dict_['ID'] = np.array(ids).tolist()
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)

    # Delete Cn2_w1 column since weights sum up to 1 anyway
    df.drop(columns=['Cn2_w1'], inplace=True)
    return df


#%
singular_df = make_singulars_dataset(singular_dict)

J_df  = make_polychrome_dataset_spline(fitted_dict_raw, 'J')
dx_df = make_polychrome_dataset_spline(fitted_dict_raw, 'dx')
dy_df = make_polychrome_dataset_spline(fitted_dict_raw, 'dy')
bg_df = make_polychrome_dataset_spline(fitted_dict_raw, 'bg')
F_df  = make_polychrome_dataset_spline(fitted_dict_raw, 'F')

Cn2_w_df = make_Cn2_weights_dataset(fitted_dict_raw['Cn2_weights'])
LO_df    = make_LO_dataset(fitted_dict_raw['LO_coefs'])

SR_fit_df  = make_polychrome_dataset(fitted_dict_raw['SR fit'])
SR_data_df = make_polychrome_dataset(fitted_dict_raw['SR data'])

FWHM_fit_data  = make_polychrome_dataset(fitted_dict_raw['FWHM fit'])
FWHM_data_data = make_polychrome_dataset(fitted_dict_raw['FWHM data'])

FWHM_fit_x   = FWHM_fit_data.map(lambda x: x[0])
FWHM_fit_y   = FWHM_fit_data.map(lambda x: x[1])
FWHM_data_x  = FWHM_data_data.map(lambda x: x[0])
FWHM_data_y  = FWHM_data_data.map(lambda x: x[1])

FWHM_fit_df  = FWHM_fit_x.apply(lambda x: x**2)  + FWHM_fit_y.apply(lambda x: x**2)
FWHM_data_df = FWHM_data_x.apply(lambda x: x**2) + FWHM_data_y.apply(lambda x: x**2)

FWHM_fit_df  = FWHM_fit_df.apply(np.sqrt)
FWHM_data_df = FWHM_data_df.apply(np.sqrt)

# Postprocess some values
J_df = J_df.abs()
if 'alpha' in singular_df.columns:
    singular_df['alpha'] = singular_df['alpha'].apply(lambda x: np.abs(x))

singular_df['r0'] = singular_df['r0'].apply(lambda x: np.abs(x))
singular_df[singular_df['r0'] > 1] = np.nan

#%%
# Preparing values for storage
def collapse_dataframe_to_lists(df, name):
    """Convert DataFrame rows to numpy arrays and set column name."""
    collapsed = df.apply(lambda row: row.tolist(), axis=1)
    collapsed.name = name
    return collapsed

to_collapse = [
    (F_df,  'F'),
    (J_df,  'J'),
    (dx_df, 'dx'),
    (dy_df, 'dy'),
    (LO_df, 'LO_coefs'),
    (bg_df, 'bg'),
    (Cn2_w_df, 'Cn2_weights')
]
fitted_values_df = pd.concat([collapse_dataframe_to_lists(df, name) for df, name in to_collapse] + [singular_df], axis=1)


# Store dataframes
with open(STD_FOLDER / 'muse_fitted_df.pickle', 'wb') as handle:
    pickle.dump(
        {
            'fitted_values': fitted_values_df,
            'FWHM_fit_df':   FWHM_fit_df,
            'FWHM_data_df':  FWHM_data_df,
            'SR_fit_df':     SR_fit_df,
            'SR_data_df':    SR_data_df,
        },
        handle
    )

#%%
# Plot histograms for every J wavelengths
def plot_param_histograms_wvls(df, name):
    a_s, b_s = [], []
    for i in df.columns:
        plt.figure(figsize=(10, 6))
        df[i].hist(bins=30)
        plt.axvline(med := df[i].median(), color='r', linestyle='dashed', linewidth=1, label='Median')
        plt.axvline(med + (std := df[i].std()), color='y', linestyle='dashed', linewidth=1, label='+1 Std')
        plt.axvline(med - std, color='y', linestyle='dashed', linewidth=1, label='-1 Std')
        plt.legend()
        plt.title(f'{name} at {i} nm, median: {med:.3g}, std: {std:.3g}, a = {med - std:.3g}, b = {med + std:.3g}')
        plt.show()
        a_s.append(med - std)
        b_s.append(med + std)

    a_s = np.array(a_s)
    b_s = np.array(b_s)
    print(f'Suggested normalization parameters for {name}: a={a_s.mean():.4f}, b={b_s.mean():.4f}')


def plot_param_histograms_singular(df):
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        data_ = df[col]
        data_.hist(bins=30)
        med = data_.median()
        std = data_.std()

        plt.axvline(med, color='r', linestyle='dashed', linewidth=1, label='Median')
        plt.axvline(med + std, color='y', linestyle='dashed', linewidth=1, label='+1 Std')
        plt.axvline(med - std, color='y', linestyle='dashed', linewidth=1, label='-1 Std')
        plt.legend()
        plt.title(f'{col}, median: {med:.3g}, std: {std:.3g}, a = {med - std:.3g}, b = {med + std:.3g}')
        plt.show()
        print(f'Suggested normalization parameters for {col}: a={med - std:.4f}, b={med + std:.4f}')


def plot_spline_param_distribution(df, wavelengths, title='Strehl Ratio Distribution'):
    x_ = df.to_numpy()
    x_median = np.median(x_, axis=0)
    x_le = np.percentile(x_, 15.87, axis=0)  # -1 std
    x_ue = np.percentile(x_, 84.13, axis=0)  # +1 std

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, x_median, label='Median')
    plt.fill_between(wavelengths, x_le, x_ue, alpha=0.3, label='1-sigma percentile')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Strehl Ratio')
    plt.title(title)
    plt.legend()
    plt.show()


#%%
plot_param_histograms_wvls(J_df, 'J')
#%%
plot_param_histograms_wvls(LO_df, 'LO_coefs')
#%%
plot_param_histograms_singular(singular_df)
#%%
plot_spline_param_distribution(J_df, norm_wvl.inverse(λ_ctrl)*1e9, title='J Strehl Ratio Distribution')
#%%
print("LO NCPAs coefs stats:")
print(LO_df.describe().T)

#%%
from tools.utils import r0, seeing

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

#%%

'L0Tot',
'r0Tot',
'seeingTot'
'Seeing (header)'


id_max = 224
time_before = muse_df.iloc[id_max].time

x = muse_df['r0Tot'].to_numpy()
y = r0(muse_df['seeingTot'], 500e-9).to_numpy()
y[:id_max] *= 1.17 # Ad hox corrective coefficient

plt.scatter(x, y)
plt.plot([0, 0.3], [0, 0.3], 'r--')

#%%
r0_fitted = singular_df['r0']
r0_data = r0(muse_df['Seeing (header)'], 500e-9)
r0_data = r0_data.reindex(r0_fitted.index)

sns.scatterplot(x=r0_data, y=r0_fitted)
plt.plot([0, 0.3], [0, 0.3], 'r--')
plt.xlabel('r0 from seeing [m]')
plt.ylabel('Fitted r0 [m]')
plt.title('Fitted vs Seeing-derived r0')
plt.show()

#%%
# Same but for L0
L0_fitted = singular_df['L0']
L0_data = muse_df['L0Tot']
L0_data = L0_data.reindex(L0_fitted.index)
sns.scatterplot(x=L0_data, y=L0_fitted)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('L0 from header [m]')
plt.ylabel('Fitted L0 [m]')
plt.title('Fitted vs Header L0')
plt.show()

#%% Same for the wind speed
wind_fitted = singular_df['wind_speed_single']
wind_data = muse_df['Wind speed (header)']
wind_data = wind_data.reindex(wind_fitted.index)
sns.scatterplot(x=wind_data, y=wind_fitted)
plt.plot([0, 30], [0, 30], 'r--')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel('Wind speed from header [m/s]')
plt.ylabel('Fitted wind speed [m/s]')
plt.title('Fitted vs Header wind speed')
plt.show()

#%%
# from collections import Counter

# # Check lengths of arrays in fitted_dict_raw['SR fit'])
# # Check lengths of arrays in fitted_dict_raw['SR fit']
# length_counter = Counter(len(arr) for arr in fitted_dict_raw['FWHM fit'])
# print(f"Array length distribution in 'SR fit': {dict(length_counter)}")
# print(f"Expected length: {len(wavelength)}")

# for length, count in length_counter.items():
#     if length != len(wavelength):
#         print(f"Found {count} arrays with length {length} (expected {len(wavelength)})")
#         # Optionally print the IDs with mismatched lengths
#         mismatched_ids = [ids[i] for i, arr in enumerate(fitted_dict_raw['FWHM fit']) if len(arr) == length]
#         print(f"  IDs: {mismatched_ids}")


#%%
from tools.plotting import wavelength_to_rgb
import seaborn as sns

SR_err_df   = (SR_fit_df - SR_data_df).abs() / SR_data_df * 100
FWHM_err_df = (FWHM_fit_df - FWHM_data_df).abs() / FWHM_data_df * 100

wvl_normed = np.linspace(381, 750, len(wavelength)) # Approximate visible spectrum range to normalize to for better plotting
bins = np.linspace(0, 40, 21) # Define constant bins

# Plot as overlapping histograms iterating by wavelength
plt.figure(figsize=(10, 6))
for i, wvl in enumerate(wavelength[::2]):  # Skip every second wavelength
    data = SR_err_df[wvl].dropna().clip(upper=40)
    sns.histplot(data, bins=bins+i*0.05, alpha=0.1, label=f'{wvl} nm', color=wavelength_to_rgb(wvl_normed[i*2]),  element="step", fill=True, linewidth=2)

plt.xlim(0, 40)
plt.ylim(0, 250)
plt.xlabel('SR relative error, [%]')
plt.ylabel('Number of samples')
plt.title(f'Strehl Ratio Relative Error Distribution ({experiment})')
plt.legend(title='Wavelength, [nm]')
plt.xticks(np.arange(0, 45, 5), [str(x) if x < 40 else '>40' for x in np.arange(0, 45, 5)])
plt.show()

#%%
# Plot spectrally averaged SR error distribution
plt.figure(figsize=(10, 6))
data = SR_err_df.mean(axis=1).dropna().clip(upper=40)
sns.histplot(data, bins=bins, alpha=0.5, element="step", fill=True, linewidth=2)
plt.xlabel('Mean SR relative error, [%]')
plt.ylabel('Number of samples')
plt.title(f'Mean Strehl Ratio Relative Error Distribution ({experiment})')
plt.xticks(np.arange(0, 45, 5), [str(x) if x < 40 else '>40' for x in np.arange(0, 45, 5)])
plt.show()

# ID of a max SR error sample
# id_max_SR_err = SR_err_df.mean(axis=1).idxmax()
# print(f"ID of max SR error sample: {id_max_SR_err}")

# images_dict['ID'].index(id_max_SR_err)

#%%
from tools.plotting import plot_radial_PSF_profiles

data_cube   = np.stack(images_data, axis=0).squeeze()
fitted_cube = np.stack(images_fitted, axis=0).squeeze()

N_wvl = len(wavelength)

PSF_disp = lambda x, w: (x[:,w,...])
wvl_select = np.s_[0, N_wvl//2, -1]

# Polychromatic median profiles
fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSF_disp(data_cube, lmbd),
        PSF_disp(fitted_cube, lmbd),
        'Data',
        'TipTorch',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        return_profiles=True,
        ax=ax[i]
    )
plt.show()

#%%
# Spectrally averaged profile
fig = plt.figure(figsize=(10, 6))
plt.title('Polychromatic PSFs profile')
PSF_avg = lambda x: np.mean(x, axis=1)
plot_radial_PSF_profiles( PSF_avg(data_cube), PSF_avg(fitted_cube), 'Data', 'TipTorch', cutoff=40, ax=fig.add_subplot(111) )
plt.show()
# %%
