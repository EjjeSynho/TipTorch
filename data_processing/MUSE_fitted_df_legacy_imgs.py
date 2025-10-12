#%%
import sys
sys.path.append('..')

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_processing.MUSE_STD_dataset_utils import STD_FOLDER, CUBES_CACHE
from data_processing.MUSE_data_utils import wvl_bins


wavelength = np.linspace(475, 935, 13)  # nm

#%%
IRLOS_phases_dict = {}

for file in tqdm(os.listdir(CUBES_CACHE)):
    with open(CUBES_CACHE / file, 'rb') as handle:
        data,_ = pickle.load(handle)

        # wavelength = data['wavelength']

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
# fitted_samples_folder = STD_FOLDER / 'fitted_no_Moffat/'
fitted_samples_folder = STD_FOLDER / 'old/MUSE_fitted_neg_b/'

files = os.listdir(fitted_samples_folder)

with open(fitted_samples_folder / files[0], 'rb') as handle:
    data = pickle.load(handle)
    for x in data.keys():
        print(x, end=', ')

df_relevant_entries = [
    'bg', 'F', 'dx', 'dy', 'r0', 'dn', 'Jx', 'Jy', 'sausage_pow', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta', 'wind_speed',
    'SR data', 'SR fit', 'FWHM fit', 'FWHM data',
]

df_relevant_entries = list(set(df_relevant_entries) & set(data.keys()))
df_relevant_entries.sort()

#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in df_relevant_entries}
ids = []

images_data, images_fitted = [], []

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
# Store images
images_dict = {
    'data': images_data,
    'fitted': images_fitted,
    'ID': ids
}


#%%
from tools.plotting import plot_radial_PSF_profiles

data_cube   = np.stack(images_data, axis=0).squeeze()
fitted_cube = np.stack(images_fitted, axis=0).squeeze()

N_wvl = 30
PSF_disp = lambda x, w: (x[:,w,...])
wvl_select = np.s_[0, N_wvl//2, -1]

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
singular_dict = {}
singular_entries = list( set(['r0', 'dn', 'Jxy', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta', 'sausage_pow']).intersection( df_relevant_entries) )

for key in singular_entries:
    singular_dict[key] = np.squeeze(np.array(fitted_dict_raw[key])).tolist()

# Jx_dict = np.squeeze(np.array(fitted_dict_raw['Jx']))
# Jy_dict = np.squeeze(np.array(fitted_dict_raw['Jy']))
# dx_dict = np.squeeze(np.array(fitted_dict_raw['dx']))
# dy_dict = np.squeeze(np.array(fitted_dict_raw['dy']))
# bg_dict = np.squeeze(np.array(fitted_dict_raw['bg']))
# F_dict  = np.squeeze(np.array(fitted_dict_raw['F']))

def make_df_from_dict(dict_):
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


# def make_LO_dataset(data):
#     data_ = np.squeeze(np.stack(data))
#     dict_ = {}
#     for i in range(data_.shape[1]):
#         dict_[f'Z{i+1}'] = data_[:,i].tolist()

#     dict_['ID'] = np.array(ids).tolist()
#     df = pd.DataFrame(dict_)
#     df.set_index('ID', inplace=True)
#     df.sort_index(inplace=True)

#     df.rename(columns={'Z1': 'Phase bump'}, inplace=True)

#     return df


#%%
singular_df = make_df_from_dict(singular_dict)

Jx_df  = make_polychrome_dataset(fitted_dict_raw['Jx'])
Jy_df  = make_polychrome_dataset(fitted_dict_raw['Jy'])
dx_df = make_polychrome_dataset(fitted_dict_raw['dx'])
dy_df = make_polychrome_dataset(fitted_dict_raw['dy'])
bg_df = make_polychrome_dataset(fitted_dict_raw['bg'])
F_df  = make_polychrome_dataset(fitted_dict_raw['F'])

# LO_df = make_LO_dataset(fitted_dict_raw['LO_coefs'])

FWHM_fit_data  = make_polychrome_dataset(fitted_dict_raw['FWHM fit'])
FWHM_data_data = make_polychrome_dataset(fitted_dict_raw['FWHM data'])

SR_fit_df  = make_polychrome_dataset(fitted_dict_raw['SR fit'])
SR_data_df = make_polychrome_dataset(fitted_dict_raw['SR data'])

FWHM_fit_x  = FWHM_fit_data.applymap(lambda x: x[0])
FWHM_fit_y  = FWHM_fit_data.applymap(lambda x: x[1])
FWHM_data_x = FWHM_data_data.applymap(lambda x: x[0])
FWHM_data_y = FWHM_data_data.applymap(lambda x: x[1])

FWHM_fit_df  = FWHM_fit_x.apply(lambda x: x**2)  + FWHM_fit_y.apply(lambda x: x**2)
FWHM_data_df = FWHM_data_x.apply(lambda x: x**2) + FWHM_data_y.apply(lambda x: x**2)

FWHM_fit_df  = FWHM_fit_df.apply(np.sqrt)
FWHM_data_df = FWHM_data_df.apply(np.sqrt)

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
# Postprocess dataset
Jx_df = Jx_df.abs()
Jy_df = Jy_df.abs()
if 'alpha' in singular_df.columns:
    singular_df['alpha'] = singular_df['alpha'].apply(lambda x: np.abs(x))
singular_df['r0'] = singular_df['r0'].apply(lambda x: np.abs(x))
singular_df[singular_df['r0'] > 1] = np.nan


#%%
from tools.plotting import wavelength_to_rgb
import seaborn as sns

SR_err_df   = (SR_fit_df - SR_data_df).abs() / SR_data_df * 100
FWHM_err_df = (FWHM_fit_df - FWHM_data_df).abs() / FWHM_data_df * 100

wvl_normed = np.linspace(380, 750, len(wavelength))

# print(SR_err_df.describe().T)

# Plot as overlapping histograms iterating by wavelength
plt.figure(figsize=(10, 6))
for i, wvl in enumerate(wavelength[::2]):  # Skip every second wavelength
    data = SR_err_df[wvl].dropna()
    sns.histplot(data, bins=30, alpha=0.2, label=f'{wvl} nm',  color=wavelength_to_rgb(wvl_normed[i*2]),  element="step", fill=True, linewidth=2)
    
plt.xlabel('SR relative error (%)')
plt.ylabel('Number of samples')
plt.title('Strehl Ratio Relative Error Distribution')
plt.legend()
plt.show()


#%%
from tools.plotting import plot_radial_PSF_profiles

N_wvl = len(wavelength)

PSF_disp = lambda x, w: (x[:,w,...])
wvl_select = np.s_[0, N_wvl//2, -1]

PSF_0 = images_dict['data'][images_dict['ID'].index(344)]
PSF_1 = images_dict['fitted'][images_dict['ID'].index(344)]

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSF_disp(PSF_0, lmbd),
        PSF_disp(PSF_1, lmbd),
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
from tools.plotting import plot_radial_PSF_profiles

# rand_array = np.random.randint(0, len(ids), 100)

PSF_0 = np.squeeze(np.stack(images_data,   axis=0))
PSF_1 = np.squeeze(np.stack(images_fitted, axis=0))

wvl_select = np.s_[0, 6, 12]
PSF_disp = lambda x, w: (x[:,w,...])

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=40,  ax=ax[i] )
plt.show()

# %%
fig = plt.figure(figsize=(10, 6))
plt.title('Polychromatic PSF')
PSF_avg = lambda x: np.mean(x, axis=1)
plot_radial_PSF_profiles( PSF_avg(PSF_0), PSF_avg(PSF_1), 'Data', 'TipTorch', cutoff=40, ax=fig.add_subplot(111) )
plt.show()


