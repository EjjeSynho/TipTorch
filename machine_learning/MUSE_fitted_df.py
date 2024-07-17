#%%
import sys
sys.path.append('..')

import os
import pickle
from project_globals import MUSE_DATA_FOLDER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.utils import cropper


reduced_samples_folder = MUSE_DATA_FOLDER + 'DATA_reduced/'
#%%
file = os.listdir(reduced_samples_folder)[200]

with open(reduced_samples_folder + file, 'rb') as handle:
    data = pickle.load(handle)

wavelength = data['spectral data']['wvls binned'].round().astype(int)


IRLOS_phases_dict = {}

for file in tqdm(os.listdir(reduced_samples_folder)):
    with open(reduced_samples_folder + file, 'rb') as handle:
        data = pickle.load(handle)

    if 'IRLOS phases' in data.keys():
        if data['IRLOS phases']['OPD aperture'] is not None:
            IRLOS_phases_dict[int(file.split('_')[0])] = data['IRLOS phases']['OPD aperture'].mean(axis=0)


# Save phases to disk
with open(MUSE_DATA_FOLDER + 'IRLOS_phases_dict.pkl', 'wb') as handle:
    pickle.dump(IRLOS_phases_dict, handle)


#%%
# fitted_samples_folder = MUSE_DATA_FOLDER + 'MUSE_fitted_no_S_no_M/'
fitted_samples_folder = MUSE_DATA_FOLDER + 'MUSE_fitted_derot/'

files = os.listdir(fitted_samples_folder)

with open(fitted_samples_folder + files[0], 'rb') as handle:
    data = pickle.load(handle)

#%%
for x in data.keys():
    print(x, end=', ')

df_relevant_entries = [
    'bg', 'F', 'dx', 'dy', 'r0', 'dn', 'Jx', 'Jy', 'Jxy', 'amp', 'b', 'alpha',
    'SR data', 'SR fit', 'FWHM fit', 'FWHM data',
]

#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in df_relevant_entries}
ids = []

images_data, images_fitted = [], []

for file in tqdm(files):
    id = int(file.split('.')[0])

    with open(fitted_samples_folder + file, 'rb') as handle:
        data = pickle.load(handle)
    
    images_data.append( data['Img. data'] )
    images_fitted.append( data['Img. fit'] )
    
    for key in fitted_dict_raw.keys():
        fitted_dict_raw[key].append(data[key])
    ids.append(id)

#%%
# Store images
with open(MUSE_DATA_FOLDER + 'MUSE_images_data.pkl', 'wb') as handle:
    images_dict = {
        'data': images_data,
        'fitted': images_fitted,
        'ID': ids
    }
    pickle.dump(images_dict, handle)
    
#%%
singular_dict = {}
for key in ['r0', 'dn', 'Jxy', 'amp', 'b', 'alpha']:
    singular_dict[key] = np.squeeze(np.array(fitted_dict_raw[key])).tolist()

Jx_dict = np.squeeze(np.array(fitted_dict_raw['Jx']))
Jy_dict = np.squeeze(np.array(fitted_dict_raw['Jy']))
dx_dict = np.squeeze(np.array(fitted_dict_raw['dx']))
dy_dict = np.squeeze(np.array(fitted_dict_raw['dy']))
bg_dict = np.squeeze(np.array(fitted_dict_raw['bg']))
F_dict  = np.squeeze(np.array(fitted_dict_raw['F']))

def make_df_from_dict(dict_):
    dict_['ID'] = np.array(ids)
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)
    return df

def make_polychrome_dataset(data):
    dict_ = {}
    for i,wvl in enumerate(wavelength):
        dict_[wvl] = data[:,i].tolist()

    dict_['ID'] = np.array(ids).tolist()
    df = pd.DataFrame(dict_)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)
    return df

singular_df = make_df_from_dict(singular_dict)
singular_df['r0'] = singular_df['r0'].apply(lambda x: np.abs(x))

Jx_df = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['Jx'])))
Jy_df = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['Jy'])))
dx_df = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['dx'])))
dy_df = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['dy'])))
bg_df = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['bg'])))
F_df  = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['F'])))

FWHM_fit_x  = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['FWHM fit'])[...,0]))
FWHM_fit_y  = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['FWHM fit'])[...,1]))

FWHM_data_x = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['FWHM data'])[...,0]))
FWHM_data_y = make_polychrome_dataset(np.squeeze(np.stack(fitted_dict_raw['FWHM data'])[...,1]))

FWHM_fit_df  = FWHM_fit_x.apply(lambda x: x**2) + FWHM_fit_y.apply(lambda x: x**2)
FWHM_data_df = FWHM_data_x.apply(lambda x: x**2) + FWHM_data_y.apply(lambda x: x**2)

FWHM_fit_df  = FWHM_fit_df.apply(np.sqrt)
FWHM_data_df = FWHM_data_df.apply(np.sqrt)

#%%
# Store dataframes
with open(MUSE_DATA_FOLDER + 'MUSE_fitted_df.pkl', 'wb') as handle:
    pickle.dump(
        {
            'singular_vals_df': singular_df,
            'Jx_df': Jx_df,
            'Jy_df': Jy_df,
            'dx_df': dx_df,
            'dy_df': dy_df,
            'bg_df': bg_df,
            'F_df': F_df,
            'FWHM_fit_df': FWHM_fit_df,
            'FWHM_data_df': FWHM_data_df,
        },
        handle
    )

#%%
from tools.utils import plot_radial_profiles_new

# rand_array = np.random.randint(0, len(ids), 100)

PSF_0 = np.squeeze(np.stack(images_data,   axis=0))
PSF_1 = np.squeeze(np.stack(images_fitted, axis=0))

wvl_select = np.s_[0, 6, 12]
PSF_disp = lambda x, w: (x[:,w,...])

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=40,  ax=ax[i] )
plt.show()

# %%
fig = plt.figure(figsize=(10, 6))
plt.title()
PSF_avg = lambda x: np.mean(x, axis=1)
plot_radial_profiles_new( PSF_avg(PSF_0), PSF_avg(PSF_1), 'Data', 'TipTorch', cutoff=40, ax=fig.add_subplot(111) )
plt.show()
