#%%
import sys
sys.path.append('..')

import os
import pickle
from project_settings import MUSE_DATA_FOLDER
import pandas as pd
import seaborn as sns
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
# fitted_samples_folder = MUSE_DATA_FOLDER + 'MUSE_fitted_derot/'
# fitted_samples_folder = MUSE_DATA_FOLDER + 'MUSE_fitted_new_norm/'
fitted_samples_folder = MUSE_DATA_FOLDER + 'MUSE_fitted_neg_b/'

files = os.listdir(fitted_samples_folder)

with open(fitted_samples_folder + files[0], 'rb') as handle:
    data = pickle.load(handle)

#%%
for x in data.keys():
    print(x, end=', ')

df_relevant_entries = [
    'bg', 'F', 'dx', 'dy', 'r0', 'dn', 'Jx', 'Jy', 'Jxy', 'sausage_pow',
    'amp', 'b', 'alpha', 'beta', 'ratio', 'theta',
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
for key in ['r0', 'dn', 'Jxy', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta', 'sausage_pow']:
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

FWHM_fit_df  = FWHM_fit_x.apply(lambda x: x**2)  + FWHM_fit_y.apply(lambda x: x**2)
FWHM_data_df = FWHM_data_x.apply(lambda x: x**2) + FWHM_data_y.apply(lambda x: x**2)

FWHM_fit_df  = FWHM_fit_df.apply(np.sqrt)
FWHM_data_df = FWHM_data_df.apply(np.sqrt)

#%%
# Postprocess dataset
Jx_df = Jx_df.abs()
Jy_df = Jy_df.abs()
singular_df['alpha'] = singular_df['alpha'].apply(lambda x: np.abs(x))
singular_df['r0']    = singular_df['r0'].apply(lambda x: np.abs(x))

singular_df[singular_df['r0'] > 1] = np.nan

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
            'F_df':  F_df,
            'FWHM_fit_df':  FWHM_fit_df,
            'FWHM_data_df': FWHM_data_df,
        },
        handle
    )

#%%
from tools.plotting import plot_radial_profiles

# rand_array = np.random.randint(0, len(ids), 100)

PSF_0 = np.squeeze(np.stack(images_data,   axis=0))
PSF_1 = np.squeeze(np.stack(images_fitted, axis=0))

wvl_select = np.s_[0, 6, 12]
PSF_disp = lambda x, w: (x[:,w,...])

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=40,  ax=ax[i] )
plt.show()

# %%
fig = plt.figure(figsize=(10, 6))
plt.title('Polychromatic PSF')
PSF_avg = lambda x: np.mean(x, axis=1)
plot_radial_profiles( PSF_avg(PSF_0), PSF_avg(PSF_1), 'Data', 'TipTorch', cutoff=40, ax=fig.add_subplot(111) )
plt.show()


# %% -------------------- Cross check with TipTorch --------------------
from tools.utils import SausageFeature
from PSF_models.TipToy_MUSE_multisrc import TipTorch
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF
from project_settings import device
import torch

id = 396

PSF_1_fitted = torch.tensor(images_fitted[ids.index(id)], device=device).float()

sample = LoadMUSEsampleByID(id)
PSF_0, _, norms, bgs = LoadImages(sample, device)
config_file, PSF_0 = GetConfig(sample, PSF_0, None, device)
N_wvl = PSF_0.shape[1]

PSF_0 = rotate_PSF(PSF_0, -sample['All data']['Pupil angle'].item()) 
config_file['telescope']['PupilAngle'] = 0

#%%
#% Initialize the model
model = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)
sausage_absorber = SausageFeature(model)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

model.PSD_include['fitting'] = True
model.PSD_include['WFS noise'] = True
model.PSD_include['spatio-temporal'] = True
model.PSD_include['aliasing'] = False
model.PSD_include['chromatism'] = True
model.PSD_include['Moffat'] = True

model.to_float()

inputs_tiptorch = {
    'F':     torch.tensor([ F_df.loc[id].to_numpy()], device=model.device),
    'dx':    torch.tensor([dx_df.loc[id].to_numpy()], device=model.device),
    'dy':    torch.tensor([dy_df.loc[id].to_numpy()], device=model.device),
    'bg':    torch.tensor([bg_df.loc[id].to_numpy()], device=model.device),
    'Jx':    torch.tensor([Jx_df.loc[id].to_numpy()], device=model.device),
    'Jy':    torch.tensor([Jy_df.loc[id].to_numpy()], device=model.device),
    'r0':    torch.tensor([singular_df.loc[id, 'r0'   ]], device=model.device),
    'dn':    torch.tensor([singular_df.loc[id, 'dn'   ]], device=model.device),
    'Jxy':   torch.tensor([singular_df.loc[id, 'Jxy'  ]], device=model.device),
    'amp':   torch.tensor([singular_df.loc[id, 'amp'  ]], device=model.device),
    'b':     torch.tensor([singular_df.loc[id, 'b'    ]], device=model.device),
    'alpha': torch.tensor([singular_df.loc[id, 'alpha']], device=model.device),
    'beta':  torch.tensor([singular_df.loc[id, 'beta' ]], device=model.device),
    'ratio': torch.tensor([singular_df.loc[id, 'ratio']], device=model.device),
    'theta': torch.tensor([singular_df.loc[id, 'theta']], device=model.device)
}
setattr(model, 's_pow', torch.tensor([singular_df.loc[id,'sausage_pow']], device=model.device))

PSF_1 = model(inputs_tiptorch, None, lambda: sausage_absorber(model.s_pow.flatten()))

#%%
from tools.plotting import plot_radial_profiles, draw_PSF_stack

center = np.array([PSF_0.shape[-2]//2, PSF_0.shape[-1]//2])

wvl_select = np.s_[0, 6, 12]

draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )
draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1_fitted.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )

PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
plt.show()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1_fitted, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
plt.show()


# %%
