#%%
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.plotting import plot_radial_PSF_profiles, LWE_basis, cropper, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from managers.config_manager import GetSPHEREonsky, ConfigManager
from project_settings import SPHERE_DATASET_FOLDER, SPHERE_DATA_FOLDER, device
from data_processing.normalizers import CreateTransformSequenceFromFile
from managers.input_manager import InputsTransformer
from copy import copy, deepcopy

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/psf_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

config_manager = ConfigManager(GetSPHEREonsky())

transformer = InputsTransformer({
    'F':   df_transforms_fitted['F L'],
    'bg':  df_transforms_fitted['bg L'], # bg R and bg L, as well as dx,dy, and F and are the same
    'r0':  df_transforms_fitted['r0'],
    'dx':  df_transforms_fitted['dx L'],
    'dy':  df_transforms_fitted['dy L'], 
    'Jx':  df_transforms_fitted['Jx'],
    'Jy':  df_transforms_fitted['Jy'],
    'Jxy': df_transforms_fitted['Jxy'],
    'dn':  df_transforms_fitted['dn'],
    'basis_coefs': df_transforms_fitted['LWE coefs'],
    'wind_dir':    df_transforms_fitted['Wind dir'],
    'wind_speed':  df_transforms_fitted['Wind speed']
})

fitted_entres = ['F L', 'F R', 'bg L', 'bg R', 'dx L', 'dx R', 'dy L', 'dy R', 'Wind dir', 'r0', 'Jx', 'Jy', 'Jxy', 'dn', 'LWE coefs']

#%%
def get_fixed_inputs(batch, entries=[]):
    if len(entries) == 0:
        return {}
    
    fixed_inputs = { entry: batch['fitted data'][entry] for entry in entries }

    entries_to_remove = []
    dict_to_add = {}

    for entry, value in fixed_inputs.items():
        if entry.endswith(' L'):
            entry_ = entry.replace(' L', '')
            
            dict_to_add[entry_] = torch.stack([
                torch.tensor(np.array(fixed_inputs[entry_+' L'])).float().to(device),
                torch.tensor(np.array(fixed_inputs[entry_+' R'])).float().to(device)
            ], axis=1)
            
            entries_to_remove.append(entry_+' L')
            entries_to_remove.append(entry_+' R')

        elif entry == 'LWE coefs':
            dict_to_add['basis_coefs'] = torch.tensor(np.array(fixed_inputs[entry])).float().to(device)
            entries_to_remove.append('LWE coefs')
            
        elif entry == 'Wind dir':
            dict_to_add['wind_dir'] = torch.tensor(np.array(fixed_inputs[entry])).float().to(device)
            entries_to_remove.append('Wind dir')   
                    
        else:
            fixed_inputs[entry] = torch.tensor(np.array(value)).float().to(device)
            
    for entry in entries_to_remove: _ = fixed_inputs.pop(entry)
    
    return fixed_inputs | dict_to_add


def run_model(model, batch, predicted_inputs, fixed_inputs={}):
    current_configs = batch['configs']
    config_manager.Convert(current_configs, framework='pytorch', device=device)

    model.config = current_configs
    model.Update(reinit_grids=False, reinit_pupils=False)

    x_unpacked = predicted_inputs | fixed_inputs # It overides the values from the destacked dictionary
    if 'basis_coefs' in x_unpacked:
        return model(x_unpacked, None, lambda: basis(x_unpacked['basis_coefs'].float()))
    else:
        return model(x_unpacked)


def get_data(batch, fixed_entries):
    x_0    = batch['NN input'].float().to(device)
    PSF_0  = batch['PSF_0'].float().to(device)
    config = batch['configs']
    fixed_inputs = get_fixed_inputs(batch, fixed_entries)
    config_manager.Convert(config, framework='pytorch', device=device)
    return x_0, fixed_inputs, PSF_0, config


#%%
with open(SPHERE_DATASET_FOLDER + 'batch_test.pkl', 'rb') as handle:
    batch_data = pickle.load(handle)

x0, fixed_inputs, PSF_0, init_config = get_data(batch_data, fitted_entres)

toy = TipTorch(init_config, None, device)
_ = toy()

basis = LWE_basis(toy)


#%%
PSFs_data, PSFs_fitted = [], []

with torch.no_grad():
    x0, fixed_inputs, PSF_0, init_config = get_data(batch_data, fitted_entres)
    toy.config = init_config
    toy.Update(reinit_grids=True, reinit_pupils=True)
    
    PSFs_data.append(PSF_0.cpu())
    fitted_dict = get_fixed_inputs(batch_data, fitted_entres)
    PSFs_fitted.append(run_model(toy, batch_data, {}, fixed_inputs=fitted_dict).cpu())


PSFs_data   = torch.cat(PSFs_data,   dim=0)[:,0,...].numpy()
PSFs_fitted = torch.cat(PSFs_fitted, dim=0)[:,0,...].numpy()

index_ = batch_data['IDs'].index(2818)

# plot_radial_PSF_profiles(PSFs_data[index_,...], PSFs_fitted[index_,...], 'Data', 'TipTorch', title='Fitted')
# plt.show()

print(batch_data['IDs'][2])
plot_radial_PSF_profiles(PSFs_data[2,...], PSFs_fitted[2,...], 'Data', 'TipTorch', title='Fitted')
plt.show()

# plot_radial_PSF_profiles(PSFs_data, PSFs_fitted, 'Data', 'TipTorch', title='Fitted')
# plt.show()

#%%
new_dict = {}

for entry in fitted_dict:
    new_dict[entry] = fitted_dict[entry][index_].unsqueeze(0)

_, _, merged_config = SPHERE_preprocess(
    sample_ids    = [2818],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

toy_one = TipTorch(merged_config, None, device)
_ = toy_one()

PSF_test = toy_one(new_dict, None, lambda: basis(new_dict['basis_coefs'].float())).cpu().numpy()[0,0,...]#[index_,0,...]

#%
plot_radial_PSF_profiles(PSFs_data[index_,...], PSF_test, 'Data', 'TipTorch', title='Fitted')
plt.show()


#%%
# A = (PSF_test - PSF_test.min()) / PSF_test.max()
# B = (PSFs_fitted[index_,...] - PSFs_fitted[index_,...].min()) / PSFs_fitted[index_,...].max()
A = PSF_test
B = PSFs_fitted[index_,...]

# draw_PSF_stack(PSF_test, PSFs_fitted[index_,...], average=True, crop=40, min_val=1e-6, max_val=1e-1)
# plt.show()

# draw_PSF_stack(A, B, average=True, crop=40, min_val=1e-6, max_val=1e-1)
plot_radial_PSF_profiles(A, B, 'Data', 'TipTorch', title='Fitted')
plt.show()
