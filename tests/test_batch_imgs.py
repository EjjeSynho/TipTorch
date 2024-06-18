#%%
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky
from tools.utils import cropper
from project_globals import SPHERE_DATA_FOLDER
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
with open(SPHERE_DATA_FOLDER + 'images_df.pickle', 'rb') as handle:
    images = pickle.load(handle)


#%%
def batch_config_and_images(ids):
    data_dict = {}

    PSF_data, _, merged_config = SPHERE_preprocess(
        sample_ids    = ids,
        norm_regime   = 'sum',
        split_cube    = False,
        PSF_loader    = lambda x: SamplesByIds(x, synth=False),
        config_loader = GetSPHEREonsky,
        framework     = 'pytorch',
        device        = 'cpu')

    pack_data = lambda entry: torch.stack( [PSF_data[i][entry] for i in range(len(PSF_data))], axis=0 )
        
    data_dict['PSF_0']   = pack_data('PSF (mean)')
    data_dict['PSF_var'] = pack_data('PSF (var)')
    data_dict['norms']   = pack_data('norm (mean)')
    data_dict['masks']   = pack_data('mask (mean)')
    data_dict['configs'] = merged_config
    data_dict['IDs']     = ids
    
    return data_dict

ids_ = [1115, 2818, 637, 869, 1370, 159, 2719, 1588]

sample_id = 2818

index_ = ids_.index(sample_id)

data_dict = batch_config_and_images(ids_)

croppa = cropper(data_dict['PSF_0'][index_,...], 80)

PSF_data, _, merged_config = SPHERE_preprocess(
    sample_ids    = [sample_id],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = 'cpu')

PSF_0    = PSF_data[0]['PSF (mean)']
PSF_var  = PSF_data[0]['PSF (var)']
PSF_mask = PSF_data[0]['mask (mean)']
norms    = PSF_data[0]['norm (mean)']
del PSF_data

PSF_0_stack = data_dict['PSF_0'][index_,...]

dPSF = PSF_0 - PSF_0_stack

plt.imshow(np.log(PSF_0[0,...]))
plt.show()
plt.imshow(dPSF[0,...])
plt.colorbar()
plt.show()

# %%
with open(SPHERE_DATA_FOLDER + 'IRDIS_fitted/2818.pickle', 'rb') as handle:
    data = pickle.load(handle)

PSF_0_fitto = torch.tensor(data['Img. data'].squeeze())

image = images[sample_id][0].squeeze() / norms.unsqueeze(-1).unsqueeze(-1)

plt.imshow((PSF_0_fitto-PSF_0)[1,...])
plt.colorbar()
plt.show()

# %%
