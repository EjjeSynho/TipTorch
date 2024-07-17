#%%
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
from project_globals import MUSE_DATA_FOLDER
import pandas as pd
from tqdm import tqdm
from data_processing.MUSE_preproc_utils import GetMUSEonsky


#%%
with open(MUSE_DATA_FOLDER+'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

# with open(MUSE_DATA_FOLDER+'fitted_df.pickle', 'rb') as handle:
#     fitted_df = pickle.load(handle)

# with open('../data/temp/fitted_df_norm.pickle', 'rb') as handle:
#     fitted_df_norm = pickle.load(handle)

with open(MUSE_DATA_FOLDER+'MUSE_fitted_df.pkl', 'rb') as handle:
    fitted_df = pickle.load(handle)

with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

muse_df = muse_df[muse_df['Corrupted']   == False]
muse_df = muse_df[muse_df['Bad quality'] == False]

muse_df_norm = muse_df_norm.loc[muse_df.index]

valid_ids = (test:=fitted_df['dx_df'].mean(axis=1))[~test.isna()].index

muse_df_norm = muse_df_norm.loc[valid_ids]
muse_df = muse_df.loc[valid_ids]

#%%
PSF_0, merged_config = GetMUSEonsky([100], derotate_PSF=True, device=torch.device('cpu'))
wvls = (merged_config['sources_science']['Wavelength'].numpy() * 1e9).round(0).astype(int).flatten()

column_names = wvls.tolist()
index_names  = muse_df.index.values

# dx_fitted_df  = pd.DataFrame(np.zeros([len(muse_df.index), len(wvls)]), columns=column_names, index=index_names)
# dy_fitted_df  = pd.DataFrame(np.zeros([len(muse_df.index), len(wvls)]), columns=column_names, index=index_names)
# bg_fitted_df  = pd.DataFrame(np.zeros([len(muse_df.index), len(wvls)]), columns=column_names, index=index_names)
# Jxy_fitted_df = pd.DataFrame(np.zeros([len(muse_df.index), len(wvls)]), columns=column_names, index=index_names)

dx_fitted_df  = fitted_df['dx_df'].loc[valid_ids]
dy_fitted_df  = fitted_df['dy_df'].loc[valid_ids]
bg_fitted_df  = fitted_df['bg_df'].loc[valid_ids]
Jxy_fitted_df = fitted_df['singular_vals_df']['Jxy'].loc[valid_ids]


#%% 
BATCH_SIZE = 4
selected_entries_input = muse_df_norm.columns.values.tolist()

def batch_config_and_images(ids):
    data_dict = {}
    PSF_0, merged_config = GetMUSEonsky(ids, device=torch.device('cpu'))
        
    data_dict['PSF_0']   = PSF_0
    data_dict['configs'] = merged_config
    data_dict['IDs']     = ids
    data_dict['Wvls']    = (merged_config['sources_science']['Wavelength'].numpy() * 1e9).round(0).astype(int).flatten()
 
    return data_dict


def batch_get_data_dicts(ids):
    return { 
        'fitted data': {
            'Jxy': torch.from_numpy(dx_fitted_df.loc[ids].to_numpy()).float(),
            'dx':  torch.from_numpy(dx_fitted_df.loc[ids].to_numpy()).float(),
            'dy':  torch.from_numpy(dy_fitted_df.loc[ids].to_numpy()).float(), 
            'bg':  torch.from_numpy(bg_fitted_df.loc[ids].to_numpy()).float()   
        },
        'onsky data':  muse_df_norm.loc[ids].to_dict(orient='list')
    }


def batch_get_inp_tensor(ids):
    return { 'NN input': torch.tensor(muse_df_norm[selected_entries_input].loc[ids].to_numpy()) }

#%%
def CreateDataset():
    
    train_folder      = MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/train/'
    validation_folder = MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/validation/'

    # Make folders if they don't exist
    if not os.path.exists(MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/'):
        os.makedirs(MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/')
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    ids_all = muse_df_norm.index.values.astype(int)
    np.random.shuffle(ids_all)

    ids_split = np.array_split(ids_all, np.ceil(len(ids_all)/BATCH_SIZE))

    num_batches = len(ids_split)

    ids_train_batches = ids_split[:int(0.8*num_batches)]
    ids_val_batches   = ids_split[int(0.8*num_batches):]

    print('Processing training batches...')
    for batch_id in tqdm(range(len(ids_train_batches))):
        ids = ids_train_batches[batch_id]
        batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)
        
        filename_batch = train_folder + f'batch_{batch_id}_train.pkl'
        with open(filename_batch, 'wb') as handle:
            # print('Writing:', filename_batch)
            pickle.dump(batch_data, handle)

    print('Processing validation batches...')
    for batch_id in tqdm(range(len(ids_val_batches))):
        ids = ids_val_batches[batch_id]
        batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)

        filename_batch = validation_folder + f'batch_{batch_id}_val.pkl'
        with open(filename_batch, 'wb') as handle:
            # print('Writing:', filename_batch)
            pickle.dump(batch_data, handle)
    
    print('\n')


def WriteTestBatch():
    ids = [210, 327, 355, 311]
    batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)

    filename_batch = MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/' + f'batch_test.pkl'
    with open(filename_batch, 'wb') as handle:
        print('Writing:', filename_batch)
        pickle.dump(batch_data, handle)
        
# %%
CreateDataset()
WriteTestBatch()

# %%
