#%%
%reload_ext autoreload
%autoreload 2

import os
import sys

sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
from data_processing.MUSE_data_utils import MUSE_DATA_FOLDER, wvl_bins
from data_processing.MUSE_STD_dataset_utils import LoadSTDStarData, STD_FOLDER
from tqdm import tqdm


#%%
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

with open(STD_FOLDER / 'muse_fitted_df.pkl', 'rb') as handle:
    fitted_df = pickle.load(handle)

with open(STD_FOLDER / 'muse_STD_stars_telemetry.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)['telemetry normalized imputed df']

muse_df = muse_df[muse_df['Corrupted']   == False]
muse_df = muse_df[muse_df['Bad quality'] == False]

muse_df_norm = muse_df_norm.loc[muse_df.index]

dx_df = fitted_df['dx_df']

valid_ids = (dx_df)[~dx_df.isna()].index.values.tolist()

muse_df_norm = muse_df_norm.loc[valid_ids]
muse_df = muse_df.loc[valid_ids]

#%%
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 2), a_min=0, a_max=N_wvl_max-1)
wvls = wvl_bins[wvl_ids]

F_fitted_df  = fitted_df['F_df'].loc[valid_ids]
dx_fitted_df = fitted_df['dx_df'].loc[valid_ids]
dy_fitted_df = fitted_df['dy_df'].loc[valid_ids]
bg_fitted_df = fitted_df['bg_df'].loc[valid_ids]
J_fitted_df  = fitted_df['J_df'].loc[valid_ids]

Jxy_fitted_df   = fitted_df['singular_vals_df']['Jxy'].loc[valid_ids]
r0_fitted_df    = fitted_df['singular_vals_df']['r0'].loc[valid_ids]
dn_fitted_df    = fitted_df['singular_vals_df']['dn'].loc[valid_ids]

amp_fitted_df   = fitted_df['singular_vals_df']['amp'].loc[valid_ids]
b_fitted_df     = fitted_df['singular_vals_df']['b'].loc[valid_ids]
alpha_fitted_df = fitted_df['singular_vals_df']['alpha'].loc[valid_ids]
beta_fitted_df  = fitted_df['singular_vals_df']['beta'].loc[valid_ids]
ratio_fitted_df = fitted_df['singular_vals_df']['ratio'].loc[valid_ids]
theta_fitted_df = fitted_df['singular_vals_df']['theta'].loc[valid_ids]

LO_fitted_df    = fitted_df['LO_df'].loc[valid_ids]

#%% 
BATCH_SIZE = 16
selected_entries_input = muse_df_norm.columns.values.tolist()

def batch_config_and_images(ids):
    data_dict = {}
 
    PSFs, configs, norms, bgs  = LoadSTDStarData(
        ids = ids,
        derotate_PSF = True,
        normalize = True,
        subtract_background = True,
        ensure_odd_pixels = True,
        device = torch.device('cpu')
    )

    data_dict['PSF_0']   = PSFs
    data_dict['norms']   = norms
    data_dict['bgs']     = bgs
    data_dict['IDs']     = ids
    data_dict['Wvls']    = wvls
    data_dict['configs'] = configs

    return data_dict


def batch_get_data_dicts(ids):
    return {
        'fitted data': {
            'r0':    torch.from_numpy(r0_fitted_df.loc[ids].to_numpy()).float(),
            'F':     torch.from_numpy(F_fitted_df.loc[ids].to_numpy()).float(),
            'dx':    torch.from_numpy(dx_fitted_df.loc[ids].to_numpy()).float(),
            'dy':    torch.from_numpy(dy_fitted_df.loc[ids].to_numpy()).float(), 
            'bg':    torch.from_numpy(bg_fitted_df.loc[ids].to_numpy()).float(),
            'dn':    torch.from_numpy(dn_fitted_df.loc[ids].to_numpy()).float(),
            'J':     torch.from_numpy(J_fitted_df.loc[ids].to_numpy()).float(),
            'Jxy':   torch.from_numpy(Jxy_fitted_df.loc[ids].to_numpy()).float(),
            'amp':   torch.from_numpy(amp_fitted_df.loc[ids].to_numpy()).float(),
            'b':     torch.from_numpy(b_fitted_df.loc[ids].to_numpy()).float(),
            'alpha': torch.from_numpy(alpha_fitted_df.loc[ids].to_numpy()).float(),
            'beta':  torch.from_numpy(beta_fitted_df.loc[ids].to_numpy()).float(),
            'ratio': torch.from_numpy(ratio_fitted_df.loc[ids].to_numpy()).float(),
            'theta': torch.from_numpy(theta_fitted_df.loc[ids].to_numpy()).float(),
            'LO':    torch.from_numpy(LO_fitted_df.loc[ids].to_numpy()).float()
        },
        'onsky data':  muse_df_norm.loc[ids].to_dict(orient='list')
    }


def batch_get_inp_tensor(ids):
    return { 'NN input': torch.tensor(muse_df_norm[selected_entries_input].loc[ids].to_numpy()) }

#%%
def CreateDataset(ids_train_batches=None, ids_val_batches=None):
    
    train_folder      = MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/train/'
    validation_folder = MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/validation/'

    # Make folders if they don't exist
    if not os.path.exists(MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/'):
        os.makedirs(MUSE_DATA_FOLDER + f'MUSE_dataset_{BATCH_SIZE}/')
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    if ids_train_batches is None or ids_val_batches is None:
        # Compute new random split
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
train_ids, val_ids = [], []

folder_train = MUSE_DATA_FOLDER + 'dataset_legacy/train/'
folder_val   = MUSE_DATA_FOLDER + 'dataset_legacy/validation/'

if not os.path.exists(folder_train):
    print('Copying IDs from the existent dataset...')
    print('Loading train batches...')
    for file in tqdm([ folder_train+file for file in os.listdir(folder_train) if '.pkl' in file ]):
        with open(file, 'rb') as handle:
            train_ids.append( pickle.load(handle)['IDs'].tolist() )
            
    print('Loading validation batches...')
    for file in tqdm([ folder_val+file   for file in os.listdir(folder_val)   if '.pkl' in file ]):
        with open(file, 'rb') as handle:
            val_ids.append( pickle.load(handle)['IDs'].tolist() )

else:
    print('Creating new dataset...')
    CreateDataset()
    WriteTestBatch()

# %%

