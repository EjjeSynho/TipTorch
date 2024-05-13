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
from project_globals import SPHERE_DATA_FOLDER
import pandas as pd
from tqdm import tqdm

#%%
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

with open(SPHERE_DATA_FOLDER+'fitted_df.pickle', 'rb') as handle:
    fitted_df = pickle.load(handle)

with open('../data/temp/fitted_df_norm.pickle', 'rb') as handle:
    fitted_df_norm = pickle.load(handle)

with open('../data/temp/psf_df_norm.pickle', 'rb') as handle:
    psf_df_norm = pickle.load(handle)

get_ids = lambda df: set( df.index.values.tolist() )

ids_0 = get_ids(fitted_df)
ids_1 = get_ids(psf_df)
ids_2 = get_ids(fitted_df_norm)
ids_3 = get_ids(psf_df_norm)
ids   = list( ids_0.intersection(ids_1).intersection(ids_2).intersection(ids_3) )

print('Final number of good samples:', len(ids))

psf_df = psf_df.loc[ids]
fitted_df = fitted_df.loc[ids]
psf_df_norm = psf_df_norm.loc[ids]
fitted_df_norm = fitted_df_norm.loc[ids]

#%% Group by wavelengths
print('Grouping data by wavelengths...')

psf_df['λ group'] = psf_df.groupby(['λ left (nm)', 'λ right (nm)']).ngroup()
unique_wvls = psf_df[['λ left (nm)', 'λ right (nm)', 'λ group']].drop_duplicates().set_index('λ group')
unique_wvls.sort_values(by='λ group', inplace=True)
group_counts = psf_df.groupby(['λ group']).size()
group_counts = pd.concat([group_counts, unique_wvls], axis=1)
group_counts.sort_values(by='λ group', inplace=True)
group_counts.columns = ['Count', 'λ left (nm)', 'λ right (nm)']
group_counts.sort_values(by='Count', ascending=False, inplace=True)
group_counts = group_counts[group_counts['Count'] > 10]

print(group_counts)

# Splitting into the groups by the wavelength
ids_wvl_split = {}
for wvl_group in group_counts.index.values:
    ids_wvl_split[wvl_group] = psf_df[psf_df['λ group'] == wvl_group].index.values
    
#%%
def generate_binary_sequence(size, fraction_of_ones):
    sequence = np.random.choice([0, 1], size=size-1, p=[1-fraction_of_ones, fraction_of_ones])
    sequence = np.append(sequence, 1) # Make sure that at least one will be in validation
    np.random.shuffle(sequence)
    return sequence.astype(bool)

def split_in_batches(ids, batch_size):
    n_batches = np.ceil(len(ids) / batch_size).astype(int)
    np.random.shuffle(ids)
    return np.array_split(ids, n_batches)

BATCH_SIZE = 64 #16

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
        # device        = device)

    pack_data = lambda entry: torch.stack( [PSF_data[i][entry] for i in range(len(PSF_data))], axis=0 )
        
    data_dict['PSF_0']   = pack_data('PSF (mean)')
    data_dict['PSF_var'] = pack_data('PSF (var)')
    data_dict['norms']   = pack_data('norm (mean)')
    data_dict['masks']   = pack_data('mask (mean)')
    data_dict['configs'] = merged_config
    data_dict['IDs']     = ids
    
    return data_dict


def batch_get_data_dicts(ids):
    data_dict = {
        'fitted data': fitted_df.loc[ids].to_dict(orient='list'),
        'onsky data':  psf_df.loc[ids].to_dict(orient='list')
    }
    return data_dict


selected_entries_input = [
    'Airmass',
    'r0 (SPARTA)',
    # 'Seeing (SPARTA)',
    # 'FWHM',
    'Wind direction (header)',
    # 'Wind direction (MASSDIMM)',
    'Wind speed (header)',
    'Wind speed (SPARTA)',
    # 'Wind speed (MASSDIMM)',
    'Tau0 (header)',
    'Tau0 (SPARTA)',
    # 'Jitter X', 'Jitter Y',
    'Focus',
    'ITTM pos (avg)',
    'ITTM pos (std)',
    'DM pos (avg)',
    'DM pos (std)',
    'Pressure',
    # 'Humidity',
    # 'Temperature',
    'Nph WFS',
    # 'Flux WFS',
    'Rate',
    # 'DIT Time',
    'Sequence time',
    'λ left (nm)',  'λ right (nm)',
    'Strehl'
    # 'Δλ left (nm)', 'Δλ right (nm)',
    # 'mag V', 'mag R', 'mag G', 'mag J', 'mag H', 'mag K'
]

def batch_get_inp_tensor(ids):
    return { 'NN input': torch.tensor(psf_df_norm[selected_entries_input].loc[ids].to_numpy()) }


def CreateDataset():
    for current_group in group_counts.index.values:
        wvl_L = group_counts.loc[current_group]['λ left (nm)']
        wvl_R = group_counts.loc[current_group]['λ right (nm)']
        print(f'>>>>>> Processing wavelength group {current_group}: {wvl_L}, {wvl_R} [nm]')
        ids_temp = ids_wvl_split[current_group]

        rand_selection = generate_binary_sequence(len(ids_temp), 0.8)
        ids_train = ids_temp[ rand_selection]
        ids_val   = ids_temp[~rand_selection]

        ids_train_batches = split_in_batches(ids_train, BATCH_SIZE)
        ids_val_batches   = split_in_batches(ids_val,   BATCH_SIZE)

        print('Processing training batches...')
        for batch_id in tqdm(range(len(ids_train_batches))):
            ids = ids_train_batches[batch_id]
            batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)
            
            filename_batch = SPHERE_DATA_FOLDER + f'SPHERE_dataset_{BATCH_SIZE}/train/' + f'batch_{batch_id}_train_grp_{current_group}.pkl'
            with open(filename_batch, 'wb') as handle:
                # print('Writing:', filename_batch)
                pickle.dump(batch_data, handle)

        print('Processing validation batches...')
        for batch_id in tqdm(range(len(ids_val_batches))):
            ids = ids_val_batches[batch_id]
            batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)

            filename_batch = SPHERE_DATA_FOLDER + f'SPHERE_dataset_{BATCH_SIZE}/validation/' + f'batch_{batch_id}_val_grp_{current_group}.pkl'
            with open(filename_batch, 'wb') as handle:
                # print('Writing:', filename_batch)
                pickle.dump(batch_data, handle)
        
        print('\n')


def WriteTestBatch():
    ids = [1115, 2818, 637, 869, 1370, 159, 2719, 1588]
    batch_data = batch_config_and_images(ids) | batch_get_data_dicts(ids) | batch_get_inp_tensor(ids)

    filename_batch = SPHERE_DATA_FOLDER + f'SPHERE_dataset_{BATCH_SIZE}/' + f'batch_test.pkl'
    with open(filename_batch, 'wb') as handle:
        print('Writing:', filename_batch)
        pickle.dump(batch_data, handle)
        
# %%
CreateDataset()
WriteTestBatch()

# %%
