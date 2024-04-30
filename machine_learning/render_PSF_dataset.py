
#%%
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.insert(0, '..')

import os
import pickle
import numpy as np
from project_globals import SPHERE_DATA_FOLDER
from tqdm import tqdm
from tools.utils import plot_radial_profiles_new
import matplotlib.pyplot as plt

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

#%%
profiles = {}
bad_ids  = []

for id in tqdm(ids):
    # if not f'{id}.png' in os.listdir(SPHERE_DATA_FOLDER+'fitted_imgs/'):
    file = f'{id}.pickle'
    with open(SPHERE_DATA_FOLDER+'IRDIS_fitted/'+file, 'rb') as handle:
        data = pickle.load(handle)

    try:
        norms = data['Data norms'][None,:,None,None]
        PSF_0 = data['Img. data'] / norms
        PSF_1 = data['Img. fit']  / norms

        p_0, p_1, p_err = plot_radial_profiles_new(PSF_0[:,0,...], PSF_1[:,0,...], 'Data','Fitted', title=str(id), dpi=200, return_profiles=True)
        plt.savefig(SPHERE_DATA_FOLDER+f'fitted_imgs/{id}.png')
        profiles[id] = (p_0, p_1, p_err)
        
    except:
        print(f'Error with {id}')
        bad_ids.append(id)

with open(SPHERE_DATA_FOLDER + 'fitted_and_data_PSF_profiles.pickle', 'wb') as handle:
    pickle.dump(profiles, handle)

#%%
central_deviations = {}

for id in profiles:
    profile = profiles[id][-1]
    profile_m_err = np.median(profile, axis=0)
    central_deviations[id] = profile_m_err[0]

#%%
ids_deviants = [id for id in central_deviations if central_deviations[id] > 30]

# Find the max values
max_val = 0
max_id  = None
for id in ids_deviants:
    if central_deviations[id] > max_val:
        max_val = central_deviations[id]
        max_id  = id
        
# Find the min values
min_val = 100
min_id  = None
for id in central_deviations:
    if central_deviations[id] < min_val:
        min_val = central_deviations[id]
        min_id  = id

# Find the median
dev_vals   = np.array( [central_deviations[id] for id in central_deviations] )
median_val = np.median(dev_vals)

#%%
dev_vals = np.array( [central_deviations[id] for id in central_deviations] )
# _ = plt.hist(dev_vals, bins=100)

print(f'Max. deviation from data profiles: {max_val:.2f}% for id: {max_id}')
print(f'Min. deviation from data profiles: {min_val:.2f}% for id: {min_id}')
print(f'Median deviation from data profiles: {median_val:.2f}%')

#%%
# A = PSF_0[:,0,...]
# B = PSF_1[:,0,...]
# max_A = A.max().item()
# max_B = B.max().item()
# print( f'{np.abs(max_B-max_A)/max_A * 100}:.2f %')
