#%%
import sys
sys.path.append('..')

import os
import pickle
from project_globals import SPHERE_DATA_FOLDER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.utils import cropper

#%%
fitted_samples_folder = SPHERE_DATA_FOLDER + 'IRDIS_fitted/'

files = os.listdir(fitted_samples_folder)

with open(fitted_samples_folder + files[0], 'rb') as handle:
    data = pickle.load(handle)

for x in data.keys():
    print(x)

df_relevant_entries = [
    'bg', 'F', 'dx', 'dy', 'r0', 'n', 'dn', 'Jx', 'Jy', 'Jxy', 'Nph WFS (new)',
    'SR data', 'SR fit', 'FWHM fit', 'FWHM data', 'LWE coefs'
]

#%% Create fitted parameters dataset
fitted_dict_raw = {key: [] for key in df_relevant_entries}
ids = []

images_data, images_fitted = [], []
Hessians, variances = [], []

for file in tqdm(files):
    id = int(file.split('.')[0])

    with open(fitted_samples_folder + file, 'rb') as handle:
        data = pickle.load(handle)
        
    images_data.append( data['Img. data'] )
    images_fitted.append( data['Img. fit'] )
    Hessians.append( data['Hessian'] )
    variances.append( data['Variances'] )
    
    for key in fitted_dict_raw.keys():
        fitted_dict_raw[key].append(data[key])
    ids.append(id)
    
#%%
fitted_dict = {}
fitted_dict['ID'] = np.array(ids)

for key in fitted_dict_raw.keys():
    fitted_dict[key] = np.squeeze(np.array(fitted_dict_raw[key]))

for key in fitted_dict.keys():
    fitted_dict[key] = fitted_dict[key].tolist()

fitted_df = pd.DataFrame(fitted_dict)
fitted_df.set_index('ID', inplace=True)
fitted_df.sort_index(inplace=True)

FWHM_norm = lambda x: [np.sqrt(x[0][0]**2 + x[0][1]**2), np.sqrt(x[1][0]**2 + x[1][1]**2)]

fitted_df['FWHM fit' ] = fitted_df['FWHM fit' ].apply(FWHM_norm)
fitted_df['FWHM data'] = fitted_df['FWHM data'].apply(FWHM_norm)

def separate_L_and_R(entry):
    fitted_df[entry + ' L'] = fitted_df[entry].apply(lambda x: x[0])
    fitted_df[entry + ' R'] = fitted_df[entry].apply(lambda x: x[1])
    fitted_df.pop(entry)
    
for entry in ['F', 'bg', 'SR data', 'SR fit', 'FWHM fit', 'FWHM data']:
    separate_L_and_R(entry)

fitted_df['LWE coefs'] = fitted_df['LWE coefs'].apply(lambda x: np.array(x))

#%%
fitted_df_filtered = fitted_df.dropna()
print('\n>>>> Samples remained:', len(fitted_df_filtered))

with open(SPHERE_DATA_FOLDER + 'fitted_df.pickle', 'wb') as handle:
    pickle.dump(fitted_df_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
from data_processing.normalizers import CreateTransformSequenceFromFile

df_transforms = CreateTransformSequenceFromFile('../data/temp/fitted_df_norm_transforms.pickle')

fitted_df_normalized = fitted_df_filtered.copy()

for entry in df_transforms:
    fitted_df_normalized[entry] = df_transforms[entry].forward(fitted_df_normalized[entry].values)


with open('../data/temp/fitted_df_norm.pickle', 'wb') as handle:
    pickle.dump(fitted_df_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
   
#%% Write images

crop_imgs = cropper(images_data[0], 80)

im_d_ = [img[crop_imgs] for img in images_data  ]
im_f_ = [img[crop_imgs] for img in images_fitted]

images_df = { id: (im_d_[i], im_f_[i]) for i, id in enumerate(ids) }

with open(SPHERE_DATA_FOLDER + 'images_df.pickle', 'wb') as handle:
    pickle.dump(images_df, handle, protocol=pickle.HIGHEST_PROTOCOL)       
    
 
#%% =============================================================================
names = ['Piston',]*4 + ['Tip',]*4 + ['Tilt',]*4

for i in range(0, 4):
    names[i]   += f' {i+1}'
    names[i+4] += f' {i+1}'
    names[i+8] += f' {i+1}'
names.pop(0)

names = ['r0', 'F L', 'F R', 'dx', 'dy', 'bg L', 'bg R', 'dn', 'Jx', 'Jy', 'Jxy'] + names
transform_entries = ['r0', 'F L', 'F R', 'dx', 'dy', 'bg L', 'bg R', 'dn', 'Jx', 'Jy', 'Jxy'] + ['LWE coefs',]*11

#%%
variances_  = np.stack([variance for variance in variances if variance is not None])

variances_m = np.mean(variances_, axis=0)**0.5
variances_s = np.std(variances_, axis=0)

fig = plt.figure(figsize=(10, 5))
plt.bar(names, variances_m)
plt.errorbar(names, variances_m, yerr=variances_s, fmt='.', color='black')
plt.yscale('log')
plt.grid()
plt.xticks(rotation=45)
plt.title('Fitted variables standard deviation')

#%%
# %matplotlib qt
# %matplotlib inline

from matplotlib.colors import SymLogNorm

norm = SymLogNorm(linthresh=1e-8, vmin=0, vmax=1)
# norm = 

Hessians_cube = np.stack(Hessians)

scale_matrix = np.zeros(Hessians_cube.shape[1:])

for i,j in np.ndindex(Hessians_cube.shape[1:]):
    scale_i = df_transforms[transform_entries[i]].transforms[0].scale()
    scale_j = df_transforms[transform_entries[j]].transforms[0].scale()
    scale_matrix[i,j] = scale_i*scale_j
  
Hessians_normalized = Hessians_cube / scale_matrix[None,...]

Hessians_m = np.nanmean(Hessians_cube, axis=0)
Hessians_v = np.nanvar(Hessians_cube, axis=0)

#%%
a = np.diag(Hessians_m)
b = np.diag(Hessians_v)

plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.bar(names, a)
plt.errorbar(names, a, yerr=b, fmt='.', color='black')
plt.xticks(rotation=45, ha='right')
plt.xlim(-0.5, 10.5)
plt.ylim(-0.005, 0.005)
plt.show()

#%%
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.bar(names, a)
plt.errorbar(names, a, yerr=b, fmt='.', color='black')
plt.xticks(rotation=45, ha='right')
plt.xlim(10.5, 21.5)
# plt.ylim(-0.005, 0.005)

plt.show()

#%%
plt.imshow(Hessians_m, interpolation='nearest')
plt.gca().xaxis.set_ticks_position('top')
plt.xticks(np.arange(len(names)), names, rotation=45, ha='left')
plt.yticks(np.arange(len(names)), names, rotation=0)
plt.colorbar()

#%%
Hessians_Snyders_cut = Hessians_m[:11,:11]

plt.imshow(np.flip(Hessians_Snyders_cut, 0), interpolation='nearest')
plt.gca().xaxis.set_ticks_position('top')
plt.xticks(np.arange(Hessians_Snyders_cut.shape[0]), names[:Hessians_Snyders_cut.shape[0]], rotation=45, ha='left')
plt.yticks(np.flip(np.arange(Hessians_Snyders_cut.shape[1]),0), names[:Hessians_Snyders_cut.shape[0]], rotation=0)
plt.xlim(-0.5,10.5)
plt.ylim(-0.5,10.5)
plt.colorbar()

