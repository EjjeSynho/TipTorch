#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from project_globals import SPHERE_DATA_FOLDER
import pickle
from tqdm import tqdm

samples_path = SPHERE_DATA_FOLDER / Path('IRDIS_synthetic')
save_path    = SPHERE_DATA_FOLDER / Path('OOPAO_images')

crop = 64
el_croppo = np.s_[crop:-crop, crop:-crop]

#%% Save images
files = os.listdir(samples_path)
for file in tqdm(files):
    with open(samples_path / Path(file), 'rb') as handle:
        sample = pickle.load(handle)
    imag = sample['PSF L'][0,...]
    id = file.split('_')[0]

    plt.figure(figsize=(5,5))
    plt.title(id)
    plt.imshow(np.log10(imag[el_croppo]))
    if not os.path.exists(save_path / Path(id + '.png')):
        plt.savefig(save_path / Path(id + '.png'))
    plt.axis('off')
    
#%% filter good images
files = os.listdir(save_path)
good_ids = [ int(file.split('.')[0]) for file in files ]

#%%
synth_df = []

files = os.listdir(samples_path)
for file in tqdm(files):
    with open(samples_path / Path(file), 'rb') as handle:
        sample = pickle.load(handle)
        
    id = int( file.split('_')[0] )
    sample_dict = { 'ID': id }
    
    if id in good_ids:
        sample_dict['invalid'] = False
    else:
        sample_dict['invalid'] = True
        
    synth_df.append(sample_dict)

synth_df = pd.DataFrame(synth_df)
synth_df = synth_df.set_index('ID')
synth_df.sort_index(inplace=True)

#%% Save dataframe
with open(SPHERE_DATA_FOLDER / Path('synth_df.pickle'), 'wb') as handle:
    pickle.dump(synth_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

