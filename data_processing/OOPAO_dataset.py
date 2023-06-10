#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from project_globals import SPHERE_DATA_FOLDER
from data_processing.SPHERE_preproc_utils import GetJitter

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

entries_df = [
    'ID', 'Airmass', 'r0', 'Seeing', 'Strehl (IR)',
    'Strehl (vis)', 'Wind direction', 'Wind direction (200 mbar)', 'Jitter X', 'Jitter Y',
    'Wind speed', 'Wind speed (200 mbar)', 'Nph WFS', 'Rate', 'DIT Time', 'Num. DITs', 'WFS noise (nm)',
    'Flux WFS', 'Sequence time', 'λ left (nm)', 'λ right (nm)', 'λ WFS (nm)', 'invalid']

synth_df = []

files = os.listdir(samples_path)
for file in tqdm(files):
    with open(samples_path / Path(file), 'rb') as handle:
        sample = pickle.load(handle)

    sample_dict = {}
    
    id = int( file.split('_')[0] )
    sample_dict                   = { 'ID': id }
    sample_dict['invalid']        = not (id in good_ids)
    sample_dict['Strehl (IR)']    = sample['Strehl (IRDIS)']
    sample_dict['Strehl (vis)']   = sample['Strehl (SAXO)']
    sample_dict['Airmass']        = sample['r0']
    sample_dict['r0']             = sample['seeing']
    sample_dict['Seeing']         = sample['telescope']['airmass']
    sample_dict['Rate']           = sample['RTC']['loop rate']
    sample_dict['Nph WFS']        = sample['WFS']['Nph vis']
    sample_dict['λ WFS (nm)']     = sample['WFS']['wavelength']
    sample_dict['λ left (nm)']    = sample['spectra']['central R']
    sample_dict['λ right (nm)']   = sample['spectra']['central L']
    sample_dict['Num. DITs']      = 1
    sample_dict['DIT Time']       = 1.0 #TODO: redo in future!!!!
    sample_dict['Sequence time']  = sample_dict['DIT Time'] * sample_dict['Num. DITs']
    sample_dict['Wind speed']     = sample['Wind speed'][0]
    sample_dict['Wind direction'] = sample['Wind direction'][0]
    sample_dict['WFS noise (nm)'] = sample['WFS']['Reconst. error']
    sample_dict['Flux WFS']       = sample['parameters']['nPhotonPerSubaperture']
    sample_dict['Wind speed (200 mbar)']     = np.nan if len(sample['Wind speed']) == 1 else sample['Wind speed'][1]
    sample_dict['Wind direction (200 mbar)'] = np.nan if len(sample['Wind direction']) == 1 else sample['Wind direction'][1]

    Jx, Jy = GetJitter(sample, sample['config'])

    sample_dict['Jitter X'] = Jx
    sample_dict['Jitter Y'] = Jy
    
    synth_df.append(sample_dict)
    
synth_df = pd.DataFrame(synth_df)
synth_df = synth_df.set_index('ID')
synth_df.sort_index(inplace=True)

# Save dataframe
with open(SPHERE_DATA_FOLDER / Path('synth_df.pickle'), 'wb') as handle:
    pickle.dump(synth_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
