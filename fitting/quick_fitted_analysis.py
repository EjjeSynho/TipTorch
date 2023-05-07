#%%
from tqdm import tqdm
import numpy as np
import pickle
import os

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_NP2NI/'
fitted_files = os.listdir(fitted_folder)

# Read the fitted data

r0 = []

for file in tqdm(fitted_files):
    id = int(file.split('.')[0])

    with open(fitted_folder + file, 'rb') as handle:
        data = pickle.load(handle)
        r0.append(data['r0'])
