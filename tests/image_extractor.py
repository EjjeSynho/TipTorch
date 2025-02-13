#%%
import numpy as np
import os
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

color_map = cm.get_cmap('inferno')

list_strange = 'C:/Users/akuznets/Desktop/MUSE_NFM_strange_PSFs.txt'
save_folder = 'C:/Users/akuznets/Desktop/strange_PSFs/'

# Read lines
with open(list_strange, 'r') as f:
    lines = f.readlines()
    
im_folder = 'F:/ESO/Data/MUSE/MUSE_images/'

im_files = os.listdir(im_folder)

im_selected = [file_ for file_ in im_files for line in lines if line.strip() in file_]

#%%
for i in tqdm(range(len(im_selected))):
    im = PIL.Image.open(im_folder + im_selected[i])
    im_array = np.array(im)[37:710, 16:688, ...].mean(axis=-1)
    im = PIL.Image.fromarray(np.uint8(color_map(im_array/255)*255))
    im.save(save_folder+im_selected[i].split('_')[1])

# %%

aa = np.array([842, 781, 588, 488, 266,  79,  30,  14,  12,  11])