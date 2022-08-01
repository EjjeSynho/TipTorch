#%%
import numpy as np
from astropy.io import fits
import os
from os import path
import matplotlib.pyplot as plt
import json
from configparser import ConfigParser

folder = 'C:/Users/akuznets/Data/SPHERE/simulated/SPHERE4ARSENIY/20220616_175310.0/'

folder_PSF = folder[:-1] + '_PSF/'
PSF_files = os.listdir(folder_PSF)
#PSF_files.pop(PSF_files.index('PSF_DL.fits'))

k = 1
pathh = path.join(folder_PSF,PSF_files[k])
with fits.open(pathh) as hdul:
    print(pathh)
    img = hdul[0].data
    print(hdul[0].header)

crop = 16

img = img[img.shape[0]//2-crop:img.shape[0]//2+crop,img.shape[1]//2-crop:img.shape[1]//2+crop]
plt.imshow(np.log(img))

#%%
# Read simulation parameters
with open(folder + 'params.txt') as json_file:
    buffer = json_file.read()
    buffer = buffer.replace('Inf', 'Infinity')
    params = json.loads(buffer)

#%%

wvl = 5e-7 #[m]
k2 = (2*np.pi/5e-7)**2

rad2arc = 3600.0 * 180.0/np.pi

Cn2  = np.array(params['ATMO']['CN2'])
h  = np.array(params['ATMO']['HEIGHTS'])
a0 = params['SEEING']['CONSTANT'] / rad2arc

γ = 0.0 * np.pi/180.0

r0_base = 0.976*wvl/a0
r0_sim  = (0.423*k2*np.cos(γ)**-1 * np.trapz(Cn2,h))**(-3/5)



print(r0_base, r0_sim)

#H = ( np.trapz(F*h**(5/3),h) / np.trapz(F,h) )**(3/5)
#Theta_0 = 0.314 * r0/H * rad2arc #[asec]

#print(Theta_0)
