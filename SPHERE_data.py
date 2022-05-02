#%%
import os
import torch
import numpy as np
from astropy.io import fits
from os import path
import matplotlib.pyplot as plt
import pandas as pd

data_path = 'C:/Users/akuznets/Data/SPHERE/SPHERE_DC_DATA/'
path_dtts = 'C:/Users/akuznets/Data/SPHERE/DATA/DTTS/'

folders = os.listdir(data_path)


#%%
fits_files = []
for folder in folders:
    files = os.listdir(path.join(data_path,folder))
    for file in files:
        fits_files.append(path.join(data_path,folder,file))

#%%

def FilterNoiseBG(img, center, radius=80):
    # Mask empty area of the image
    mask_empty = np.ones(img.shape[0:2])
    mask_empty[np.where(img[:,:]==img[0,0])] = 0.
    # Mask center
    xx,yy = np.meshgrid( np.arange(0,img.shape[1]), np.arange(0,img.shape[0]) )
    mask_PSF = np.ones(img.shape[0:2])
    mask_PSF[np.sqrt((yy-center[0])**2 + (xx-center[1])**2) < radius] = 0.
    # Noise mask
    PSF_background = img[:,:]*mask_PSF*mask_empty
    noise_stats_mask = np.ones(img.shape[0:2])
    PSF_background[PSF_background == 0.0] = np.nan
    PSF_background[np.abs(PSF_background) > 3*np.nanstd(PSF_background)] = np.nan
    noise_stats_mask[np.abs(PSF_background) > 3*np.nanstd(PSF_background)] = 0.0
    return PSF_background, mask_empty*noise_stats_mask


id = 200

hdr = fits.open(fits_files[id])
im = hdr[0].data.sum(axis=0)

# Crop image
crop_size = 256
ind = np.unravel_index(np.argmax(im, axis=None), im.shape)
win_x = [ ind[0]-crop_size//2, ind[0]+crop_size//2 ]
win_y = [ ind[1]-crop_size//2, ind[1]+crop_size//2 ]

crop_x = slice(win_x[0], win_x[1])
crop_y = slice(win_y[0], win_y[1])

im = im[crop_x, crop_y]

ind = np.unravel_index(np.argmax(im, axis=None), im.shape)
background, mask = FilterNoiseBG(im, ind, radius=120)

median = np.nanmedian(background)
print(median)
im -= median

plt.imshow(np.log(im))
plt.show()

#%%
import sys
sys.path.insert(1, 'C:\\Users\\akuznets\\Projects\\TIPTOP\\P3\\')
import telemetry.sphereUtils as sphereUtils


def get_telescope_pointing():
    TELAZ = float(hdr[0].header['HIERARCH ESO TEL AZ'])
    TELALT = float(hdr[0].header['HIERARCH ESO TEL ALT'])
    airmass = 0.5*(float(hdr[0].header['HIERARCH ESO TEL AIRM END']) + float(hdr[0].header['ESO TEL AIRM START']))
    return TELAZ, TELALT, airmass


def get_wavelength():
    filter_data = {
        'B_Y':  [1043,140],
        'B_J':  [1245,240],
        'B_H':  [1625,290],
        'B_Ks': [2182,300]
    }
    filter_name = hdr[0].header['HIERARCH ESO INS1 FILT NAME']

    wvl = 1e-9*filter_data[filter_name][0]
    bw  = 1e-9*filter_data[filter_name][1]
    return wvl, bw


def get_ambi_parameters():
    tau0 = float(hdr[0].header['HIERARCH ESO TEL AMBI TAU0'])
    wDir = float(hdr[0].header['HIERARCH ESO TEL AMBI WINDDIR'])
    wSpeed = float(hdr[0].header['HIERARCH ESO TEL AMBI WINDSP'])
    RHUM  = float(hdr[0].header['HIERARCH ESO TEL AMBI RHUM'])
    pressure = 0.5*(float(hdr[0].header['HIERARCH ESO TEL AMBI PRES START']) + float(hdr[0].header['HIERARCH ESO TEL AMBI PRES END']))
    fwhm_linobs = float(hdr[0].header['HIERARCH ESO TEL IA FWHMLINOBS'])
    return tau0, wDir, wSpeed, RHUM, pressure , fwhm_linobs


#%%


data_path = 'C:/Users/akuznets/Data/SPHERE/SPHERE_DC_DATA/'
path_dtts = 'C:/Users/akuznets/Data/SPHERE/DATA/DTTS/'
date = hdr[0].header['DATE'][:10]

path_data=data_path
date_obs=date
which='last'
n_subap=1240

'''
path_sparta = path_data + '/ird_convert_recenter_dc5-SPH_SPARTA_PSFDATA-psf_sparta_data.fits'
if os.path.isfile(path_sparta):
    sparta = fits.getdata(path_sparta)
    # number of acquisitions during the observation
    nPSF = sparta.shape[1]-1
    # note : nPSF must == nFiles/2 with raw images or im.shape[0]//2 with processed images
    r0 = sparta[0, :, :]
    vspeed = sparta[1, :, :]
    SR = sparta[2, :, :]
    seeing = sparta[3, :, :]

    if which == 'last':
        r0 = r0[nPSF, :]
        vspeed = vspeed[nPSF, :]
        SR = SR[nPSF, :]
        seeing = seeing[nPSF, :]
'''
# grab the number of photons
def find_closest(df, date, Texp):
    df["date"] = pd.to_datetime(df["date"])
    id_closest = np.argmin(abs(df["date"] - date))
    date_min = df["date"][id_closest] - pd.DateOffset(seconds=Texp/2)
    date_max = df["date"][id_closest] + pd.DateOffset(seconds=Texp/2)
    id_min = np.argmin(abs(df["date"] - date_min))
    id_max = np.argmin(abs(df["date"] - date_max))
    return id_min, id_max

n_ph = np.nan
rate = np.nan

if path_dtts is not None and date_obs is not None:
    year = date_obs[:4]
    path_sub = path_dtts + year + "/" + date_obs
    if os.path.isdir(path_sub):
        # grab the exposure time and the exact date
        #hdr = fits.getheader(path_sparta)
        Texp  = hdr[0].header['HIERARCH ESO DET SEQ1 DIT'] * hdr[0].header['HIERARCH ESO DET NDIT']
        date = pd.to_datetime(date_obs)
        # Get the number of photons
        tmp = [file for file in os.listdir(path_sub) if "sparta_visible_WFS" in file]
        if len(tmp)>0:
            file_name = tmp[0]
            df = pd.read_csv(path_sub + "/" + file_name)
            if 'flux_VisLoop[#photons/aperture/frame]' in df:
                df_flux = df['flux_VisLoop[#photons/aperture/frame]']
                id_min, id_max = find_closest(df, date, Texp)
                n_ph = df_flux[id_min:id_max+1].median()/n_subap
            if 'Frame rate [Hz]' in df:
                df_rate = df['Frame rate [Hz]']
                rate = df_rate[id_min:id_max+1].median()

#return r0, vspeed, SR, seeing, n_ph, rate




#%%

TELAZ, TELALT, airmass = get_telescope_pointing() # get telescope pointing
wvl, bw = get_wavelength() # get spectral information
#VMAG, RMAG, GMAG, JMAG, HMAG, KMAG, RA, DEC = sphereUtils.get_star_magnitudes(hdr)
fwhm, tau0, wSpeed, wDir, RHUM, pressure = get_ambi_parameters() # get atmospheric parameters
#SRMEAN, SRMIN, SRMAX = sphereUtils.read_strehl_value(hdr) # get Strehl values
#r0, vspeed, SR, seeing, n_ph, rate = sphereUtils.read_sparta_data(fits_files[id], date, path_dtts, 'last') # get SPARTA data
#psInMas, gain, ron, DIT, NDIT, _ = sphereUtils.get_detector_config(hdr) # get detector config



# %%


params['atmosphere']['Seeing'] = seeing
params['atmosphere']['WindSpeed'] = [wSpeed]
params['atmosphere']['WindDirection'] = [wDir]
params['sensor_science']['SigmaRON'] = ron
params['sensor_science']['Gain'] = gain