#%%
import numpy as np
from query_eso_archive import query_simbad
from scipy.ndimage import center_of_mass

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import os
from os import path
import matplotlib.pyplot as plt

'''
data_dir = path.normpath('C:/Users/akuznets/Data/MUSE/DATA/')
listData = os.listdir(data_dir)
sample_id = 5
sample_name = listData[sample_id]
path_im = path.join(data_dir, sample_name)
'''

'''
angle = np.zeros([len(listData)])
angle[0] = -46
angle[5] = -44
angle = angle[sample_id]
'''

class MUSEcube():

    def FilterNoiseBG(self, img, center, radius=80):
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


    def Center(self,im):
        WoG_ROI = 16
        center = np.array(np.unravel_index(np.argmax(im), im.shape))
        crop = slice(center[0]-WoG_ROI//2, center[1]+WoG_ROI//2)
        crop = (crop, crop)
        WoG = np.array(center_of_mass(im[crop])) + im.shape[0]//2-WoG_ROI//2
        return WoG


    def GetSpectrum(self):
        pix_radius = 7
        xx,yy = np.meshgrid( np.arange(0,self.cube_img.shape[1]), np.arange(0,self.cube_img.shape[0]) )
        fluxes = np.zeros(self.cube_img.shape[2])

        for i in range(self.cube_img.shape[2]):
            mask = np.copy(self.cube_img[:,:,i])
            ind = self.Center(mask)
            mask[np.sqrt((yy-ind[0])**2 + (xx-ind[1])**2) > pix_radius] = 0.0
            fluxes[i] = mask.sum()

        return fluxes / fluxes.max()


    def MaskAO(self, img, radius):
        xx,yy = np.meshgrid( np.arange(0,img.shape[1]), np.arange(0,img.shape[0]) )

        ind = np.unravel_index(np.argmax(img, axis=None), img.shape)
        mask = np.zeros(img.shape[0:2])
        mask[np.sqrt((yy-ind[0])**2 + (xx-ind[1])**2) < radius] = 1.0

        ind = center_of_mass(img*mask)
        mask = np.ones(img.shape[0:2])
        mask[np.sqrt((yy-ind[0])**2 + (xx-ind[1])**2) < radius] = 0.0
        return mask


    def __init__(self, path_im, crop_size, angle=0):
        cube_img = []
        cube_var = []
        wavelengths = []

        with fits.open(path_im) as hdul:
            self.obs_info = hdul[0].header
            for bandwidth_id in range(0,10):
                cube_img.append(hdul[bandwidth_id*4+1].data)
                cube_var.append(hdul[bandwidth_id*4+2].data)
                wavelengths.append(hdul[bandwidth_id*4+1].header['LAMBDAOB']*1e-10)
            
        if angle:
            from scipy.ndimage import rotate
            for i in range(len(cube_img)):
                cube_img[i] = rotate(cube_img[i], angle)
                cube_var[i] = rotate(cube_var[i], angle)

        self.wavelengths = np.array(wavelengths)
        cube_img = np.dstack(cube_img)
        cube_var = np.dstack(cube_var)
        polychrome = cube_img.sum(axis=2)

        #Find maximum of the composite PSF to crop it
        ind = np.unravel_index(np.argmax(polychrome, axis=None), polychrome.shape)

        win_x = [ ind[0]-crop_size//2, ind[0]+crop_size//2 ]
        win_y = [ ind[1]-crop_size//2, ind[1]+crop_size//2 ]

        # Estimate background
        meaningful_masks = []
        medians = []
        means = []
        for i in range(cube_img.shape[2]):
            background, meaningful_mask = self.FilterNoiseBG(cube_img[:,:,i], ind)
            medians.append( np.nanmedian(background) )
            means.append(   np.nanmean(background) )
            meaningful_masks.append(meaningful_mask)
        medians = np.array(medians)
        means   = np.array(means)
        meaningful_masks = np.dstack(meaningful_masks)

        #Crop image cubes
        crop_x = slice(int(win_x[0]), int(win_x[1]))
        crop_y = slice(int(win_y[0]), int(win_y[1]))

        self.cube_img   = cube_img  [crop_x, crop_y, :] #- medians
        self.cube_var   = cube_var  [crop_x, crop_y, :]
        self.polychrome = polychrome[crop_x, crop_y]
        self.meaningful_masks = meaningful_masks[ crop_x, crop_y, :]

        self.radii = np.linspace(12,19,10) # approximate AO-cutoff radius (empirical)


    def Layer(self, wvl_id):
        wvl = self.wavelengths[wvl_id]
        im  = self.cube_img[:,:,wvl_id]
        var = self.cube_var[:,:,wvl_id]
        return im, var, wvl


    def Masks(self, wvl_id):
        im, _, _ = self.Layer(wvl_id)
        weights_core = (1.0-self.MaskAO(np.copy(im), self.radii[wvl_id])) * self.meaningful_masks[:,:,wvl_id]
        weights_halo = self.MaskAO(np.copy(im), self.radii[wvl_id]) * self.meaningful_masks[:,:,wvl_id]
        return weights_core, weights_halo
