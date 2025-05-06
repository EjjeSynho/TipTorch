import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import os
import re
import pickle
from .SPHERE_data import LoadSPHEREsampleByID
from tools.utils import mask_circle
from managers.parameter_parser import ParameterParser
from managers.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from copy import deepcopy
from project_globals import MAX_NDIT, SPHERE_DATA_FOLDER #, device
from tools.utils import rad2mas, pad_lists, cropper, gaussian_centroid
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
import warnings
from skimage.restoration import inpaint
from skimage.morphology import binary_dilation, disk, binary_erosion, square


# delay = lambda r: (0.0017+81e-6)*r #81 microseconds is the constant SPARTA latency, 17e-4 is the imperical constant
# frame_delay = lambda r: r/1e3 * 2.3  if (r/1e3*2.3) > 1.0 else 1.0 # delay of 2.3 frames for 1000 Hz loop rate
frame_delay = lambda r: torch.clamp(r/1e3 * 2.3, min=1.0)


def SPHERE_PSF_spiders_mask(crop, thick=9):
    def draw_line(start_point, end_point, thickness=10):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (256,256), (255,255,255))
        draw = ImageDraw.Draw(image)
        draw.line([start_point, end_point], fill=(0,0,0), width=thickness)
        return np.round(np.array(image).mean(axis=2) / 256).astype('uint')

    w, h = 256, 256
    dw, dh = 7, 7
    line_mask = draw_line((w, h-dh), (w//2+dw, h//2), thick) * \
                draw_line((0, h-dh), (w//2-dw, h//2), thick) * \
                draw_line((w, dh),   (w//2+dw, h//2), thick) * \
                draw_line((0, dh),   (w//2-dw, h//2), thick)
    
    crop = slice(line_mask.shape[0]//2-crop//2, line_mask.shape[0]//2+crop//2+crop%2)
    return line_mask[crop,crop]


def LoadSPHEREsynthByID(target_id):
    directory = SPHERE_DATA_FOLDER+'IRDIS_synthetic/'
    target_id = str(target_id)

    for filename in os.listdir(directory):
        match = re.match(r'(\d+)_synth.pickle', filename)
        if match and match.group(1) == target_id:
            # Load pickle file
            with open(directory + filename, 'rb') as f:
                data = pickle.load(f)
                return data
    return None


def SamplesByIds(ids, synth=False):
    data_samples = []
    f = LoadSPHEREsynthByID if synth else LoadSPHEREsampleByID
    for id in ids:
        data_samples.append(f(id))
    return data_samples


def SamplesFromDITs(init_sample):
    data_samples1 = []
    N_DITs = init_sample['PSF L'].shape[0]
    
    if N_DITs > MAX_NDIT: 
        warnings.warn(f'Warning: {N_DITs} DITs might be too many to fit into VRAM!')

    for i in range(init_sample['PSF L'].shape[0]):
        data_samples1.append( deepcopy(init_sample) )

    for i, sample in enumerate(data_samples1):
        sample['PSF L'] = init_sample['PSF L'][i,...][None,...]
        sample['PSF R'] = init_sample['PSF R'][i,...][None,...]

    return data_samples1


def process_mask(mask):
    mask_modified = np.zeros_like(mask.cpu().numpy())
    N_src, N_wvl = mask.shape[0], mask.shape[1]

    for i in range(N_src):
        for j in range(N_wvl):
            mask_layer = mask[i,j,...].cpu().numpy()
            mask_layer = binary_dilation(
                binary_erosion(mask_layer, square(3)),
                disk(3)
            )
            mask_modified[i,j,...] = mask_layer
            
    return torch.tensor(mask_modified).to(mask.device)


def separate_background(img, mask=None):
    bkg_estimator = MedianBackground()
    bkg = Background2D(img, (25,)*2, filter_size=(3,), bkg_estimator=bkg_estimator)
    
    if mask is None:
        return bkg.background
    else:
        return inpaint.inpaint_biharmonic(bkg.background, mask)


def OnlyCentralWvl(samples):
    if isinstance(samples, list):
        for i in range(len(samples)):
            buf = samples[i]['spectra'].copy()
            samples[i]['spectra'] = [buf['central L']*1e-9, buf['central R']*1e-9]
    else:
        buf = samples['spectra'].copy()
        samples['spectra'] = [buf['central L']*1e-9, buf['central R']*1e-9]


def GetJitter(synth_sample, synth_config):
    TT_res = synth_sample['WFS']['tip/tilt residuals']
    D = synth_config['telescope']['TelescopeDiameter']
    ang_pix = synth_sample['Detector']['psInMas'] / rad2mas
    jitter = lambda a: 2*2*a/D/ang_pix
    TT_jitter = jitter(TT_res)
    Jx = TT_jitter[:,0].std() * ang_pix * rad2mas * 2.355
    Jy = TT_jitter[:,1].std() * ang_pix * rad2mas * 2.355
    return Jx, Jy


def PSF_mask(img, center=(0,0)):
    N_pix          = img.shape[0]
    mask_PSF_inner = 1.0 - mask_circle(N_pix, 30, center=(0,0), centered=True)
    mask_PSF_outer = 1.0 - mask_circle(N_pix, 80, center=(0,0), centered=True)
    mask_noise     = (SPHERE_PSF_spiders_mask(N_pix, thick=12) + mask_PSF_outer) * mask_PSF_inner
    mask_noise     = np.roll(mask_noise, center[0],  axis=0)
    mask_noise     = np.roll(mask_noise, center[1], axis=1)
    return np.clip(mask_noise, 0, 1)


def process_PSF(PSF, bg_map, cropper): 
    check_center = lambda x: x if x[x.shape[-2]//2, x.shape[-1]//2] > 0 else x*-1 # for some reason, some images apperead flipped in sign
    
    PSF_subtr = check_center(PSF) - bg_map
    
    mask_valid_pix = np.ones_like(PSF_subtr)
    
    _, bg_median, std = sigma_clipped_stats(PSF_subtr, sigma=(N_sigma := 2))

    mask_valid_pix[np.abs(PSF_subtr) < bg_median + N_sigma * std] = 0.0
    
    PSF_subtr = PSF_subtr[cropper]
    bg_map = bg_map[cropper]
    mask_valid_pix = mask_valid_pix[cropper]
    
    norma = np.sum(PSF_subtr * mask_valid_pix) # normalization factor
    
    mask_valid_pix[PSF_subtr < 0] = 0.0 # to prevent a bias by only-positives

    return PSF_subtr, norma, mask_valid_pix.astype(np.int8)


def process_PSF_cube(PSF_cube, bg_map, cropper):
    '''This function normalizes and removes background for every DIT in a given PSF cube'''
    N_DIT  = PSF_cube.shape[0]
    PSFs   = np.zeros_like(PSF_cube[cropper], dtype=PSF_cube.dtype)
    masks  = np.zeros_like(PSF_cube[cropper], dtype=np.int8)
    norms  = np.zeros([N_DIT], dtype=PSF_cube.dtype)
    
    for dit in range(N_DIT): 
        PSFs[dit,...], norms[dit], masks[dit,...] = process_PSF(PSF_cube[dit,...].copy(), bg_map, cropper)
    
    return PSFs, norms, masks


def ProcessPSFCubes(data_samples, size):

    def compute_from_cube(cube):        
        PSF_mean = cube.mean(axis=0)
        PSF_var  = cube.var (axis=0)
        
        # center_mask = mask_circle(PSF_mean.shape[0], 50, center=(0,0), centered=True)
        # y,x = gaussian_centroid(PSF_mean*center_mask)
        
        crop     = cropper(PSF_mean, size)#, (x,y))
        mask_PSF = 1-PSF_mask(PSF_mean, center=(0,0)).astype(int) # cover the PSF with a wings
        bg_map   = separate_background(PSF_mean, mask_PSF)
        
        PSF_mean, norm_mean, mask_mean = process_PSF(PSF_mean, bg_map, crop)
        PSF_cube, norms, masks = process_PSF_cube(cube, bg_map, crop)

        # Normalize
        PSF_mean /= norm_mean
        PSF_var   = PSF_var[crop] / norm_mean**2
        bg_map   /= norm_mean

        return PSF_mean, PSF_var, PSF_cube, norm_mean, norms, mask_mean, masks, bg_map 
    

    PSF_data = []
    for data_sample in data_samples:
    
        PSF_L_mean, PSF_L_var, PSFs_L, norm_L_mean, norms_L, mask_L_mean, masks_L, bg_map_L = compute_from_cube(data_sample['PSF L'])
        PSF_R_mean, PSF_R_var, PSFs_R, norm_R_mean, norms_R, mask_R_mean, masks_R, bg_map_R = compute_from_cube(data_sample['PSF R'])

        data_record = {
            'norm (cube)':   np.stack([norms_L,     norms_R    ], axis=-1),
            'PSF (cube)':    np.stack([PSFs_L,      PSFs_R     ], axis= 1),
            'PSF (mean)':    np.stack([PSF_L_mean,  PSF_R_mean ], axis= 0),
            'PSF (var)':     np.stack([PSF_L_var,   PSF_R_var  ], axis= 0),
            'mask (mean)':   np.stack([mask_L_mean, mask_R_mean], axis= 0),
            'nask (cube)':   np.stack([masks_L,     masks_R    ], axis= 1),
            'bg map (mean)': np.stack([bg_map_L,    bg_map_R   ], axis= 0),
            'norm (mean)':   np.array([norm_L_mean, norm_R_mean])
        }
        PSF_data.append(data_record)
        
    return PSF_data


def SPHERE_preprocess(sample_ids, split_cube, PSF_loader, config_loader, framework, device):

    make_tensor  = lambda x: torch.tensor(x, device=device) if type(x) is not torch.Tensor else x

    if not isinstance(sample_ids, list):
        data_samples = [sample_ids]
        
    if not split_cube:
        data_samples = PSF_loader(sample_ids)
    else:
        data_samples = []
        for id in sample_ids:
            data_samples += SamplesFromDITs( PSF_loader([id])[0] )
            
    OnlyCentralWvl(data_samples)

    PSF_data = ProcessPSFCubes(data_samples, size=111)

    # Manage config files
    config_manager = ConfigManager()
    config_file    = ParameterParser('../data/parameter_files/irdis.ini').params
    merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample, *config_loader()) for sample in data_samples])

    merged_config['atmosphere']['Cn2Weights']    = pad_lists(merged_config['atmosphere']['Cn2Weights'], 0)
    merged_config['atmosphere']['Cn2Heights']    = pad_lists(merged_config['atmosphere']['Cn2Heights'], 1e4)
    merged_config['atmosphere']['WindDirection'] = pad_lists(merged_config['atmosphere']['WindDirection'], 0)
    merged_config['atmosphere']['WindSpeed']     = pad_lists(merged_config['atmosphere']['WindSpeed'], 0)

    config_manager.Convert(merged_config, framework=framework, device=device)

    merged_config['sensor_science']['FieldOfView'] = PSF_data[0]['PSF (mean)'].shape[-1]

    # To save memory, delete PSFs from the samples
    for sample in data_samples:
        del sample['PSF L']
        del sample['PSF R']
  
    if framework.lower() == 'pytorch':
        for data_record in PSF_data:
            for key in data_record.keys():
                data_record[key] = make_tensor(data_record[key])

    return PSF_data, data_samples, merged_config
