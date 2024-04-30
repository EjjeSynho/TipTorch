import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import os
import re
import pickle
from .SPHERE_data import LoadSPHEREsampleByID
from tools.utils import mask_circle
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from copy import deepcopy
from project_globals import MAX_NDIT, SPHERE_DATA_FOLDER #, device
from tools.utils import rad2mas, pad_lists, cropper

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
        print('***** WARNING! '+str(N_DITs)+' DITs might be too many to fit into VRAM! *****')
    else:
        # print('Split into '+str(N_DITs)+' samples')
        pass

    for i in range(init_sample['PSF L'].shape[0]):
        data_samples1.append( deepcopy(init_sample) )

    for i, sample in enumerate(data_samples1):
        sample['PSF L'] = init_sample['PSF L'][i,...][None,...]
        sample['PSF R'] = init_sample['PSF R'][i,...][None,...]

    return data_samples1


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


def ProcessPSFCubes(data_samples, norm_regime, bg_subtraction, size):
    # Auxiliary functions
    N_pix        = data_samples[0]['PSF L'].shape[-1]
    mask_noise   = (1-mask_circle(N_pix, 80, center=(0,0), centered=True)) * SPHERE_PSF_spiders_mask(N_pix, thick=10)
    check_center = lambda x: x if x[x.shape[-2]//2, x.shape[-1]//2] > 0 else x*-1 # for some reason, some images apperead flipped in sign
    PSF_norm     = lambda x: x.sum() if norm_regime == 'sum' else x.max() if norm_regime == 'max' else 1

    def process_PSF(x):
        PSF_  = check_center(x)
        bg_   = np.median(PSF_[mask_noise > 0])
        PSF_ -= bg_ # Remove background to compute normalizing factor more precisely
        
        norm_ = PSF_norm(PSF_*(1-mask_noise))
        
        if bg_subtraction:
            return PSF_/norm_, 0.0, norm_
        else:
            return (PSF_+bg_)/norm_, bg_/norm_, norm_


    def process_PSF_cube(PSF_cube):
        '''This function normalizes and removes background for every DIT in a given PSF cube'''
        N_DIT = PSF_cube.shape[0]
        PSFs  = np.zeros_like(PSF_cube)
        bgs   = np.zeros([N_DIT])
        norms = np.zeros([N_DIT])
        
        for dit in range(N_DIT): 
            PSFs[dit,...], bgs[dit], norms[dit] = process_PSF(PSF_cube[dit,...])
        
        return PSFs, bgs, norms

    PSF_data = []

    # crop_cube  = np.s_[:, :, N_pix//2-crop_windows//2:N_pix//2+crop_windows//2+a, N_pix//2-crop_windows//2:N_pix//2+crop_windows//2+a]
    # crop_slice = np.s_[:,    N_pix//2-crop_windows//2:N_pix//2+crop_windows//2+a, N_pix//2-crop_windows//2:N_pix//2+crop_windows//2+a]

    # Reduce the size of the output PSFs
    crop_cube  = cropper(data_samples[0]['PSF L'], size)
    crop_slice = crop_cube#cropper(data_samples[0]['PSF L'][0], size)

    # Process for all data samples
    for data_sample in data_samples:

        N_dits = data_sample['PSF L'].shape[0]
        assert N_dits == data_sample['PSF R'].shape[0]
    
        if 'PSF L' in data_sample.keys() and data_sample['PSF L'] is not None:
            PSFs_L, bgs_L, norms_L = process_PSF_cube(data_sample['PSF L'])
            
            PSF_L_mean = data_sample['PSF L'].mean(axis=0)
            PSF_L_var  = data_sample['PSF L'].var(axis=0)
            
            PSF_L_mean, bg_L_mean, norm_L_mean = process_PSF(PSF_L_mean)
            PSF_L_var /= norm_L_mean**2    

        if 'PSF R' in data_sample.keys() and data_sample['PSF R'] is not None:
            PSFs_R, bgs_R, norms_R = process_PSF_cube(data_sample['PSF R'])
            
            PSF_R_mean = data_sample['PSF R'].mean(axis=0)
            PSF_R_var  = data_sample['PSF R'].var(axis=0)
            
            PSF_R_mean, bg_R_mean, norm_R_mean = process_PSF(PSF_R_mean)
            PSF_R_var /= norm_R_mean**2

        data_record = {
            'PSF (cube)':  np.stack([PSFs_L,  PSFs_R],   axis= 1)[crop_cube],
            'bg (cube)':   np.stack([bgs_L,   bgs_R],     axis=-1),
            'norm (cube)': np.stack([norms_L, norms_R], axis=-1),
            
            'PSF (mean)':  np.stack([PSF_L_mean,  PSF_R_mean], axis= 0)[crop_slice],
            'PSF (var)':   np.stack([PSF_L_var,   PSF_R_var],  axis= 0)[crop_slice],
            'bg (mean)':   np.array([bg_L_mean,   bg_R_mean]),
            'norm (mean)': np.array([norm_L_mean, norm_R_mean])
        }
        PSF_data.append(data_record)
        
    return PSF_data


def SPHERE_preprocess(sample_ids, norm_regime, split_cube, PSF_loader, config_loader, framework, device):
    """
    This function outputs the PSF cubes corresponding to specific samples, while all configs are merged into a single config.
    If only a single sample is inputted, the function by default will output a single PSF cube and a single config.
    But you can split the PSF cube into multiple samples by using the 'split_cube' option.
    'loader_func' specifies how and from where to load the data from.
    'config_loader' specifies how to init TipTorch from input data samples. For example, synthetic data has a different config file than on-sky data
    """

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

    PSF_data = ProcessPSFCubes(data_samples, norm_regime, bg_subtraction=False, size=121)

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
