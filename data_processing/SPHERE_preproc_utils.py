import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import os
import re
import pickle
from .SPHERE_data import LoadSPHEREsampleByID
from tools.utils import SPHERE_PSF_spiders_mask, mask_circle
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from copy import deepcopy
from globals import MAX_NDIT, SPHERE_DATA_FOLDER, device


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
    for i in range(len(samples)):
        buf = samples[i]['spectra'].copy()
        samples[i]['spectra'] = [buf['central L']*1e-9, buf['central R']*1e-9]


def GenerateImages(samples, norm_regime, device):
    ims = []
    bgs = []
    normas = []

    N_pix = samples[0]['PSF L'].shape[-1]
    
    # Required functions
    check_center = lambda x: x if x[x.shape[-2]//2, x.shape[-1]//2] > 0 else x*-1 # for some reason, some images apperead flipped in sign
    make_tensor = lambda x: torch.tensor(x, device=device) if type(x) is not torch.Tensor else x
    mask_noise = (1-mask_circle(N_pix, 80, center=(0,0), centered=True)) * SPHERE_PSF_spiders_mask(N_pix, thick=10)

    # Prepare images for TipTop
    for i in range(len(samples)):
        # this function copllapses a DITs and normalizes it
        PSF_norm = lambda x: x.sum() if norm_regime == 'sum' else x.max() if norm_regime == 'max' else 1
        
        buf_norms = np.zeros([2])
        buf_bg    = np.zeros([2])
        buf_im    = []

        def process_PSF(entry):
            PSF_  = check_center(samples[i][entry].squeeze())
            bg_   = np.median(PSF_[mask_noise > 0])
            PSF_ -= bg_
            norm_ = PSF_norm(PSF_*(1-mask_noise))
            return (PSF_+bg_)/norm_, bg_/norm_, norm_

        if 'PSF L' in samples[i].keys():
            PSF, bg, norm = process_PSF('PSF L')
            buf_norms[0] = norm
            buf_bg[0] = bg
            buf_im.append(PSF)

        if 'PSF R' in samples[i].keys():
            PSF, bg, norm = process_PSF('PSF R')
            buf_norms[1] = norm
            buf_bg[1] = bg
            buf_im.append(PSF)

        ims.append(np.stack(buf_im))
        bgs.append(buf_bg)
        normas.append(buf_norms)

    # outputs torch parameters
    return make_tensor(np.stack(ims)), make_tensor(np.stack(bgs)), make_tensor(np.stack(normas)).squeeze()


def SPHERE_preprocess(sample_ids, regime, norm_regime, synth=False):
    if regime == '1P21I':
        data_samples = SamplesByIds(sample_ids, synth)
        data_samples[0]['PSF L'] = data_samples[0]['PSF L'].mean(axis=0)[None,...]
        data_samples[0]['PSF R'] = data_samples[0]['PSF L'].mean(axis=0)[None,...]

    elif regime == 'NP2NI' or regime == '1P2NI':
        if len(sample_ids) > 1:
            print('****** Warning: Only one sample ID can be used in this regime! ******')
        f = LoadSPHEREsynthByID if synth else LoadSPHEREsampleByID
        data_samples = SamplesFromDITs( f(sample_ids[0]) )

    if regime == '1P2NI':
        data_samples = [data_samples[0]]

    OnlyCentralWvl(data_samples)
    PSF_0, bg, norms = GenerateImages(data_samples, norm_regime, device)

    # Manage config files
    path_ini = '../data/parameter_files/irdis.ini'

    config_file = ParameterParser(path_ini).params
    config_manager = ConfigManager( GetSPHEREsynth() if synth else GetSPHEREonsky() )
    merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample) for sample in data_samples])
    config_manager.Convert(merged_config, framework='pytorch', device=device)

    return PSF_0, bg, norms, data_samples, merged_config

