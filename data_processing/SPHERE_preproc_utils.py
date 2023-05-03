import sys
sys.path.insert(0, '..')

import numpy as np
import torch

from .SPHERE_data import LoadSPHEREsampleByID
from tools.utils import BackgroundEstimate
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from copy import deepcopy

from globals import MAX_NDIT

def SamplesByIds(ids):
    data_samples = []
    for id in ids:
        data_samples.append( LoadSPHEREsampleByID(id) )
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

    make_tensor = lambda x: torch.tensor(x, device=device) if type(x) is not torch.Tensor else x

    # Preprocess input data so TipToy can understand it
    for i in range(len(samples)):
        bg_est = lambda x: BackgroundEstimate(x, radius=80).item()
        check_center = lambda x: x[x.shape[0]//2, x.shape[1]//2] > 0 # for some reason, ssome images apperead flipped in sign

        def process_PSF(x): # this function copllapses a DITs and normalizes it
            x = x.sum(axis=0)
            if   norm_regime == 'sum': x /= x.sum()
            elif norm_regime == 'max': x /= x.max()
            return x

        buf_im = []
        buf_bg = []
        buf_norms = []

        if 'PSF L' in samples[i].keys():
            buf_norms.append( samples[i]['PSF L'].sum(axis=(1,2)) )
            buf_im.append( process_PSF(samples[i]['PSF L']) )
            if not check_center(buf_im[-1]): buf_im[-1] *= -1
            buf_bg.append( bg_est(buf_im[-1]) )

        if 'PSF R' in samples[i].keys():
            buf_norms.append( samples[i]['PSF R'].sum(axis=(1,2)) )
            buf_im.append( process_PSF(samples[i]['PSF R']) )
            if not check_center(buf_im[-1]): buf_im[-1] *= -1
            buf_bg.append( bg_est(buf_im[-1]) )

        ims.append(np.stack(buf_im))
        bgs.append(buf_bg)
        normas.append(buf_norms)

    # outputs torch parameters
    return make_tensor(np.stack(ims)), make_tensor(np.stack(bgs)), make_tensor(np.stack(normas)).squeeze()


def SPHERE_preprocess(sample_ids, regime, norm_regime, device):
    if regime == '1P21I':
        data_samples = SamplesByIds(sample_ids)
    elif regime == 'NP2NI' or regime == '1P2NI':
        if len(sample_ids) > 1:
            print('****** Warning: Only one sample ID can be used in this regime! ******')
        data_samples = SamplesFromDITs(LoadSPHEREsampleByID(sample_ids[0]))

    OnlyCentralWvl(data_samples)
    PSF_0, bg, norms = GenerateImages(data_samples, norm_regime, device)

    if regime == '1P2NI':
        data_samples = [data_samples[0]]
        bg = bg.mean(dim=0)

    # Manage config files
    path_ini = '../data/parameter_files/irdis.ini'

    config_file = ParameterParser(path_ini).params
    config_manager = ConfigManager(GetSPHEREonsky())
    merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample) for sample in data_samples])
    config_manager.Convert(merged_config, framework='pytorch', device=device)

    return data_samples, PSF_0, bg, norms, merged_config

