#%%
# %reload_ext autoreload
# %autoreload 2

import os
import sys
sys.path.insert(0, '..')

import numpy as np
import pickle
import torch
from torch import nn
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from tools.utils import OptimizeLBFGS, SR, pdims, FitGauss2D
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from tools.config_manager import ConfigManager, GetSPHEREonsky

from project_globals import SPHERE_DATA_FOLDER, SPHERE_FITTING_FOLDER, device

# device = torch.device('cuda:0')

#% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]

good_ids = psf_df.index.values.tolist()

ids_fitted = [ int(file.split('.')[0]) for file in os.listdir(SPHERE_FITTING_FOLDER) ]
good_ids = list(set(good_ids) - set(ids_fitted))

#%%
regime = '1P21I'
# regime = '1P2NI'
# regime = 'NP2NI'
norm_regime = 'sum'

def gauss_fitter(PSF_stack):
    FWHMs = np.zeros([PSF_stack.shape[0], PSF_stack.shape[1], 2])
    for i in range(PSF_stack.shape[0]):
        for l in range(PSF_stack.shape[1]):
            f_x, f_y = FitGauss2D(PSF_stack[i,l,:,:].float())
            FWHMs[i,l,0] = f_x.item()
            FWHMs[i,l,1] = f_y.item()
    return FWHMs

to_store = lambda x: x.detach().cpu().numpy()


def load_and_fit_sample(id):
    sample_ids = [id]
    PSF_0, bg, norms, _, merged_config = SPHERE_preprocess(sample_ids, regime, norm_regime, device)

    # Jx = merged_config['sensor_HO']['Jitter X'].abs()
    # Jy = merged_config['sensor_HO']['Jitter Y'].abs()

    # toy = TipTorch(merged_config, norm_regime, device)
    toy = TipTorch(merged_config, None, device)

    optimizables = ['F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']
    toy.optimizables = optimizables
    _ = toy({
        'F':   torch.tensor([0.89, 0.91]*toy.N_src, device=toy.device).flatten(),
        'Jx':  torch.tensor([28.8]*toy.N_src, device=toy.device).flatten(),
        'Jy':  torch.tensor([28.8]*toy.N_src, device=toy.device).flatten(),
        'Jxy': torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
        'bg':  bg.to(device)
    })

    PSF_1 = toy()
    PSF_DL = toy.DLPSF()

    # mask_in  = toy.mask_rim_in.unsqueeze(1).float()
    # mask_out = toy.mask_rim_out.unsqueeze(1).float()

    loss = nn.L1Loss(reduction='sum')

    '''
    window_loss = lambda x, x_max: \
        torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2

    def loss_fn(a,b):
        z = loss(a,b) + \
            window_loss(toy.r0, 0.5).sum() * 1.0 + \
            window_loss(toy.Jx, 50).sum() * 0.5 + \
            window_loss(toy.Jy, 50).sum() * 0.5 + \
            window_loss(toy.Jxy, 400).sum() * 0.5 + \
            window_loss(toy.dn + toy.NoiseVariance(toy.r0), toy.NoiseVariance(toy.r0).max()*1.5).sum() * 0.1
            torch.lt(torch.max(toy.PSD*mask_in), torch.max(toy.PSD*mask_out)).float() #+ \        
        return z
    '''
    
    loss_fn = loss

    optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn)
    
    optimizer_lbfgs.Optimize(PSF_0, [toy.bg], 5)
    for _ in range(10):
        optimizer_lbfgs.Optimize(PSF_0, [toy.F], 3)
        optimizer_lbfgs.Optimize(PSF_0, [toy.dx, toy.dy], 3)
        # optimizer_lbfgs.Optimize(PSF_0, [toy.r0, toy.dn], 5)
        # optimizer_lbfgs.Optimize(PSF_0, [toy.wind_dir, toy.wind_speed], 3)
        optimizer_lbfgs.Optimize(PSF_0, [toy.Jx, toy.Jy], 3)
        optimizer_lbfgs.Optimize(PSF_0, [toy.Jxy], 3)

    PSF_1 = toy()
    
    config_manager = ConfigManager(GetSPHEREonsky())
    config_manager.Convert(merged_config, framework='numpy')
    # config_manager.process_dictionary(merged_config)

    save_data = {
        'comments':    'Photons are multiplied by rate, no PSD regularization',
        'optimized':   optimizables,
        'config':      merged_config,
        'bg':          to_store(bg),
        'F':           to_store(toy.F),
        'dx':          to_store(toy.dx),
        'dy':          to_store(toy.dy),
        'r0':          to_store(toy.r0),
        'n':           to_store(toy.NoiseVariance(toy.r0.abs())),
        'dn':          to_store(toy.dn),
        'Jx':          to_store(toy.Jx),
        'Jy':          to_store(toy.Jy),
        'Jxy':         to_store(toy.Jxy),
        'Nph WFS':     to_store(toy.WFS_Nph),
        'SR data':     SR(PSF_0, PSF_DL).detach().cpu().numpy(),
        'SR fit':      SR(PSF_1, PSF_DL).detach().cpu().numpy(),
        'FWHM fit':    gauss_fitter(PSF_0), 
        'FWHM data':   gauss_fitter(PSF_1),
        'Img. data':   to_store(PSF_0*pdims(norms,2)),
        'Img. fit':    to_store(PSF_1*pdims(norms,2)),
        'PSD':         to_store(toy.PSD),
        'Data norms':  to_store(norms),
        'Model norms': to_store(toy.norm_scale),
        'loss':        loss_fn(PSF_1, PSF_0).item()
    }
    return save_data

#%%
for id in good_ids:
    filename = SPHERE_FITTING_FOLDER + str(id) + '.pickle'
    try:
        save_data = load_and_fit_sample(id)
        with open(filename, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(e)
        print('Failed to fit sample', id)
        continue
    