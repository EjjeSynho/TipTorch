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
from tools.config_manager import ConfigManager, GetSPHEREsynth
from tools.utils import rad2mas, rad2arc

from project_globals import SPHERE_DATA_FOLDER, SPHERE_OOPAO_FITTING_FOLDER, device

# device = torch.device('cuda:0')

#% Initialize data sample
with open(SPHERE_DATA_FOLDER+'synth_df.pickle', 'rb') as handle:
    synth_df = pickle.load(handle)

synth_df = synth_df[synth_df['invalid'] == False]
good_ids = synth_df.index.values.tolist()

ids_fitted = [ int(file.split('.')[0]) for file in os.listdir(SPHERE_OOPAO_FITTING_FOLDER) ]
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
    PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True, device=device)

    def GetJitter():
        TT_res = synth_samples[0]['WFS']['tip/tilt residuals']
        D = synth_config['telescope']['TelescopeDiameter']
        ang_pix = synth_samples[0]['Detector']['psInMas'] / rad2mas
        jitter = lambda a: 2*2*a/D/ang_pix
        TT_jitter = jitter(TT_res)
        Jx = TT_jitter[:,0].std() * ang_pix * rad2mas * 2.355
        Jy = TT_jitter[:,1].std() * ang_pix * rad2mas * 2.355
        return Jx, Jy

    Jx, Jy = GetJitter()

    toy = TipTorch(synth_config, norm_regime, device=device, TipTop=True, PSFAO=False, oversampling=1)

    optimizables = ['F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy', 'dn']
    toy.optimizables = optimizables
    _ = toy({ 
        'Jxy': torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
        'Jx':  torch.tensor([Jx]*toy.N_src, device=toy.device).flatten(),
        'Jy':  torch.tensor([Jy]*toy.N_src, device=toy.device).flatten(),
        'dx':  torch.tensor([0.0]*toy.N_src, device=toy.device).flatten(),
        'dy':  torch.tensor([0.0]*toy.N_src, device=toy.device).flatten(),
        'bg':  bg.to(device)
    })

    PSF_3 = toy()
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
    
    optimizer_lbfgs.Optimize(PSF_2, [toy.bg], 5)
    for _ in range(10):
        optimizer_lbfgs.Optimize(PSF_2, [toy.F], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.bg], 2)
        optimizer_lbfgs.Optimize(PSF_2, [toy.dx, toy.dy], 3)
        # optimizer_lbfgs.Optimize(PSF_2, [toy.r0, toy.dn], 5)
        optimizer_lbfgs.Optimize(PSF_2, [toy.dn], 5)
        # optimizer_lbfgs.Optimize(PSF_2, [toy.wind_dir, toy.wind_speed], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.Jx, toy.Jy], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.Jxy], 3)

    PSF_3 = toy()
    
    config_manager = ConfigManager(GetSPHEREsynth())
    config_manager.Convert(synth_config, framework='numpy')
    # config_manager.process_dictionary(merged_config)

    save_data = {
        'comments':    'Photons are NOT multiplied by rate, no PSD regularization',
        'optimized':   optimizables,
        'config':      synth_config,
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
        'Jx init':     to_store(torch.tensor([Jx]*toy.N_src, device=toy.device).flatten()),
        'Jy init':     to_store(torch.tensor([Jy]*toy.N_src, device=toy.device).flatten()),
        'Nph WFS':     to_store(toy.WFS_Nph),
        'SR data':     SR(PSF_2, PSF_DL).detach().cpu().numpy(),
        'SR fit':      SR(PSF_3, PSF_DL).detach().cpu().numpy(),
        'FWHM fit':    gauss_fitter(PSF_2), 
        'FWHM data':   gauss_fitter(PSF_3),
        'Img. data':   to_store(PSF_2*pdims(norms,2)),  #FIX WLV PROBLEM n only left is presents
        'Img. fit':    to_store(PSF_3*pdims(norms,2)),
        'PSD':         to_store(toy.PSD),
        'Data norms':  to_store(norms),
        'Model norms': to_store(toy.norm_scale),
        'loss':        loss_fn(PSF_2, PSF_3).item()
    }
    return save_data

# save_data = load_and_fit_sample(321)

#%%
for id in good_ids:
    filename = SPHERE_OOPAO_FITTING_FOLDER + str(id) + '.pickle'
    try:
        save_data = load_and_fit_sample(id)
        with open(filename, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(e)
        print('Failed to fit sample', id)
        continue
    