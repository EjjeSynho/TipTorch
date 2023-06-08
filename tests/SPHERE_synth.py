#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
from torch import nn, optim
import numpy as np
from tools.utils import OptimizeLBFGS, ParameterReshaper, plot_radial_profiles, SR, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
import matplotlib.pyplot as plt
from tools.utils import rad2mas, rad2arc
from project_globals import SPHERE_DATA_FOLDER, device


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]

# open synth df
with open(SPHERE_DATA_FOLDER+'synth_df.pickle', 'rb') as handle:
    synth_df = pickle.load(handle)
    

#%% ================================= Read OOPAO sample =================================
regime = '1P21I'
norm_regime = 'sum'

# sample_ids = [132]
# sample_ids = [1209]
sample_ids = [1452]

#% Initialize model
PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True, device=device)
toy = TipTorch(synth_config, norm_regime, device=device, TipTop=True, PSFAO=False, oversampling=1)


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

toy.optimizables = ['F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']
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

draw_PSF_stack(PSF_2, PSF_3, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_2), destack(PSF_3), 'OOPAO', 'TipTorch', title='IRDIS PSF', dpi=200)

#%%
# loss = nn.L1Loss(reduction='sum')
# # Confines a value between 0 and the specified value
# window_loss = lambda x, x_max: \
#     torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2
# def loss_fn(a,b):
#     z = loss(a,b) + \
#         window_loss(toy.r0, 0.5).sum() * 5.0 + \
#         window_loss(toy.Jx, 50).sum() * 0.5 + \
#         window_loss(toy.Jy, 50).sum() * 0.5 + \
#         window_loss(toy.Jxy, 400).sum() * 0.5 + \
#         window_loss(toy.dn + toy.NoiseVariance(toy.r0), 1.5).sum()
#     return z

loss_fn = nn.L1Loss(reduction='sum')

PSF_3  = toy()
PSF_DL = toy.DLPSF()

optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn)

for i in range(10):
    optimizer_lbfgs.Optimize(PSF_2, [toy.F], 3)
    optimizer_lbfgs.Optimize(PSF_2, [toy.bg], 2)
    optimizer_lbfgs.Optimize(PSF_2, [toy.dx, toy.dy], 3)
    optimizer_lbfgs.Optimize(PSF_2, [toy.r0, toy.dn], 5)
    optimizer_lbfgs.Optimize(PSF_2, [toy.wind_dir, toy.wind_speed], 3)
    optimizer_lbfgs.Optimize(PSF_2, [toy.Jx, toy.Jy, toy.Jxy], 3)

PSF_3 = toy()
print('\nStrehl ratio: ', SR(PSF_3, PSF_DL))

#%%
draw_PSF_stack(PSF_2, PSF_3, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_2), destack(PSF_3), 'OOPAO', 'TipTorch', title='IRDIS PSF', dpi=200)
