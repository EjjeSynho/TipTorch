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
from PSF_models.TipToy_SPHERE_multisrc import TipToy

from project_globals import SPHERE_DATA_FOLDER, SPHERE_FITTING_FOLDER, device


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
    PSF_0, bg, norms, data_samples, merged_config = SPHERE_preprocess(sample_ids, regime, norm_regime)

    toy = TipToy(merged_config, norm_regime, device, TipTop=False, PSFAO=True)

    toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy', 'amp', 'b', 'alpha', 'beta', 'ratio', 'theta']
    _ = toy({
        'Jxy': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
        'Jx':  torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
        'Jy':  torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
        'bg':  bg.to(device)
    })

    PSF_1 = toy()
    PSF_DL = toy.DLPSF()

    loss_fn = nn.L1Loss(reduction='sum')

    optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn)

    for _ in range(20):
        optimizer_lbfgs.Optimize(PSF_0, [toy.F, toy.dx, toy.dy], 2)
        optimizer_lbfgs.Optimize(PSF_0, [toy.bg], 2)
        optimizer_lbfgs.Optimize(PSF_0, [toy.b], 2)
        optimizer_lbfgs.Optimize(PSF_0, [toy.r0, toy.amp, toy.alpha, toy.beta], 3)
        optimizer_lbfgs.Optimize(PSF_0, [toy.ratio, toy.theta], 3)
        optimizer_lbfgs.Optimize(PSF_0, [toy.Jx, toy.Jy, toy.Jxy], 3)

    PSF_1 = toy()

    save_data = {
        'config': merged_config,
        'F':   to_store(toy.F),
        'b':   to_store(toy.b),
        'dx':  to_store(toy.dx),
        'dy':  to_store(toy.dy),
        'r0':  to_store(toy.r0),
        'amp':  to_store(toy.amp),
        'beta':  to_store(toy.beta),
        'alpha':  to_store(toy.alpha),
        'theta':  to_store(toy.theta),
        'ratio':  to_store(toy.ratio),
        'bg':  to_store(toy.bg),
        'Jx':  to_store(toy.Jx),
        'Jy':  to_store(toy.Jy),
        'Jxy': to_store(toy.Jxy),
        'SR data': SR(PSF_0, PSF_DL),
        'SR fit':  SR(PSF_1, PSF_DL),
        'FWHM fit':  gauss_fitter(PSF_0), 
        'FWHM data': gauss_fitter(PSF_1),
        'Img. data': to_store(PSF_0*pdims(norms,2)),
        'Img. fit':  to_store(PSF_1*pdims(norms,2)),
        'loss': loss_fn(PSF_1, PSF_0).item()
    }
    return save_data


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

#%%
# from tools.utils import SR, plot_radial_profiles, draw_PSF_stack

# print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))

# draw_PSF_stack(PSF_0, PSF_1)

# PSF_refs   = [ x for x in torch.split(PSF_0[:,0,...].squeeze().cpu(), 1, dim=0) ]
# PSF_estims = [ x for x in torch.split(PSF_1[:,0,...].squeeze().cpu(), 1, dim=0) ]

# plot_radial_profiles(PSF_refs, PSF_estims, 'TipToy', title='IRDIS PSF', dpi=200)
# # # for i in range(PSF_0.shape[0]): plot_radial_profile(PSF_0[i,0,:,:], PSF_1[i,0,:,:], 'TipToy', title='IRDIS PSF', dpi=200)
