#%%
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tools.plotting import plot_radial_PSF_profiles, SR, draw_PSF_stack, rad2mas, cropper
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_STD_dataset_utils import SPHERE_preprocess, SamplesByIds
from managers.config_manager import GetSPHEREonsky
from project_settings import SPHERE_DATA_FOLDER, device
from torchmin import minimize

import dynesty
from dynesty.pool import Pool
from dynesty import plotting as dyplot

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
# psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
# psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]
#%
subset_df = psf_df[psf_df['High quality'] == True]
subset_df = subset_df[subset_df['High SNR'] == True]
# subset_df = subset_df[subset_df['LWE'] == False]
subset_df = subset_df[subset_df['LWE'] == True]
subset_df = subset_df[subset_df['Central hole'] == False]

#%%
from matplotlib.colors import LogNorm
from data_processing.SPHERE_STD_dataset_utils import LoadSPHEREsampleByID

# 448, 452, 465, 552, 554, 556, 564, 576, 578, 580, 581, 578, 576, 992
# 1209 # high noise
# 1452 # high noise
# 1660 # LWE
# 456
# 465
# 1393 #50 DITs
# 1408
# 898

# Too high blur
# 2423
# 3365

# [114, 549, 811, 816, 1176, 1192, 1304, 1573, 2146,
# 2726, 3121, 3613, 3651, 3706, 3875, 3882, 3886, 3906, 3909, 4002, 405]

sample_id = 1374 #2112 #1921 #3909

# LWE_flag = psf_df.loc[sample_id]['LWE']
#%
PSF_data, _, merged_config = SPHERE_preprocess(
    sample_ids    = [sample_id],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

PSF_0    = PSF_data[0]['PSF (mean)'].unsqueeze(0)
PSF_var  = PSF_data[0]['PSF (var)'].unsqueeze(0)
PSF_mask = PSF_data[0]['mask (mean)'].unsqueeze(0)
norms    = PSF_data[0]['norm (mean)']
del PSF_data

# if psf_df.loc[sample_id]['Nph WFS'] < 10:
PSF_mask   = PSF_mask * 0 + 1
# LWE_flag   = psf_df.loc[sample_id]['LWE']
LWE_flag = True
wings_flag = psf_df.loc[sample_id]['Wings']


#%% Initialize model
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

toy = TipTorch(merged_config, None, device)

_ = toy()

# print(toy.WFS_Nph.item())

# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']
# toy.optimizables = []

# _ = toy({ 'bg': bg.unsqueeze(0).to(device) })
# _ = toy({ 'dx': torch.tensor([[0.0, 0.0]]).to(device) })
# _ = toy({ 'dy': torch.tensor([[0.0, 0.0]]).to(device) })

PSF_1 = toy()
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()


draw_PSF_stack(PSF_0*PSF_mask, PSF_1, average=True, crop=80, scale='log')
# mask_in  = toy.mask_rim_in.unsqueeze(1).float()
# mask_out = toy.mask_rim_out.unsqueeze(1).float()


#%% PSF fitting (no early-stopping)
from tools.utils import LWE_basis
from tools.normalizers import TransformSequence, Uniform, InputsTransformer

basis = LWE_basis(toy)

norm_F   = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0)  ])
norm_bg  = TransformSequence(transforms=[ Uniform(a=-1e-6, b=1e-6) ])
norm_r0  = TransformSequence(transforms=[ Uniform(a=0,     b=0.5)  ])
norm_dxy = TransformSequence(transforms=[ Uniform(a=-1,    b=1)    ])
norm_J   = TransformSequence(transforms=[ Uniform(a=0,     b=30)   ])
norm_Jxy = TransformSequence(transforms=[ Uniform(a=0,     b=50)   ])
norm_LWE = TransformSequence(transforms=[ Uniform(a=-20,   b=20)   ])
norm_dn  = TransformSequence(transforms=[ Uniform(a=-0.02, b=0.02) ])
norm_wind_spd = TransformSequence(transforms=[ Uniform(a=0, b=20)  ])
norm_wind_dir = TransformSequence(transforms=[ Uniform(a=0, b=360) ])


transformer = InputsTransformer({
    'r0':  norm_r0,
    'F':   norm_F,
    'dx':  norm_dxy,
    'dy':  norm_dxy,
    'bg':  norm_bg,
    'dn':  norm_dn,
    'Jx':  norm_J,
    'Jy':  norm_J,
    'Jxy': norm_Jxy,
    'wind_speed': norm_wind_spd,
    'wind_dir':   norm_wind_dir,
    'basis_coefs': norm_LWE
})


inp_dict = {}

# Loop through the class attributes
for attr in ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']:
    inp_dict[attr] = getattr(toy, attr)

if wings_flag:
    # inp_dict['wind_speed'] = toy.wind_speed
    inp_dict['wind_dir']   = toy.wind_dir
    
if LWE_flag:
    inp_dict['basis_coefs'] = basis.coefs

_ = transformer.stack(inp_dict) # to create index mapping

#%%
x0 = [norm_r0.forward(toy.r0).item(),
      1.0, 1.0,
      0.0, 0.0,
      0.0, 0.0,
      0.0, 0.0,
      0.0,
      0.5,
      0.5,
      0.1]

if wings_flag:
    x0 = x0 + [
        norm_wind_dir.forward(toy.wind_dir).item(),
        # norm_wind_spd.forward(toy.wind_speed).item()
    ]   

if LWE_flag:
    x0 = x0 + [0,]*4 + [0,]*8

x0 = torch.tensor(x0).float().to(device).unsqueeze(0)

def func(x_):
    x_torch = transformer.unstack(x_)
    if 'basis_coefs' in x_torch:
        return toy(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))
    else:
        return toy(x_torch)

if LWE_flag:
    gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))

    pattern_pos = torch.tensor([[0,0,0,0,  0,-1,1,0,  1,0,0,-1]]).to(device).float() * 50.0
    pattern_neg = torch.tensor([[0,0,0,0,  0,1,-1,0, -1,0,0, 1]]).to(device).float() * 50.0

    def loss_fn(x_):
        img_punish = ( (func(x_)-PSF_0) * PSF_mask ).flatten().abs().sum()
        LWE_punish = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, 40)).flatten().abs().sum()
        coefs_ = transformer.unstack(x_)['basis_coefs']
        loss = img_punish + LWE_punish(pattern_pos, coefs_) + LWE_punish(pattern_neg, coefs_)
        return loss
else:
    
    def loss_fn(x_):
        loss = (func(x_)-PSF_0)*PSF_mask
        return loss.flatten().abs().sum()

result = minimize(loss_fn, x0, method='bfgs', disp=2)

x0 = result.x
# x0_buf = x0.clone()

#%%
def GetNewPhotons():
    WFS_noise_var = toy.dn + toy.NoiseVariance(toy.r0.abs())

    N_ph_0 = toy.WFS_Nph.clone()

    def func_Nph(x):
        toy.WFS_Nph = x
        var = toy.NoiseVariance(toy.r0.abs())
        return (WFS_noise_var-var).flatten().abs().sum()

    result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
    toy.WFS_Nph = N_ph_0.clone()

    return result_photons.x

Nph_new = GetNewPhotons()

print(toy.WFS_Nph.item(), Nph_new.item())

#%%
from tools.utils import BuildPTTBasis, decompose_WF, project_WF, calc_WFE

LWE_coefs = transformer.unstack(x0)['basis_coefs'].clone()
PTT_basis = BuildPTTBasis(toy.pupil.cpu().numpy(), True).to(device).float()

TT_max = PTT_basis.abs()[1,...].max().item()
pixel_shift = lambda coef: 4 * TT_max * rad2mas / toy.psInMas / toy.D * 1e-9 * coef

LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, basis.modal_basis)
PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, toy.pupil)
PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, toy.pupil)

#%
x0_new = transformer.unstack(x0)
x0_new['basis_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, basis.modal_basis, toy.pupil) 
x0_new['dx'] -= pixel_shift(PTT_coefs[:, 2])
x0_new['dy'] -= pixel_shift(PTT_coefs[:, 1])
x0 = transformer.stack(x0_new)

#%
# x0 = x0_buf.clone()

#%
with torch.no_grad():
    PSF_1 = func(x0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_radial_PSF_profiles( PSF_0[:,0,...].cpu().numpy(), PSF_1[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
    plot_radial_PSF_profiles( PSF_0[:,1,...].cpu().numpy(), PSF_1[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
    plt.show()
  
    draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)#, scale=None)

#%%

'''
for attr in ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']:
    inp_dict[attr] = getattr(toy, attr)
    # inp_dict['wind_speed'] = toy.wind_speed
    inp_dict['wind_dir']   = toy.wind_dir
    inp_dict['basis_coefs'] = basis.coefs
'''

def func_dynesty(x_):
    x_torch = transformer.unstack(torch.tensor(x_).float().to(device).unsqueeze(0))
    if 'basis_coefs' in x_torch:
        return toy(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))
    else:
        return toy(x_torch)

def likelihood(x):
    return (func_dynesty(x)-PSF_0).abs().sum().cpu().numpy()

sample_range = 1.0

def prior_transform(u):
    return sample_range*u

print("Starting nested sampling...")

# toy = toy.to(torch.device('cpu'))

n_dim = x0.shape[1]

sampler = dynesty.NestedSampler(
    loglikelihood   = likelihood,
    prior_transform = prior_transform,
    ndim            = n_dim,
    nlive           = 2000
)

sampler.run_nested()
results = sampler.results

#%%

fg, ax = dyplot.cornerpoints(
    results,
    cmap='plasma',
    kde=False
)

#%%

buf = np.copy(results.samples)

sample = np.zeros_like(results.samples)

for i in range(0, sample.shape[1]):
    sample[:,i] = transformer.unstack(torch.tensor(sample[:,i]).float().unsqueeze(0).to(device)).cpu().numpy()



#%%

fg, ax = dyplot.cornerplot(results, color='blue',
                           truth_color='black', show_titles=True,
                           max_n_ticks=3, quantiles=None)

#%%
from torch.autograd.functional import hessian, jacobian

hessian_mat = hessian(lambda x_: loss_fn_all(func(x_), PSF_0).log(), x0).squeeze()
hessian_mat_ = hessian_mat.clone()
hessian_mat_[1:,0] = hessian_mat_[0,1:]
hessian_mat_[0,0] = 0.0
hessian_mat_[hessian_mat.abs() < 1e-11] = 1e-11


#%%
# plt.figure(dpi=200)
plt.imshow(hessian_mat_.cpu().detach().numpy())

# Change labels of x ticks
xticks = np.arange(0, hessian_mat_.shape[-1])
xticklabels = \
    list(decomposed_variables.keys())[:-1] + \
    [f"Piston {i}" for i in range(2, 5)] + \
    [f"Tip {i}" for i in range(1, 5)] + \
    [f"Tilt {i}" for i in range(1, 5)]

# Insert value after 'bg'
xticklabels.insert(xticklabels.index('bg')+1, 'bg R')
xticklabels.insert(xticklabels.index('F')+1,  'F R')

xticklabels[xticklabels.index('bg')] = 'bg L'
xticklabels[xticklabels.index('F')]  = 'F L'
plt.xticks(xticks, xticklabels, rotation=45, ha="right")
plt.yticks(xticks, xticklabels, rotation=45, ha="right")

#%%
hessian_inv = torch.inverse(hessian_mat_)
variance_estimates = torch.diag(hessian_inv).abs().cpu().numpy()

plt.bar(np.arange(0, len(variance_estimates)), variance_estimates)
plt.xticks(np.arange(0, len(variance_estimates)), xticklabels, rotation=45, ha="right")
plt.yscale('log')
plt.ylabel('Variance of the estimated paraemters')
plt.title('Fisher information per parameter')

#%%
L_complex, V_complex = torch.linalg.eig(hessian_mat_)

L = L_complex.real.cpu().numpy()
V = V_complex.real.cpu().numpy()
V = V / np.linalg.norm(V, axis=0)

for eigenvalue_id in range(0, L.shape[0]):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].set_title('Eigenvalues')
    ax[0].bar(np.arange(0, L.shape[0]), np.abs(L))
    ax[0].bar(eigenvalue_id, np.abs(L[eigenvalue_id]), color='red')

    ax[1].set_title(f'Eigenvector #{eigenvalue_id}')
    ax[1].bar(np.arange(0, V.shape[0]), np.abs(V[:, eigenvalue_id]), color='green')
    ax[1].set_xticks(xticks, xticklabels, rotation=45, ha="right")

    # plt.show()
    numeration = str(eigenvalue_id)
    if eigenvalue_id < 10:
        numeration = '0' + numeration
    plt.savefig(f'C:/Users/akuznets/Desktop/buf/couplings/eigen_{numeration}.png', dpi=200)


#%%
