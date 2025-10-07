#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.plotting import plot_radial_PSF_profiles, SR, draw_PSF_stack, rad2mas, mask_circle, GradientLoss, OptimizableLO, ZernikeLO
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds, process_mask
from managers.config_manager import GetSPHEREonsky
from project_settings import SPHERE_DATA_FOLDER, device
from torchmin import minimize


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

subset_df = psf_df[psf_df['Corrupted'] == False]
# subset_df = subset_df[subset_df['Low quality'] == False]
# subset_df = subset_df[subset_df['Medium quality'] == False]
# subset_df = subset_df[subset_df['LWE'] == True]
# subset_df = subset_df[subset_df['mag R'] < 7]
# subset_df = subset_df[subset_df['Num. DITs'] < 50]
# subset_df = subset_df[subset_df['Class A'] == True]
# subset_df = subset_df[np.isfinite(subset_df['λ left (nm)']) < 1700]
# subset_df = subset_df[subset_df['Δλ left (nm)'] < 80]
subset_df = subset_df[subset_df['High quality'] == True]
subset_df = subset_df[subset_df['High SNR'] == True]
subset_df = subset_df[subset_df['LWE'] == True]
# subset_df = subset_df[subset_df['Central hole'] == False]

#%%

windy_entries = [x for x in psf_df.columns.values if 'Wind' in x]


'Wind direction (header)',
'Wind direction (MASSDIMM)',
'Wind direction (200 mbar)',
'Wind speed (header)',
'Wind speed (SPARTA)',
'Wind speed (MASSDIMM)',
'Wind speed (200 mbar)'


#%%
sample_id = 768
# sample_id = 2649
# sample_id = 438
# sample_id = 97
# sample_id = 173 # -- interesting wings
# sample_id = 399 # -- interesting wings
# sample_id = 1248 # -- interesting rotated-LWE PSF
# sample_id = 2219 # -- interesting LWE-cross
# sample_id = 1778 # -- eplicit wings


PSF_data, data_sample, config_file = SPHERE_preprocess(
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


config_file['sensor_science']['FieldOfView'] = PSF_0.shape[-1]

config_file['NumberSources'] = config_file['NumberSources'].int().item()
config_file['DM']['DmHeights'] = torch.tensor(config_file['DM']['DmHeights'], device=device)
config_file['sources_HO']['Wavelength'] = config_file['sources_HO']['Wavelength']
config_file['sources_HO']['Height'] = torch.inf


# if psf_df.loc[sample_id]['Nph WFS'] < 10:
# PSF_mask   = PSF_mask * 0 + 1
PSF_mask = process_mask(PSF_mask)
# LWE_flag   = psf_df.loc[sample_id]['LWE']
LWE_flag = True
wings_flag = True

#psf_df.loc[sample_id]['Wings']
# wings_flag = False

if psf_df.loc[sample_id]['Central hole'] == True:
    circ_mask = 1-mask_circle(PSF_0.shape[-1], 3, centered=True)
    PSF_mask *= torch.tensor(circ_mask[None, None, ...]).to(device)

'''
PSF_data = {
    'PSF': PSF_0,
    'mask': PSF_mask,
    'config': config_file,
    'ob.file': data_sample[0]['observation']['date']
}

import pickle
with open('../data/samples/IRDIS_sample_data.pkl', 'wb') as f:
    pickle.dump(PSF_data, f)
'''
# plt.imshow(circ_mask * np.squeeze(PSF_0[0,0,...].cpu().numpy()), norm=LogNorm())

#%% Initialize model
# from PSF_models.TipTorch import TipTorch
# from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch
from tools.utils import LWE_basis

# tiptorch = TipTorch(merged_config, None, device, oversampling=1)
# _ = tiptorch()

PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          False
}
model = TipTorch(config_file, 'SCAO', None, PSD_include, 'sum', device, oversampling=1)
model.to_float()

basis = LWE_basis(model)

PSF_1  = model()
# PSF_DL = tiptorch.DLPSF()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, average=True, min_val=1e-5, crop=80, scale='log')

#%%
A = PSF_0[0,0,...].abs().log10().cpu().numpy()
B = PSF_0[0,1,...].abs().log10().cpu().numpy()

A = A - np.median(A)
B = B - np.median(B)

A = A / A.max()
B = B / B.max()

C = np.stack([A, B, np.zeros_like(A)], axis=-1)

plt.imshow(C)
plt.axis('off')

#%% PSF fitting (no early-stopping)
from tools.normalizers import Uniform
from tools.utils import OptimizableLO, ZernikeLO
from managers.input_manager import InputsManager


use_Zernike = False

if use_Zernike:
    N_modes = 300
    LO_basis = ZernikeLO(model, N_modes, device)
else:
    LO_map_size = 31
    LO_basis = OptimizableLO(model, ignore_pupil=True)

inputs_manager = InputsManager()

norm_F        = Uniform(a=0.0,   b=1.0)
norm_bg       = Uniform(a=-5e-6, b=5e-6)
norm_r0       = Uniform(a=0.05,  b=0.5)
norm_dxy      = Uniform(a=-1,    b=1)
norm_J        = Uniform(a=0,     b=40)
norm_Jxy      = Uniform(a=-180,  b=180)
norm_LWE      = Uniform(a=-20,   b=20)
norm_dn       = Uniform(a=-0.02, b=0.02)
norm_wind_spd = Uniform(a=0,     b=20)
norm_wind_dir = Uniform(a=0,     b=360)
norm_LO       = Uniform(a=-10,   b=10)

# For old
# norm_J       = Uniform(a=0,     b=30)
# norm_Jxy     = Uniform(a=0,     b=50)

# Add parameters to InputsManager with their normalizers
inputs_manager.add('r0',  model.r0,                 norm_r0)
inputs_manager.add('F',   torch.tensor([[1.0,]*2]), norm_F)
inputs_manager.add('dx',  torch.tensor([[0.0,]*2]), norm_dxy)
inputs_manager.add('dy',  torch.tensor([[0.0,]*2]), norm_dxy)
inputs_manager.add('bg',  torch.tensor([[0.0,]*2]), norm_bg)
inputs_manager.add('dn',  torch.tensor([0.0]),      norm_dn)
inputs_manager.add('Jx',  torch.tensor([[7.5]]),    norm_J)
inputs_manager.add('Jy',  torch.tensor([[7.5]]),    norm_J)
inputs_manager.add('Jxy', torch.tensor([[18]]),     norm_Jxy)

if wings_flag:
    # inputs_manager.add('wind_speed', model.wind_speed, norm_wind_spd)
    inputs_manager.add('wind_dir', model.wind_dir,  norm_wind_dir)
if LWE_flag:
    inputs_manager.add('basis_coefs', torch.zeros([1,12]), norm_LWE)

if use_Zernike:
    if N_modes is not None:
        inputs_manager.add('LO_coefs', torch.zeros([1, N_modes]), norm_LO)
else:
    if LO_map_size is not None:
        inputs_manager.add('LO_coefs', torch.zeros([1, LO_map_size**2]), norm_LO)

inputs_manager.to_float()
inputs_manager.to(device)

if use_Zernike:
    if N_modes is not None:
        inputs_manager.set_optimizable('LO_coefs', False)
else:
    if LO_map_size is not None:
        inputs_manager.set_optimizable('LO_coefs', False)


#%%
def func(x_):
    model_inp = inputs_manager.unstack(x_)
    if 'basis_coefs' in model_inp:
        return model(model_inp, None, lambda: basis(model_inp['basis_coefs']))
    else:
        return model(model_inp)
    
img_loss = lambda x: ( (func(x)-PSF_0) * PSF_mask ).flatten().abs().sum()

if LWE_flag:
    A = 50.0
    patterns = [
        [0,0,0,0,  0,-1,1,0,  1,0,0,-1], # pattern_pos
        [0,0,0,0,  0,1,-1,0, -1,0,0, 1], # pattern_neg
        [0,0,0,0,  0,-1,1,0, -1,0,0, 1], # pattern_1
        [0,0,0,0,  0,1,-1,0,  1,0,0,-1], # pattern_2
        [0,0,0,0,  1,0,0,-1,  0,1,-1,0], # pattern_3
        [0,0,0,0,  -1,0,0,1,  0,-1,1,0], # pattern_4
        [-1,1,1,-1,  0,0,0,0,  0,0,0,0], # pattern_piston_horiz
        [1,-1,-1,1,  0,0,0,0,  0,0,0,0]  # pattern_piston_vert
    ]
    patterns = [torch.tensor([p]).to(device).float() * A for p in patterns]
    
    gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
    Gauss_err = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, A/2)).flatten().abs().sum()

    LWE_regularizer = lambda c: sum(Gauss_err(pattern, c) for pattern in patterns)

    def loss_fn(x_):
        loss = img_loss(x_)
        coefs_ = inputs_manager['basis_coefs']
        loss += LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4
        return loss

else:
    def loss_fn(x_):
        return img_loss(x_)


x0 = inputs_manager.stack()

# Add small random perturbation to initial guess
# x0 += torch.randn_like(x0) * 1e-1

#%%
result = minimize(loss_fn, x0, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x0 = result.x

PSF_1 = func(x0)

#%%
from tools.utils import BuildPTTBasis, decompose_WF, project_WF

LWE_coefs = inputs_manager['basis_coefs']
PTT_basis = BuildPTTBasis(model.pupil.cpu().numpy(), True).to(device).float()

TT_max = PTT_basis.abs()[1,...].max().item()
pixel_shift = lambda coef: 2 * TT_max * rad2mas * 1e-9 * coef / model.psInMas / model.D  / (1-7/model.pupil.shape[-1])

LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, basis.modal_basis)
PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, model.pupil)
PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, model.pupil)

inputs_manager['basis_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, basis.modal_basis, model.pupil)
inputs_manager['dx'] -= pixel_shift(PTT_coefs[:, 2])
inputs_manager['dy'] -= pixel_shift(PTT_coefs[:, 1])
x0_new = inputs_manager.stack()
PSF_1 = func(x0_new)

inputs_manager_new = inputs_manager.copy()

#%
plt.imshow((LWE_OPD-PPT_OPD)[0,...].cpu().numpy())#, vmin=-0.05, vmax=0.05)
# plt.imshow((LWE_OPD)[0,...].cpu().numpy())#, vmin=-0.05, vmax=0.05)
# plt.imshow(PPT_OPD[0,...].cpu().numpy())#, vmin=-0.05, vmax=0.05)
plt.colorbar()
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

#%%
inputs_manager = inputs_manager_new.copy()

#%%
def func_LO(x_):
    model_inp = inputs_manager.unstack(x_)
    
    if use_Zernike:  
        phase_func = lambda: \
            basis(inputs_manager['basis_coefs']) * \
            LO_basis(inputs_manager['LO_coefs'])
    else:
        phase_func = lambda: \
            basis(inputs_manager['basis_coefs']) * \
            LO_basis(inputs_manager['LO_coefs'].view(1, LO_map_size, LO_map_size))

    return model(model_inp, None, phase_func)

if not use_Zernike:  
    grad_loss_fn = GradientLoss(p=1, reduction='mean')

img_loss_LO = lambda x: ( (func_LO(x)-PSF_0) * PSF_mask ).flatten().abs().sum()

def loss_fn_LO(x_):
    loss = img_loss_LO(x_)
    if use_Zernike:  
        grad_loss = 0.0
    else:  
        grad_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_map_size, LO_map_size)) * 1e-4
    l2_loss = inputs_manager['LO_coefs'].pow(2).sum()*5e-9
    return loss+ l2_loss + grad_loss 


if use_Zernike:  
    inputs_manager.set_optimizable(['LO_coefs', 'basis_coefs'], True)
    inputs_manager.set_optimizable([
        'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir'
    ], False)
else:
    inputs_manager.set_optimizable(['LO_coefs'], True)
    inputs_manager.set_optimizable([
        'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'basis_coefs'
    ], False)


x1 = inputs_manager.stack()

#%%
result = minimize(loss_fn_LO, x1, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x1 = result.x

PSF_1 = func_LO(x1)
#%%
if use_Zernike:  
    OPD_map = LO_basis.compute_OPD(inputs_manager['LO_coefs'])[0].detach().cpu().numpy() * 1e9
else:  
    OPD_map = inputs_manager['LO_coefs'].view(1, LO_map_size, LO_map_size)[0].detach().cpu().numpy()

plt.imshow(OPD_map) #, cmap='RdBu')
plt.colorbar()
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)


#%%
inputs_manager.set_optimizable([
    'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy'
], True)
inputs_manager.set_optimizable(['LO_coefs', 'wind_dir', 'basis_coefs'], False)

x2 = inputs_manager.stack()

#%%
result = minimize(img_loss_LO, x2, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x2 = result.x

PSF_1 = func_LO(x2)

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

#%%
LWE_map = LWE_OPD-PPT_OPD

if use_Zernike:
    fitted_map = LO_basis.compute_OPD(inputs_manager['LO_coefs'])[0] * 1e9
else:
    fitted_map = LO_basis.interp_upscale(inputs_manager['LO_coefs'].view(1, LO_map_size, LO_map_size))

summa = (model.pupil*(fitted_map+LWE_map)).cpu().numpy().squeeze()

plt.imshow(summa) #, cmap='RdBu')
plt.colorbar()

#%%
def GetNewPhotons():
    WFS_noise_var = model.dn + model.NoiseVariance(model.r0.abs())

    N_ph_0 = model.WFS_Nph.clone()

    def func_Nph(x):
        model.WFS_Nph = x
        var = model.NoiseVariance(model.r0.abs())
        return (WFS_noise_var-var).flatten().abs().sum()

    result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
    model.WFS_Nph = N_ph_0.clone()

    return result_photons.x

Nph_new = GetNewPhotons()

print(model.WFS_Nph.item(), Nph_new.item())


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
plt.ylabel('Variance of the estimated parameters')
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
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))
draw_PSF_stack(PSF_0, PSF_1, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_PSF_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)

#%%
with torch.no_grad():
    LWE_phase = torch.einsum('mn,nwh->mwh', basis.coefs, basis.modal_basis).cpu().numpy()[0,...]
    plt.imshow(LWE_phase)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('LWE OPD, [nm] RMS')

#%%
WFE = torch.mean(model.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = model.D/4 * 1e9*(model.Jx+model.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / model.wvl.flatten()[0]

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')
