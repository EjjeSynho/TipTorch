#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas
from data_processing.SPHERE_preproc_utils import process_mask
from torchmin import minimize

from project_globals import device

#%% Initialize data sample
with open('../data/samples/IRDIS_sample_data.pkl', 'rb') as handle:
    sample_data = pickle.load(handle)

config_file = sample_data['config']
PSF_data    = sample_data['PSF']
PSF_mask    = sample_data['mask']
PSF_mask    = process_mask(PSF_mask)


#%% Initialize model
from PSF_models.TipTorch import TipTorch

# Configure which error terms PSDs to include in the PSF model
PSD_include = {
    'fitting':         True,  # fitting error
    'WFS noise':       True,  # wavefront sensor noise
    'spatio-temporal': True,  # spatio-temporal errors
    'aliasing':        True,  # aliasing error
    'chromatism':      True,  # chromatic effects
    'diff. refract':   True,  # differential refraction error
    'Moffat':          False  # Moffat "absorber" PSD. Switched off for this example
}

# Initialize the TipTorch PSF model with the loaded configuration
model = TipTorch(
    AO_config    = config_file, # configuration parameters dictionary
    AO_type      = 'SCAO',      # selected AO mode
    pupil        = None,        # using default pupil (and apodizer) defined in config
    PSD_include  = PSD_include, # which error terms to include
    norm_regime  = 'sum',       # normalization to sum = 1 over the PSF
    device       = device,      # device to run computations on
    oversampling = 1            # oversampling factor
)

# In float regime, model is faster and only marginally less accurate, so recommended
model.to_float()

#%% Manage optimizable static phase
# For practical reason, static phase in the pupil plane is managed externaly and not included in the model.
# However, this can be done by adding a new input to the model. For example, we can add Zernike-driven static
# aberrations map. In addition to this, we can include quasi-static modes associated with Low Wind Effect (LWE).

from tools.static_phase import ZernikeLO, LWE_basis


LWE_static_phase = LWE_basis(model=model)

N_modes = 300
LO_basis = ZernikeLO(model=model, N_modes=300, ignore_pupil=True)



#%%

PSF_pred = model()

draw_PSF_stack(PSF_data*PSF_mask, PSF_pred*PSF_mask, average=True, min_val=1e-5, crop=80, scale='log')


#%%
from data_processing.normalizers import Uniform
from managers.input_manager import InputsManager


inputs_manager = InputsManager()

inputs_manager.add('r0',  model.r0,                 Uniform(a=0.05,  b=0.5))
inputs_manager.add('F',   torch.tensor([[1.0,]*2]), Uniform(a=0.0,   b=1.0))
inputs_manager.add('dx',  torch.tensor([[0.0,]*2]), Uniform(a=-1,    b=1))
inputs_manager.add('dy',  torch.tensor([[0.0,]*2]), Uniform(a=-1,    b=1))
inputs_manager.add('bg',  torch.tensor([[0.0,]*2]), Uniform(a=-5e-6, b=5e-6))
inputs_manager.add('dn',  torch.tensor([0.0]),      Uniform(a=-0.02, b=0.02))
inputs_manager.add('Jx',  torch.tensor([[7.5]]),    Uniform(a=0,     b=40))
inputs_manager.add('Jy',  torch.tensor([[7.5]]),    Uniform(a=0,     b=40))
inputs_manager.add('Jxy', torch.tensor([[18]]),     Uniform(a=-180,  b=180))

inputs_manager.add('basis_coefs', torch.zeros([1,12]),    Uniform(a=-20,   b=20))
inputs_manager.add('LO_coefs', torch.zeros([1, N_modes]), Uniform(a=-10,   b=10))

inputs_manager.to_float()
inputs_manager.to(device)

print(inputs_manager)


#%%
inputs_manager.set_optimizable('LO_coefs', False)

def func(x_):
    model_inp = inputs_manager.unstack(x_)
    if 'basis_coefs' in model_inp:
        return model(model_inp, None, lambda: LWE_static_phase(model_inp['basis_coefs']))
    else:
        return model(model_inp)
    
img_loss = lambda x: ( (func(x)-PSF_data) * PSF_mask ).flatten().abs().sum()

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

LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, LWE_static_phase.modal_basis)
PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, model.pupil)
PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, model.pupil)

inputs_manager['basis_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, LWE_static_phase.modal_basis, model.pupil)
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
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

#%%
inputs_manager = inputs_manager_new.copy()

#%%
def func_LO(x_):
    model_inp = inputs_manager.unstack(x_)
    
    if use_Zernike:  
        phase_func = lambda: \
            LWE_static_phase(inputs_manager['basis_coefs']) * \
            LO_basis(inputs_manager['LO_coefs'])
    else:
        phase_func = lambda: \
            LWE_static_phase(inputs_manager['basis_coefs']) * \
            LO_basis(inputs_manager['LO_coefs'].view(1, LO_map_size, LO_map_size))

    return model(model_inp, None, phase_func)

if not use_Zernike:  
    grad_loss_fn = GradientLoss(p=1, reduction='mean')

img_loss_LO = lambda x: ( (func_LO(x)-PSF_data) * PSF_mask ).flatten().abs().sum()

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
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)


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
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

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

hessian_mat = hessian(lambda x_: loss_fn_all(func(x_), PSF_data).log(), x0).squeeze()
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
draw_PSF_stack(PSF_data, PSF_1, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_data), destack(PSF_1), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)

#%%
with torch.no_grad():
    LWE_phase = torch.einsum('mn,nwh->mwh', LWE_static_phase.coefs, LWE_static_phase.modal_basis).cpu().numpy()[0,...]
    plt.imshow(LWE_phase)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('LWE OPD, [nm] RMS')

#%%
WFE = torch.mean(model.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = model.D/4 * 1e9*(model.Jx+model.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / model.wvl.flatten()[0]

S_0 = SR(PSF_data, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')
