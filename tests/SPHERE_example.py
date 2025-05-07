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
from torchmin import minimize

from project_settings import device

#%% Initialize data sample
with open('../data/samples/IRDIS_sample_data.pkl', 'rb') as handle:
    sample_data = pickle.load(handle)

config   = sample_data['config']
PSF_data = sample_data['PSF']
PSF_mask = sample_data['mask']


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
    AO_config    = config, # configuration parameters dictionary
    AO_type      = 'SCAO',      # selected AO mode
    pupil        = None,        # using default pupil (and apodizer) defined in config
    PSD_include  = PSD_include, # which error terms to include
    norm_regime  = 'sum',       # normalize PSFs to sum = 1 over the PSF
    device       = device,      # device to run computations on (CPU or GPU)
    oversampling = 1            # oversampling factor
)

# In float regime, the model is faster and only marginally less accurate, so recommended
model.to_float()

# Running model with the parameters defined in the config. Values unspecifiiied in the config are just set to default values
PSF_test = model()

#%% Manage optimizable static phase
# For practical reason, static phase in the pupil plane is managed externaly and not included in the model.
# However, this can be done by adding a new input to the model. For example, we can add Zernike-driven static
# aberrations map. In addition to this, we can include quasi-static modes associated with Low Wind Effect (LWE).

from tools.static_phase import ZernikeBasis, LWEBasis

# Pupil iis managed inside the PSF model, so we need to ignore it in the basis computation
LWE_basis = LWEBasis(model=model, ignore_pupil=True)
Z_basis = ZernikeBasis(model=model, N_modes=300, ignore_pupil=True)

# Compute static phase from the modal coefficients
def compute_static_phase(input_dict):
    return Z_basis(input_dict['Z_coefs']) * LWE_basis(input_dict['LWE_coefs']) * model.pupil * model.apodizer

#%% Managing model inputs
# In this example, pytorch-minimize is used to optimize the model inputs. The inputs are managed by the InputsManager class.
# InputsManager allows to normalize model inputs that mau take ildly different ranges of values. In addition, it allows to
# easily stack inputs dictionary to a single vector/matrix for optimization and unstack it back to a dictionary.

from data_processing.normalizers import Uniform
from managers.input_manager import InputsManager


model_inputs = InputsManager()
'''
Note, that it is possible to add parameters with arbitrary shapes and names into the manager.
The only ones which named the same as internakl variables of the model will be used in the model.
Other parameters then must be handled manually by the user, like Z_coefs and LWE_coefs in this example.
'''
# The dimensionality of inputs is very important, since PSF model doesn't do any checking itself
model_inputs.add('r0',  model.r0,                 Uniform(a=0.05,  b=0.5))
model_inputs.add('F',   torch.tensor([[1.0,]*2]), Uniform(a=0.0,   b=1.0))
model_inputs.add('dx',  torch.tensor([[0.0,]*2]), Uniform(a=-1,    b=1))
model_inputs.add('dy',  torch.tensor([[0.0,]*2]), Uniform(a=-1,    b=1))
model_inputs.add('bg',  torch.tensor([[0.0,]*2]), Uniform(a=-5e-6, b=5e-6))
model_inputs.add('dn',  torch.tensor([0.0]),      Uniform(a=-0.02, b=0.02))
model_inputs.add('Jx',  torch.tensor([[7.5]]),    Uniform(a=0,     b=40))
model_inputs.add('Jy',  torch.tensor([[7.5]]),    Uniform(a=0,     b=40))
model_inputs.add('Jxy', torch.tensor([[18]]),     Uniform(a=-180,  b=180))

model_inputs.add('LWE_coefs', torch.zeros([1,12]),    Uniform(a=-20,   b=20))
model_inputs.add('Z_coefs',   torch.zeros([1, Z_basis.N_modes]), Uniform(a=-10,   b=10))

model_inputs.to_float()
model_inputs.to(device)

print(model_inputs)

#%%
def simulate(x_):
    '''
    Simulates the PSF given the inputs stacked into a single vector. Used in the optimization process.
    Given this vector, it is correct to say that model is fully defined by:
     -- the internal values pre-set with config during the initialization,
     -- the input dictionary which overloads some of these parameters every time this function is called,
     -- the external phase generator which is also called every time this function is called.
    '''
    # Note, that every call to model_inputs.unstack() will update the internal state of model_inputs
    # Switching the update off helps to leave the internal state of model_inputs intact
    input_dict = model_inputs.unstack(x_, update=True)
    # PSD = None means that the PSD will be computed inside the model and not provided from outside.
    return model(x=input_dict, PSD=None, phase_generator=lambda: compute_static_phase(input_dict))

x0 = model_inputs.stack()

# Direct prediction without any calibration or fitting, quite inaccurate
PSF_pred = simulate(x0)

draw_PSF_stack(PSF_data*PSF_mask, PSF_pred*PSF_mask, average=True, min_val=1e-5, crop=80, scale='log')

#%%
class LWERegularizer:
    """
    The purpose of this class is to regularize the optimization of LWE coefficients to avoid overfitting
    """
    # These initial values are completely empirical
    def __init__(self, device, amplitude=50.0, gaussian_sigma_factor=2.0, gauss_penalty_weight=5.0, l2_weight=1e-4):
        self.device = device
        self.amplitude = amplitude
        self.gaussian_sigma = amplitude / gaussian_sigma_factor
        self.l2_weight = l2_weight
        self.gauss_penalty_weight = gauss_penalty_weight

        # Define patterns to avoid while optimizing because they are unlikely to appear physically and thus may mean overfitting
        pattern_templates = [
            [0,0,0,0,  0,-1,1,0,  1,0,0,-1],  # pattern_outwards
            [0,0,0,0,  0,1,-1,0, -1,0,0, 1],  # pattern_inwards
            [0,0,0,0,  0,-1,1,0, -1,0,0, 1],  # pattern_1
            [0,0,0,0,  0,1,-1,0,  1,0,0,-1],  # pattern_2
            [0,0,0,0,  1,0,0,-1,  0,1,-1,0],  # pattern_3
            [0,0,0,0,  -1,0,0,1,  0,-1,1,0],  # pattern_4
            [-1,1,1,-1,  0,0,0,0,  0,0,0,0],  # pattern_piston_horiz
            [1,-1,-1,1,  0,0,0,0,  0,0,0,0]   # pattern_piston_vert
        ]

        # Create tensor patterns from templates
        self.patterns = [torch.tensor([p]).to(device).float() * self.amplitude for p in pattern_templates]

    def gaussian_penalty(self, amplitude, x, x_0, sigma):
        # Calculate Gaussian penalty between coefficient vector and pattern template
        return amplitude * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))

    def pattern_error(self, pattern, coefficients):
        # Calculate error term for a specific pattern
        return (pattern * self.gaussian_penalty(self.gauss_penalty_weight, coefficients, pattern, self.gaussian_sigma)).flatten().abs().sum()

    def __call__(self, coefficients):
        # Calculate the full LWE regularization loss for given coefficients
        pattern_loss = sum(self.pattern_error(pattern, coefficients) for pattern in self.patterns)
        # L2 regularization
        LWE_l2_loss = (coefficients**2).mean() * self.l2_weight

        return pattern_loss + LWE_l2_loss


# Initialize the regularizer
LWE_regularizer = LWERegularizer(device)

L1_loss_custom = lambda x: ( (simulate(x)-PSF_data) * PSF_mask ).flatten().abs().sum()

def loss_fn(x_):
    # You can also update the models_inputs entries directly if needed
    return L1_loss_custom(x_) + LWE_regularizer(model_inputs['LWE_coefs']) * float(model_inputs.is_optimizable('LWE_coefs'))


#%%
# Switch off optimization of Zernike coefficients for now, so they are notstacked into the model inputs vector
model_inputs.set_optimizable('Z_coefs', False)

x0 = model_inputs.stack()
# x0 += torch.randn_like(x0) * 1e-1 # Add random perturbations to the initial guess

result = minimize(loss_fn, x0, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x0 = result.x

PSF_fitted = simulate(x0)


#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_fitted*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

#%%
model_inputs = model_inputs.copy()

#%%
def func_LO(x_):
    model_inp = model_inputs.unstack(x_)
    
    if use_Zernike:  
        phase_func = lambda: \
            LWE_basis(model_inputs['LWE_coefs']) * \
            Z_basis(model_inputs['Z_coefs'])
    else:
        phase_func = lambda: \
            LWE_basis(model_inputs['LWE_coefs']) * \
            Z_basis(model_inputs['Z_coefs'].view(1, LO_map_size, LO_map_size))

    return model(model_inp, None, phase_func)

if not use_Zernike:  
    grad_loss_fn = GradientLoss(p=1, reduction='mean')

img_loss_LO = lambda x: ( (func_LO(x)-PSF_data) * PSF_mask ).flatten().abs().sum()

def loss_fn_LO(x_):
    loss = img_loss_LO(x_)
    if use_Zernike:  
        grad_loss = 0.0
    else:  
        grad_loss = grad_loss_fn(model_inputs['Z_coefs'].view(1, 1, LO_map_size, LO_map_size)) * 1e-4
    l2_loss = model_inputs['Z_coefs'].pow(2).sum()*5e-9
    return loss+ l2_loss + grad_loss 


if use_Zernike:  
    model_inputs.set_optimizable(['Z_coefs', 'LWE_coefs'], True)
    model_inputs.set_optimizable([
        'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir'
    ], False)
else:
    model_inputs.set_optimizable(['Z_coefs'], True)
    model_inputs.set_optimizable([
        'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'LWE_coefs'
    ], False)


x1 = model_inputs.stack()

#%%
result = minimize(loss_fn_LO, x1, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x1 = result.x

PSF_fitted = func_LO(x1)
#%%
if use_Zernike:  
    OPD_map = Z_basis.compute_OPD(model_inputs['Z_coefs'])[0].detach().cpu().numpy() * 1e9
else:  
    OPD_map = model_inputs['Z_coefs'].view(1, LO_map_size, LO_map_size)[0].detach().cpu().numpy()

plt.imshow(OPD_map) #, cmap='RdBu')
plt.colorbar()
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_fitted*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)


#%%
model_inputs.set_optimizable([
    'F','r0', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy'
], True)
model_inputs.set_optimizable(['Z_coefs', 'wind_dir', 'LWE_coefs'], False)

x2 = model_inputs.stack()

#%%
result = minimize(img_loss_LO, x2, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x2 = result.x

PSF_fitted = func_LO(x2)

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,0,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_data*PSF_mask)[:,1,...].cpu().numpy(), (PSF_fitted*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_data*PSF_mask, PSF_fitted*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)

#%%
LWE_map = LWE_OPD-PPT_OPD

if use_Zernike:
    fitted_map = Z_basis.compute_OPD(model_inputs['Z_coefs'])[0] * 1e9
else:
    fitted_map = Z_basis.interp_upscale(model_inputs['Z_coefs'].view(1, LO_map_size, LO_map_size))

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

hessian_mat = hessian(lambda x_: loss_fn_all(simulate(x_), PSF_data).log(), x0).squeeze()
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
print('\nStrehl ratio: ', SR(PSF_fitted, PSF_DL))
draw_PSF_stack(PSF_data, PSF_fitted, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_data), destack(PSF_fitted), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)

#%%
with torch.no_grad():
    LWE_phase = torch.einsum('mn,nwh->mwh', LWE_basis.coefs, LWE_basis.modal_basis).cpu().numpy()[0,...]
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
