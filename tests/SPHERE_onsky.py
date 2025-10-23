#%%
%reload_ext autoreload
%autoreload 2

import sys

from tests.SPHERE_dynesty import LWE_coefs
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
from tools.utils import mask_circle, rad2mas, GradientLoss
from data_processing.SPHERE_STD_dataset_utils import LoadSTDStarData, process_mask, STD_FOLDER
from project_settings import device
from torchmin import minimize

from PSF_models.IRDIS_wrapper import PSFModelIRDIS


#%% Initialize data sample
with open(STD_FOLDER / 'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

subset_df = psf_df[psf_df['Corrupted'] == False]
subset_df = subset_df[subset_df['High quality'] == True]
subset_df = subset_df[subset_df['High SNR'] == True]
subset_df = subset_df[subset_df['LWE'] == True]

#%%
# sample_id = 768
# sample_id = 2649
# sample_id = 438
# sample_id = 97
# sample_id = 173 # -- interesting wings
# sample_id = 399 # -- interesting wings
# sample_id = 1248 # -- interesting rotated-LWE PSF
# sample_id = 2219 # -- interesting LWE-cross
sample_id = 1778  # -- explicit wings


PSF_data, data_sample, model_config = LoadSTDStarData(
    sample_id,
    normalize=True,
    subtract_background=True,
    ensure_odd_pixels=True,
    device=device
)

PSF_0 = PSF_data[0]['PSF (mean)']
PSF_var = PSF_data[0]['PSF (var)']
PSF_mask = PSF_data[0]['mask (mean)']
norms = PSF_data[0]['norm (mean)']
del PSF_data

# Process mask
PSF_mask = process_mask(PSF_mask)
LWE_flag = True
fit_wind = True
LO_NCPAs = False

# Handle central hole if present
if psf_df.loc[sample_id]['Central hole'] == True:
    circ_mask = 1-mask_circle(PSF_0.shape[-1], 3, centered=True)
    PSF_mask *= torch.tensor(circ_mask[None, None, ...]).to(device)

#%% Initialize PSF model
PSF_model = PSFModelIRDIS(
    model_config,
    LWE_flag=LWE_flag,
    fit_wind=fit_wind,
    LO_NCPAs=LO_NCPAs,
    use_Zernike=True,
    N_modes=9,
    LO_map_size=31,
    device=device
)

#%%
func = lambda x_: PSF_model(PSF_model.inputs_manager.unstack(x_))
PSF_1 = func(x0 := PSF_model.inputs_manager.stack())

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, average=True, min_val=1e-5, crop=80, scale='log')

#%% Create loss functions (like in original SPHERE_onsky.py)
img_loss = lambda x: ((func(x) - PSF_0) * PSF_mask).flatten().abs().sum()

LWE_gauss_penalty_weight = 50.0
# LWE_gauss_penalty_weight = 100.0
LO_NCPAs_penalty_weight  = 5e-9
LWE_coefs_penalty_weight = 1e-4
# LWE_coefs_penalty_weight = 1e-5

if LWE_flag:
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
    patterns = [torch.tensor([p]).to(device).float() * LWE_gauss_penalty_weight for p in patterns]
    
    gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
    Gauss_err = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, LWE_gauss_penalty_weight/2)).flatten().abs().sum()

    LWE_regularizer = lambda c: sum(Gauss_err(pattern, c) for pattern in patterns)


if not PSF_model.use_Zernike:  
    grad_loss_fn = GradientLoss(p=1, reduction='mean')

if PSF_model.LWE_flag or PSF_model.LO_NCPAs:
    def loss_fn(x_):
        loss = img_loss(x_)
        
        # Update inputs manager to access coefficients
        PSF_model.inputs_manager.unstack(x_, include_all=True, update=True)
        
        if PSF_model.LWE_flag:
            coefs_LWE = PSF_model.inputs_manager['LWE_coefs']
            loss += LWE_regularizer(coefs_LWE) + (coefs_LWE**2).mean() * LWE_coefs_penalty_weight
        
        if PSF_model.LO_NCPAs:
            coefs_LO = PSF_model.inputs_manager['LO_coefs']

            if PSF_model.use_Zernike:  
                grad_loss = 0.0
            else:  
                grad_loss = grad_loss_fn(coefs_LO.view(1, 1, PSF_model.LO_map_size, PSF_model.LO_map_size)) * 1e-4

            L2_loss_coefs = coefs_LO.pow(2).sum() * LO_NCPAs_penalty_weight
        else:
            L2_loss_coefs = 0.0
            grad_loss = 0.0
            
        return loss + L2_loss_coefs + grad_loss 

else:
    def loss_fn(x_):
        return img_loss(x_)

x0 = PSF_model.inputs_manager.stack()

#%% Single-stage PSF fitting
result = minimize(loss_fn, x0, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x0 = result.x

PSF_1 = func(x0)

#%% PTT compensation
LWE_OPD, PPT_OPD, NCPA_OPD = PSF_model.compensate_PTT_coupling()

x0_compensated = PSF_model.inputs_manager.stack()

PSF_1 = func(x0_compensated).detach()

if NCPA_OPD is None:
    NCPA_OPD = 0.0

OPD_map = LWE_OPD + NCPA_OPD

plt.imshow(OPD_map[0,...].detach().cpu().numpy())
plt.colorbar()
plt.title('Combined OPD Map')
plt.show()

#%% Results visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), 
    (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Left PSF', ax=ax[0]
)
plot_radial_PSF_profiles(
    (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), 
    (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 
    'Data', 'TipTorch', title='Right PSF', ax=ax[1]
)
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)


#%% Photon count estimation
Nph_new = PSF_model.GetNewPhotons()
print(f"Original WFS photons: {PSF_model.model.WFS_Nph.item():.1f}")
print(f"Estimated WFS photons: {Nph_new.item():.1f}")

#%% Combined phase map visualization
if LWE_flag and hasattr(PSF_model, 'NCPAs_basis'):
    LWE_map = LWE_OPD - PPT_OPD

    if use_Zernike:
        fitted_map = PSF_model.NCPAs_basis.compute_OPD(PSF_model.inputs_manager['LO_coefs'])[0] * 1e9
    else:
        fitted_map = PSF_model.NCPAs_basis.interp_upscale(PSF_model.inputs_manager['LO_coefs'].view(1, PSF_model.LO_map_size, PSF_model.LO_map_size))

    combined_map = (PSF_model.model.pupil * (fitted_map + LWE_map)).cpu().numpy().squeeze()

    plt.imshow(combined_map)
    plt.colorbar()
    plt.title('Combined LWE + NCPA Phase Map')
    plt.show()

#%% Error budget analysis
WFE = torch.mean(PSF_model.model.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = PSF_model.model.D/4 * 1e9*(PSF_model.model.Jx+PSF_model.model.Jy)*0.5/rad2mas
WFE_total = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / PSF_model.model.wvl.flatten()[0]
theoretical_SR = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE (no LWE): {WFE_total:.2f} nm')
print(f'Theoretical Strehl ratio: {theoretical_SR:.3f}')

# Calculate actual Strehl ratio if we had a reference PSF
# Note: This would require a diffraction-limited PSF for comparison
# print(f'Measured Strehl ratio: {SR(PSF_1, PSF_DL):.3f}')