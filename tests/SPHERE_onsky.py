#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
from tools.utils import SR, mask_circle, GradientLoss, rad2mas
from data_processing.SPHERE_create_STD_dataset import LoadSTDStarData, process_mask, STD_FOLDER
from project_settings import device
from torchmin import minimize


#%% Initialize data sample
with open(STD_FOLDER / 'sphere_df.pickle', 'rb') as handle:
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
# sample_id = 768
# sample_id = 2649
# sample_id = 438
# sample_id = 97
# sample_id = 173 # -- interesting wings
# sample_id = 399 # -- interesting wings
# sample_id = 1248 # -- interesting rotated-LWE PSF
# sample_id = 2219 # -- interesting LWE-cross
sample_id = 1778 # -- explicit wings


PSF_data, data_sample, model_config = LoadSTDStarData(
    sample_id,
    normalize = True,
    subtract_background = True,
    ensure_odd_pixels = True,
    device = device
)

PSF_0    = PSF_data[0]['PSF (mean)']
PSF_var  = PSF_data[0]['PSF (var)']
PSF_mask = PSF_data[0]['mask (mean)']
norms    = PSF_data[0]['norm (mean)']
del PSF_data


# if psf_df.loc[sample_id]['Nph WFS'] < 10:
# PSF_mask   = PSF_mask * 0 + 1
PSF_mask = process_mask(PSF_mask)
# LWE_flag   = psf_df.loc[sample_id]['LWE']
LWE_flag = True
fit_wind = True
use_Zernike = True

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
from PSF_models.TipTorch import TipTorch
from tools.static_phase import LWEBasis, ArbitraryBasis, PixelmapBasis, ZernikeBasis

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
model = TipTorch(model_config, 'SCAO', None, PSD_include, 'sum', device, oversampling=1)
model.to_float()

LWE_basis = LWEBasis(model, ignore_pupil=False)

PSF_1  = model()

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
from managers.input_manager import InputsManager



if use_Zernike:
    N_modes = 3 # Starts with defocus
    NCPAs_basis = ZernikeBasis(model, N_modes+2, ignore_pupil=False)
    NCPAs_basis.basis = NCPAs_basis.basis[2:,...]  # remove tip/tilt
    NCPAs_basis.N_modes = N_modes-2
    
    # Z_basis = ZernikeBasis(self.model, N_modes=self.LO_N_params, ignore_pupil=False)
    # sausage_basis = MUSEPhaseBump(self.model, ignore_pupil=False)

    # # LO NCPAs + phase bump optimized jointly
    # composite_basis = torch.concat([
    #     (sausage_basis.OPD_map).unsqueeze(0).flip(-2)*5e6*self.model.pupil.unsqueeze(0),
    #     Z_basis.basis[2:self.Z_mode_max,...]
    # ], dim=0)

    # self.LO_basis = ArbitraryBasis(self.model, composite_basis, ignore_pupil=False)
    # self.LO_N_params = self.LO_basis.N_modes
    
else:
    LO_map_size = 31
    NCPAs_basis = PixelmapBasis(model, ignore_pupil=False)

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
inputs_manager.add('Jxy', torch.tensor([[0]]),     norm_Jxy)

if fit_wind:
    # inputs_manager.add('wind_speed', model.wind_speed, norm_wind_spd)
    inputs_manager.add('wind_dir', model.wind_dir,  norm_wind_dir)
if LWE_flag:
    inputs_manager.add('LWE_coefs', torch.zeros([1,12]), norm_LWE)

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
        inputs_manager.set_optimizable('LO_coefs', True)
else:
    if LO_map_size is not None:
        inputs_manager.set_optimizable('LO_coefs', True)


#%%
def phase_func(LWE_coefs, LO_coefs):
    LWE_OPD  = LWE_basis.compute_OPD(LWE_coefs)
    NCPA_OPD = NCPAs_basis.compute_OPD(LO_coefs)
    return LWE_basis.OPD2Phase(LWE_OPD + NCPA_OPD)


def func(x_):
    model_inp = inputs_manager.unstack(x_)
    if 'LWE_coefs' in model_inp:
        # Incorporate func_LO logic for phase function creation
        if 'LO_coefs' in model_inp:
            if use_Zernike:  
                phase_func_lambda = lambda: \
                    LWE_basis(model_inp['LWE_coefs']) * \
                    NCPAs_basis(model_inp['LO_coefs'])
            else:
                phase_func_lambda = lambda: \
                    LWE_basis(model_inp['LWE_coefs']) * \
                    NCPAs_basis(model_inp['LO_coefs'].view(1, LO_map_size, LO_map_size))
        else:
            # Fallback to original phase_func if LO_coefs not present
            phase_func_lambda = lambda: phase_func(
                model_inp['LWE_coefs'],
                torch.zeros_like(model_inp['LWE_coefs'])  # dummy LO_coefs
            )
        return model(model_inp, None, phase_func_lambda)
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

    if not use_Zernike:  
        grad_loss_fn = GradientLoss(p=1, reduction='mean')

    def loss_fn(x_):
        loss = img_loss(x_)
        
        coefs_LWE = inputs_manager['LWE_coefs']
        coefs_LO  = inputs_manager['LO_coefs']
        
        if use_Zernike:  
            grad_loss = 0.0
        else:  
            grad_loss = grad_loss_fn(coefs_LO.view(1, 1, LO_map_size, LO_map_size)) * 1e-4
        L2_loss_coefs = coefs_LO.pow(2).sum()*5e-9
        
        loss += LWE_regularizer(coefs_LWE) + (coefs_LWE**2).mean()*1e-4
        return loss + L2_loss_coefs + grad_loss 

else:
    def loss_fn(x_):
        return img_loss(x_)


x0 = inputs_manager.stack()


#%%
result = minimize(loss_fn, x0, max_iter=300, tol=1e-5, method='l-bfgs', disp=2)
x0 = result.x

PSF_1 = func(x0)

#%%
from tools.static_phase import BuildPTTBasis, decompose_WF, project_WF

# LWE modes can couple with PTT modes - remove this effect
LWE_coefs   = inputs_manager['LWE_coefs']
NCPAs_coefs = inputs_manager['LO_coefs']
PTT_basis = BuildPTTBasis(model.pupil.cpu().numpy(), True).to(device).float()

TT_max = PTT_basis.abs()[1,...].max().item()
pixel_shift = lambda coef: 2 * TT_max * rad2mas * 1e-9 * coef / model.psInMas / model.D  / (1-7/model.pupil.shape[-1])

if use_Zernike:
    NCPA_OPD  = NCPAs_basis.compute_OPD(NCPAs_coefs) * 1e9 # [nm]
else:
    # TODO: fix strange normalization here
    NCPA_OPD  = NCPAs_basis.interp_upscale(NCPAs_coefs.view(1, LO_map_size, LO_map_size))[0,...] #* 1e9  # [nm]

LWE_OPD   = LWE_basis.compute_OPD(LWE_coefs) * 1e9 # [nm]
PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, model.pupil)
PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, model.pupil)

inputs_manager['LWE_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, LWE_basis.modal_basis, model.pupil)
inputs_manager['dx'] -= pixel_shift(PTT_coefs[:, 2])
inputs_manager['dy'] -= pixel_shift(PTT_coefs[:, 1])
x0_compensated = inputs_manager.stack()

PSF_1 = func(x0_compensated).detach()

# inputs_manager_new = inputs_manager.copy()

OPD_map = LWE_OPD - PPT_OPD + NCPA_OPD

plt.imshow(OPD_map[0,...].detach().cpu().numpy())#, vmin=-0.05, vmax=0.05)
# plt.imshow((LWE_OPD)[0,...].cpu().numpy())#, vmin=-0.05, vmax=0.05)
# plt.imshow(PPT_OPD[0,...].cpu().numpy())#, vmin=-0.05, vmax=0.05)
plt.colorbar()
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1*PSF_mask)[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
plot_radial_PSF_profiles( (PSF_0*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1*PSF_mask)[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_0*PSF_mask, PSF_1*PSF_mask, min_val=1e-6, average=True, crop=80)


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
WFE = torch.mean(model.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = model.D/4 * 1e9*(model.Jx+model.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / model.wvl.flatten()[0]

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')
