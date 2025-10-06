#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from torchmin import minimize
from managers.input_manager import InputsManager

from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack, plot_chromatic_PSF_slice
from tools.utils import PupilVLT, GradientLoss, RadialProfileLossSimple
# from tools.utils import mask_circle
from data_processing.MUSE_STD_dataset_utils import STD_FOLDER, LoadSTDStarData
from data_processing.normalizers import Uniform, Uniform0_1, Atanh, Uniform0_1
from project_settings import device

derotate_PSF    = True
Moffat_absorber = False
fit_LO          = True
fit_wind_speed  = True
chrom_defocus   = False and fit_LO
spline_fit      = False
LO_N_params     = 75
Z_mode_max      = 9
N_spline_ctrl   = 5

#%
# import pickle

# with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
#     muse_df = pickle.load(handle)
    
# muse_df.index[muse_df['Wind speed (header)'] > 16]

# 6, 96, 151, 152, 188, 354 # strong wind ones

#%%
# with open(MUSE_DATA_FOLDER+'/muse_df.pickle', 'rb') as handle:
#     muse_df = pickle.load(handle)

# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 5), a_min=0, a_max=N_wvl_max-1)
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 3), a_min=0, a_max=N_wvl_max-1)
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 2), a_min=0, a_max=N_wvl_max-1)

# ids = 96 # strongest wind
# ids = 394 # strong wind
# ids = 278 # strong wind
# ids = 176 # strong wind
# ids = 296 # sausage
# ids = 324
# ids = 230 # PSF with chromatic displacement
# ids = 231
# ids = 344 # intense phase bump
# ids = [344, 179, 451] # intense phase bump
# ids = 423 # relatively good one

# ids = 404 # intense streaks
# ids = 462
# ids = 465 # slight sausage
# ids = 359
# ids = 121
# ids = 184 # weak wind patterns
# ids = 338 # blurry
# ids = 470 # blurry
# ids = 346 # blurry
# ids = 206 # blurry, good for red debugging
# ids = 179 # blurry
# ids = 174

# ids = 428 # good one, no DM mismathc yet
# ids = 434 # good one, no DM mismathc yet

# ids = 446 # does not converge with L-BFGS

# ids = 440 # good one, but DM correction mismatch
# ids = 449 # good one, but DM correction mismatch
# ids = 451 # good one, but DM correction mismatch
# ids = 453 # good one, but DM correction mismatch

# ids = 455 # good one

# ids = 457 # surprisingly poor blue fitting
# ids = 477 # surprisingly poor blue fitting
ids = 467 # surprisingly poor blue fitting
# ids = 458 # surprisingly poor blue fitting
ids = 468 # surprisingly poor blue fitting

# ids = 482 # good one
# ids = 494 # good one
# ids = 462 # good one
# ids = 475 # good one

PSF_0, norms, bgs, model_config = LoadSTDStarData(
    ids = ids,
    derotate_PSF = derotate_PSF,
    normalize = True,
    subtract_background = True,
    wvl_ids = wvl_ids,
    ensure_odd_pixels = True,
    device = device
)

# PSF_0[:,-1,...] /= 0.97
# PSF_0[:,-2,...] /= 0.985
# PSF_0[:,-1,...] /= 0.99
# PSF_0[:,-2,...] /= 0.999

if derotate_PSF:
    pupil_angle = 0.0
else:
    pupil_angle = model_config['telescope']['PupilAngle'].cpu().numpy().item()

N_wvl = PSF_0.shape[1]
N_src = PSF_0.shape[0]

wavelengths = model_config['sources_science']['Wavelength'].squeeze()

#%
cmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
cmap.set_bad(color='black')

for j in range(N_src):
    for i in range(N_wvl):
        im = PSF_0[j,i,...].cpu().numpy()
        vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
        vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()

        plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.axis('off')
        plt.show()
        
    plt.show()

#%
# plot_chromatic_PSF_slice(PSF_0, wavelengths, norms, window_size=40)
# plot_chromatic_PSF_slice(PSF_0, wavelengths, window_size=100, slices=['horizontal'])

#%% Initialize the model
from PSF_models.TipTorch import TipTorch
from tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis, MUSEPhaseBump

# model_config['DM']['DmPitchs'] = torch.tensor([0.245], dtype=torch.float32, device=device)  # [m]

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=pupil_angle), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          Moffat_absorber
}
PSF_model = TipTorch(model_config, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
# PSF_model.apodizer = PSF_model.make_tensor(1.0)
PSF_1 = PSF_model()

#%%
# LO_basis = PixelmapBasis(PSF_model, ignore_pupil=False)
Z_basis = ZernikeBasis(PSF_model, N_modes=LO_N_params, ignore_pupil=False)
sausage_basis = MUSEPhaseBump(PSF_model, ignore_pupil=False)

#%
# LO NCPAs + phase bump optimized jointly
composite_basis = torch.concat([
    (sausage_basis.OPD_map).unsqueeze(0).flip(-2)*5e6*PSF_model.pupil.unsqueeze(0),
    # sausage_basis.OPD_map.unsqueeze(0).flip(-2)*5e6,
    Z_basis.zernike_basis[2:Z_mode_max,...]
], dim=0)

defocus_mode_id = 1 # the index of defocus mode

LO_basis = ArbitraryBasis(PSF_model, composite_basis, ignore_pupil=False)
LO_N_params = LO_basis.N_modes

# for i in range(LO_basis.N_modes):
#     plt.imshow(LO_basis.basis[i,...].cpu().numpy())
#     plt.title(f"LO Mode {i+1}")
#     plt.colorbar()
#     plt.show()

#%% Initialize inputs manager and set up parameters
inputs_manager = InputsManager()

# Initialize normalizers/transforms
norm_F           = Uniform(a=0.0,   b=1.0)
norm_bg          = Uniform(a=-5e-6, b=5e-6)
norm_r0          = Uniform(a=0,     b=1)
norm_dxy         = Uniform(a=-1,    b=1)
norm_J           = Uniform(a=0,     b=50)
norm_Jxy         = Uniform(a=-180,  b=180)
norm_dn          = Uniform(a=0,     b=5)
norm_amp         = Uniform(a=0,     b=10)
norm_b           = Uniform(a=0,     b=0.1)
norm_alpha       = Uniform(a=-1,    b=10)
norm_beta        = Uniform(a=0,     b=2)
norm_ratio       = Uniform(a=0,     b=2)
norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
norm_wind_speed  = Uniform(a=0, b=10)
norm_wind_dir    = Uniform(a=0, b=360)
norm_sausage_pow = Uniform(a=0, b=1)
norm_LO          = Uniform(a=-100, b=100)
# norm_GL_h        = Uniform(a=0.0, b=2000.0)
# norm_GL_frac     = Atanh()


# TODO: must be scaling of spline values, too
norm_wvl = Uniform0_1(a=wavelengths.min().item(), b=wavelengths.max().item())
wvl_norm = norm_wvl(wavelengths).clone().unsqueeze(0)
x_ctrl = torch.linspace(0, 1, N_spline_ctrl, device=device)

# polychromatic_params = ['F', 'dx', 'dy', 'Jx', 'Jy']
polychromatic_params = ['F', 'dx', 'dy'] + (['chrom_defocus'] if chrom_defocus else ['J'])

# Add base parameters
if spline_fit:
    inputs_manager.add('F_ctrl',  torch.tensor([[1.0,]*N_spline_ctrl]*N_src),  norm_F)
    inputs_manager.add('dx_ctrl', torch.tensor([[0.0,]*N_spline_ctrl]*N_src),  norm_dxy)
    inputs_manager.add('dy_ctrl', torch.tensor([[0.0,]*N_spline_ctrl]*N_src),  norm_dxy)
    inputs_manager.add('bg_ctrl', torch.tensor([[0.0,]*N_spline_ctrl]*N_src),  norm_bg)
else:
    inputs_manager.add('F',  torch.tensor([[1.0,]*N_wvl]*N_src),  norm_F)
    inputs_manager.add('dx', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy)
    inputs_manager.add('dy', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_dxy)
    inputs_manager.add('bg', torch.tensor([[0.0,]*N_wvl]*N_src),  norm_bg)

if chrom_defocus:
    inputs_manager.add('J', torch.tensor([[25.0]]*N_src), norm_J)
else:
    if spline_fit:
        inputs_manager.add('J_ctrl', torch.tensor([[25.0,]*N_spline_ctrl]*N_src), norm_J)
        # inputs_manager.add('Jx_ctrl', torch.tensor([[10.0,]*N_spline_ctrl]*N_src), norm_dxy)
        # inputs_manager.add('Jy_ctrl', torch.tensor([[10.0,]*N_spline_ctrl]*N_src), norm_dxy)
    else:
        inputs_manager.add('J', torch.tensor([[25.0]*N_wvl]*N_src), norm_J)
        # inputs_manager.add('Jx', torch.tensor([[25.0]*N_wvl]*N_src),  norm_J)
        # inputs_manager.add('Jy', torch.tensor([[25.0]*N_wvl]*N_src),  norm_J)
        
inputs_manager.add('r0', PSF_model.r0.clone(), norm_r0)
if fit_wind_speed:
    # inputs_manager.add('wind_dir_single',   PSF_model.wind_dir[0,0].clone().unsqueeze(-1),   norm_wind_dir)
    inputs_manager.add('wind_speed_single', PSF_model.wind_speed[:,0].clone().unsqueeze(-1), norm_wind_speed)

inputs_manager.add('Jxy', torch.tensor([[0.0]]*N_src), norm_Jxy, optimizable=False)
inputs_manager.add('dn',  torch.tensor([0.25]*N_src),  norm_dn)

# GL_frac = np.maximum(PSF_model.Cn2_weights[0,-1].detach().cpu().numpy().item(), 0.9)
# GL_h    = PSF_model.h[0,-1].detach().cpu().numpy().item()

# inputs_manager.add('GL_frac', torch.tensor([GL_frac]), norm_GL_frac)
# inputs_manager.add('GL_h',    torch.tensor([GL_h]), norm_GL_h)

# Add Moffat parameters if needed
if Moffat_absorber:
    inputs_manager.add('amp',   torch.tensor([1e-4]*N_src), norm_amp)
    inputs_manager.add('b',     torch.tensor([0.0]*N_src), norm_b)
    inputs_manager.add('alpha', torch.tensor([4.5]*N_src), norm_alpha)
    inputs_manager.add('beta',  torch.tensor([2.5]*N_src), norm_beta)
    inputs_manager.add('ratio', torch.tensor([1.0]*N_src), norm_ratio)
    inputs_manager.add('theta', torch.tensor([0.0]*N_src), norm_theta)

if fit_LO:
    if isinstance(LO_basis, PixelmapBasis):
        inputs_manager.add('LO_coefs', torch.zeros([N_src, LO_N_params**2]), norm_LO)
        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_N_params, LO_N_params))
        
    elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
        inputs_manager.add('LO_coefs', torch.zeros([N_src, LO_N_params]), norm_LO)

        if chrom_defocus:
            inputs_manager.add('chrom_defocus',  torch.tensor([[0.0,]*N_wvl]*N_src),  norm_LO, optimizable=chrom_defocus)

            def phase_func():
                coefs_chromatic = inputs_manager["LO_coefs"].view(N_src, LO_N_params).unsqueeze(1).repeat(1, N_wvl, 1)
                coefs_chromatic[:, :, defocus_mode_id] += inputs_manager["chrom_defocus"].view(N_src, N_wvl) # add chromatic defocus
                return LO_basis(coefs_chromatic)
        else:
            phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(N_src, LO_N_params))
    else:
        raise ValueError('Wrong LO type specified.')
else:
    phase_func = None

inputs_manager.to(device)
inputs_manager.to_float()

# print(inputs_manager)

_ = inputs_manager.stack()

#%%
def func(x_, include_list=None):
    x_torch = inputs_manager.unstack(x_)

    if spline_fit:
        for entry in polychromatic_params:
            spline = NaturalCubicSpline(natural_cubic_spline_coeffs(x_ctrl, x_torch[entry+'_ctrl'].T))
            x_torch[entry] = spline.evaluate(wvl_norm).squeeze(-1)
    
    # Clone J entry to Jx and Jy
    x_torch['Jx'] = x_torch['J']
    x_torch['Jy'] = x_torch['J']
    
    if fit_wind_speed:
        # x_torch['wind_dir']   = x_torch['wind_dir_single'].unsqueeze(-1).repeat(1, PSF_model.N_L)
        x_torch['wind_speed'] = x_torch['wind_speed_single'].unsqueeze(-1).repeat(1, PSF_model.N_L)

    # x_torch['Cn2_weights'] = torch.hstack([x_torch['GL_frac'], 1.0 - x_torch['GL_frac']]).unsqueeze(0)
    # x_torch['h']           = torch.hstack([torch.tensor([0.0], device=device), x_torch['GL_h'].abs()]).unsqueeze(0)

    x_ = { key: x_torch[key] for key in include_list } if include_list is not None else x_torch

    return PSF_model(x_torch, None, phase_generator=phase_func)

x_ = inputs_manager.stack()
PSF_1 = func(x_)

#%%
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)
wvl_weights = N_wvl / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

wvl_weights = wvl_weights * 0 + 1

# mask = torch.tensor(mask_circle(PSF_0.shape[-1], 5)).view(1, 1, *PSF_0.shape[-2:]).to(device)
# mask_inv = 1.0 - mask

# loss_radial_fn = RadialProfileLossSimple(
#     n_bins=64,
#     loss="fvu",      # or "mse"
#     bin_weight="r",  # "uniform" or "counts" also available
#     log_profile=False
# )

loss_Huber = torch.nn.HuberLoss(reduction='mean', delta=0.05)
loss_MAE   = torch.nn.L1Loss(reduction='mean')
loss_MSE   = torch.nn.MSELoss(reduction='mean')


def loss_fn(x_, w_MSE, w_MAE):
    diff = (func(x_)-PSF_0) * wvl_weights
    w = 2e4
    MSE_loss = diff.pow(2).mean() * w * w_MSE
    MAE_loss = diff.abs().mean()  * w * w_MAE
    LO_loss = loss_LO_fn() if fit_LO else 0.0

    return MSE_loss + MAE_loss + LO_loss


def loss_LO_fn():
    if isinstance(LO_basis, PixelmapBasis):
        LO_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_N_params, LO_N_params)) * 5e-5
    elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
        LO_loss = inputs_manager['LO_coefs'].abs().sum()**2 * 1e-7
        # Constraint to enforce first element of LO_coefs to be positive
        first_coef_penalty = torch.clamp(-inputs_manager['LO_coefs'][0, 0], min=0).pow(2) * 5e-5
        LO_loss += first_coef_penalty
    return LO_loss


def loss_fn_Huber(x_):
    PSF_1 = func(x_)
    huber_loss = loss_Huber(PSF_1*wvl_weights*5e5, PSF_0*wvl_weights*5e5)
    MSE_loss = loss_MSE(PSF_1*wvl_weights, PSF_0*wvl_weights) * 2e4 * 800.0
    LO_loss = loss_LO_fn() if fit_LO else 0.0

    return huber_loss + LO_loss + MSE_loss

loss_fn1 = lambda x_: loss_fn(x_, w_MSE=800.0, w_MAE=1.6)
loss_fn2 = lambda x_: loss_fn(x_, w_MSE=1.0, w_MAE=2.0)
grad_loss_fn = GradientLoss(p=1, reduction='mean')


#%
# def loss_radial(x_):
#     PSF_1 = func(x_)
#     diff = (PSF_1-PSF_0) * wvl_weights
#     mse_loss = (diff * 4000).pow(2).mean()
#     mae_loss = (diff * 32000).abs().mean()
#     if fit_LO:
#         if isinstance(LO_basis, PixelmapBasis):
#             LO_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_N_params, LO_N_params)) * 5e-5
#         elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
#             LO_loss = inputs_manager['LO_coefs'].abs().sum()**2 * 1e-7
#             # Constraint to enforce first element of LO_coefs to be positiv
#             first_coef_penalty = torch.clamp(-inputs_manager['LO_coefs'][0, 0], min=0).pow(2) * 5e-5
#             LO_loss += first_coef_penalty
#         else:
#             raise ValueError('Wrong LO type specified.')
#     else:
#         LO_loss = 0.0

#     loss_rad = loss_radial_fn(PSF_1, PSF_0) * 5000
#     return mse_loss + mae_loss + LO_loss + loss_rad


#%%
def minimize_params(loss_fn, include_list, exclude_list, max_iter, verbose=True):
    if len(include_list) > 0:
        inputs_manager.set_optimizable(include_list, True)
    else:
        raise ValueError('include_list is empty')
        
    inputs_manager.set_optimizable(exclude_list, False)

    # print(inputs_manager)

    x_backup = inputs_manager.stack().clone()

    result = minimize(loss_fn, inputs_manager.stack(), max_iter=max_iter, tol=1e-4, method='l-bfgs', disp=2 if verbose else 0)
    if result['nit'] < max_iter * 0.3:
        if verbose:
            print("Warning: minimization stopped too early. Perhaps, convergence wasn't reached? Trying BFGS...")
        inputs_manager.unstack(x_backup, include_all=True, update=True)
        result = minimize(loss_fn, inputs_manager.stack(), max_iter=max_iter, tol=1e-5, method='bfgs', disp=2 if verbose else 0)
    
    if result['fun'] > 1:
        if verbose:
            print("Warning: final loss is high. Perhaps, convergence wasn't reached? Trying BFGS...")
        inputs_manager.unstack(x_backup, include_all=True, update=True)
        result = minimize(loss_fn, inputs_manager.stack(), max_iter=max_iter, tol=1e-5, method='bfgs', disp=2 if verbose else 0)

    if fit_LO:
        if isinstance(LO_basis, PixelmapBasis):
            OPD_map = inputs_manager['LO_coefs'].view(N_src, LO_N_params, LO_N_params).detach().cpu().numpy()
        elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
            OPD_map = LO_basis.compute_OPD(inputs_manager["LO_coefs"].view(N_src, LO_N_params)).detach().cpu().numpy()
    else:
        OPD_map = None
        
    if verbose:
        print('-'*50)

    return result.x, func(result.x), OPD_map

                #   ['wind_speed'] + \
include_general = ['r0', 'dn'] + \
                  (['amp', 'alpha', 'beta', 'b'] if Moffat_absorber else []) + \
                  (['LO_coefs'] if fit_LO else []) + (['chrom_defocus'] if chrom_defocus else []) + \
                  ([x+'_ctrl' for x in polychromatic_params] if spline_fit else polychromatic_params)

exclude_general = ['ratio', 'theta'] if Moffat_absorber else []
include_LO = (['LO_coefs'] if fit_LO else []) + (['chrom_defocus'] if chrom_defocus else [])
exclude_LO = list(set(include_general + exclude_general) - set(include_LO))

# inc_minus_Moffat = list(set(include_general) - set(['amp', 'alpha', 'beta', 'b']))
# inc_only_Moffat = ['amp', 'alpha', 'beta', 'b']

#%%
x0, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 150)
# x0, PSF_1, OPD_map = minimize_params(loss_fn_Huber, include_general, exclude_general, 150)

# if fit_LO:
    # x1, PSF_1, OPD_map = minimize_params(loss_fn2, include_LO, exclude_LO, 50)
    # x2, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 50)

#%
# import pickle

# with open(f'PSF_fitted_{ids}.pickle', 'wb') as handle:
#     pickle.dump({
#         'PSF_1': PSF_1.detach().cpu().numpy(),
#         'x0': OPD_map,
#         'IDS': ids,
#     }, handle)

#%%
from tools.plotting import plot_radial_PSF_profiles

id_src = 0

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
wvl_select = np.s_[0, N_wvl//2, -1]

draw_PSF_stack(
    PSF_0.cpu().numpy()[id_src, wvl_select, ...],
    PSF_1.cpu().numpy()[id_src, wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=100
)

PSF_disp = lambda x, w: (x[id_src,w,...]).cpu().numpy()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSF_disp(PSF_0, lmbd),
        PSF_disp(PSF_1, lmbd),
        'Data',
        'TipTorch',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        return_profiles=True,
        ax=ax[i]        
    )
plt.show()

if fit_LO:
    plt.imshow(OPD_map[id_src,...]*1e9)
    plt.colorbar()
    plt.show()

#%%
from data_processing.MUSE_STD_dataset_utils import GetROIaroundMax, GetSpectrum

diff_im = (PSF_1-PSF_0).abs()[0,...].squeeze().cpu().numpy()

_, _, max_id = GetROIaroundMax(diff_im.mean(0).squeeze(), 10) # TODO: this function is redundant
spectrum_diff = GetSpectrum(diff_im, max_id, radius=2)
_, _, max_id = GetROIaroundMax(PSF_0[0,...].squeeze().cpu().numpy().mean(0), 10) # TODO: this function is redundant
spectrum_data = GetSpectrum(PSF_0[0,...].squeeze().cpu().numpy(), max_id, radius=5)

plt.plot(wavelengths.squeeze().cpu().numpy()*1e9, spectrum_diff / spectrum_data * 100)

# %%
from tools.plotting import plot_radial_PSD_profiles

plot_radial_PSD_profiles(PSF_model)

#%%
if spline_fit:
    x_torch = inputs_manager.unstack(x2)
    spline = NaturalCubicSpline(natural_cubic_spline_coeffs(x_ctrl, x_torch['F_ctrl'].T))
    AA = spline.evaluate(torch.linspace(0, 1, 100).to(device)).squeeze(-1)
    plt.plot(np.linspace(wavelengths.min().item()*1e9, wavelengths.max().item()*1e9, 100), AA.detach().cpu().numpy())

plt.plot(wavelengths.squeeze(0).cpu().numpy()*1e9, PSF_model.F.squeeze().detach().cpu().numpy())
plt.show()

#%%
from tools.plotting import plot_chromatic_PSF_slice

# plot_chromatic_PSF_slice(PSF_1.abs(), wavelengths, window_size=40)
plot_chromatic_PSF_slice(PSF_0.abs(), wavelengths, window_size=40, scale='linear')
# plot_chromatic_PSF_slice(PSF_0-PSF_1, wavelengths, window_size=40)


#%%

for j in range(N_src):
    for i in range(N_wvl):
        im = (PSF_1-PSF_0).abs()[j,i,...].cpu().numpy()
        vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
        vmax = np.percentile(im[im > 0], 99.9975) if np.any(im > 0) else im.max()

        plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.axis('off')
        plt.show()
        
    plt.show()



#%%
F_copy = PSF_model.F.detach().clone()

#%%
inputs_manager['F'] = torch.tensor([[1.0]*N_wvl], device=device)

x_tmp = inputs_manager.stack()

PSF_1_flat = func(x_tmp)

plot_chromatic_PSF_slice((PSF_1_flat).abs(), wavelengths, window_size=40, scale='linear')

#%%
plot_chromatic_PSF_slice((PSF_1).abs(), wavelengths, window_size=40, scale='linear')

#%%
plot_chromatic_PSF_slice((PSF_1-PSF_0)/PSF_0, wavelengths, window_size=40, scale='linear', slices=['horizontal','vertical'])

#%%
# PSF_diff = (PSF_0 - PSF_1)[0,:,...].mean(0).squeeze()
PSF_diff = (PSF_0 - PSF_1)[0,-5:,...].mean(0).squeeze()#.abs()

center_y, center_x = PSF_diff.shape[-2] // 2, PSF_diff.shape[-1] // 2
half_window = 50 // 2

# Calculate slice bounds
y_start = max(0, center_y - half_window)
y_end = min(PSF_diff.shape[-2], center_y + half_window)
x_start = max(0, center_x - half_window)
x_end = min(PSF_diff.shape[-1], center_x + half_window)
    
h_slice = PSF_diff[center_y, x_start:x_end].cpu().numpy()
v_slice = PSF_diff[y_start:y_end, center_x].cpu().numpy()
# Calculate diagonal slice (top-left to bottom-right)
min_len = min(len(h_slice), len(v_slice))

d_slice = []
for i in range(min_len):
    y_idx = y_start + i
    x_idx = x_start + i
    if y_idx < PSF_diff.shape[-2] and x_idx < PSF_diff.shape[-1]:
        d_slice.append(PSF_diff[y_idx, x_idx].cpu().numpy())
d_slice = np.array(d_slice)

# Make slices the same length and average
# slice_ = 0.5 * (h_slice[:min_len] + v_slice[:min_len])
# slice_ = v_slice[:min_len]
slice_ = d_slice[:min_len]
plt.plot(slice_)

# %%
from photutils.profiles import RadialProfile

xycen = (PSF_diff.shape[-1]//2, PSF_diff.shape[-2]//2)  # (x, y) position tuple
edge_radii = np.arange(PSF_diff.shape[-1]//2)
rp = RadialProfile(PSF_diff.cpu().numpy().squeeze(), xycen, edge_radii)

plt.plot(rp.profile)
plt.xlim(0, 25)


# %%
# TODO: curently improperly normalized
# Compute error budget over PSDs with proper integration
error_budget = {}
dk = PSF_model.dk
dk_squared = dk**2  # Area element in frequency space

for entry in PSF_model.PSD_include:
    PSD = PSF_model.PSDs[entry]

    if len(PSD.shape) > 1:            
        PSD_norm = (PSF_model.dk*PSF_model.wvl_atm*1e9/2/torch.pi)**2
        PSD = PSF_model.half_PSD_to_full(PSD * PSD_norm).real # [nm^2]
        
        # Proper integration: multiply by dk^2 for area element
        error_budget[entry] = (PSD * dk_squared).sum().item()
        
        print(f"{entry:15s}: {np.sqrt(error_budget[entry]):.2f} nm RMS")
        
total_PSD = PSF_model.PSD.real
error_budget['Total PSD'] = (total_PSD * dk_squared).sum().item()
print(f"{'Total PSD':15s}: {np.sqrt(error_budget['Total PSD']):.2f} nm RMS")


# %%
# for i in range(N_wvl):
im = PSF_0[0,-1,...].cpu().numpy()
vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()

plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.show()

im = PSF_1[0,-1,...].cpu().numpy()
vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()

plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
plt.show()
# %%
