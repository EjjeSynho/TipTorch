#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from torchmin import minimize

from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack, plot_chromatic_PSF_slice
# from tools.utils import GradientLoss
from data_processing.MUSE_STD_dataset_utils import STD_FOLDER, LoadSTDStarData
from PSF_models.NFM_wrapper import PSFModelNFM
from project_settings import device

#%%
import pickle

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

#%%
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 5), a_min=0, a_max=N_wvl_max-1)
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 3), a_min=0, a_max=N_wvl_max-1)
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 2), a_min=0, a_max=N_wvl_max-1)

# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 2), a_min=0, a_max=N_wvl_max-1)[:14]

#%%
# ids = 96 # strongest wind (not really, it was a tomo error, likely)
# ids = 394 # strong wind (not really, it was a tomo error, likely)
# ids = 278 # strong wind (not really, it was a tomo error, likely)
# ids = 176 # strong wind
# ids = 296 # sausage
# ids = 230 # PSF with chromatic displacement
# ids = 324
# ids = 344 # intense phase bump
# ids = [344, 179, 451] # intense phase bump
# ids = 423 # relatively good one
# ids = 231

# ids = 404 # intense streaks
# ids = 465 # slight sausage
# ids = 184 # weak wind patterns
# ids = 470 # blurry
# ids = 346 # blurry (very)
# ids = 179 # blurry

# ids = 121
# ids = 174

# ids = 428 # good one, no DM mismathc yet
# ids = 434 # good one, no DM mismathc yet

# ids = 446 # does not converge with L-BFGS

# ids = 440 # good one, but DM correction mismatch
# ids = 449 # good one, but DM correction mismatch
# ids = 451 # good one, but DM correction mismatch
# ids = 453 # good one, but DM correction mismatch


# ids = 338 # poor blue fitting
# ids = 206 # poor blue fitting, wind wings (?)
# ids = 457 # poor blue fitting
# ids = 477 # poor blue fitting
# ids = 467 # poor blue fitting
# ids = 458 # poor blue fitting
# ids = 494 # poor blue fitting

# ids = 455 # good one, slight sausage
# ids = 359 # good one
# ids = 482 # good one
# ids = 462 # good one
# ids = 475 # good one, mb DM correction radius mismatch
# ids = 468 # good one

# ids = 455 # large tomographic error (?)
# ids = 456 # large tomographic error

ids = 125

PSF_0, norms, bgs, configs = LoadSTDStarData(
    ids                 = ids,
    derotate_PSF        = True,
    normalize           = True,
    subtract_background = True,
    wvl_ids             = wvl_ids,
    ensure_odd_pixels   = True,
    device              = device
)

#%%
PSF_model = PSFModelNFM(
    configs,
    multiple_obs    = True,
    LO_NCPAs        = True,
    chrom_defocus   = False,
    use_splines     = True,
    Moffat_absorber = False,
    Z_mode_max      = 9,
    device          = device
)

func = lambda x_: PSF_model( PSF_model.inputs_manager.unstack(x_) )
PSF_1 = func(x0 := PSF_model.inputs_manager.stack() )

N_wvl = PSF_0.shape[1]
N_src = PSF_0.shape[0]

wavelengths = PSF_model.wavelengths

#%%
cmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
cmap.set_bad(color='black')

for j in range(N_src):
    for i in range(N_wvl):
        im = PSF_0[j,i,...].cpu().numpy()
        vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
        vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()
        # vmin = np.percentile(im[im > 0], 97) if np.any(im > 0) else 1
        # vmax = np.percentile(im[im > 0], 99.95) if np.any(im > 0) else im.max()

        plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.axis('off')
        # plt.show()
        
    plt.show()

# plot_chromatic_PSF_slice(PSF_0, wavelengths, norms, window_size=40)
# plot_chromatic_PSF_slice(PSF_0, wavelengths, window_size=100, slices=['horizontal'])

#%%
from tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis

# wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)
wvl_weights = torch.linspace(0.5, 1.0, N_wvl).to(device).view(1, N_wvl, 1, 1)
wvl_weights = N_wvl / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

# wvl_weights = wvl_weights * 0 + 1

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
    diff = (func(x_)-PSF_0) #* wvl_weights
    w = 2e4
    MSE_loss = diff.pow(2).mean() * w * w_MSE
    MAE_loss = diff.abs().mean()  * w * w_MAE
    LO_loss  = loss_LO_fn() if PSF_model.LO_NCPAs else 0.0

    return MSE_loss + MAE_loss + LO_loss + Moffat_loss_fn()

suppress_sausage = False
suppress_LO = False

suppress_sausage = 1e3 if suppress_sausage else 1
suppress_LO = 1e3 if suppress_LO else 1

def loss_LO_fn():
    if isinstance(PSF_model.LO_basis, PixelmapBasis):
        # LO_loss = grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5
        raise NotImplementedError("Gradient loss for PixelmapBasis is not implemented yet.")
        
    elif isinstance(PSF_model.LO_basis, ZernikeBasis) or isinstance(PSF_model.LO_basis, ArbitraryBasis):
        LO_loss = PSF_model.inputs_manager['LO_coefs'].pow(2).sum(-1).mean() * 1e-7 * suppress_LO
        # Constraint to enforce first element of LO_coefs to be positive
        first_coef_penalty = torch.clamp(-PSF_model.inputs_manager['LO_coefs'][:, 0], min=0).pow(2).mean() * 5e-5 * suppress_sausage
        LO_loss += first_coef_penalty
        
    return LO_loss


def Moffat_loss_fn():
    if PSF_model.Moffat_absorber is False:
        return 0.0
    
    amp = PSF_model.inputs_manager['amp']
    # alpha = PSF_model.inputs_manager['alpha']
    # beta = PSF_model.inputs_manager['beta']
    # b = PSF_model.inputs_manager['b']
    
    # Enforce positive amplitude
    amp_penalty = amp.pow(2).mean() * 2.5e-2
    
    # Enforce beta > 1.5
    # beta_penalty = torch.clamp(1.5 - beta, min=0).pow(2).mean() * 1e-3
    
    # # Enforce alpha > 0
    # alpha_penalty = torch.clamp(-alpha, min=0).pow(2).mean() * 1e-3
    
    # # Enforce b > 0
    # b_penalty = torch.clamp(-b, min=0).pow(2).mean() * 1e-3
    
    return amp_penalty #+ beta_penalty + alpha_penalty + b_penalty


def loss_fn_Huber(x_):
    PSF_1 = func(x_)
    huber_loss = loss_Huber(PSF_1*wvl_weights*5e5, PSF_0*wvl_weights*5e5)
    MSE_loss = loss_MSE(PSF_1*wvl_weights, PSF_0*wvl_weights) * 2e4 * 800.0
    LO_loss = loss_LO_fn() if PSF_model.LO_NCPAs else 0.0

    return huber_loss + LO_loss #+ MSE_loss


loss_fn1 = lambda x_: loss_fn(x_, w_MSE=900.0, w_MAE=1.6)
# loss_fn2 = lambda x_: loss_fn(x_, w_MSE=1.0,   w_MAE=2.0)
# grad_loss_fn = GradientLoss(p=1, reduction='mean')


#%
# def loss_radial(x_):
#     PSF_1 = func(x_)
#     diff = (PSF_1-PSF_0) * wvl_weights
#     mse_loss = (diff * 4000).pow(2).mean()
#     mae_loss = (diff * 32000).abs().mean()
#     if fit_LO:
#         if isinstance(LO_basis, PixelmapBasis):
#             LO_loss = grad_loss_fn(PSF_model.OPD_func.unsqueeze(1)) * 5e-5
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
    if not include_list:
        raise ValueError('include_list is empty')
        
    PSF_model.inputs_manager.set_optimizable(include_list, True)
    PSF_model.inputs_manager.set_optimizable(exclude_list, False)

    x_backup = PSF_model.inputs_manager.stack().clone()
    
    lim_lambda = lambda method, tol: minimize(
        loss_fn, PSF_model.inputs_manager.stack(), 
        max_iter=max_iter, tol=tol, method=method, 
        disp=2 if verbose else 0
    )

    # Try L-BFGS first
    result = lim_lambda('l-bfgs', 1e-4)
    
    # Retry with BFGS if convergence issues detected
    if result['nit'] < max_iter * 0.3 or result['fun'] > 1: # If converged earlier than 30% of the iterations or final loss is too high
        if verbose:
            reason = "stopped too early" if result['nit'] < max_iter * 0.3 else "final loss is high"
            print(f"Warning: minimization {reason}. Perhaps, convergence wasn't reached? Trying BFGS...")
            
        _ = PSF_model.inputs_manager.unstack(x_backup, include_all=True, update=True)
        result = lim_lambda('bfgs', 1e-5)

    OPD_map = PSF_model.OPD_func(PSF_model.inputs_manager['LO_coefs']).detach().cpu().numpy() if PSF_model.LO_NCPAs else None
        
    if verbose:
        print('-'*50)

    return result.x, func(result.x), OPD_map

fit_wind_speed  = True
fit_outerscale  = True
fit_Cn2_profile = True
    
include_general = ['r0', 'dn'] + \
                  (['amp', 'alpha', 'b'] if PSF_model.Moffat_absorber else []) + \
                  (['LO_coefs'] if PSF_model.LO_NCPAs else []) + \
                  ([x+'_ctrl' for x in PSF_model.polychromatic_params] if PSF_model.use_splines else PSF_model.polychromatic_params) + \
                  (['L0'] if fit_outerscale else []) + \
                  (['wind_speed_single'] if fit_wind_speed else []) + \
                  (['Cn2_weights'] if fit_Cn2_profile else [])

exclude_general = ['ratio', 'theta', 'beta'] if PSF_model.Moffat_absorber else []

# include_LO = (['LO_coefs'] if PSF_model.LO_NCPAs else []) + (['chrom_defocus'] if PSF_model.chrom_defocus else [])
# exclude_LO = list(set(include_general + exclude_general) - set(include_LO))

# inc_minus_Moffat = list(set(include_general) - set(['amp', 'alpha', 'beta', 'b']))
# inc_only_Moffat = ['amp', 'alpha', 'beta', 'b']

#%%
x0, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 150)
# x0, PSF_1, OPD_map = minimize_params(loss_fn_Huber, include_general, exclude_general, 150)

# if fit_LO:
#     x1, PSF_1, OPD_map = minimize_params(loss_fn2, include_LO, exclude_LO, 50)
#     x2, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 50)

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

if PSF_model.LO_NCPAs:
    plt.imshow(OPD_map[id_src,...]*1e9)
    plt.colorbar()
    plt.show()


#%%
print('λ: ', *[f"{x:.0f}" for x in wavelengths.detach().cpu().numpy().flatten()*1e9])
print('r0:', *[f"{x:.3f}" for x in PSF_model.model.r0.detach().cpu().numpy().flatten()])
print('L0:', *[f"{x:.3f}" for x in PSF_model.model.L0.detach().cpu().numpy().flatten()])
print('dn:', *[f"{x:.3f}" for x in PSF_model.model.dn.detach().cpu().numpy().flatten()])
print('dx:', *[f"{x:.2f}" for x in PSF_model.model.dx.detach().cpu().numpy().flatten()])
print('dy:', *[f"{x:.2f}" for x in PSF_model.model.dy.detach().cpu().numpy().flatten()])
print('F: ', *[f"{x:.3f}" for x in PSF_model.model.F.detach().cpu().numpy().flatten()])
print('LO:', *[f"{x:.2f}" for x in PSF_model.inputs_manager['LO_coefs'].detach().cpu().numpy().flatten()])
print('wind_speed_single:', *[f"{x:.2f}" for x in PSF_model.inputs_manager['wind_speed_single'].detach().cpu().numpy().flatten()])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
wvl_plot = wavelengths.squeeze().cpu().numpy()*1e9

ax[0].plot(wvl_plot, PSF_model.model.F.detach().cpu().numpy().flatten())
ax[0].set_title('Flux factor')
ax[0].set_xlabel('Wavelength [nm]')
ax[0].set_xlim(485, 927.81)
ax[0].set_ylim(0.93, 1.05)
ax[1].plot(wvl_plot, PSF_model.model.dx.detach().cpu().numpy().flatten(), label='dx')
ax[1].plot(wvl_plot, PSF_model.model.dy.detach().cpu().numpy().flatten(), label='dy')
ax[1].legend()
ax[1].set_title('Position offset')
ax[1].set_xlabel('Wavelength [nm]')
ax[1].set_xlim(485, 927.81)
ax[1].set_ylim(-1.6, 0.2)

plt.show()


#%%
import torch.nn.functional as F

N_λ = len(wavelengths) // 8
W_orig = H_orig = 69

PSF_model.SetWavelengths(wavelengths[::8])  # in nm
PSF_model.SetImageSize(W_orig)

torch.cuda.empty_cache()

#%%
center_y, center_x = PSF_0.shape[-2] // 2, PSF_0.shape[-1] // 2
half_crop = H_orig // 2
ROI = slice( center_y - half_crop, center_y + half_crop+(PSF_0.shape[0] % 2) )

# Downsampling factor to reduce memory usage
pool_size = 3
W = H = W_orig // pool_size

PSF_0_pooled = F.max_pool2d(PSF_0[:, ::8, ROI, ROI], kernel_size=pool_size, stride=pool_size)

# TODO: test if PSF stays the same when the image size is shrunk down

#%%
def run_diff_small(x_):  
    """Compute PSF and apply maxpooling to reduce memory footprint."""
    psf = func(x_)
    # Apply 2D maxpooling: (N_src, N_λ, H, W) -> (N_src, N_λ, H//pool, W//pool)
    psf_pooled = F.max_pool2d(psf, kernel_size=pool_size, stride=pool_size)
    return psf_pooled.reshape(-1)

x_grad = x0.clone().detach().requires_grad_(True)

N_params = x_grad.shape[-1]

# Jacobian of PSF wrt parameters: shape (N, P)
J = torch.autograd.functional.jacobian(
    run_diff_small,
    x_grad,
    vectorize=True   # if supported; otherwise drop this argument
)   # (N, P)
torch.cuda.empty_cache()

#%%
J_per_param = PSF_model.inputs_manager.inputs_transformer.unstack(J.squeeze())
for k, v in J_per_param.items():
    J_per_param[k] = J_per_param[k].reshape(N_λ, H, W, -1)

#%%
# Downsample target PSF for comparison
res_vec = (run_diff_small(x_grad).reshape_as(PSF_0_pooled) - PSF_0_pooled).reshape(-1)   # (N,)
res_vec = res_vec.reshape(N_λ, H, W).unsqueeze(-1)  # (1, N_λ, H, W)

sensitivity_map = {}
J_per_param = PSF_model.inputs_manager.inputs_transformer.unstack(J.squeeze())
for k, v in J_per_param.items():
    J_per_param[k] = J_per_param[k].reshape(N_λ, H, W, -1)
    # sensitivity_map[k] = res_vec * J_per_param[k]
    sensitivity_map[k] = J_per_param[k]

#%%

# List all shapes of sensitivity maps
for k, v in J_per_param.items():
    print(f"{k:15s}: {v.shape}")

#%%

studied_param = 'dn'

final_map = sensitivity_map[studied_param][0,...,0].cpu().detach().numpy()
final_map -= final_map.mean()

plt.imshow(final_map, cmap='seismic', vmin=-np.percentile(np.abs(final_map), 99), vmax=np.percentile(np.abs(final_map), 99))
plt.colorbar()
plt.title(rf"Sensitivity map $\partial$PSF/$\partial${studied_param}")
plt.show()


#%%
from data_processing.MUSE_STD_dataset_utils import GetROIaroundMax, GetSpectrum

diff_im = (PSF_1-PSF_0).abs()[0,...].squeeze().cpu().numpy()

_, _, max_id  = GetROIaroundMax(diff_im.mean(0).squeeze(), 10) # TODO: this function is redundant
spectrum_diff = GetSpectrum(diff_im, max_id, radius=2)
_, _, max_id  = GetROIaroundMax(PSF_0[0,...].squeeze().cpu().numpy().mean(0), 10) # TODO: this function is redundant
spectrum_data = GetSpectrum(PSF_0[0,...].squeeze().cpu().numpy(), max_id, radius=5)

plt.plot(wavelengths.squeeze().cpu().numpy()*1e9, spectrum_diff / spectrum_data * 100)

# %%
from tools.plotting import plot_radial_PSD_profiles

plot_radial_PSD_profiles(PSF_model.model)

#%%
if PSF_model.use_splines:
    N_HD_bins = 20
    λ_min, λ_max = wavelengths.min().item()*1e9, wavelengths.max().item()*1e9
    x_ctrl = PSF_model.inputs_manager['F_x_ctrl'].squeeze()
    λ_ctrl = x_ctrl * (λ_max - λ_min) + λ_min
        
    A = PSF_model.evaluate_splines('F', torch.linspace(0, 1, N_HD_bins).to(device)).squeeze(-1)
    C = PSF_model.evaluate_splines('F', x_ctrl).squeeze(-1)
    
    plt.plot(np.linspace(λ_min, λ_max, N_HD_bins), A.squeeze().detach().cpu().numpy())
    plt.scatter(λ_ctrl.cpu(), C.squeeze().detach().cpu().numpy())
    # plt.plot(λ_ctrl, A.squeeze().detach().cpu().numpy(), 'o', label='Control Points')
    
else:
    plt.plot(wavelengths.squeeze().cpu().numpy()*1e9, PSF_model.F.squeeze().detach().cpu().numpy())
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
PSF_model.inputs_manager['F'] = torch.tensor([[1.0]*N_wvl], device=device)

x_tmp = PSF_model.inputs_manager.stack()

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
