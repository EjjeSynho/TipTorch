#%%
%reload_ext autoreload
%autoreload 2

import sys

sys.path.append('../')

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LogNorm
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack, plot_chromatic_PSF_slice
from datasets.MUSE_STD_dataset.STD_dataset_utils import STD_FOLDER, LoadSTDStarCache, KC_MISMATCH_FLAG, KC_FRACTION_FEATURE, KC_GROUP_FEATURE, apply_kc_fraction, kc_fraction_from_config
# The fitting regime (model flags, loss, priors, multi-start fit) is the one of STD_fitter.py: this script only picks the samples, builds the
# model and the objective through the fitter's functions and looks at the result. Override a fitter flag here before building the model
# (e.g. fitter.WING_WIDTH_T_PX = 3.0, fitter.DEFOCUS_STARTS = (20.0,), fitter.PSF_LOSS_CFG['peak_weight'] = 0.0) to test a change
from datasets.MUSE_STD_dataset import STD_fitter as fitter
from tiptorch._config import default_device


#%%
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

#%%
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 5), a_min=0, a_max=N_wvl_max-1)
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 3), a_min=0, a_max=N_wvl_max-1)
wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 2), a_min=0, a_max=N_wvl_max-1)
# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=31)+1, 2), a_min=0, a_max=N_wvl_max-1)[:14]

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
# ids = 174 # fail converge wth Cn2, strong wind

# ids = 428 # good one, no DM mismathc yet
# ids = 434 # good one, no DM mismathc yet

# The 'DM correction mismatch' samples below are flagged 'k_c mismatch' in muse_df (k_c ~93 % of the nominal); their configs get the scaled DM pitch automatically
# ids = 440 # good one, but DM correction mismatch
# ids = 449 # good one, but DM correction mismatch
# ids = 446 # good one, but DM correction mismatch
# ids = 451 # good one, but DM correction mismatch
# ids = 453 # good one, but DM correction mismatch

# ids = 338 # poor blue fitting
# ids = 457 # poor blue fitting
ids = 477 # poor blue fitting
# ids = 467 # poor blue fitting
# ids = 458 # poor blue fitting
# ids = 494 # poor blue fitting

# ids = 482 # good one
# ids = 475 # good one, mb DM correction radius mismatch
# ids = 468 # good one
# ids = 462 # good one
# ids = 455 # good one, slight sausage
# ids = 438 # good one, slight sausage

# ids = 359 # good one
# ids = 368 # kinda bad in blue
# ids = 511 # bad in blue

# ids = [206, 359, 462, 468, 475]
# ids = 206
# ids = 96
# ids = 456 # large tomographic error
# ids = 125

# Several OBs stacked without derotation: the data stay in the raw frame and the model rotates its pupil (spider streaks), its DM / WFS
# frame (tomography, AO cut-off) and its static phase (phase bump, NCPAs) per OB by the header pupil angle of the config. The samples
# are phase-bump stars ('Non-point' in muse_df) with visible streaks and good quality, spread over the pupil angle, so that the rotation
# of the bump and of the tomographic residual relative to the streaks can be checked in the plots below (see the rotation check cell)
# ids = [462, 144, 344, 109, 525]
DEROTATE_PSF = False

# Samples recorded with fewer corrected DM modes are flagged 'k_c mismatch' (see datasets/MUSE_STD_dataset/kc_mismatch.py):
# LoadSTDStarCache scales the DM pitch of their configs so that the model's k_c = 'k_c fraction' x nominal. With APPLY_KC_CORRECTION = False
# every sample is fitted with the nominal k_c = 1 / (2 x 0.22 m) instead (same flag as in STD_fitter.py). KC_FRACTION forces the model k_c
# (relative to the nominal one) for all the ids, overriding both. One model holds a single k_c, so without a forced value all the ids
# of one model must belong to the same 'k_c group' (0 = nominal)
APPLY_KC_CORRECTION = True
KC_FRACTION = None

ids_ = ids if isinstance(ids, list) else [ids]

print(muse_df.loc[ids_, [KC_MISMATCH_FLAG, KC_FRACTION_FEATURE, KC_GROUP_FEATURE, 'Pupil angle']])
if APPLY_KC_CORRECTION and KC_FRACTION is None and muse_df.loc[ids_, KC_GROUP_FEATURE].nunique() > 1:
    raise ValueError(f"The ids {ids_} belong to different k_c groups {muse_df.loc[ids_, KC_GROUP_FEATURE].tolist()}: fit them in separate models")

PSF_0, norms, bgs, configs = LoadSTDStarCache(
    ids                 = ids,
    derotate_PSF        = DEROTATE_PSF,
    normalize           = True,
    subtract_background = True,
    wvl_ids             = wvl_ids,
    ensure_odd_pixels   = True,
    device              = default_device
)

# The configs carry the k_c of the samples' labels (scaled DM pitch): rescale them to the nominal / forced k_c when asked
kc_fraction = KC_FRACTION if KC_FRACTION is not None else (None if APPLY_KC_CORRECTION else 1.0)
if kc_fraction is not None:
    configs = [apply_kc_fraction(c, kc_fraction / kc_fraction_from_config(c)) for c in configs]

N_src, N_wvl = PSF_0.shape[:2]

#%%
PSF_model = fitter.build_PSF_model(configs, default_device)
wavelengths = PSF_model.λ_sim

print(f"Model k_c = {100 * kc_fraction_from_config(configs[0]):.1f} % of the nominal ({PSF_model.model.kc:.3f} 1/m, "
      f"DM pitch {np.asarray(configs[0]['DM']['DmPitchs']).ravel()[0]:.4f} m)")
print(f"chromatic defocus: {fitter.CHROM_DEFOCUS} | anisotropic jitter: {fitter.ANISOTROPIC_JITTER} | wing: {fitter.WING_KERNEL} "
      f"({fitter.WING_WIDTH_PX} x {fitter.WING_WIDTH_T_PX} px) | defocus starts {fitter.DEFOCUS_STARTS} nm selected by {fitter.SELECT_BY}")

#%%
cmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
cmap.set_bad(color='black')

for j in range(N_src):
    for i in range(N_wvl):
        im = PSF_0[j,i,...].cpu().numpy()
        vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
        vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()

        plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.axis('off')
        # plt.show()

    plt.show()

# plot_chromatic_PSF_slice(PSF_0, wavelengths, norms, window_size=40)
# plot_chromatic_PSF_slice(PSF_0, wavelengths, window_size=100, slices=['horizontal'])

#%%
# Objective and parameter sets of the fitter: quality flags of the sample (blurry: bump suppressed, LO coefficients and wing fraction pulled
# to the dataset priors, L0 / wind speed / Cn2 at the config values; no 'Non-point' flag: bump suppressed), PSF loss (fitter.PSF_LOSS_CFG) +
# LO regularization + MAP priors (fitter.PRIOR_STD), L-BFGS with a BFGS retry
flags = fitter.sample_fit_flags(ids_)
print('fit flags:', flags)

func, loss_fn    = fitter.build_loss(PSF_model, PSF_0, flags, default_device)
minimize_params  = fitter.make_minimizer(PSF_model, func, loss_fn)
include, exclude = fitter.parameter_sets(PSF_model, flags)
print('fitted:', sorted(include), '| frozen:', sorted(exclude))

#%%
# Multi-start fit (fitter.DEFOCUS_STARTS, selection by fitter.SELECT_BY, even-mode sign convention fitter.DEFOCUS_SIGN per start)
fit = fitter.multistart_fit(PSF_model, PSF_0, loss_fn, minimize_params, include, exclude, flags, verbose=True)
x0, PSF_1, OPD_map = fit['x'], fit['PSF'], fit['OPD_map']

print(f"kept the start at {fit['start']:+.0f} nm by {fitter.SELECT_BY}: loss {fit['loss']:.4f}, median |peak err| {100 * fit['peak_error']:.2f} %, "
      f"defocus {fit['defocus']:+.1f} nm, even modes flipped {fit['flipped']}")

for name in ['J_angle'] if PSF_model.anisotropic_jitter else []:
    print(f'{name}:', *[f'{v:.2f}' for v in PSF_model.inputs_manager[name].detach().cpu().numpy().flatten()])
if PSF_model.chrom_defocus:
    print('chromatic defocus at the spline nodes [nm]:', *[f'{v:.1f}' for v in PSF_model.inputs_manager['chrom_defocus_ctrl'].detach().cpu().numpy().flatten()])
if PSF_model.wing_kernel:
    print('wing fraction at the spline nodes:', *[f'{abs(v):.4f}' for v in PSF_model.inputs_manager['wing_frac_ctrl'].detach().cpu().numpy().flatten()],
          '| widths [px]:', *[f'{abs(PSF_model.inputs_manager[k][0].item()):.2f}' for k in ('wing_width', 'wing_width_t')],
          '| angle [deg]:', f"{PSF_model.inputs_manager['wing_angle'][0].item():.1f}")

#%%
from tools.plotting import plot_radial_PSF_profiles

for id_src, dataset_index in enumerate(ids_):
    
    if PSF_model.model.N_src > 1:
        print("Plotting results for dataset index:", dataset_index)
    
    vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
    vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)
    wvl_select = np.s_[0, N_wvl//2, -1]

    draw_PSF_stack(
        PSF_0.cpu().numpy()[id_src, wvl_select, ...],
        PSF_1.cpu().numpy()[id_src, wvl_select, ...],
        average = True,
        min_val = vmin,
        max_val = vmax,
        crop = 100
    )

    PSF_disp = lambda x, w: (x[id_src,w,...]).cpu().numpy()

    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_PSF_profiles(
            PSF_disp(PSF_0, lmbd),
            PSF_disp(PSF_1, lmbd),
            'Data',
            'TipTorch',
            cutoff = 40,
            y_min = 3e-2,
            linthresh = 1e-2,
            return_profiles = True,
            ax = ax[i]        
        )
    plt.show()

    if PSF_model.LO_NCPAs:
        plt.imshow(OPD_map[id_src,...]*1e9)
        plt.colorbar()
        plt.show()

#%%
# Rotation check per OB (raw frames): the data and the model at the red end, where the streaks are the most visible, the model's
# oriented pupil and the fitted static OPD (phase bump + NCPAs) rotated to the image frame as the model applies it. The streaks of the
# model must follow the data's, and the bump, attached to the DM frame, must turn with the pupil: a bump amplitude (first LO
# coefficient) fitted positive and comparable across OBs means the rotated bump matches the data
if PSF_model.LO_NCPAs:
    m = PSF_model.model
    angles  = m.pupil_angle.expand(N_src).cpu().numpy()
    pupils  = m.pupil_rot.expand(N_src, -1, -1).cpu().numpy()
    OPD_rot = PSF_model.LO_basis.rotate_OPD(PSF_model.OPD_func(PSF_model.inputs_manager['LO_coefs']).unsqueeze(1))[:, 0].detach().cpu().numpy() * 1e9 # [N_src, H, W] nm
    bump    = PSF_model.inputs_manager['LO_coefs'][:, 0].detach().cpu().numpy()
    lim     = np.abs(OPD_rot[j]).max()

    fig, ax = plt.subplots(N_src, 4, figsize=(15, 3.6*N_src), squeeze=False)
    for j in range(N_src):
        D, M = PSF_0[j, -1].cpu().numpy(), PSF_1[j, -1].detach().cpu().numpy()
        norm = LogNorm(vmin=np.percentile(D[D > 0], 10), vmax=D.max())
        ax[j, 0].imshow(D, cmap=cmap, norm=norm)
        ax[j, 0].set_title(f'ID {ids_[j]}: data, {wavelengths.flatten()[-1].item()*1e9:.0f} nm, pupil angle {angles[j]:.1f}°', fontsize=9)
        ax[j, 1].imshow(M, cmap=cmap, norm=norm)
        ax[j, 1].set_title('model', fontsize=9)
        ax[j, 2].imshow(pupils[j], cmap='gray')
        ax[j, 2].set_title('oriented pupil of the model', fontsize=9)
        ax[j, 3].imshow(OPD_rot[j] * (pupils[j] > 0), cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax[j, 3].set_title(f'static OPD [nm] (bump coef. {bump[j]:.1f})', fontsize=9)
        
        for a in ax[j]:
            a.axis('off')
            
    plt.tight_layout()
    plt.show()

    print('Pupil angles [deg]:', *[f'{a:.1f}' for a in angles])
    print('Phase bump coefs.: ', *[f'{b:.1f}' for b in bump])

#%
# print('λ: ', *[f"{x:.0f}" for x in wavelengths.detach().cpu().numpy().flatten()*1e9])
# print('r0:', *[f"{x:.3f}" for x in PSF_model.model.r0.detach().cpu().numpy().flatten()])
# print('L0:', *[f"{x:.3f}" for x in PSF_model.model.L0.detach().cpu().numpy().flatten()])
# print('dn:', *[f"{x:.3f}" for x in PSF_model.model.dn.detach().cpu().numpy().flatten()])
# print('dx:', *[f"{x:.2f}" for x in PSF_model.model.dx.detach().cpu().numpy().flatten()])
# print('dy:', *[f"{x:.2f}" for x in PSF_model.model.dy.detach().cpu().numpy().flatten()])
# print('F: ', *[f"{x:.3f}" for x in PSF_model.model.F.detach().cpu().numpy().flatten()])
# print('LO:', *[f"{x:.2f}" for x in PSF_model.inputs_manager['LO_coefs'].detach().cpu().numpy().flatten()])
# print('wind_speed_single:', *[f"{x:.2f}" for x in PSF_model.inputs_manager['wind_speed_single'].detach().cpu().numpy().flatten()])

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# wvl_plot = wavelengths.squeeze().cpu().numpy()*1e9

# ax[0].plot(wvl_plot, PSF_model.model.F.detach().cpu().numpy().flatten())
# ax[0].set_title('Flux factor')
# ax[0].set_xlabel('Wavelength [nm]')
# ax[0].set_xlim(485, 927.81)
# ax[0].set_ylim(0.93, 1.05)
# ax[1].plot(wvl_plot, PSF_model.model.dx.detach().cpu().numpy().flatten(), label='dx')
# ax[1].plot(wvl_plot, PSF_model.model.dy.detach().cpu().numpy().flatten(), label='dy')
# ax[1].legend()
# ax[1].set_title('Position offset')
# ax[1].set_xlabel('Wavelength [nm]')
# ax[1].set_xlim(485, 927.81)
# ax[1].set_ylim(-1.6, 0.2)

# plt.show()


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
from data_processing.MUSE.STD_dataset.STD_dataset_utils import GetROIaroundMax, GetSpectrum

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
        
    A = PSF_model.evaluate_splines('F', torch.linspace(0, 1, N_HD_bins).to(default_device)).squeeze(-1)
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
PSF_model.inputs_manager['F'] = torch.tensor([[1.0]*N_wvl], device=default_device)

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

_ = PSF_model.model.ErrorBudget(True)


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
