#%%
%reload_ext autoreload
%autoreload 2

# *****************************************************************************************************
# ********************* FIX BACK PUPIL TRANSPOSE IN TIPTORCH  **********************************
# *****************************************************************************************************
# *****************************************************************************************************

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm, SymLogNorm
from photutils.centroids import centroid_2dg
from torchmin import minimize

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tiptorch.PSF_models.HARMONI_wrapper import PSFModelHARMONI
from tiptorch.managers.config_manager import ConfigManager
from tiptorch._config import default_device, default_torch_type
from tiptorch.tools.utils import BinCn2, mask_circle
from tools.plotting import plot_radial_PSF_profiles
from astropy.io import fits

#%%
N_pix    = 151  # Desired number of pixels in the final PSF cube (N_pix x N_pix)
# N_layers = 10  # Number of binned atmospheric layers
N_layers = None
N_λ_bins = 40  # Number of spectral bins; set to None to use Δλ_bin

#%%
pupil_path = '/home/aosimul/akuznets/Data/HARMONI/pupils/EELT480pp0.0803m_obs0.283_spider2023.fits'

with fits.open(pupil_path) as hdul:
    pupil_data = hdul[1].data

plt.imshow(pupil_data, cmap='gray', origin='lower')
plt.axis('off')
plt.show()

#%%
def fft_propagate_pupil_to_focal(pupil):
    pupil = np.asarray(pupil, dtype=np.complex64)
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    focal_intensity = np.abs(focal_field) ** 2
    return focal_intensity / focal_intensity.sum()


PSF_fft = fft_propagate_pupil_to_focal(pupil_data)

plt.figure(figsize=(5, 5))
plt.imshow(np.log10(np.clip(PSF_fft, 1e-12, None)), origin='lower', cmap='inferno')
plt.title('Minimal FFT PSF from pupil_data')
plt.colorbar(label='log10 intensity')
plt.tight_layout()
plt.show()

#%%
# >>>>>>> Load simulated PSF cube for comparison
# cube_path = '/home/aosimul/akuznets/Data/HARMONI/simulated_cubes_6mas/01_Feige110_MCAO_6MAS_Iz_CLEAR_ERM_DATACUBE.fits'
cube_path = '/home/aosimul/akuznets/Data/HARMONI/simulated_cubes_6mas/01_Feige110_MCAO_6MAS_H_CLEAR_ERM_DATACUBE.fits'

with fits.open(cube_path) as hdul:
    PSF_0 = hdul[1].data  # Shape: (N_wvl, N_pix, N_pix) or (N_wvl, N_y, N_x)
    PSF_header = hdul[1].header
    
print(f"\nLoaded PSF cube from {cube_path}")
print(f"PSF cube shape: {PSF_0.shape}")
print(f"Number of wavelengths: {PSF_0.shape[0]}")

Δλ        = PSF_header['CD3_3']  / 10.0
λ_ref_pix = PSF_header['CRPIX3'] / 10.0
λ_ref_val = PSF_header['CRVAL3'] / 10.0

pix_idx = np.arange(PSF_0.shape[0]) + 1  # FITS pixel indices are 1-based
λ_full = λ_ref_val + (pix_idx - λ_ref_pix) * Δλ


valid_slices = slice(212, -212)

# Crop border slices:
PSF_0  = PSF_0[valid_slices, ...]
λ_fits = λ_full[valid_slices]

print(f"Wavelength range: {λ_fits[0]:.1f} to {λ_fits[-1]:.1f} nm")


def bin_spectral_cube(cube, wavelengths, n_bins=None, Δλ_bin=None):
    """
    Sum neighboring spectral slices into approximately equal-width bins.

    As in the MUSE-NFM cube processing, each output image is the NaN-safe sum of its input planes, its wavelength is the mean wavelength of the
    chunk, and the per-pixel spectral scatter is retained. If both controls are provided, n_bins takes precedence.
    """
    cube = np.asarray(cube)
    wavelengths = np.asarray(wavelengths, dtype=float)

    if cube.ndim != 3:
        raise ValueError(f"Expected a [wavelength, y, x] cube, got {cube.shape}.")
    
    if wavelengths.ndim != 1 or len(wavelengths) != cube.shape[0]:
        raise ValueError("Wavelengths must be one-dimensional and match the cube.")
    
    if len(wavelengths) < 2:
        raise ValueError("At least two wavelength slices are required for binning.")
    
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be strictly increasing.")

    if n_bins is None and Δλ_bin is None:
        return cube, None, wavelengths, None

    spectral_span = wavelengths[-1] - wavelengths[0]
    
    if n_bins is None:
        if Δλ_bin <= 0:
            raise ValueError("Δλ_bin must be positive.")
        
        n_bins = max(int(round(spectral_span / float(Δλ_bin))), 1)

    n_bins = int(n_bins)
    
    if n_bins < 1 or n_bins > len(wavelengths):
        raise ValueError(f"N_bins must be between 1 and {len(wavelengths)}, got {n_bins}.")

    bin_ids = np.rint(np.linspace(0, len(wavelengths), n_bins + 1)).astype(int)
    bin_ids[0], bin_ids[-1] = 0, len(wavelengths)

    data_binned = np.empty((n_bins, *cube.shape[1:]), dtype=cube.dtype)
    STD_binned  = np.empty_like(data_binned)
    wavelengths_binned = np.empty(n_bins, dtype=float)

    for bin_id, (start, stop) in enumerate(zip(bin_ids[:-1], bin_ids[1:])):
        chunk = cube[start:stop]
        data_binned[bin_id] = np.nansum(chunk, axis=0)
        STD_binned [bin_id] = np.nanstd(chunk, axis=0)
        wavelengths_binned[bin_id] = np.nanmean(wavelengths[start:stop])

    native_Δλ = np.median(np.diff(wavelengths))
    wavelength_edges = np.empty(n_bins + 1, dtype=float)
    wavelength_edges[0]  = wavelengths[0]  - native_Δλ / 2
    wavelength_edges[-1] = wavelengths[-1] + native_Δλ / 2
    
    for edge_id, sample_id in enumerate(bin_ids[1:-1], start=1):
        wavelength_edges[edge_id] = (wavelengths[sample_id - 1] + wavelengths[sample_id]) / 2

    return data_binned, STD_binned, wavelengths_binned, wavelength_edges


# Find PSF cube center using maximum position
n_wvl, n_y, n_x = PSF_0.shape
centroid = centroid_2dg( PSF_0.mean(axis=0) )
center_x, center_y = int(round(centroid[0])), int(round(centroid[1]))

# Preserved under different names since center_y/center_x get reassigned later on:
# needed to embed the fitted cutout back into the original (raw) pixel grid.
orig_frame_shape  = (n_y, n_x)
orig_frame_center = (center_y, center_x)

print(f"\nPSF cube center (from maximum): ({center_y}, {center_x})")
print(f"PSF cube original shape: {PSF_0.shape}")


# Function to embed PSF into a new cube with specified dimensions
def embed_PSF_in_new_cube(PSF_cube, target_n_y, target_n_x, center_y=None, center_x=None):
    """
    Embed PSF cube into a new spectral cube with specified dimensions.
    The PSF is centered in the new cube.
    
    Parameters:
    -----------
    psf_cube : ndarray
        Input PSF cube with shape (N_wvl, N_y, N_x)
    target_n_y : int
        Target number of pixels in y dimension
    target_n_x : int
        Target number of pixels in x dimension
    center_y : int, optional
        Y-coordinate of PSF center. If None, uses PSF maximum.
    center_x : int, optional
        X-coordinate of PSF center. If None, uses PSF maximum.
    
    Returns:
    --------
    new_cube : ndarray
        New PSF cube with shape (N_wvl, target_n_y, target_n_x)
    """
    n_wvl, n_y, n_x = PSF_cube.shape
    
    # Use provided center or find maximum position
    if center_y is None or center_x is None:
        # Find the maximum position using photutils
        psf_0 = PSF_cube[0]
        centroid = centroid_2dg(psf_0)
        center_y, center_x = int(round(centroid[0])), int(round(centroid[1]))
    
    # Calculate center positions in new cube
    center_y_new, center_x_new = target_n_y // 2, target_n_x // 2
    
    # Calculate offsets
    offset_y = center_y_new - center_y
    offset_x = center_x_new - center_x
    
    # Create new cube filled with zeros
    new_cube = np.zeros( (n_wvl, target_n_y, target_n_x), dtype=PSF_cube.dtype)
    
    # Calculate source and destination slices
    src_y_start = max(0, -offset_y)
    src_x_start = max(0, -offset_x)
    dst_y_start = max(0, offset_y)
    dst_x_start = max(0, offset_x)
    
    src_y_end = min(n_y, target_n_y - offset_y)
    src_x_end = min(n_x, target_n_x - offset_x)
    dst_y_end = min(target_n_y, n_y + offset_y)
    dst_x_end = min(target_n_x, n_x + offset_x)
    
    # Copy the overlapping region
    new_cube[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = PSF_cube[:, src_y_start:src_y_end, src_x_start:src_x_end]
    
    return new_cube


# Embed PSF cube into new dimensions (N_wvl x N_pix x N_pix)
print("\nEmbedding PSF cube into new dimensions...")
target_n_y, target_n_x = N_pix, N_pix
PSF_0_full = embed_PSF_in_new_cube(PSF_0, target_n_y, target_n_x, center_y, center_x)
print(f"Embedded PSF cube shape: {PSF_0_full.shape}")

# Generate non-zero pixels mask for embedded PSF cube (full valid footprint, single star in FoV)
PSF_mask = np.any(PSF_0_full != 0, axis=0)

# Background-only mask: valid footprint excluding the central PSF core (r < 30 px)
center_y, center_x = PSF_mask.shape[0] // 2, PSF_mask.shape[1] // 2
central_mask = mask_circle(PSF_mask.shape[0], 30, center=(center_y, center_x), centered=False)
PSF_mask_bg = PSF_mask & ~central_mask.astype(bool)

# Compute median noise (additive background offset) from background-only pixels
valid_pixels = PSF_0_full[:, PSF_mask_bg]
median_noise = np.median(valid_pixels, axis=1)

# Subtract the background only within the valid footprint, keeping the zero-padded border untouched
PSF_0_full = np.where(PSF_mask[np.newaxis, :, :], PSF_0_full - median_noise[:, np.newaxis, np.newaxis], 0.0)

# Normalize each spectral slice to unit flux over the full footprint (core + wings), not just the background
valid_sums = np.sum(PSF_0_full, axis=(1, 2))
PSF_0_full /= valid_sums[:, np.newaxis, np.newaxis]

PSF_binned, _, _, λ_bin_edges = bin_spectral_cube(PSF_0_full, λ_fits, n_bins=N_λ_bins)
PSF_binned /= PSF_binned.sum(axis=(1, 2), keepdims=True)
λ_binned = np.round(λ_bin_edges[:-1] + np.diff(λ_bin_edges) / 2, 1)

bin_widths = np.diff(λ_bin_edges)
print(f"Spectrally binned cube shape: {PSF_binned.shape}")
print(
    f"Spectral bins: {len(λ_binned)}; "
    f"Δλ_bin = {np.median(bin_widths):.2f} nm "
    f"({bin_widths.min():.2f}-{bin_widths.max():.2f} nm)"
)

sparse_ids = np.round(np.linspace(0, len(λ_binned) - 1, 8)).astype(int)
print(f"\nSparse wavelength indices: {sparse_ids}")
print(f"Sparse wavelengths: {λ_binned[sparse_ids]} nm ({len(sparse_ids)} slices)")

PSF_0 = PSF_binned[sparse_ids, ...]
PSF_0 /= PSF_0.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]  # Normalize each slice to unit flux

λ_sparse = λ_binned[sparse_ids]
N_λ_sparse = len(λ_sparse)

plt.imshow( np.log(np.clip(np.abs(PSF_0[0]), 5e-6, None)), cmap='viridis', origin='lower')
plt.scatter(target_n_x // 2, target_n_y // 2, color='red', marker='x', label='Original Center')
plt.show()

#%%
def ConfigInit(N_layers=None, verbose=False):
    config_manager = ConfigManager()
    config_torch = config_manager.Load('/home/aosimul/akuznets/Data/HARMONI/HARMONI_MCAO_Med.ini')
    config_torch = config_manager.Convert(config_torch, framework='pytorch', device=default_device, dtype=default_torch_type)

    config_torch['sources_science']['Wavelength']   = torch.tensor(λ_sparse, device=default_device).view(1,-1) * 1e-9 # [m]
    config_torch['sensor_science']['PixelScale']    = 6.0
    config_torch['sensor_science']['FieldOfView']   = N_pix
    config_torch['telescope']['PupilAngle']         = torch.tensor(22.0, device=default_device)  # [deg]
    config_torch['telescope']['PathPupil']          = '/home/aosimul/akuznets/Data/HARMONI/pupils/EELT480pp0.0803m_obs0.283_spider2023.fits'
    config_torch['telescope']['PathStaticOn']       = '/home/aosimul/akuznets/Data/HARMONI/pupils/ELT_M1_MORFEO_DMs_static_wfe_480px.fits'
    config_torch['DM']['NumberReconstructedLayers'] = N_layers

    if N_layers is not None:
        # >>>>>>> Perform Cn2 layers binning
        Cn2_weights     = config_torch['atmosphere']['Cn2Weights'].flatten()
        layer_altitudes = config_torch['atmosphere']['Cn2Heights'].flatten()
        wind_speed      = config_torch['atmosphere']['WindSpeed'].flatten()
        wind_direction  = config_torch['atmosphere']['WindDirection'].flatten()

        Cn2_binned, h_binned, wind_direction_binned, wind_speed_binned = BinCn2(
            Cn2_weights,
            layer_altitudes,
            wind_direction = wind_direction,
            wind_speed = wind_speed,
            N_binned = N_layers,
        )

        # Update config with binned Cn2 and wind profiles
        config_torch['atmosphere']['Cn2Weights']    = Cn2_binned.unsqueeze(0)  # Add batch dimension
        config_torch['atmosphere']['Cn2Heights']    = h_binned.unsqueeze(0)    # Add batch dimension
        config_torch['atmosphere']['WindSpeed']     = wind_speed_binned.unsqueeze(0)
        config_torch['atmosphere']['WindDirection'] = wind_direction_binned.unsqueeze(0)

    else:
        N_layers = config_torch['atmosphere']['Cn2Weights'].shape[-1]

    if verbose:
        print(f"Simulated wavelengths:  {λ_sparse.tolist()} nm ({len(λ_sparse)} slices)")
        print(f"Cn2 heights:     {[int(round(x)) for x in config_torch['atmosphere']['Cn2Heights'].flatten().cpu().numpy().tolist()]}")
        print(f"Cn2 weights:     {[round(x, 2)   for x in config_torch['atmosphere']['Cn2Weights'].flatten().cpu().numpy().tolist()]}")
        print(f"Wind speeds:     {[round(x, 2)   for x in config_torch['atmosphere']['WindSpeed'].flatten().cpu().numpy().tolist()]}")
        print(f"Wind directions: {[round(x, 2)   for x in config_torch['atmosphere']['WindDirection'].flatten().cpu().numpy().tolist()]}")

        # Display pupil and static WFE in a 1x2 subplot
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

        with fits.open(config_torch['telescope']['PathPupil']) as hdul:
            im1 = axes[0].imshow(hdul[1].data, cmap='gray', origin='lower')
            axes[0].set_title('ELT Pupil')
            axes[0].set_xlabel('Pixel')
            axes[0].set_ylabel('Pixel')
            plt.colorbar(im1, ax=axes[0], label='Pupil Transmission')

        with fits.open(config_torch['telescope']['PathStaticOn']) as hdul_static:
            im2 = axes[1].imshow(hdul_static[0].data, cmap='jet', origin='lower')
            axes[1].set_title('ELT Static WFE (MORFEO DMs)')
            axes[1].set_xlabel('Pixel')
            axes[1].set_ylabel('Pixel')
            plt.colorbar(im2, ax=axes[1], label='WFE [nm]')

        plt.tight_layout()
        plt.show()
        
    return config_torch


#%
print("Initializing the TipTorch HARMONI wrapper...")
model = PSFModelHARMONI(
    config = ConfigInit(N_layers=N_layers, verbose=True),
    LO_NCPAs = True,
    use_Zernike = True,
    use_static_WFE = True,
    Z_mode_max = 3,
    use_Moffat = False,
    N_spline_nodes = 5,
    device = default_device,
    retain_PSDs = False,
    dtype = default_torch_type,
    λ_min = float(λ_full[0]  * 1e-9),
    λ_max = float(λ_full[-1] * 1e-9),
    num_λ_slices=len(λ_full),
)
tiptorch_model = model.model

# tiptorch_model.pupil = tiptorch_model.pupil.T

model.inputs_manager.delete('wind_speed_single')
model.inputs_manager.delete('wind_dir_single')

# model.inputs_manager.set_optimizable(['LO_coefs', 'F_norm', 'bg_ctrl', 'Cn2_weights', 'L0', 'r0'], False)
model.inputs_manager.set_optimizable(['LO_coefs', 'F_norm', 'bg_ctrl', 'Cn2_weights', 'L0'], False)
x_dict = model.inputs_manager.to_dict()

x_dict['J_ctrl'] = x_dict['J_ctrl'] * 0.0 + 6.0


PSF_1 = model(x_dict) # update the model with the initial parameters to ensure all internal states are consistent


cmap_viridis = plt.cm.get_cmap('viridis').copy()
cmap_viridis.set_bad(color='#440154')  # viridis darkpurple

vmax = max(PSF_1.max(), PSF_1.max())
vmin = vmax * 1e-4

plt.imshow(PSF_1.cpu().squeeze().sum(dim=0), cmap=cmap_viridis, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.axis('off')
plt.show()


#%%
print(model.inputs_manager)

print("\n" + "="*60)
print("Model Initialization Summary".center(60))
print("="*60)
print(f"{'Parameter':<35} {'Value':>20}")
print("-"*60)
print(f"{'Number of sources':<35} {tiptorch_model.N_src:>20}")
print(f"{'Sparse wavelengths':<35} {tiptorch_model.N_wvl:>20}")
print(f"{'Full-spectrum wavelengths':<35} {model.num_λ_slices:>20}")
print(f"{'Telescope diameter':<35} {tiptorch_model.D:>19.1f} m")
print(f"{'Pixel scale':<35} {tiptorch_model.psInMas:>15.3f} mas/pixel")
print(f"{'Field of view':<35} {f'{tiptorch_model.N_pix} × {tiptorch_model.N_pix} pixels':>20}")
print(f"{'Number of atmospheric layers':<35} {tiptorch_model.N_L:>20}")
print(f"{'Number of DMs':<35} {tiptorch_model.N_DM:>20}")
print(f"{'Number of guide stars':<35} {tiptorch_model.N_GS:>20}")
print("="*60)

static_WFE = model.LO_basis.basis[model.static_WFE_mode]
static_WFE_rms = static_WFE[tiptorch_model.pupil.squeeze() > 0].std()
print(f"Static WFE loaded with RMS = {static_WFE_rms:.2f} nm")

print(model.inputs_manager)

_ = model()


#%
# Fit the sparse HARMONI cube using the same managed-parameter workflow as the
# MUSE on-sky example. The zero-padded border is excluded from the objective.
PSF_data = torch.as_tensor(PSF_0, device=default_device, dtype=default_torch_type,).unsqueeze(0)
fit_mask = torch.as_tensor(PSF_mask, device=default_device, dtype=torch.bool)

λ_weighting = False

if λ_weighting:
    wavelength_weights = torch.linspace(0.5, 1.0, PSF_data.shape[1], device=default_device, dtype=default_torch_type, ).view(1, -1, 1)
    wavelength_weights *= PSF_data.shape[1] / wavelength_weights.sum()
else:
    wavelength_weights = 1.0


def run_model(x):
    """Unpack an optimizer vector, update the wrapper, and render sparse PSFs."""
    inputs = model.inputs_manager.unstack(x, include_all=True, update=True)
    return model(inputs)


def loss_PSF(PSF_data, model, w_MSE, w_MAE, w_log):
    """MUSE-style image loss evaluated only on measured HARMONI pixels."""
    diff = (model - PSF_data)[..., fit_mask] * wavelength_weights
    MSE_loss = diff.pow(2).mean() * w_MSE
    MAE_loss = diff.abs().mean() * w_MAE

    # Robust (Huber) log-loss: fits the PSF wings, where linear MSE/MAE have little effect,
    # while being insensitive to residual outliers; eps is adaptive to the data peak to keep
    # both logarithm arguments strictly positive at all flux levels.
    eps = PSF_data.detach().amax().clamp(min=1e-10) * 1e-5
    log_diff = (torch.log(model.clamp(min=eps)) - torch.log(PSF_data.clamp(min=eps)))[..., fit_mask] * wavelength_weights
    log_loss = torch.nn.functional.huber_loss(log_diff, torch.zeros_like(log_diff), delta=1.0) * w_log

    return 2e4 * (MSE_loss + MAE_loss)  + log_loss * 1e-3

if model.use_Moffat:
    def Moffat_penalty():
        amp = model.inputs_manager['amp']
        # alpha = model.inputs_manager['alpha']
        # beta = model.inputs_manager['beta']
        # b = model.inputs_manager['b']
        
        # Enforce positive amplitude
        amp_penalty = amp.pow(2).mean() * 0.5
        
        # Enforce beta > 1.5
        # beta_penalty = torch.clamp(1.5 - beta, min=0).pow(2).mean() * 1e-3
        
        # # Enforce alpha > 0
        # alpha_penalty = torch.clamp(-alpha, min=0).pow(2).mean() * 1e-3
        
        # # Enforce b > 0
        # b_penalty = torch.clamp(-b, min=0).pow(2).mean() * 1e-3
        
        return amp_penalty #+ b_penalty + beta_penalty + alpha_penalty
else:
    def Moffat_penalty():
        return 0.0


def jitter_penalty():
    """Encourage small jitter values (in mas) to avoid unphysical PSF broadening."""
    jitter = model.inputs_manager['J_ctrl'].abs().sum()
    return jitter / N_λ_sparse * 0.5


def dn_penalty():
    """Encourage small differential piston values (in nm) to avoid unphysical PSF broadening."""
    dn = model.inputs_manager['dn'].abs()
    return dn * 0.02


def loss_fn(x):
    PSF_fitted = run_model(x)
    PSF_loss = loss_PSF(PSF_data, PSF_fitted, w_MSE=2000.0, w_MAE=2.6, w_log=500.0)
    # metrics_loss = metric_mismatch_loss(PSF_fitted, PSF_data)
    return PSF_loss + dn_penalty() + Moffat_penalty()  #+ jitter_penalty() 


def minimize_params(loss_function, max_iter, verbose=True, force_BFGS=False):
    """Fit selected managed inputs, retrying with BFGS after a poor early stop."""
        
    x_backup = model.inputs_manager.stack().clone()

    def run_minimizer(method, tolerance):
        return minimize(
            loss_function,
            model.inputs_manager.stack(),
            max_iter=max_iter,
            tol=tolerance,
            method=method,
            disp=2 if verbose else 0,
        )

    result = run_minimizer('bfgs' if force_BFGS else 'l-bfgs', 1e-4)
    acceptable_loss = 1.0
    stopped_early = result['nit'] < max_iter * 0.3
    high_loss = result['fun'] > acceptable_loss

    if not force_BFGS and stopped_early and high_loss:
        if verbose:
            print("Warning: L-BFGS stopped early with a high loss. Retrying from the initial parameters with BFGS...")
            
        model.inputs_manager.unstack(x_backup, include_all=True, update=True)
        result = run_minimizer('bfgs', 1e-5)

    # Make the manager state deterministic even if the optimizer's last loss
    # evaluation was not performed exactly at result.x.
    model.inputs_manager.unstack(result.x, include_all=True, update=True)
    PSF_fitted = run_model(result.x)

    if verbose:
        print('-' * 50)

    success = bool(result['fun'] < acceptable_loss)
    if not success:
        print("Warning: Minimization did not reach the target loss.")

    return result.x, PSF_fitted, success, float(result['fun'])


print("\nFitting sparse HARMONI PSFs...")
x_fit, PSF_1, success, final_loss = minimize_params(loss_fn, 250)

print(f"Fit success: {success}; final loss: {final_loss:.6f}")
print("Fitted parameters:")
print(model.inputs_manager)

# Find wavelength closest to 1600 nm in sparse array
target_wavelength = 1600.0  # nm
wvl_idx = np.argmin(np.abs(λ_sparse - target_wavelength))
print(f"\nComparing at wavelength: {λ_sparse[wvl_idx]:.1f} nm (closest to {target_wavelength:.1f} nm)")

# Crop to central 121 pixels
crop_size = 121
center_y, center_x = N_pix // 2, N_pix // 2
half_crop = crop_size // 2
crop_ = np.s_[center_y-half_crop:center_y+half_crop+1, center_x-half_crop:center_x+half_crop+1]

PSF_data_cropped = PSF_0[wvl_idx][crop_]
PSF_simulated_cropped = PSF_1.detach().cpu().numpy()[0, wvl_idx][crop_]

# Determine common vmin and vmax for log normalization
vmax = max(PSF_data_cropped.max(), PSF_simulated_cropped.max())
vmin = vmax * 1e-4

# Compute difference
PSF_difference = PSF_data_cropped - PSF_simulated_cropped

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Set bad pixels to dark purple (viridis-like)
cmap_viridis = plt.cm.get_cmap('viridis').copy()
cmap_viridis.set_bad(color='#440154')  # viridis darkpurple

cmap_rdbu = plt.cm.get_cmap('RdBu_r').copy()
cmap_rdbu.set_bad(color='darkblue')

# Data PSF (log scale)
im1 = axes[0].imshow(PSF_data_cropped, cmap=cmap_viridis, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
axes[0].set_title(f'Data PSF at {λ_sparse[wvl_idx]:.1f} nm\n(Log Scale, {crop_size}x{crop_size})')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity (log)')

# Fitted PSF (log scale)
im2 = axes[1].imshow(PSF_simulated_cropped, cmap=cmap_viridis, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
axes[1].set_title(f'Fitted PSF at {λ_sparse[wvl_idx]:.1f} nm\n(Log Scale, {crop_size}x{crop_size})')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

# Difference (data - fitted model)
diff_vmax = np.abs(PSF_difference).max()
linthresh = diff_vmax * 1e-3  # Linear threshold: values below this are shown linearly
im3 = axes[2].imshow(PSF_difference, cmap=cmap_rdbu, origin='lower', norm=SymLogNorm(linthresh=linthresh, vmin=-diff_vmax, vmax=diff_vmax, base=10))
axes[2].set_title(f'Difference (Data - Fit)\n(SymLog Scale, {crop_size}x{crop_size})')
axes[2].set_xlabel('Pixel')
axes[2].set_ylabel('Pixel')
plt.colorbar(im3, ax=axes[2], label='Intensity Difference')

plt.tight_layout()
plt.show()

#%
# Radial profile comparison
wvl_select = [0, len(λ_sparse)//2, -1]

fig, ax = plt.subplots(1, len(wvl_select), figsize=(15, 4))
for i, lmbd_idx in enumerate(wvl_select):
    plot_radial_PSF_profiles(
        PSF_0[lmbd_idx],
        PSF_1[0, lmbd_idx, ...].detach().cpu().numpy(),
        'Data',
        'Fitted',
        cutoff = 40,
        y_min = 3e-2,
        linthresh = 1e-2,
        return_profiles = False,
        ax = ax[i]
    )
    ax[i].set_title(f'λ = {λ_sparse[lmbd_idx]:.1f} nm')

plt.tight_layout()
plt.show()

#%%
print("\nSimulating the fitted model on the binned HARMONI H-band spectrum...")
PSF_1_full = model.SimulateFullSpectrum(src_ids=0, λ_batch_size=50, verbose=True, force_cpu=True)[0]

full_flux = PSF_1_full.sum(dim=(-2, -1))
print(f"Full-spectrum PSF shape: {tuple(PSF_1_full.shape)}")
print(f"Per-slice flux range: {full_flux.min().item():.6f} to {full_flux.max().item():.6f}")

sample_ids = [0, len(λ_binned) // 2, len(λ_binned) - 1]
fig, axes = plt.subplots(2, len(sample_ids), figsize=(15, 9))

for column, spectral_id in enumerate(sample_ids):
    data_slice = PSF_binned[spectral_id]
    model_slice = PSF_1_full[spectral_id].numpy()
    vmax = max(data_slice.max(), model_slice.max())
    norm = LogNorm(vmin=max(vmax * 1e-4, 1e-16), vmax=vmax)
    wavelength = λ_binned[spectral_id]

    axes[0, column].imshow(data_slice, cmap='viridis', origin='lower', norm=norm)
    axes[0, column].set_title(f'Data at {wavelength:.1f} nm')
    axes[1, column].imshow(model_slice, cmap='viridis', origin='lower', norm=norm)
    axes[1, column].set_title(f'Model at {wavelength:.1f} nm')

for axis in axes.flat:
    axis.set_xlabel('Pixel')
    axis.set_ylabel('Pixel')

plt.tight_layout()
plt.show()

# Averaged full-spectrum radial profiles comparison
wvl_select = [0, len(λ_sparse)//2, -1]

avg_white = ( lambda x: x.mean(dim=0) if isinstance(x, torch.Tensor) else np.mean(x, axis=0) )

plot_radial_PSF_profiles(
    avg_white(PSF_0_full),
    avg_white(PSF_1_full[valid_slices,...]),
    'Data',
    'Fitted',
    cutoff = 40,
    y_min = 3e-2,
    linthresh = 1e-2,
    return_profiles = False,
)
plt.tight_layout()
plt.show()

#%%
# Save the fitted full-spectrum PSF cube as a FITS file, mirroring the original HARMONI
# data cube's single-extension layout and WCS conventions (spatial + AWAV spectral axis).
def save_fitted_cube_fits(
    output_path, psf_cube, λ_full_nm, reference_header,
    orig_frame_shape=None, orig_frame_center=None,
    compress=True, compression_type='GZIP_2'
):
    """
    Save a TipTorch-fitted full-spectrum PSF cube (N_λ, N_y, N_x) as a FITS file whose
    header follows the same spatial/spectral WCS layout as the original simulated HARMONI
    cube (reference_header), so the two files can be compared/overlaid directly.

    Tile compression follows the same pattern as MUSEObservation.SaveModelCubeFITS.

    Parameters
    ----------
    output_path       : str or Path
    psf_cube          : array-like or torch.Tensor, shape (N_λ, N_y, N_x)
    λ_full_nm         : 1-D array, wavelengths in nm, matching psf_cube's first axis
    reference_header  : fits.Header of the original data cube (HDU[1]), used as a WCS template
    orig_frame_shape  : (N_y, N_x) of the original (raw) cube this cutout was cropped/embedded
                        from, if the cutout is meant to be placed back into it. Optional.
    orig_frame_center : (y, x), 0-based pixel location in the original frame that this cutout's
                        centre (N_pix // 2, N_pix // 2) corresponds to. Optional.
    compress          : bool, apply tile compression (default True)
    compression_type  : str, FITS tile compression algorithm (default 'GZIP_2')
    """
    data = psf_cube.detach().cpu().numpy() if torch.is_tensor(psf_cube) else np.asarray(psf_cube)
    data = data.astype(np.float32)
    n_λ, n_y, n_x = data.shape

    λ_full_A = np.asarray(λ_full_nm, dtype=float) * 10.0  # nm -> angstrom, matching the reference header's units

    hdr = fits.Header()
    hdr['OBJECT'] = reference_header.get('OBJECT', 'UNKNOWN')

    # Spatial WCS: same sky position and plate scale as the reference cube; only the
    # reference pixel changes since the fitted cube uses a different (N_pix x N_pix) footprint.
    hdr['CTYPE1'] = reference_header.get('CTYPE1', 'RA---TAN'); hdr['CUNIT1'] = reference_header.get('CUNIT1', 'deg')
    hdr['CTYPE2'] = reference_header.get('CTYPE2', 'DEC--TAN'); hdr['CUNIT2'] = reference_header.get('CUNIT2', 'deg')
    hdr['CD1_1']  = reference_header.get('CD1_1', 1.0);  hdr['CD1_2'] = reference_header.get('CD1_2', 0.0)
    hdr['CD2_1']  = reference_header.get('CD2_1', 0.0);  hdr['CD2_2'] = reference_header.get('CD2_2', 1.0)
    hdr['CRVAL1'] = reference_header.get('CRVAL1', 0.0); hdr['CRVAL2'] = reference_header.get('CRVAL2', 0.0)
    hdr['CRPIX1'] = n_x / 2 + 0.5
    hdr['CRPIX2'] = n_y / 2 + 0.5

    # Spectral WCS: AWAV axis in angstrom, re-derived from the actually simulated wavelengths.
    hdr['CTYPE3'] = reference_header.get('CTYPE3', 'AWAV'); hdr['CUNIT3'] = reference_header.get('CUNIT3', 'angstrom')
    hdr['CRPIX3'] = 1.0
    hdr['CRVAL3'] = float(λ_full_A[0])
    hdr['CD3_3']  = float(np.median(np.diff(λ_full_A)))

    hdr['BUNIT']   = 'normalized flux'
    hdr['EXTNAME'] = 'DATA'
    hdr['COMMENT'] = 'TipTorch-fitted PSF cube; WCS follows the original HARMONI simulated cube.'

    # Raw-frame placement: lets this cutout be re-embedded at the correct pixel location
    # in the original (uncropped) cube, e.g. via embed_PSF_in_new_cube(cube, ONAXIS2, ONAXIS1, OCENTY, OCENTX).
    if orig_frame_shape is not None and orig_frame_center is not None:
        hdr['ONAXIS1'] = (int(orig_frame_shape[1]), 'Original (raw) frame width, in pixels')
        hdr['ONAXIS2'] = (int(orig_frame_shape[0]), 'Original (raw) frame height, in pixels')
        hdr['OCENTX']  = (int(orig_frame_center[1]), '0-based x pixel in the raw frame at this cutout centre')
        hdr['OCENTY']  = (int(orig_frame_center[0]), '0-based y pixel in the raw frame at this cutout centre')



    if compress:
        hdu = fits.CompImageHDU(data, header=hdr, compression_type=compression_type)
    else:
        hdu = fits.ImageHDU(data, header=hdr)

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(str(output_path), overwrite=True)
    comp_label = f' ({compression_type} compressed)' if compress else ''
    print(f"Saved fitted full-spectrum PSF cube{comp_label} to {output_path}")


output_cube_path = Path(cube_path).with_name(Path(cube_path).stem + '_TipTorch_fit.fits')
save_fitted_cube_fits(
    output_cube_path,
    PSF_1_full,
    model.λ_full.detach().cpu().numpy() * 1e9,  # model.λ_full is in metres
    reference_header=PSF_header,
    orig_frame_shape=orig_frame_shape,
    orig_frame_center=orig_frame_center,
)

#%%
# Load the saved cube back and display it, as a sanity check that the FITS round-trips correctly.
with fits.open(output_cube_path) as hdul_check:
    hdul_check.info()
    PSF_1_full_reloaded = hdul_check['DATA'].data
    reloaded_header = hdul_check['DATA'].header

n_λ_reloaded = reloaded_header['NAXIS3']
λ_reloaded_A = reloaded_header['CRVAL3'] + np.arange(n_λ_reloaded) * reloaded_header['CD3_3']
λ_reloaded_nm = λ_reloaded_A / 10.0

print(f"Reloaded cube shape: {PSF_1_full_reloaded.shape}")
print(f"Reloaded wavelength range: {λ_reloaded_nm[0]:.1f} to {λ_reloaded_nm[-1]:.1f} nm")
print(f"Original raw-frame shape: ({reloaded_header['ONAXIS2']}, {reloaded_header['ONAXIS1']})")
print(f"Cutout centre in raw frame: ({reloaded_header['OCENTY']}, {reloaded_header['OCENTX']})")

# Embed the cutout back into the original (raw) pixel grid, using the ONAXIS1/2 and
# OCENTX/OCENTY entries stored in the FITS header (see save_fitted_cube_fits above).
def embed_cutout_at_position(cutout, raw_shape, center_yx):
    """
    Place a centred PSF cutout (N_λ, cut_ny, cut_nx) into a zero-padded cube of shape
    (N_λ, *raw_shape), such that the cutout's own centre pixel lands at center_yx = (y, x)
    in the raw frame. Clips at the raw-frame boundary, mirroring embed_PSF_in_new_cube.
    """
    n_λ, cut_ny, cut_nx = cutout.shape
    raw_ny, raw_nx = raw_shape
    center_y, center_x = center_yx

    offset_y = center_y - cut_ny // 2
    offset_x = center_x - cut_nx // 2

    raw_cube = np.zeros((n_λ, raw_ny, raw_nx), dtype=cutout.dtype)

    src_y_start, src_x_start = max(0, -offset_y), max(0, -offset_x)
    dst_y_start, dst_x_start = max(0, offset_y),  max(0, offset_x)
    src_y_end = min(cut_ny, raw_ny - offset_y)
    src_x_end = min(cut_nx, raw_nx - offset_x)
    dst_y_end = min(raw_ny, cut_ny + offset_y)
    dst_x_end = min(raw_nx, cut_nx + offset_x)

    raw_cube[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        cutout[:, src_y_start:src_y_end, src_x_start:src_x_end]

    return raw_cube


raw_shape_reloaded = (reloaded_header['ONAXIS2'], reloaded_header['ONAXIS1'])
center_yx_reloaded = (reloaded_header['OCENTY'], reloaded_header['OCENTX'])

PSF_1_full_embedded = embed_cutout_at_position(PSF_1_full_reloaded, raw_shape_reloaded, center_yx_reloaded)
print(f"Embedded cube shape: {PSF_1_full_embedded.shape}")

sample_ids = [0, n_λ_reloaded // 2, n_λ_reloaded - 1]
fig, axes = plt.subplots(1, len(sample_ids), figsize=(15, 5))

for ax, spectral_id in zip(axes, sample_ids):
    im = ax.imshow(
        PSF_1_full_reloaded[spectral_id],
        cmap='viridis', origin='lower',
        norm=LogNorm(vmin=max(PSF_1_full_reloaded[spectral_id].max() * 1e-4, 1e-16), vmax=PSF_1_full_reloaded[spectral_id].max())
    )
    ax.set_title(f'Reloaded fit at {λ_reloaded_nm[spectral_id]:.1f} nm')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    plt.colorbar(im, ax=ax, label='Intensity (log)')

plt.tight_layout()
plt.show()

#%%
