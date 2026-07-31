#%%
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
N_layers = 5
N_λ_bins = 40  # Number of spectral bins; set to None to use Δλ_bin

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
λ_fits = λ_ref_val + (pix_idx - λ_ref_pix) * Δλ

# Crop border slices:
PSF_0  = PSF_0 [212:-212, ...]
λ_fits = λ_fits[212:-212]

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
    
    src_y_end   = min(n_y, target_n_y - offset_y)
    src_x_end   = min(n_x, target_n_x - offset_x)
    dst_y_end   = min(target_n_y, n_y + offset_y)
    dst_x_end   = min(target_n_x, n_x + offset_x)
    
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
config_manager = ConfigManager()
config_torch = config_manager.Load('/home/aosimul/akuznets/Data/HARMONI/HARMONI_MCAO_Med.ini')
config_torch = config_manager.Convert(config_torch, framework='pytorch', device=default_device, dtype=default_torch_type)

config_torch['sources_science']['Wavelength']   = torch.tensor(λ_sparse, device=default_device).view(1,-1) * 1e-9 # [m]
config_torch['sensor_science']['PixelScale']    = 6.0
config_torch['sensor_science']['FieldOfView']   = N_pix
config_torch['telescope']['PupilAngle']         = torch.tensor(20.0, device=default_device)  # [deg]
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
    layer_altitudes = config_torch['atmosphere']['Cn2Heights'].flatten()

print(f"Simulated wavelengths: {λ_sparse.tolist()} nm ({len(λ_sparse)} slices)")
print(f"Cn2 profile binned from {len(layer_altitudes)} layers to {N_layers} equivalent layers")
print(f"Binned heights:         {[int(round(x)) for x in h_binned.cpu().numpy().tolist()]}")
print(f"Binned weights:         {[round(x, 2) for x in Cn2_binned.cpu().numpy().tolist()]}")
print(f"Binned wind speeds:     {[round(x, 2) for x in wind_speed_binned.cpu().numpy().tolist()]}")
print(f"Binned wind directions: {[round(x, 2) for x in wind_direction_binned.cpu().numpy().tolist()]}")

# Display pupil and static WFE in a 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

#%%
print("Initializing the TipTorch HARMONI wrapper...")
model = PSFModelHARMONI(
    config = config_torch,
    LO_NCPAs = True,
    use_Zernike = False,
    use_static_WFE = True,
    Z_mode_max = 9,
    N_spline_nodes = 5,
    device = default_device,
    retain_PSDs = False,
    dtype = default_torch_type,
    λ_min = float(λ_binned[0]  * 1e-9),
    λ_max = float(λ_binned[-1] * 1e-9),
    num_λ_slices=len(λ_binned),
)

model.inputs_manager.delete('wind_speed_single')
model.inputs_manager.delete('wind_dir_single')

model.inputs_manager.set_optimizable(['LO_coefs', 'F_norm', 'wind_speed_single', 'Cn2_weights', 'L0', 'r0'], False)
x_dict = model.inputs_manager.to_dict()

x_dict['J_ctrl'] = x_dict['J_ctrl'] * 0.0 + 2.0

_ = model(x_dict) # update the model with the initial parameters to ensure all internal states are consistent

print(model.inputs_manager)

tiptorch_model = model.model

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

print("\nSimulating sparse PSFs with static WFE...")
PSF_initial = model()

#%%
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


def loss_PSF(PSF_data, PSF_model, w_MSE, w_MAE):
    """MUSE-style image loss evaluated only on measured HARMONI pixels."""
    diff = (PSF_model - PSF_data)[..., fit_mask] * wavelength_weights
    MSE_loss = diff.pow(2).mean() * w_MSE
    MAE_loss = diff.abs().mean() * w_MAE
    return 2e4 * (MSE_loss + MAE_loss)

# def force_positive(x):
#     return torch.clamp(-x, min=0).pow(2).mean()

def force_small_jitter():
    """Encourage small jitter values (in mas) to avoid unphysical PSF broadening."""
    jitter = model.inputs_manager['J_ctrl'].abs().sum()
    return jitter / N_λ_sparse * 2.0

def loss_fn(x):
    PSF_fitted = run_model(x)
    PSF_loss = loss_PSF(PSF_data, PSF_fitted, w_MSE=900.0, w_MAE=3.6)
    return PSF_loss #+ force_small_jitter()


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
            print(
                "Warning: L-BFGS stopped early with a high loss. "
                "Retrying from the initial parameters with BFGS..."
            )
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
x_fit, PSF_1, success, final_loss = minimize_params(loss_fn, 200)

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

# Data PSF (log scale)
im1 = axes[0].imshow(PSF_data_cropped, cmap='viridis', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
axes[0].set_title(f'Data PSF at {λ_sparse[wvl_idx]:.1f} nm\n(Log Scale, {crop_size}x{crop_size})')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity (log)')

# Fitted PSF (log scale)
im2 = axes[1].imshow(PSF_simulated_cropped, cmap='viridis', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
axes[1].set_title(f'Fitted PSF at {λ_sparse[wvl_idx]:.1f} nm\n(Log Scale, {crop_size}x{crop_size})')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

# Difference (data - fitted model)
diff_vmax = np.abs(PSF_difference).max()
linthresh = diff_vmax * 1e-3  # Linear threshold: values below this are shown linearly
im3 = axes[2].imshow(PSF_difference, cmap='RdBu_r', origin='lower', norm=SymLogNorm(linthresh=linthresh, vmin=-diff_vmax, vmax=diff_vmax, base=10))
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
        return_profiles = True,
        ax = ax[i]
    )
    ax[i].set_title(f'λ = {λ_sparse[lmbd_idx]:.1f} nm')
plt.tight_layout()
plt.show()

#%%
print("\nSimulating the fitted model on the binned HARMONI H-band spectrum...")
PSF_full = model.SimulateFullSpectrum(
    src_ids=0,
    λ_batch_size=50,
    verbose=True,
    force_cpu=True,
)[0]

full_flux = PSF_full.sum(dim=(-2, -1))
print(f"Full-spectrum PSF shape: {tuple(PSF_full.shape)}")
print(f"Per-slice flux range: {full_flux.min().item():.6f} to {full_flux.max().item():.6f}")

sample_ids = [0, len(λ_binned) // 2, len(λ_binned) - 1]
fig, axes = plt.subplots(2, len(sample_ids), figsize=(15, 9))

for column, spectral_id in enumerate(sample_ids):
    data_slice = PSF_binned[spectral_id]
    model_slice = PSF_full[spectral_id].numpy()
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

#%%
