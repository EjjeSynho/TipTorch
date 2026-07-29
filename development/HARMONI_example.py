#%%
# Import required libraries
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.centroids import centroid_2dg

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager
from tiptorch._config import default_device, default_torch_type
from tiptorch.tools.utils import BinCn2
from tiptorch.tools.static_phase import ArbitraryBasis
from astropy.io import fits
import torch.nn.functional as F

"""
HARMONI PSF Simulation Example
===============================
This example demonstrates PSF simulation for ELT/HARMONI using TipTorch.

Features:
- Multi-Conjugate AO (MCAO) configuration
- Atmospheric turbulence profile binning (Cn2, wind speed, wind direction)
- Static wavefront error (WFE) support from FITS files
- Multi-wavelength PSF simulation
- PSF quality metrics and visualization

The example includes:
1. Loading and displaying ELT pupil and static WFE maps
2. Binning atmospheric profiles for computational efficiency
3. Initializing TipTorch with MCAO configuration
4. Incorporating static WFE using ArbitraryBasis
5. Simulating PSFs with/without static aberrations
6. Analyzing chromatic effects across wavelengths
7. Sensitivity analysis of static WFE amplitude
"""

#%%
PathPupil    = '/home/aosimul/akuznets/Data/HARMONI/pupils/EELT480pp0.0803m_obs0.283_spider2023.fits'
PathStaticOn = '/home/aosimul/akuznets/Data/HARMONI/pupils/ELT_M1_MORFEO_DMs_static_wfe_480px.fits'

# Read the FITS files:

hdul = fits.open(PathPupil)
hdul_static = fits.open(PathStaticOn)

# Display pupil and static WFE in a 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pupil
im1 = axes[0].imshow(hdul[1].data, cmap='gray', origin='lower')
axes[0].set_title('ELT Pupil')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Pupil Transmission')

# Static WFE
im2 = axes[1].imshow(hdul_static[0].data, cmap='jet', origin='lower')
axes[1].set_title('ELT Static WFE (MORFEO DMs)')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='WFE [m]')

plt.tight_layout()
plt.show()


#%%
config_manager = ConfigManager()
config_torch = config_manager.Load('/home/aosimul/akuznets/Data/HARMONI/HARMONI_MCAO_Med.ini')
config_torch = config_manager.Convert(config_torch, framework='pytorch', device=default_device, dtype=default_torch_type)

# >>>>>>> Perform Cn2 layers binning
Cn2_weights     = config_torch['atmosphere']['Cn2Weights'].flatten()
layer_altitudes = config_torch['atmosphere']['Cn2Heights'].flatten()
wind_speed      = config_torch['atmosphere']['WindSpeed'].flatten()
wind_direction  = config_torch['atmosphere']['WindDirection'].flatten()

N_binned = config_torch['DM']['NumberReconstructedLayers'].item()

Cn2_binned, h_binned, wind_direction_binned, wind_speed_binned = BinCn2(
    Cn2_weights,
    layer_altitudes,
    wind_direction=wind_direction,
    wind_speed=wind_speed,
    N_binned=N_binned,
)

# Update config with binned Cn2 and wind profiles
config_torch['atmosphere']['Cn2Weights']    = Cn2_binned.unsqueeze(0)  # Add batch dimension
config_torch['atmosphere']['Cn2Heights']    = h_binned.unsqueeze(0)    # Add batch dimension
config_torch['atmosphere']['WindSpeed']     = wind_speed_binned.unsqueeze(0)
config_torch['atmosphere']['WindDirection'] = wind_direction_binned.unsqueeze(0)

print(f"Cn2 profile binned from {len(layer_altitudes)} layers to {N_binned} equivalent layers")
print(f"Binned heights:   {[int(round(x)) for x in h_binned.cpu().numpy().tolist()]}")
print(f"Binned weights:   {[round(x, 2) for x in Cn2_binned.cpu().numpy().tolist()]}")
print(f"Binned wind speeds:     {[round(x, 2) for x in wind_speed_binned.cpu().numpy().tolist()]}")
print(f"Binned wind directions: {[round(x, 2) for x in wind_direction_binned.cpu().numpy().tolist()]}")

#%%
# >>>>>>> Load simulated PSF cube for comparison
# cube_path = '/home/aosimul/akuznets/Data/HARMONI/simulated_cubes_6mas/01_Feige110_MCAO_6MAS_Iz_CLEAR_ERM_DATACUBE.fits'
cube_path = '/home/aosimul/akuznets/Data/HARMONI/simulated_cubes_6mas/01_Feige110_MCAO_6MAS_H_CLEAR_ERM_DATACUBE.fits'

with fits.open(cube_path) as hdul:
    PSF_cube   = hdul[1].data  # Shape: (N_wvl, N_pix, N_pix) or (N_wvl, N_y, N_x)
    PSF_header = hdul[1].header
    
print(f"\nLoaded PSF cube from {cube_path}")
print(f"PSF cube shape: {PSF_cube.shape}")
print(f"Number of wavelengths: {PSF_cube.shape[0]}")

# Get wavelength information from header
# This cube uses CD3_3/CRPIX3/CRVAL3 (WCS convention) instead of CDELT3,
# with wavelengths stored in angstroms (CUNIT3 = 'angstrom')

Δλ        = PSF_header['CD3_3']  / 10.0
λ_ref_pix = PSF_header['CRPIX3'] / 10.0
λ_ref_val = PSF_header['CRVAL3'] / 10.0

pix_idx = np.arange(PSF_cube.shape[0]) + 1  # FITS pixel indices are 1-based
wavelengths_fits = λ_ref_val + (pix_idx - λ_ref_pix) * Δλ

print(f"Wavelength range: {wavelengths_fits[0]:.1f} to {wavelengths_fits[-1]:.1f} nm")
    
#%%
# Find PSF cube center using maximum position
n_wvl, n_y, n_x = PSF_cube.shape

# Use the first wavelength slice to find the PSF center
psf_0 = PSF_cube.mean(axis=0)
# Find the maximum position using photutils
centroid = centroid_2dg(psf_0)
center_y, center_x = int(round(centroid[0])), int(round(centroid[1]))

print(f"\nPSF cube center (from maximum): ({center_y}, {center_x})")
print(f"PSF cube original shape: {PSF_cube.shape}")


# Function to embed PSF into a new cube with specified dimensions
def embed_PSF_in_new_cube(psf_cube, target_n_y=151, target_n_x=151, center_y=None, center_x=None):
    """
    Embed PSF cube into a new spectral cube with specified dimensions.
    The PSF is centered in the new cube.
    
    Parameters:
    -----------
    psf_cube : ndarray
        Input PSF cube with shape (N_wvl, N_y, N_x)
    target_n_y : int
        Target number of pixels in y dimension (default: 151)
    target_n_x : int
        Target number of pixels in x dimension (default: 151)
    center_y : int, optional
        Y-coordinate of PSF center. If None, uses PSF maximum.
    center_x : int, optional
        X-coordinate of PSF center. If None, uses PSF maximum.
    
    Returns:
    --------
    new_cube : ndarray
        New PSF cube with shape (N_wvl, target_n_y, target_n_x)
    """
    n_wvl, n_y, n_x = psf_cube.shape
    
    # Use provided center or find maximum position
    if center_y is None or center_x is None:
        # Find the maximum position using photutils
        psf_0 = psf_cube[0]
        centroid = centroid_2dg(psf_0)
        center_y, center_x = int(round(centroid[0])), int(round(centroid[1]))
    
    # Calculate center positions in new cube
    center_y_new, center_x_new = target_n_y // 2, target_n_x // 2
    
    # Calculate offsets
    offset_y = center_y_new - center_y
    offset_x = center_x_new - center_x
    
    # Create new cube filled with zeros
    new_cube = np.zeros((n_wvl, target_n_y, target_n_x))
    
    # Calculate source and destination slices
    src_y_start = max(0, -offset_y)
    src_y_end   = min(n_y, target_n_y - offset_y)
    src_x_start = max(0, -offset_x)
    src_x_end   = min(n_x, target_n_x - offset_x)
    
    dst_y_start = max(0, offset_y)
    dst_y_end   = min(target_n_y, n_y + offset_y)
    dst_x_start = max(0, offset_x)
    dst_x_end   = min(target_n_x, n_x + offset_x)
    
    # Copy the overlapping region
    new_cube[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = psf_cube[:, src_y_start:src_y_end, src_x_start:src_x_end]
    
    return new_cube


# Embed PSF cube into new dimensions (N_wvl x 151 x 151)
print(f"\nEmbedding PSF cube into new dimensions...")
target_n_y, target_n_x = 151, 151
PSF_cube_embedded = embed_PSF_in_new_cube(PSF_cube, target_n_y, target_n_x, center_x, center_y)
print(f"Embedded PSF cube shape: {PSF_cube_embedded.shape}")

#%%
plt.imshow(np.log(np.abs(PSF_cube_embedded[1000])), cmap='viridis', origin='lower')
# plt.scatter(target_n_x // 2, target_n_y // 2, color='red', marker='x', label='Original Center')
# plt.imshow(np.log(np.abs(PSF_cube[4000])), cmap='viridis', origin='lower')
plt.show()

#%%
from tiptorch.tools.utils import mask_circle

# Generate non-zero pixels mask for embedded PSF cube (full valid footprint, single star in FoV)
PSF_mask = np.any(PSF_cube_embedded != 0, axis=0)

# Background-only mask: valid footprint excluding the central PSF core (r < 30 px)
center_y, center_x = PSF_mask.shape[0] // 2, PSF_mask.shape[1] // 2
central_mask = mask_circle(PSF_mask.shape[0], 30, center=(center_y, center_x), centered=False)
PSF_mask_bg = PSF_mask & ~central_mask.astype(bool)

# Compute median noise (additive background offset) from background-only pixels
valid_pixels = PSF_cube_embedded[:, PSF_mask_bg]
median_noise = np.median(valid_pixels, axis=1)

# Subtract the background only within the valid footprint, keeping the zero-padded border untouched
PSF_cube_embedded = np.where(
    PSF_mask[np.newaxis, :, :],
    PSF_cube_embedded - median_noise[:, np.newaxis, np.newaxis],
    0.0,
)

# Normalize each spectral slice to unit flux over the full footprint (core + wings), not just the background
valid_sums = np.sum(PSF_cube_embedded, axis=(1, 2))
PSF_cube_embedded /= valid_sums[:, np.newaxis, np.newaxis]


#%%
# Function to find closest wavelengths in the cube
def find_closest_wavelengths(user_wavelengths, cube_wavelengths):
    """Find indices of closest wavelengths in the cube for user-defined wavelengths."""
    user_wavelengths = np.array(user_wavelengths)
    cube_wavelengths = np.array(cube_wavelengths)
    
    indices = []
    for wvl in user_wavelengths:
        idx = np.argmin(np.abs(cube_wavelengths - wvl))
        indices.append(idx)
    
    return np.array(indices)


# Define wavelengths for PSF cube comparison
wavelengths = [475e-9, 650e-9, 1600e-9]  # Blue, middle, near-IR (HARMONI range)

# Extract PSF cube data for user-defined wavelengths
print(f"\nExtracting PSF cube data for user wavelengths...")
print(f"User wavelengths: {[f'{w*1e9:.1f}' for w in wavelengths]} nm")

closest_indices = find_closest_wavelengths([w*1e9 for w in wavelengths], wavelengths_fits)

print(f"Closest cube wavelengths: {wavelengths_fits[closest_indices]} nm")
print(f"Closest cube indices: {closest_indices}")

# Extract PSFs from embedded cube
PSF_cube_selected = PSF_cube_embedded[closest_indices]
print(f"\nExtracted PSF cube shape: {PSF_cube_selected.shape}")

# Visualize PSF from cube for a user-defined wavelength (e.g., 1600 nm)
target_wvl_idx = 2  # Index for 1600 nm in the wavelengths list
target_wvl = wavelengths[target_wvl_idx]
target_wvl_nm = target_wvl * 1e9

# Find the closest index in the cube for this wavelength
closest_idx = closest_indices[target_wvl_idx]
PSF_from_cube = PSF_cube_selected[target_wvl_idx]

print(f"\nVisualizing PSF from cube at {target_wvl_nm:.1f} nm (cube index {closest_idx})")

# Normalize PSF for visualization
PSF_from_cube_norm = PSF_from_cube / PSF_from_cube.sum()

# Crop to central 121 pixels
crop_size = 121
center_y, center_x = PSF_from_cube_norm.shape[0] // 2, PSF_from_cube_norm.shape[1] // 2
half_crop = crop_size // 2
PSF_from_cube_cropped = PSF_from_cube_norm[center_y - half_crop:center_y + half_crop + 1, center_x - half_crop:center_x + half_crop + 1]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear scale
im1 = axes[0].imshow(PSF_from_cube_cropped, cmap='viridis', origin='lower')
axes[0].set_title(f'PSF from Embedded Cube at {target_wvl_nm:.1f} nm (Linear, {crop_size}x{crop_size} cropped)')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity')

# Log scale
im2 = axes[1].imshow(PSF_from_cube_cropped, cmap='viridis', origin='lower', norm=LogNorm(vmin=PSF_from_cube_cropped.max()*1e-4, vmax=PSF_from_cube_cropped.max()))
axes[1].set_title(f'PSF from Embedded Cube at {target_wvl_nm:.1f} nm (Log, {crop_size}x{crop_size} cropped)')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

plt.tight_layout()
plt.show()

#%%
plt.imshow(np.log(np.abs(PSF_cube[4000])), cmap='viridis', origin='lower')

#%%
print("Initializing TipTorch model for ELT/HARMONI...")
model = TipTorch(
    AO_config=config_torch,
    device=default_device,
    retain_PSDs=False,
    dtype=default_torch_type,
    oversampling=3
)

print(f"Model initialized successfully!")
print(f"Number of sources: {model.N_src}")
print(f"Number of wavelengths: {model.N_wvl}")
print(f"Telescope diameter: {model.D:.1f} m")
print(f"Pixel scale: {model.psInMas:.3f} mas/pixel")
print(f"Field of view: {model.N_pix} × {model.N_pix} pixels")
print(f"Number of atmospheric layers: {model.N_L}")
print(f"Number of DMs: {model.N_DM}")
print(f"Number of guide stars: {model.N_GS}")

#%%
# ============================================================================
# STEP 3.5: Prepare Static WFE
# ============================================================================
# Load and prepare the static WFE map for use in PSF simulation

static_WFE = torch.tensor(hdul_static[0].data.astype(np.float32), dtype=default_torch_type, device=default_device) * model.pupil.squeeze()
static_basis = ArbitraryBasis(model, static_WFE.unsqueeze(0), ignore_pupil=False)

# Coefficient to control the amplitude of static WFE (1.0 = full amplitude)
static_WFE_amplitude = torch.tensor([1.0], device=default_device, dtype=default_torch_type)

print(f"Static WFE prepared with RMS = {static_WFE[model.pupil.squeeze() > 0].std():.2f} nm")

#%%
# ============================================================================
# STEP 4: Simulate PSF
# ============================================================================
# The forward method can be called with or without additional parameters.
# When called without arguments, it uses the current model state.
# We now include the static WFE by passing the phase parameter.

print("\nSimulating PSF at HARMONI wavelength (1600 nm) with static WFE...")
# Simulate PSF with static WFE
PSF = model(phase=static_basis(static_WFE_amplitude))  # Pass the static phase to the forward method

print(f"PSF shape: {PSF.shape}")
print(f"PSF sum (should be ~1.0): {PSF.sum().item():.6f}")
print(f"PSF max: {PSF.max().item():.6f}")
print(f"PSF min: {PSF.min().item():.6f}")

#%%
# ============================================================================
# STEP 5: Visualize Results
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Convert to numpy for plotting
PSF_np = PSF.detach().cpu().numpy().squeeze()

# Crop to central 121 pixels
crop_size = 121
center_y, center_x = PSF_np.shape[0] // 2, PSF_np.shape[1] // 2
half_crop = crop_size // 2
PSF_cropped = PSF_np[center_y - half_crop:center_y + half_crop + 1, center_x - half_crop:center_x + half_crop + 1]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Full PSF (linear scale)
im1 = axes[0].imshow(PSF_cropped, cmap='viridis', origin='lower')
axes[0].set_title(f'HARMONI PSF (Linear Scale, {crop_size}x{crop_size} cropped)')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity')

# PSF in log scale (to see wings)
im2 = axes[1].imshow(PSF_cropped, cmap='viridis', origin='lower', norm=LogNorm(vmin=PSF_cropped.max()*1e-4, vmax=PSF_cropped.max()))
axes[1].set_title(f'HARMONI PSF (Log Scale, {crop_size}x{crop_size} cropped)')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

plt.tight_layout()
plt.show()

#%%
# ============================================================================
# STEP 5.5: Compare PSF with and without Static WFE
# ============================================================================
print("\nComparing PSFs with and without static WFE...")

# Simulate PSF without static WFE
PSF_no_static = model().detach().cpu().numpy().squeeze()

# PSF with static WFE (already computed)
PSF_with_static = PSF_np

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PSF without static WFE
im1 = axes[0].imshow(PSF_no_static, cmap='viridis', origin='lower', norm=LogNorm(vmin=PSF_no_static.max()*1e-4, vmax=PSF_no_static.max()))
axes[0].set_title('PSF without Static WFE')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity (log)')

# PSF with static WFE
im2 = axes[1].imshow(PSF_with_static, cmap='viridis', origin='lower', norm=LogNorm(vmin=PSF_with_static.max()*1e-4, vmax=PSF_with_static.max()))
axes[1].set_title('PSF with Static WFE')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

# Difference (normalized)
diff = np.abs(PSF_with_static - PSF_no_static)
im3 = axes[2].imshow(diff, cmap='hot', origin='lower')
axes[2].set_title('Absolute Difference')
axes[2].set_xlabel('Pixel')
axes[2].set_ylabel('Pixel')
plt.colorbar(im3, ax=axes[2], label='|Difference|')

plt.tight_layout()
plt.show()

# Compute Strehl ratios (approximate)
print(f"Peak intensity without static WFE: {PSF_no_static.max():.6f}")
print(f"Peak intensity with static WFE:    {PSF_with_static.max():.6f}")
print(f"Relative Strehl degradation:       {(1 - PSF_with_static.max()/PSF_no_static.max())*100:.2f}%")

#%%
# ============================================================================
# STEP 6: Extract and Analyze PSF Properties
# ============================================================================

# Compute PSF properties
from astropy.modeling import models, fitting

# Center of mass
y_idx, x_idx = np.indices(PSF_np.shape)
x_com = np.sum(x_idx * PSF_np) / np.sum(PSF_np)
y_com = np.sum(y_idx * PSF_np) / np.sum(PSF_np)

# Gaussian fit to estimate FWHM
gauss_init = models.Gaussian2D(amplitude=PSF_np.max(), 
                                x_mean=x_com, y_mean=y_com,
                                x_stddev=1.0, y_stddev=1.0, theta=0)
fitter = fitting.LevMarLSQFitter()
y_grid, x_grid = np.mgrid[:PSF_np.shape[0], :PSF_np.shape[1]]
gauss_fit = fitter(gauss_init, x_grid, y_grid, PSF_np)

# FWHM calculation (FWHM = 2*sqrt(2*ln(2))*stddev ≈ 2.355*stddev)
fwhm_x = 2.355 * gauss_fit.x_stddev.value * model.psInMas.item() / 1000  # Convert to arcsec
fwhm_y = 2.355 * gauss_fit.y_stddev.value * model.psInMas.item() / 1000  # Convert to arcsec

print("\nPSF Properties:")
print(f"Center of mass: ({x_com:.2f}, {y_com:.2f})")
print(f"Gaussian FWHM: {fwhm_x:.3f} x {fwhm_y:.3f} arcsec")

#%%
# ============================================================================
# STEP 7: Simulate at Different Wavelengths
# ============================================================================
# HARMONI covers 0.475-0.935 um in the visible, but also has near-IR capability.
# Let's simulate at a few wavelengths across the HARMONI range.

fwhms = []

fig, axes = plt.subplots(1, len(wavelengths), figsize=(15, 5))

for i, wvl in enumerate(wavelengths):
    # Update wavelength
    model.SetWavelengths(torch.tensor([wvl], device=default_device))
    
    # Regenerate static phase for the new wavelength (phase is wavelength-dependent)
    static_phase = static_basis(static_WFE_amplitude)
    
    # Simulate PSF with static WFE
    PSF_wvl = model(phase=static_phase).detach().cpu().numpy().squeeze()
    
    # Gaussian fit
    y_grid, x_grid = np.mgrid[:PSF_wvl.shape[0], :PSF_wvl.shape[1]]
    gauss_init = models.Gaussian2D(amplitude=PSF_wvl.max(), 
                                    x_mean=x_com, y_mean=y_com,
                                    x_stddev=1.0, y_stddev=1.0, theta=0)
    gauss_fit = fitter(gauss_init, x_grid, y_grid, PSF_wvl)
    fwhm = 2.355 * gauss_fit.x_stddev.value * model.psInMas.item() / 1000
    fwhms.append(fwhm)
    
    # Plot
    im = axes[i].imshow(PSF_wvl, cmap='viridis', origin='lower', 
                        norm=LogNorm(vmin=PSF_wvl.max()*1e-4, vmax=PSF_wvl.max()))
    axes[i].set_title(f'PSF at {wvl*1e9:.0f} nm\nFWHM = {fwhm:.3f} arcsec')
    axes[i].set_xlabel('Pixel')
    axes[i].set_ylabel('Pixel')
    plt.colorbar(im, ax=axes[i], label='Intensity (log)')

plt.tight_layout()
plt.show()

#%%
# ============================================================================
# STEP 8: Chromatic PSF Analysis
# ============================================================================
# Show how FWHM changes with wavelength (diffraction-limited scaling)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot([w*1e9 for w in wavelengths], fwhms, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('FWHM [arcsec]')
ax.set_title('HARMONI PSF FWHM vs Wavelength (ELT)')
ax.grid(True, alpha=0.3)

# Theoretical diffraction limit: FWHM ≈ 1.028 * λ / D (radians)
theoretical_fwhm = 1.028 * np.array(wavelengths) / 38.5 * (180/np.pi) * 3600  # arcsec
ax.plot([w*1e9 for w in wavelengths], theoretical_fwhm, 'r--', 
        label=f'Theoretical diffraction limit')
ax.legend()

plt.tight_layout()
plt.show()

#%%
# ============================================================================
# STEP 9: Static WFE Amplitude Sensitivity Study
# ============================================================================
# Explore how different static WFE amplitudes affect the PSF quality

print("\nAnalyzing static WFE amplitude sensitivity...")

# Test different amplitudes (0 = no static WFE, 1 = full amplitude, >1 = amplified)
amplitudes = [0.0, 0.5, 1.0, 2.0]
peak_intensities = []

# Reset to 1600 nm wavelength for consistent comparison
model.SetWavelengths(torch.tensor([1600e-9], device=default_device))

fig, axes = plt.subplots(1, len(amplitudes), figsize=(20, 5))

for i, amp in enumerate(amplitudes):
    # Set static WFE amplitude
    amp_tensor = torch.tensor([amp], device=default_device, dtype=default_torch_type)
    static_phase = static_basis(amp_tensor)
    
    # Simulate PSF
    PSF_amp = model(phase=static_phase).detach().cpu().numpy().squeeze()
    peak_intensities.append(PSF_amp.max())
    
    # Plot
    im = axes[i].imshow(PSF_amp, cmap='viridis', origin='lower',
                        norm=LogNorm(vmin=PSF_amp.max()*1e-4, vmax=PSF_amp.max()))
    axes[i].set_title(f'Static WFE × {amp:.1f}\nPeak = {PSF_amp.max():.6f}')
    axes[i].set_xlabel('Pixel')
    axes[i].set_ylabel('Pixel')
    plt.colorbar(im, ax=axes[i], label='Intensity (log)')

plt.tight_layout()
plt.show()

# Plot Strehl vs amplitude
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
relative_strehl = np.array(peak_intensities) / peak_intensities[0]
ax.plot(amplitudes, relative_strehl, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Static WFE Amplitude (×nominal)')
ax.set_ylabel('Relative Strehl Ratio')
ax.set_title('PSF Quality vs Static WFE Amplitude')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No degradation')
ax.legend()
plt.tight_layout()
plt.show()

print(f"\nStatic WFE sensitivity analysis:")
for i, amp in enumerate(amplitudes):
    degradation = (1 - relative_strehl[i]) * 100
    print(f"  Amplitude × {amp:.1f}: Strehl = {relative_strehl[i]:.4f} (degradation: {degradation:.2f}%)")

print("\nSimulation complete!")
print(f"TipTorch successfully simulated ELT/HARMONI PSFs at {len(wavelengths)} wavelengths.")
print(f"Telescope: ELT with {config_torch['telescope']['TelescopeDiameter'].item():.1f}m primary mirror")
print(f"AO mode: MCAO with {config_torch['DM']['NumberActuators']} DMs")
print(f"Science wavelength: {config_torch['sources_science']['Wavelength'][0].item()*1e9:.0f} nm")
print(f"Static WFE RMS: {static_WFE[model.pupil.squeeze() > 0].std():.2f} nm")
