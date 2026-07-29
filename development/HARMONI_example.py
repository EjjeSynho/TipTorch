#%%
# Import required libraries
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager
from tiptorch._config import default_device, default_torch_type
from tiptorch.tools.utils import BinCn2
from astropy.io import fits

#%%
config_manager = ConfigManager()
config_torch = config_manager.Load('/home/aosimul/akuznets/Data/HARMONI/HARMONI_MCAO_Med.ini')

#%%
config_torch = config_manager.Convert(config_torch, framework='pytorch', device=default_device, dtype=default_torch_type)

# >>>>>>> Perform Cn2 layers binning
# Bin the Cn2 profile to reduce the number of atmospheric layers for computational efficiency
# The config file may have many layers (e.g., 36), but we can use fewer (e.g., 5-10) for simulation

# Extract original Cn2 weights and layer altitudes from config
Cn2_weights     = config_torch['atmosphere']['Cn2Weights'].flatten()
layer_altitudes = config_torch['atmosphere']['Cn2Heights'].flatten()

# Number of binned layers (reduce from original to this number)
N_binned = 10  # Use 10 equivalent layers
# Perform Cn2 binning using the vectorized BinCn2 function
Cn2_binned, h_binned = BinCn2(Cn2_weights, layer_altitudes, N_binned=N_binned)

#%%
# Update config with binned Cn2 profile
config_torch['atmosphere']['Cn2Weights'] = Cn2_binned.unsqueeze(0)  # Add batch dimension
config_torch['atmosphere']['Cn2Heights'] = h_binned.unsqueeze(0)    # Add batch dimension

print(f"Cn2 profile binned from {len(layer_altitudes)} layers to {N_binned} equivalent layers")
print(f"Binned heights:   {h_binned.cpu().numpy().tolist()}")
print(f"Binned weights:   {Cn2_binned.cpu().numpy().tolist()}")

# >>>>>>> Load simulated PSF cube for comparison
psf_cube_path = '/home/aosimul/akuznets/Data/HARMONI/simulated_cubes_6mas/01_Feige110_MCAO_6MAS_Iz_CLEAR_ERM_DATACUBE.fits'
try:
    with fits.open(psf_cube_path) as hdul:
        psf_cube   = hdul[1].data  # Shape: (N_wvl, N_pix, N_pix) or (N_wvl, N_y, N_x)
        psf_header = hdul[1].header
        
    print(f"\nLoaded PSF cube from {psf_cube_path}")
    print(f"PSF cube shape: {psf_cube.shape}")
    print(f"Number of wavelengths: {psf_cube.shape[0]}")
    
    # Get wavelength information from header
    # This cube uses CD3_3/CRPIX3/CRVAL3 (WCS convention) instead of CDELT3,
    # with wavelengths stored in angstroms (CUNIT3 = 'angstrom')
    if 'CD3_3' in psf_header:
        wave_step = psf_header['CD3_3']
        wave_ref_pix = psf_header.get('CRPIX3', 1.)
        wave_ref_val = psf_header.get('CRVAL3', 0)
        wave_unit = psf_header.get('CUNIT3', 'angstrom')

        pix_idx = np.arange(psf_cube.shape[0]) + 1  # FITS pixel indices are 1-based
        wavelengths_fits = wave_ref_val + (pix_idx - wave_ref_pix) * wave_step

        # Convert to nanometers
        if wave_unit.strip().lower() == 'angstrom':
            wavelengths_fits = wavelengths_fits / 10.0
        elif wave_unit.strip().lower() in ('nm', 'nanometer', 'nanometers'):
            pass
        elif wave_unit.strip().lower() == 'm':
            wavelengths_fits = wavelengths_fits * 1e9

        print(f"Wavelength range: {wavelengths_fits[0]:.1f} to {wavelengths_fits[-1]:.1f} nm")
        
except FileNotFoundError:
    print(f"\nPSF cube file not found at {psf_cube_path}")
    print("Continuing without PSF cube loading...")

#%%
print("Initializing TipTorch model for ELT/HARMONI...")
model = TipTorch(
    AO_config=config_torch,
    AO_type='MCAO',  # MCAO (Multi-Conjugate AO) for ELT with multiple DMs
    pupil=None,  # Let TipTorch load the pupil from config file
    norm_regime='sum',  # Normalize PSF to sum=1
    device=default_device,
    oversampling=1,
    retain_PSDs=False,
    dtype=default_torch_type
)

print(f"Model initialized successfully!")
print(f"Device: {model.device}")
print(f"Number of sources: {model.N_src}")
print(f"Number of wavelengths: {model.N_wvl}")
print(f"Telescope diameter: {model.D:.1f} m")
print(f"Pixel scale: {model.psInMas:.3f} mas/pixel")
print(f"Field of view: {model.N_pix} x {model.N_pix} pixels")
print(f"Number of atmospheric layers: {model.N_L}")
print(f"Number of DMs: {model.N_DM}")
print(f"Number of guide stars: {model.N_GS}")

#%%
# ============================================================================
# STEP 4: Simulate PSF
# ============================================================================
# The forward method can be called with or without additional parameters.
# When called without arguments, it uses the current model state.

print("\nSimulating PSF at HARMONI wavelength (1600 nm)...")
PSF = model()  # Equivalent to model.forward()

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

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Full PSF (linear scale)
im1 = axes[0].imshow(PSF_np, cmap='viridis', origin='lower')
axes[0].set_title('HARMONI PSF (Linear Scale)')
axes[0].set_xlabel('Pixel')
axes[0].set_ylabel('Pixel')
plt.colorbar(im1, ax=axes[0], label='Intensity')

# PSF in log scale (to see wings)
im2 = axes[1].imshow(PSF_np, cmap='viridis', origin='lower', 
                     norm=LogNorm(vmin=PSF_np.max()*1e-4, vmax=PSF_np.max()))
axes[1].set_title('HARMONI PSF (Log Scale)')
axes[1].set_xlabel('Pixel')
axes[1].set_ylabel('Pixel')
plt.colorbar(im2, ax=axes[1], label='Intensity (log)')

plt.tight_layout()
plt.show()

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

wavelengths = [475e-9, 650e-9, 1600e-9]  # Blue, middle, near-IR (HARMONI range)
fwhms = []

fig, axes = plt.subplots(1, len(wavelengths), figsize=(15, 5))

for i, wvl in enumerate(wavelengths):
    # Update wavelength
    model.SetWavelengths(torch.tensor([wvl], device=default_device))
    
    # Simulate PSF
    PSF_wvl = model().detach().cpu().numpy().squeeze()
    
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

print("\nSimulation complete!")
print(f"TipTorch successfully simulated ELT/HARMONI PSFs at {len(wavelengths)} wavelengths.")
print(f"Telescope: ELT with {config_torch['telescope']['TelescopeDiameter'].item():.1f}m primary mirror")
print(f"AO mode: MCAO with {config_torch['DM']['NumberActuators']} DMs")
print(f"Science wavelength: {config_torch['sources_science']['Wavelength'][0].item()*1e9:.0f} nm")
