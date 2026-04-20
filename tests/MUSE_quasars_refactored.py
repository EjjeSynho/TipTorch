#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
        import linecache
        ipy.events.register('post_execute', lambda: linecache.clearcache())
except NameError:
    pass

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_settings import PROJECT_PATH, device, default_torch_type

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from torchmin import minimize
from tqdm import tqdm
from pathlib import Path

from tools.utils import mask_circle, mask_square
from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE, MUSE_DATA_FOLDER


#%%
# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data
# data_folder = MUSE_DATA_FOLDER / 'clumpy_galaxies/' # change to your actual path with the MUSE NFM data

if not isinstance(data_folder, Path):
    data_folder = Path(data_folder)

raw_path   = data_folder / "J0259/MUSE.2024-12-05T03_15_37.598.fits.fz"
cube_path  = data_folder / "J0259/J0259-0901_all.fits"
cache_path = data_folder / "J0259/J0259-0901_all.pickle"

# cube_path  = data_folder / "J0144/J0144-5745.fits"
# cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
# raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"

# cube_path  = data_folder / "reduced_cubes/CUBE_0001.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T04_56_21.169.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0001.pickle"

# We need to pre-process the data before using it with the model and asssociate the reduced telemetry - this is done by the LoadDataCache function
# You need to run this function at least ones to generate the data cache file. Then, te function will automatically reduce it ones it's found
spectral_cubes, spectral_info, _, model_config = LoadCachedDataMUSE(raw_path, cube_path, cache_path, save_cache=True, device=device, verbose=True)   
# Extract full and binned spectral cubes. Sparse cube selects a set of 7 binned wavelengths ranges
cube_full, cube_binned, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_binned"], spectral_cubes["mask"]

#%%
# To save memory and compute time, we don't need neither the full spectral cube, nor even the binned one. It's enough to have a sparse subset of spectral
# slices that cover the whole wavelength range, which is enough to constrain the chromatic behavior of PSFs. It is referred to as "sparse" cube.
λ_full,   Δλ_full   = spectral_info['λ_full'],   spectral_info['Δλ_full']
λ_binned, Δλ_binned = spectral_info['λ_binned'], spectral_info['Δλ_binned']

# Here, it's assumed to be every 5th bin, but it can be changed to any other selection strategy.
ids_λ_sparse = np.arange(0, λ_binned.shape[-1], 5)
λ_sparse  =  λ_binned[..., ids_λ_sparse]
Δλ_sparse = Δλ_binned[..., ids_λ_sparse]

#  The model config is also updated to simulate only sparse λs
model_config['sources_science']['Wavelength'] = torch.tensor(λ_sparse, device=device, dtype=torch.float32) * 1e-9 #[m]
cube_sparse = cube_binned[ids_λ_sparse, ...] # Select the sparse subset ofspectral slices
N_wvl = cube_sparse.shape[0]

# Since spectral bins are the sum, they need to be re-normalized to averages to be compatible with the full spectrum
# Δλ_sparse ≡ Δλ_binned
flux_λ_norm = torch.tensor(Δλ_full / Δλ_sparse, device=device, dtype=torch.float32)
cube_sparse *= flux_λ_norm[:, None, None]

model_config['atmosphere']['Cn2Heights'] = torch.tensor([[0.0, 1e4]], device=device)
model_config['atmosphere']['Cn2Weights'] = torch.tensor([[0.99, 0.01]], device=device)
model_config['atmosphere']['WindDirection'] = model_config['atmosphere']['WindDirection'][0,:2].unsqueeze(0)
model_config['atmosphere']['WindSpeed']     = model_config['atmosphere']['WindSpeed'][0,:2].unsqueeze(0)

#%%
from tools.multisources import add_ROIs, DetectSources, ExtractSources
from tools.utils import rad2mas, rad2arc

PSF_size = 111  # Define the size of each extracted PSF

# sources = DetectSources(cube_sparse, threshold='auto', nsigma=10, display=True, draw_win_size=20)
sources = DetectSources(cube_sparse, threshold='auto', nsigma=25, display=True, draw_win_size=20)
# Extract separate source images from the data + other data, necessary for later fitting and performance evaluation
srcs_image_data = ExtractSources(cube_sparse, sources, box_size=PSF_size, filter_sources=True, debug_draw=False)

N_src   = srcs_image_data["count"]
sources = srcs_image_data["coords"]
ROIs    = srcs_image_data["images"]

pixel_scale = model_config['sensor_science']['PixelScale'] # [mas/pix]

# Compute the center of mass for valid mask assuming it's the center of the science field
yy, xx = torch.where(valid_mask.squeeze() > 0)
field_center = np.stack([xx.float().mean().item(), yy.float().mean().item()])[None,...]

# Sources coordinates that can be understood by TipTorch model
sources_coords  = np.stack([sources['x_peak'].values, sources['y_peak'].values], axis=1)
sources_coords -= field_center
sources_coords  = sources_coords*pixel_scale / rad2mas  # [pix] -> [rad]

# Convert to zenith and azimuth angles
sources_zenith  = np.arctan(np.sqrt(sources_coords[:,0]**2 + sources_coords[:,1]**2)) * rad2arc # [arcsec]
sources_azimuth = np.degrees(np.arctan2(sources_coords[:,1], sources_coords[:,0]))  # [deg]

# Update the model config with the sources coordinates
model_config['NumberSources'] = N_src
model_config['telescope']['PupilAngle'] = torch.tensor(model_config['telescope']['PupilAngle'], device=device) # Assuming the pupil angle is 0 for simplicity, it can be updated if known
model_config['sources_science']['Zenith']  = torch.tensor(sources_zenith, device=device).unsqueeze(-1)
model_config['sources_science']['Azimuth'] = torch.tensor(sources_azimuth, device=device).unsqueeze(-1)

#%%
# Correct for the difference in energy per λ bin
flux_core_radius = 2  # [pix]
N_core_pixels = (flux_core_radius*2 + 1)**2  # [pix^2], assuming a square mask for the core flux estimation

# Contains the spectrum per source AVERAGED across the PSF core pixels. The core mask here MUST match EXACTLY the one
# used for the flux normalization factor estimation later
src_spectra_sparse = [GetSpectrum(cube_sparse, sources.iloc[i], radius=flux_core_radius, mask_type='square') for i in range(N_src)]
src_spectra_full   = [GetSpectrum(cube_full,   sources.iloc[i], radius=flux_core_radius, mask_type='square') for i in range(N_src)]

src_spectra_sparse = torch.stack(src_spectra_sparse, dim=0) # This one is stored on GPU
src_spectra_full   = np.stack(src_spectra_full, axis=0) # This one is stored on CPU to save memory
src_spectra_full   = torch.tensor(src_spectra_full, device='cpu', dtype=torch.float32)

colors = [f'tab:{color}' for color in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:N_src]]

plt.figure(figsize=(6, 4))
for i_src in range(N_src):
    plt.plot(λ_full, src_spectra_full[i_src], linewidth=0.25, alpha=0.3, color=colors[i_src], label=f'Source {i_src+1} (full spectrum)')
    plt.scatter(λ_sparse,
                src_spectra_sparse[i_src].cpu().numpy(),
                color=colors[i_src], marker='o', s=30, alpha=0.8,
                label=f'Source {i_src+1} (sparse samples)')

plt.xlabel('Wavelength, [nm]')
plt.xlim(λ_full.min(), λ_full.max())
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.title('Sources spectra preview')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% For predicted and fitted model inputs, it is convenient to organize them using inputs_manager
from PSF_models.NFM_wrapper import PSFModelNFM

PSF_model = PSFModelNFM(
    model_config,
    multiple_obs    = False,
    LO_NCPAs        = True,
    chrom_defocus   = False,
    Moffat_absorber = False,
    N_spline_nodes  = 3,
    Z_mode_max      = 9,
    device          = device
)

PSF_model.inputs_manager.set_optimizable('bg_ctrl', False)

#%%
@torch.no_grad()
def EvaluateFluxNormalizationFactor(PSF_model, N_core_pixels, quasi_inf_PSF_size=511):
    ''' Computes composite chromatic flux normalization factor. '''

    current_PSF_size = PSF_model.model.N_pix # the actual size of the simulated PSFs
    # Ensure that if PSF's small size is odd, then quasi-infinite PSF size is also odd to avoid sub-pixel shifts of the PSF core
    quasi_inf_PSF_size -= (1-current_PSF_size % 2)

    if PSF_model.use_splines:
        wvl_current = PSF_model.λ_sim.clone()
        PSF_model.SetWavelengths(PSF_model.λ_ctrl) # isntead of anchor λs, evaluate at spline nodes

    # backup the original coordinates
    coords_backup = (
        PSF_model.inputs_manager['src_dirs_x'].clone(),
        PSF_model.inputs_manager['src_dirs_y'].clone()
    )
    # Assume on-axis PSF for computing the ratio
    PSF_model.inputs_manager['src_dirs_x'] *= 0.0
    PSF_model.inputs_manager['src_dirs_y'] *= 0.0

    PSF_small = PSF_model.forward(src_ids=0) # compute only for the first source ignoring the field variability (just for speed's sake)
    PSF_model.SetImageSize(quasi_inf_PSF_size) # quasi-infinite PSF image to compute how much flux is lost while cropping

    PSF_inf = PSF_model.forward(src_ids=0)
    PSF_model.SetImageSize(current_PSF_size)

    if PSF_model.use_splines:
        PSF_model.SetWavelengths(wvl_current) # switch back to the original wavelengths

    # Restore the original coordinates
    PSF_model.inputs_manager['src_dirs_x'], PSF_model.inputs_manager['src_dirs_y'] = coords_backup

    # How much flux is cropped by assuming the finite size of the PSF. Since PSFs are normalized to ∑PSF≈1 per wavelength,
    # the crop ratio is given by the ratio of the max pixel values in the small and quasi-infinite PSF images
    crop_ratio = (PSF_inf.amax(dim=(-2,-1)) / PSF_small.amax(dim=(-2,-1))).squeeze()

    # How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob". In other words,
    # compute the ratio of the flux in the PSF core (defined by a mask) to the total flux in the quasi-infinite PSF image.
    # core_mask       = torch.tensor(mask_square(current_PSF_size,   flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)
    core_mask_inf   = torch.tensor(mask_square(quasi_inf_PSF_size, flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)
    core_flux_ratio = torch.squeeze((PSF_inf * core_mask_inf).sum(dim=(-2,-1), keepdim=True) / PSF_inf.sum(dim=(-2,-1), keepdim=True))

    # Compute composite normalization factor. Since the spectrum extracted from the PSF core is averaged over the core,
    # we need to multiply it by the number of core pixels to get the total flux in the PSF core
    PSF_norm_factor = N_core_pixels / core_flux_ratio / crop_ratio
    torch.cuda.empty_cache()
    
    if PSF_model.use_splines:
        PSF_model.inputs_manager['F_norm_λ_ctrl'] = PSF_norm_factor.clone() # store it, we'll need it later when simulating full spectrum
        # For convenience, reproject back on the simulated λs
        return PSF_model.evaluate_splines(PSF_norm_factor, PSF_model.λ_sim_normed), PSF_inf, PSF_small
    else:
        PSF_model.inputs_manager['F_norm_λ'] = PSF_norm_factor.unsqueeze(0).clone() # we'll need it later when simulating full spectrum
        return PSF_norm_factor.float(), PSF_inf, PSF_small


# from tools.plotting import plot_radial_PSF_profiles

# core_mask     = torch.tensor(mask_square(current_PSF_size,   flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)
# core_mask_inf   = torch.tensor(mask_square(quasi_inf_PSF_size, flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)

# _c = (quasi_inf_PSF_size - current_PSF_size) // 2
# inf_crop = np.s_[..., _c:_c+current_PSF_size, _c:_c+current_PSF_size]

# PSF_small_ = PSF_small / PSF_small.sum(dim=(-2,-1), keepdim=True) # Normalize to unit flux
# PSF_inf_crop_ = PSF_inf[inf_crop] / PSF_inf[inf_crop].sum(dim=(-2,-1), keepdim=True) # Normalize to unit flux

# d = (PSF_small_ - PSF_inf_crop_).abs().mean(dim=(0,1))
# d[current_PSF_size//2, current_PSF_size//2] = 0 # Set the central pixel to zero to avoid it dominating the difference due to the high flux concentration in the PSF core

# plot_radial_PSF_profiles(PSF_small_.mean(dim=(0,1)), PSF_inf_crop_.mean(dim=(0,1)), 'Data', 'Model', title='Windsong', cutoff=16, y_min=5e-2)

PSF_norm_factor, _, _ = EvaluateFluxNormalizationFactor(PSF_model, N_core_pixels)

#%%
empty_img   = torch.zeros([N_wvl, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) + 1 #TODO: fix it
# wvl_weights = wvl_weights * 0 + 1

# TODO: weighting factor from the averae spectrum of the soucres in the field or even per-source?
# TODO: relative weights for different brigtness?

def func(x):
    x_dict = PSF_model.inputs_manager.unstack(x, include_all=True, update=True)
    PSFs_ = PSF_model(x_dict)
    flux_normalization = x_dict['F_norm'].unsqueeze(-1)  * PSF_norm_factor * src_spectra_sparse
    PSFs_ *= flux_normalization.view(N_src, N_wvl, 1, 1)
    return add_ROIs( empty_img*0.0, PSFs_, srcs_image_data["img_crops"], srcs_image_data["img_slices"] )


def loss_PSF(PSF_data, PSF_pred, w_total, w_MSE, w_MAE):
    residuals = PSF_data - PSF_pred
    diff = residuals * wvl_weights
    MSE_loss = diff.pow(2).mean() * w_MSE
    MAE_loss = diff.abs().mean()  * w_MAE
    # Since x input in func() updates the internal values for PSF_model (inluding F_norm),
    # they now can directly use them to compute the flux normalization factor
    F_penalty = (PSF_model.inputs_manager['F_ctrl'] - 1.0).abs().mean()
    # Soft non-negativity penalty: penalize when residuals (data - model) are negative
    non_negativity_penalty = torch.nn.functional.relu(-residuals).mean()
    return w_total * (MSE_loss + MAE_loss) + F_penalty * 0.2 + non_negativity_penalty * 0.05


def loss(x_, data, func):
    model = func(x_)
    return loss_PSF(data, model, w_total=1e-3,  w_MSE=900.0, w_MAE=1.6)

#%%
x0 = PSF_model.inputs_manager.stack().clone()
x1 = x0.clone()

for i in range(3):
    result_global = minimize(lambda x: loss(x, cube_sparse, func), x1, max_iter=250, tol=1e-3, method='bfgs', disp=2)
    x1 = result_global.x.clone()

#%%
PSF_norm_factor_1, _, _ = EvaluateFluxNormalizationFactor(PSF_model, N_core_pixels)

# Compute the updated normalization factor to account for the new PSF morphology after the first fitting step.
# This is necessary to ensure that the flux normalization is consistent with the actual PSF shape, which may have changed during the optimization.
F_norm_correction = (PSF_norm_factor_1 / PSF_norm_factor).mean().item()

PSF_model.inputs_manager['F_norm'] /= F_norm_correction
# F_norm_λ is already updated inside the EvaluateFluxNormalizationFactor function
PSF_norm_factor = PSF_norm_factor_1.clone()

#TODO: fit only F_norm
# Update the parameters to account for the new normalization factor
result_global = minimize(lambda x: loss(x, cube_sparse, func), x1, max_iter=500, tol=1e-3, method='l-bfgs', disp=2)
x2 = result_global.x.clone()

#%%
from tools.multisources import VisualizeSources, PlotSourcesProfiles, ROI_from_valid_mask

with torch.no_grad():
    model_fit = func(x1).detach()

ROI_plot = ROI_from_valid_mask(valid_mask)["slice"]
norm_field = LogNorm(vmin=1, vmax=cube_sparse.sum(dim=0).max()) # again, rather empirical values

VisualizeSources(cube_sparse, model_fit, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_sparse, model_fit, sources, radius=16, title='Fitted PSFs')

#%% Compute Strehl ratio
with torch.no_grad():
    PSFs_pred = PSF_model()
    PSF_DL = PSF_model.model.DLPSF().squeeze()
    
Strehls_per_λ = PSFs_pred[0].amax(dim=(-2,-1)) / PSF_DL.amax(dim=(-2,-1))

plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()




#%%
empty_img_full = torch.zeros([PSF_model.num_λ_slices, cube_sparse.shape[-2], cube_sparse.shape[-1]], device='cpu')

@torch.no_grad()
def func_full():
    # No inputs since SimulateFullSpectrum() fully relies on the inputs stored in the inputs_manager, which are already up-to-date
    PSFs_combined = PSF_model.SimulateFullSpectrum(verbose=True)
    # Now, chromatic PSF normalization factor must be re-evaluated for the full spectrum
    PSF_norm_factor_full = PSF_model.evaluate_splines(PSF_model.inputs_manager['F_norm_λ_ctrl'], PSF_model.λ_full_normed).cpu()
    flux_normalization   = PSF_model.inputs_manager['F_norm'].unsqueeze(-1).cpu() * PSF_norm_factor_full * src_spectra_full
    # Apply the flux normalization to the PSFs similar to func()
    PSFs_ = PSFs_combined * flux_normalization.unsqueeze(-1).unsqueeze(-1)

    return add_ROIs( empty_img_full, PSFs_, srcs_image_data["img_crops"], srcs_image_data["img_slices"] )


#%%

model_full = func_full()

#%%
diff_img_full = (cube_full - model_full.numpy()) * valid_mask.cpu().numpy()

# VisualizeSources(cube_full, model_full, norm=LogNorm(vmin=1e1, vmax=25000*10), mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_full, model_full, sources, radius=16, title='Fitted PSFs')

#%%
plt.plot(λ_full, PSF_model.evaluate_splines(PSF_model.inputs_manager['F_ctrl'][0,...], PSF_model.λ_full_normed).squeeze().cpu().numpy())

#%% ============== Plotting the residual spectrum ===================
from astropy.convolution import convolve, Box1DKernel

def get_src_coords_in_pixels(src_idx):
    return np.round(sources['x_peak'].iloc[src_idx]).astype(int), \
           np.round(sources['y_peak'].iloc[src_idx]).astype(int)
           
# NOTE: these coords are hard-coded for J0529 obs
host_coords  = [178, 162]
AGN_coords_1 = list(get_src_coords_in_pixels(0))
AGN_coords_2 = list(get_src_coords_in_pixels(1))
bg_coords    = [135, 183]
lens_coords  = [180, 191]

targets_diff_info = [
    {
        'coords': host_coords, # defines coordinates in pixels
        'name': 'Host galaxy?', # target's label
        'color': 'tab:blue', # display color
        'radius': 2 # radius of the region around the window to compute the spectrum from, the full size of this win is 2*radius + 1
    },
    {
        'coords': bg_coords,
        'name': 'Background',
        'color': 'tab:orange',
        'radius': 5
    },
    {
        'coords': AGN_coords_2,
        'name': 'AGN im. #2 (after subtraction)',
        'color': 'tab:green',
        'radius': 2
    },
    {
        'coords': lens_coords,
        'name': 'Lens',
        'color': 'tab:purple',
        'radius': 2
    }
]

# Plots the spectra of the targets after subtraction
fig, ax = plt.subplots(figsize=(10,6))
for info in targets_diff_info:
    spectrum_sharp = GetSpectrum(diff_img_full, info['coords'], radius=info['radius'], debug_show_ROI=False )
    spectrum_avg   = convolve(spectrum_sharp, Box1DKernel(10), boundary='extend')
       
    plt.plot(λ_full, spectrum_sharp, linewidth=0.5, alpha=0.25, color=info['color'], label=info['name'])
    plt.plot(λ_full, spectrum_avg,   linewidth=1,   alpha=1,    color=info['color'], linestyle='--')

plt.legend()
plt.xlim(λ_full.min(), λ_full.max())
plt.grid(alpha=0.2)
plt.title('Spectra after subtraction')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()

#%% Plotting the actual spectrum
targets_info = [
    {
        'coords': bg_coords,
        'name': 'Background',
        'color': 'tab:orange',
        'radius': 5
    },
    {
        'coords': AGN_coords_1,
        'name': 'AGN im. #1',
        'color': 'tab:red',
        'radius': 1
    },
    {
        'coords': AGN_coords_2,
        'name': 'AGN im. #2',
        'color': 'tab:green',
        'radius': 2
    },
    {
        'coords': lens_coords,
        'name': 'Lens',
        'color': 'tab:purple',
        'radius': 2
    }
]

# Plots the spectra of the targets after subtraction
fig, ax = plt.subplots(figsize=(10,6))
for info in targets_info:
    spectrum_sharp = GetSpectrum(cube_full, info['coords'], radius=info['radius'], debug_show_ROI=False )
    spectrum_avg   = convolve(spectrum_sharp, Box1DKernel(10), boundary='extend')
       
    plt.plot(λ_full, spectrum_sharp, linewidth=0.5, alpha=0.25, color=info['color'], label=info['name'])
    plt.plot(λ_full, spectrum_avg,   linewidth=1,   alpha=1,    color=info['color'], linestyle='--')

plt.legend()
plt.xlim(λ_full.min(), λ_full.max())
plt.grid(alpha=0.2)
plt.title('Spectra')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()
    
#%% Plot multispectral cubes as RGB images
from tools.plotting import plot_wavelength_rgb_log, plot_wavelength_rgb_linear
from photutils.aperture import RectangularAperture

# Mapping MUSE λs range to visible spectrum range for RGB conversion
λ_vis = np.linspace(440, 750, diff_img_full.shape[0])

diff_rgb = plot_wavelength_rgb_log(
    diff_img_full[ROI_plot],
    wavelengths=λ_vis,
    title="Difference",
    min_val=500, max_val=60000, show=False
)

# for info in targets_diff_info:
#     # Note, that boxes position is shifted according to plotting ROI
#     aperture = RectangularAperture([info['coords'][0]-ROI_plot[1].start, info['coords'][1]-ROI_plot[2].start], info['radius']*2+1, info['radius']*2+1, theta=0)
#     aperture.plot(color=info['color'], lw=1, label=info['name'])
# plt.legend()
# plt.show()

diff_rgb = plot_wavelength_rgb_log(
    cube_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"Data",
    min_val=500, max_val=200000, show=True
)

diff_rgb = plot_wavelength_rgb_log(
    model_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"Model",
    min_val=500, max_val=200000, show=True
)


# %%
from astropy.io import fits
import numpy as np
import os

split_PSFs = True

if split_PSFs:
    # Convert model_full_split to float32 to save space (this is a 4D array with objects as first dimension)
    # Create primary HDU

    hdu = fits.PrimaryHDU(model_full_split.astype(np.float32))

    hdu.header['CRVAL1'] = 1                     # Reference pixel value for x axis
    hdu.header['CDELT1'] = 1                     # Pixel step size for x axis
    hdu.header['CUNIT1'] = 'pixel'               # Pixel unit for x axis
    hdu.header['CTYPE1'] = 'PIXEL'                # Axis type is pixel for x axis
    hdu.header['CRPIX1'] = 1                     # Reference pixel for x axis
    
    hdu.header['CRVAL2'] = 1                     # Reference pixel value for y axis
    hdu.header['CDELT2'] = 1                     # Pixel step size for y axis
    hdu.header['CUNIT2'] = 'pixel'               # Pixel unit for y axis
    hdu.header['CTYPE2'] = 'PIXEL'                # Axis type is pixel for y axis
    hdu.header['CRPIX2'] = 1                     # Reference pixel for y axis
        
    hdu.header['CRVAL3'] = λ_full[0]             # Reference wavelength value
    hdu.header['CDELT3'] = λ_full[1] - λ_full[0] # Wavelength step size
    hdu.header['CUNIT3'] = 'nm'                  # Wavelength unit
    hdu.header['CTYPE3'] = 'WAVE'                # Axis type is wavelength
    hdu.header['CRPIX3'] = 1                     # Reference pixel

    hdu.header['CTYPE4'] = 'OBJECT'              # Fourth dimension is for different objects
    hdu.header['CRPIX4'] = 1                     # Reference pixel for object dimension
    hdu.header['CRVAL4'] = 1                     # First object has index 1
    hdu.header['CDELT4'] = 1                     # Step size for object dimension
    
    # Create HDUList
    hdul = fits.HDUList([hdu])

    # Save to file
    output_file = data_folder / f'{os.path.splitext(os.path.basename(cube_path))[0]}_modeled_cube_objects.fits'
    hdul.writeto(output_file, overwrite=True)
    print(f"Saved 4D model cube to {output_file}")
    
else:
    # Create primary HDU
    hdu = fits.PrimaryHDU(model_full.astype(np.float32))

    hdu.header['CRVAL1'] = 1                     # Reference pixel value for x axis
    hdu.header['CDELT1'] = 1                     # Pixel step size for x axis
    hdu.header['CUNIT1'] = 'pixel'               # Pixel unit for x axis
    hdu.header['CTYPE1'] = 'PIXEL'                # Axis type is pixel for x axis
    hdu.header['CRPIX1'] = 1                     # Reference pixel for x axis
    
    hdu.header['CRVAL2'] = 1                     # Reference pixel value for y axis
    hdu.header['CDELT2'] = 1                     # Pixel step size for y axis
    hdu.header['CUNIT2'] = 'pixel'               # Pixel unit for y axis
    hdu.header['CTYPE2'] = 'PIXEL'                # Axis type is pixel for y axis
    hdu.header['CRPIX2'] = 1                     # Reference pixel for y axis
        
    hdu.header['CRVAL3'] = λ_full[0]             # Reference wavelength value
    hdu.header['CDELT3'] = λ_full[1] - λ_full[0] # Wavelength step size
    hdu.header['CUNIT3'] = 'nm'                  # Wavelength unit
    hdu.header['CTYPE3'] = 'WAVE'                # Axis type is wavelength
    hdu.header['CRPIX3'] = 1                     # Reference pixel

    # Create HDUList
    hdul = fits.HDUList([hdu])
    output_file = data_folder / f'{os.path.splitext(os.path.basename(cube_path))[0]}_modeled_cube.fits'
    hdul.writeto(output_file, overwrite=True)


# %%
