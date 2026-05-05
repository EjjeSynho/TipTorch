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

from tiptorch._config import WEIGHTS_FOLDER, default_device, default_torch_type, project_settings

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pathlib import Path

from tiptorch.tools.utils import mask_square
from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE

# Define the location of your NFM data. It can be whenever. One option is to add it to the project config file 
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

device = default_device

#%%
# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
# data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data
data_folder = MUSE_DATA_FOLDER / 'clumpy_galaxies/' # change to your actual path with the MUSE NFM data

if not isinstance(data_folder, Path):
    data_folder = Path(data_folder)

# raw_path   = data_folder / "J0259/MUSE.2024-12-05T03_15_37.598.fits.fz"
# cube_path  = data_folder / "J0259/J0259-0901_all.fits"
# cache_path = data_folder / "J0259/J0259-0901_all.pickle"

# cube_path  = data_folder / "J0144/J0144-5745.fits"
# cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
# raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"

cube_path  = data_folder / "reduced_cubes/CUBE_0001.fits"
raw_path   = data_folder / "raw_data/MUSE.2023-04-27T04_56_21.169.fits.fz"
cache_path = data_folder / "reduced_telemetry/CUBE_0001.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0002.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_11_59.783.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0002.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0003.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_28_41.014.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0003.pickle"


# cube_path  = data_folder / "reduced_cubes/CUBE_0021.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-06-17T00_04_47.319.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0021.pickle"


# We need to pre-process the data before using it with the model and asssociate the reduced telemetry - this is done by the LoadDataCache function
# You need to run this function at least ones to generate the data cache file. Then, te function will automatically reduce it ones it's found
spectral_cubes, spectral_info, data_cache, model_config = LoadCachedDataMUSE(raw_path, cube_path, cache_path, save_cache=True, device=device, verbose=True)   
# Extract full and binned spectral cubes. Sparse cube selects a set of 7 binned wavelengths ranges
cube_full, cube_binned, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_binned"], spectral_cubes["mask"]

#NOTE: MUSE cube flux units are [10^-20 erg s^-1 cm^-2 Å^-1], all cubes are normalized to this flux unit

# Compute the center of mass for valid mask assuming it's the center of the science field
yy, xx = torch.where(valid_mask.squeeze() > 0)
field_center = np.stack([xx.float().mean().item(), yy.float().mean().item()])[None,...] # [pix]
del yy, xx

reduced_telemetry = data_cache['All data'] # this is the telemetry reduced to the format compatible with the model, it can be used to update the model config with the actual telemetry values
del data_cache # free up memory, we won't need the rest of the data cache for now, but it can be useful for debugging and further analysis if needed

#%%
# To save memory and compute time, we don't need neither the full spectral cube, nor even the binned one. It's enough to have a sparse subset of spectral
# slices that cover the whole wavelength range, which is enough to constrain the chromatic behavior of PSFs. It is referred to as "sparse" cube.
λ_full,   Δλ_full   = spectral_info['λ_full'],   spectral_info['Δλ_full']
λ_binned, Δλ_binned = spectral_info['λ_binned'], spectral_info['Δλ_binned']

# Here, it's assumed to be every 5th bin, but it can be changed to any other selection strategy
ids_λ_sparse = np.arange(0, λ_binned.shape[-1], 5)
λ_sparse  =  λ_binned[..., ids_λ_sparse]
Δλ_sparse = Δλ_binned[..., ids_λ_sparse] # Δλ_sparse ≡ Δλ_binned

cube_sparse = cube_binned[ids_λ_sparse, ...] # Select the sparse subset ofspectral slices
N_wvl = cube_sparse.shape[0]

# Since spectral bins are the sum, they need to be re-normalized to averages to be compatible with the full spectrum
flux_λ_norm = torch.tensor(Δλ_full / Δλ_sparse, device=device, dtype=torch.float32)
cube_sparse *= flux_λ_norm[:, None, None]

#%%
def AddSourcesToModelConfig(model_config, sources):
    from tiptorch.tools.utils import rad2mas, rad2arc

    pixel_scale = model_config['sensor_science']['PixelScale'] # [mas/pix]
    
    # Sources coordinates that can be understood by TipTorch model
    sources_coords  = np.stack([sources['x_peak'].values, sources['y_peak'].values], axis=1)
    sources_coords -= field_center
    sources_coords  = sources_coords * pixel_scale / rad2mas  # [pix] -> [rad]

    # Convert to zenith and azimuth angles
    sources_zenith  = np.arctan(np.sqrt(sources_coords[:,0]**2 + sources_coords[:,1]**2)) * rad2arc # [arcsec]
    sources_azimuth = np.degrees(np.arctan2(sources_coords[:,1], sources_coords[:,0]))  # [deg]

    # Update the model config with the sources coordinates
    model_config['NumberSources'] = len(sources)
    model_config['sources_science']['Zenith']  = torch.tensor(sources_zenith, device=device).unsqueeze(-1)
    model_config['sources_science']['Azimuth'] = torch.tensor(sources_azimuth, device=device).unsqueeze(-1)


def ExtractSpectraFromCore(sources, cube_full, cube_sparse, flux_core_radius):
    N_core_pixels = (flux_core_radius*2 + 1)**2  # [pix²], this expression assumes a square mask for the core flux estimation

    # Contains the spectrum per source AVERAGED across the PSF core pixels. The core mask here MUST match EXACTLY the one
    # used for the flux normalization factor estimation later
    src_spectra_full = [GetSpectrum(cube_full,   sources.iloc[i], radius=flux_core_radius, mask_type='square') for i in range(N_src)]
    src_spectra_full = np.stack(src_spectra_full, axis=0) # This one is stored on CPU to save memory
    src_spectra_full = torch.tensor(src_spectra_full, device='cpu', dtype=torch.float32)

    if cube_sparse is not None:
        src_spectra_sparse = [GetSpectrum(cube_sparse, sources.iloc[i], radius=flux_core_radius, mask_type='square') for i in range(N_src)]
        src_spectra_sparse = torch.stack(src_spectra_sparse, dim=0) # This one is stored on GPU
    else:
        src_spectra_sparse = None
    
    return src_spectra_sparse, src_spectra_full, N_core_pixels


#%%
from tools.multisources import DisplaySources, add_ROIs, DetectSources, AddSources, ExtractSources

PSF_size = 111  # Define the size of each extracted PSF
model_config['sensor_science']['FieldOfView'] = PSF_size

# Simple sources detector function. For now, make sure that only point sources are adressed
# This function also defines the order in which sources are indexed ad processed later, so it's important to use it before extracting the source images
# and spectra. The order is defined by the brightness of the sources, so the brightest source will be indexed as 0, the second brightest as 1, and so on
sources = DetectSources(cube_sparse, threshold='auto', nsigma=35, box_size=11, sort_by_brightness=True, weight_from_flux=False)
sources = AddSources(cube_sparse, [[100, 200]], sources, weights=0.0, weight_from_flux=False)

DisplaySources(cube_sparse, sources, draw_box_size=20, vmin=10, vmax=sources['peak_value'].max()*0.85)

# --------------- If some sources must be filtered out, here is the right place to do it --------------------------------------

# Extract separate source images + other auxilliary data. It's necessary for later fitting and performance evaluation
srcs_image_data = ExtractSources(cube_sparse, sources, box_size=PSF_size, filter_sources=True, debug_draw=False)

N_src   = srcs_image_data["count" ]
sources = srcs_image_data["coords"]
ROIs    = srcs_image_data["images"]

#%%
from tiptorch.tools.utils import generate_random_colors

flux_core_radius = 2  # [pix]

# This function computes average spectrum around the PSF core for each source within a square with side size of 2*flux_core_radius + 1
src_spectra_sparse, src_spectra_full, N_core_pixels = ExtractSpectraFromCore(sources, cube_full, cube_sparse, flux_core_radius)

colors = generate_random_colors(N_src)

def PlotSourceSpectra(
    λ_full, src_spectra_full,
    λ_sparse = None, src_spectra_sparse = None,
    smooth_kernel = None,
    title='Sources spectra preview', figsize=(10, 6)):
    
    from astropy.convolution import convolve, Box1DKernel
    
    show_smooth = False
    if smooth_kernel is not None:
        if smooth_kernel > 1:
            show_smooth = True
                
    show_sparse = (src_spectra_sparse is not None) and (λ_sparse is not None)
                
    plt.figure(figsize=figsize)
    
    N = src_spectra_sparse.shape[0] if src_spectra_sparse is not None else src_spectra_full.shape[0]  # Number of sources
    
    vmin = min(src_spectra_full.min().item(), src_spectra_sparse.min().item() if show_sparse else float('inf'))
    
    for i_src in range(N):
        plt.plot(λ_full, src_spectra_full[i_src], linewidth=0.25, alpha=(0.5 if show_smooth else 0.8), color=colors[i_src], label=f'Source {i_src+1} (full spectrum)')
        if show_sparse:
            plt.scatter(λ_sparse, src_spectra_sparse[i_src].cpu().numpy(),
                        color=colors[i_src], marker='o', s=30, alpha=1.0,
                        label=f'Source {i_src+1} (sparse samples)')
        if show_smooth:
            spectrum_smooth = convolve(src_spectra_full[i_src].numpy(), Box1DKernel(smooth_kernel), boundary='extend')
            plt.plot(λ_full, spectrum_smooth, linewidth=0.65, alpha=1, color=colors[i_src], label=f'Source {i_src+1}')
            
        plt.axhline(0, color='gray', linewidth=0.8)

    plt.xlabel('Wavelength, [nm]')
    plt.xlim(λ_full.min(), λ_full.max())
    plt.ylim(vmin, None)
    plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
    plt.title(title)
    if N < 10: # Don't plot it when too many sources are displayed to avoid cluttering the legend
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


PlotSourceSpectra(λ_full, src_spectra_full, λ_sparse=λ_sparse, src_spectra_sparse=src_spectra_sparse)

#%% For predicted and fitted model inputs, it is convenient to organize them using inputs_manager
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from machine_learning.calibrators.NFM_calibrator import NFMCalibrator

def InitPSFModel():
    # The model config is also updated to simulate only sparse λs
    model_config['sources_science']['Wavelength'] = torch.tensor(λ_sparse, device=device, dtype=torch.float32) * 1e-9 #[m]
    model_config['telescope']['PupilAngle']       = torch.tensor(model_config['telescope']['PupilAngle'], device=device)

    AddSourcesToModelConfig(model_config, sources)

    PSF_model = PSFModelNFM(
        model_config,
        multiple_obs    = False,
        LO_NCPAs        = True,
        chrom_defocus   = False,
        Moffat_absorber = False,
        N_spline_nodes  = 5,
        Z_mode_max      = 9,
        device          = device
    )
    #  Get the initial guess for the PSF model parameters
    calibrator = NFMCalibrator(WEIGHTS_FOLDER / 'NFM_calibrator/NFM_calibrator_bundle.pth', device=device)
    _ = calibrator.check_compatibility(PSF_model)

    calibrator.calibrate(reduced_telemetry, PSF_model)

    # Disable optimization of some variables
    PSF_model.inputs_manager.set_optimizable('bg_ctrl', False)
    PSF_model.inputs_manager.set_optimizable('wind_dir', False)
    
    return PSF_model


PSF_model = InitPSFModel()

#%%
@torch.no_grad()
def EvaluateCoreFluxFactor(PSF_model, N_core_pixels, quasi_inf_PSF_size=511) -> None:
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
    else:
        PSF_model.inputs_manager['F_norm_λ'] = PSF_norm_factor.unsqueeze(0).clone() # we'll need it later when simulating full spectrum


@torch.no_grad()
def UpdateFluxNormalization(PSF_model) -> None:
    PSF_norm_factor_old = PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()

    EvaluateCoreFluxFactor(PSF_model, N_core_pixels) # update the core flux normalization factor based on the current PSF morphology
    PSF_norm_factor_new = PSF_model.inputs_manager['F_norm_λ_ctrl'].clone()

    # Compute the updated normalization factors
    F_norm_correction = (PSF_norm_factor_new / PSF_norm_factor_old).mean().item()
    PSF_model.inputs_manager['F_norm'] /= F_norm_correction

    # Since F_ctrl is the parameter that directly controls the overall flux normalization in the model, we can use its mean value to correct
    # for per-source flux normalization. This is a rather empirical correction
    F_mean_correction = PSF_model.inputs_manager['F_ctrl'].mean().item()
    PSF_model.inputs_manager['F_ctrl'] /= F_mean_correction
    PSF_model.inputs_manager['F_norm'] *= F_mean_correction


#%%
# Empty image to store the simulated PSFs while adding them to the right locations in the field. This is more memory efficient than storing all PSFs
# separately and also allows to overlap sources on top of each other, which is important for a realistic simulation and fitting
canvas = torch.zeros([N_wvl, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device)

# Compute loss weighting factors per source based on total flux per source
src_fluxes = torch.tensor(sources['peak_value'].to_numpy(), device=device, dtype=torch.float32)
src_mask   = torch.tensor(sources['weight'].to_numpy(), device=device, dtype=torch.float32) # sources importance in loss function computation

max_flux = src_fluxes.max().item()
src_relative_weights = torch.clamp(src_fluxes / max_flux, min=0.2, max=1.0) * src_mask# limit the loss influence limit
# Compute weighted mean of the src_fluxes to get a more robust estimate of the typical source flux in the field, which can be used for normalization
w_total = src_relative_weights.sum() / (src_fluxes * src_relative_weights).sum() # this normalization ensures that loss is ~10⁰-10¹

# Compute chromatic loss weighting factors per source based on it's spectrum
w_spectral = src_spectra_sparse.amax(dim=-1, keepdim=True) / src_spectra_sparse
w_spectral /= w_spectral.mean(dim=0, keepdim=True) # normalize to the mean to avoid changing the overall loss scale
w_spectral = torch.clamp(w_spectral, min=0.2, max=2.0) # limit the loss influence of certain wavelengths
w_spectral = w_spectral.view(N_src, N_wvl, 1, 1) # reshape for broadcasting

w_spectral *= 0
w_spectral += 1.0 # for now, use uniform spectral weighting, but this can be easily changed to give more importance to certain wavelengths if needed


# ---------- Source selection for fitting ----------
# Select which sources to include in the fit. None = all sources (original behavior using field-uniform PSF from source 0).
# When a subset is specified (e.g., [0, 2, 5]), only those sources are simulated via NFM_wrapper's src_ids mechanism
# (each getting its own field-position-dependent PSF), and only their ROI regions contribute to the loss and backpropagation.
# fit_src_ids = None  # e.g., [0, 2, 5] or a single int like 0

fit_src_ids = 0

if fit_src_ids is not None:
    _fit_ids = [fit_src_ids] if isinstance(fit_src_ids, (int, np.integer)) else list(fit_src_ids)
    N_fit = len(_fit_ids)
    fit_spectra_sparse = src_spectra_sparse[_fit_ids]
    fit_img_crops  = [srcs_image_data["img_crops"][i]  for i in _fit_ids]
    fit_img_slices = [srcs_image_data["img_slices"][i] for i in _fit_ids]
    fit_src_mask   = src_mask[_fit_ids]

    # Recompute loss normalization weights for the selected subset
    fit_src_fluxes = src_fluxes[_fit_ids]
    fit_max_flux   = fit_src_fluxes.max().item()
    fit_relative_weights = torch.clamp(fit_src_fluxes / fit_max_flux, min=0.2, max=1.0) * fit_src_mask
    fit_w_total = fit_relative_weights.sum() / (fit_src_fluxes * fit_relative_weights).sum()
    fit_w_spectral = w_spectral[_fit_ids]

    # Spatial mask covering only selected source ROIs to exclude unmodeled source regions from the loss
    fit_spatial_mask = torch.zeros([1, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device)
    for i in _fit_ids:
        (y_min_img, y_max_img), (x_min_img, x_max_img) = srcs_image_data["img_slices"][i]
        fit_spatial_mask[:, y_min_img:y_max_img, x_min_img:x_max_img] = 1.0
else:
    _fit_ids = None  # signals "all sources" mode
    N_fit = N_src
    fit_spectra_sparse = src_spectra_sparse
    fit_img_crops  = srcs_image_data["img_crops"]
    fit_img_slices = srcs_image_data["img_slices"]
    fit_src_mask   = src_mask
    fit_w_total    = w_total
    fit_w_spectral = w_spectral
    fit_spatial_mask = None  # no spatial masking needed when all sources are fitted


def simulate_sparse(x):
    x_dict = PSF_model.inputs_manager.unstack(x, include_all=True, update=True) # update model inputs with the values from the stacked vector
    # Save F_norm for the fitted sources before forward() modifies x_dict in-place (it selects only src_ids entries)
    F_norm_fit = x_dict['F_norm'][_fit_ids] if _fit_ids is not None else x_dict['F_norm']
    # When _fit_ids is set, NFM_wrapper computes per-source PSFs only for the selected sources (field-varying PSF morphology).
    # When _fit_ids is None, the original field-uniform approximation is used (PSF shape from source 0 applied to all).
    PSFs_  = PSF_model(x_dict, src_ids=(_fit_ids if _fit_ids is not None else 0))
    PSF_norm_factor = PSF_model.evaluate_splines(PSF_model.inputs_manager['F_norm_λ_ctrl'], PSF_model.λ_sim_normed)
    flux_normalization = F_norm_fit.unsqueeze(-1) * PSF_norm_factor * fit_spectra_sparse
    # Use regular multiplication (not *=) so that PSFs_ shape [1, N_wvl, H, W] can broadcast to [N_fit, N_wvl, H, W]
    PSFs_ = PSFs_ * flux_normalization.view(N_fit, N_wvl, 1, 1)

    return add_ROIs( canvas*0.0, PSFs_, fit_img_crops, fit_img_slices )


def loss_PSF(PSF_data, PSF_pred, weights_λ, weight_total, w_MSE, w_MAE):
    residuals = PSF_data - PSF_pred
    if fit_spatial_mask is not None:
        residuals = residuals * fit_spatial_mask  # zero out regions outside selected source ROIs
    diff = residuals * weights_λ * fit_src_mask.view(-1, 1, 1, 1) # apply both spectral and source weights to the residuals
    MSE_loss = diff.pow(2).mean() * w_MSE
    MAE_loss = diff.abs().mean()  * w_MAE
    # Since x input in simulate_sparse() updates the internal values for the PSF_model (inluding F_norm),
    # they now can be used directly
    F_penalty = (PSF_model.inputs_manager['F_ctrl'] - 1.0).abs().mean()
    # Soft non-negativity penalty: penalize when residuals (data - model) are negative to prevent ouversubtracting simulated PSFs
    non_negativity_penalty = torch.nn.functional.relu(-residuals).mean()
    return (MSE_loss + MAE_loss) * weight_total + F_penalty * 0.2 + non_negativity_penalty * 2.0


suppress_bump_flag = False
suppress_LO_flag   = False

# LO fitting weights
w_suppress_bump = 1e3 if suppress_bump_flag else 1
w_suppress_LO   = 1e3 if suppress_LO_flag   else 1

force_positive = lambda x: torch.clamp(-x, min=0).pow(2).mean()

def loss_LO(w_bump, w_LO):
    # L2 regularization on all LO coefficients
    LO_loss = PSF_model.inputs_manager['LO_coefs'].pow(2).sum(-1).mean() * w_LO * w_suppress_LO
    # Constraint to enforce first element of LO_coefs to be positive
    phase_bump_positive = force_positive(PSF_model.inputs_manager['LO_coefs'][:, 0]) * w_bump * w_suppress_bump
    # Force defocus to be positive to mitigate sign ambiguity
    first_defocus_penalty = force_positive(PSF_model.inputs_manager['LO_coefs'][:, 2]) * w_LO * w_suppress_LO #NOTE: won't work with the chromatic defocus
    
    LO_loss += phase_bump_positive + first_defocus_penalty
    return LO_loss


def loss(x_, data, func):
    model = func(x_)
    LO_loss  = loss_LO(w_bump=5e-5, w_LO=1e-7)
    PSF_loss = loss_PSF(data, model, weights_λ=fit_w_spectral, weight_total=fit_w_total, w_MSE=900.0, w_MAE=2.6)
    
    return LO_loss + PSF_loss


#%%
from fitting.PSF_optimizer import OptimizePSFModel

def FitPSFModel(PSF_model, loss_fn, repeat=2, max_iter=200):
    x_params = None
    for _ in range(repeat):
        UpdateFluxNormalization(PSF_model)
        x_params, _ = OptimizePSFModel(
            PSF_model,
            loss_fn,
            x_initial  = x_params.clone() if x_params is not None else None,
            max_iter   = max_iter,
            n_attempts = 1,
            verbose    = True,
            force_bfgs = True
        )
    return x_params

x_params = FitPSFModel(PSF_model, lambda x: loss(x, cube_sparse, simulate_sparse), repeat=3, max_iter=200)

#%%
from tools.multisources import VisualizeSources, PlotSourcesProfiles, ROI_from_valid_mask

with torch.no_grad():
    model_fit = simulate_sparse(x_params)

ROI_plot = ROI_from_valid_mask(valid_mask)["slice"]
norm_field = LogNorm(vmin=1, vmax=cube_sparse.sum(dim=0).max()) # again, rather empirical values

VisualizeSources(cube_sparse, model_fit, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_sparse, model_fit, sources, radius=16, title='Fitted PSFs')

#%% 
Strehls_per_λ = PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()

#%%
canvas_full = torch.zeros([PSF_model.num_λ_slices, cube_sparse.shape[-2], cube_sparse.shape[-1]], device='cpu')

@torch.no_grad()
def simulate_full(): #TODO: spplit output
    # No inputs since SimulateFullSpectrum() fully relies on the inputs stored in the inputs_manager, which are already up-to-date
    PSFs_combined = PSF_model.SimulateFullSpectrum(verbose=True)
    # Now, chromatic PSF normalization factor must be re-evaluated for the full spectrum
    PSF_norm_factor_full = PSF_model.evaluate_splines(PSF_model.inputs_manager['F_norm_λ_ctrl'], PSF_model.λ_full_normed).cpu()
    flux_normalization   = PSF_model.inputs_manager['F_norm'].unsqueeze(-1).cpu() * PSF_norm_factor_full * src_spectra_full
    # Apply the flux normalization to the PSFs similar to func()
    PSFs_ = PSFs_combined * flux_normalization.unsqueeze(-1).unsqueeze(-1)

    return add_ROIs(canvas_full, PSFs_, srcs_image_data["img_crops"], srcs_image_data["img_slices"]), PSFs_


model_full, PSFs_separated = simulate_full()
diff_img_full = (cube_full - model_full.numpy()) * valid_mask.cpu().numpy()

PlotSourcesProfiles(cube_full, model_full, sources, radius=16, title='Fitted PSFs')

#%% Plotting the residual spectrum

_, src_diff_full, _ = ExtractSpectraFromCore(sources, diff_img_full, cube_sparse=None, flux_core_radius=flux_core_radius)

PlotSourceSpectra(λ_full, src_diff_full)

#%% Plot multispectral cubes as RGB images
from tools.plotting import PlotSpetralCubeInRGB

# Mapping MUSE spectral range to visible spectrum range for RGB conversion
λ_vis = np.linspace(440, 750, diff_img_full.shape[0])

_ = PlotSpetralCubeInRGB(
    diff_img_full[ROI_plot],
    wavelengths=λ_vis,
    title="Difference",
    min_val=500, max_val=60000,
    show=False
)

_ = PlotSpetralCubeInRGB(
    cube_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"Data",
    min_val=500, max_val=200000,
    show=True
)

_ = PlotSpetralCubeInRGB(
    model_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"Model",
    min_val=500, max_val=200000,
    show=True
)


# %%
from astropy.io import fits
import os


def SaveModelCubeFITS(cube, output_path, λ_full, sources=None, compress=True, compression_type='GZIP_2'):
    """
    Save a 3D (N_wvl, H, W) or 4D (N_src, N_wvl, H, W) model cube to a FITS file.

    3D input → single image HDU (optionally compressed).

    4D input with sources=None → single zero-padded 4D image HDU (mostly zeros, poor use
    of space even with compression).

    4D input with sources=<DataFrame> → multi-extension FITS: one compact HDU per source
    (N_wvl, psf_H, psf_W), with the WCS in each extension encoding where the PSF belongs
    in the full field. No zeros are stored at all — this is the recommended format.
    Reading back: for source i, hdul[i+1].data gives its PSF cube; CRVAL1/2 give its
    1-based (x, y) centroid in the full-field pixel frame.

    Available compression types (all lossless for float32):
      'GZIP_2'  - byte-shuffle + gzip, best for float PSF data (default)
      'RICE_1'  - fast, good general-purpose
      'HCOMPRESS_1' - hierarchical, slightly lossy unless scale=0 enforced

    Parameters
    ----------
    cube             : array-like, shape (N_wvl, H, W) or (N_src, N_wvl, H, W)
    output_path      : str or Path
    λ_full           : 1-D array, wavelengths in nm
    sources          : pd.DataFrame with 'x_peak' and 'y_peak' columns (0-based pixel coords),
                       required for the compact multi-extension 4D format
    compress         : bool, apply tile compression (default True)
    compression_type : str, FITS tile compression algorithm (default 'GZIP_2')
    """
    data = np.asarray(cube, dtype=np.float32)
    is_4d = data.ndim == 4

    def _wcs_λ(hdr, λ_full):
        hdr['CRVAL3'] = float(λ_full[0]);  hdr['CDELT3'] = float(λ_full[1] - λ_full[0])
        hdr['CUNIT3'] = 'nm';              hdr['CTYPE3'] = 'WAVE';  hdr['CRPIX3'] = 1

    def _make_hdu(arr, hdr):
        if compress:
            return fits.CompImageHDU(arr, header=hdr, compression_type=compression_type)
        return fits.ImageHDU(arr, header=hdr)

    if is_4d and sources is not None:
        # --- compact multi-extension format: one HDU per source, no zeros stored ---
        primary = fits.PrimaryHDU()
        primary.header['NSOURCES'] = (data.shape[0], 'Number of source PSF extensions')
        primary.header['NWVL']     = (data.shape[1], 'Number of wavelength slices')
        primary.header['COMMENT']  = 'Each extension contains one source PSF cube (N_wvl, H, W).'
        primary.header['COMMENT']  = 'CRVAL1/2 gives the source centroid in full-field pixel coords (1-based).'
        hdul = fits.HDUList([primary])

        psf_h, psf_w = data.shape[2], data.shape[3]
        crpix_x = (psf_w + 1) / 2.0  # centre of the PSF cutout (1-based)
        crpix_y = (psf_h + 1) / 2.0

        for i in range(data.shape[0]):
            hdr = fits.Header()
            # Spatial WCS: reference pixel is the PSF centre; CRVAL encodes its position
            # in the full field (FITS 1-based convention, hence +1)
            hdr['CRPIX1'] = crpix_x;               hdr['CRVAL1'] = float(sources.iloc[i]['x_peak']) + 1
            hdr['CDELT1'] = 1;                      hdr['CUNIT1'] = 'pixel'; hdr['CTYPE1'] = 'PIXEL'
            hdr['CRPIX2'] = crpix_y;               hdr['CRVAL2'] = float(sources.iloc[i]['y_peak']) + 1
            hdr['CDELT2'] = 1;                      hdr['CUNIT2'] = 'pixel'; hdr['CTYPE2'] = 'PIXEL'
            _wcs_λ(hdr, λ_full)
            hdr['SRCIDX']  = (i,     'Source index (0-based)')
            hdul.append(_make_hdu(data[i], hdr))

        hdul.writeto(str(output_path), overwrite=True)
        comp_label = f' ({compression_type} compressed)' if compress else ''
        print(f"Saved {data.shape[0]}-source multi-extension FITS{comp_label} to {output_path}")

    else:
        # --- single HDU fallback (3D, or 4D without source positions) ---
        hdr = fits.Header()
        hdr['CRVAL1'] = 1; hdr['CDELT1'] = 1; hdr['CUNIT1'] = 'pixel'; hdr['CTYPE1'] = 'PIXEL'; hdr['CRPIX1'] = 1
        hdr['CRVAL2'] = 1; hdr['CDELT2'] = 1; hdr['CUNIT2'] = 'pixel'; hdr['CTYPE2'] = 'PIXEL'; hdr['CRPIX2'] = 1
        _wcs_λ(hdr, λ_full)
        if is_4d:
            hdr['CTYPE4'] = 'OBJECT'; hdr['CRPIX4'] = 1; hdr['CRVAL4'] = 1; hdr['CDELT4'] = 1

        if compress:
            hdul = fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, header=hdr, compression_type=compression_type)])
        else:
            hdul = fits.HDUList([fits.PrimaryHDU(data, header=hdr)])

        hdul.writeto(str(output_path), overwrite=True)
        ndim_label = '4D' if is_4d else '3D'
        comp_label = f' ({compression_type} compressed)' if compress else ''
        print(f"Saved {ndim_label} model cube{comp_label} to {output_path}")


# stem = os.path.splitext(os.path.basename(cube_path))[0]
# suffix = '_modeled_cube_objects' if np.asarray(PSFs_separated).ndim == 4 else '_modeled_cube'
# output_file = data_folder / f'{stem}{suffix}.fits'
# Pass PSFs_separated (N_src, N_wvl, 111, 111) + source positions → compact multi-extension format, no zeros
# SaveModelCubeFITS(PSFs_separated, output_file, λ_full, sources=sources)

# %%
