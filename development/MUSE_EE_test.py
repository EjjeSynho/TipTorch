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

# cube_path  = data_folder / "reduced_cubes/CUBE_0001.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T04_56_21.169.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0001.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0002.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_11_59.783.fits.fz"
# cache_path = data_folder / "reduced_telemetry/CUBE_0002.pickle"

cube_path  = data_folder / "reduced_cubes/CUBE_0003.fits"
raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_28_41.014.fits.fz"
cache_path = data_folder / "reduced_telemetry/CUBE_0003.pickle"

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


#%% For predicted and fitted model inputs, it is convenient to organize them using inputs_manager
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from machine_learning.calibrators.NFM_calibrator import NFMCalibrator

# The model config is also updated to simulate only sparse λs
model_config['sources_science']['Wavelength'] = torch.tensor(λ_sparse, device=device, dtype=torch.float32) * 1e-9 #[m]
model_config['telescope']['PupilAngle']       = torch.tensor(model_config['telescope']['PupilAngle'], device=device)

PSF_model = PSFModelNFM(
    model_config,
    multiple_obs    = False,
    LO_NCPAs        = True,
    chrom_defocus   = False,
    use_Moffat = False,
    retain_PSDs     = True,
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

PSF_model.model.F *= 0.0
PSF_model.model.F += 1.0


#%%
with torch.inference_mode():
    quasi_inf_PSF_size = 911

    current_PSF_size = PSF_model.model.N_pix # the actual size of the simulated PSFs
    # Ensure that if PSF's small size is odd, then quasi-infinite PSF size is also odd to avoid sub-pixel shifts of the PSF core
    quasi_inf_PSF_size -= (1-current_PSF_size % 2)

    if PSF_model.use_splines:
        wvl_current = PSF_model.λ_sim.clone()
        PSF_model.SetWavelengths(PSF_model.λ_ctrl) # isntead of anchor λs, evaluate at spline nodes

    PSF_small = PSF_model.forward(src_ids=0) # compute only for the first source ignoring the field variability (just for speed's sake)
    PSDs_small = { k: v.clone().detach() for k,v in PSF_model.model.PSDs.items() }

    PSF_model.SetImageSize(quasi_inf_PSF_size) # quasi-infinite PSF image to compute how much flux is lost while cropping

    PSF_inf = PSF_model.forward(src_ids=0)
    PSDs_inf = { k: v.clone().detach() for k,v in PSF_model.model.PSDs.items() }
    
    PSF_model.SetImageSize(current_PSF_size)

    if PSF_model.use_splines:
        PSF_model.SetWavelengths(wvl_current) # switch back to the original wavelengths

    flux_core_radius = 2 # [pix] 
    N_core_pixels = (flux_core_radius*2 + 1)**2  # [pix²], this expression assumes a square mask for the core flux estimation

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


# %%
von_Karman_small = PSDs_small['fitting'].real.squeeze()
von_Karman_inf   = PSDs_inf['fitting'].real.squeeze()

#%%

inf_crop = np.s_[
    von_Karman_inf.shape[0]//2 - von_Karman_small.shape[0]//2 : von_Karman_inf.shape[0]//2 + von_Karman_small.shape[0]//2 + 1,
     - von_Karman_small.shape[1] : 
]

von_Karman_inf_cropped = von_Karman_inf[inf_crop]

#%
# fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# vmin = min(von_Karman_small.min().item(), von_Karman_inf_cropped.min().item()) + 1e-8
# vmax = max(von_Karman_small.max().item(), von_Karman_inf_cropped.max().item())
# norm = LogNorm(vmin=vmin, vmax=vmax)

# axes[0].imshow(von_Karman_small.cpu().numpy(),       norm=norm); axes[0].set_title('Small')
# axes[1].imshow(von_Karman_inf_cropped.cpu().numpy(), norm=norm); axes[1].set_title('Inf (cropped)')
# axes[2].imshow((von_Karman_inf_cropped - von_Karman_small).cpu().numpy()); axes[2].set_title('Difference')
# plt.tight_layout()
# plt.show()

PSD_inf_outside = von_Karman_inf.clone()
PSD_inf_outside[inf_crop] = 0

PSD_crop_ratio = PSD_inf_outside.sum().item() / von_Karman_inf.sum().item() * 100

print('Energy ratio:', PSD_crop_ratio, '%')

#%%
λ_anchor = np.round(PSF_model.λ_ctrl.cpu().numpy() * 1e9).astype(int)

el_croppo = (1.0 - crop_ratio.cpu().numpy()) * 100.0

_ = plt.figure(figsize=(12,8))

plt.plot(λ_anchor, el_croppo, label='PSF crop ratio', color='blue', linewidth=2)
plt.axhline(y=PSD_crop_ratio, label='PSD crop ratio', color='orange', linewidth=2)
plt.ylim(0, 8)
plt.xlabel('Wavelength, [nm]')
plt.ylabel('Crop ratio, [%]')
plt.title('Crop ratio vs. λ for MUSE NFM')
plt.legend()
plt.grid()
plt.show()

# %%