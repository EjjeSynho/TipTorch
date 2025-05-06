#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torchmin import minimize

from astropy.stats import sigma_clipped_stats

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, RectangularAperture
from sklearn.cluster import DBSCAN
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages
from tools.utils import plot_radial_profiles_new, draw_PSF_stack, mask_circle
from managers.config_manager import ConfigManager
from data_processing.normalizers import CreateTransformSequenceFromFile
from managers.input_manager import InputsTransformer
from tqdm import tqdm
from project_globals import MUSE_DATA_FOLDER, device
from astropy.io import fits
from scipy.ndimage import binary_dilation

from machine_learning.MUSE_onsky_df import *


#%%
# Load the FITS file

reduced_name = 'J0259_all'
# reduced_name = 'J0259_2024-12-05T03_04_07.007'
# reduced_name = 'J0259_2024-12-05T03_49_24.768'
# reduced_name = 'J0259_2024-12-05T03_15_37.598'

if reduced_name == 'J0259_all':
    cube_path = MUSE_DATA_FOLDER + f"quasars/J0259_cubes/J0259-0901_all.fits"
else:
    cube_path = MUSE_DATA_FOLDER + f"quasars/J0259_cubes/J0259-0901_DATACUBE_FINAL_{reduced_name[6:]}.fits"

test_fits = fits.open(cube_path)

data_full = np.nan_to_num(test_fits[1].data, nan=0.0)

# Compute the mask of valid pixels
nan_mask = np.abs(np.nansum(test_fits[1].data, axis=0)) < 1e-12
nan_mask = binary_dilation(nan_mask, iterations=2, )
valid_mask = ~nan_mask

test_fits.close()

data_full = data_full * valid_mask[np.newaxis, :, :]

valid_mask = torch.tensor(valid_mask, device=device).float().unsqueeze(0)

# Load processed data file
with open(MUSE_DATA_FOLDER + f"quasars/J0259_reduced/{reduced_name}.pickle", 'rb') as f:
    data_sample = pickle.load(f)

# Compute wavelength data
λ_min, λ_max, Δλ_full = data_sample['spectral data']['wvl range']
λ_bins = data_sample['spectral data']['wvl bins']
Δλ_binned = np.median(np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])]))

if hasattr(λ_max, 'item'): # To compensate for a small error in the data reduction routine
    λ_max = λ_max.item()

λ = np.linspace(λ_min, λ_max, np.round((λ_max-λ_min)/Δλ_full+1).astype('int'))
assert len(λ) == data_full.shape[0]

if reduced_name == 'J0259_2024-12-05T03_15_37.598':
    host_coords  = [53, 37]
    AGN_coords_1 = [47, 39]
    bg_coords    = [56, 39]
    AGN_coords_2 = [61, 37]
    lens_coords  = [55, 65]
    
elif reduced_name == 'J0259_2024-12-05T03_49_24.768':
    host_coords  = [44, 30]
    AGN_coords_1 = [37, 28]
    bg_coords    = [12, 62]
    AGN_coords_2 = [52, 30]
    lens_coords  = [46, 58]
    
elif reduced_name == 'J0259_2024-12-05T03_04_07.007':
    host_coords  = [46, 37]
    AGN_coords_1 = [38, 35]
    bg_coords    = [10, 57]
    AGN_coords_2 = [53, 36]
    lens_coords  = [47, 65]
  
elif reduced_name == 'J0259_all':
    host_coords  = [53, 37]
    AGN_coords_1 = [46, 35]
    bg_coords    = [10, 58]
    AGN_coords_2 = [60, 37]
    lens_coords  = [55, 66]
else: 
    print('No coordinates for this reduced_name')

#%%
data_onsky, _, _, _ = LoadImages(data_sample, device=device, subtract_background=False, normalize=False, convert_images=True)
data_onsky = data_onsky.squeeze()
data_onsky *= valid_mask

# Correct the flux to match MUSE cube
data_onsky = data_onsky * (data_full.sum(axis=0).max() /  data_onsky.sum(axis=0).max())

# Extract config file and update it
config_file, data_onsky = GetConfig(data_sample, data_onsky)
data_onsky = data_onsky.squeeze()

config_file['NumberSources'] = 1 #N_src
# The bigger size of initialized PSF is needed to extract the flux loss due to cropping to the box_size later
config_file['sensor_science']['FieldOfView'] = 511
# Select only a subset of predicted wavelengths and modify the config file accordingly
wavelength = config_file['sources_science']['Wavelength'].clone()
# Assumes that we are not in the pupil tracking mode
config_file['telescope']['PupilAngle'] = torch.zeros(1, device=device)

ids_wavelength_selected = np.arange(0, wavelength.shape[-1], 2)
wavelength_selected = wavelength[..., ids_wavelength_selected]
config_file['sources_science']['Wavelength'] = wavelength_selected

N_wvl = len(ids_wavelength_selected)

data_sparse = data_onsky.clone()[ids_wavelength_selected,...]

#%%
from managers.multisrc_manager import detect_sources

data_src = data_onsky.sum(dim=0).cpu().numpy()
# mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)

PSF_size = 111  # Define the size of each ROI (in pixels)
thres = 50000

sources   = detect_sources(data_src, threshold=thres, box_size=11, verbose=True)
srcs_pos  = np.transpose((sources['x_peak'], sources['y_peak']))
srcs_flux = sources['peak_value'].to_numpy()

# Draw the detected sources
apertures = CircularAperture(srcs_pos, r=5)
apertures_box = RectangularAperture(srcs_pos, PSF_size//2, PSF_size//2)

norm_field = LogNorm(vmin=10, vmax=thres*10)

plt.imshow(np.abs(data_src), norm=norm_field, origin='lower', cmap='gray')
apertures_box.plot(color='gold', lw=2, alpha=0.45)
plt.show()

#%%
from managers.multisrc_manager import extract_ROIs, add_ROIs


ROIs, local_coords, global_coords, _ = extract_ROIs(data_sparse, sources, box_size=PSF_size)
N_src = len(ROIs)
# plot_ROIs_as_grid(ROIs, cols=np.ceil(np.sqrt(len(ROIs))).astype('uint'))  # Adjust the number of columns as needed

# Extract the spectrum per target near the PSF peak
def GetSpectrum(data, point, radius=1, debug_show_ROI=False):
    if type(point) == list:
        point = np.array(point)
    elif type(point) == pd.Series:
        point = point.to_numpy()
    elif type(point) == torch.Tensor:
        point = point.cpu().numpy()
    else:
        raise ValueError('Point must be a list, ndarray, pandas Series, or torch Tensor')

    x, y = point[:2].astype('int')
    
    if radius == 0:
        return data[:, y, x]
    
    y_min = max(0, y - radius)
    y_max = min(data.shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(data.shape[2], x + radius + 1)

    if torch.is_tensor(data):
        if debug_show_ROI:
            plt.imshow(data[:, y_min:y_max, x_min:x_max].sum(dim=0).cpu().numpy(), origin='lower')
            plt.show()
        return data[:, y_min:y_max, x_min:x_max].mean(dim=(1,2))
    else:
        if debug_show_ROI:
            plt.imshow(data[:, y_min:y_max, x_min:x_max].sum(axis=0), origin='lower')
            plt.show()
        return np.nanmean(data[:, y_min:y_max, x_min:x_max], axis=(1,2))


flux_λ_norm = Δλ_full / Δλ_binned  # Correct for the difference in energy per λ bin

flux_core_radius = 2  # [pix]
N_core_pixels = (flux_core_radius*2 + 1)**2  # [pix^2]

# It tells the average flux in the PSF core for each source
src_spectra_sparse = [GetSpectrum(data_sparse, sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_binned = [GetSpectrum(data_onsky,  sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_full   = [GetSpectrum(data_full,   sources.iloc[i], radius=flux_core_radius) for i in range(N_src)]

# STRANGE LINE IN THE FIRST SPECTRUM!!!!!
#%%
i_src = 1

plt.plot(λ, src_spectra_full[i_src], linewidth=0.25, alpha=0.3)
plt.scatter(wavelength.squeeze().cpu().numpy()*1e9, src_spectra_binned[i_src].cpu().numpy())
plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_sparse[i_src].cpu().numpy())


#%%
with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

df = data_sample['All data']
df['ID'] = 0
df.loc[0, 'Pupil angle'] = 0.0

df_pruned  = prune_columns(df.copy())
df_reduced = reduce_columns(df_pruned.copy())
df_transforms = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_norm = normalize_df(df_reduced, df_transforms)
df_norm = df_norm.fillna(0)

selected_entries_input = muse_df_norm.columns.values.tolist()


#%
'''
config_file['DM']['OptimizationZenith']  = config_file['DM']['OptimizationZenith'].repeat(N_src)
config_file['DM']['OptimizationAzimuth'] = config_file['DM']['OptimizationAzimuth'].repeat(N_src)

config_file['RTC']['SensorFrameRate_HO'] = config_file['RTC']['SensorFrameRate_HO'].repeat(N_src)
config_file['RTC']['LoopDelaySteps_HO'] = config_file['RTC']['LoopDelaySteps_HO'].repeat(N_src)
config_file['RTC']['LoopGain_HO'] = config_file['RTC']['LoopGain_HO'].repeat(N_src)

config_file['sensor_HO']['NumberPhotons'] = config_file['sensor_HO']['NumberPhotons'].repeat(N_src, 1)
config_file['atmosphere']['Seeing'] = config_file['atmosphere']['Seeing'].repeat(N_src)

config_file['atmosphere']['WindSpeed'] = config_file['atmosphere']['WindSpeed'].repeat(N_src, 1)
config_file['atmosphere']['WindDirection'] = config_file['atmosphere']['WindDirection'].repeat(N_src, 1)
# config_file['atmosphere']['Cn2Weights'] = config_file['atmosphere']['Cn2Weights'].repeat(N_src, 1)
# config_file['atmosphere']['Cn2Heights'] = config_file['atmosphere']['Cn2Heights'].repeat(N_src, 1)

config_file['sources_science']['Zenith'] = config_file['sources_science']['Zenith'].repeat(N_src)
config_file['sources_science']['Azimuth'] = config_file['sources_science']['Azimuth'].repeat(N_src)
config_file['sources_HO']['Wavelength'] = config_file['sources_HO']['Wavelength'].repeat(N_src, 1)
'''


#%%
from tools.utils import PupilVLT, OptimizableLO
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch
# from tools.utils import SausageFeature

LO_map_size = 31

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          False
}
model = TipTorch(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
model.apodizer = model.make_tensor(1.0)

model.to_float()
model.to(device)
#%
# PSF_1 = model()

#%%
from data_processing.normalizers import Uniform
from managers.input_manager import InputsManager

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')

inputs_manager = InputsManager()

inputs_manager.add('r0',    torch.tensor([model.r0[0].item()]), df_transforms_fitted['r0'])
inputs_manager.add('F',     torch.tensor([[1.0,]*N_wvl]),    df_transforms_fitted['F'])
inputs_manager.add('bg',    torch.tensor([[0,]*N_wvl]),      df_transforms_fitted['bg'])
inputs_manager.add('dx',    torch.tensor([[0.0,]*N_wvl]),    df_transforms_fitted['dx'])
inputs_manager.add('dy',    torch.tensor([[0.0,]*N_wvl]),    df_transforms_fitted['dy'])
inputs_manager.add('dn',    torch.tensor([1.5]),             df_transforms_fitted['dn'])
inputs_manager.add('Jx',    torch.tensor([[10,]*N_wvl]),     df_transforms_fitted['Jx'])
inputs_manager.add('Jy',    torch.tensor([[10,]*N_wvl]),     df_transforms_fitted['Jy'])
inputs_manager.add('Jxy',   torch.tensor([[0]]), df_transforms_fitted['Jxy'])
inputs_manager.add('amp',   torch.tensor([0.0]), df_transforms_fitted['amp'])
inputs_manager.add('b',     torch.tensor([0.0]), df_transforms_fitted['b'])
inputs_manager.add('alpha', torch.tensor([4.5]), df_transforms_fitted['alpha'])
inputs_manager.add('beta',  torch.tensor([2.5]), df_transforms_fitted['beta'])
inputs_manager.add('ratio', torch.tensor([1.0]), df_transforms_fitted['ratio'])
inputs_manager.add('theta', torch.tensor([0.0]), df_transforms_fitted['theta']) 
inputs_manager.add('s_pow', torch.tensor([0.0]), df_transforms_fitted['s_pow'])


if LO_map_size is not None:
    inputs_manager.add('LO_coefs', torch.zeros([1, LO_map_size**2]), Uniform(a=-100, b=100))
    inputs_manager.set_optimizable('LO_coefs', False)

inputs_manager.set_optimizable(['ratio', 'theta', 'alpha', 'beta', 'amp', 'b'], False)
inputs_manager.set_optimizable(['Jxy'], False)

inputs_manager.to_float()
inputs_manager.to(device)

print(inputs_manager)


inputs_manager_objs = InputsManager()

inputs_manager_objs.add('dx', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dx'])
inputs_manager_objs.add('dy', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dy'])
inputs_manager_objs.add('F_norm', torch.tensor([[1.0,]]*N_src), df_transforms_fitted['F'])

inputs_manager_objs.to_float()
inputs_manager_objs.to(device)

inputs_manager_objs.set_optimizable(['F_norm'], False)

print(inputs_manager_objs)


#%%
from machine_learning.calibrator import Calibrator, Gnosis


calibrator = Calibrator(
    inputs_manager=inputs_manager,
    predicted_values = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha'],
    device=device,
    calibrator_network = {
        'artichitecture': Gnosis,
        'inputs_size': len(selected_entries_input),
        'NN_kwargs': {
            'hidden_size': 200,
            'dropout_p': 0.1
        },
        'weights_folder': '../data/weights/gnosis_MUSE_v3_7wvl_yes_Mof_no_ssg.dict'
    }
)

calibrator.eval()


#%%
# pred_inputs = normalizer.unstack(net(NN_inp))
pred_inputs = calibrator(df_norm[selected_entries_input].loc[0])

with torch.no_grad():
    PSF_pred_big = model(pred_inputs).clone() # First initial prediction of the "big" PSF

#%%
config_file['sensor_science']['FieldOfView'] = PSF_size # Set the actual size of the simulated PSFs
# model.Update(reinit_grids=True, reinit_pupils=True)
model.Update(init_grids=True, init_pupils=True, init_tomography=True)

with torch.no_grad():
    PSF_pred_small = model(pred_inputs)

inputs_manager.update(pred_inputs)

#%%
# How much flux is cropped by assuming the finite size of the PSF box (PSF_predbif is assumed to be quasi-infinite)
crop_ratio = (PSF_pred_big.amax(dim=(-2,-1)) / PSF_pred_small.amax(dim=(-2,-1))).squeeze()

core_mask     = torch.tensor(mask_circle(PSF_size, flux_core_radius+1)[None,None,...]).to(device).float()
core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], flux_core_radius+1)[None,None,...]).to(device).float()
 
# How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob"
core_flux_ratio = torch.squeeze((PSF_pred_big*core_mask_big).sum(dim=(-2,-1), keepdim=True) / PSF_pred_big.sum(dim=(-2,-1), keepdim=True))
PSF_norm_factor = N_core_pixels / flux_λ_norm / core_flux_ratio / crop_ratio

#%%

def func_dxdy(x_):
    dxdy_inp = inputs_manager_objs.unstack(x_.unsqueeze(0), update=False) # Don't update interal values yet
    dxdy_inp['dx'] = dxdy_inp['dx'].repeat(1, N_wvl) # Extend to simulated number of wavelength
    dxdy_inp['dy'] = dxdy_inp['dy'].repeat(1, N_wvl) # assuming the same shift for all wavelengths
    return model(pred_inputs | dxdy_inp)


def fit_dxdy(i_src, verbose=0):
    dxdy_0 = inputs_manager_objs.stack()[i_src,:]
    PSF_data = torch.nan_to_num(ROIs[i_src].unsqueeze(0)) * flux_λ_norm / src_spectra_sparse[i_src][None,:,None,None]
    peaks_scaler = PSF_pred_big.amax(dim=(-2,-1)) / (PSF_data*core_mask).amax(dim=(-2,-1))
    PSF_data *= peaks_scaler[:, :, None, None] # a sort of normalizing fetch-factor
    
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_data*core_mask, func_dxdy(dxdy_)*core_mask, reduction='sum')*1e3
    result = minimize(loss, dxdy_0, max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    # Update managers internal values with the new fitted values
    dxdy_1 = inputs_manager_objs.unstack(result.x.unsqueeze(0), update=False)
    inputs_manager_objs['dx'][i_src,:] = dxdy_1['dx'].flatten()
    inputs_manager_objs['dy'][i_src,:] = dxdy_1['dy'].flatten()
    
    return func_dxdy(result.x).detach().clone().squeeze(), result.x.detach().clone()


PSFs_fitted = []

# The PSF model generates PSFs normalized to sum of 1 per wavelength. We need to normalize them to the flux of the sources.
# To do so, we need to account for: 1. the flux normalization factor, 2. the crop ratio, 3. the core to wings flux ratio.
for i in tqdm(range(N_src)):
    PSF_fitted, dxdy = fit_dxdy(i, verbose=0)
    PSFs_fitted.append(PSF_fitted * (src_spectra_sparse[i] * PSF_norm_factor)[:,None,None])
    
PSFs_fitted = torch.stack(PSFs_fitted, dim=0)


#%%
model_sparse = add_ROIs(
    torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device), # blanck baase image
    [PSFs_fitted[i,...] for i in range(N_src)], # predicted flux-normalized PSFs after coordinates tuning
    local_coords,
    global_coords
)

# It tells the average flux in the PSF core for each source
src_spectra_sparse = [GetSpectrum(data_sparse,  sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_fitted = [GetSpectrum(model_sparse, sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]

# STRANGE LINE IN THE FIRST SPECTRUM!!!!!
#%
# i_src = 1
# plt.plot(λ, src_spectra_full[i_src], linewidth=0.25, alpha=0.3)
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_fitted[i_src].cpu().numpy())
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_sparse[i_src].cpu().numpy())

#%%
from managers.multisrc_manager import VisualizeSources, PlotSourcesProfiles

ROI_plot = np.s_[..., 125:225, 125:225]

VisualizeSources(data_sparse, model_sparse, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(data_sparse, model_sparse, sources, radius=16, title='Predicted PSFs')


#%% ====================================================== Fitting =======================================================
from tools.utils import OptimizableLO

LO_basis = OptimizableLO(model, ignore_pupil=False)

inputs_manager.set_optimizable('LO_coefs', False)
inputs_manager.set_optimizable('Jxy', False)

inputs_manager.delete('amp')
inputs_manager.delete('beta')
inputs_manager.delete('alpha')
inputs_manager.delete('b')
inputs_manager.delete('ratio')
inputs_manager.delete('theta')

inputs_manager.delete('dx')
inputs_manager.delete('dy')
inputs_manager.delete('s_pow')

print(inputs_manager)

inputs_manager_objs.set_optimizable('F_norm', True)

#%%
x0 = torch.cat([
    inputs_manager.stack().flatten(),
    inputs_manager_objs.stack().flatten(),
])
x_size = inputs_manager.get_stacked_size()

empty_img   = torch.zeros([N_wvl, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 0 + 1

#%%
def func_fit(x): # TODO: relative weights for different brigtness
    PSFs_fit = []
    for i in range(N_src):
        params_dict = inputs_manager.unstack(x[:x_size].unsqueeze(0))
        F_dxdy_dict = inputs_manager_objs.unstack(x[x_size:].view(N_src, -1))

        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))

        F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | F_dxdy_dict
        flux_norm = (src_spectra_sparse[i] * PSF_norm_factor)[:,None,None] * F_dxdy_dict['F_norm'][i]

        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
        
    return add_ROIs( empty_img*0.0, PSFs_fit, local_coords, global_coords )


def loss_fit(x_):
    PSFs_ = func_fit(x_)
    l1 = F.smooth_l1_loss(data_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    l2 = F.mse_loss(data_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    return l1 * 1e-3 + l2 * 5e-6


_ = func_fit(x0)

#%%
result_global = minimize(loss_fit, x0, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x0 = result_global.x.clone()
x_fit_dict = inputs_manager.unstack(x0.unsqueeze(0), include_all=False)

#%% 
# with torch.no_grad():
model_fit = func_fit(result_global.x).detach()

VisualizeSources(data_sparse, model_fit, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(data_sparse, model_fit, sources, radius=16, title='Fitted PSFs')


#%% 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
from tools.misc import QuadraticModel


λ_sparse = wavelength_selected.flatten() * 1e9 # [nm]

F_model    = QuadraticModel(λ_sparse)
Jx_model   = QuadraticModel(λ_sparse)
Jy_model   = QuadraticModel(λ_sparse)
norm_model = QuadraticModel(λ_sparse)

params_Jx   = Jx_model.fit(inputs_manager['Jx'].flatten())
params_Jy   = Jy_model.fit(inputs_manager['Jy'].flatten())
params_F    = F_model.fit(inputs_manager['F'].flatten())
params_norm = norm_model.fit(PSF_norm_factor.flatten(), [2, 1, 1e3])

Fx_curve_fit   = Jx_model(params_Jx)
Fy_curve_fit   = Jy_model(params_Jy)
F_curve_fit    = F_model (params_F)
norm_curve_fit = norm_model(params_norm)


#%%
curve_inputs = InputsManager()

curve_inputs.add('Jx_A', torch.tensor([params_Jx[0]]), Uniform(a=-5e-5, b=5e-5))
curve_inputs.add('Jx_B', torch.tensor([params_Jx[1]]), Uniform(a=-2e-2, b=2e-2))
curve_inputs.add('Jx_C', torch.tensor([params_Jx[2]]), Uniform(a=30,    b=60))

curve_inputs.add('Jy_A', torch.tensor([params_Jy[0]]), Uniform(a=-5e-5, b=5e-5))
curve_inputs.add('Jy_B', torch.tensor([params_Jy[1]]), Uniform(a=-5e-2, b=5e-2))
curve_inputs.add('Jy_C', torch.tensor([params_Jy[2]]), Uniform(a=30,    b=60))

curve_inputs.add('F_A',   torch.tensor([params_F[0]]),  Uniform(a=5e-7,  b=10e-7))
curve_inputs.add('F_B',   torch.tensor([params_F[1]]),  Uniform(a=-2e-3, b=0))
curve_inputs.add('F_C',   torch.tensor([params_F[2]]),  Uniform(a=0,     b=2))

curve_inputs.add('norm_A', torch.tensor([params_norm[0]]), Uniform(a=0, b=2e-2))
curve_inputs.add('norm_B', torch.tensor([params_norm[1]]), Uniform(a=50, b=-20))
curve_inputs.add('norm_C', torch.tensor([params_norm[2]]), Uniform(a=2e4, b=4e4))

curve_inputs.set_optimizable(['norm_A', 'norm_B', 'norm_C'], False)

curve_inputs.to_float()
curve_inputs.to(device)

x_0_curve = curve_inputs.stack()
# print(x_0_curve.shape)

#%%
curve_params_ = curve_inputs.unstack(x_0_curve)

curve_sample = lambda x, curve_p, p_name: Jx_model.quadratic_function(
    x,
    curve_p[f'{p_name}_A'],
    curve_p[f'{p_name}_B'],
    curve_p[f'{p_name}_C']
)

Jx_new   = curve_sample(λ_sparse, curve_params_, 'Jx')
Jy_new   = curve_sample(λ_sparse, curve_params_, 'Jy')
F_new    = curve_sample(λ_sparse, curve_params_, 'F')
norm_new = curve_sample(λ_sparse, curve_params_, 'norm')

#%
plt.figure(figsize=(10, 6))

plt.plot(λ_sparse.cpu(), inputs_manager['Jx'].flatten().cpu(), label='Data', color='tab:blue')
plt.plot(λ_sparse.cpu(), Fx_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:blue')
plt.scatter(λ_sparse.cpu(), Jx_new.cpu(), label='New Quadratic Curve', color='tab:blue', marker='x')

plt.plot(λ_sparse.cpu(), inputs_manager['Jy'].flatten().cpu(), label='Data', color='tab:orange')
plt.plot(λ_sparse.cpu(), Fy_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:orange')
plt.scatter(λ_sparse.cpu(), Jy_new.cpu(), label='New Quadratic Curve', color='tab:orange', marker='x')
plt.xlabel('Wavelength [nm]')
plt.legend()
plt.grid(True)
plt.show()

# plt.plot(λ_sparse.cpu(), inputs_manager['norm'].flatten().cpu(), label='Data', color='tab:orange')
plt.plot(λ_sparse.cpu(), PSF_norm_factor.flatten().cpu(), label='Data', color='tab:orange')
plt.plot(λ_sparse.cpu(), norm_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:orange')
plt.scatter(λ_sparse.cpu(), norm_new.cpu(), label='New Quadratic Curve', color='tab:orange', marker='x')
plt.xlabel('Wavelength [nm]')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(λ_sparse.cpu(), inputs_manager['F'].flatten().cpu(), label='Data', color='tab:green')
plt.plot(λ_sparse.cpu(), F_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:green')
plt.scatter(λ_sparse.cpu(), F_new.cpu(), label='New Quadratic Curve', color='tab:green', marker='x')
plt.xlabel('Wavelength [nm]')
plt.legend()
plt.grid(True)
plt.show()

#%%
inputs_manager.set_optimizable(['F', 'Jx', 'Jy'],  False)
inputs_manager_objs.set_optimizable(['dx', 'dy'], False)

print(inputs_manager)

#%%
x_size_model = inputs_manager.get_stacked_size()
x_size_curve = curve_inputs.get_stacked_size()
x_size_total = x_size_curve + x_size_model

x2 = torch.cat([
    inputs_manager.stack().flatten(),
    curve_inputs.stack().flatten(),
    inputs_manager_objs.stack().flatten(),
])

empty_img = torch.zeros([N_wvl, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)

#%%
def func_fit_curve(x):
    PSFs_fit = []
    for i in range(N_src):
        params_dict = inputs_manager.unstack(x[:x_size_model].unsqueeze(0))
        curve_p_    = curve_inputs.unstack(x[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
        F_dxdy_dict = inputs_manager_objs.unstack(x[x_size_curve+x_size_model:].view(N_src, -1))
        
        curve_dict = {p: curve_sample(λ_sparse, curve_p_, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
        
        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        # F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        # F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        F_dxdy_dict['dx'] = inputs_manager_objs['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = inputs_manager_objs['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | curve_dict | F_dxdy_dict
        flux_norm = (src_spectra_sparse[i] * norm_new)[:,None,None] * F_dxdy_dict['F_norm'][i]
        
        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
    
    return add_ROIs( empty_img*0.0, PSFs_fit, local_coords, global_coords )


#%%
def loss_fit_curve(x_):
    PSFs_ = func_fit_curve(x_)
    l1 = F.smooth_l1_loss(data_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    l2 = F.mse_loss(data_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    return l1 * 1e-3 + l2 * 5e-6


_ = loss_fit_curve(x2)
    
#%%
result_global = minimize(loss_fit_curve, x2, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x2 = result_global.x.clone().detach()

x_fit_dict = inputs_manager.unstack(x2[:x_size_model].unsqueeze(0))
x_curve_fit_dict = curve_inputs.unstack(x2[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
flux_corrections = inputs_manager_objs['F_norm']

#%% 
model_fit_curves = func_fit_curve(result_global.x).detach()

VisualizeSources(data_sparse, model_fit_curves, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(data_sparse, model_fit_curves, sources, radius=16, title='Fitted PSFs')

#%%
torch.cuda.empty_cache()
model_inputs_full_λ = {p: curve_sample(torch.as_tensor(λ, device=device), x_curve_fit_dict, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
norms_new_full_λ = curve_sample(torch.as_tensor(λ, device=device), x_curve_fit_dict, 'norm')

λ_split_size = 100
# Split λ array into batches
λ_batches = [λ[i:i + λ_split_size] for i in range(0, len(λ), λ_split_size)]

model_full = []

for batch_id in tqdm(range(len(λ_batches))):
    batch_size = len(λ_batches[batch_id])
    config_file['sources_science']['Wavelength'] = torch.as_tensor(λ_batches[batch_id]*1e-9, device=device).unsqueeze(0)
    model.Update(init_grids=True, init_pupils=True, init_tomography=True)

    empty_img = torch.zeros([batch_size, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)
    PSF_batch = []

    for i in range(N_src):
        batch_ids = slice(batch_id*batch_size, (batch_id+1)*batch_size)

        dict_selected = {
            key: model_inputs_full_λ[key][:,batch_ids]
            for key in model_inputs_full_λ.keys()
        }

        dxdy_dict = {
            'dx': inputs_manager_objs['dx'][i].unsqueeze(-1).repeat(batch_size).unsqueeze(0), # Extend to simulated number of wavelength
            'dy': inputs_manager_objs['dy'][i].unsqueeze(-1).repeat(batch_size).unsqueeze(0) # assuming the same shift for all wavelengths
        }

        dict_selected = x_fit_dict | dict_selected | dxdy_dict
        dict_selected['bg'] = torch.zeros([1, batch_size], device=device)

        del dict_selected['LO_coefs']

        flux_norm = norms_new_full_λ[batch_ids]\
            * flux_corrections[i]\
            * torch.as_tensor(src_spectra_full[i][batch_ids], device=device) \
            * flux_λ_norm

        PSF_batch.append( (model(dict_selected).squeeze() * flux_norm[:,None,None]).detach() )

    model_full.append( add_ROIs( empty_img*0.0, PSF_batch, local_coords, global_coords ).cpu().numpy() )

model_full = np.vstack(model_full)


#%%
norm_full = LogNorm(vmin=1e1, vmax=thres*10)

VisualizeSources(data_full, model_full, norm=norm_full, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(data_full, model_full, sources, radius=16, title='Fitted PSFs')

diff_img_full = (data_full - model_full) * valid_mask.cpu().numpy()

#%%
from astropy.convolution import convolve, Box1DKernel


# Define the coordinates and their properties
targets_info = [
    {
        'coords': host_coords,
        'name': 'Host galaxy?',
        'color': 'tab:blue',
        'radius': 2
    },
    {
        'coords': bg_coords,
        'name': 'Background',
        'color': 'tab:orange',
        'radius': 5
    },
    # {
    #     'coords': AGN_coords_1,
    #     'name': 'AGN im. #1 (after subtraction)',
    #     'color': 'tab:red',
    #     'radius': 1
    # },
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

# Create kernel for convolution
kernel = Box1DKernel(10)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10,6))

labels, colors = [], []
dict_store = {}
# Loop through each spectrum
for info in targets_info:
    spectrum_full = GetSpectrum(diff_img_full[ROI_plot], info['coords'], radius=info['radius'], debug_show_ROI=False )
    spectrum_avg = convolve(spectrum_full, kernel, boundary='extend')
    dict_store[info['name']] = spectrum_full
    
    labels.append(info['name'])
    colors.append(info['color'])
    
    plt.plot(λ, spectrum_full, linewidth=0.5, alpha=0.25, label=info['name'], color=info['color'])
    plt.plot(λ, spectrum_avg, linewidth=1, alpha=1, color=info['color'], linestyle='--')
    

    
    
plt.legend()
# plt.ylim(0, None)
plt.xlim(λ.min(), λ.max())
plt.grid(alpha=0.2)
# plt.title('Full-spectrum: (AGN, AGN residue, host(?), background)')
plt.title('Full-spectrum: (AGN residue, Host(?), background)')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()

with open(MUSE_DATA_FOLDER+f'quasars/spectra/{reduced_name}.pkl', 'wb') as f:
    pickle.dump(dict_store, f)
    
#%%
from tools.utils import wavelength_to_rgb

λ_vis = np.linspace(440, 750, diff_img_full.shape[0])

def plot_wavelength_rgb(image, wavelengths=None, min_val=1e-3, max_val=1e1, title=None, show=True):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        
    wavelengths = np.asarray(wavelengths)
    rgb_weights = np.array([wavelength_to_rgb(λ, show_invisible=True) for λ in wavelengths]).T

    weighted = rgb_weights[:, :, None, None] * image[None, :, :, :]
    image_RGB = np.abs(weighted.sum(axis=1))  # shape: (3, height, width)
    image_RGB = np.moveaxis(image_RGB, 0, -1)

    log_min, log_max = np.log10(min_val), np.log10(max_val)
    image_log = np.log10(image_RGB+1e-10)

    image_clipped = np.clip(image_log, log_min, log_max)
    norm_image = (image_clipped - log_min) / (log_max - log_min)

    if show:
        plt.figure()
        plt.imshow(norm_image, origin="lower")
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    return image_RGB


diff_rgb = plot_wavelength_rgb(
    diff_img_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"{reduced_name}\nDifference",
    min_val=500, max_val=60000, show=True
)


diff_rgb = plot_wavelength_rgb(
    data_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"{reduced_name}\nObservation",
    min_val=500, max_val=200000, show=True
)

diff_rgb = plot_wavelength_rgb(
    model_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"{reduced_name}\nModel",
    min_val=500, max_val=200000, show=True
)



# %%
spectra_dicts = []

reduced_names = ['J0259_all',  'J0259_2024-12-05T03_04_07.007',  'J0259_2024-12-05T03_49_24.768',  'J0259_2024-12-05T03_15_37.598']
for name in reduced_names:
    with open(MUSE_DATA_FOLDER+f'quasars/spectra/{name}.pkl', 'rb') as f:
        dict_spectrum = pickle.load(f)
    spectra_dicts.append(dict_spectrum)

#%%

# Define the targets and their plotting properties
targets = [
    ('Host galaxy?', 'tab:blue'),
    ('Background', 'tab:orange'),
    ('AGN im. #2 (after subtraction)', 'tab:green'),
    ('Lens', 'tab:purple')
]

# Initialize dictionaries to store means and stds
means = {}
stds = {}

# Calculate means and standard deviations for each target
for target, _ in targets:
    spectra = [spectra_dicts[i][target] for i in range(len(spectra_dicts))]
    means[target] = np.mean(spectra, axis=0)
    stds[target] = np.std(spectra, axis=0)


# no_smooth = True
no_smooth = False

# Create the plot
plt.figure()
for target, color in targets:
    if no_smooth:
        plt.plot(λ, means[target], label=target, color=color, linewidth=0.25)
        plt.fill_between(λ,
                        means[target] - stds[target],
                        means[target] + stds[target],
                        color=color, alpha=0.2)
    else:
        spectrum_avg = convolve(means[target], kernel, boundary='extend')

        plt.plot(λ, means[target], label=target, color=color, linewidth=0.5, alpha=0.25)
        plt.plot(λ, spectrum_avg, label=target+' (smoothed)', color=color, linewidth=1)
        
plt.xlim(λ.min(), λ.max())
plt.ylim(-7, 12.5)  
plt.legend()
plt.grid(alpha=0.2)

if no_smooth:
    plt.title('Mean spectra with std')
else:
    plt.title('Mean spectra (smoothed)')

plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()

# %%
with open(MUSE_DATA_FOLDER+f'quasars/spectra/wavelengths.pkl', 'wb') as f:
    pickle.dump(λ, f)