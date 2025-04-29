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
from tools.utils import plot_radial_profiles_new, draw_PSF_stack, mask_circle, rad2mas
from tools.config_manager import ConfigManager, MultipleTargetsInOneObservation
from data_processing.normalizers import CreateTransformSequenceFromFile
from tqdm import tqdm
from project_globals import MUSE_DATA_FOLDER, device
from astropy.io import fits
from scipy.ndimage import binary_dilation
from copy import deepcopy

from machine_learning.MUSE_onsky_df import *

predict_Moffat = True
predict_phase_bump = True

# device = torch.device('cpu')

#%%
# Load the FITS file

cube_path = MUSE_DATA_FOLDER + "wide_field/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"

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

reduced_name = 'DATACUBEFINALexpcombine_20200224T050448_7388e773'

with open(MUSE_DATA_FOLDER + f"wide_field/reduced/{reduced_name}.pickle", 'rb') as f:
    data_sample = pickle.load(f)

# Compute wavelength data
λ_min, λ_max, Δλ_full = data_sample['spectral data']['wvl range']
λ_bins = data_sample['spectral data']['wvl bins']
Δλ_binned = np.median(np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])]))

if hasattr(λ_max, 'item'): # To compensate for a small error in the data reduction routine
    λ_max = λ_max.item()

λ = np.linspace(λ_min, λ_max, np.round((λ_max-λ_min)/Δλ_full+1).astype('int'))
assert len(λ) == data_full.shape[0]


#%%
data_onsky, _, _, _ = LoadImages(data_sample, device=device, subtract_background=False, normalize=False, convert_images=True)
data_onsky = data_onsky.squeeze()
data_onsky *= valid_mask

# Correct the flux to match MUSE cube
data_onsky = data_onsky * (data_full.sum(axis=0).max() / data_onsky.sum(axis=0).max())

# Extract config file and update it
config_file, data_onsky = GetConfig(data_sample, data_onsky, device=device)
data_onsky = data_onsky.squeeze()

config_file['NumberSources'] = 1
# The bigger size of initialized PSF is needed to extract the flux loss due to cropping to the box_size later
config_file['sensor_science']['FieldOfView'] = 511
# Select only a subset of predicted wavelengths and modify the config file accordingly
wavelength = config_file['sources_science']['Wavelength'].clone()
# Assumes that we are not in the pupil tracking mode
config_file['telescope']['PupilAngle'] = torch.zeros(1, device=device)

ids_wavelength_selected = np.arange(0, wavelength.shape[-1], 2)
wavelength_selected = wavelength[..., ids_wavelength_selected]
config_file['sources_science']['Wavelength'] = wavelength_selected

config_file['atmosphere']['Cn2Heights'] = torch.tensor([[0., 10000.]], device=device)

N_wvl = len(ids_wavelength_selected)

data_sparse = data_onsky.clone()[ids_wavelength_selected,...]

#%%
from tools.multisrc import detect_sources

data_src = data_onsky.sum(dim=0).cpu().numpy()
# mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)

PSF_size = 111  # Define the size of each ROI (in pixels)
thres = 1200000
# thres = 1000000
# thres = 700000
# thres = 2500000

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
from tools.multisrc import extract_ROIs, add_ROIs, extract_ROIs_from_coords

ROIs, local_coords, global_coords, valid_srcs = extract_ROIs(data_sparse, sources, box_size=PSF_size)
N_src = len(ROIs)
# plot_ROIs_as_grid(ROIs, cols=np.ceil(np.sqrt(len(ROIs))).astype('uint'))  # Adjust the number of columns as needed

sources_valid = sources.iloc[valid_srcs]
# del sources

yy, xx = torch.where(valid_mask.squeeze() > 0) # Compute center of mass for valid mask assuming it's the center of the field

field_center    = np.stack([xx.float().mean().item(), yy.float().mean().item()])[None,...]
sources_coords  = np.stack([sources_valid['x_peak'].values, sources_valid['y_peak'].values], axis=1)
sources_coords -= field_center
sources_coords  = sources_coords*25 / rad2mas  # [pix] -> [rad]

# src_dirs_x = torch.tensor(sources_coords[:,0], device=device).float()
# src_dirs_y = torch.tensor(sources_coords[:,1], device=device).float()

#%%
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
src_spectra_sparse = [GetSpectrum(data_sparse, sources_valid.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_binned = [GetSpectrum(data_onsky,  sources_valid.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_full   = [GetSpectrum(data_full,   sources_valid.iloc[i], radius=flux_core_radius) for i in range(N_src)]

#%%
i_src = 2
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

#%%
from tools.utils import PupilVLT, OptimizableLO
from PSF_models.TipTorch import TipTorch_new

LO_map_size = 31

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          predict_Moffat
}

model = TipTorch_new(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
model.apodizer = model.make_tensor(1.0)

model.to_float()
model.to(device)
model.on_axis = False

# PSF_1 = model()

#%%
from data_processing.normalizers import Uniform, Uniform
from tools.input_manager import InputsManager, InputsManagersUnion

df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')
df_transforms_src_coords  = Uniform(None, -1.8e-5, 1.8e-5)
df_transforms_Cn2_weights = Uniform(None, 0, 1)
df_transforms_GL_frac = Uniform(None, 0, 5)
df_transforms_LO = Uniform(None, -100, 100)

shared_inputs = InputsManager()

shared_inputs.add('r0',    torch.tensor([model.r0[0].item()]), df_transforms_fitted['r0'])
shared_inputs.add('F',     torch.tensor([[1.0,]*N_wvl]), df_transforms_fitted['F'])
shared_inputs.add('bg',    torch.tensor([[0,]*N_wvl]),   df_transforms_fitted['bg'])
shared_inputs.add('dx',    torch.tensor([[0.0,]*N_wvl]), df_transforms_fitted['dx'])
shared_inputs.add('dy',    torch.tensor([[0.0,]*N_wvl]), df_transforms_fitted['dy'])
shared_inputs.add('dn',    torch.tensor([1.5]),          df_transforms_fitted['dn'])
shared_inputs.add('Jx',    torch.tensor([[10,]*N_wvl]),  df_transforms_fitted['Jx'])
shared_inputs.add('Jy',    torch.tensor([[10,]*N_wvl]),  df_transforms_fitted['Jy'])
shared_inputs.add('Jxy',   torch.tensor([[0]]), df_transforms_fitted['Jxy'])
shared_inputs.add('amp',   torch.tensor([0.0]), df_transforms_fitted['amp'])
shared_inputs.add('b',     torch.tensor([0.0]), df_transforms_fitted['b'])
shared_inputs.add('alpha', torch.tensor([4.5]), df_transforms_fitted['alpha'])
# inputs_manager.add('beta',  torch.tensor([2.5]), df_transforms_fitted['beta'])
# inputs_manager.add('ratio', torch.tensor([1.0]), df_transforms_fitted['ratio'])
# inputs_manager.add('theta', torch.tensor([0.0]), df_transforms_fitted['theta'])
shared_inputs.add('s_pow', torch.tensor([0.0]), df_transforms_fitted['s_pow'])

# inputs_manager.add('Cn2_weights', torch.tensor([[0.9, 0.1]]), df_transforms_Cn2_weights)
shared_inputs.add('GL_frac', torch.tensor([np.arctanh(config_file['atmosphere']['Cn2Weights'][0][0].item())]), df_transforms_GL_frac)


if LO_map_size is not None:
    shared_inputs.add('LO_coefs', torch.zeros([1, LO_map_size**2]), df_transforms_LO)
    shared_inputs.set_optimizable('LO_coefs', False)

# inputs_manager.set_optimizable(['ratio', 'theta', 'alpha', 'beta', 'amp', 'b'], predict_Moffat)
shared_inputs.set_optimizable(['alpha', 'amp', 'b'], predict_Moffat)
shared_inputs.set_optimizable(['s_pow'], predict_phase_bump)
shared_inputs.set_optimizable(['Jxy'], False)

shared_inputs.to_float()
shared_inputs.to(device)

# print(shared_inputs)

individual_inputs = InputsManager()

individual_inputs.add('dx', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dx'])
individual_inputs.add('dy', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dy'])
individual_inputs.add('F_norm', torch.tensor([[1.0,]]*N_src), df_transforms_fitted['F'])
individual_inputs.add('src_dirs_x', torch.tensor(sources_coords[:,0]), df_transforms_src_coords)
individual_inputs.add('src_dirs_y', torch.tensor(sources_coords[:,1]), df_transforms_src_coords)

individual_inputs.to_float()
individual_inputs.to(device)

individual_inputs.set_optimizable(['F_norm'], False)
individual_inputs.set_optimizable(['src_dirs_x'], False)
individual_inputs.set_optimizable(['src_dirs_y'], False)

# print(individual_inputs)

all_inputs = InputsManagersUnion([shared_inputs, individual_inputs])

# _ = inputs_managers_union.unstack(inputs_managers_union.stack())
#%%
print(all_inputs)




#%%
from machine_learning.calibrator import Calibrator, Gnosis

calibrator = Calibrator(
    inputs_manager=shared_inputs,
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
# shared_inputs.update(pred_inputs)
all_inputs.update(pred_inputs)

with torch.no_grad():
    config_file['NumberSources'] = 1
    config_file['sensor_science']['FieldOfView'] = 511
    model.Update(config=config_file, init_grids=True, init_pupils=True, init_tomography=True)
    PSF_pred_big = model(pred_inputs) # First initial prediction of the "big" PSF
    
    config_file['sensor_science']['FieldOfView'] = PSF_size
    model.Update(config=config_file, init_grids=True, init_pupils=True, init_tomography=True)
    PSF_pred_small = model(pred_inputs) # Second initial prediction of the "small" PSF
    torch.cuda.empty_cache()

#%%
# How much flux is cropped by assuming the finite size of the PSF box (PSF_predbif is assumed to be quasi-infinite)
crop_ratio = (PSF_pred_big.amax(dim=(-2,-1)) / PSF_pred_small.amax(dim=(-2,-1))).squeeze()

core_mask     = torch.tensor(mask_circle(PSF_size, flux_core_radius+1)[None,None,...]).to(device).float()
core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], flux_core_radius+1)[None,None,...]).to(device).float()

# How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob"
core_flux_ratio = torch.squeeze((PSF_pred_big*core_mask_big).sum(dim=(-2,-1), keepdim=True) / PSF_pred_big.sum(dim=(-2,-1), keepdim=True))
PSF_norm_factor = N_core_pixels / flux_λ_norm / core_flux_ratio / crop_ratio

#%% --------------------------
def func_dxdy(x_, i_src):
    dxdy_inp = individual_inputs.unstack(x_.unsqueeze(0), update=False) # Don't update internal values
    dxdy_inp['dx'] = dxdy_inp['dx'].repeat(1, N_wvl) # Extend to simulated number of wavelength
    dxdy_inp['dy'] = dxdy_inp['dy'].repeat(1, N_wvl) # assuming the same shift for all wavelengths
    dxdy_inp['src_dirs_x'] = dxdy_inp['src_dirs_x'][i_src].unsqueeze(0)
    dxdy_inp['src_dirs_y'] = dxdy_inp['src_dirs_y'][i_src].unsqueeze(0)

    return model(pred_inputs | dxdy_inp)



def fit_dxdy(i_src, verbose=0):
    if verbose > 0:
        print(f'Predicting source {i_src}...')
       
    PSF_0 = torch.nan_to_num(ROIs[i_src].unsqueeze(0)) / src_spectra_sparse[i_src][None,:,None,None]
    F_norm = (PSF_0 * core_mask).sum(dim=(-2,-1), keepdim=True) / core_flux_ratio[None,:,None,None]
    PSF_0 /= F_norm

    dxdy_0 = individual_inputs.stack()[i_src,:]

    _ = func_dxdy(dxdy_0, i_src)
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_0, func_dxdy(dxdy_, i_src), reduction='sum')*1e3
    result = minimize(loss, dxdy_0, max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    dxdy_1 = individual_inputs.unstack(result.x.unsqueeze(0), update=False)
    individual_inputs['dx'][i_src] = dxdy_1['dx'].flatten()
    individual_inputs['dy'][i_src] = dxdy_1['dy'].flatten()

    return PSF_0.clone(), func_dxdy(result.x, i_src).detach().clone(), result.x.clone(), F_norm.clone()


#%%
PSFs_data, PSFs_fitted, fluxes, dxdys = [], [], [], []

for i in tqdm(range(N_src)):
    PSF_0, PSF_1, dxdy, flux = fit_dxdy(i, verbose=0)
    PSFs_data.append(PSF_0)
    PSFs_fitted.append(PSF_1)
    fluxes.append(flux)
    dxdys.append(dxdy)

PSFs_data = torch.vstack(PSFs_data)
PSFs_fitted = torch.vstack(PSFs_fitted)
fluxes = torch.vstack(fluxes)
dxdys  = torch.stack(dxdys)

norm_factors = torch.stack([src_spectra_sparse[i][:,None,None] * fluxes[i, ...] for i in range(N_src)])

#%%
display_mask = mask_circle(PSF_size, 18)[None,...]

PSFs_0_white = np.mean(PSFs_data.cpu().cpu().numpy(), axis=1) * display_mask
PSFs_1_white = np.mean(PSFs_fitted.cpu().cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)

#%% ---------------------------
model_sparse = add_ROIs(
    torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device),
    [PSFs_fitted[i,...]*norm_factors[i] for i in range(N_src)],
    local_coords,
    global_coords
)

# It tells the average flux in the PSF core for each source
# src_spectra_sparse = [GetSpectrum(data_sparse,  sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
# src_spectra_fitted = [GetSpectrum(model_sparse, sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]

# STRANGE LINE IN THE FIRST SPECTRUM!!!!!
#%
# i_src = 1
# plt.plot(λ, src_spectra_full[i_src], linewidth=0.25, alpha=0.3)
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_fitted[i_src].cpu().numpy())
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_sparse[i_src].cpu().numpy())

#%
from tools.multisrc import VisualizeSources, PlotSourcesProfiles

# ROI_plot = np.s_[..., 125:225, 125:225]

VisualizeSources(data_sparse, model_sparse, norm=norm_field, mask=valid_mask)
PlotSourcesProfiles(data_sparse, model_sparse, sources_valid, radius=16, title='Predicted PSFs')

#%%
# merged_config = MultipleTargetsInOneObservation(config_file, N_batch := 16)
merged_config = MultipleTargetsInOneObservation(config_file, N_src)

# for key, value in sources_inputs.items():
#     print(f'{key}: {value.shape}')

def select_sources(src_dict: dict, selected_ids: list) -> dict:
    result_dict = {}
    for key, tensor in src_dict.items():
        if hasattr(tensor, 'shape') and tensor.shape[0] == N_src:
            result_dict[key] = tensor[selected_ids]
        else:
            result_dict[key] = tensor
    return result_dict


#%%
'''
shared_inputs.set_optimizable(['dx'], False)
shared_inputs.set_optimizable(['dy'], False)
shared_inputs.set_optimizable(['bg'], False)
shared_inputs.set_optimizable(['s_pow'], False)
# inputs_manager.set_optimizable(['theta'], False)
# inputs_manager.set_optimizable(['ratio'], False)
shared_inputs.set_optimizable(['amp'], False)
shared_inputs.set_optimizable(['alpha'], False)
shared_inputs.set_optimizable(['b'], False)
shared_inputs.set_optimizable(['Jx'], False)
shared_inputs.set_optimizable(['Jy'], False)
shared_inputs.set_optimizable(['Jxy'], False)
shared_inputs.set_optimizable(['GL_frac'], True)
shared_inputs.set_optimizable(['F'], False)

individual_inputs.set_optimizable(['F_norm'], True)
individual_inputs.set_optimizable(['dx'], False)
individual_inputs.set_optimizable(['dy'], False)
'''

all_inputs.set_optimizable(['dx', 'dy', 'bg', 's_pow', 'amp', 'alpha', 'b', 'Jx', 'Jy', 'Jxy', 'F'], False)
all_inputs.set_optimizable(['GL_frac', 'F_norm'], True)

#%%
x0 = all_inputs.stack()
empty_img   = torch.zeros([N_wvl, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 0 + 1

torch.cuda.empty_cache()
model.Update(config=merged_config, init_grids=True, init_pupils=True, init_tomography=True)

#%%
def func_fit(x_):
    sources_inputs = all_inputs.unstack(x_, update=False)
    sources_inputs['dx'] = sources_inputs['dx'].unsqueeze(-1).repeat(1,N_wvl)
    sources_inputs['dy'] = sources_inputs['dy'].unsqueeze(-1).repeat(1,N_wvl)
    
    GL_frac = nn.functional.tanh(sources_inputs['GL_frac'].abs())
    sources_inputs['Cn2_weights'] = torch.stack([GL_frac, 1-GL_frac]).T

    PSF_ = model(sources_inputs) * norm_factors * sources_inputs['F_norm'].view(-1,1,1,1)
      
    return add_ROIs( empty_img*0.0, PSF_, local_coords, global_coords )


def loss_fit(x_):
    simulated_field = func_fit(x_)
    
    l1 = F.smooth_l1_loss(data_sparse*wvl_weights, simulated_field*wvl_weights, reduction='mean')
    # l2 = F.mse_loss(data_sparse*wvl_weights, simulated_field*wvl_weights, reduction='mean')
    return l1 * 1e-3 #+ l2 * 5e-6


#%%
torch.cuda.empty_cache()
result_global = minimize(loss_fit, x0, max_iter=300, tol=1e-3, method='l-bfgs', disp=2)

#%%
with torch.no_grad():
    x1 = result_global.x.clone()
    # x1 = x0
    sources_inputs_fitted = all_inputs.unstack(x1, update=True)
    field_fitted = func_fit(x1)


#%%
# VisualizeSources(data_sparse, model_sparse, norm=norm_field, mask=valid_mask)
# PlotSourcesProfiles(data_sparse, model_sparse, sources_valid, radius=16, title='Predicted PSFs')

VisualizeSources(data_sparse, field_fitted, norm=norm_field, mask=valid_mask)
PlotSourcesProfiles(data_sparse, field_fitted, sources_valid, radius=16, title='Fitted PSFs')


#%% ------ Adam implementation

# Convert to PyTorch parameters for optimization
x0_param = nn.Parameter(all_inputs.stack().clone(), requires_grad=True)

# Create Adam optimizer
optimizer = optim.Adam([x0_param], lr=1e-3)

# Hyperparameters for optimization
num_epochs = 30

# Progress tracking
losses = []

# Main optimization loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()  # Zero gradients at the beginning of each epoch




#%% --------------- Tiled optimization ------------------------------
def find_closest_sources(i_src: int, N: int) -> np.ndarray:
    """
    Find N closest sources to the source with index i_src.

    Args:
        i_src: Index of the reference source
        N: Number of closest sources to return

    Returns:
        numpy array of indices of N closest sources, sorted by distance
    """
    sources_ = sources_valid.to_numpy()[:, :-1]  # Exclude the last column
    deltas = sources_ - sources_[i_src]
    dist_sq = np.einsum('ij,ij->i', deltas, deltas)  # Efficient way to compute squared distances
    indices = np.argpartition(dist_sq, N)[:N]  # Get indices of N smallest distances
    return indices[np.argsort(dist_sq[indices])]  # Sort indices by distance


def create_proximity_table(sources_data: pd.DataFrame) -> np.ndarray:
    """
    Create a proximity table for all sources.

    Args:
        sources_data: DataFrame containing source positions

    Returns:
        N_src x N_src numpy array where element [i,j] is the squared distance
        between source i and source j
    """
    sources_ = sources_data.to_numpy()[:, :-1]  # Exclude the last column
    N = len(sources_)
    proximity_table = np.zeros((N, N))

    for i in range(N):
        deltas = sources_ - sources_[i]
        proximity_table[i, :] = np.einsum('ij,ij->i', deltas, deltas)

    return proximity_table # [pix^2]


def get_sources_in_ROI(i_src: int, roi_radius: float, proximity_table: np.ndarray) -> list:
    """
    Find all sources within a specified ROI around the selected source.

    Args:
        i_src: Index of the center source
        roi_radius: Radius of ROI in pixels
        proximity_table: Pre-computed proximity table with squared distances

    Returns:
        List of indices of sources within the ROI, sorted by distance
    """
    # Get squared distances from source i_src to all other sources
    distances = proximity_table[i_src, :]

    # Find indices of sources within the ROI radius (using squared distance)
    indices = np.where(distances <= roi_radius**2)[0]

    # Sort indices by distance
    return indices[np.argsort(distances[indices])].tolist()


def get_N_closest_sources(i_src: int, n: int, proximity_table: np.ndarray) -> list:
    """
    Find the N closest sources to the selected source.

    Args:
        i_src: Index of the center source
        n: Number of closest sources to return (including the center source)
        proximity_table: Pre-computed proximity table with squared distances

    Returns:
        List of indices of N closest sources, sorted by distance
    """
    # Get squared distances from source i_src to all other sources
    distances = proximity_table[i_src, :]

    # Get indices of n smallest distances (including the center source)
    indices = np.argpartition(distances, min(n, len(distances)-1))[:n]
    # Sort indices by distance
    return indices[np.argsort(distances[indices])].tolist()


def select_sources_in_tile(
    sources_data: pd.DataFrame,
    proximity_table: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    d_offset: float,
    N: int
) -> list:
    """
    Select all sources within a specified tile plus some sources outside the tile
    but within d_offset distance, ensuring that exactly N sources are returned.

    Args:
        sources_data: DataFrame containing source positions and other info
        proximity_table: Pre-computed proximity table with squared distances
        x_range: (x_min, x_max) defining the tile's x boundaries
        y_range: (y_min, y_max) defining the tile's y boundaries
        d_offset: Maximum distance outside the tile to consider additional sources
        N: Exact number of sources to return

    Returns:
        List of indices of selected sources
    """
    sources_pos = sources_data[['x_peak', 'y_peak']].to_numpy()

    # Extract x and y range values
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Find sources within the tile
    in_tile_mask = ((sources_pos[:, 0] >= x_min) &
                   (sources_pos[:, 0] <= x_max) &
                   (sources_pos[:, 1] >= y_min) &
                   (sources_pos[:, 1] <= y_max))

    in_tile_indices = np.where(in_tile_mask)[0].tolist()

    # If we have exactly N sources, return them
    if len(in_tile_indices) == N:
        return in_tile_indices

    # If we have fewer than N sources within the tile, add nearby sources
    if len(in_tile_indices) < N:
        # Find sources outside the tile but within d_offset
        outside_tile_mask = ~in_tile_mask
        outside_sources = sources_pos[outside_tile_mask]
        outside_indices = np.where(outside_tile_mask)[0]

        # Calculate distances to the nearest tile boundary for each outside source
        distances = np.zeros(len(outside_sources))
        for i, (x, y) in enumerate(outside_sources):
            # Calculate distance to nearest x and y boundaries
            dx = max(0, x_min - x, x - x_max)
            dy = max(0, y_min - y, y - y_max)
            # Euclidean distance to nearest boundary
            distances[i] = np.sqrt(dx**2 + dy**2)
        # Find indices within d_offset of the tile boundary
        near_tile_mask = distances <= d_offset
        near_tile_indices = outside_indices[near_tile_mask].tolist()

        # Sort near tile indices by peak value (brightness)
        if len(near_tile_indices) > 0:
            peak_values = sources_data.iloc[near_tile_indices]['peak_value'].to_numpy()
            sorted_indices = np.argsort(peak_values)[::-1]  # Sort descending
            near_tile_indices = [near_tile_indices[i] for i in sorted_indices]

        # Add more sources from proximity if needed
        if len(in_tile_indices) + len(near_tile_indices) < N:
            # Find remaining sources
            remaining_indices = np.setdiff1d(np.arange(len(sources_data)),
                                            np.concatenate([in_tile_indices, near_tile_indices]))

            # If we have tile sources, find closest to those
            if len(in_tile_indices) > 0:
                # Use the closest source from the tile as reference
                ref_source = in_tile_indices[0]
                distances = proximity_table[ref_source, remaining_indices]
                sorted_indices = np.argsort(distances)
                additional_indices = remaining_indices[sorted_indices][:N - len(in_tile_indices) - len(near_tile_indices)]
            else:
                # Use the brightest source as reference
                peak_values = sources_data.iloc[remaining_indices]['peak_value'].to_numpy()
                sorted_indices = np.argsort(peak_values)[::-1]  # Sort descending
                additional_indices = remaining_indices[sorted_indices][:N - len(in_tile_indices) - len(near_tile_indices)]

            all_indices = in_tile_indices + near_tile_indices + additional_indices.tolist()
        else:
            # Just add near tile indices until we have N
            all_indices = in_tile_indices + near_tile_indices[:N - len(in_tile_indices)]

    # If we have more than N sources within the tile, keep the brightest ones
    else:
        peak_values = sources_data.iloc[in_tile_indices]['peak_value'].to_numpy()
        sorted_indices = np.argsort(peak_values)[::-1]  # Sort descending
        all_indices = [in_tile_indices[i] for i in sorted_indices[:N]]

    return all_indices


proximity_table = create_proximity_table(sources_valid)

# brightest_id = sources_valid['peak_value'].argmax()
# brightest_pos = sources_valid.iloc[brightest_id][['x_peak', 'y_peak']].to_numpy()
# x_range = (brightest_pos[0] - 50, brightest_pos[0] + 10)
# y_range = (brightest_pos[1] - 50, brightest_pos[1] + 10)

# # testo = get_n_closest_sources(brightest_id, N_batch, proximity_table)
# testo = select_sources_in_tile(sources_valid, proximity_table, x_range, y_range, 50, N_batch)

# model_sparse = add_ROIs(
#     torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device),
#     [PSFs_fitted[i,...]*norm_factors[i] for i in testo],
#     [local_coords[i]  for i in testo],
#     [global_coords[i] for i in testo]
# )

# VisualizeSources(data_sparse, model_sparse, norm=norm_field, mask=valid_mask)


#%%
import matplotlib.patches as patches

def split_image_into_tiles(
    image_shape: tuple,
    n_tiles_x: int,
    n_tiles_y: int,
    border_offset: int = 0
) -> list:
    """
    Split an image into NxM tiles with an optional border offset.

    Args:
        image_shape: Tuple of (height, width) for the image
        n_tiles_x: Number of tiles along x-axis
        n_tiles_y: Number of tiles along y-axis
        border_offset: Pixels to exclude from borders

    Returns:
        List of dictionaries containing tile information with x_range and y_range
    """
    height, width = image_shape[-2:]

    # Calculate effective dimensions after applying border offset
    eff_height = height - 2 * border_offset
    eff_width  = width  - 2 * border_offset

    # Calculate tile sizes
    tile_height = eff_height // n_tiles_y
    tile_width = eff_width // n_tiles_x

    tiles = []

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            # Calculate tile boundaries with offset
            y_min = border_offset + i * tile_height
            y_max = border_offset + (i + 1) * tile_height
            x_min = border_offset + j * tile_width
            x_max = border_offset + (j + 1) * tile_width

            # Ensure the last tiles include any remaining pixels
            if i == n_tiles_y - 1:
                y_max = height - border_offset
            if j == n_tiles_x - 1:
                x_max = width - border_offset

            tile_info = {
                'x_range': (x_min, x_max),
                'y_range': (y_min, y_max),
                'id': (i, j)
            }

            tiles.append(tile_info)

    return tiles


def visualize_tiles(
    image,
    tiles: list,
    title: str = 'Image Tiles',
    cmap: str = 'gray',
    norm = None,
    alpha: float = 0.7
) -> None:
    """
    Visualize the tiling of an image.

    Args:
        image: The image data to display
        tiles: List of tile dictionaries as returned by split_image_into_tiles
        title: Plot title
        cmap: Colormap for the image display
        norm: Normalization for the image display
        alpha: Alpha value for the rectangle overlay
    """
    if torch.is_tensor(image):
        if image.dim() > 2:
            # If multi-channel/wavelength, take mean or sum
            display_img = image.mean(dim=0).cpu().numpy()
        else:
            display_img = image.cpu().numpy()
    else:
        if image.ndim > 2:
            # If multi-channel/wavelength, take mean or sum
            display_img = image.mean(axis=0)
        else:
            display_img = image

    plt.figure(figsize=(10, 8))
    plt.imshow(display_img, origin='lower', cmap=cmap, norm=norm)

    colors = plt.cm.tab10.colors

    for i, tile in enumerate(tiles):
        x_min, x_max = tile['x_range']
        y_min, y_max = tile['y_range']
        width = x_max - x_min
        height = y_max - y_min

        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=alpha
        )
        plt.gca().add_patch(rect)

        # Add tile ID label
        plt.text(
            x_min + width/2, y_min + height/2,
            f"Tile {tile['id']}", color=color,
            ha='center', va='center', fontweight='bold'
        )

    plt.title(title)
    plt.tight_layout()
    plt.show()


tiles = split_image_into_tiles(data_full.shape, n_tiles_x=4, n_tiles_y=4, border_offset=10)
visualize_tiles(data_full.sum(axis=0), tiles, title='Image Tiles', norm=norm_field)

#%%
sources_inputs = all_inputs.unstack(all_inputs.stack())
sources_inputs['dx'] = sources_inputs['dx'].unsqueeze(-1).repeat(1,N_wvl)
sources_inputs['dy'] = sources_inputs['dy'].unsqueeze(-1).repeat(1,N_wvl)


def simulate_tile(model_inputs, tile, proximity_table, sources, max_sources):
    
    x_range, y_range = tile['x_range'], tile['y_range']

    source_indices = select_sources_in_tile(
        sources,
        proximity_table,
        tile['x_range'],
        tile['y_range'],
        d_offset = 30,
        N = max_sources
    )

    if len(source_indices) == 0:
        return torch.zeros([N_wvl, tile['y_range']-tile['y_range'], tile['x_range']-tile['x_range']], device=device)

    # Create model for this tile using the selected sources
    # model_tile = add_ROIs(
    #     torch.zeros([N_wvl, y_range[1]-y_range[0], x_range[1]-x_range[0]], device=device),
    #     [PSFs_fitted[i, ...] * norm_factors[i] for i in source_indices],
    #     [local_coords[i] - torch.tensor([x_range[0], y_range[0]], device=device) for i in source_indices],
    #     [global_coords[i] for i in source_indices]
    # )
    
    batch_inputs = select_sources(model_inputs, source_indices)
    PSF_1 = model(batch_inputs)

    model_tile = add_ROIs(
        torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device),
        [PSF_1[i,...]*norm_factors[src_id] for i, src_id in enumerate(source_indices)],
        [local_coords [src_id] for src_id in source_indices],
        [global_coords[src_id] for src_id in source_indices]
    )
    
    return model_tile[..., y_range[0]:y_range[1], x_range[0]:x_range[1]]


torch.cuda.empty_cache()
with torch.no_grad():
    tiles_model = [simulate_tile(sources_inputs, tile, proximity_table, sources_valid, max_sources=N_batch) for tile in tqdm(tiles)]

# Concatenate tiles back into image based on their positions
model_sparse_tiled = torch.zeros_like(data_sparse, device=device)
for tile, model_tile in zip(tiles, tiles_model):
    x_min, x_max = tile['x_range']
    y_min, y_max = tile['y_range']
    model_sparse_tiled[:, y_min:y_max, x_min:x_max] = model_tile[...]

    
#%%
VisualizeSources(data_sparse, model_sparse_tiled.detach(), norm=norm_field, mask=valid_mask)

#%%
# Create a parameter initialization for inputs_manager_objs
x0 = individual_inputs.stack()

# Convert to PyTorch parameters for optimization
x0_param = nn.Parameter(x0.clone(), requires_grad=True)

# Create Adam optimizer
optimizer = optim.Adam([x0_param], lr=1e-3)

# Hyperparameters for optimization
num_epochs = 30
accumulation_steps = len(tiles)  # Accumulate gradients over all tiles
log_interval = 5

# Progress tracking
losses = []

# Main optimization loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()  # Zero gradients at the beginning of each epoch

    # Process each tile with gradient accumulation
    for tile_idx, tile in enumerate(tqdm(tiles, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # Extract tile boundaries
        x_range, y_range = tile['x_range'], tile['y_range']

        # Select sources relevant to this tile
        source_indices = select_sources_in_tile(
            sources_valid,
            proximity_table,
            x_range, y_range,
            d_offset=30,
            N=min(N_batch, N_src)
        )

        if len(source_indices) == 0:
            continue  # Skip empty tiles

        # Get tile data
        tile_data = data_sparse[:, y_range[0]:y_range[1], x_range[0]:x_range[1]]

        # Prepare empty image for this tile
        tile_empty = torch.zeros([N_wvl, y_range[1]-y_range[0], x_range[1]-x_range[0]], device=device)

        # Forward pass - create model for this batch of sources
        PSFs_fit = []
        for i, src_idx in enumerate(source_indices):
            # Get parameters for this source
            src_params = x0_param[src_idx].unsqueeze(0)
            dxdy_inp = individual_inputs.unstack(src_params, update=False)

            # Extend dx and dy for all wavelengths
            dxdy_inp['dx'] = dxdy_inp['dx'].repeat(1, N_wvl)
            dxdy_inp['dy'] = dxdy_inp['dy'].repeat(1, N_wvl)

            # Set source directions
            dxdy_inp['src_dirs_x'] = dxdy_inp['src_dirs_x'][0].unsqueeze(0)
            dxdy_inp['src_dirs_y'] = dxdy_inp['src_dirs_y'][0].unsqueeze(0)

            # Generate PSF
            psf = model(pred_inputs | dxdy_inp)

            # Apply normalization
            flux_norm = norm_factors[src_idx]
            PSFs_fit.append(psf.squeeze() * flux_norm)

        # Add all PSFs to the tile
        local_tile_coords = [local_coords[idx] - torch.tensor([x_range[0], y_range[0]], device=device) for idx in source_indices]
        model_tile = add_ROIs(tile_empty, PSFs_fit, local_tile_coords, [global_coords[idx] for idx in source_indices])

        # Calculate loss for this tile
        tile_loss = F.smooth_l1_loss(tile_data, model_tile, reduction='sum')

        # Normalize loss by tile size and accumulate
        normalized_loss = tile_loss / (tile_data.shape[0] * tile_data.shape[1] * tile_data.shape[2])
        normalized_loss = normalized_loss / accumulation_steps  # Scale by accumulation steps
        normalized_loss.backward()

        # Track loss
        epoch_loss += normalized_loss.item()

        # Clear cache to save memory
        torch.cuda.empty_cache()

    # Update weights
    optimizer.step()

    # Record average loss
    avg_loss = epoch_loss / len(tiles)
    losses.append(avg_loss)

    # Print progress
    if (epoch + 1) % log_interval == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Adaptive learning rate - reduce if loss plateaus
    if epoch > 10 and losses[-1] > 0.98 * losses[-2]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
            print(f"Reducing learning rate to {param_group['lr']:.6f}")

# Update the inputs_manager_objs with optimized parameters
with torch.no_grad():
    for i in range(N_src):
        src_params = individual_inputs.unstack(x0_param[i].unsqueeze(0), update=False)
        individual_inputs['dx'][i] = src_params['dx'].flatten()
        individual_inputs['dy'][i] = src_params['dy'].flatten()
        individual_inputs['F_norm'][i] = src_params['F_norm'].flatten()

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Optimization Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Generate final model with optimized parameters
with torch.no_grad():
    # Use the tiled approach for the final model to save memory
    model_sparse_optimized = torch.zeros_like(data_sparse, device=device)

    for tile in tqdm(tiles, desc="Generating final model"):
        x_range, y_range = tile['x_range'], tile['y_range']
        source_indices = select_sources_in_tile(
            sources_valid, proximity_table, x_range, y_range, d_offset=30, N=min(N_batch, N_src)
        )

        if len(source_indices) == 0:
            continue

        tile_empty = torch.zeros([N_wvl, y_range[1]-y_range[0], x_range[1]-x_range[0]], device=device)
        PSFs_fit = []

        for src_idx in source_indices:
            # Get optimized parameters
            dxdy_inp = individual_inputs.unstack(x0_param[src_idx].unsqueeze(0), update=False)
            dxdy_inp['dx'] = dxdy_inp['dx'].repeat(1, N_wvl)
            dxdy_inp['dy'] = dxdy_inp['dy'].repeat(1, N_wvl)
            dxdy_inp['src_dirs_x'] = dxdy_inp['src_dirs_x'][0].unsqueeze(0)
            dxdy_inp['src_dirs_y'] = dxdy_inp['src_dirs_y'][0].unsqueeze(0)

            # Generate PSF with optimized parameters
            psf = model(pred_inputs | dxdy_inp)
            flux_norm = norm_factors[src_idx] * dxdy_inp['F_norm'][0]
            PSFs_fit.append(psf.squeeze() * flux_norm)

        # Add all PSFs to the tile
        local_tile_coords = [local_coords[idx] - torch.tensor([x_range[0], y_range[0]], device=device) for idx in source_indices]
        model_tile = add_ROIs(tile_empty, PSFs_fit, local_tile_coords, [global_coords[idx] for idx in source_indices])

        # Add to final model
        model_sparse_optimized[:, y_range[0]:y_range[1], x_range[0]:x_range[1]] = model_tile

        # Clear cache to save memory
        torch.cuda.empty_cache()

# Visualize the optimized model
VisualizeSources(data_sparse, model_sparse_optimized, norm=norm_field, mask=valid_mask)
PlotSourcesProfiles(data_sparse, model_sparse_optimized, sources_valid, radius=16, title='Optimized PSFs')


#%% ====================================================== Fitting =======================================================
from tools.utils import OptimizableLO

LO_basis = OptimizableLO(model, ignore_pupil=False)

shared_inputs.set_optimizable('LO_coefs', False)
shared_inputs.set_optimizable('Jxy', False)

shared_inputs.delete('amp')
shared_inputs.delete('beta')
shared_inputs.delete('alpha')
shared_inputs.delete('b')
shared_inputs.delete('ratio')
shared_inputs.delete('theta')

shared_inputs.delete('dx')
shared_inputs.delete('dy')
shared_inputs.delete('s_pow')

print(shared_inputs)

individual_inputs.set_optimizable('F_norm', True)

#%%
x0 = torch.cat([
    shared_inputs.stack().flatten(),
    individual_inputs.stack().flatten(),
])
x_size = shared_inputs.get_stacked_size()

empty_img   = torch.zeros([N_wvl, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 0 + 1

#%%
torch.cuda.empty_cache()

def func_fit(x): # TODO: relative weights for different brigtness
    # global selected_ids
    PSFs_fit = []
    for i in range(N_src):
    # for i in selected_ids:
        params_dict = shared_inputs.unstack(x[:x_size].unsqueeze(0))
        F_dxdy_dict = individual_inputs.unstack(x[x_size:].view(N_src, -1))

        # phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))

        F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | F_dxdy_dict
        flux_norm = norm_factors[i] * F_dxdy_dict['F_norm'][i]

        # PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
        PSFs_fit.append( model(inputs).squeeze() * flux_norm )

    # np.random.shuffle(all_ids)
    # selected_ids = all_ids[:12].tolist()
    # print(selected_ids)
        
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
x_fit_dict = shared_inputs.unstack(x0.unsqueeze(0), include_all=False)

#%% 
# with torch.no_grad():
# model_fit = func_fit(result_global.x).detach()
with torch.no_grad():
    model_fit = func_fit(x0).detach()

VisualizeSources(data_sparse, model_fit, norm=norm_field, mask=valid_mask)
PlotSourcesProfiles(data_sparse, model_fit, sources_valid, radius=16, title='Fitted PSFs')


#%% 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
from tools.misc import QuadraticModel


λ_sparse = wavelength_selected.flatten() * 1e9 # [nm]

F_model    = QuadraticModel(λ_sparse)
Jx_model   = QuadraticModel(λ_sparse)
Jy_model   = QuadraticModel(λ_sparse)
norm_model = QuadraticModel(λ_sparse)

params_Jx   = Jx_model.fit(shared_inputs['Jx'].flatten())
params_Jy   = Jy_model.fit(shared_inputs['Jy'].flatten())
params_F    = F_model.fit(shared_inputs['F'].flatten())
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

plt.plot(λ_sparse.cpu(), shared_inputs['Jx'].flatten().cpu(), label='Data', color='tab:blue')
plt.plot(λ_sparse.cpu(), Fx_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:blue')
plt.scatter(λ_sparse.cpu(), Jx_new.cpu(), label='New Quadratic Curve', color='tab:blue', marker='x')

plt.plot(λ_sparse.cpu(), shared_inputs['Jy'].flatten().cpu(), label='Data', color='tab:orange')
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

plt.plot(λ_sparse.cpu(), shared_inputs['F'].flatten().cpu(), label='Data', color='tab:green')
plt.plot(λ_sparse.cpu(), F_curve_fit.cpu(), label='Fitted Quadratic Curve', linestyle='--', color='tab:green')
plt.scatter(λ_sparse.cpu(), F_new.cpu(), label='New Quadratic Curve', color='tab:green', marker='x')
plt.xlabel('Wavelength [nm]')
plt.legend()
plt.grid(True)
plt.show()

#%%
shared_inputs.set_optimizable(['F', 'Jx', 'Jy'],  False)
individual_inputs.set_optimizable(['dx', 'dy'], False)

print(shared_inputs)

#%%
x_size_model = shared_inputs.get_stacked_size()
x_size_curve = curve_inputs.get_stacked_size()
x_size_total = x_size_curve + x_size_model

x2 = torch.cat([
    shared_inputs.stack().flatten(),
    curve_inputs.stack().flatten(),
    individual_inputs.stack().flatten(),
])

empty_img = torch.zeros([N_wvl, data_sparse.shape[-2], data_sparse.shape[-1]], device=device)

#%%
def func_fit_curve(x):
    PSFs_fit = []
    for i in range(N_src):
        params_dict = shared_inputs.unstack(x[:x_size_model].unsqueeze(0))
        curve_p_    = curve_inputs.unstack(x[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
        F_dxdy_dict = individual_inputs.unstack(x[x_size_curve+x_size_model:].view(N_src, -1))
        
        curve_dict = {p: curve_sample(λ_sparse, curve_p_, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
        
        phase_func = lambda: LO_basis(shared_inputs["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        # F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        # F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        F_dxdy_dict['dx'] = individual_inputs['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = individual_inputs['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

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

x_fit_dict = shared_inputs.unstack(x2[:x_size_model].unsqueeze(0))
x_curve_fit_dict = curve_inputs.unstack(x2[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
flux_corrections = individual_inputs['F_norm']

#%% 
model_fit_curves = func_fit_curve(result_global.x).detach()

VisualizeSources(data_sparse, model_fit_curves, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(data_sparse, model_fit_curves, sources_valid, radius=16, title='Fitted PSFs')

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
            'dx': individual_inputs['dx'][i].unsqueeze(-1).repeat(batch_size).unsqueeze(0), # Extend to simulated number of wavelength
            'dy': individual_inputs['dy'][i].unsqueeze(-1).repeat(batch_size).unsqueeze(0) # assuming the same shift for all wavelengths
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
PlotSourcesProfiles(data_full, model_full, sources_valid, radius=16, title='Fitted PSFs')

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

# Create the plot
plt.figure()
for target, color in targets:
    plt.plot(λ, means[target], label=target, color=color, linewidth=0.25)
    plt.fill_between(λ,
                     means[target] - stds[target],
                     means[target] + stds[target],
                     color=color, alpha=0.2)
plt.legend()
plt.grid(alpha=0.2)
plt.title('Mean spectra with std')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()

# %%
with open(MUSE_DATA_FOLDER+f'quasars/spectra/wavelengths.pkl', 'wb') as f:
    pickle.dump(λ, f)