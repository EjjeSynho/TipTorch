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
from astropy.visualization import simple_norm
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, RectangularAperture
from sklearn.cluster import DBSCAN
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages
from tools.utils import plot_radial_profiles_new, draw_PSF_stack, mask_circle
from tools.config_manager import ConfigManager
from data_processing.normalizers import CreateTransformSequenceFromFile, InputsTransformer
from tqdm import tqdm
from project_globals import MUSE_DATA_FOLDER, device
from astropy.io import fits
from scipy.ndimage import binary_dilation

from machine_learning.MUSE_onsky_df import *


#%%
# Load the FITS file

# reduced_name = 'J0259_2024-12-05T03_04_07.007'
# reduced_name = 'J0259_2024-12-05T03_49_24.768'
reduced_name = 'J0259_2024-12-05T03_15_37.598'

cube_path = MUSE_DATA_FOLDER + f"quasars/J0259_cubes/J0259-0901_DATACUBE_FINAL_{reduced_name[6:]}.fits"
test_fits = fits.open(cube_path)

data_onsky_full = np.nan_to_num(test_fits[1].data, nan=0.0)

# Compute the mask of valid pixels
nan_mask = np.abs(np.nansum(test_fits[1].data, axis=0)) < 1e-12
nan_mask = binary_dilation(nan_mask, iterations=2, )
valid_mask = ~nan_mask

test_fits.close()

data_onsky_full = data_onsky_full * valid_mask[np.newaxis, :, :]

# Load processed data file
with open(MUSE_DATA_FOLDER + f"quasars/J0259_reduced/{reduced_name}.pickle", 'rb') as f:
    data = pickle.load(f)

# Compute wavelength data
λ_min, λ_max, Δλ_full = data['spectral data']['wvl range']
λ_bins = data['spectral data']['wvl bins']
Δλ_binned = np.median(np.concatenate([np.diff(λ_bins[λ_bins < 589]), np.diff(λ_bins[λ_bins > 589])]))

if hasattr(λ_max, 'item'): # To compensate for a small error in the data reduction routine
    λ_max = λ_max.item()

λ = np.linspace(λ_min, λ_max, np.round((λ_max-λ_min)/Δλ_full+1).astype('int'))
assert len(λ) == data_onsky_full.shape[0]

#%%
data_onsky, _, _, _ = LoadImages(data, device=device, subtract_background=False, normalize=False, convert_images=True)
data_onsky = data_onsky.squeeze()
data_onsky *= torch.tensor(valid_mask, device=device).float().unsqueeze(0)

# Correct the flux to match MUSE cube
data_onsky = data_onsky * (data_onsky_full.sum(axis=0).max() /  data_onsky.sum(axis=0).max())

# Extract config file and update it
config_file, data_onsky = GetConfig(data, data_onsky)
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

data_onsky_sparse = data_onsky.clone()[ids_wavelength_selected,...]

#%
# test_A = data_onsky.sum(dim=0).cpu().numpy()
# test_B = np.nansum(data_onsky_full, axis=0)
# test_C = np.abs(test_A - test_B)

# plt.imshow(test_A, norm=LogNorm(vmin=1e-0, vmax=500000), origin='lower', cmap='gray')
# plt.show()
# plt.imshow(test_B, norm=LogNorm(vmin=1e-0, vmax=500000), origin='lower', cmap='gray')
# plt.show()
# plt.imshow(test_C, norm=LogNorm(vmin=1e-0, vmax=500000), origin='lower', cmap='gray')
# plt.show()

#%%
data_src = data_onsky.sum(dim=0).cpu().numpy()
mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)

box_size = 111  # Define the size of each ROI (in pixels)
thres = 50000

sources = find_peaks(data_src, threshold=thres, box_size=11)
print(f"Detected {len(sources)} sources")

def merge_sources(sources, eps=2):
    ''' Helps in the case if a single source was detected as multiple peaks '''
    positions = np.transpose((sources['x_peak'], sources['y_peak']))
    
    db = DBSCAN(eps=eps, min_samples=1).fit(positions)

    unique_labels = set(db.labels_)
    merged_positions = np.array([ positions[db.labels_ == label].mean(axis=0) for label in unique_labels ])
    merged_fluxes    = np.array([ data_src[int(pos[1]), int(pos[0])] for pos in merged_positions ])

    merged_sources = pd.DataFrame(merged_positions, columns=['x_peak', 'y_peak'])
    merged_sources['peak_value'] = merged_fluxes
    return merged_sources

sources   = merge_sources(sources, eps=2)
srcs_pos  = np.transpose((sources['x_peak'], sources['y_peak']))
srcs_flux = sources['peak_value'].to_numpy()

print(f"Merged to {len(sources)} sources")

# Draw the detected sources
apertures = CircularAperture(srcs_pos, r=5)
apertures_box = RectangularAperture(srcs_pos, box_size//2, box_size//2)

norm_field = LogNorm(vmin=10, vmax=thres*10)

plt.imshow(np.abs(data_src), norm=norm_field, origin='lower', cmap='gray')
apertures_box.plot(color='gold', lw=2, alpha=0.45)
plt.show()

#%%
def extract_ROIs(image, sources, box_size=20, max_nan_fraction=0.3):
    torch_flag=False
    if isinstance(image, np.ndarray):
        xp = np
    # elif isinstance(image, cp.array):
    #     xp = cp
    elif isinstance(image, torch.Tensor):
        xp = torch
        torch_flag=True
    else:
        raise TypeError("Unexpected image data type")
        
    ROIs = []
    roi_local_coords  = []  # To store the local image indexes inside NaN-padded ROI
    roi_global_coords = []  # To store the coordinates relative to the original image
    positions = np.transpose((sources['x_peak'], sources['y_peak']))

    D = image.shape[0]  # Depth dimension

    half_box = box_size // 2
    extra_pixel = box_size % 2  # 1 if box_size is odd, 0 if even

    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        
        # Calculate the boundaries, ensuring they don't exceed the image size
        x_min = x - half_box
        x_max = x + half_box + extra_pixel
        y_min = y - half_box
        y_max = y + half_box + extra_pixel
        
        # Extract the ROI with NaN-padding if the ROI goes outside the image bounds
        if torch_flag:
            roi = torch.full((D, box_size, box_size), float('nan'), device=image.device)  # Create a blank 3D box filled with NaNs
        else:
            roi = xp.full((D, box_size, box_size), float('nan'))  # Create a blank 3D box filled with NaNs
        
        # Calculate the actual overlapping region between image and ROI
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, image.shape[-1])
        y_max_img = min(y_max, image.shape[-2])

        # Determine where the image will go in the NaN-padded ROI
        x_min_roi = max(0, -x_min)
        y_min_roi = max(0, -y_min)
        x_max_roi = x_min_roi + (x_max_img - x_min_img)
        y_max_roi = y_min_roi + (y_max_img - y_min_img)

        # Copy image data into the padded ROI
        roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi] = image[:, y_min_img:y_max_img, x_min_img:x_max_img]
        
        # Filter out ROIs with too many NaNs
        if torch_flag:
            nan_fraction = torch.isnan(roi).sum().item() / roi.numel()
        else:
            nan_fraction = xp.isnan(roi).sum() / roi.size
            
        if nan_fraction <= max_nan_fraction:
            ROIs.append(roi)
            # Store the local coordinates where the actual image data is inside the NaN-padded ROI
            roi_local_coords.append(((y_min_roi, y_max_roi), (x_min_roi, x_max_roi)))
            # Store the global coordinates relative to the original image
            roi_global_coords.append(((y_min_img, y_max_img), (x_min_img, x_max_img)))
    
    return ROIs, roi_local_coords, roi_global_coords


def add_ROIs(image, ROIs, local_coords, global_coords):    
    for roi, local_idx, global_idx in zip(ROIs, local_coords, global_coords):
        (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
        (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx

        image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    
    return image


def add_ROI(image, ROI, local_coord, global_coord):    
    (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coord
    (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coord
    image[:, y_min_img:y_max_img, x_min_img:x_max_img] += ROI[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    
    return image


def plot_ROIs_as_grid(ROIs, cols=5):
    """Display the ROIs in a grid of subplots."""
    n_ROIs = len(ROIs)
    rows = (n_ROIs + cols - 1) // cols  # Calculate number of rows needed
    _, axes = plt.subplots(rows, cols) #, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, roi in enumerate(ROIs):
        ax = axes[i]
        roi_ = roi.cpu() if roi.ndim == 2 else roi.sum(dim=0).cpu()  # Remove the channel dimension if it exists
        norm = simple_norm(roi_, 'log', percent=100-1e-1)
        ax.imshow(roi_, origin='lower', cmap='gray', norm=norm)
        # ax.set_title(f'Source {i+1}')
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # plt.tight_layout()
    plt.show()


ROIs, local_coords, global_coords = extract_ROIs(data_onsky_sparse, sources, box_size=box_size)
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
src_spectra_sparse = [GetSpectrum(data_onsky_sparse, sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_binned = [GetSpectrum(data_onsky,        sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_full   = [GetSpectrum(data_onsky_full,   sources.iloc[i], radius=flux_core_radius) for i in range(N_src)]


# STRANGE LINE IN THE FIRST SPECTRUM!!!!!
#%%
i_src = 1

plt.plot(λ, src_spectra_full[i_src], linewidth=0.25, alpha=0.3)
plt.scatter(wavelength.squeeze().cpu().numpy()*1e9, src_spectra_binned[i_src].cpu().numpy())
plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_sparse[i_src].cpu().numpy())


#%%
with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

df = data['All data']
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
from PSF_models.TipTorch import TipTorch_new
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
model = TipTorch_new(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
model.apodizer = model.make_tensor(1.0)

model.to_float()
model.to(device)
#%
# PSF_1 = model()

#%%
from data_processing.normalizers import Uniform, InputsManager

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


#%%
class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.25):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x


class Calibrator(nn.Module):

    def __init__(self, inputs_manager, calibrator_network, predicted_values, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.predicted_values = predicted_values
        
        # Initialiize inputs normalizer and staker/unstacker
        self.normalizer = InputsTransformer({ inp: inputs_manager.get_transform(inp) for inp in predicted_values })
        _ = self.normalizer.stack({ inp: inputs_manager[inp] for inp in predicted_values }, no_transform=True)

        # Initialize the calibrator network
        net_class      = calibrator_network['artichitecture']
        inputs_size    = calibrator_network['inputs_size']
        outputs_size   = calibrator_network['outputs_size'] if 'outputs_size' in calibrator_network else self.normalizer.get_stacked_size()
        weights_folder = calibrator_network['weights_folder']
        NN_kwargs      = calibrator_network['NN_kwargs']

        self.net = net_class(inputs_size, outputs_size, **NN_kwargs)
        self.net.to(device)
        self.net.float() # TODO: support double precision
        self.net.load_state_dict(torch.load(weights_folder, map_location=torch.device('cpu')))

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def forward(self, x):
        if type(x) is pd.DataFrame or type(x) is pd.Series:
            NN_inp = torch.as_tensor(x.to_numpy(), device=device)
        elif type(x) is list or type(x) is np.ndarray:
            NN_inp = torch.as_tensor(x, device=device)
        elif type(x) is torch.Tensor:
            NN_inp = x
        else:
            raise ValueError('NN_inputs must be a pandas DataFrame, numpy array, list, or torch tensor')

        if NN_inp.ndim == 1: NN_inp = NN_inp.unsqueeze(0)

        NN_inp = NN_inp.float() # TODO: support double precision

        # Scale the inputs back to the original range and pack them into the dictionary format
        return self.normalizer.unstack(self.net(NN_inp))


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
config_file['sensor_science']['FieldOfView'] = box_size # Set the actual size of the simulated PSFs
# model.Update(reinit_grids=True, reinit_pupils=True)
model.Update(init_grids=True, init_pupils=True, init_tomography=True)

with torch.no_grad():
    PSF_pred_small = model(pred_inputs)

inputs_manager.update(pred_inputs)

#%%
# How much flux is cropped by assuming the finite size of the PSF box (PSF_predbif is assumed to be quasi-infinite)
crop_ratio = (PSF_pred_big.amax(dim=(-2,-1)) / PSF_pred_small.amax(dim=(-2,-1))).squeeze()

def masked_flux_ratio(PSF, mask):
    F_norm = (PSF*mask).sum(dim=(-2,-1), keepdim=True) / PSF.sum(dim=(-2,-1), keepdim=True)
    return F_norm

core_mask     = torch.tensor(mask_circle(box_size, flux_core_radius+1)[None,None,...]).to(device).float()
core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], flux_core_radius+1)[None,None,...]).to(device).float()

# How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob"
core_flux_ratio = masked_flux_ratio(PSF_pred_big, core_mask_big).squeeze()

transformer_dict_astrometry = { 'dx': df_transforms_fitted['dx'], 'dy': df_transforms_fitted['dy'] }

transformer_dxdy = InputsTransformer(transformer_dict_astrometry)
_ = transformer_dxdy.stack({ attr: getattr(model, attr) for attr in transformer_dict_astrometry })

expand_dxdy = lambda x_, N: x_.unsqueeze(0).T.repeat(1, N).flatten().unsqueeze(0)
func = lambda x_: model(pred_inputs | transformer_dxdy.unstack(expand_dxdy(x_, N_wvl)))

def fit_dxdy(source_id, verbose=0):
    PSF_data = torch.nan_to_num(ROIs[source_id].unsqueeze(0)) * flux_λ_norm / src_spectra_sparse[source_id][None,:,None,None]
    # the last one is a sort of normalizing fetch-factor 
    peaks_scaler = PSF_pred_big.amax(dim=(-2,-1)) / (PSF_data*core_mask).amax(dim=(-2,-1))
    PSF_data *= peaks_scaler[:, :, None, None]
    
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_data*core_mask, func(dxdy_)*core_mask, reduction='sum')*1e3
    result = minimize(loss, torch.zeros([2], device=device).float(), max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    return func(result.x).detach().clone().squeeze(), result.x.detach().clone()

#%%
PSFs_coords_fitted, dxdys = [], []

# The PSF model generates PSFs normalized to sum of 1 per wavelength. We need to normalize them to the flux of the sources.
# To do so, we need to account for: 1. the flux normalization factor, 2. the crop ratio, 3. the core to wings flux ratio.

PSF_norm_factor = N_core_pixels / flux_λ_norm / core_flux_ratio / crop_ratio

for i in tqdm(range(N_src)):
    PSF_fitted, dxdy = fit_dxdy(i, verbose=0)
    PSFs_coords_fitted.append(PSF_fitted * (src_spectra_sparse[i] * PSF_norm_factor)[:,None,None])
    dxdys.append(dxdy)
    
PSFs_coords_fitted = torch.stack(PSFs_coords_fitted, dim=0)
dxdys = torch.stack(dxdys, dim=0)

# y = flux_λ_norm / core_flux_ratio / crop_ratio * N_core_pixels
# y = y.cpu().numpy()

# x = wavelength_selected.squeeze().cpu().numpy()*1e9
# plt.plot(x, y, label='Normalization factor')


#%%
composite_img = add_ROIs(
    torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device), # blanck baase image
    [PSFs_coords_fitted[i,...] for i in range(N_src)], # predicted flux-normalized PSFs after coordinates tuning
    local_coords,
    global_coords
)

# It tells the average flux in the PSF core for each source
src_spectra_sparse = [GetSpectrum(data_onsky_sparse, sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]
src_spectra_fitted = [GetSpectrum(composite_img,     sources.iloc[i], radius=flux_core_radius) * flux_λ_norm for i in range(N_src)]

# STRANGE LINE IN THE FIRST SPECTRUM!!!!!
#%
# i_src = 1
# plt.plot(λ, src_spectra_full[i_src], linewidth=0.25, alpha=0.3)
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_fitted[i_src].cpu().numpy())
# plt.scatter(wavelength_selected.squeeze().cpu().numpy()*1e9, src_spectra_sparse[i_src].cpu().numpy())

#%%
ROI_plot = np.s_[..., 125:225, 125:225]

diff_img = (data_onsky_sparse-composite_img) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()
plt.imshow(np.maximum(composite_img[ROI_plot].sum(dim=0).abs().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
plt.show()
plt.imshow(diff_img[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()

#%%
ROIs_0, _, _ = extract_ROIs(data_onsky_sparse, sources, box_size=box_size)
ROIs_1, _, _ = extract_ROIs(composite_img, sources, box_size=box_size)

# display_mask = torch.tensor(mask_circle(PSFs_coords_fitted.shape[-2], 16)[None,None,...]).to(device).float()

PSFs_0_white = np.mean(torch.stack(ROIs_0).cpu().numpy(), axis=1)
PSFs_1_white = np.mean(torch.stack(ROIs_1).cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)
plt.show()

#%% ====================================================== Fitting =======================================================
from tools.utils import OptimizableLO

LO_basis = OptimizableLO(model, ignore_pupil=False)

inputs_manager.set_optimizable('LO_coefs', False)
# inputs_manager.set_optimizable('s_pow', False)
inputs_manager.set_optimizable('Jxy', False)

inputs_manager.delete('amp')
inputs_manager.delete('beta')
inputs_manager.delete('alpha')
inputs_manager.delete('b')
inputs_manager.delete('ratio')
inputs_manager.delete('theta')
inputs_manager.delete('s_pow')
inputs_manager.delete('dx')
inputs_manager.delete('dy')

inputs_manager.add('PSF_norm', PSF_norm_factor, Uniform(a=1e4, b=2e4))
inputs_manager.set_optimizable('PSF_norm', False)

# inputs_manager.set_optimizable('dx', False)
# inputs_manager.set_optimizable('dy', False)
# inputs_manager.set_optimizable(['amp', 'beta', ], False)

print(inputs_manager)


#%%
x0 = inputs_manager.stack().squeeze().clone().detach()
x_size = inputs_manager.get_stacked_size()

x0_dict = inputs_manager.unstack(x0.unsqueeze(0), include_all=False)

#%%
Fs_flat = torch.ones([N_src], device=device)
x0 = torch.cat([x0, Fs_flat, dxdys.flatten()]) # [PSF params, flux, dx/dy] are packed into one vector
x0.requires_grad = True

empty_img   = torch.zeros([N_wvl, data_onsky_sparse.shape[-2], data_onsky_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 0 + 1

#%%
def func_fit(x): # TODO: relative weights for different brigtness
    PSFs_fit = []
    for i in range(N_src):
        x_fit_dict = inputs_manager.unstack(x[:x_size].unsqueeze(0))
        
        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        dxdy_ = x[x_size+N_src+i*2 : x_size+N_src+(i+1)*2]
        dxdy_dict  = transformer_dxdy.unstack(expand_dxdy(dxdy_, N_wvl))

        inputs = x_fit_dict | dxdy_dict
        flux_norm = (src_spectra_sparse[i] * PSF_norm_factor)[:,None,None] * x[x_size+i]
        
        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
    
    return add_ROIs( empty_img*0.0, PSFs_fit, local_coords, global_coords )


def loss_fit(x_):
    PSFs_ = func_fit(x_)
    l1 = F.smooth_l1_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    l2 = F.mse_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    return l1 * 1e-3 + l2 * 5e-6


_ = func_fit(x0)
    
#%%
result_global = minimize(loss_fit, x0, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x0 = result_global.x.clone()
x_fit_dict = inputs_manager.unstack(x0.unsqueeze(0), include_all=False)

dxdy_dicts = []
for i in range(N_src):
    dxdy_ = result_global.x[x_size+N_src+i*2 : x_size+N_src+(i+1)*2]
    dxdy_dicts.append( transformer_dxdy.unstack(expand_dxdy(dxdy_, N_wvl)) )

flux_corrections = result_global.x[x_size:x_size+N_src]

#%% 
# with torch.no_grad():
composite_img_fit = func_fit(result_global.x).detach()
diff_img = (data_onsky_sparse-composite_img_fit) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()
plt.imshow(np.maximum(composite_img_fit[ROI_plot].sum(dim=0).abs().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
plt.show()
plt.imshow(diff_img[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()

#%%
ROIs_0, _, _ = extract_ROIs(data_onsky_sparse, sources, box_size=box_size)
ROIs_1, _, _ = extract_ROIs(composite_img_fit, sources, box_size=box_size)

PSFs_0_white = np.mean(torch.stack(ROIs_0).cpu().numpy(), axis=1)
PSFs_1_white = np.mean(torch.stack(ROIs_1).cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-2)
plt.show()


#%% 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
from scipy.optimize import curve_fit

class LineModel:
    """
    A PyTorch-compatible line model for non-linear optimization.

    The model function is defined as:
        f(x; k, y_intercept) = k * x + y_intercept
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the LineModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def line_function(x, k, y_intercept):
        """
        Compute the line function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            k (float): Slope.
            y_intercept (float): Y-intercept.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return k * x + y_intercept

    def fit(self, y: torch.Tensor):
        """
        Fit the line model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [k, y_intercept].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [k, y_intercept].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, k, y_intercept):
            return self.line_function(x, k, y_intercept)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=[1e-6, 1e-6])
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (k, y_intercept).

        Returns:
            Tensor: Computed y-values.
        """
        return self.line_function(self.x, *params)


class QuadraticModel:
    """
    A PyTorch-compatible quadratic model for non-linear optimization.

    The model function is defined as:
        f(x; a, b, c) = a * x^2 + b * x + c
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the QuadraticModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def quadratic_function(x, a, b, c):
        """
        Compute the quadratic function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            a (float): Quadratic coefficient.
            b (float): Linear coefficient.
            c (float): Constant term.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return a * x ** 2 + b * x + c

    def fit(self, y: torch.Tensor, p0=[1e-6, 1e-6, 1e-6]):
        """
        Fit the quadratic model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [a, b, c].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [a, b, c].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, a, b, c):
            return self.quadratic_function(x, a, b, c)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=p0)
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (a, b, c).

        Returns:
            Tensor: Computed y-values.
        """
        return self.quadratic_function(self.x, *params)



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
# inputs_manager.set_optimizable(['ratio', 'theta', 'alpha', 'beta', 'amp', 'b'], False)
inputs_manager.set_optimizable(['F', 'Jx', 'Jy'],  False)

print(inputs_manager)

#%%
x_size_curve = curve_inputs.get_stacked_size()
x_size_model = inputs_manager.get_stacked_size()
x_size_total = x_size_curve + x_size_model

x2_1 = curve_inputs.stack().squeeze().clone().detach()
x2_2 = inputs_manager.stack().squeeze().clone().detach()
x2 = torch.cat([x2_1, x2_2, Fs_flat, dxdys.flatten()]) # [curve params, PSF params, flux, dx/dy] are packed into one vector
x2.requires_grad = True

empty_img = torch.zeros([N_wvl, data_onsky_sparse.shape[-2], data_onsky_sparse.shape[-1]], device=device)

#%%
def func_fit_curve(x):
    PSFs_fit = []
    for i in range(N_src):
        x_fit_dict = inputs_manager.unstack(x[x_size_curve:x_size_total].unsqueeze(0))
        
        curve_dict_ = curve_inputs.unstack(x[:x_size_curve].unsqueeze(0))
        x_curve_dict = {p: curve_sample(λ_sparse, curve_dict_, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
        
        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        dxdy_ = x[x_size_total+N_src+i*2 : x_size_total+N_src+(i+1)*2]
        dxdy_dict  = transformer_dxdy.unstack(expand_dxdy(dxdy_, N_wvl))

        inputs = x_fit_dict | x_curve_dict | dxdy_dict
        flux_norm = (src_spectra_sparse[i] * norm_new)[:,None,None] * x[x_size_total+i]
        
        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
    
    return add_ROIs( empty_img*0.0, PSFs_fit, local_coords, global_coords )


def loss_fit_curve(x_):
    PSFs_ = func_fit_curve(x_)
    l1 = F.smooth_l1_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    l2 = F.mse_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    return l1 * 1e-3 + l2 * 5e-6


_ = func_fit_curve(x2)

    
#%%
result_global = minimize(loss_fit_curve, x2, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x2 = result_global.x.clone()
x_fit_dict = inputs_manager.unstack(x2[x_size_curve:x_size_total].unsqueeze(0))
x_curve_fit_dict = curve_inputs.unstack(x2[:x_size_curve].unsqueeze(0))
flux_corrections = result_global.x[x_size_total:x_size_total+N_src]

#%% 
# with torch.no_grad():
composite_img_fit = func_fit_curve(result_global.x).detach()
diff_img = (data_onsky_sparse-composite_img_fit) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse[ROI_plot].abs().sum(dim=0).cpu(), norm=norm_field, origin='lower')
plt.show()
plt.imshow(np.maximum(composite_img_fit[ROI_plot].sum(dim=0).abs().cpu(), 5e3), norm=norm_field, origin='lower')
plt.show()
plt.imshow(diff_img[ROI_plot].abs().sum(dim=0).cpu(), norm=norm_field, origin='lower')
plt.show()

#%%
ROIs_0, _, _ = extract_ROIs(data_onsky_sparse, sources, box_size=box_size)
ROIs_1, _, _ = extract_ROIs(composite_img_fit, sources, box_size=box_size)

PSFs_0_white = np.mean(torch.stack(ROIs_0).cpu().numpy(), axis=1)
PSFs_1_white = np.mean(torch.stack(ROIs_1).cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-2)
plt.show()

#%%
host_coords = [51, 38]
AGN_coords_1  = [47, 39]
bg_coords   = [14, 72]

host_spectrum = GetSpectrum(diff_img[ROI_plot], host_coords, radius=1, debug_show_ROI=False) * flux_λ_norm
bg_spectrum  = GetSpectrum(diff_img[ROI_plot], bg_coords, radius=7, debug_show_ROI=False) * flux_λ_norm
AGN_residue  = GetSpectrum(diff_img[ROI_plot], AGN_coords_1, radius=1, debug_show_ROI=False) * flux_λ_norm

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(λ_sparse.cpu(), host_spectrum.cpu(), linewidth=0.5, alpha=0.75, label='Host galaxy?')
plt.plot(λ_sparse.cpu(), bg_spectrum.cpu(), linewidth=0.5, alpha=0.75, label='Background')
plt.plot(λ_sparse.cpu(), AGN_residue.cpu(), linewidth=0.5, alpha=0.75, label='AGN residue (after subtraction)')
plt.plot(λ_sparse.cpu(), src_spectra_sparse[0].cpu(), linewidth=0.5, alpha=0.75, label='AGN image (before subtraction)')

plt.legend()
plt.ylim(0, None)
plt.xlim(λ.min(), λ.max())
plt.grid(alpha=0.2)
plt.title('Full-spectral range subtraction')
plt.show()

#%%
torch.cuda.empty_cache()
model_inputs_full_λ = {p: curve_sample(torch.as_tensor(λ, device=device), x_curve_fit_dict, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
norms_new_full_λ = curve_sample(torch.as_tensor(λ, device=device), x_curve_fit_dict, 'norm')

# idx = (np.abs(λ - wvl)).argmin()
split_size = 100
# Split λ array into batches of size 100
λ_batches = [λ[i:i + split_size] for i in range(0, len(λ), split_size)]

composite_img_full = []

for batch_id in tqdm(range(len(λ_batches))):
    batch_size = len(λ_batches[batch_id])
    config_file['sources_science']['Wavelength'] = torch.as_tensor(λ_batches[batch_id]*1e-9, device=device).unsqueeze(0)
    model.Update(init_grids=True, init_pupils=True, init_tomography=True)

    empty_img = torch.zeros([batch_size, data_onsky_sparse.shape[-2], data_onsky_sparse.shape[-1]], device=device)
    PSF_batch = []

    for i in range(N_src):
        batch_ids = slice(batch_id*batch_size, (batch_id+1)*batch_size)

        dict_selected = {
            key: model_inputs_full_λ[key][:,batch_ids]
            for key in model_inputs_full_λ.keys()
        }

        dxdy_ = x2[x_size_total+N_src+i*2 : x_size_total+N_src+(i+1)*2]
        dxdy_dict = {
            'dx': expand_dxdy(transformer_dict_astrometry['dx'](dxdy_[0]), batch_size),
            'dy': expand_dxdy(transformer_dict_astrometry['dy'](dxdy_[1]), batch_size) 
        }

        dict_selected = x_fit_dict |dict_selected | dxdy_dict
        dict_selected['bg'] = torch.zeros([1, batch_size], device=device)

        del dict_selected['LO_coefs']
        del dict_selected['PSF_norm']

        flux_norm = norms_new_full_λ[batch_ids]\
            * flux_corrections[i]\
            * torch.as_tensor(src_spectra_full[i][batch_ids], device=device) \
            * flux_λ_norm

        PSF_batch.append( model(dict_selected).squeeze() * flux_norm[:,None,None] )

    composite_img_full.append( add_ROIs( empty_img*0.0, PSF_batch, local_coords, global_coords ).cpu().numpy() )

composite_img_full = np.vstack(composite_img_full)


#%%
diff_img_full = data_onsky_full-composite_img_full * valid_mask[None,...]
norm_full = LogNorm(vmin=1e1, vmax=thres*10)

plt.imshow(np.abs(data_onsky_full[ROI_plot].sum(axis=0)), norm=norm_full, origin='lower')
plt.imshow(np.abs((data_onsky_full).sum(axis=0)), norm=norm_full, origin='lower')
plt.show()
plt.imshow(composite_img_full[ROI_plot].sum(axis=0), norm=norm_full, origin='lower')
plt.show()
plt.imshow(np.abs((diff_img_full[ROI_plot]).sum(axis=0)), norm=norm_full, origin='lower')
plt.show()

#%%
ROIs_0, _, _ = extract_ROIs(data_onsky_full, sources, box_size=box_size)
ROIs_1, _, _ = extract_ROIs(composite_img_full, sources, box_size=box_size)

PSFs_0_white = np.mean(np.stack(ROIs_0), axis=1)
PSFs_1_white = np.mean(np.stack(ROIs_1), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-2)
plt.show()

#%%
from astropy.convolution import convolve, Box1DKernel

host_coords  = [53, 37]
AGN_coords_1 = [47, 39]
bg_coords    = [56, 39]
AGN_coords_2 = [61, 37]
lens_coords  = [55, 65]

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
kernel = Box1DKernel(20)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10,6))

labels, colors = [], []

# Loop through each spectrum
for info in targets_info:
    spectrum_full = GetSpectrum(diff_img_full[ROI_plot], info['coords'], radius=info['radius'], debug_show_ROI=False )
    spectrum_avg = convolve(spectrum_full, kernel, boundary='extend')
    
    labels.append(info['name'])
    colors.append(info['color'])
    
    plt.plot(λ, spectrum_full,
             linewidth=0.5,
             alpha=0.25,
             label=info['name'],
             color=info['color'])

    plt.plot(λ, spectrum_avg,
             linewidth=1,
             alpha=1,
             color=info['color'],
             linestyle='--')
    
plt.legend()
# plt.ylim(0, None)
plt.xlim(λ.min(), λ.max())
plt.grid(alpha=0.2)
# plt.title('Full-spectrum: (AGN, AGN residue, host(?), background)')
plt.title('Full-spectrum: (AGN residue, Host(?), background)')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.xlabel('Wavelength, [nm]')
plt.show()


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
    data_onsky_full[ROI_plot],
    wavelengths=λ_vis,
    title=f"{reduced_name}\nObservation",
    min_val=500, max_val=200000, show=True
)
# %%
