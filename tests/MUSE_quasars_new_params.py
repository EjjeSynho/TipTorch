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


nan_mask = np.abs(np.nansum(test_fits[1].data, axis=0)) < 1e-12
nan_mask = binary_dilation(nan_mask, iterations=2, )
valid_mask = ~nan_mask

test_fits.close()

with open(MUSE_DATA_FOLDER + f"quasars/J0259_reduced/{reduced_name}.pickle", 'rb') as f:
    data = pickle.load(f)
    
#%%
# box_size = 31  # Define the size of each ROI (in pixels)
# box_size = 61  # Define the size of each ROI (in pixels)
box_size = 111  # Define the size of each ROI (in pixels)

data_onsky, var_mask, norms, bgs = LoadImages(data, device=device, subtract_background=False, normalize=False, convert_images=True)
data_onsky = data_onsky.squeeze()
data_onsky *= torch.tensor(valid_mask, device=device).float().unsqueeze(0)

#%%
data_src = data_onsky.sum(dim=0).cpu().numpy()
mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)

thres = 50000

sources = find_peaks(data_src, threshold=thres, box_size=11)
print(f"Detected {len(sources)} sources")

#%%
def merge_sources(sources, eps=2):
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

#%%
apertures = CircularAperture(srcs_pos, r=5)
apertures_box = RectangularAperture(srcs_pos, box_size//2, box_size//2)

norm_field = LogNorm(vmin=1e-0, vmax=500000)

plt.imshow(data_src, norm=norm_field, origin='lower', cmap='gray')
apertures_box.plot(color='gold', lw=2, alpha=0.45)
plt.show()

#%%
def extract_ROIs(image, sources, box_size=20, max_nan_fraction=0.3):   
    ROIs = []
    roi_local_coords = []  # To store the local image indexes inside NaN-padded ROI
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
        roi = torch.full((D, box_size, box_size), float('nan'), device=image.device)  # Create a blank 3D box filled with NaNs
        
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
        nan_fraction = torch.isnan(roi).sum().item() / roi.numel()
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


ROIs, local_coords, global_coords = extract_ROIs(data_onsky, sources, box_size=box_size)
N_src = len(ROIs)
# plot_ROIs_as_grid(ROIs, cols=np.ceil(np.sqrt(len(ROIs))).astype('uint'))  # Adjust the number of columns as needed

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


#%%
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new
# from tools.utils import SausageFeature

config_file, data_onsky = GetConfig(data, data_onsky)
data_onsky = data_onsky.squeeze()

config_file['NumberSources'] = 1 #N_src
config_file['sensor_science']['FieldOfView'] = 511

wavelength = config_file['sources_science']['Wavelength'].clone()

ids_wavelength_selected = np.arange(0, wavelength.shape[-1], 2)
wavelength_selected = wavelength[..., ids_wavelength_selected]
config_file['sources_science']['Wavelength'] = wavelength_selected

N_wvl = len(ids_wavelength_selected)

data_onsky_sparse = data_onsky.clone()[ids_wavelength_selected,...]

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

# inputs_manager.set_optimizable(['ratio', 'theta', 'alpha', 'beta', 'amp', 'b'], False)
inputs_manager.set_optimizable(['ratio', 'theta'], False)
           
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
    PSF_pred_big = model(pred_inputs).clone()

#%%
config_file['sensor_science']['FieldOfView'] = box_size
# model.Update(reinit_grids=True, reinit_pupils=True)
model.Update(init_grids=True, init_pupils=True, init_tomography=True)

with torch.no_grad():
    PSF_pred_small = model(pred_inputs)

inputs_manager.update(pred_inputs)

#%%
cut_middle = lambda n,m: np.s_[..., n//2-m//2 : n//2 + m//2 + m%2, n//2-m//2 : n//2 + m//2 + m%2 ]

ROI_1 = cut_middle(PSF_pred_big.shape[-2], box_size)

ratio_crop = (PSF_pred_big.amax(dim=(-2,-1))/PSF_pred_small.amax(dim=(-2,-1))).unsqueeze(-1).unsqueeze(-1)

# diffa = PSF_pred_big.clone()
# diffa[ROI_1] -= PSF_pred_small * ratio_crop
# plt.imshow(diffa.squeeze().sum(dim=0).detach().abs().cpu().numpy(), norm=LogNorm(), origin='lower')

def masked_flux_ratio(PSF, mask):
    F_norm = (PSF*mask).sum(dim=(-2,-1), keepdim=True) / PSF.sum(dim=(-2,-1), keepdim=True)
    return F_norm

core_mask = torch.tensor(mask_circle(box_size, 3)[None,None,...]).to(device).float()
# more_mask = torch.tensor(mask_circle(box_size, 8)[None,None,...]).to(device).float()

# core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], 4)[None,None,...]).to(device).float()


#%%
from data_processing.MUSE_preproc_utils import GetRadialBackround
from scipy.ndimage import gaussian_filter

transformer_dict_astrometry = {
    'dx': df_transforms_fitted['dx'],
    'dy': df_transforms_fitted['dy']
}

ratio_core = masked_flux_ratio(PSF_pred_small, core_mask) #* ratio_crop

transformer_dxdy = InputsTransformer(transformer_dict_astrometry)
_ = transformer_dxdy.stack({ attr: getattr(model, attr) for attr in transformer_dict_astrometry })

expand_dxdy = lambda x_: x_.unsqueeze(0).T.repeat(1, N_wvl).flatten().unsqueeze(0)
func = lambda x_: model(pred_inputs | transformer_dxdy.unstack(expand_dxdy(x_)))


def fit_dxdy(source_id, verbose=0):
    if verbose > 0: 
        print(f'Fitting source {source_id}...')
    
    PSF_0 = torch.nan_to_num(ROIs[source_id].clone()[ids_wavelength_selected,...].unsqueeze(0))
    F_norm = (PSF_0 * core_mask).sum(dim=(-2,-1), keepdim=True) / ratio_core
    PSF_0 /= F_norm

    dxdy_0 = torch.zeros([2], device=device).float()

    _ = func(dxdy_0)
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_0, func(dxdy_), reduction='sum')*1e3
    result = minimize(loss, dxdy_0, max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    return PSF_0.clone(), func(result.x).detach().clone(), result.x.clone(), F_norm.clone()


PSFs_0, PSFs_1, fluxes, dxdys = [], [], [], []

for i in tqdm(range(len(ROIs))):
    PSF_0, PSF_1, dxdy, flux = fit_dxdy(i, verbose=0)
    PSFs_0.append(PSF_0)
    PSFs_1.append(PSF_1)
    fluxes.append(flux)
    dxdys.append(dxdy)
    
PSFs_0 = torch.vstack(PSFs_0)
PSFs_1 = torch.vstack(PSFs_1)
fluxes = torch.vstack(fluxes)
dxdys  = torch.stack(dxdys)

#%%
composite_img = add_ROIs(
    torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device),
    [(PSFs_1*fluxes)[i, ...] for i in range(PSFs_1.shape[0])],
    local_coords,
    global_coords
)

ROI_plot = np.s_[..., 125:225, 125:225]

diff_img = (data_onsky_sparse-composite_img) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()

plt.imshow(np.maximum(composite_img[ROI_plot].sum(dim=0).abs().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
plt.show()

plt.imshow(diff_img[ROI_plot].abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
plt.show()

#%
ROIs_1, _, _ = extract_ROIs(composite_img, sources, box_size=box_size)
PSFs_2 = torch.nan_to_num(torch.stack(ROIs_1) / fluxes * ratio_crop, nan=0.0)

PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1)
PSFs_2_white = np.mean(PSFs_2.cpu().cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_2_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)
plt.show()

#%%
from tools.utils import OptimizableLO

LO_basis = OptimizableLO(model, ignore_pupil=False)

inputs_manager.set_optimizable('LO_coefs', True)
inputs_manager.set_optimizable('s_pow', False)
inputs_manager.set_optimizable('Jxy', False)
# inputs_manager.set_optimizable('dx', False)
# inputs_manager.set_optimizable('dy', False)
# inputs_manager.set_optimizable(['amp', 'beta', ], False)

# _ = inputs_manager.stack()

print(inputs_manager)

# print(inputs_manager.get_stacked_size())

#%% ----- Fitting -----
x_size = inputs_manager.get_stacked_size()

x0 = inputs_manager.stack().squeeze().clone().detach()
Fs_flat = torch.ones([N_src], device=device)
x0 = torch.cat([x0, Fs_flat, dxdys.flatten()]) # [PSF params, flux, dx/dy] are packed into one vector
x0.requires_grad = True

empty_img = torch.zeros([N_wvl, data_onsky_sparse.shape[-2], data_onsky_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)

#%%
def func_fit(x):
    PSFs_fit = []
    for i in range(N_src):
        x_fit_dict = inputs_manager.unstack(x[:x_size].unsqueeze(0))
        
        phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        dxdy_ = x[x_size+N_src+i*2 : x_size+N_src+(i+1)*2]
        dxdy_dict  = transformer_dxdy.unstack(expand_dxdy(dxdy_))

        inputs = x_fit_dict | dxdy_dict
        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * fluxes[i, ...] * x[x_size+i] )
    
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
x_fit_dict = calibrator.normalizer.unstack(x0[:x_size].unsqueeze(0))

dxdy_dicts = []

for i in range(N_src):
    dxdy_ = result_global.x[x_size+N_src+i*2 : x_size+N_src+(i+1)*2]
    dxdy_dicts.append( transformer_dxdy.unstack(expand_dxdy(dxdy_)) )

flux_corrections = result_global.x[x_size:x_size+N_src]

#%
for key in pred_inputs.keys():
    percent_err = 100 * (x_fit_dict[key] - pred_inputs[key]) / pred_inputs[key]
    percent_err = percent_err.squeeze().detach().cpu().numpy()
    percent_err = np.round(percent_err, 1)
    print(f'{key} : {percent_err} %')


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
ROIs_1, _, _ = extract_ROIs(composite_img_fit, sources, box_size=box_size)
PSFs_2 = torch.nan_to_num(torch.stack(ROIs_1) / fluxes, nan=0.0)

PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1)
# PSFs_2_white = np.mean(PSFs_2.cpu().cpu().numpy(), axis=1) #*0.9
PSFs_2_white = np.mean((PSFs_2*ratio_crop).cpu().cpu().numpy(), axis=1)#*0.9

plot_radial_profiles_new(PSFs_0_white, PSFs_2_white, 'Data', 'TipTorch', title='PSFs fitted over the field', cutoff=16, y_min=1e-1)


#%%
from tools.utils import wavelength_to_rgb


wavelength_display = wavelength_selected.cpu().numpy()[0,...] * 1e9

wavelength_display = np.linspace(380, 750, wavelength_display.shape[0])


def plot_wavelength_rgb(image, wavelengths=None, ROI=None, log_offset=1.5, clip_min=0, rgb_scale=3,
                       title=None, origin='lower', show=True):
    """
    Create and display a logarithmic RGB visualization of a wavelength-dependent image.

    Parameters
    ----------
    image : torch.Tensor or numpy.ndarray
        Input image with shape (n_wavelengths, height, width)
    wavelengths : array-like, optional
        Wavelengths in nanometers. If None, uses evenly spaced values between 380-750nm
    ROI : slice or tuple, optional
        Region of interest to display. If None, shows full image
    log_offset : float, optional
        Offset subtracted from log-scaled image (higher values make image darker)
    clip_min : float, optional
        Minimum value after log_offset subtraction (values below are clipped to 0)
    rgb_scale : float, optional
        Multiplication factor for final RGB values
    title : str, optional
        Plot title
    origin : str, optional
        Plot origin ('lower' or 'upper')
    show : bool, optional
        If True, displays the plot immediately

    Returns
    -------
    numpy.ndarray
        The RGB image array
    """
    # Convert to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    # Apply ROI if specified
    if ROI is not None:
        image = image[ROI]

    # Generate wavelengths if not provided
    if wavelengths is None:
        wavelengths = np.linspace(380, 750, image.shape[0])
    wavelengths = np.asarray(wavelengths)

    # Generate RGB values for each wavelength
    R, G, B = np.zeros_like(wavelengths), np.zeros_like(wavelengths), np.zeros_like(wavelengths)
    for i, λ in enumerate(wavelengths):
        R[i], G[i], B[i] = wavelength_to_rgb(λ, show_invisible=True)
    RGB_weights = np.stack([R, G, B], axis=0)

    # Create RGB image
    image_log = np.log10(np.abs(image) + 1e-10)  # Add small constant to avoid log(0)
    image_RGB = (image_log[None, ...] * RGB_weights[...,None,None]).transpose(1,2,3,0)

    # Apply scaling and clipping
    image_RGB = image_RGB - log_offset
    image_RGB = np.clip(image_RGB, clip_min, None)
    # Normalize each color channel independently
    image_RGB = image_RGB / image_RGB.max(axis=(-3,-2), keepdims=True)
    image_RGB = np.nan_to_num(image_RGB, nan=0.0) * rgb_scale

    # Display the result
    if show:
        plt.figure()
        plt.imshow(image_RGB.mean(axis=0), origin=origin)
        if title:
            plt.title(title)
            plt.axis('off')
            plt.show()

    return image_RGB

# Example usage:
# For the original case:
rgb_image = plot_wavelength_rgb(
    diff_img,
    wavelengths=wavelength_display,
    ROI=ROI_plot,
    log_offset=1.5,
    clip_min=0,
    rgb_scale=3,
    title=f"{reduced_name}\nDifference"
)

# For a different image with custom parameters:
# rgb_image = plot_wavelength_rgb(
#     some_other_image,
#     log_offset=2.0,
#     clip_min=0.1,
#     rgb_scale=2
# )

rgb_image = plot_wavelength_rgb(
    data_onsky_sparse,
    wavelengths=wavelength_display,
    ROI=ROI_plot,
    log_offset=1.5,
    clip_min=0,
    rgb_scale=3,
    title=f"{reduced_name}\nData"
)


# %%
