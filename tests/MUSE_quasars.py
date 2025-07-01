#%%
try:
    ipy = get_ipython()        # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_settings import PROJECT_PATH, MUSE_DATA_FOLDER, device, default_torch_type

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from torchmin import minimize
from tqdm import tqdm
from pathlib import Path

from tools.utils import mask_circle
from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE
from data_processing.normalizers import CreateTransformSequenceFromFile
from data_processing.MUSE_onsky_df import *


#%
'''
import os
from astropy.io import fits
from astroquery.eso import Eso
from astroquery.exceptions import NoResultsWarning
import warnings


def parse_muse_cube_header(cube_path):
    hdr = fits.getheader(cube_path, ext=0)
    prog_id   = hdr.get('HIERARCH ESO OBS PROG ID')
    obs_block = hdr.get('HIERARCH ESO OBS ID')
    if prog_id is None or obs_block is None:
        raise KeyError(f"Could not find ESO PROG ID / OBS ID in header of {cube_path}")
    return prog_id.strip(), obs_block


def find_raw_muse_exposures(prog_id, obs_block, eso=None):
    eso = eso or Eso()
    # If you haven’t yet, authenticate for proprietary data:
    # eso.login(store_password=True)
    
    # (Optional) Inspect what filters and columns the MUSE‐form supports:
    # eso.query_instrument('muse', help=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoResultsWarning)
        raw_table = eso.query_instrument(
            'muse',
            column_filters={
                # These keys must match exactly the HTML form inputs;
                # if in doubt, run the help=True step above.
                'prog_id':   prog_id,
                'obs_id':    obs_block,
            },
            # explicitly request the filename & download URL columns
            columns=['filename', 'productURL']
        )
    if raw_table is None or len(raw_table) == 0:
        raise RuntimeError(f"No raw MUSE exposures found for {prog_id}, {obs_block}")
    return raw_table


def download_raw_files(raw_table, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eso = Eso()
    paths = []
    for row in raw_table:
        fn = row['filename']
        url= row['productURL']
        dest = os.path.join(out_dir, fn)
        if not os.path.exists(dest):
            print(f"Downloading {fn} …")
            eso.retrieve_data(url, location=out_dir)
        else:
            print(f"Already have {fn}; skipping.")
        paths.append(dest)
    return paths


def fetch_raw_for_cube(cube_path, out_dir='.'):
    prog_id, obs_block = parse_muse_cube_header(cube_path)
    print(f"Programme: {prog_id}, OB: {obs_block}")
    tbl = find_raw_muse_exposures(prog_id, obs_block)
    print(f"Found {len(tbl)} raw exposures; downloading …")
    return download_raw_files(tbl, out_dir)
'''

#%%
# Define the paths to the raw and reduced MUSE NFM cubes. The cached data cube will be generated based on them
data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data
data_folder = MUSE_DATA_FOLDER / 'quasars/' # change to your actual path with the MUSE NFM data

if not isinstance(data_folder, Path):
    data_folder = Path(data_folder)

# raw_path   = data_folder / "J0259/MUSE.2024-12-05T03_15_37.598.fits.fz"
# cube_path  = data_folder / "J0259/J0259-0901_all.fits"
# cache_path = data_folder / "J0259/J0259-0901_all.pickle"

cube_path  = data_folder / "J0144/J0144-5745.fits"
cache_path = data_folder / "J0144/J0144-5745_cache.pickle"
raw_path   = data_folder / "J0144/J0144_raw.114.fits.fz"

#%
# We need to pre-process the data before using it with the model and asssociate the reduced telemetry - this is done by the LoadDataCache function
# You need to run this function at least ones to generate the data cache file. Then, te function will automatically reduce it ones it's found
spectral_cubes, spectral_info, data_cached, model_config = LoadCachedDataMUSE(raw_path, cube_path, cache_path, save_cache=True, device=device, verbose=True)   
# Extract full and binned spectral cubes. Sparse cube selects a set of 7 binned wavelengths ranges
cube_full, cube_sparse, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_sparse"], spectral_cubes["mask"]

N_wvl = cube_sparse.shape[0] # dims are [N_wvls, H, W]

λ_full,   λ_sparse = spectral_info['λ_full'],  spectral_info['λ_sparse']
Δλ_full, Δλ_binned = spectral_info['Δλ_full'], spectral_info['Δλ_binned']

#TODO: fix binned and sparse consistency
flux_λ_norm = torch.tensor(Δλ_full / Δλ_binned, device=device, dtype=torch.float32)
cube_sparse *= flux_λ_norm[:, None, None]

#%%
from managers.multisrc_manager import add_ROIs, DetectSources, ExtractSources

PSF_size = 111  # Define the size of each extracted PSF

sources = DetectSources(cube_sparse, threshold='auto', nsigma=20, display=True, draw_win_size=20)
# Extract separate source images from the data + other data, necessary for later fitting and performance evaluation
srcs_image_data = ExtractSources(cube_sparse, sources, box_size=PSF_size, filter_sources=True, debug_draw=False)

N_src   = srcs_image_data["count"]
sources = srcs_image_data["coords"]
ROIs    = srcs_image_data["images"]

def get_src_coords_in_pixels(src_idx):
    return \
        np.round(sources['x_peak'].iloc[src_idx]).astype(int), \
        np.round(sources['y_peak'].iloc[src_idx]).astype(int)

#%%
# Correct for the difference in energy per λ bin
flux_core_radius = 2  # [pix]
N_core_pixels = (flux_core_radius*2 + 1)**2  # [pix^2]

# It tells the average flux in the PSF core for each source
src_spectra_sparse = [GetSpectrum(cube_sparse, sources.iloc[i], radius=flux_core_radius) for i in range(N_src)]
src_spectra_full   = [GetSpectrum(cube_full,   sources.iloc[i], radius=flux_core_radius) for i in range(N_src)]

colors = [f'tab:{color}' for color in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:N_src]]

plt.figure(figsize=(6, 4))
for i_src in range(N_src):
    plt.plot(λ_full, src_spectra_full[i_src], linewidth=0.25, alpha=0.3, color=colors[i_src], label=f'Source {i_src+1} (full spectrum)')
    
    plt.scatter(λ_sparse.cpu().numpy(), src_spectra_sparse[i_src].cpu().numpy(), color=colors[i_src], marker='o', s=30, alpha=0.8,
                label=f'Source {i_src+1} (sparse samples)')

plt.xlabel('Wavelength, [nm]')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.title('Sources spectra preview')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
from tools.utils import PupilVLT
from PSF_models.TipTorch import TipTorch

LO_map_size = 31

# Initialize the PSF model
pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device, dtype=default_torch_type)
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          False
}
model = TipTorch(model_config, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1, dtype=default_torch_type)
model.to_float()
model.to(device)

#%% For predicted and fitted model inputs, it is convenient to organize them using inputs_manager
from data_processing.normalizers import Uniform
from managers.input_manager import InputsManager, InputsManagersUnion

df_transforms_fitted = CreateTransformSequenceFromFile(PROJECT_PATH / 'data/reduced_telemetry/MUSE/muse_df_fitted_transforms.pickle')

shared_inputs = InputsManager()

shared_inputs.add('r0',    torch.tensor([model.r0[0].item()]), df_transforms_fitted['r0'])
shared_inputs.add('F',     torch.tensor([[1.0,]*N_wvl]),    df_transforms_fitted['F'])
shared_inputs.add('bg',    torch.tensor([[0,]*N_wvl]),      df_transforms_fitted['bg'])
# shared_inputs.add('dx',    torch.tensor([[0.0,]*N_wvl]),    df_transforms_fitted['dx'])
# shared_inputs.add('dy',    torch.tensor([[0.0,]*N_wvl]),    df_transforms_fitted['dy'])
shared_inputs.add('dn',    torch.tensor([1.5]),             df_transforms_fitted['dn'])
shared_inputs.add('Jx',    torch.tensor([[10,]*N_wvl]),     df_transforms_fitted['Jx'])
shared_inputs.add('Jy',    torch.tensor([[10,]*N_wvl]),     df_transforms_fitted['Jy'])
shared_inputs.add('Jxy',   torch.tensor([[0]]), df_transforms_fitted['Jxy'])
shared_inputs.add('amp',   torch.tensor([0.0]), df_transforms_fitted['amp'])
shared_inputs.add('b',     torch.tensor([0.0]), df_transforms_fitted['b'])
shared_inputs.add('alpha', torch.tensor([4.5]), df_transforms_fitted['alpha'])
shared_inputs.add('beta',  torch.tensor([2.5]), df_transforms_fitted['beta'])
shared_inputs.add('ratio', torch.tensor([1.0]), df_transforms_fitted['ratio'])
shared_inputs.add('theta', torch.tensor([0.0]), df_transforms_fitted['theta']) 
shared_inputs.add('s_pow', torch.tensor([0.0]), df_transforms_fitted['s_pow'])

if LO_map_size is not None:
    shared_inputs.add('LO_coefs', torch.zeros([1, LO_map_size**2]), Uniform(a=-100, b=100))
    shared_inputs.set_optimizable('LO_coefs', False)

shared_inputs.set_optimizable(['ratio', 'theta', 'alpha', 'beta', 'amp', 'b'], False)
shared_inputs.set_optimizable(['Jxy'], False)

shared_inputs.to_float()
shared_inputs.to(device)

print(shared_inputs)


individual_inputs = InputsManager()

individual_inputs.add('dx', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dx'])
individual_inputs.add('dy', torch.tensor([[0.0,]]*N_src),     df_transforms_fitted['dy'])
individual_inputs.add('F_norm', torch.tensor([[1.0,]]*N_src), df_transforms_fitted['F'])

individual_inputs.to_float()
individual_inputs.to(device)

individual_inputs.set_optimizable(['F_norm'], False)

print(individual_inputs)


#%%
from machine_learning.calibrator import Calibrator, Gnosis

def GetReducedTelemetryInputs(cached_data):
    with open(PROJECT_PATH / 'data/reduced_telemetry/MUSE/muse_df_norm_imputed.pickle', 'rb') as handle:
        muse_df_norm = pickle.load(handle)

    df = cached_data['All data']
    df['ID'] = 0
    df.loc[0, 'Pupil angle'] = 0.0

    df_pruned  = prune_columns(df.copy())
    df_reduced = reduce_columns(df_pruned.copy())
    df_transforms = CreateTransformSequenceFromFile(PROJECT_PATH / 'data/reduced_telemetry/MUSE/muse_df_norm_transforms.pickle')
    df_norm = normalize_df(df_reduced, df_transforms)
    df_norm = df_norm.fillna(0)

    selected_entries_input = muse_df_norm.columns.values.tolist()
    return df_norm[selected_entries_input].loc[0]


telemetry_inputs = GetReducedTelemetryInputs(data_cached)


calibrator = Calibrator(
    inputs_manager=shared_inputs,
    predicted_values = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow', 'amp', 'b', 'alpha'],
    device=device,
    dtype=default_torch_type,
    calibrator_network = {
        'artichitecture': Gnosis,
        'inputs_size': len(telemetry_inputs),
        'NN_kwargs': {
            'hidden_size': 200,
            'dropout_p': 0.1
        },
        'weights_folder': PROJECT_PATH / 'data/weights/MUSE_calibrator.dict'
    }
)
calibrator.eval()

predicted_model_inputs = calibrator(telemetry_inputs)
shared_inputs.update(predicted_model_inputs) # update the internal values of the inputs_manager class

#%%
with torch.no_grad():
    # Quasi-infinite PSF image to compute how much flux is lost while cropping
    model_config['sensor_science']['FieldOfView'] = 511
    model.Update(config=model_config, init_grids=True, init_pupils=True, init_tomography=True)
    PSF_pred_big = model(predicted_model_inputs).clone() # First initial prediction of the "big" PSF

    # The actual size of the simulated PSFs
    model_config['sensor_science']['FieldOfView'] = PSF_size
    model.Update(config=model_config, init_grids=True, init_pupils=True, init_tomography=True)
    PSF_pred_small = model(predicted_model_inputs)


#%% Compute spectral flux normalization factor
# How much flux is cropped by assuming the finite size of the PSF box (PSF_predbif is assumed to be quasi-infinite)
crop_ratio = (PSF_pred_big.amax(dim=(-2,-1)) / PSF_pred_small.amax(dim=(-2,-1))).squeeze()

core_mask     = torch.tensor(mask_circle(PSF_size, flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)
core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], flux_core_radius+1)[None,None,...], dtype=default_torch_type, device=device)

# How much flux is spread out of the PSF core because PSF is not a single pixel but rather "a blob"
core_flux_ratio = torch.squeeze((PSF_pred_big*core_mask_big).sum(dim=(-2,-1), keepdim=True) / PSF_pred_big.sum(dim=(-2,-1), keepdim=True))
# PSF_norm_factor = N_core_pixels / flux_λ_norm / core_flux_ratio / crop_ratio
PSF_norm_factor = N_core_pixels / core_flux_ratio / crop_ratio

norm_transform = Uniform(PSF_norm_factor.min().int().item(),  PSF_norm_factor.max().int().item())
shared_inputs.add('PSF_norm_factor', PSF_norm_factor.float().to(device), norm_transform, False)

#%% Fine tunes sources astrometry but don't touch the PSF model parameters
def func_dxdy(x_):
    dxdy_inp = individual_inputs.unstack(x_.unsqueeze(0), update=False) # Don't update interal values
    return model(predicted_model_inputs | dxdy_inp)


def fit_dxdy(i_src, verbose=0):
    dxdy_0 = individual_inputs.stack()[i_src,:]
    PSF_data = torch.nan_to_num(ROIs[i_src].unsqueeze(0)) * (src_spectra_sparse[i_src]).view(1,-1,1,1)
    # [None,:,None,None]
    peaks_scaler = PSF_pred_big.amax(dim=(-2,-1)) / (PSF_data*core_mask).amax(dim=(-2,-1))
    PSF_data *= peaks_scaler[:, :, None, None] # a sort of normalizing fetch-factor
    
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_data*core_mask, func_dxdy(dxdy_)*core_mask, reduction='sum')*1e3
    result = minimize(loss, dxdy_0, max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    # Update managers internal values with the new fitted values
    dxdy_1 = individual_inputs.unstack(result.x.unsqueeze(0), update=False)
    individual_inputs['dx'][i_src,:] = dxdy_1['dx'].flatten()
    individual_inputs['dy'][i_src,:] = dxdy_1['dy'].flatten()
    
    return func_dxdy(result.x).detach().clone().squeeze(), result.x.detach().clone()


PSFs_fitted = []

# The PSF model generates PSFs normalized to sum of 1 per wavelength. We need to normalize them to the flux of the sources.
# To do so, we need to account for: 1. the flux normalization factor, 2. the crop ratio, 3. the core to wings flux ratio.
for i in tqdm(range(N_src)):
    PSF_fitted, dxdy = fit_dxdy(i, verbose=0)
    PSFs_fitted.append(PSF_fitted * (src_spectra_sparse[i] * shared_inputs['PSF_norm_factor'])[:,None,None])
    
PSFs_fitted = torch.stack(PSFs_fitted, dim=0)


#%% Initial guess, will be inaccurate
from managers.multisrc_manager import VisualizeSources, PlotSourcesProfiles

model_sparse = add_ROIs(
    torch.zeros([N_wvl, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device), # blanck baase image
    [PSFs_fitted[i,...] for i in range(N_src)], # predicted flux-normalized PSFs after coordinates tuning
    srcs_image_data["img_crops"],
    srcs_image_data["img_slices"]
)

ROI_plot = np.s_[..., 125:225, 125:225]
norm_field = LogNorm(vmin=1, vmax=cube_sparse.sum(dim=0).max()) # again, rather empirical values

VisualizeSources(cube_sparse, model_sparse, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_sparse, model_sparse, sources, radius=16, title='Initiall guess')

#%% Manage PSF model input params 
from tools.static_phase import PixelmapBasis

LO_basis = PixelmapBasis(model, ignore_pupil=False)

# inputs_manager.set_optimizable('LO_coefs', False)
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

empty_img   = torch.zeros([N_wvl, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device)
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) + 1 #TODO: fix it

def func_fit(x): # TODO: relative weights for different brigtness
    PSFs_fit = []
    for i in range(N_src):
        params_dict = shared_inputs.unstack(x[:x_size].unsqueeze(0))
        F_dxdy_dict = individual_inputs.unstack(x[x_size:].view(N_src, -1))

        phase_func = lambda: LO_basis(shared_inputs["LO_coefs"].view(1, LO_map_size, LO_map_size))

        F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | F_dxdy_dict
        flux_norm = (src_spectra_sparse[i] * PSF_norm_factor)[:,None,None] * F_dxdy_dict['F_norm'][i]

        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
        
    return add_ROIs( empty_img*0.0, PSFs_fit, srcs_image_data["img_crops"], srcs_image_data["img_slices"] )


def loss(x_, data, func):
    model_ = func(x_)
    l1 = F.smooth_l1_loss(data*wvl_weights, model_*wvl_weights, reduction='mean')
    l2 = F.mse_loss      (data*wvl_weights, model_*wvl_weights, reduction='mean')

    # Enforce positivity of the residual
    residual = data - model_
    negative_residual_penalty = F.relu(-residual).mean()
    
    # Enforce Jx/Jy ratio being close to 1
    J_ratio_penalty = (1.0 - shared_inputs['Jx']/shared_inputs['Jy']).abs().mean()
    
    # Combine the loss terms
    # return l1 * 1e-3 + l2 * 5e-6 + negative_residual_penalty * 1e-2 * J_ratio_penalty * 0.1
    return l1*1.5 + l2*0.25 + negative_residual_penalty*2 * J_ratio_penalty*0.05


_ = func_fit(x0)

result_global = minimize(lambda x: loss(x, cube_sparse, func_fit), x0, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x0 = result_global.x.clone()
x_fit_dict = shared_inputs.unstack(x0.unsqueeze(0), include_all=False)

#%%
with torch.no_grad():
    model_fit = func_fit(result_global.x).detach()

VisualizeSources(cube_sparse, model_fit, norm=norm_field, mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_sparse, model_fit, sources, radius=16, title='Fitted PSFs')

#%% Compute Strehl ratio
def func_just_for_Strehl(x): # TODO: relative weights for different brigtness
    PSFs_fit = []
    for i in range(N_src):
        params_dict = shared_inputs.unstack(x[:x_size].unsqueeze(0))
        F_dxdy_dict = individual_inputs.unstack(x[x_size:].view(N_src, -1))

        phase_func = lambda: LO_basis(shared_inputs["LO_coefs"].view(1, LO_map_size, LO_map_size))

        F_dxdy_dict['dx'] = F_dxdy_dict['dx'][i].unsqueeze(-1).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = F_dxdy_dict['dy'][i].unsqueeze(-1).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | F_dxdy_dict

        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() )
    return PSFs_fit

with torch.no_grad():
    PSFs_pred = func_just_for_Strehl(result_global.x)
    PSF_DL = model.DLPSF().squeeze()
    Strehls_per_λ = PSFs_pred[0].amax(dim=(-2,-1)) / PSF_DL.amax(dim=(-2,-1))

plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(λ_sparse.cpu(), 100* Strehls_per_λ.cpu())
plt.ylabel('Strehl ratio [%]')
plt.xlabel('Wavelength [nm]')
plt.grid()
plt.show()

#%% Interpolate PSF model parameters over the full wavelengths range assuming they change smooth
from tools.misc import QuadraticModel

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

curve_inputs = InputsManager()

curve_inputs.add('Jx_A', torch.tensor([params_Jx[0]]), Uniform(a=-5e-5, b=5e-5))
curve_inputs.add('Jx_B', torch.tensor([params_Jx[1]]), Uniform(a=-2e-2, b=2e-2))
curve_inputs.add('Jx_C', torch.tensor([params_Jx[2]]), Uniform(a=30,    b=60))

curve_inputs.add('Jy_A', torch.tensor([params_Jy[0]]), Uniform(a=-5e-5, b=5e-5))
curve_inputs.add('Jy_B', torch.tensor([params_Jy[1]]), Uniform(a=-5e-2, b=5e-2))
curve_inputs.add('Jy_C', torch.tensor([params_Jy[2]]), Uniform(a=30,    b=60))

curve_inputs.add('F_A',  torch.tensor([params_F[0]]),  Uniform(a=5e-7,  b=10e-7))
curve_inputs.add('F_B',  torch.tensor([params_F[1]]),  Uniform(a=-2e-3, b=0))
curve_inputs.add('F_C',  torch.tensor([params_F[2]]),  Uniform(a=0,     b=2))

curve_inputs.add('norm_A', torch.tensor([params_norm[0]]), Uniform(a=0, b=2e-2))
curve_inputs.add('norm_B', torch.tensor([params_norm[1]]), Uniform(a=50, b=-20))
curve_inputs.add('norm_C', torch.tensor([params_norm[2]]), Uniform(a=2e4, b=4e4))

curve_inputs.set_optimizable(['norm_A', 'norm_B', 'norm_C'], False)

curve_inputs.to_float()
curve_inputs.to(device)


curve_params_ = curve_inputs.unstack(curve_inputs.stack())

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

if False:
    plt.figure(figsize=(10, 6))

    plt.plot(λ_sparse.cpu(), shared_inputs['Jx'].flatten().cpu(), label='Fitted Jx', color='tab:blue')
    plt.plot(λ_sparse.cpu(), Fx_curve_fit.cpu(), label='Interpolated quadratic curve', linestyle='--', color='tab:blue')
    plt.scatter(λ_sparse.cpu(), Jx_new.cpu(), label='Tuned quadratic curve', color='tab:blue', marker='x')

    plt.plot(λ_sparse.cpu(), shared_inputs['Jy'].flatten().cpu(), label='Fitted Jy', color='tab:orange')
    plt.plot(λ_sparse.cpu(), Fy_curve_fit.cpu(), label='Interpolated quadratic curve', linestyle='--', color='tab:orange')
    plt.scatter(λ_sparse.cpu(), Jy_new.cpu(), label='Tuned quadratic curve', color='tab:orange', marker='x')
    plt.xlabel('Wavelength [nm]')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.plot(λ_sparse.cpu(), inputs_manager['norm'].flatten().cpu(), label='Data', color='tab:orange')
    plt.plot(λ_sparse.cpu(), PSF_norm_factor.flatten().cpu(), label='Flux normalization factor', color='tab:orange')
    plt.plot(λ_sparse.cpu(), norm_curve_fit.cpu(), label='Interpolated quadratic curve', linestyle='--', color='tab:orange')
    plt.scatter(λ_sparse.cpu(), norm_new.cpu(), label='Tuned quadratic curve', color='tab:orange', marker='x')
    plt.xlabel('Wavelength [nm]')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(λ_sparse.cpu(), shared_inputs['F'].flatten().cpu(), label='PSF normalization factor', color='tab:green')
    plt.plot(λ_sparse.cpu(), F_curve_fit.cpu(), label='Interpolated quadratic curve', linestyle='--', color='tab:green')
    plt.scatter(λ_sparse.cpu(), F_new.cpu(), label='Tuned quadratic curve', color='tab:green', marker='x')
    plt.xlabel('Wavelength [nm]')
    plt.legend()
    plt.grid(True)
    plt.show()


#%% Do fine-adjustment of the parameters extrapolators by fitting their parameters
shared_inputs.set_optimizable(['F', 'Jx', 'Jy'],  False)
individual_inputs.set_optimizable(['dx', 'dy'], False)

print(shared_inputs)

x_size_model = shared_inputs.get_stacked_size()
x_size_curve = curve_inputs.get_stacked_size()
x_size_total = x_size_curve + x_size_model

x2 = torch.cat([
    shared_inputs.stack().flatten(),
    curve_inputs.stack().flatten(),
    individual_inputs.stack().flatten(),
])


def func_fit_curve(x):
    PSFs_fit = []
    for i in range(N_src):
        params_dict = shared_inputs.unstack(x[:x_size_model].unsqueeze(0))
        curve_p_    = curve_inputs.unstack(x[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
        F_dxdy_dict = individual_inputs.unstack(x[x_size_curve+x_size_model:].view(N_src, -1))
        
        curve_dict = {p: curve_sample(λ_sparse, curve_p_, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
        
        phase_func = lambda: LO_basis(shared_inputs["LO_coefs"].view(1, LO_map_size, LO_map_size))
        
        F_dxdy_dict['dx'] = individual_inputs['dx'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # Extend to simulated number of wavelength
        F_dxdy_dict['dy'] = individual_inputs['dy'][i].unsqueeze(-1).repeat(N_wvl).unsqueeze(0) # assuming the same shift for all wavelengths

        inputs = params_dict | curve_dict | F_dxdy_dict
        flux_norm = (src_spectra_sparse[i] * norm_new)[:,None,None] * F_dxdy_dict['F_norm'][i]
        
        PSFs_fit.append( model(inputs, phase_generator=phase_func).squeeze() * flux_norm )
    
    return add_ROIs( empty_img*0.0, PSFs_fit, srcs_image_data["img_crops"], srcs_image_data["img_slices"] )


# _ = loss_fit_curve(x2)

result_global = minimize(lambda x: loss(x, cube_sparse, func_fit_curve), x2, max_iter=300, tol=1e-3, method='bfgs', disp=2)
x2 = result_global.x.clone().detach()

x_fit_dict = shared_inputs.unstack(x2[:x_size_model].unsqueeze(0))
x_curve_fit_dict = curve_inputs.unstack(x2[x_size_model:x_size_curve+x_size_model].unsqueeze(0))
flux_corrections = individual_inputs['F_norm']

#%% Predict PSFs over the full wavelengths range
from managers.multisrc_manager import add_ROIs_separately

print('Extending the prediction over the whole wavelengths range...')
torch.cuda.empty_cache()

model_inputs_full_λ = {p: curve_sample(torch.as_tensor(λ_full, device=device), x_curve_fit_dict, p).unsqueeze(0) for p in ['Jx', 'Jy', 'F']}
norms_new_full_λ = curve_sample(torch.as_tensor(λ_full, device=device), x_curve_fit_dict, 'norm')

# Split λ array into batches
λ_split_size = 100
λ_batches = [λ_full[i:i + λ_split_size] for i in range(0, len(λ_full), λ_split_size)]

model_full = []
model_full_split = []

wvl_temp = model_config['sources_science']['Wavelength'].clone()

with torch.no_grad():
    for batch_id in tqdm(range(len(λ_batches))):
        batch_size = len(λ_batches[batch_id])
        model_config['sources_science']['Wavelength'] = torch.as_tensor(λ_batches[batch_id]*1e-9, device=device).unsqueeze(0)
        model.Update(init_grids=True, init_pupils=True, init_tomography=True)

        empty_img = torch.zeros([batch_size, cube_sparse.shape[-2], cube_sparse.shape[-1]], device=device)
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
                * torch.as_tensor(src_spectra_full[i][batch_ids], device=device)

            PSF_batch.append( (model(dict_selected).squeeze() * flux_norm[:,None,None]).detach() )

        model_full.append( add_ROIs( empty_img*0.0, PSF_batch, srcs_image_data["img_crops"], srcs_image_data["img_slices"] ).cpu().numpy() )
        model_full_split.append( add_ROIs_separately( empty_img*0.0, PSF_batch, srcs_image_data["img_crops"], srcs_image_data["img_slices"] ).cpu().numpy() )
        
model_config['sources_science']['Wavelength'] = torch.as_tensor(λ_batches[batch_id]*1e-9, device=device).unsqueeze(0)
model.Update(init_grids=True, init_pupils=True, init_tomography=True)

model_full = np.vstack(model_full)
model_full_split = np.concatenate(model_full_split, axis=1)
diff_img_full = (cube_full - model_full) * valid_mask.cpu().numpy()
torch.cuda.empty_cache()


# VisualizeSources(cube_full, model_full, norm=LogNorm(vmin=1e1, vmax=25000*10), mask=valid_mask, ROI=ROI_plot)
PlotSourcesProfiles(cube_full, model_full, sources, radius=16, title='Fitted PSFs')

#%% ============== Plotting the residual spectrum ===================
from astropy.convolution import convolve, Box1DKernel

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

for info in targets_diff_info:
    # Note, that boxes position is shifted according to plotting ROI
    aperture = RectangularAperture([info['coords'][0]-ROI_plot[1].start, info['coords'][1]-ROI_plot[2].start], info['radius']*2+1, info['radius']*2+1, theta=0)
    aperture.plot(color=info['color'], lw=1, label=info['name'])
plt.legend()
plt.show()

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
