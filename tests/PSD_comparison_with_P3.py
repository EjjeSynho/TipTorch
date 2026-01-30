#%%
%reload_ext autoreload
%autoreload 2

import sys
import os
import tempfile
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tools.utils import pdims

sys.path.append('..')

from project_settings import DATA_FOLDER, PROJECT_PATH, DATA_FOLDER

TIPTOP_PATH = PROJECT_PATH / '../astro-tiptop'

# import TIPTOP dependencies
for module in ['MASTSEL', 'P3', 'SEEING', 'SYMAO', 'TIPTOP']:
    sys.path.append(str(TIPTOP_PATH / f'{module}'))
    sys.path.append(str(TIPTOP_PATH / f'{module}/{module.lower()}'))

sys.path.append(str(TIPTOP_PATH))
from aoSystem.fourierModel import fourierModel

#%%
path_ini = str(DATA_FOLDER / "parameter_files/muse_ltao.ini")

# Create a temporary modified version of the ini file
with open(path_ini, 'r') as f:
    ini_content = f.read()

# Replace the calibration path to the one understandable by P3
modified_content = ini_content.replace('$PROJECT_PATH$/data/calibrations/', '/aoSystem/data/')
temp_dir = os.path.dirname(path_ini)
temp_fd, temp_path_ini = tempfile.mkstemp(suffix='.ini', dir=temp_dir)

with os.fdopen(temp_fd, 'w') as temp_file:
    temp_file.write(modified_content)

# Run P3 Fourier model
P3_model = fourierModel(
    temp_path_ini,
    path_root=None,
    calcPSF=False,
    verbose=False,
    display=False,
    getErrorBreakDown=False,
    getFWHM=False,
    getEncircledEnergy=False,
    getEnsquaredEnergy=False,
    displayContour=False
)

W_tomo_P3    = P3_model.tomographicReconstructor().get() # extract tomographic reconstructor
W_alpha_P3   = P3_model.Walpha.get()               # extract layer-to-alpha-direction projector
P_beta_DM_P3 = P3_model.PbetaDM[0].get()         # extract projection from DMs to directions of interest
# Remove temp config
os.remove(temp_path_ini)


#%%
# Compute different P3 PSD contributors and pack into dictionary
P3_PSDs = {
    'fitting':         P3_model.fittingPSD().squeeze().copy(),
    'aliasing':        P3_model.aliasingPSD().squeeze().copy(),
    'diff. refract':   P3_model.differentialRefractionPSD().squeeze().copy(),
    'chromatism':      P3_model.chromatismPSD().squeeze().copy(),
    'spatio-temporal': P3_model.spatioTemporalPSD().squeeze().copy(),
    'WFS noise':       P3_model.noisePSD().squeeze().copy()
}

# Manually compute spatio-temporal terms for verification
nK = P3_model.freq.resAO
nH = P3_model.ao.atm.nL
Hs = P3_model.ao.atm.heights * P3_model.strechFactor
deltaT = P3_model.ao.rtc.holoop['delay'] / P3_model.ao.rtc.holoop['rate']
wDir_x = np.cos(P3_model.ao.atm.wDir*np.pi/180)
wDir_y = np.sin(P3_model.ao.atm.wDir*np.pi/180)

freq_t = [0,] * nH

s = 0 # single source

Beta = [P3_model.ao.src.direction[0,s], P3_model.ao.src.direction[1,s]]

PbetaL = cp.zeros([nK, nK, 1, nH], dtype=complex)
fx = Beta[0] * P3_model.freq.kxAO_
fy = Beta[1] * P3_model.freq.kyAO_

for j in range(nH):
    freq_t[j] = wDir_x[j]*P3_model.freq.kxAO_+ wDir_y[j]*P3_model.freq.kyAO_
    delta_h = Hs[j]*(fx+fy) - deltaT * P3_model.ao.atm.wSpeed[j]*freq_t[j]
    PbetaL[: , :, 0, j] = cp.exp(1j*2*cp.pi*delta_h)

proj = PbetaL - np.matmul(P3_model.PbetaDM[s], P3_model.Walpha)
proj_t = np.conj(proj.transpose(0, 1, 3, 2))
tmp = np.matmul(proj,np.matmul(P3_model.Cphi, proj_t))

P_beta_L_P3 = PbetaL.get()
freq_t_P3 = cp.stack(freq_t, axis=2).get()

#%%
from PSF_models.TipTorch import TipTorch
from managers.config_manager import ConfigManager
from project_settings import device, DATA_FOLDER, default_torch_type
import torch

#%%
config_manager = ConfigManager()
config_dict = config_manager.Load(path_ini)
config_dict = config_manager.Convert(config_dict, framework='pytorch', device=device, dtype=default_torch_type)

# Initialize TipTorch model
tiptorch_model = TipTorch(
    AO_config=config_dict,
    AO_type='LTAO',
    norm_regime=None,
    device=device,
    oversampling=1
)

# Update oversampling to match P3 model exactly
match_sampling = (P3_model.freq.k_.min().get() / tiptorch_model.sampling_factor.cpu().numpy().max()) * 1.001
tiptorch_model.oversampling = match_sampling
tiptorch_model.Update(grids=True, pupils=True, tomography=True)

wvl_src = tiptorch_model.wvl.item()
wvl_atm = tiptorch_model.wvl_atm.item()
wvl_GS  = tiptorch_model.GS_wvl.item()

norm_factor = (wvl_src / wvl_atm)**2 # Scaling factor for PSDs due to the wavelength difference

PSF_1 = tiptorch_model()


#%%
# tiptorch_model.IOR_src_wvl = n_air_P3(tiptorch_model.wvl)
# tiptorch_model.IOR_wvl_atm = n_air_P3(tiptorch_model.wvl_atm)
# tiptorch_model.IOR_GS_wvl  = n_air_P3(tiptorch_model.GS_wvl) # GS_wvl may depend on the filter for SCAO or on LGS wavelength

# Restore from half to full size
def half_to_full_reconstructor(W_half):
    """Convert half reconstructor matrix to full size by mirroring."""
    W_half = W_half[0,...].detach().clone() # Remove batch dim and clone to avoid in-place ops
    return torch.cat([W_half, torch.flip(W_half[:,:-1,...], dims=(0,1))], dim=1).cpu().numpy()[:-1,:-1,...]

AO_mask = tiptorch_model.mask_corrected_AO.unsqueeze(-1).unsqueeze(-1)

W_tomo_torch    = half_to_full_reconstructor(tiptorch_model.W_tomo)
W_alpha_torch   = half_to_full_reconstructor(tiptorch_model.W_alpha * AO_mask)
P_beta_DM_torch = half_to_full_reconstructor(tiptorch_model.P_beta_DM * AO_mask)
P_beta_L_torch  = half_to_full_reconstructor(tiptorch_model.P_beta_L * AO_mask)
freq_t_torch    = half_to_full_reconstructor(tiptorch_model.freq_t / tiptorch_model.wind_speed.view(1,1,1,-1))

#%
# n2 =  23.7+6839.4/(130-(GS_wvl*1.e6)**(-2))+45.47/(38.9-(GS_wvl*1.e6)**(-2))
# n1 =  23.7+6839.4/(130-(wvl_ref*1.e6)**(-2))+45.47/(38.9-(wvl_ref*1.e6)**(-2))

def refractionIndex(wvl):
    c = [64.328, 29498.1, 146.0, 255.4, 41.0]
    wvlRef = wvl * 1e6  # Convert from [m] to [um]
    return 1e-6 * (c[0] +  c[1]/(c[2]-1.0/wvlRef**2) + c[3]/(c[4] - 1.0/wvlRef**2) )

#%%
C = 1 / norm_factor

# Make PSDs displayable and comparable for both models
def PSD_preprocess(psd_data):
    """ Convert both PSDs to numpy arrays and get real part. Handles half PSD for TipTorch, too. """
    if hasattr(psd_data, 'cpu'):
        PSD_buf = tiptorch_model.half_PSD_to_full(psd_data).squeeze().cpu().numpy().real
        PSD_buf[PSD_buf.shape[0]//2, PSD_buf.shape[1]//2] = 0.0
        return PSD_buf[:-1,:-1] * C
    
    elif hasattr(psd_data, 'get'): # Convert from Cupy to Numpy
        return psd_data.squeeze().get().real
    else:
        return np.array(psd_data).squeeze().real # Numpy array already

TipTorch_PSDs = {}
for key in P3_PSDs.keys():
    TipTorch_PSDs[key] = PSD_preprocess(tiptorch_model.PSDs[key].clone())
    
    P3_PSDs[key] = PSD_preprocess(P3_PSDs[key])

#%%
noise_variance_P3 = P3_model.ao.wfs.computeNoiseVarianceAtWavelength(
    wvl_science=wvl_src,
    wvl_wfs=wvl_GS,
    r0_at_500nm=tiptorch_model.r0_().item(), 
)[0] * (wvl_src / wvl_atm)**2 # scaled to atmosphere wavelength

noise_variance_tiptorch = tiptorch_model.NoiseVariance()[0,0].item() # at atmosphere wavelength by default

print(f"Noise var. - P3: {noise_variance_P3:.4f}, TipTorch: {noise_variance_tiptorch:.4f}, Diff.: {noise_variance_P3 - noise_variance_tiptorch:.4f}")
# print(np.nanmean(P3_model.Cb.get() / half_to_full_reconstructor(tiptorch_model.C_b) - C).real)


#%%
PSD_noise_P3 = cp.zeros((P3_model.freq.resAO, P3_model.freq.resAO, P3_model.ao.src.nSrc), dtype=complex)
# noise level is considered in the covariance matrix Cb
# and the noise gain is considered as follows (0.6 - 1.0)
noise_gain = min(0.8, 0.4 + 0.1333 * P3_model.ao.rtc.holoop['delay'])**2

for j in range(P3_model.ao.src.nSrc):
    PW = cp.matmul(P3_model.PbetaDM[j], P3_model.W)
    PW_t = cp.conj(PW.transpose(0,1,3,2))
    tmp  = cp.matmul( PW, cp.matmul(P3_model.Cb, PW_t) )
    PSD_noise_P3[:,:,j] = P3_model.freq.mskInAO_ * tmp[:, :, 0, 0] * P3_model.freq.pistonFilterAO_ * noise_gain

PSD_noise_P3 = PSD_noise_P3.squeeze().real.get()

#%%
noise_gain_torch = min(0.8, 0.4 + 0.1333 * tiptorch_model.HOloop_delay.item())**2
PW_torch = torch.matmul(tiptorch_model.P_beta_DM, tiptorch_model.W)
PW_t_torch = torch.conj(PW_torch.transpose(-2, -1))
tmp_torch = torch.matmul(PW_torch, torch.matmul(tiptorch_model.C_b, PW_t_torch))

PSD_noise_tiptorch = pdims(tiptorch_model.mask_corrected_AO * tiptorch_model.piston_filter, 2) * tmp_torch * noise_gain_torch
PSD_noise_tiptorch = half_to_full_reconstructor(PSD_noise_tiptorch).squeeze().real


#%%

key = 'WFS noise'

WFS_P3 = P3_PSDs[key]
WFS_TipTorch = TipTorch_PSDs[key]

rel_diff = WFS_TipTorch / WFS_P3

med_rel_diff = np.nanmedian(rel_diff)


#%%
def display_map(im, title='', cmap='viridis', show_axis=True, fontsize=12, ax=None, vmin=None, vmax=None, scale='log', colorbar=True):
    """ Display a PSD with log or linear normalization. Returns axis object and image for colorbar creation. """
    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))
    
    # Use provided vmin/vmax or calculate percentiles for normalization
    if vmin is None or vmax is None:
        if scale == 'log':
            vmin = np.percentile(im[im > 0], 1e1)    if np.any(im > 0) else 1
            vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()
        else:  # linear
            vmin = np.percentile(im, 1)
            vmax = np.percentile(im, 99)
    
    if scale == 'log':
        im_plot = ax.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    elif scale == 'linear':
        im_plot = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Scale must be either 'log' or 'linear'.")
    
    ax.tick_params(axis='both', which='major', labelsize=fontsize) if not show_axis else ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=fontsize)
    
    if colorbar:
        plt.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
    
    return ax, im_plot


def plot_side_by_side(torch_data, P3_data, title, scale='log'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
    # Calculate common vmin/vmax from both images
    combined_positive = np.concatenate([P3_data[P3_data > 0].flatten(), torch_data[torch_data > 0].flatten()])
    vmin = np.percentile(combined_positive, 10)
    vmax = np.percentile(combined_positive, 99.975)    

    ax1, im1 = display_map(torch_data, title=f'TipTorch {title}', ax=ax1, vmin=vmin, vmax=vmax, scale=scale, colorbar=False)
    ax2, im2 = display_map(P3_data, title=f'P3 {title}', ax=ax2, vmin=vmin, vmax=vmax, scale=scale, colorbar=False)
        
    # Add shared colorbar - thicker and on the right
    cbar = fig.colorbar(im2, ax=[ax1, ax2], label='Value', fraction=0.03, pad=0.02, aspect=15, shrink=1.8)
    cbar.ax.tick_params(labelsize=12)
        
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.show()


def plot_difference_map(torch_data, P3_data, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Calculate percentage difference: (TipTorch - P3) / P3 * 100
    # Avoid division by zero by adding small epsilon
    diff_normalized = ((torch_data - P3_data) / (P3_data + 1e-15)) * 100
    
    # Set reasonable limits for the difference plot
    vmin = np.percentile(diff_normalized, 2.5)
    vmax = np.percentile(diff_normalized, 97.5)
    
    # Plot the normalized difference
    im = ax.imshow(diff_normalized, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f'Normalized Difference: {title}\n(TipTorch - P3) / P3 x 100%', fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Percentage Difference (%)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()
    

def compute_difference_stats(torch_data, P3_data, title='', verbose=True):
    diff_normalized = ((torch_data - P3_data) / (P3_data + 1e-15)) * 100  # [%]
    diff_stats = {
        'diff_array': diff_normalized,
        'median': np.median(diff_normalized),
        'mean': np.mean(diff_normalized),
        'std':  np.std(diff_normalized),
        'max':  np.max(np.abs(diff_normalized))
    }
    if verbose:
        # print(f"{title}: Mean: {diff_stats['mean']:.1f}%, Median: {diff_stats['median']:.1f}%, STD: {diff_stats['std']:.1f}%, Max: {diff_stats['max']:.1f}%")
        print(f"{title}: Median: {diff_stats['median']:.1f}%, Max: {diff_stats['max']:.1f}%")
        
    return diff_stats 


#%%
# Iterate over all PSD contributors and create 1 x 2 subplots for each
for key in P3_PSDs.keys():
    if key in TipTorch_PSDs:     
        plot_side_by_side(TipTorch_PSDs[key], P3_PSDs[key], title=key+' PSD')

# Calculate and plot normalized difference maps (as percentages)
for key in P3_PSDs.keys():
    if key in TipTorch_PSDs:
        plot_difference_map(TipTorch_PSDs[key], P3_PSDs[key], title=key+' PSD')

# Print error statistics
for key in TipTorch_PSDs.keys():
    diff_stats = compute_difference_stats(TipTorch_PSDs[key], P3_PSDs[key], title=key+' PSD')

# %%
from photutils.profiles import RadialProfile

# choice = ['fitting', 'WFS noise', 'spatio-temporal', 'aliasing', 'diff. refract', 'chromatism']
choice = ['fitting', 'WFS noise', 'chromatism']

def plot_PSD_radial_profile(img, title, linestyle='-', color=None):
    xycen = (img.shape[-1]//2, img.shape[-2]//2)
    edge_radii = np.arange(img.shape[-1]//2)
    rp = RadialProfile(img, xycen, edge_radii)

    spatial_freq = rp.radius * tiptorch_model.dk.cpu().numpy()
    plt.plot(spatial_freq, rp.profile, linestyle=linestyle, color=color, label=title, linewidth=1)

    
plt.figure(figsize=(12,8))

for i, key in (enumerate(choice)):
    plot_PSD_radial_profile(P3_PSDs[key], f'P3 {key}', linestyle='--', color=f'C{i}')
    plot_PSD_radial_profile(TipTorch_PSDs[key], f'TipTorch {key}', linestyle='-', color=f'C{i}')


plt.xscale('symlog', linthresh=5e-2)
plt.yscale('symlog', linthresh=1e-3)

plt.xlim(1e-1, 1e1)

plt.grid(True, which='both', alpha=0.3)
plt.xlabel('Spatial frequency (1/m)')

plt.title('PSD radial profile comparison')
plt.ylabel(rf'PSD [rad$^2$/(1/m)$^2$]')

plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# %%
# Compare TipTorch and P3 tomographic reconstructors

i_layer = 0
j_GS = 0

W_tomo_torch_data = np.abs(W_tomo_torch)[..., i_layer, j_GS]
W_tomo_P3_data    = np.abs(W_tomo_P3)   [..., i_layer, j_GS]

W_alpha_torch_data = np.abs(W_alpha_torch)[..., 0, i_layer]
W_alpha_P3_data    = np.abs(W_alpha_P3)   [..., 0, i_layer]

P_beta_DM_torch_data = np.abs(P_beta_DM_torch)[..., 0, 0]
P_beta_DM_P3_data    = np.abs(P_beta_DM_P3)   [..., 0, 0]

P_beta_L_torch_data = np.abs(P_beta_L_torch)[..., 0, i_layer]
P_beta_L_P3_data    = np.abs(P_beta_L_P3)   [..., 0, i_layer]

freq_t_torch_data = np.abs(freq_t_torch)[..., i_layer]
freq_t_P3_data    = np.abs(freq_t_P3)   [..., i_layer]

# plot_side_by_side(W_tomo_torch_data, W_tomo_P3_data, title='W_tomo')
plot_difference_map(W_tomo_torch_data, W_tomo_P3_data, title='W_tomo')

# plot_side_by_side(W_alpha_torch_data, W_alpha_P3_data, title='W_alpha')
# plot_difference_map(W_alpha_torch_data, W_alpha_P3_data, title='W_alpha')

plot_difference_map(P_beta_DM_torch_data, P_beta_DM_P3_data, title='P_beta_DM')


#%%
_ = compute_difference_stats(W_tomo_torch_data, W_tomo_P3_data, title='W_tomo')
_ = compute_difference_stats(W_alpha_torch_data, W_alpha_P3_data, title='W_alpha')
_ = compute_difference_stats(P_beta_DM_torch_data, P_beta_DM_P3_data, title='P_beta_DM')
_ = compute_difference_stats(freq_t_torch_data, freq_t_P3_data, title='freq_t')
_ = compute_difference_stats(P_beta_L_torch_data, P_beta_L_P3_data, title='P_beta_L')


#%%
# display_map(freq_t_torch_data, scale='linear', colorbar=True)
plot_side_by_side(freq_t_torch_data, freq_t_P3_data, title='freq_t')

