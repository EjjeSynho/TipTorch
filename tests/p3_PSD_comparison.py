#%%
%reload_ext autoreload
%autoreload 2

import sys
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

sys.path.append('..')

from project_settings import DATA_FOLDER, PROJECT_PATH, DATA_FOLDER

# import TIPTOP dependencies
for module in ['MASTSEL', 'P3', 'SEEING', 'SYMAO', 'TIPTOP']:
    sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop') / f'{module}'))
    sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop') / f'{module}/{module.lower()}'))

sys.path.append(str(TIPTOP_PATH:=(PROJECT_PATH / '../astro-tiptop')))
from aoSystem.fourierModel import fourierModel

# Verify paths
# for i, path in enumerate(sys.path):
    # print(f"{i}: {path}")

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
tiptop_HO_model = fourierModel(
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

# Remove temp config
os.remove(temp_path_ini)

#%%
# Compute different P3 PSD contributors and pack into dictionary
P3_PSDs = {
    'fitting':         tiptop_HO_model.fittingPSD().squeeze().copy(),
    'aliasing':        tiptop_HO_model.aliasingPSD().squeeze().copy(),
    'diff. refract':   tiptop_HO_model.differentialRefractionPSD().squeeze().copy(),
    'chromatism':      tiptop_HO_model.chromatismPSD().squeeze().copy(),
    'spatio-temporal': tiptop_HO_model.spatioTemporalPSD().squeeze().copy(),
    'WFS noise':       tiptop_HO_model.noisePSD().squeeze().copy()
}

#%%
# %reload_ext autoreload
# %autoreload 2

from PSF_models.TipTorch import TipTorch
from managers.config_manager import ConfigManager
from project_settings import device, DATA_FOLDER, default_torch_type

#%%
# path_ini = DATA_FOLDER / "parameter_files/muse_ltao.ini"
config_manager = ConfigManager()
config_dict = config_manager.Load(path_ini)
config_dict = config_manager.Convert(config_dict, framework='pytorch', device=device, dtype=default_torch_type)

# Initialize TipTorch model
tiptorch_model = TipTorch(
    AO_config=config_dict,
    AO_type='LTAO',
    norm_regime='sum',
    device=device,
    oversampling=1
)

# Update oversampling to match P3 model exactly
match_sampling = (tiptop_HO_model.freq.k_.min().get() / tiptorch_model.sampling_factor.item()) * 1.001
tiptorch_model.oversampling = match_sampling
tiptorch_model.Update(init_grids=True, init_pupils=True, init_tomography=True)

PSF_1 = tiptorch_model()

#%%
def PSD_preprocess(psd_data):
    """ Convert both PSDs to numpy arrays and get real part. Handles half PSD for TipTorch, too. """
    if hasattr(psd_data, 'cpu'):
        PSD_buf = tiptorch_model.half_PSD_to_full(psd_data).squeeze().cpu().numpy().real
        PSD_buf[PSD_buf.shape[0]//2, PSD_buf.shape[1]//2] = 0.0
        return PSD_buf[:-1,:-1]
    elif hasattr(psd_data, 'get'): # Convert from Cupy to Numpy
        return psd_data.squeeze().get().real
    else:
        return np.array(psd_data).squeeze().real # Numpy array already

TipTorch_PSDs = {}
for key in P3_PSDs.keys():
    TipTorch_PSDs[key] = PSD_preprocess(tiptorch_model.PSDs[key].clone())
    P3_PSDs[key] = PSD_preprocess(P3_PSDs[key])

# print(P3_PSDs['fitting'].shape)
# print(TipTorch_PSDs['fitting'].shape)

#%%
def display_PSD(im, title=None, cmap='viridis', show_axis=True, fontsize=12, ax=None, vmin=None, vmax=None):
    """ Display a PSD with log normalization. Returns axis object and image for colorbar creation. """
    
    # Use provided vmin/vmax or calculate percentiles for log normalization
    if vmin is None or vmax is None:
        vmin = np.percentile(im[im > 0], 1e1)    if np.any(im > 0) else 1
        vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()
    
    im_plot = ax.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    
    if not show_axis:
        ax.axis('off')
    else:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    if title: ax.set_title(title, fontsize=fontsize)
    
    return ax, im_plot


# Iterate over all PSD contributors and create 1x2 subplots for each
for key in P3_PSDs.keys():
    if key in TipTorch_PSDs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
        torch_data = TipTorch_PSDs[key]
        p3_data = P3_PSDs[key]
        
        # Calculate common vmin/vmax from both datasets
        combined_positive = np.concatenate([p3_data[p3_data > 0].flatten(), torch_data[torch_data > 0].flatten()])
        vmin = np.percentile(combined_positive, 10)
        vmax = np.percentile(combined_positive, 99.975)
        
        # Plot P3 PSD on the left
        ax1, im1 = display_PSD(P3_PSDs[key], title=f'P3 {key} PSD', ax=ax1, vmin=vmin, vmax=vmax)
        # Plot TipTorch PSD on the right
        ax2, im2 = display_PSD(TipTorch_PSDs[key], title=f'TipTorch {key} PSD', ax=ax2, vmin=vmin, vmax=vmax)
        # Add shared colorbar - thicker and on the right
        cbar = fig.colorbar(im2, ax=[ax1, ax2], label='PSD Value', fraction=0.03, pad=0.02, aspect=15, shrink=1.8)
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)
        plt.show()

#%%
# Calculate and plot normalized difference maps (as percentages)
for key in P3_PSDs.keys():
    if key in TipTorch_PSDs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        torch_data = TipTorch_PSDs[key]
        p3_data = P3_PSDs[key]
        
        # Calculate percentage difference: (TipTorch - P3) / P3 * 100
        # Avoid division by zero by adding small epsilon where P3 is zero
        epsilon = 1e-15
        diff_normalized = ((torch_data - p3_data) / (p3_data + epsilon)) * 100
        
        # Set reasonable limits for the difference plot
        vmin = np.percentile(diff_normalized, 2.5)
        vmax = np.percentile(diff_normalized, 97.5)
        
        # Plot the normalized difference
        im = ax.imshow(diff_normalized, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'Normalized Difference: {key} PSD\n(TipTorch - P3) / P3 x 100%', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label='Percentage Difference (%)', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.show()

#%%
diffs = {}

# Print error statistics
for key in P3_PSDs.keys():
    if key in TipTorch_PSDs:
        torch_data = TipTorch_PSDs[key]
        p3_data    = P3_PSDs[key]
        
        epsilon = 1e-15 
        diff_normalized = ((torch_data - p3_data) / (p3_data + epsilon)) * 100 # [%]
        diffs[key] = diff_normalized.copy()
        
        mean_diff = np.mean(diff_normalized)
        std_diff  = np.std(diff_normalized)
        max_diff  = np.max(np.abs(diff_normalized))
        
        print(f"{key}: Mean: {mean_diff:.2f}%, STD: {std_diff:.2f}%, Max: {max_diff:.2f}%")

# %%
from photutils.profiles import RadialProfile

def plot_PSD_radial_profile(img, title, linestyle='-', color=None):
    xycen = (img.shape[-1]//2, img.shape[-2]//2)
    edge_radii = np.arange(img.shape[-1]//2)
    rp = RadialProfile(img, xycen, edge_radii)

    spatial_freq = rp.radius * tiptorch_model.dk.cpu().numpy()
    plt.plot(spatial_freq, rp.profile, linestyle=linestyle, color=color, label=title, linewidth=1)

    # plt.ylabel(rf'PSD [nm$^2$/(1/m)$^2$]')
    plt.xscale('symlog', linthresh=5e-2)

    plt.xlim(1e-1, 1e1)

    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Spatial frequency (1/m)')
    
    
plt.figure(figsize=(12,8))

plot_PSD_radial_profile(P3_PSDs['fitting'], 'P3 fitting', linestyle='--', color='C0')
plot_PSD_radial_profile(TipTorch_PSDs['fitting'], 'TipTorch fitting', linestyle='-', color='C0')

plot_PSD_radial_profile(P3_PSDs['WFS noise'], 'P3 WFS noise', linestyle='--', color='C1')
plot_PSD_radial_profile(TipTorch_PSDs['WFS noise'], 'TipTorch WFS noise', linestyle='-', color='C1')

plt.title('PSD Radial Profile Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# %%
