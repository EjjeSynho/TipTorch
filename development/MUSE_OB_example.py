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

from tiptorch._config import default_device, project_settings
import matplotlib.pyplot as plt
from tools.observations import MUSEObservation

from pathlib import Path

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
# cache_path = data_folder / "cached_cubes/CUBE_0001.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0002.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_11_59.783.fits.fz"
# cache_path = data_folder / "cached_cubes/CUBE_0002.pickle"

# cube_path  = data_folder / "reduced_cubes/CUBE_0003.fits"
# raw_path   = data_folder / "raw_data/MUSE.2023-04-27T05_28_41.014.fits.fz"
# cache_path = data_folder / "cached_cubes/CUBE_0003.pickle"

cube_path  = data_folder / "reduced_cubes/CUBE_0021.fits"
raw_path   = data_folder / "raw_data/MUSE.2023-06-17T00_04_47.319.fits.fz"
cache_path = data_folder / "cached_cubes/CUBE_0021.pickle"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

#%%
ob.DetectSources(nsigma=35, threshold='auto')
# ob.DetectSources(nsigma=35, threshold=4e2, verbose=True)
# ob.AddSources([[100, 200]], weights=0.0)
# NOTE: If some sources must be filtered out, do it here by modifying the ob.sources_table before initializing the sources for simulation

# ob.DeleteSources([1, 2])  # Example: delete sources with IDs 1 and 2

ob.DisplayField()

#%%
ob.InitSimulation()

#%%
ob.FitPSFModel(repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry'], repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry', 'photometry'], repeat=3, max_iter=200)

#%%
model = ob.SimulateField()

#%%
ob.DisplaySimulation(plot_profiles=True)

#%%
_ = ob.SimulateField(full_spectrum=True)

#%%
ob.DisplaySimulation(plot_profiles=True, plot_full_spectrum=True)

#%%
center_ = ob.field_center[0].tolist()

_, PSF_sparse, _ = ob.SimulatePSFAtPosition(*center_)
_, PSF_full, _   = ob.SimulatePSFAtPosition(*center_, full_spectrum=True)

#%%
import numpy as np
import torch

ids = []
# Before and after Na-filter region
region_1 = np.where(ob.λ_bins <  589)[0]
region_2 = np.where(ob.λ_bins >= 600)[0]

for l in ob.λ_bins:
    id_λ = np.argmin(np.abs(ob.λ_full - l)).item()
    ids.append(id_λ)
ids = np.array(ids)

ids1 = ids[region_1]
ids2 = ids[region_2]

#%%
PSF_full_binned = []
for i in range(len(ids1)-1):
    PSF_full_binned.append(PSF_full[ids1[i]:ids1[i+1]].mean(dim=0))

for i in range(len(ids2)-1):
    PSF_full_binned.append(PSF_full[ids2[i]:ids2[i+1]].mean(dim=0))

PSF_full_binned = torch.stack(PSF_full_binned, dim=0)[ob.ids_λ_sparse, ...].to(ob.device)

#%%
max_diffs = PSF_sparse.amax(dim=(-2, -1)) / PSF_full_binned.amax(dim=(-2, -1))

print(max_diffs)

#%%
from tools.plotting import plot_radial_PSF_profiles

# n_l = 6

plot_radial_PSF_profiles(
    PSF_0 = PSF_sparse, #[n_l, ...],
    PSF_1 = (PSF_full_binned * max_diffs.view(-1, 1, 1)), #[n_l, ...],
    label_0 = 'Full',
    label_1 = 'Sparse',
    cutoff=30,
)

#%%
ob.PlotSourceSpectra(title='Sources spectra (residual)', show_sparse=False, plot_residual=True, smooth_kernel=15)

#%%
simul_backup = ob.simulated_full.clone()
data_backup  = ob.cube_full.clone()
N_wvl_backup = ob.N_wvl_full

#%%
ob.simulated_full = simul_backup.clone()
ob.cube_full = data_backup.clone()
ob.N_wvl_full = N_wvl_backup

# %%
remove_region = [ids1[-1].item(), ids2[0].item()]  # Remove the Na-filter region

# Remove slices corresponding to the Na-filter region from the simulated and data cubes
ob.simulated_full = torch.cat([ob.simulated_full[:remove_region[0]], ob.simulated_full[remove_region[1]:]], dim=0)
ob.cube_full      = torch.cat([ob.cube_full[:remove_region[0]], ob.cube_full[remove_region[1]:]], dim=0)

ob.N_wvl_full = ob.simulated_full.shape[0]

# %%
ob.DisplaySimulation(plot_profiles=True, plot_full_spectrum=True, focus_on_src=0)  # Focus on the first source


#%%
from tools.multisources import PlotSourcesProfiles

ii = 0

data_  = ob.cube_full[ids[ii]:ids[ii+1], ...]
bg_ = ob.cube_full[ids[ii]:ids[ii+1], 50:150, 50:150].mean(dim=(-2, -1), keepdim=True)
data_ -= bg_

model_ = ob.simulated_full[ids[ii]:ids[ii+1], ...]

# PlotSourcesProfiles(data_, model_, ob.sources.table, radius=16, title=f'Source radial profiles')
        

#%%
import torch
import numpy as np
from tools.multisources import extract_ROIs
from tools.plotting import plot_radial_PSF_profiles

data_  = data_  = ob.cube_full
model_ = model_ = ob.simulated_full

radius = 16

box_size = np.round(radius * 2 + 4).astype(int)

ROIs_0, coords_0, _, _ = extract_ROIs(data_,  ob.sources.table, box_size=box_size)
ROIs_1, coords_1, _, _ = extract_ROIs(model_, ob.sources.table, box_size=box_size)

#%% Flag bad spectral slices (cosmic rays etc.) for all ROIs in batch
N_src = len(ROIs_0)

# Stack all ROIs into a batch: [N_src, N_wvl, H, W]
ROIs_batch = torch.stack(ROIs_0)

# Central pixel coordinates per source
cps = torch.tensor([
    [int(np.round(np.sum(c[0]))/2), int(np.round(np.sum(c[1]))/2)]
    for c in coords_0
])  # [N_src, 2]

# Extract peak values and surrounding ring for each source
peaks  = ROIs_batch[torch.arange(N_src), :, cps[:,0], cps[:,1]]  # [N_src, N_wvl]
donuts = torch.stack([
    ROIs_batch[i, :, cps[i,0]-1:cps[i,0]+2, cps[i,1]-1:cps[i,1]+2]
    for i in range(N_src)
])  # [N_src, N_wvl, 3, 3]
donuts[..., 1, 1] = 0.0
ring_mean = donuts.sum(dim=(-2, -1)) / 8.0  # [N_src, N_wvl]

# Compute ring-to-peak ratio
ratio = ring_mean.clamp(min=1e-12) / peaks
ratio = ratio.nan_to_num(nan=2.0, posinf=2.0, neginf=2.0)

# Center ratio by subtracting the baseline median (per source)
baseline = ratio[:, 1200:].median(dim=1, keepdim=True).values
ratio_centered = ratio - baseline

# Adaptive threshold per source
Ps = torch.arange(50, 100, 1)
quantiles = Ps.float() / 100.0

threshs = torch.stack([
    torch.quantile(ratio_centered[i, 1200:], quantiles) for i in range(N_src)
])  # [N_src, len(Ps)]
d_threshs = threshs.diff(dim=1).abs()
best_pct = (Ps[1:][d_threshs.argmax(dim=1)] - 0.5) / 100.0  # [N_src]

thresh_vals = torch.stack([
    torch.quantile(ratio_centered[i, 1200:], best_pct[i]) for i in range(N_src)
])  # [N_src]

# Flag slices exceeding threshold (only after slice 1200)
bad_mask = (ratio_centered > thresh_vals[:, None]) & (torch.arange(ratio.shape[1])[None, :] > 1200)

# Apply NaN mask to central pixels of flagged slices
ROIs_batch_fixed = ROIs_batch.clone()
for i in range(N_src):
    ROIs_batch_fixed[i, bad_mask[i], cps[i, 0], cps[i, 1]] = float('nan')

ROI_fixed = [ROIs_batch_fixed[i] for i in range(N_src)]

#%% Diagnostic plot for source 0
plt.plot(ratio_centered[0].cpu().numpy(), linewidth=0.8, label='Ring / peak ratio (centered)')
plt.axhline(thresh_vals[0].item(), color='r', linestyle='--', linewidth=0.8,
            label=f'{best_pct[0].item()*100:.0f}th pct threshold')
plt.xlabel('Spectral slice')
plt.ylabel('Ring / peak flux ratio')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Spectrally average the PSFs
avg_white = (lambda x: torch.stack(x).mean(dim=1) if isinstance(x[0], torch.Tensor) else np.mean(np.stack(x), axis=1))

# PSFs_0_white = avg_white(ROIs_0)
PSFs_0_white = avg_white(ROI_fixed)
PSFs_1_white = avg_white(ROIs_1)

fig = plt.figure(figsize=(6, 4), dpi=300)
ax = fig.gca()

plot_radial_PSF_profiles(
    PSF_0 = PSFs_0_white,
    PSF_1 = PSFs_1_white,
    label_0 = 'Data',
    label_1 = 'Prediction',
    cutoff=radius,
    ax=ax,
)
plt.show()

#%%
ii = 3075-5

pic_data  = ROIs_0[0]
pic_model = ROIs_1[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
    
ax1.imshow(pic_data[ii], cmap='gray')
ax1.set_title('Data')
ax1.axis('off')

ax2.imshow(pic_model[ii], cmap='gray')
ax2.set_title('Model')
ax2.axis('off')

plt.show()

# %%
%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import LogNorm, Normalize

try:
    import torch
except ImportError:
    torch = None


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().abs().cpu().numpy()
    return np.asarray(x)


data_cube  = _to_numpy(pic_data)
model_cube = _to_numpy(pic_model)

# Get wavelength information if available
try:
    wavelengths = _to_numpy(ob.λ_full)
    has_wavelengths = True
except:
    wavelengths = None
    has_wavelengths = False

if data_cube.ndim < 3 or model_cube.ndim < 3:
    raise ValueError(
        "pic_data and pic_model must contain a stack of 2D slices."
    )

if data_cube.shape != model_cube.shape:
    raise ValueError(
        f"Shape mismatch: data={data_cube.shape}, model={model_cube.shape}"
    )

n_slices = data_cube.shape[0]
if n_slices == 0:
    raise ValueError("The input cubes contain no slices.")

init_idx = int(np.clip(ii, 0, n_slices - 1))

# Compute global vmin for fixed normalization across all slices
all_data_finite = data_cube[np.isfinite(data_cube)]
all_model_finite = model_cube[np.isfinite(model_cube)]
all_finite = np.concatenate([all_data_finite, all_model_finite])

if all_finite.size > 0:
    global_vmin_linear = float(all_finite.min())
    positive = all_finite[all_finite > 0]
    global_vmin_log = max(float(positive.min()), 1e-12) if positive.size > 0 else 1e-12
else:
    global_vmin_linear = 0.0
    global_vmin_log = 1e-12


def _combined_slice_norm(data_slice, model_slice, log_scale):
    """Compute shared normalization for both data and model slices."""
    # Combine both slices to find common vmax
    combined = np.concatenate([
        data_slice[np.isfinite(data_slice)].flatten(),
        model_slice[np.isfinite(model_slice)].flatten()
    ])
    
    if combined.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    
    if log_scale:
        positive = combined[combined > 0]
        if positive.size == 0:
            return Normalize(vmin=global_vmin_linear, vmax=max(combined.max(), global_vmin_linear + 1))
        
        vmin = global_vmin_log
        vmax = float(positive.max())
        
        if vmax <= vmin:
            vmax = vmin * 10.0
        
        return LogNorm(vmin=vmin, vmax=vmax)
    
    vmin = global_vmin_linear
    vmax = float(combined.max())
    
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    return Normalize(vmin=vmin, vmax=vmax)


def _slice_errors(d, m):
    valid = np.isfinite(d) & np.isfinite(m)

    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan

    d_valid = d[valid]
    m_valid = m[valid]
    diff = np.abs(d_valid - m_valid)

    peak_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    peak_ref = float(np.max(np.abs(d_valid)))
    mean_ref = float(np.mean(np.abs(d_valid)))

    peak_rel = peak_err / peak_ref * 100.0 if peak_ref > 0 else np.nan
    mean_rel = mean_err / mean_ref * 100.0 if mean_ref > 0 else np.nan

    return peak_err, mean_err, peak_rel, mean_rel


def _format_percent(value):
    return f"{value:.2f}%" if np.isfinite(value) else "N/A"


# Prevent duplicate widget figures when rerunning the notebook cell.
plt.close("all")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), dpi=100, constrained_layout=False)

fig.subplots_adjust(bottom=0.25, right=0.80, top=0.86)

initial_data  = data_cube [init_idx]
initial_model = model_cube[init_idx]

initial_norm = _combined_slice_norm(initial_data, initial_model, log_scale=False)

im1 = ax1.imshow(initial_data, cmap="inferno", norm=initial_norm)
ax1.set_title("Data")
ax1.axis("off")

im2 = ax2.imshow(initial_model, cmap="inferno", norm=initial_norm)
ax2.set_title("Model")
ax2.axis("off")

err_text = fig.suptitle("", y=0.96, fontsize=10)

ax_slider = fig.add_axes([0.15, 0.10, 0.55, 0.03])
slice_slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=n_slices - 1, valinit=init_idx, valstep=1)

ax_radio = fig.add_axes([0.83, 0.38, 0.14, 0.16])
norm_radio = RadioButtons(ax_radio, ("linear", "log"), active=0)


def update(_=None):
    idx = int(round(slice_slider.val))
    log_scale = norm_radio.value_selected == "log"

    d = data_cube[idx]
    m = model_cube[idx]

    # Use shared normalization for both images with fixed vmin
    shared_norm = _combined_slice_norm(d, m, log_scale)

    im1.set_data(d)
    im1.set_norm(shared_norm)

    im2.set_data(m)
    im2.set_norm(shared_norm)

    peak_err, mean_err, peak_rel, mean_rel = _slice_errors(d, m)

    # Build title with wavelength info if available
    title_parts = [f"Slice {idx}/{n_slices - 1}"]
    
    if has_wavelengths and idx < len(wavelengths):
        title_parts.append(f"λ = {wavelengths[idx]:.2f} nm")
    
    title_parts.extend([
        f"Peak error: {peak_err:.3g} ({_format_percent(peak_rel)})",
        f"Mean error: {mean_err:.3g} ({_format_percent(mean_rel)})"
    ])
    
    err_text.set_text("  |  ".join(title_parts))

    fig.canvas.draw_idle()


def on_key_press(event):
    """Handle keyboard navigation with left/right arrow keys."""
    if event.key == 'left':
        new_val = max(0, slice_slider.val - 1)
        slice_slider.set_val(new_val)
    
    elif event.key == 'right':
        new_val = min(n_slices - 1, slice_slider.val + 1)
        slice_slider.set_val(new_val)


slice_slider.on_changed(update)
norm_radio.on_clicked(update)
fig.canvas.mpl_connect('key_press_event', on_key_press)

update()
plt.show()

# %%
# 2184 - big jump
# 2185 - fine

# 2192 - fine
# 2193-2195 - big jump

# 2202 - fine
# 2203 - big jump

# 2958 - fine
# 2959 - starts jumping
# 2960 - severe jump
# 2961 - fine

#3206 - missing core
#3207 - fine again

#3232 - missing core
#3233 - fine again

#3217 - missing core
#3318 - fine again

#3250 - missing core
#3351 - fine again
