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

# %%
ob.DisplaySimulation(plot_profiles=True, full_spectrum=True, focus_on_src=0)  # Focus on the first source

#%%
ob.PlotSourceSpectra()

#%%
ob.SaveState(data_folder/'metadata'/(cache_path.stem+'.pkl'), save_full_cubes=False)

#%%
ob = MUSEObservation(raw_path, cube_path, cache_path, device=device)

ob.LoadState(data_folder/'metadata'/(cache_path.stem+'.pkl'), device=device)
_ = ob.SimulateField(full_spectrum=True)
ob.DisplaySimulation(plot_profiles=True, full_spectrum=True, focus_on_src=0)  # Focus on the first source


# %%
%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import LogNorm, Normalize
import torch

# Cut out the ROI around source 0 from the full-spectrum data/model cubes,
# same convention as DisplaySimulation(..., focus_on_src=...) uses internally.
src_idx = 0
(y0, y1), (x0, x1) = ob.sources.slices_global[src_idx]

pic_data  = ob.cube_full[:, y0:y1, x0:x1]
pic_model = ob.simulated_full[:, y0:y1, x0:x1]

# Crop to the very central 11x11 pixels of the ROI
crop_size = 11
cy, cx = pic_data.shape[-2] // 2, pic_data.shape[-1] // 2
half = crop_size // 2
cy0, cy1 = cy - half, cy - half + crop_size
cx0, cx1 = cx - half, cx - half + crop_size

pic_data  = pic_data [:, cy0:cy1, cx0:cx1]
pic_model = pic_model[:, cy0:cy1, cx0:cx1]

ii = ob.N_wvl_full // 2  # initial slice shown in the widget below


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
    raise ValueError("pic_data and pic_model must contain a stack of 2D slices.")

if data_cube.shape != model_cube.shape:
    raise ValueError(f"Shape mismatch: data={data_cube.shape}, model={model_cube.shape}")

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

