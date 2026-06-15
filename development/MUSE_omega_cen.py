#%%
%reload_ext autoreload
%autoreload 2

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from tiptorch._config import default_device, project_settings
from tools.observations import MUSEObservation
from datasets.MUSE_STD_dataset.STD_dataset_utils import MatchRawWithCubes

data_folder = Path(project_settings["MUSE_data_folder"]) / "omega_cluster"

RAW_FOLDER   = data_folder / "raw_data"
CUBES_FOLDER = data_folder / "reduced_cubes"
CACHE_FOLDER = data_folder / "cached_cubes"

#%%
files_matches, _ = MatchRawWithCubes(RAW_FOLDER, CUBES_FOLDER, verbose=False)

reduced_cube = "DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
# reduced_cube = "ADP.2019-05-30T08-10-58.821.fits" # meh
# reduced_cube = "ADP.2021-05-19T05-26-01.345.fits" # meh
# reduced_cube = "ADP.2021-05-19T05-26-01.337.fits"
# reduced_cube = "ADP.2021-05-19T05-26-01.303.fits"

raw_file   = RAW_FOLDER   / files_matches.loc[files_matches['cube'] == reduced_cube]['raw'].values[0]
cube_path  = CUBES_FOLDER / reduced_cube
cache_path = CACHE_FOLDER / f"{cube_path.stem}.pickle"


ob = MUSEObservation(
    raw_file,
    cube_path,
    cache_path,
    PSF_size = 111,
    # model_type='Moffat', #'psfao', #'TipTorch',
    # model_type='psfao', #'TipTorch',
    model_type='TipTorch',
    device=default_device
)
ob.λ_batch_size = ob.λ_full.shape[0] // 3 + 1


if reduced_cube.replace('.fits', '').endswith('303'):
    bg_custom = np.array([-6.7931, -5.4455, -4.1980, -3.6123, -2.8540, -2.2158, -1.5991])
    ob.bg_prior = torch.tensor(bg_custom, device=ob.device, dtype=ob.cube_sparse.dtype)
    
elif reduced_cube.replace('.fits', '').endswith('345'):
    bg_custom = np.array([-8, -7, -5, -4.5, -4, -3.5, -5])
    ob.bg_prior = torch.tensor(bg_custom, device=ob.device, dtype=ob.cube_sparse.dtype)
    
elif reduced_cube.replace('.fits', '').endswith('337'):
    bg_custom = np.array([-8, -6, -4, -3.5, -3.5, -3, -5])
    bg_addend = np.array([2.3296, 1.8386, 1.1218, 1.0374, 1.3269, 1.2559, 3.5082]) * 0.9
    bg_custom += bg_addend
    ob.bg_prior = torch.tensor(bg_custom, device=ob.device, dtype=ob.cube_sparse.dtype)

# ob.cube_sparse -= torch.tensor(bg_custom, device=ob.device, dtype=ob.cube_sparse.dtype).view(-1, 1, 1)
# ob.bg_prior     = torch.zeros(ob.N_wvl, device=ob.device, dtype=ob.cube_sparse.dtype)

#%%
from tiptorch.tools.utils import seeing, rad2arc
from astropy.io import fits

with fits.open(cube_path) as hdul:
    try:
        OBID1 = hdul[0].header['OBID1']
    except KeyError:
        OBID1 = hdul[0].header['HIERARCH ESO OBS ID']
        
    EXPTIME = hdul[0].header['EXPTIME'] / 60.0

print(f"Metadata for {reduced_cube}:\n")

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]

seeings = np.array([seeing(ob.reduced_telemetry[e].item(), 500e-9) for e in ['r0Tot', 'R0'] if np.isfinite(ob.reduced_telemetry[e].item())]).mean()
seeings = (seeings + ob.reduced_telemetry['Seeing (header)'].item()) / 2.0

NGS_mag = np.array([ob.reduced_telemetry[e].item() for e in ['NGS mag (header)', 'NGS mag (from ph.)', 'NGS mag (from 2x2)']])
NGS_mag = NGS_mag[np.isfinite(NGS_mag)].mean()

col_w = max(len(e) for e in ['RA (science)', 'DEC (science)', 'Airmass'])

print(f"{'OB ID':<{col_w}}  {OBID1}")
print(f"{'Exposure time':<{col_w}}  {EXPTIME:.1f} min")

for e in ['RA (science)', 'DEC (science)', 'Airmass']:
    print(f"{e:<{col_w}}  {ob.reduced_telemetry[e].item():.4f}")

print(f"{'Tau0 (header)':<{col_w}}  {ob.reduced_telemetry['Tau0 (header)'].item()*1e3:1.1f}")
print(f"{'Seeing':<{col_w}}  {seeings:.2f} arcsec")
print(f"{'NGS mag':<{col_w}}  {NGS_mag:.2f}")

#%%
from tools.plotting import PlotSpetralCubeInRGB
from matplotlib.colors import LogNorm


img = ob.cube_sparse.cpu().numpy() - bg_custom.reshape(-1, 1, 1)

# display_norm = LogNorm(vmin=1, vmax=img.sum(axis=0).max().item()) # again, rather empirical values
# plt.imshow(img.sum(axis=0), norm=display_norm, origin='lower')


idx = 0

display_norm = LogNorm(vmin=1, vmax=img[idx, ...].max().item()) # again, rather empirical values
plt.imshow(img[idx, ...], norm=display_norm, origin='lower')

plt.axis('off')
plt.show()

# color_kwargs = {
#     'saturation':  2.0,
#     'contrast'  :  1.0,
#     'wb_shift'  : -0.1,
#     'mg_shift'  :  0.075,
#     'min_val'   :  1,
#     'max_val'   :  5e4,
# }

# λ_vis_sparse = np.linspace(440, 750, ob.N_wvl)

# _ = PlotSpetralCubeInRGB(
#     img,
#     wavelengths = λ_vis_sparse,
#     title = "Model (full spectrum)",
#     **color_kwargs
# )


#%%
# Read pre-processed HST data from DataFrame
sources_file = data_folder / f'metadata/HST_srcs_{cube_path.stem}.csv'

with open(sources_file, 'r') as f:
    sources_df = pd.read_csv(f)

sources_df.set_index('ID', inplace=True)
sources_df.dropna(inplace=True)
sources = sources_df[['x, [asec]', 'y, [asec]', 'flux (total, normalized)']].copy()
sources.rename(
    columns={
        'x, [asec]': 'x_peak',
        'y, [asec]': 'y_peak',
        'flux (total, normalized)': 'peak_value'
    },
    inplace=True
)
sources[['x_peak', 'y_peak']] = sources[['x_peak', 'y_peak']] * 1e3 / 25.0 + ob.field_center # [asec] -> [pix]

#%
# Find almost dublicates
# coords = sources[['x_peak', 'y_peak']].values
# from sklearn.neighbors import NearestNeighbors
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
# distances, indices = nbrs.kneighbors(coords)

#%
# Add weight column for later use (e.g. in loss function)
sources['weight'] = 1.0

# Leave the first N brighest sources
# sources = sources.nlargest(200, 'peak_value')

ob.sources_table = sources
ob.ExtractSources(verbose=True, max_nan_fraction=0.7)

# ob.DisplaySources(draw_box_size=5)

ob.InitSimulation()

#%%
ob.FitPSFModel(repeat=3, max_iter=200)
# ob.FitPSFModel(repeat=1, max_iter=500)

#%%
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

model_cache = data_folder / f"metadata/PSF_model_predicted_{ob.model_type}_{cube_path.stem}.pt"

if model_cache.exists():
    print("Loading PSF model from cache...")
    model_data = torch.load(model_cache)
    ob.PSF_model = PSFModelNFM.load(model_data, device=ob.device, config=ob.model_config)
else:
    print("Fitting PSF model...")
    ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)
    model_data = ob.PSF_model.save()#cpu=True)
    torch.save(model_data, model_cache)
    print(f"PSF model saved to {model_cache}")

#%%
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

model_cache = data_folder / f"metadata/PSF_model_fitted_{ob.model_type}_{cube_path.stem}.pt"

if model_cache.exists():
    print("Loading PSF model from cache...")
    model_data = torch.load(model_cache)
    ob.PSF_model = PSFModelNFM.load(model_data, device=ob.device, config=ob.model_config)
else:
    print("Fitting PSF model...")
    ob.FitPSFModel(repeat=3, max_iter=200)
    model_data = ob.PSF_model.save()
    torch.save(model_data, model_cache)
    print(f"PSF model saved to {model_cache}")

#%%
# field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=True, force_cpu=False)
ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)

#%%
field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=True, force_cpu=False)
# field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=False, force_cpu=False)
data_img = ob.cube_sparse.clone() - ob.background_sparse.view(ob.N_wvl, 1, 1)

#%%
residue_sparse = (ob.cube_sparse - field_disentangled) * ob.valid_mask
bg_correct = ob.ExtractBackgroundFromResidue(residue_sparse, min_radius=2, max_radius=18, border_margin = 30, regime='median', show=True)
bg_testo = ob.background_sparse + bg_correct.squeeze()

#%%
ob.bg_prior = bg_testo
ob.background_sparse *= 0.0

field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=True, force_cpu=False)
# field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=False, force_cpu=False)
data_img = ob.cube_sparse.clone() #- ob.background_sparse.view(ob.N_wvl, 1, 1)

#%%
from tools.multisources import VisualizeSources, PlotSourcesProfiles, extract_ROIs, add_ROIs
from matplotlib.colors import LogNorm

display_norm = LogNorm(vmin=1, vmax=data_img.sum(dim=0).max().item()) # again, rather empirical values
_ = VisualizeSources(data_img, field_disentangled, norm=display_norm, mask=ob.valid_mask, ROI=ob.ROI_plot)

PlotSourcesProfiles(data_img, field_disentangled, ob.sources.table, show=False,radius=16, title='Predicted profiles + astrometry correction (spectrally binned)', y_max=350, y_min=0.25)

if 'fitted' in model_cache.stem:
    prefix = 'fitted'
elif 'predicted' in model_cache.stem:
    prefix = 'predicted'

if   prefix == 'fitted':
    plt.savefig(data_folder / f"plots/profile_fitted_{ob.model_type}_{cube_path.stem}.pdf", dpi=300)
elif prefix == 'predicted':
    plt.savefig(data_folder / f"plots/profile_predicted_{ob.model_type}_{cube_path.stem}.pdf", dpi=300)

#%%
# _, PSFs, _ = ob.simulate_sparse(x=None, src_subset=None, return_PSFs=True)
# PSF_mean = PSFs.mean(dim=0).cpu().numpy()

#%%
Strehls_per_λ = ob.PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(ob.λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()


#%%
from tiptorch.tools.utils import  FitMoffat2D_astropy

crop_size = 7

ROIs_0, _, _, _ = extract_ROIs(data_img,           ob.sources.table, box_size=crop_size)
ROIs_1, _, _, _ = extract_ROIs(field_disentangled, ob.sources.table, box_size=crop_size)

# Spectrally average the PSFs
avg_white = ( lambda x: torch.stack(x).mean(dim=1) if isinstance(x[0], torch.Tensor) else np.mean(np.stack(x), axis=1) )

PSFs_0_white = avg_white(ROIs_0)
PSFs_1_white = avg_white(ROIs_1)

PSF_0_avg = torch.stack(ROIs_0).mean(dim=0).cpu().numpy()
PSF_1_avg = torch.stack(ROIs_1).mean(dim=0).cpu().numpy()


FWHM_λ = []

for i in range(ob.N_wvl):
    f0_x, f0_y, model_0, fit_0 = FitMoffat2D_astropy(PSF_0_avg[i])
    f1_x, f1_y, model_1, fit_1 = FitMoffat2D_astropy(PSF_1_avg[i])

    f0 = np.sqrt(f0_x**2 + f0_y**2)
    f1 = np.sqrt(f1_x**2 + f1_y**2)
    FWHM_λ.append( np.abs(f0-f1) / f0 )
    
FWHM_λ = np.array(FWHM_λ) * 100
mean_fwhm_λ = np.mean(FWHM_λ)

print(f'Mean FWHM difference: {mean_fwhm_λ:.1f}%')

# ---------------------------------------------------
peak_0 = torch.stack(ROIs_0).mean(dim=0).amax(dim=(-2,-1)).cpu()
peak_1 = torch.stack(ROIs_1).mean(dim=0).amax(dim=(-2,-1)).cpu()
d = (peak_0 - peak_1).abs() / (peak_0 + 1e-10) * 100.0
mean_peak_λ = np.mean(d.cpu().numpy())

print(f'Mean peak value difference: {mean_peak_λ:.1f}%')


#%
FWHM_ob = []

for i in range(len(ob.sources)):
    try:
        f0_x, f0_y, _, _ = FitMoffat2D_astropy(PSFs_0_white[i,...].cpu().numpy())
        f1_x, f1_y, _, _ = FitMoffat2D_astropy(PSFs_1_white[i,...].cpu().numpy())
        f0 = np.sqrt(f0_x**2 + f0_y**2)
        f1 = np.sqrt(f1_x**2 + f1_y**2)
        FWHM_ob.append(np.abs(f0 - f1) / f0 if f0 > 0 else np.nan)
    except Exception:
        FWHM_ob.append(np.nan)
    
FWHM_ob = np.array(FWHM_ob) * 100

# Median and flux-weighted stats for FWHM errors
fwhm_flux = np.asarray(ob.sources.table['peak_value'].values[:len(FWHM_ob)], dtype=float)
fwhm_weights = np.where(fwhm_flux > 0, fwhm_flux, 0.0)

fwhm_valid = ~np.isnan(FWHM_ob)
fwhm_median   = np.median(FWHM_ob[fwhm_valid])
fwhm_p16, fwhm_p84 = np.percentile(FWHM_ob[fwhm_valid], [16, 84])

fwhm_w_avg = np.average(FWHM_ob[fwhm_valid], weights=fwhm_weights[fwhm_valid]) \
    if fwhm_weights[fwhm_valid].sum() > 0 else np.nan
_si = np.argsort(FWHM_ob[fwhm_valid])
_ds, _ws = FWHM_ob[fwhm_valid][_si], fwhm_weights[fwhm_valid][_si]
_cw = np.cumsum(_ws) / (_ws.sum() + 1e-12)
fwhm_wp16 = float(np.interp(0.16, _cw, _ds))
fwhm_wp84 = float(np.interp(0.84, _cw, _ds))

col_w = 22
print(f"\n{'FWHM relative error summary (per source, white-light)':^{3*col_w}}")
print('-' * 3 * col_w)
print(f"{'Statistic':<{col_w}}{'Value [%]':>{col_w}}{'1-sigma interval [%]':>{col_w}}")
print('-' * 3 * col_w)
print(f"{'Median':<{col_w}}{fwhm_median:>{col_w}.2f}{f'[{fwhm_p16:.2f}, {fwhm_p84:.2f}]':>{col_w}}")
print(f"{'Flux-weighted mean':<{col_w}}{fwhm_w_avg:>{col_w}.2f}{f'[{fwhm_wp16:.2f}, {fwhm_wp84:.2f}]':>{col_w}}")
print('-' * 3 * col_w)

#%
cutoff_id = None

a = PSFs_0_white[:cutoff_id, ...].amax(dim=(-2,-1)).cpu()
b = PSFs_1_white[:cutoff_id, ...].amax(dim=(-2,-1)).cpu()

d = (a-b).abs() / (a + 1e-10)

flux = np.asarray(ob.sources.table['peak_value'].values[:cutoff_id], dtype=float)
d_np = d.cpu().numpy() * 100
flux_valid = flux[:len(d_np)]  # guard against length mismatch
weights = np.where(flux_valid > 0, flux_valid, 0.0)
weighted_avg = np.average(d_np, weights=weights) if weights.sum() > 0 else np.nan

p16, p84 = np.percentile(d_np, [16, 84])

# Weighted 1-sigma percentiles
sorted_idx = np.argsort(d_np)
d_sorted = d_np[sorted_idx]
w_sorted = weights[sorted_idx]
cum_w    = np.cumsum(w_sorted) / (w_sorted.sum() + 1e-12)
wp16 = float(np.interp(0.16, cum_w, d_sorted))
wp84 = float(np.interp(0.84, cum_w, d_sorted))
median_val = np.median(d_np)

# plt.hist(d_np, bins=50, range=(0, 100))

# Median line + 1-sigma band
# plt.axvline(median_val, color='red',    linestyle='--', label=f'Median: {median_val:.2f}% [{p16:.2f}, {p84:.2f}]')
# plt.axvspan(p16, p84, alpha=0.10, color='red')
# plt.axvline(weighted_avg, color='blue', linestyle='--', label=f'Flux-weighted mean: {weighted_avg:.2f}% [{wp16:.2f}, {wp84:.2f}]')
# plt.axvspan(wp16, wp84, alpha=0.10, color='blue')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.xlim(0, None)
# plt.ylabel('Number of sources')
# plt.xlabel('Relative difference in peak values, [%]')
# plt.show()

# Summary table
col_w = 22
print(f"\n{'Peak-value relative error summary':^{3*col_w}}")
print('-' * 3 * col_w)
print(f"{'Statistic':<{col_w}}{'Value [%]':>{col_w}}{'1-sigma interval [%]':>{col_w}}")
print('-' * 3 * col_w)
print(f"{'Median':<{col_w}}{median_val:>{col_w}.2f}{f'[{p16:.2f}, {p84:.2f}]':>{col_w}}")
print(f"{'Flux-weighted mean':<{col_w}}{weighted_avg:>{col_w}.2f}{f'[{wp16:.2f}, {wp84:.2f}]':>{col_w}}")
print('-' * 3 * col_w)


# Write results to a text file
report_path = data_folder / f"plots/errors_{ob.model_type}_{prefix}_{cube_path.stem}.txt"
with open(report_path, 'w') as _f:
    _f.write(f"Sample: {cube_path.stem}\n\n")
    _f.write(f"Spectrally-averaged metrics (white-light stack)\n")
    _f.write('-' * 3 * col_w + '\n')
    _f.write(f"  Mean FWHM difference (across wavelengths):      {mean_fwhm_λ:.2f}%\n")
    _f.write(f"  Mean peak-value difference (across wavelengths): {mean_peak_λ:.2f}%\n")
    _f.write('-' * 3 * col_w + '\n\n')
    _f.write(f"{'FWHM relative error (per source, white-light)':^{3*col_w}}\n")
    _f.write('-' * 3 * col_w + '\n')
    _f.write(f"{'Statistic':<{col_w}}{'Value [%]':>{col_w}}{'1-sigma interval [%]':>{col_w}}\n")
    _f.write('-' * 3 * col_w + '\n')
    _f.write(f"{'Median':<{col_w}}{fwhm_median:>{col_w}.2f}{f'[{fwhm_p16:.2f}, {fwhm_p84:.2f}]':>{col_w}}\n")
    _f.write(f"{'Flux-weighted mean':<{col_w}}{fwhm_w_avg:>{col_w}.2f}{f'[{fwhm_wp16:.2f}, {fwhm_wp84:.2f}]':>{col_w}}\n")
    _f.write('-' * 3 * col_w + '\n\n')
    _f.write(f"{'Peak-value relative error':^{3*col_w}}\n")
    _f.write('-' * 3 * col_w + '\n')
    _f.write(f"{'Statistic':<{col_w}}{'Value [%]':>{col_w}}{'1-sigma interval [%]':>{col_w}}\n")
    _f.write('-' * 3 * col_w + '\n')
    _f.write(f"{'Median':<{col_w}}{median_val:>{col_w}.2f}{f'[{p16:.2f}, {p84:.2f}]':>{col_w}}\n")
    _f.write(f"{'Flux-weighted mean':<{col_w}}{weighted_avg:>{col_w}.2f}{f'[{wp16:.2f}, {wp84:.2f}]':>{col_w}}\n")
    _f.write('-' * 3 * col_w + '\n')
print(f"Report saved to: {report_path}")


#%%
from scipy.stats import gaussian_kde

flux = np.asarray(ob.sources.table['peak_value'].values, dtype=float)
d_pct = d.cpu().numpy() * 100
valid = (flux > 0) & (d_pct > 0)
xy = np.vstack([np.log10(flux[valid]), np.log10(d_pct[valid])])
z_valid = gaussian_kde(xy)(xy)
z = np.zeros(len(flux))
z[valid] = z_valid

plt.xlabel('Flux (normalized), [a.u.]')
plt.ylabel('Difference, [%]')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.scatter(flux, d_pct, c=z, s=20, alpha=0.7, cmap='viridis')
plt.colorbar(label='Density')
plt.show()

#%%
i_src = np.random.randint(0, ob.N_src)

p0   = PSFs_0_white[i_src].cpu()
p1   = PSFs_1_white[i_src].cpu()
diff = np.abs(p0 - p1)

vmin = 0.0
vmax = max(p0.max(), p1.max()).item()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im0 = axes[0].imshow(p0,   origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
im1 = axes[1].imshow(p1,   origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
im2 = axes[2].imshow(diff, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
axes[0].set_title('Data (white-light)')
axes[1].set_title('Model (white-light)')
axes[2].set_title('Difference')
fig.colorbar(im2, ax=axes, shrink=0.7)
plt.show()



#%%
simulated_full = ob.SimulateField(full_spectrum=True, N_λ_per_batch=10, disentangle_spectra=True, force_cpu=False)

#%% Plot multispectral cubes as RGB images
from tools.plotting import PlotSpetralCubeInRGB

# Mapping MUSE spectral range to visible spectrum range for RGB conversion
λ_vis = np.linspace(440, 750, ob.N_wvl_full)  # MUSE covers ~465-930nm, so we map it to 440-750nm for visualization

color_kwargs ={
    'saturation' :  2.0,
    'contrast'   :  1.75,
    'wb_shift'   : -0.3,
    'mg_shift'   :  0.15,
    'min_val'    :  1000,
    'max_val'    :  4*7.5e6,
    'show'       :  False,    
    'wavelengths':  λ_vis
}

model_full = simulated_full[ob.ROI_plot] + ob.background_full.view(-1, 1, 1).numpy()
data_full  = ob.cube_full[ob.ROI_plot]
diff_full  = np.abs(model_full - data_full)

_ = PlotSpetralCubeInRGB( model_full, title="Model (full spectrum)", **color_kwargs )
plt.savefig(data_folder / f"plots/model_full_{ob.model_type}_{cube_path.stem}.pdf", dpi=300)

_ = PlotSpetralCubeInRGB( data_full, title="Data (full spectrum)", **color_kwargs )
plt.savefig(data_folder / f"plots/data_full_{ob.model_type}_{cube_path.stem}.pdf", dpi=300)

_ = PlotSpetralCubeInRGB( diff_full, title="Difference", **color_kwargs )
plt.savefig(data_folder / f"plots/diff_full_{ob.model_type}_{cube_path.stem}.pdf", dpi=300)

# del model_full, data_full, diff_full
# torch.cuda.empty_cache()

#%%
PlotSourcesProfiles(
    data_full - ob.background_full.view(-1, 1, 1).numpy(),
    simulated_full,
    ob.sources.table,
    radius=16,
    title='Radial profiles',
    y_max=350,
    y_min=0.25
)

#%%
# Compute HST spectra for each source
HST_weights = np.array([1.0, 1.0, 0.93, 0.82, 1.0])[None, ...] * 2.35  # empirical multiplier
src_spectra_HST = sources_df.copy().drop(columns=['x, [asec]', 'y, [asec]', 'flux (total, normalized)']).loc[sources.index]
λ_HST = np.array([float(col[1:-1]) for col in src_spectra_HST.columns]) # [nm]
src_spectra_HST = src_spectra_HST.to_numpy() * HST_weights * flux_λ_norm.median().item()


# Generate as many random colors as N_src
colors = []
for _ in range(N_src):
    r = random.random()
    g = random.random()
    b = random.random()
    colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
colors[:10] = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plot spectra for NFM and HST
plt.figure(figsize=(6, 4))
# for i_src in range(N_src):
for i_src in range(130, 137):
    # Full MUSE spectrum (in true units)
    plt.plot(λ_full, src_spectra_full[i_src], linewidth=0.25, alpha=0.2, color=colors[i_src])
    
    # Sparse and renormalized NFM spectrum
    plt.scatter(λ_sparse.cpu().numpy(), src_spectra_sparse[i_src].cpu().numpy(), color=colors[i_src], marker='o', s=30, alpha=0.8)
    # plt.plot(λ_sparse.cpu().numpy(), src_spectra_sparse[i_src].cpu().numpy(), linestyle='--', linewidth=0.5, alpha=1, color=colors[i_src])
    
    # Renormalized and corrected HST spectrum
    # plt.scatter(λ_HST, src_spectra_HST[i_src], color=colors[i_src], marker='+', s=30, alpha=0.8)
    plt.plot(   λ_HST, src_spectra_HST[i_src], linewidth=0.25, alpha=1, color=colors[i_src])


plt.xlabel('Wavelength, [nm]')
plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
plt.title('Sources spectra preview')
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.yscale('log')
# plt.ylim(1e2, None)
plt.show()


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
            sources,
            proximity_table,
            x_range, y_range,
            d_offset=30,
            N=min(N_batch, N_src)
        )

        if len(source_indices) == 0:
            continue  # Skip empty tiles

        # Get tile data
        tile_data = cube_sparse[:, y_range[0]:y_range[1], x_range[0]:x_range[1]]

        # Prepare empty image for this tile
        tile_empty = torch.zeros([N_wvl, y_range[1]-y_range[0], x_range[1]-x_range[0]], device=default_device)

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
        local_tile_coords = [local_coords[idx] - torch.tensor([x_range[0], y_range[0]], device=default_device) for idx in source_indices]
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
    model_sparse_optimized = torch.zeros_like(cube_sparse, device=default_device)

    for tile in tqdm(tiles, desc="Generating final model"):
        x_range, y_range = tile['x_range'], tile['y_range']
        source_indices = select_sources_in_tile(
            sources, proximity_table, x_range, y_range, d_offset=30, N=min(N_batch, N_src)
        )

        if len(source_indices) == 0:
            continue

        tile_empty = torch.zeros([N_wvl, y_range[1]-y_range[0], x_range[1]-x_range[0]], device=default_device)
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
        local_tile_coords = [local_coords[idx] - torch.tensor([x_range[0], y_range[0]], device=default_device) for idx in source_indices]
        model_tile = add_ROIs(tile_empty, PSFs_fit, local_tile_coords, [global_coords[idx] for idx in source_indices])

        # Add to final model
        model_sparse_optimized[:, y_range[0]:y_range[1], x_range[0]:x_range[1]] = model_tile

        # Clear cache to save memory
        torch.cuda.empty_cache()

# Visualize the optimized model
VisualizeSources(cube_sparse, model_sparse_optimized, norm=norm_field, mask=valid_mask)
PlotSourcesProfiles(cube_sparse, model_sparse_optimized, sources, radius=16, title='Optimized PSFs')


