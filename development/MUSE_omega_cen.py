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

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

#%%
raw_path   = MUSE_DATA_FOLDER / "omega_cluster/raw_data/MUSE.2020-02-24T05-16-30.566.fits.fz"
cube_path  = MUSE_DATA_FOLDER / "omega_cluster/reduced_cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
cache_path = MUSE_DATA_FOLDER / "omega_cluster/cached_cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.pickle"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=default_device)
ob.λ_batch_size = ob.λ_full.shape[0] // 3 + 1  # Process all wavelengths at once (adjust if memory issues arise)

#%%
# Read pre-processed HST data from DataFrame
sources_file = MUSE_DATA_FOLDER / 'omega_cluster/OmegaCentaury_data/HST_sources_in_FoV.csv'

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

# Add weight column for later use (e.g. in loss function)
sources['weight'] = 1.0

# Leave the first N brighest sources
# sources = sources.nlargest(50, 'peak_value')

ob.sources_table = sources
ob.ExtractSources(verbose=True, max_nan_fraction=0.7)
#%%
# ob.DisplaySources(draw_box_size=5)
# ob.PlotSourceSpectra()

#%
ob.InitSimulation()

#%%
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

model_cache = MUSE_DATA_FOLDER / "omega_cluster/PSF_model_predicted.pt"

if model_cache.exists():
    print("Loading PSF model from cache...")
    model_data = torch.load(model_cache)
    ob.PSF_model = PSFModelNFM.load(model_data, device=ob.device)
else:
    print("Fitting PSF model...")
    ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)
    model_data = ob.PSF_model.save()#cpu=True)
    torch.save(model_data, model_cache)
    print(f"PSF model saved to {model_cache}")

#%%
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

model_cache = MUSE_DATA_FOLDER / "omega_cluster/PSF_model_fitted.pt"

if model_cache.exists():
    print("Loading PSF model from cache...")
    model_data = torch.load(model_cache)
    ob.PSF_model = PSFModelNFM.load(model_data, device=ob.device)
else:
    print("Fitting PSF model...")
    ob.FitPSFModel(repeat=3, max_iter=200)
    model_data = ob.PSF_model.save()
    torch.save(model_data, model_cache)
    print(f"PSF model saved to {model_cache}")


#%
# _, PSFs, _ = ob.simulate_sparse(return_PSFs=True)

#%%
from tools.multisources import VisualizeSources, add_ROIs_separately, PlotSourcesProfiles, extract_ROIs, add_ROIs
from tools.observations import SourcesSubset
from matplotlib.colors import LogNorm


# I_sim, fluxes, bg_solved = DisentangleFlux(ob.sources.select(None), PSFs, ob.cube_sparse, solver='nonlinear')

# fluxes_ = fluxes.cpu().numpy().T  # [N_src, N_wvl]
# F_norm = ob.PSF_model['F_norm'].unsqueeze(-1).cpu().numpy()
# PSF_norm_factor = ob.PSF_model.evaluate_splines(ob.PSF_model.inputs_manager['F_norm_λ_ctrl'], ob.PSF_model.λ_sim_normed).cpu().numpy()
# fluxes_ /= F_norm * PSF_norm_factor

# for i in range(ob.N_src):
#     ob.sources.spectra_sparse[i] = torch.tensor(fluxes_[i], device=ob.device, dtype=ob.PSF_model.dtype)

# colors = []
# for _ in range(0,8):
#     r = random.random()
#     g = random.random()
#     b = random.random()
#     colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')

# colors[:10] = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# plt.figure(figsize=(6, 4))

# for i_src in range(0, 8):
#     plt.plot(ob.λ_sparse, ob.sources.spectra_sparse[i_src].cpu().numpy(), color=colors[i_src], alpha=0.8)
#     plt.plot(ob.λ_sparse, fluxes_[i_src], color=colors[i_src], alpha=0.8, linestyle='--')

# plt.xlabel('Wavelength, [nm]')
# plt.ylabel(r'Flux, [ $10^{-20} \frac{erg} {s \, \cdot \, cm^2 \, \cdot \, Å} ]$')
# plt.title('Sources spectra preview')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# field_disentangled = ob.simulate_sparse(return_PSFs=False)[0]
#%
field_disentangled = ob.SimulateField(full_spectrum=False, disentangle_spectra=True, force_cpu=False)

#%%
from matplotlib.colors import LogNorm

bg = ob.residue_sparse.clone()

# Adjustable parameters
min_radius = 2
max_radius = 14
border_margin = 15

# Create circular masks around sources with radii proportional to log(flux)
src_positions = ob.sources_table[['x_peak', 'y_peak']].values
src_fluxes = ob.sources_table['peak_value'].values

# Compute radii: log-scale mapping from min/max flux to min_radius-max_radius pixel radius
log_fluxes = np.log10(src_fluxes + 1e-10)  # avoid log(0)
log_min, log_max = log_fluxes.min(), log_fluxes.max()
radii = min_radius + (log_fluxes - log_min) / (log_max - log_min + 1e-10) * (max_radius - min_radius)

# Create coordinate grids
y_grid, x_grid = torch.meshgrid(
    torch.arange(bg.shape[-2], device=bg.device),
    torch.arange(bg.shape[-1], device=bg.device),
    indexing='ij'
)

# Build composite mask (True = masked out)
mask = torch.zeros(bg.shape[-2:], dtype=torch.bool, device=bg.device)

# Mask circular regions around sources
for (x_src, y_src), r in zip(src_positions, radii):
    dist_sq = (x_grid - x_src)**2 + (y_grid - y_src)**2
    mask |= (dist_sq <= r**2)

# Mask border regions
mask |= (x_grid < border_margin) | (x_grid >= bg.shape[-1] - border_margin)
mask |= (y_grid < border_margin) | (y_grid >= bg.shape[-2] - border_margin)

# Apply mask (set masked pixels to NaN)
bg[:, mask] = float('nan')

display_norm = LogNorm(vmin=1, vmax=ob.cube_sparse.sum(dim=0).max()) # again, rather empirical values
plt.imshow(bg.sum(dim=0).cpu(), origin='lower', norm=display_norm, cmap='inferno')

bg = torch.nanmean(bg, dim=(-2,-1))  # Average over the field, ignoring NaNs
bg = bg.view(ob.N_wvl, 1, 1)

#%%
data_img = ob.cube_sparse.clone() - ob.background_sparse.view(ob.N_wvl, 1, 1)

#%%
# display_norm = LogNorm(vmin=1, vmax=data_img.sum(dim=0).max()) # again, rather empirical values
# _ = VisualizeSources(data_img, ob.simulated_sparse, norm=display_norm, mask=ob.valid_mask, ROI=ob.ROI_plot)

display_norm = LogNorm(vmin=1, vmax=data_img.sum(dim=0).max().item()) # again, rather empirical values
_ = VisualizeSources(data_img, field_disentangled, norm=display_norm, mask=ob.valid_mask, ROI=ob.ROI_plot)

#%%
# w = ob.sources.table['peak_value'].values.copy() # get flux values as weights
# w = w / w.max() # Normalize weights by flux to [0, 1]
# min_thresh = 0.1
# w += min_thresh
# w /= 1. + min_thresh

PlotSourcesProfiles(data_img, field_disentangled, ob.sources.table, radius=16, title='Radial profiles', y_max=350, y_min=0.25)
# PlotSourcesProfiles(ob.cube_sparse, ob.simulated_sparse, ob.sources.table, radius=16, title='Source radial profiles (sparse spectrum)')

#%%
Strehls_per_λ = ob.PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(ob.λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()


#%%
ROIs_0, _, _, _ = extract_ROIs(data_img, ob.sources.table, box_size=7)
ROIs_1, _, _, _ = extract_ROIs(I_sim_,   ob.sources.table, box_size=7)

# Spectrally average the PSFs
avg_white = (
    lambda x: torch.stack(x).mean(dim=1)
    if isinstance(x[0], torch.Tensor)
    else np.mean(np.stack(x), axis=1)
)

PSFs_0_white = avg_white(ROIs_0)
PSFs_1_white = avg_white(ROIs_1)


cutoff_id = None

a = PSFs_0_white[:cutoff_id, ...].amax(dim=(-2,-1)).cpu()
b = PSFs_1_white[:cutoff_id, ...].amax(dim=(-2,-1)).cpu()

d = (a-b).abs() / (a + 1e-10)



plt.hist(d.cpu().numpy() * 100, bins=50, range=(0, 100))

# Median line
median_val = np.median(d.cpu().numpy() * 100)
plt.axvline(median_val, color='red', linestyle='--', label=f'Median (all sources): {median_val:.2f}%')
plt.legend()


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
# from tools.observations import split_field_into_tiles, visualize_field_tiles

# tiles = split_field_into_tiles(ob.cube_sparse.shape, N_tiles_x=4, N_tiles_y=4, border_offset=0)
# visualize_field_tiles(ob.cube_sparse.sum(axis=0).cpu().numpy(), tiles, title='Field Tiles', norm=display_norm)

# for tile in tiles:
#     tile['srcs'] = ob.sources.select_sources_in_tile(tile['ROI'], d_offset=30)
#     tile['srcs_light'] = tile['srcs'].clone(light_weight=True).rebase_coords(crop_ROI=tile['ROI'], filter_empty=False)
#     print(f"Tile {tile['ID']}: {len(tile['srcs'])} sources in tile")


# I_sim_full, fluxes_full, bg_full = DisentangleFluxBatched(ob, λ_batch_size=10, solver='nonlinear')
# I_sim_full, fluxes_full, bg_full = DisentangleFluxBatched(ob, λ_batch_size=10, solver='linear')


#%%
simulated_full = ob.SimulateField(full_spectrum=True, N_λ_per_batch=10, disentangle_spectra=True, force_cpu=False)

#%% Plot multispectral cubes as RGB images
from tools.plotting import PlotSpetralCubeInRGB

# Mapping MUSE spectral range to visible spectrum range for RGB conversion
λ_vis = np.linspace(440, 750, ob.N_wvl_full)  # MUSE covers ~465-930nm, so we map it to 440-750nm for visualization


_ = PlotSpetralCubeInRGB(
    simulated_full[ob.ROI_plot],
    wavelengths=λ_vis,
    title="Model",
    min_val=500, max_val=7.5e6,
    show=False
)

_ = PlotSpetralCubeInRGB(
    ob.cube_full[ob.ROI_plot] - ob.background_full.view(-1, 1, 1).numpy(),
    wavelengths=λ_vis,
    title="Data",
    min_val=500, max_val=7.5e6,
    show=False
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


