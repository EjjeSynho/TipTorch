#%%
%reload_ext autoreload
%autoreload 2

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tiptorch._config import default_device, project_settings
from tools.observations import MUSEObservation

from pathlib import Path

MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])

#%%
raw_path   = MUSE_DATA_FOLDER / "omega_cluster/raw_data/MUSE.2020-02-24T05-16-30.566.fits.fz"
cube_path  = MUSE_DATA_FOLDER / "omega_cluster/reduced_cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
cache_path = MUSE_DATA_FOLDER / "omega_cluster/cached_cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.pickle"

ob = MUSEObservation(raw_path, cube_path, cache_path, device=default_device)

ob.λ_batch_size = 3681//3

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
sources = sources.nlargest(50, 'peak_value')

#%%
ob.sources_table = sources

ob.ExtractSources(verbose=True, max_nan_fraction=0.7)
ob.DisplaySources(draw_box_size=5)

#%%
ob.PlotSourceSpectra()
ob.InitSimulation()

#%%
# ob.FitPSFModel(repeat=3, max_iter=200)
# ob.FitPSFModel(fit=['astrometry', 'photometry'], repeat=3, max_iter=200)
ob.FitPSFModel(fit=['astrometry'], repeat=1, max_iter=200)
# ob.FitPSFModel(repeat=3, max_iter=200)
# ob.FitPSFModel(repeat=3, max_iter=200)

#%%
model, PSFs, flux_normalization = ob.SimulateField(return_PSFs=True)
ob.DisplaySimulation(plot_profiles=True)

#%%
from tools.multisources import VisualizeSources, add_ROIs_separately

def DisentangleFlux(ob, PSFs, rcond=1e-2):
    """
    Reconstruct spectral cube by solving for optimal flux coefficients per wavelength.
    
    Parameters
    ----------
    ob : MUSEObservation
        Observation object containing cube data and source information
    PSFs : torch.Tensor
        PSF models for each source [N_src, N_wvl, H, W]
    rcond : float
        Cutoff for small singular values in least-squares solver
        
    Returns
    -------
    I_sim : torch.Tensor
        Reconstructed spectral cube [N_wvl, H, W]
    """
    canvas_ = torch.zeros_like(ob.cube_sparse, device=ob.cube_sparse.device)
    srcs_stack = add_ROIs_separately(canvas_, PSFs, ob.sources.slices_local, ob.sources.slices_global)
    
    P = srcs_stack.view(ob.N_src, ob.N_wvl, -1).permute(1, 2, 0)  # [N_wvl, pixels, N_src]
    I_data = ob.cube_sparse.view(ob.N_wvl, -1, 1)  # [N_wvl, pixels, 1]
    
    # Solve flux for all wavelengths
    spectrum = torch.linalg.lstsq(P, I_data, rcond=rcond).solution  # [N_wvl, N_src, 1]
    
    # Reconstruct the full spectral cube
    I_sim = torch.matmul(P, spectrum).squeeze(-1).view(ob.N_wvl, ob.cube_sparse.shape[-2], ob.cube_sparse.shape[-1])
    
    return I_sim, spectrum.squeeze(-1)  # Return both the reconstructed cube and the flux coefficients


I_sim, F = DisentangleFlux(ob, PSFs)

#%%
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

# plt.imshow(bg.sum(dim=0).cpu(), origin='lower', norm=display_norm, cmap='inferno')

bg = torch.nanmean(bg, dim=(-2,-1))  # Average over the field, ignoring NaNs

bg = bg.view(ob.N_wvl, 1, 1)

#%%
xx = ob.cube_sparse.clone()

#%%
ob.cube_sparse -= bg

#%%
from matplotlib.colors import LogNorm

display_norm = LogNorm(vmin=1, vmax=ob.cube_sparse.sum(dim=0).max()) # again, rather empirical values
_ = VisualizeSources(ob.cube_sparse, ob.simulated_sparse, norm=display_norm, mask=ob.valid_mask, ROI=ob.ROI_plot)

#%%
_ = VisualizeSources(ob.cube_sparse, I_sim, norm=display_norm, mask=ob.valid_mask, ROI=ob.ROI_plot)

#%%
Strehls_per_λ = ob.PSF_model.ComputeStrehl()
plt.title('Strehl ratio vs. λ (for the 1st source)')
plt.plot(ob.λ_sparse, 100.0 * Strehls_per_λ.flatten().cpu())
plt.ylabel('Strehl ratio, [%]')
plt.xlabel('Wavelength, [nm]')
plt.grid()
plt.show()

#%%
_ = ob.SimulateField(full_spectrum=True)
ob.DisplaySimulation(plot_profiles=True, plot_full_spectrum=True)

#%%
ob.PlotSourceSpectra(title='Sources spectra (residual)', show_sparse=False, plot_residual=True, smooth_kernel=15)

#%%
# Compute HST spectra for each source
HST_weights = np.array([1.0, 1.0, 0.93, 0.82, 1.0])[None, ...] * 2.35  # empirical multiplier
src_spectra_HST = sources_df.copy().drop(columns=['x, [asec]', 'y, [asec]', 'flux (total, normalized)']).loc[sources.index]
λ_HST = np.array([float(col[1:-1]) for col in src_spectra_HST.columns]) # [nm]
src_spectra_HST = src_spectra_HST.to_numpy() * HST_weights * flux_λ_norm.median().item()

#%%
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
    sources_ = sources.to_numpy()[:, :-1]  # Exclude the last column
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


proximity_table = create_proximity_table(sources)

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


tiles = split_image_into_tiles(cube_full.shape, n_tiles_x=4, n_tiles_y=4, border_offset=10)
visualize_tiles(cube_full.sum(axis=0), tiles, title='Image Tiles', norm=norm_field)

#%%
sources_inputs = model_inputs.unstack(model_inputs.stack())
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
        return torch.zeros([N_wvl, tile['y_range']-tile['y_range'], tile['x_range']-tile['x_range']], device=default_device)

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
        torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=default_device),
        [PSF_1[i,...]*norm_factors[src_id] for i, src_id in enumerate(source_indices)],
        [local_coords [src_id] for src_id in source_indices],
        [global_coords[src_id] for src_id in source_indices]
    )
    
    return model_tile[..., y_range[0]:y_range[1], x_range[0]:x_range[1]]


torch.cuda.empty_cache()
with torch.no_grad():
    tiles_model = [simulate_tile(sources_inputs, tile, proximity_table, sources, max_sources=N_batch) for tile in tqdm(tiles)]

# Concatenate tiles back into image based on their positions
model_sparse_tiled = torch.zeros_like(cube_sparse, device=default_device)
for tile, model_tile in zip(tiles, tiles_model):
    x_min, x_max = tile['x_range']
    y_min, y_max = tile['y_range']
    model_sparse_tiled[:, y_min:y_max, x_min:x_max] = model_tile[...]

    
#%%
VisualizeSources(cube_sparse, model_sparse_tiled.detach(), norm=norm_field, mask=valid_mask)

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


