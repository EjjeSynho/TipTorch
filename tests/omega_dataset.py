#%%
# The data taken from: https://zenodo.org/records/11104046

import os
import glob
import pickle
from attr import s
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.nddata import Cutout2D

from project_settings import MUSE_DATA_FOLDER
from tools.utils import Photometry

cube_path   = MUSE_DATA_FOLDER / "omega_cluster/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
data_folder = MUSE_DATA_FOLDER / 'omega_cluster/OmegaCentaury_data/'

#%%
print("Loading the Astrometric part of the catalog...")
t_astro = Table.read(str(data_folder / "astrometry.fits"))
print("Columns of astrometric catalog:")
print(t_astro.columns)
print()

print("Reading the photometric catalogs...")
phot_dict = {}
file_list = sorted(glob.glob(str(data_folder / "F*photometry.fits")))
for filename in file_list:
    filter_name = os.path.basename(filename)[:5]
    print(filter_name)
    phot_dict[filter_name] = Table.read(filename)
    
mags_dict = {}

for key, item in phot_dict.items():
    df_phot = item.to_pandas()
    df_phot.set_index('ID', inplace=True)
    mags_dict[key] = df_phot["corrected_mag"]

df_phot = pd.concat(mags_dict, axis=1)

df_astro = t_astro.to_pandas()
df_astro.set_index('ID', inplace=True)
df_astro = df_astro[['RA', 'DEC', 'x', 'y']]

# 1) Load your stacked image and its WCS
image_path = data_folder / "F606W_stacked_image.fits"
hdu = fits.open(image_path)[0]
stacked_image = hdu.data
wcs = WCS(hdu.header)

# 2) Define your MUSE pointing center (in deg) from the cube header
with fits.open(cube_path) as hdul:
    ra_center  = hdul[0].header['RA']  * u.deg
    dec_center = hdul[0].header['DEC'] * u.deg
    center_coord = SkyCoord(ra=ra_center, dec=dec_center)

# 3) Build a SkyCoord array for all your sources in df_astro
#    (assume df_astro has columns 'RA','DEC' in degrees)
coords = SkyCoord(ra  = df_astro['RA' ].values*u.deg,
                  dec = df_astro['DEC'].values*u.deg)
#%%
# 4) Fast search of everything within the MUSE NFM FoV radius
radius = (7*np.sqrt(2)/2) * u.arcsec * 2
#    search_around_sky builds an internal k-d tree and is O(n log n)
# idx_center, idx_sources, sep2d, _ = coords.search_around_sky(center_coord, radius)

sep = coords.separation(center_coord) # array of Angle objects
mask = sep < radius                   # boolean array

# 5) Extract only the matching sources
sel_idx = np.nonzero(mask)[0]                 # indices into your 2M+ catalog
df_sel  = df_astro.iloc[sel_idx].copy()
df_mag  = df_phot.loc[df_sel.index]    
  
# 6) Transform the selected sky coords into pixel coordinates
#    (world_to_pixel returns floats)
x_pix_sel, y_pix_sel = wcs.world_to_pixel(
    SkyCoord(ra=df_sel['RA'].values*u.deg, dec=df_sel['DEC'].values*u.deg))

# 7) Get the pixel coordinate of the center, too
x0, y0 = wcs.world_to_pixel(center_coord)

# 8) Compute offsets in arcsec
pixel_scale = 0.04  # arcsec / pixel
dx = (x_pix_sel - x0) * pixel_scale
dy = (y_pix_sel - y0) * pixel_scale

df_sel['x'] = dx
df_sel['y'] = dy

df_sel = df_sel.rename(columns={'x': 'x, [asec]', 'y': 'y, [asec]'})

# 9) Cut out image region
ROI_size_px = 280
cutout = Cutout2D(stacked_image, position=(x0, y0), size=(2*ROI_size_px, 2*ROI_size_px))
cutout_data = cutout.data

# 10) Plot
fig, ax = plt.subplots(figsize=(8,8))

extent = [ -ROI_size_px*pixel_scale,  ROI_size_px*pixel_scale,
           -ROI_size_px*pixel_scale,  ROI_size_px*pixel_scale ]

im = ax.imshow(cutout_data + 15,
               origin = 'lower',
               extent = extent,
               norm   = colors.LogNorm(vmin=10, vmax=5000),
               cmap   = 'Greys',
               zorder = 0)

# your selected sources
ax.scatter(dx, dy,
           s=40, facecolors='none',
           edgecolors='red', linewidths=0.5,
           label=f'{len(df_sel)} sources')

# mark (0,0) = the cube center
ax.scatter(0, 0, marker='o', color='green', s=80, label='MUSE center')

ax.set_xlabel('ΔRA [arcsec]')
ax.set_ylabel('ΔDEC [arcsec]')
ax.set_title('Sources within ±%.1f″ of (RA,DEC)=(%.5f,%.5f)' %
             (radius.value, ra_center.value, dec_center.value))
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#%%
import json

with open(data_folder / 'HST_filters_folder.json', 'r') as f:
    config = json.load(f)
    os.environ['PYSYN_CDBS'] = config['filters_folder']

from synphot import SourceSpectrum
from synphot.models import ConstFlux1D
from astropy import units as u
from synphot import Observation
import stsynphot as stsyn
from tqdm import tqdm

# Convert the magnitude dataframe to flux dataframe
df_flux = pd.DataFrame(index=df_mag.index)
MUSE_units = u.Unit("1e-20 erg / (s cm2 Angstrom)")

print('Computing photometry...\n')

# Process each filter separately
for filter_name in df_mag.columns:
    print(f'Processing targets for filter {filter_name}...')
    bp = stsyn.band(f'ACS,WFC1,{filter_name}')
    
    # Convert each magnitude to flux units
    flux_values = []
    for mag in tqdm(df_mag[filter_name]):
        if pd.isna(mag):
            flux_values.append(np.nan)
            continue
        
        sp   = SourceSpectrum(ConstFlux1D, amplitude=mag*u.ABmag)
        obs  = Observation(sp, bp)
        flux = obs.effstim(MUSE_units).value
        flux_values.append(flux * 1e20)  # Convert to true MUSE units
    
    df_flux[filter_name] = flux_values


filters_data = {}

for filter_name in df_mag.columns:
    filter_data = stsyn.band(f'ACS,WFC1,{filter_name}')
    filters_data[filter_name] = (filter_data.pivot().value / 10, filter_data.fwhm().value / 10)

#%%
data_store = {
    'Astrometry': df_sel,
    'AB magnitudes': df_mag,
    'Fluxes (MUSE units):': df_flux,
    'Filters data (pivot, FWHM) [nm]': filters_data
}

try:
    with open(fname := data_folder / 'omega_selected_srcs.pkl', 'wb') as f:
        pickle.dump(data_store, f)
        print(f'Data saved to {str(fname)}')
except Exception as e:
    print(f'Error saving data: {e}')
    
#%%
try:
    with open(fname := data_folder / 'omega_selected_srcs.pkl', 'rb') as f:
        data_store = pickle.load(f)
        print(f'Data loaded from {str(fname)}')

        # Extract data from the dictionary
        df_sel = data_store['Astrometry']
        df_mag = data_store['AB magnitudes']
        df_flux = data_store['Fluxes (MUSE units):']
        filters_data = data_store['Filters data (pivot, FWHM) [nm]']
        
except Exception as e:
    print(f'Error loading data: {e}')


#%% ========================= Load MUSE data to detect and match sources =========================
from data_processing.MUSE_data_utils import GetSpectrum, LoadCachedDataMUSE
from project_settings import device

raw_path   = MUSE_DATA_FOLDER / "omega_cluster/raw/MUSE.2020-02-24T05-16-30.566.fits.fz"
cache_path = MUSE_DATA_FOLDER / "omega_cluster/cached/DATACUBEFINALexpcombine_20200224T050448_7388e773.pickle"

spectral_cubes, spectral_info, data_cached, model_config = LoadCachedDataMUSE(raw_path, cube_path, cache_path, save_cache=True, device=device, verbose=True)   
cube_full, cube_sparse, valid_mask = spectral_cubes["cube_full"], spectral_cubes["cube_sparse"], spectral_cubes["mask"]

#%%
from managers.multisrc_manager import DetectSources

# Compute coordinates of sources in the field in arcsec relative to the center of pointing
sources = DetectSources(cube_sparse, threshold=50000, display=True, draw_win_size=21)

# Compute center of mass for valid mask assuming it's the center of the field
yy, xx = np.where(valid_mask.cpu().numpy().squeeze() > 0)
field_center    = np.stack([xx.mean(), yy.mean()])[None,...]
# field_center = np.array([cube_sparse.shape[1]/2, cube_sparse.shape[2]/2])
sources_coords  = np.stack([sources['x_peak'].values, sources['y_peak'].values], axis=1)
sources_coords -= field_center
sources_coords  = sources_coords*25 / 1000  # [pix] * [mas/pix] / 1000 [mas/asec] -> [asec]

df_MUSE_sources = pd.DataFrame({ 'x': sources_coords[:, 0], 'y': sources_coords[:, 1], })
df_MUSE_sources['flux'] = sources['peak_value']

df_HST_sources = df_sel.drop(columns = ['RA', 'DEC'])
df_HST_sources = df_HST_sources.rename(columns={'x, [asec]': 'x', 'y, [asec]': 'y'})
df_HST_sources['flux'] = df_flux.sum(axis=1)

df_HST_sources['x'] += 1.56297598 # Empirical coefficients
df_HST_sources['y'] -= 2.3518643

# Normalize fluxes
flux_hst  = df_HST_sources['flux'].values
flux_muse = df_MUSE_sources['flux'].values

# Sort the fluxes in descending order and take top N (where N is min of array lengths)
n_compare = min(len(flux_hst), len(flux_muse))
# Calculate median flux ratio between top HST and MUSE sources
flux_ratio_median = np.median(np.sort(flux_hst) [-n_compare:] / np.sort(flux_muse)[-n_compare:])
print(f"Median HST/MUSE flux ratio (for top {n_compare} sources): {flux_ratio_median:.4f}")

# Scale MUSE fluxes to match HST scale (initial approximation)
flux_hst_norm  = flux_hst  / np.max(flux_hst) / flux_ratio_median
flux_muse_norm = flux_muse / np.max(flux_hst) 

#%%
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform


def weighted_affine_fit_proximity(
    df_src: pd.DataFrame,
    df_dst: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    weight_col: str = 'flux',
    max_distance: float = None
) -> AffineTransform:

    # 1) Build a KD‐Tree on df_src coords
    src_coords = df_src[[x_col, y_col]].to_numpy()
    tree = cKDTree(src_coords)

    # 2) For each df_dst point, find nearest in df_src
    dst_coords = df_dst[[x_col, y_col]].to_numpy()
    dists, idxs = tree.query(dst_coords, k=1)

    # 3) Optionally filter by max_distance
    if max_distance is not None:
        mask = (dists <= max_distance)
    else:
        mask = np.ones(len(dists), dtype=bool)

    if mask.sum() < 3:
        raise ValueError(f"Need ≥3 matched pairs, got {mask.sum()}")

    # 4) Extract matched subsets
    matched_src = df_src.iloc[idxs[mask]].reset_index(drop=True)
    matched_dst = df_dst[mask].reset_index(drop=True)

    src_pts = matched_src[[x_col, y_col]].to_numpy()
    dst_pts = matched_dst[[x_col, y_col]].to_numpy()
    weights = matched_src[weight_col].to_numpy()

    N = len(src_pts)
    # 5) Build weighted least‐squares system A θ = b
    #    θ = [a, b, c, d, e, f]^T
    A = np.zeros((2*N, 6))
    b = np.zeros((2*N,))
    for i, ((x, y), (u, v), w) in enumerate(zip(src_pts, dst_pts, weights)):
        sw = np.sqrt(w)
        A[2*i    ] = sw * np.array([x, y, 1, 0, 0, 0])
        A[2*i + 1] = sw * np.array([0, 0, 0, x, y, 1])
        b[2*i    ] = sw * u
        b[2*i + 1] = sw * v

    # 6) Solve for θ via least‐squares
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)

    # 7) Pack into a 3x3 affine matrix
    M = np.array([
        [theta[0], theta[1], theta[2]],
        [theta[3], theta[4], theta[5]],
        [      0.,       0.,       1.]
    ])

    return AffineTransform(matrix=M)


# Compute transform, allowing only matches within 5 pixels:
tform = weighted_affine_fit_proximity(
    df_src=df_HST_sources,
    df_dst=df_MUSE_sources,
    x_col='x',
    y_col='y',
    weight_col='flux',
    max_distance=0.5
)

print("Affine matrix:\n", tform.params)

# To apply to all points in df1:
homog = np.vstack([df_HST_sources['x'], df_HST_sources['y'], np.ones(len(df_HST_sources))]).T
warped = (tform.params @ homog.T).T  # shape (n_sources, 3)

df_HST_sources_filtered = df_HST_sources.copy()
df_HST_sources_filtered['x'], df_HST_sources_filtered['y'] = warped[:,0], warped[:,1]

x_hst  = df_HST_sources_filtered['x'].to_numpy()
y_hst  = df_HST_sources_filtered['y'].to_numpy()

x_muse = df_MUSE_sources['x'].values
y_muse = df_MUSE_sources['y'].values


eps = -0.1 # margin to account for the MUSE field of view
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered['x'] <  3.75+eps]
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered['x'] > -3.75-eps]
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered['y'] <  3.75+eps]
df_HST_sources_filtered = df_HST_sources_filtered[df_HST_sources_filtered['y'] > -3.75-eps]
df_HST_sources_filtered = df_HST_sources_filtered.sort_values(by='flux', ascending=False)

x_hst_sel = df_HST_sources_filtered['x'].to_numpy()
y_hst_sel = df_HST_sources_filtered['y'].to_numpy()
flux_hst_norm_sel  = df_HST_sources_filtered['flux'].values  / np.max(flux_hst) / flux_ratio_median

#%%
# Plot the matches
plt.figure(figsize=(10, 10))

dot_scaler = 500

plt.scatter(x_hst, y_hst, s=dot_scaler*flux_hst_norm, facecolors='none', edgecolors='blue', label='HST sources')
plt.scatter(x_hst_sel, y_hst_sel, s=dot_scaler*flux_hst_norm_sel, facecolors='none', edgecolors='green', label='HST sources (selected)')
plt.scatter(x_muse, y_muse, marker='.', color='red', s=dot_scaler*flux_muse_norm, alpha=0.5, label='MUSE sources')
# Draw a square to specify MUSE NFM field of view of 7.5x7.5 arcsec^2, centered at (0,0)
plt.plot([-3.75, -3.75, 3.75, 3.75, -3.75], [-3.75, 3.75, 3.75, -3.75, -3.75], color='black', linestyle='--', label='MUSE NFM FoV')

plt.axis('equal')
plt.xlabel('ΔRA [arcsec]')
plt.ylabel('ΔDEC [arcsec]')
plt.title(f'HST sources matching with optimal transformation')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%%
#Save as CSV
for filter_name in df_mag.columns:
    df_HST_sources_filtered[filter_name] = df_flux[filter_name].loc[df_HST_sources_filtered.index]

df_HST_sources_filtered.rename(columns={'flux': 'flux (total, normalized)'}, inplace=True)
df_HST_sources_filtered.rename(columns={'x': 'x, [asec]'}, inplace=True)
df_HST_sources_filtered.rename(columns={'y': 'y, [asec]'}, inplace=True)

try:
    df_HST_sources_filtered.to_csv(fname := data_folder / 'HST_sources_in_FoV.csv', index=True, index_label='ID')
    print("DataFrame saved successfully to", fname)
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")

