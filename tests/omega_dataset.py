#%%
# The data taken from: https://zenodo.org/records/11104046

import os
import glob
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

from data_processing.MUSE_data_utils import MUSE_DATA_FOLDER
from tools.utils import Photometry

cube_path   = MUSE_DATA_FOLDER + "wide_field/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
data_folder = MUSE_DATA_FOLDER + 'OmegaCentaury_data/'

print("Loading the Astrometric part of the catalog...")
t_astro = Table.read(data_folder+"astrometry.fits")
print("Columns of astrometric catalog:")
print(t_astro.columns)
print()

print("Reading the photometric catalogs...")
phot_dict = {}
file_list = sorted(glob.glob(data_folder+"F*photometry.fits"))
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
image_path = data_folder + "F606W_stacked_image.fits"
hdu = fits.open(image_path)[0]
stacked_image = hdu.data
wcs = WCS(hdu.header)

# 2) Define your MUSE pointing center (in deg) from the cube header
cube_hdu = fits.open(cube_path)[0]
ra_center  = cube_hdu.header['RA']  * u.deg
dec_center = cube_hdu.header['DEC'] * u.deg
center_coord = SkyCoord(ra=ra_center, dec=dec_center)

# 3) Build a SkyCoord array for all your sources in df_astro
#    (assume df_astro has columns 'RA','DEC' in degrees)
coords = SkyCoord(ra  = df_astro['RA'].values*u.deg,
                  dec = df_astro['DEC'].values*u.deg)

# 4) Fast search of everything within the MUSE NFM FoV radius
radius = (7*np.sqrt(2)/2) * u.arcsec
#    search_around_sky builds an internal k-d tree and is O(n log n)
# idx_center, idx_sources, sep2d, _ = coords.search_around_sky(center_coord, radius)

sep = coords.separation(center_coord) # array of Angle objects
mask = sep < radius                   # boolean array

# 5) Extract only the matching sources
sel_idx = np.nonzero(mask)[0]                 # indices into your 2M+ catalog
df_sel  = df_astro.iloc[sel_idx].copy()
df_mag  = df_phot .loc[df_sel.index]    
  
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

# 9) Cut out a 500×500 pixel stamp (±250 px around the center)
ROI_size_px = 250
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



#%%
import json

with open(data_folder+'HST_filters_folder.json', 'r') as f:
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
        flux_values.append(flux)
    
    df_flux[filter_name] = flux_values * 1e20 # Convert to true MUSE units

#%%
filters_data = {}

for filter_name in df_mag.columns:
    filter_data = stsyn.band(f'ACS,WFC1,{filter_name}')
    filters_data[filter_name] = (filter_data.pivot().value / 10, filter_data.fwhm().value / 10)
        
    
#%%
import pickle

data_store = {
    'Astrometry': df_sel,
    'AB magnitudes': df_mag,
    'Fluxes (MUSE units):': df_flux,
    'Filters data (pivot, FWHM) [nm]': filters_data
}

try:
    with open(fname := data_folder+'omega_selected_srcs.pkl', 'wb') as f:
        pickle.dump(data_store, f)
        print(f'Data saved to {fname}')
except Exception as e:
    print(f'Error saving data: {e}')
        

#%%
from astropy.wcs.utils import fit_wcs_from_points
from astropy.io import fits

# 1) Suppose you have "muse_header.fits" (MUSE cube) and a list of matched points:
wcs_m = WCS("muse_header.fits")

# 2) Collect matched MUSE‐pixel coords & matched HST sky coords:
pix_muse = np.vstack([x_muse, y_muse]).T    # shape (N_match, 2)
sky_hst   = hst_match                         # a SkyCoord array (N_match,)

# 3) Fit a new WCS that maps pix_muse → sky_hst:
new_wcs = fit_wcs_from_points(pix_muse, sky_hst, projection='TAN')
# "new_wcs" is now the corrected WCS for the MUSE data that 
# lines up exactly with the HST frame, within your fit residuals.

#%%

from astropy.modeling import models, fitting

# Build a 2D “Affine” model with six free parameters:
affine_model = models.AffineTransformation2D()

# Prepare data arrays (as floats):
xy_muse = np.column_stack([x_muse_flat, y_muse_flat])  # shape (N,2)
xy_hst  = np.column_stack([x_hst_flat,  y_hst_flat])   # shape (N,2)

fit = fitting.LinearLSQFitter()
best_fit_model = fit(affine_model, xy_muse, xy_hst)

# best_fit_model.parameters gives [a, b, c, d, t_x, t_y].
Xh, Yh = best_fit_model(Xm, Ym)

#%%

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# (a) Read your MUSE catalog (e.g. ASCII, FITS table, whatever).
#     Suppose it has columns 'RA_MUSE' and 'DEC_MUSE' in degrees:
muse_tab = Table.read("muse_sources.fits")   # or .csv, .ecsv, etc.
ra_muse  = muse_tab["RA_MUSE"]  * u.deg
dec_muse = muse_tab["DEC_MUSE"] * u.deg
sc_muse  = SkyCoord(ra_muse, dec_muse)

# (b) Read your HST catalog (with 'RA_HST', 'DEC_HST'):
hst_tab = Table.read("hst_sources.fits")
ra_hst  = hst_tab["RA_HST"]  * u.deg
dec_hst = hst_tab["DEC_HST"] * u.deg
sc_hst  = SkyCoord(ra_hst, dec_hst)


idx, d2d, _ = sc_muse.match_to_catalog_sky(sc_hst)
# idx[i] is the index in sc_hst of the *closest* HST star to sc_muse[i].
# d2d[i] is the on‐sky separation between those two.

max_sep = 2.0 * u.arcsec
good   = d2d < max_sep


idx, d2d, _ = sc_muse.match_to_catalog_sky(sc_hst)
# idx[i] is the index in sc_hst of the *closest* HST star to sc_muse[i].
# d2d[i] is the on‐sky separation between those two.

max_sep = 2.0 * u.arcsec
good   = d2d < max_sep


muse_match  = sc_muse[good]
hst_match   = sc_hst[idx[good]]



from astropy.wcs import WCS

# Suppose you have the WCS objects (e.g. from the header of each cube or image):
wcs_m = WCS("muse_header.fits")
wcs_h = WCS("hst_header.fits")

# (a) Convert matched MUSE sky coords → MUSE pixel coords:
x_muse, y_muse = wcs_m.world_to_pixel(muse_match)   # units: pixels

# (b) Convert matched HST sky coords → HST pixel coords:
x_hst, y_hst = wcs_h.world_to_pixel(hst_match)



# pick a central RA/Dec somewhere near the field center:
ra0 = muse_match.ra.mean()
dec0 = muse_match.dec.mean()
center = SkyCoord(ra0, dec0)

# For each matched source, compute a small‐angle offset in arcsec:
dx_muse, dy_muse = muse_match.spherical_offsets_to(center)  # returns dDec, then dRA*cos(Dec)
dx_hst,  dy_hst  = hst_match.spherical_offsets_to(center)
# Dx = ΔRA·cos(Dec) in radians → multiply by 206265 to get arcsec, 
# same for ΔDec.
x_muse_plane = (dy_muse.to_value(u.rad)*206265,  # arcsec east‐west
                dx_muse.to_value(u.rad)*206265)  # arcsec north‐south
x_hst_plane  = (dy_hst.to_value(u.rad)*206265, 
                dx_hst.to_value(u.rad)*206265)
# Make them Nx2 arrays:
import numpy as np
pix_muse = np.vstack(x_muse_plane).T   # shape (N_good, 2)
pix_hst  = np.vstack(x_hst_plane).T
