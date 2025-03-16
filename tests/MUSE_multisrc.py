#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torchmin import minimize

from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, RectangularAperture
from sklearn.cluster import DBSCAN
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages
from tools.parameter_parser import ParameterParser
from tools.utils import plot_radial_profiles_new, draw_PSF_stack, mask_circle
from tools.config_manager import ConfigManager
from data_processing.normalizers import CreateTransformSequenceFromFile, InputsTransformer
from tqdm import tqdm
from project_globals import MUSE_DATA_FOLDER, device
from astropy.io import fits
from scipy.ndimage import binary_dilation
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

from machine_learning.MUSE_onsky_df import *

# Astrometric calibration fields
# ESOrex and MPDAF for reduction
#%%
# Load the FITS file
cube_path = MUSE_DATA_FOLDER + "wide_field/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits"
with fits.open(cube_path) as hdulist:
    header = hdulist[1].header  # Assuming data is in extension 1 (SCI)
    data = hdulist[1].data  # This loads the actual cube data if needed
    wcs = WCS(header)

# RA  = 201.696643 / [deg]  13:26:47.1 RA  (J2000) pointing           
# DEC = -47.48045  / [deg] -47:28:49.6 DEC (J2000) pointing         

ny, nx, nz = header['NAXIS2'], header['NAXIS1'], header['NAXIS3']

# Set a fixed spectral index (e.g., the first frame or middle frame)
spectral_index = nz // 2  # Middle frame, adjust as necessary

# Compute the on-sky coordinates of the four corners of the FoV
corners = [
    wcs.pixel_to_world(0, 0, spectral_index)[0],
    wcs.pixel_to_world(0, nx-1, spectral_index)[0],
    wcs.pixel_to_world(ny-1, 0, spectral_index)[0],
    wcs.pixel_to_world(ny-1, nx-1, spectral_index)[0],
]

# Calculate the center and radius of the FoV
center_ra  = sum(corner.ra.deg  for corner in corners) / 4
center_dec = sum(corner.dec.deg for corner in corners) / 4
center_coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)

# Calculate the maximum angular distance to the corners
radius = max(center_coord.separation(corner).deg for corner in corners)

# Query Simbad for objects within the FoV
custom_simbad = Simbad()
custom_simbad.TIMEOUT = 300  # Adjust timeout if necessary
result_simbad = custom_simbad.query_region(center_coord, radius=radius * u.deg)

source_coords = SkyCoord( ra=result_simbad["RA"], dec=result_simbad["DEC"], unit=(u.hourangle, u.deg) )

#%%
# Use the reference wavelength for the fixed spectral coordinate
pixel_coords = wcs.world_to_pixel_values(
    source_coords.ra.deg,  # RA in degrees
    source_coords.dec.deg, # Dec in degrees
    wcs.wcs.crval[2]       # CRVAL3: reference value for spectral axis
)

# Unpack pixel coordinates for plotting
x_pixels, y_pixels = pixel_coords[0], pixel_coords[1]  # Pixel X and Y
#%%

query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {center_coord.ra.deg}, {center_coord.dec.deg}, {(radius*u.deg).value})
) = 1
AND phot_g_mean_mag < 18
"""

job = Gaia.launch_job(query)

result_gaia = job.get_results()

#%%
gaia_coords = SkyCoord(
    ra=result_gaia['ra'],  # RA column from Gaia result
    dec=result_gaia['dec'],  # Dec column from Gaia result
    unit=(u.deg, u.deg),  # RA and Dec are in degrees
    frame='icrs'  # Gaia results are in ICRS frame
)

gaia_pixels = wcs.world_to_pixel_values(
    gaia_coords.ra.deg,  # RA in degrees
    gaia_coords.dec.deg, # Dec in degrees
    wcs.wcs.crval[2]       # CRVAL3: reference value for spectral axis
)

#%%
data_disp = np.nanmean(data, axis=0)
data_disp = np.nan_to_num(data_disp, nan=0)

# Plot the image and overlay the sources
# plt.figure(figsize=(10, 10))
#%%
norm = simple_norm(data_disp, 'log', percent=100-1e-1)
plt.imshow(data_disp, origin='lower', cmap='gray', norm=norm)

plt.scatter(pixel_coords[0], pixel_coords[1], c='red', s=5, label="SIMBAD")
plt.scatter(gaia_pixels[0], gaia_pixels[1], c='blue', s=5, label="GAIA")
plt.legend()
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.title("Mapped Sources on FITS Image")
plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()


#%%
test_fits = fits.open(MUSE_DATA_FOLDER + "wide_field/cubes/DATACUBEFINALexpcombine_20200224T050448_7388e773.fits")

nan_mask = np.abs(np.nansum(test_fits[1].data, axis=0)) < 1e-12
nan_mask = binary_dilation(nan_mask, iterations=2, )
valid_mask = ~nan_mask

test_fits.close()

with open(MUSE_DATA_FOLDER + "wide_field/reduced/DATACUBEFINALexpcombine_20200224T050448_7388e773.pickle", 'rb') as f:
    data = pickle.load(f)
    
#%%
# box_size = 31  # Define the size of each ROI (in pixels)
# box_size = 61  # Define the size of each ROI (in pixels)
box_size = 111  # Define the size of each ROI (in pixels)

data_onsky, var_mask, norms, bgs = LoadImages(data, device=device, subtract_background=False, normalize=False, convert_images=True)
data_onsky = data_onsky.squeeze()
data_onsky *= torch.tensor(valid_mask, device=device).float().unsqueeze(0)

#%%
data_src = data_onsky[-1,:,:].cpu().numpy()
mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)

# thres = 75000
# thres = 65000
thres = 50000
# thres = 45000

sources = find_peaks(data_src, threshold=thres, box_size=11)

def merge_sources(sources, eps=2):
    positions = np.transpose((sources['x_peak'], sources['y_peak']))
    
    db = DBSCAN(eps=eps, min_samples=1).fit(positions)

    unique_labels = set(db.labels_)
    merged_positions = np.array([ positions[db.labels_ == label].mean(axis=0) for label in unique_labels ])
    merged_fluxes    = np.array([ data_src[int(pos[1]), int(pos[0])] for pos in merged_positions ])

    merged_sources = pd.DataFrame(merged_positions, columns=['x_peak', 'y_peak'])
    merged_sources['peak_value'] = merged_fluxes
    return merged_sources


sources   = merge_sources(sources, eps=2)
positions = np.transpose((sources['x_peak'], sources['y_peak']))
#%%
apertures = CircularAperture(positions, r=5)
apertures_box = RectangularAperture(positions, box_size//2, box_size//2)

norm = simple_norm(data_src, 'log', percent=99.99)
# plt.imshow(data_src, norm=norm, origin='lower')
plt.imshow(data_src*0+1, origin='lower', cmap='gray')
# apertures.plot(color='red', lw=1.5, alpha=0.75)
apertures_box.plot(color='gold', lw=1, alpha=0.45)
# plt.savefig('C:/Users/akuznets/Desktop/wide_field_results/presentation/srcs_circs.pdf', dpi=300)
plt.savefig('C:/Users/akuznets/Desktop/wide_field_results/presentation/srcs_sqrs.pdf', dpi=300)
plt.show()

# def find_sources_in_range(sources, x_range, y_range):
#     sources_in_range = sources[
#         (sources['x_peak'] >= x_range[0]) & (sources['x_peak'] <= x_range[1]) &
#         (sources['y_peak'] >= y_range[0]) & (sources['y_peak'] <= y_range[1])
#     ]
#     return sources_in_range

#%%
def extract_ROIs(image, sources, box_size=20, max_nan_fraction=0.3):   
    ROIs = []
    roi_local_coords = []  # To store the local image indexes inside NaN-padded ROI
    roi_global_coords = []  # To store the coordinates relative to the original image
    positions = np.transpose((sources['x_peak'], sources['y_peak']))

    D = image.shape[0]  # Depth dimension

    half_box = box_size // 2
    extra_pixel = box_size % 2  # 1 if box_size is odd, 0 if even

    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        
        # Calculate the boundaries, ensuring they don't exceed the image size
        x_min = x - half_box
        x_max = x + half_box + extra_pixel
        y_min = y - half_box
        y_max = y + half_box + extra_pixel
        
        # Extract the ROI with NaN-padding if the ROI goes outside the image bounds
        roi = torch.full((D, box_size, box_size), float('nan'), device=image.device)  # Create a blank 3D box filled with NaNs
        
        # Calculate the actual overlapping region between image and ROI
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, image.shape[-1])
        y_max_img = min(y_max, image.shape[-2])

        # Determine where the image will go in the NaN-padded ROI
        x_min_roi = max(0, -x_min)
        y_min_roi = max(0, -y_min)
        x_max_roi = x_min_roi + (x_max_img - x_min_img)
        y_max_roi = y_min_roi + (y_max_img - y_min_img)

        # Copy image data into the padded ROI
        roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi] = image[:, y_min_img:y_max_img, x_min_img:x_max_img]
        
        # Filter out ROIs with too many NaNs
        nan_fraction = torch.isnan(roi).sum().item() / roi.numel()
        if nan_fraction <= max_nan_fraction:
            ROIs.append(roi)
            # Store the local coordinates where the actual image data is inside the NaN-padded ROI
            roi_local_coords.append(((y_min_roi, y_max_roi), (x_min_roi, x_max_roi)))
            # Store the global coordinates relative to the original image
            roi_global_coords.append(((y_min_img, y_max_img), (x_min_img, x_max_img)))
    
    return ROIs, roi_local_coords, roi_global_coords


def add_ROIs(image, ROIs, local_coords, global_coords):    
    for roi, local_idx, global_idx in zip(ROIs, local_coords, global_coords):
        (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
        (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx

        image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    
    return image


def add_ROI(image, ROI, local_coord, global_coord):    
    (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coord
    (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coord
    image[:, y_min_img:y_max_img, x_min_img:x_max_img] += ROI[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    
    return image


def plot_ROIs_as_grid(ROIs, cols=5):
    """Display the ROIs in a grid of subplots."""
    n_ROIs = len(ROIs)
    rows = (n_ROIs + cols - 1) // cols  # Calculate number of rows needed
    _, axes = plt.subplots(rows, cols) #, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, roi in enumerate(ROIs):
        ax = axes[i]
        roi_ = roi.cpu() if roi.ndim == 2 else roi.sum(dim=0).cpu()  # Remove the channel dimension if it exists
        norm = simple_norm(roi_, 'log', percent=100-1e-1)
        ax.imshow(roi_, origin='lower', cmap='gray', norm=norm)
        # ax.set_title(f'Source {i+1}')
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # plt.tight_layout()
    plt.show()


ROIs, local_coords, global_coords = extract_ROIs(data_onsky, sources, box_size=box_size)

# plot_ROIs_as_grid(ROIs, cols=np.ceil(np.sqrt(len(ROIs))).astype('uint'))  # Adjust the number of columns as needed

#%%
with open(MUSE_DATA_FOLDER+'muse_df_norm_imputed.pickle', 'rb') as handle:
    muse_df_norm = pickle.load(handle)

df = data['All data']
df['ID'] = 0
df.loc[0, 'Pupil angle'] = 0.0

df_pruned  = prune_columns(df.copy())
df_reduced = reduce_columns(df_pruned.copy())
df_transforms = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_norm = normalize_df(df_reduced, df_transforms)
df_norm = df_norm.fillna(0)

selected_entries_input = muse_df_norm.columns.values.tolist()

NN_inp = torch.tensor(df_norm[selected_entries_input].loc[0].to_numpy()).to(device).float().unsqueeze(0)

#%%
from PSF_models.TipToy_MUSE_multisrc import TipTorch
# from PSF_models.TipTorch import TipTorch_new
# from tools.utils import SausageFeature

config_file, data_onsky = GetConfig(data, data_onsky)
data_onsky = data_onsky.squeeze()

config_file['sensor_science']['FieldOfView'] = 511

wavelength = config_file['sources_science']['Wavelength'].clone()

ids_wavelength_selected = np.arange(0, wavelength.shape[-1], 2)
wavelength_selected = wavelength[..., ids_wavelength_selected]
config_file['sources_science']['Wavelength'] = wavelength_selected

N_wvl = len(ids_wavelength_selected)

data_onsky_sparse = data_onsky.clone()[ids_wavelength_selected,...]


#%%
Moffat_absorber = True
predict_Moffat = Moffat_absorber

toy = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
# toy = TipTorch_new(config_file, 'sum', device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)

toy.PSD_include['fitting'] = True
toy.PSD_include['WFS noise'] = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing'] = False
toy.PSD_include['chromatism'] = True
# toy.PSD_include['diff. refract'] = False
toy.PSD_include['Moffat'] = Moffat_absorber

toy.to_float()

inputs_tiptorch = {
    # 'r0':  torch.tensor([0.09561153075597545], device=toy.device),
    'F':   torch.tensor([[1.0,]*N_wvl], device=toy.device),
    'dx':  torch.tensor([[0.0,]*N_wvl], device=toy.device),
    'dy':  torch.tensor([[0.0,]*N_wvl], device=toy.device),
    # 'bg':  torch.tensor([[1e-06,]*N_wvl], device=toy.device),
    'bg':  torch.tensor([[0,]*N_wvl], device=toy.device),
    'dn':  torch.tensor([1.5], device=toy.device),
    'Jx':  torch.tensor([[10,]*N_wvl], device=toy.device),
    'Jy':  torch.tensor([[10,]*N_wvl], device=toy.device),
    # 'Jxy': torch.tensor([[45]], device=toy.device)
    'Jxy': torch.tensor([[0]], device=toy.device)
}

if Moffat_absorber:
    inputs_psfao = {
        'amp':   torch.ones (toy.N_src, device=toy.device)*0.0, # Phase PSD Moffat amplitude [rad²]
        'b':     torch.ones (toy.N_src, device=toy.device)*0.0, # Phase PSD background [rad² m²]
        'alpha': torch.ones (toy.N_src, device=toy.device)*0.1, # Phase PSD Moffat alpha [1/m]
        'beta':  torch.ones (toy.N_src, device=toy.device)*2,   # Phase PSD Moffat beta power law
        'ratio': torch.ones (toy.N_src, device=toy.device),     # Phase PSD Moffat ellipticity
        'theta': torch.zeros(toy.N_src, device=toy.device),     # Phase PSD Moffat angle
    }
else:
    inputs_psfao = {}

_ = toy(x=inputs_tiptorch | inputs_psfao)

#%%
df_transforms_onsky  = CreateTransformSequenceFromFile('../data/temp/muse_df_norm_transforms.pickle')
df_transforms_fitted = CreateTransformSequenceFromFile('../data/temp/muse_df_fitted_transforms.pickle')

transforms = {
    'r0':    df_transforms_fitted['r0'],
    'F':     df_transforms_fitted['F'],
    'bg':    df_transforms_fitted['bg'],
    'dx':    df_transforms_fitted['dx'],
    'dy':    df_transforms_fitted['dy'],
    'Jx':    df_transforms_fitted['Jx'],
    'Jy':    df_transforms_fitted['Jy'],
    'Jxy':   df_transforms_fitted['Jxy'],
    'dn':    df_transforms_fitted['dn'],
    's_pow': df_transforms_fitted['s_pow'],
    'amp':   df_transforms_fitted['amp'],
    'b':     df_transforms_fitted['b'],
    'alpha': df_transforms_fitted['alpha'],
    'beta':  df_transforms_fitted['beta'],
    'ratio': df_transforms_fitted['ratio'],
    'theta': df_transforms_fitted['theta']
}

predicted_entries  = ['r0', 'F', 'dn', 'Jx', 'Jy', 's_pow']

if predict_Moffat:
    predicted_entries += ['amp', 'b', 'alpha']


normalizer = InputsTransformer({ entry: transforms[entry] for entry in predicted_entries })

inp_dict = {
    'r0':    torch.ones ( toy.N_src, device=toy.device)*0.1,
    'F':     torch.ones ([toy.N_src, N_wvl], device=toy.device),
    'Jx':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    'Jy':    torch.ones ([toy.N_src, N_wvl], device=toy.device)*10,
    'dn':    torch.ones (toy.N_src, device=toy.device)*1.5,
    's_pow': torch.zeros(toy.N_src, device=toy.device),
    'amp':   torch.zeros(toy.N_src, device=toy.device),
    'b':     torch.zeros(toy.N_src, device=toy.device),
    'alpha': torch.ones (toy.N_src, device=toy.device)*0.1,
}


inp_dict_ = { entry: inp_dict[entry] for entry in predicted_entries if entry in inp_dict.keys() }
_ = normalizer.stack(inp_dict_, no_transform=True)

#%%
class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.25):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x
    
# Initialize the network, loss function and optimizer
net = Gnosis(NN_inp.shape[-1], normalizer.get_stacked_size(), 200, 0.1)
net.to(device)
net.float()

net.load_state_dict(torch.load('../data/weights/gnosis_MUSE_v3_7wvl_yes_Mof_no_ssg.dict', map_location=torch.device('cpu')))
net.eval()

with torch.no_grad():
    PSF_pred_big = toy(pred_inputs := normalizer.unstack(net(NN_inp))).clone()

#%%
config_file['sensor_science']['FieldOfView'] = box_size
toy.Update(reinit_grids=True, reinit_pupils=True)

with torch.no_grad():
    PSF_pred_small = toy(pred_inputs := normalizer.unstack(net(NN_inp)))

#%%
cut_middle = lambda n,m: np.s_[..., n//2-m//2 : n//2 + m//2 + m%2, n//2-m//2 : n//2 + m//2 + m%2 ]

ROI_1 = cut_middle(PSF_pred_big.shape[-2], box_size)

ratio_crop = (PSF_pred_big.amax(dim=(-2,-1))/PSF_pred_small.amax(dim=(-2,-1))).unsqueeze(-1).unsqueeze(-1)

# diffa = PSF_pred_big.clone()
# diffa[ROI_1] -= PSF_pred_small * ratio_crop
# plt.imshow(diffa.squeeze().sum(dim=0).detach().abs().cpu().numpy(), norm=LogNorm(), origin='lower')

def masked_flux_ratio(PSF, mask):
    F_norm = (PSF*mask).sum(dim=(-2,-1), keepdim=True) / PSF.sum(dim=(-2,-1), keepdim=True)
    return F_norm

core_mask = torch.tensor(mask_circle(box_size, 3)[None,None,...]).to(device).float()
# more_mask = torch.tensor(mask_circle(box_size, 8)[None,None,...]).to(device).float()

# core_mask_big = torch.tensor(mask_circle(PSF_pred_big.shape[-2], 4)[None,None,...]).to(device).float()

#%
# plt.im
# 
# show(PSF_pred.squeeze().sum(dim=0).detach().abs().cpu().numpy(), origin='lower')

#%
# max_record = sources[sources['peak_value'] == sources['peak_value'].max()]
# max_id = sources[(sources['x_peak'] == max_record['x_peak'].values[0]) & (sources['y_peak'] == max_record['y_peak'].values[0])].index[0]
# max_id = sources[sources['peak_value'] == 174350.47].index[0]

# max_id = 24
# max_id = 31
# max_id = 41
# max_id = 36
# max_id = 22

#%
# for i in range(len(ROIs)):
#     ROI_ = ROIs[i]
#     plt.imshow(ROI_.sum(dim=0).cpu().numpy(), norm=LogNorm(), origin='lower')
#     plt.axis('off')
#     plt.title(f'ROI {i}')
#     plt.show()

#%%
from data_processing.MUSE_preproc_utils import GetRadialBackround
from scipy.ndimage import gaussian_filter

transformer_dict_astrometry = {
    'dx': df_transforms_fitted['dx'],
    'dy': df_transforms_fitted['dy']
}

# ratio_core = torch.tensor([0.2823, 0.3473, 0.4190, 0.4625, 0.5025, 0.5410, 0.5407]).to(device).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
ratio_core = masked_flux_ratio(PSF_pred_small, core_mask) #* ratio_crop

transformer_dxdy = InputsTransformer(transformer_dict_astrometry)
_ = transformer_dxdy.stack({ attr: getattr(toy, attr) for attr in transformer_dict_astrometry })

expand_dxdy = lambda x_: x_.unsqueeze(0).T.repeat(1, N_wvl).flatten().unsqueeze(0)
func = lambda x_: toy(pred_inputs | transformer_dxdy.unstack(expand_dxdy(x_)))

def fit_dxdy(source_id, verbose=0):
    if verbose > 0: 
        print(f'Predicting source {source_id}...')
    
    PSF_0 = torch.nan_to_num(ROIs[source_id].clone()[ids_wavelength_selected,...].unsqueeze(0))
    F_norm = (PSF_0 * core_mask).sum(dim=(-2,-1), keepdim=True) / ratio_core
    PSF_0 /= F_norm

    dxdy_0 = torch.zeros([2], device=device).float()

    _ = func(dxdy_0)
    loss = lambda dxdy_: F.smooth_l1_loss(PSF_0, func(dxdy_), reduction='sum')*1e3
    result = minimize(loss, dxdy_0, max_iter=100, tol=1e-3, method='bfgs', disp=verbose)

    return PSF_0.clone(), func(result.x).detach().clone(), result.x.clone(), F_norm.clone()


PSFs_0, PSFs_1, fluxes, dxdys = [], [], [], []

for i in tqdm(range(len(ROIs))):
    PSF_0, PSF_1, dxdy, flux = fit_dxdy(i, verbose=0)
    PSFs_0.append(PSF_0)
    PSFs_1.append(PSF_1)
    fluxes.append(flux)
    dxdys.append(dxdy)
    
PSFs_0 = torch.vstack(PSFs_0)
PSFs_1 = torch.vstack(PSFs_1)
fluxes = torch.vstack(fluxes)
dxdys  = torch.stack(dxdys)

#%%
display_mask = mask_circle(box_size, 18)[None,...]

PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1) * display_mask
PSFs_1_white = np.mean(PSFs_1.cpu().cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_1_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)

# folder_write = 'C:/Users/akuznets/Desktop/wide_field_results/presentation/'


#%%
from tools.utils import FWHM_fitter, FitMoffat2D_astropy, FitGauss2D_astropy

FWHM_data = FWHM_fitter(PSFs_0_white[:,None,...], verbose=True)
FWHM_pred = FWHM_fitter(PSFs_1_white[:,None,...], verbose=True)

#%%
FWHMy = lambda FWHM: np.sqrt(FWHM[:,0,0]**2 + FWHM[:,0,1]**2)
x = FWHMy(FWHM_data)
y = FWHMy(FWHM_pred)

relative_err = np.median(np.abs(x-y) / x)*100
absolute_err = np.median(np.abs(x-y) * 25)
print(f'Median relative FWHM error: {relative_err:.1f}%')
print(f'Median absolute FWHM error: {absolute_err:.1f} [mas]')

#%%
composite_img = add_ROIs(
    torch.zeros([N_wvl, data_onsky.shape[-2], data_onsky.shape[-1]], device=device),
    [(PSFs_1*fluxes)[i, ...] for i in range(PSFs_1.shape[0])],
    local_coords,
    global_coords
)

norm_field = LogNorm(vmin=1e4, vmax=1e7)

diff_img = (data_onsky_sparse-composite_img) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse.abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_pred_data.pdf', dpi=300)

plt.imshow(np.maximum(composite_img.sum(dim=0).abs().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_pred_pred.pdf', dpi=300)

plt.imshow(diff_img.abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_pred_diff.pdf', dpi=300)

#%%
ROIs_1, _, _ = extract_ROIs(composite_img, sources, box_size=box_size)
PSFs_2 = torch.nan_to_num(torch.stack(ROIs_1) / fluxes * ratio_crop, nan=0.0)

PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1) * display_mask
PSFs_2_white = np.mean(PSFs_2.cpu().cpu().numpy(), axis=1)

plot_radial_profiles_new(PSFs_0_white, PSFs_2_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)

#%%
PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1)

for id in range(len(ROIs_1)):
    draw_PSF_stack(PSFs_0_white[id,...], PSFs_2_white[id,...], average=True, crop=41, min_val=5e-5)
    plt.savefig(folder_write + f'PSFs/pred/source_{id}_white.png', dpi=300)
    plt.tight_layout()

#%% ----- Fitting -----
N_src = len(ROIs)
x_size = normalizer.get_stacked_size()

x0 = normalizer.stack(pred_inputs).squeeze().clone().detach()
Fs_flat = torch.ones([N_src], device=device)
x0 = torch.cat([x0, Fs_flat, dxdys.flatten()])
# x0 = torch.cat([x0, dxdys.flatten()])
x0.requires_grad = True

empty_img = torch.zeros([N_wvl, data_onsky_sparse.shape[-2], data_onsky_sparse.shape[-1]], device=device)

# x = x0.clone()
# plt.imshow(test.squeeze().sum(dim=0).detach().abs().cpu().numpy(), origin='lower')
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)

# print( x0[N_src+x_size+i*2 : N_src+x_size+(i+1)*2] )

#%%

result = 0

def func_fit(x):
    global result
    PSFs_fit = []
    for i in range(N_src):
        dxdy_ = x[x_size+N_src+i*2 : 2*x_size+N_src+(i+1)*2]
        x_fit_dict = normalizer.unstack(x[:x_size].unsqueeze(0))
        dxdy_dict  = transformer_dxdy.unstack(expand_dxdy(dxdy_))

        inputs = x_fit_dict | dxdy_dict
        
        toy.StartTimer()
        PSFs_fit.append( toy(inputs).squeeze() * fluxes[i, ...] * x[x_size+i] )
        result = toy.EndTimer()
    
    return add_ROIs( empty_img*0.0, PSFs_fit, local_coords, global_coords )


_ = func_fit(x0)


#%%

# loss_fit_l1 = lambda x_: F.smooth_l1_loss(
#     data_onsky_sparse * wvl_weights,
#     func_fit(x_) * wvl_weights,
#     reduction='mean'
# )

# loss_fit_l2 = lambda x_: F.mse_loss(
#     data_onsky_sparse * wvl_weights,
#     func_fit(x_) * wvl_weights,
#     reduction='mean'
# )

# loss_fit = lambda x_: loss_fit_l1(x_) * 1e-3 + loss_fit_l2(x_) * 1e-7

def loss_fit(x_):
    PSFs_ = func_fit(x_)
    l1 = F.smooth_l1_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    l2 = F.mse_loss(data_onsky_sparse*wvl_weights, PSFs_*wvl_weights, reduction='mean')
    return l1 * 1e-3 + l2 * 1e-7
    
#%%
# print(func_fit(x0))
# composite_img = func_fit(x0)

result_global = minimize(loss_fit, x0, max_iter=50, tol=1e-3, method='bfgs', disp=2)

# plt.imshow(np.maximum(composite_img.sum(dim=0).abs().detach().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
# plt.savefig(folder_write + 'composite_pred.png', dpi=300)
# plt.show()

#%%
# with torch.no_grad():
composite_img_fit = func_fit(result_global.x).detach()

norm_field = LogNorm(vmin=1e4, vmax=1e7)

diff_img = (data_onsky_sparse-composite_img_fit) * torch.tensor(valid_mask[None,...], device=device).float()

plt.imshow(data_onsky_sparse.abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_fit_data.pdf', dpi=300)

plt.imshow(np.maximum(composite_img_fit.sum(dim=0).abs().cpu().numpy(), 5e3), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_fit_pred.pdf', dpi=300)

plt.imshow(diff_img.abs().sum(dim=0).cpu().numpy(), norm=norm_field, origin='lower')
# plt.show()
plt.savefig(folder_write + 'composite_fit_diff.pdf', dpi=300)

#%%
ROIs_1, _, _ = extract_ROIs(composite_img_fit, sources, box_size=box_size)
PSFs_2 = torch.nan_to_num(torch.stack(ROIs_1) / fluxes, nan=0.0)

PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1) * display_mask
# PSFs_2_white = np.mean(PSFs_2.cpu().cpu().numpy(), axis=1) #*0.9
PSFs_2_white = np.mean((PSFs_2*ratio_crop).cpu().cpu().numpy(), axis=1)#*0.9

plot_radial_profiles_new(PSFs_0_white, PSFs_2_white, 'Data', 'TipTorch', title='PSFs predicted over the field', cutoff=16, y_min=5e-1)
plt.savefig(folder_write + 'profiles_fit.pdf', dpi=300)

#%%
PSFs_0_white = np.mean(PSFs_0.cpu().cpu().numpy(), axis=1)

for id in range(len(ROIs_1)):
    draw_PSF_stack(PSFs_0_white[id,...], PSFs_2_white[id,...], average=True, crop=41, min_val=5e-5)
    plt.savefig(folder_write + f'PSFs/fitting/source_{id}_white.png', dpi=300)

#%% ==============================================================================================================================
class GaussianFitter(nn.Module):
    
    def __init__(self, num_targets, PSF_size, device):
        """
        A PyTorch module to fit 2D Gaussians to PSF images.

        Parameters:
        -----------
        num_targets : int
            The number of PSF images (targets) to fit Gaussians to.
        image_size : tuple of ints
            Size of the original image (height, width) where the PSFs will be embedded.
        PSF_size : tuple of ints
            Size of the PSF images (height, width).
        """
        super(GaussianFitter, self).__init__()
        self.num_targets = num_targets
        self.PSF_height, self.PSF_width = PSF_size

        # Learnable parameters for the Gaussians
        self.amplitudes = nn.Parameter(torch.randn(num_targets).abs()*4e3)  # Amplitudes for each Gaussian
        self.mu_x = nn.Parameter(torch.zeros(num_targets) * self.PSF_width  / 2)  # X mean for each Gaussian
        self.mu_y = nn.Parameter(torch.zeros(num_targets) * self.PSF_height / 2)  # Y mean for each Gaussian
        self.sigma_x = nn.Parameter(torch.abs(torch.ones(num_targets))*0.25)  # X std for each Gaussian
        self.sigma_y = nn.Parameter(torch.abs(torch.ones(num_targets))*0.25)  # Y std for each Gaussian

        # Create coordinate grid for PSF generation
        y = torch.linspace(-PSF_size[0]/self.num_targets, PSF_size[0]/self.num_targets, PSF_size[0], device=device)
        x = torch.linspace(-PSF_size[1]/self.num_targets, PSF_size[1]/self.num_targets, PSF_size[1], device=device)
        self.X, self.Y = torch.meshgrid(x, y)
        self.X = self.X.unsqueeze(0)  # Add batch dimension for broadcasting
        self.Y = self.Y.unsqueeze(0)


    def forward(self):
        """
        Generates a stack of 2D Gaussians based on the learned parameters.

        Returns:
        --------
        gaussians : torch.Tensor
            A stack of 2D Gaussian PSF images (num_targets, psf_height, psf_width).
        """
        # Expand parameters to match the shape of the grid
        mu_x = self.mu_x.view(-1, 1, 1)
        mu_y = self.mu_y.view(-1, 1, 1)
        sigma_x = self.sigma_x.view(-1, 1, 1)
        sigma_y = self.sigma_y.view(-1, 1, 1)
        amplitudes = self.amplitudes.view(-1, 1, 1)

        # Compute the 2D Gaussian for each target
        gaussian = amplitudes.abs() * torch.exp(
            -(((self.X - mu_x) ** 2) / (2 * sigma_x.abs() ** 2) + ((self.Y - mu_y) ** 2) / (2 * sigma_y.abs() ** 2))
        )

        return gaussian


#%%
gauss_fitter = GaussianFitter(num_targets=len(ROIs), PSF_size=(box_size, box_size), device=device)
gauss_fitter.to(device)

ROIs_torch = torch.tensor(np.stack(ROIs), dtype=torch.float32, device=device)
ROIs_torch = torch.nan_to_num(ROIs_torch, nan=0.0)

image_data_torch = torch.tensor(image_data, dtype=torch.float32, device=device)
image_data_torch = torch.nan_to_num(image_data_torch, nan=0.0)

image_blank  = torch.zeros_like(image_data_torch, device=device)
image_filled = add_ROIs(image_blank, gauss_fitter(), local_coords, global_coords)

#%%
iterations = 3000

optimizer = optim.Adam(gauss_fitter.parameters(), lr=2e-4)
losses = []

for i in range(iterations):
    optimizer.zero_grad()
    loss = F.mse_loss(
        image_data_torch,
        add_ROIs(
            torch.zeros_like(image_data_torch, device=device),
            gauss_fitter(),
            local_coords,
            global_coords
        )
    ) / image_data_torch.sum() * 1e2 + \
        (gauss_fitter.sigma_x.abs().sum()**2 + gauss_fitter.sigma_y.abs().sum()**2) * 1e-1 #+ \
        # (gauss_fitter.sigma_x.sum() - gauss_fitter.sigma_y.sum()).abs().sum()**2 * 1e-1
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if i % 10 == 0:
        print(f'Epoch {i+1}/{iterations}, Loss: {loss.item()}')

print(f'Epoch {iterations}/{iterations}, Loss: {loss.item()}')
        
#%%
result_simbad = (image_data_torch - \
    add_ROIs(
        torch.zeros_like(image_data_torch, device=device),
        gauss_fitter(), local_coords, global_coords
    )
).abs().cpu().detach().numpy()

plt.imshow(result_simbad, norm=norm)

#%%