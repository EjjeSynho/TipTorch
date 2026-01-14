#%%
from SPHERE_data_utils import SPHERE_DATA_FOLDER, InitIRDISConfig, device

import numpy as np
import torch
import re
import pickle
from tools.utils import mask_circle
from copy import deepcopy
from tools.utils import rad2mas, cropper#, gaussian_centroid
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
import warnings
from skimage.restoration import inpaint
from skimage.morphology import dilation, disk, erosion, footprint_rectangle


MAX_NDIT = 50  # Max number of DITs to process if processed separately

# delay = lambda r: (0.0017+81e-6)*r #81 microseconds is the constant SPARTA latency, 17e-4 is the imperical constant
# frame_delay = lambda r: r/1e3 * 2.3  if (r/1e3*2.3) > 1.0 else 1.0 # delay of 2.3 frames for 1000 Hz loop rate
# frame_delay = lambda r: torch.clamp(r/1e3 * 2.3, min=1.0)

STD_FOLDER  = SPHERE_DATA_FOLDER / 'standart_stars'
CUBES_CACHE = STD_FOLDER / 'cubes_cache'


#%%conda list
def SPHERE_PSF_spiders_mask(crop, thick=9):
    def draw_line(start_point, end_point, thickness=10):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (256,256), (255,255,255))
        draw = ImageDraw.Draw(image)
        draw.line([start_point, end_point], fill=(0,0,0), width=thickness)
        return np.round(np.array(image).mean(axis=2) / 256).astype('uint')

    w, h = 256, 256
    dw, dh = 7, 7
    line_mask = draw_line((w, h-dh), (w//2+dw, h//2), thick) * \
                draw_line((0, h-dh), (w//2-dw, h//2), thick) * \
                draw_line((w, dh),   (w//2+dw, h//2), thick) * \
                draw_line((0, dh),   (w//2-dw, h//2), thick)
    
    crop = slice(line_mask.shape[0]//2-crop//2, line_mask.shape[0]//2+crop//2+crop%2)
    return line_mask[crop, crop]


def erode_mask(mask):
    
    if   mask.ndim == 2: mask = mask[None, None, ...]
    elif mask.ndim == 3: mask = mask[None, ...]
    else:
        assert mask.ndim == 4, 'Mask must be 2D, 3D or 4D array!'
    
    mask_modified = np.zeros_like(mask)
    N_src, N_wvl = mask.shape[0], mask.shape[1]

    for i in range(N_src):
        for j in range(N_wvl):
            mask_layer = mask[i,j,...]
            mask_layer = dilation(
                erosion(mask_layer, footprint_rectangle((3, 3))),
                disk(3)
            )
            mask_modified[i,j,...] = mask_layer
            
    return mask_modified


def separate_background(img, mask=None):
    bkg_estimator = MedianBackground()
    bkg = Background2D(img, (25,)*2, filter_size=(3,), bkg_estimator=bkg_estimator)
    
    if mask is None:
        return bkg.background
    else:
        return inpaint.inpaint_biharmonic(bkg.background, mask)


# TODO: deprecated, remove
def GetJitter(synth_sample, synth_config):
    TT_res = synth_sample['WFS']['tip/tilt residuals']
    D = synth_config['telescope']['TelescopeDiameter']
    ang_pix = synth_sample['Detector']['psInMas'] / rad2mas
    jitter = lambda a: 2*2*a/D/ang_pix
    TT_jitter = jitter(TT_res)
    Jx = TT_jitter[:,0].std() * ang_pix * rad2mas * 2.355
    Jy = TT_jitter[:,1].std() * ang_pix * rad2mas * 2.355
    return Jx, Jy


def IRDIS_PSF_mask(img, center=(0,0)):
    N_pix          = img.shape[0]
    mask_PSF_inner = 1.0 - mask_circle(N_pix, 30, center=(0,0), centered=True)
    mask_PSF_outer = 1.0 - mask_circle(N_pix, 80, center=(0,0), centered=True)
    mask_noise     = (SPHERE_PSF_spiders_mask(N_pix, thick=12) + mask_PSF_outer) * mask_PSF_inner
    mask_noise     = np.roll(mask_noise, center[0],  axis=0)
    mask_noise     = np.roll(mask_noise, center[1], axis=1)
    return np.clip(mask_noise, 0, 1).astype(int)


def ProcessPSFCubes(data_samples, size, remove_background=True, normalize=True):
    '''
    This function processes PSF cubes: normalizes, removes background, crops to the specified size.
    It returns a list of dictionaries with processed PSF data for every sample in data_samples.
    '''
    def _process_PSF(PSF, bg_map, cropper): 
        check_center = lambda x: x if x[x.shape[-2]//2, x.shape[-1]//2] > 0 else x*-1 # for some reason, some images apperead flipped in sign
        
        PSF_subtr = check_center(PSF) - bg_map
        _, bg_median, std = sigma_clipped_stats(PSF_subtr, sigma=(N_sigma := 2))
        
        # Remove low-SNR pixels
        mask_valid_pix = np.ones_like(PSF_subtr)
        mask_valid_pix[np.abs(PSF_subtr) < bg_median + N_sigma * std] = 0.0
        
        # Center-crop to the specified size
        PSF_subtr      = PSF_subtr[cropper]
        bg_map         = bg_map[cropper]
        mask_valid_pix = mask_valid_pix[cropper]

        if normalize:
            norm_factor = np.sum(PSF_subtr * mask_valid_pix) # normalization factor
        else:
            norm_factor = 1.0
            
        return PSF_subtr, norm_factor, mask_valid_pix.astype(np.int8)
    
    
    def _process_PSF_cube(PSF_cube, bg_map, cropper):
        '''This function normalizes and removes background for every DIT in a given PSF cube'''
        N_DIT  = PSF_cube.shape[0]
        PSFs   = np.zeros_like(PSF_cube[cropper], dtype=PSF_cube.dtype)
        masks  = np.zeros_like(PSF_cube[cropper], dtype=np.int8)
        norms  = np.zeros([N_DIT], dtype=PSF_cube.dtype)
        
        for dit in range(N_DIT): 
            PSFs[dit,...], norms[dit], masks[dit,...] = _process_PSF(PSF_cube[dit,...].copy(), bg_map, cropper)
        
        return PSFs, norms, masks

    def _compute_from_cube(cube):        
        PSF_mean = cube.mean(axis=0)
        PSF_var  = cube.var (axis=0)
               
        crop     = cropper(PSF_mean, size)
        mask_PSF = 1-IRDIS_PSF_mask(PSF_mean, center=(0,0)) # cover the PSF with a wings
        
        if remove_background:
            bg_map = separate_background(PSF_mean, mask_PSF)
        else:
            bg_map = np.zeros_like(PSF_mean)
        
        PSF_mean, norm_mean, mask_mean = _process_PSF(PSF_mean, bg_map, crop)
        PSF_cube, norms,     masks     = _process_PSF_cube(cube, bg_map, crop)

        if normalize:
            PSF_mean /= norm_mean
            PSF_var   = PSF_var[crop] / norm_mean**2
            bg_map   /= norm_mean

        return PSF_mean, PSF_var, PSF_cube, norm_mean, norms, mask_mean, masks, bg_map 
    

    PSF_data = []
    
    for data_sample in data_samples:
        PSF_L_mean, PSF_L_var, PSFs_L, norm_L_mean, norms_L, mask_L_mean, masks_L, bg_map_L = _compute_from_cube(data_sample['PSF L'])
        PSF_R_mean, PSF_R_var, PSFs_R, norm_R_mean, norms_R, mask_R_mean, masks_R, bg_map_R = _compute_from_cube(data_sample['PSF R'])

        PSF_current = {
            'norm (cube)':   np.stack([norms_L,     norms_R    ], axis=-1),
            'PSF (cube)':    np.stack([PSFs_L,      PSFs_R     ], axis= 1),
            'PSF (mean)':    np.stack([PSF_L_mean,  PSF_R_mean ], axis= 0)[None, ...],
            'PSF (var)':     np.stack([PSF_L_var,   PSF_R_var  ], axis= 0)[None, ...],
            'mask (cube)':   np.stack([masks_L,     masks_R    ], axis= 1),
            'bg map (mean)': np.stack([bg_map_L,    bg_map_R   ], axis= 0)[None, ...],
            'norm (mean)':   np.array([norm_L_mean, norm_R_mean])[None, ...]
        }
        
        PSF_mask = np.stack([mask_L_mean, mask_R_mean], axis=0)[None, ...]
        PSF_mask = erode_mask(PSF_mask) # filters out noisy regions of PSF
        # The cube of masks is not processed this way
        # NOTE: this mask does not affect the normalization
    
        # Handle the dark PSF peak artefact present on some PSFs due to overexposure
        if data_sample['labels']['Central hole'] == True:
            circ_mask = 1.0 - mask_circle(PSF_mask.shape[-1], 3, centered=True)[None, None, ...]
            PSF_mask *= circ_mask
            PSF_current['mask (cube)'] *= circ_mask
        
        PSF_current['mask (mean)'] = PSF_mask
        PSF_data.append(PSF_current)
        
    return PSF_data


  
def LoadSTDStarData(
    ids,
    normalize = True,
    subtract_background = True,
    ensure_odd_pixels = True,
    device = device
):    
    """
    Preprocess SPHERE data, i.e., PSFs and configs for TipTorch.
    """	
    def _samples_from_IDS(ids):
        data_samples = []
        for id in ids:
            data_samples.append(LoadSTDStarCacheByID(id))
        return data_samples

    
    def _only_central_wvl(samples):
        if isinstance(samples, list):
            for i in range(len(samples)):
                buf = samples[i]['spectra'].copy()
                samples[i]['spectra'] = [buf['central L']*1e-9, buf['central R']*1e-9]
        else:
            buf = samples['spectra'].copy()
            samples['spectra'] = [buf['central L']*1e-9, buf['central R']*1e-9]

    def _samples_from_DITs(init_sample): #TODO: re-implement or deprectace
        data_samples1 = []
        N_DITs = init_sample['PSF L'].shape[0]
        
        if N_DITs > MAX_NDIT: 
            warnings.warn(f'Warning: {N_DITs} DITs might be too many to fit into VRAM!')

        for i in range(init_sample['PSF L'].shape[0]):
            data_samples1.append( deepcopy(init_sample) )

        for i, sample in enumerate(data_samples1):
            sample['PSF L'] = init_sample['PSF L'][i,...][None,...]
            sample['PSF R'] = init_sample['PSF R'][i,...][None,...]

        return data_samples1
    
    make_tensor  = lambda x: torch.tensor(x, device=device) if type(x) is not torch.Tensor else x

    # Load corresponding cached data samples containing PSFs and reduced telemetry data
    if not isinstance(ids, list):
        ids = [ids]
        
    # if not split_cube:
    data_samples = _samples_from_IDS(ids)
    # else:
    #     data_samples = []
    #     for id in sample_ids:
    #         data_samples += _samples_from_DITs( _samples_from_IDS([id])[0] )
            
    _only_central_wvl(data_samples) # select the central λ only for the used filter

    # Process PSF cubes
    PSF_size = 110 + int(ensure_odd_pixels)
    PSF_data = ProcessPSFCubes(data_samples, size=PSF_size, remove_background=subtract_background, normalize=normalize)

    # Processing configs
    framework = 'pytorch' if str(device).startswith('cuda') or str(device).startswith('cpu') else 'numpy' #TODO: debug it

    configs = []
    for sample in data_samples:
        config = InitIRDISConfig(sample, device=device, convert_config=False)
        config['sensor_science']['FieldOfView'] = PSF_data[0]['PSF (mean)'].shape[-1]
        configs.append(config)

    # To save memory, delete duplicating PSF data
    for sample in data_samples:
        del sample['PSF L'], sample['PSF R']
  
    if framework.lower() == 'pytorch':
        for data_record in PSF_data:
            for key in data_record.keys():
                data_record[key] = make_tensor(data_record[key])

    return PSF_data, data_samples, configs


def LoadSTDStarCacheByID(id):
    ''' Searches a specific STD star by its ID in the list of cached cubes. '''
    with open(STD_FOLDER / 'sphere_df.pickle', 'rb') as handle:
        request_df = pickle.load(handle)

    file = request_df.loc[request_df.index == id]['Filename'].values[0]
    full_filename = CUBES_CACHE / f'{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample = pickle.load(handle)
        
    with open(STD_FOLDER / 'PSF_classes.txt', 'r') as f:
        labels = f.readlines()

    labels_data = {}
    for label in labels:
       labels_data[label.strip()] = request_df.loc[request_df.index == id][label.strip()].item()
        
    data_sample['labels'] = labels_data
        
    return data_sample



def plot_sample(id):
    import matplotlib.pyplot as plt
    
    samp = LoadSTDStarCacheByID(id)

    buf = samp['spectra'].copy()
    buf = [buf['central L']*1e-9, buf['central R']*1e-9]
    samp['spectra'] = buf

    PSF_L_0 = samp['PSF L'].sum(axis=0)
    PSF_R_0 = samp['PSF R'].sum(axis=0)

    ROI = slice(PSF_L_0.shape[0]-32, PSF_L_0.shape[0]+32)
    ROI = (ROI, ROI)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.log(np.abs(PSF_L_0[ROI])), origin='lower')
    axs[1].imshow(np.log(np.abs(PSF_R_0[ROI])), origin='lower')
    axs[0].set_title(str(np.round(samp['spectra'][0]*1e9).astype('uint'))+' [nm]')
    axs[1].set_title(str(np.round(samp['spectra'][1]*1e9).astype('uint'))+' [nm]')
    for ax in axs: ax.axis('off')
    fig.suptitle(id)
    fig.tight_layout()
    plt.show()
    # plt.savefig(save_folder / f'{id}.png', dpi=300)

# %%
