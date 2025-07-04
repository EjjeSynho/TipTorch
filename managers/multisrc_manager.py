import numpy as np
import torch
from typing import Union
from sklearn.cluster import DBSCAN
import pandas as pd
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from astropy.visualization import LinearStretch, ImageNormalize
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.aperture import RectangularAperture
from matplotlib.colors import LogNorm
from traitlets import Instance
from tools.plotting import plot_radial_profiles


"""
This module is used to manage the multi-source simulations. It contains functions to 
- detect sources in an image,
- extract multiple source images from a single image/cube with multiple sources,
- merge multiple images into a single image,
- visualize sources.
"""

def detect_sources(data_src, threshold, box_size, eps=2, verbose=False):
    
    sources = find_peaks(data_src, threshold=threshold, box_size=box_size)
    if verbose: print(f"Detected {len(sources)} sources")

    # Helps in the case if a single source was detected as multiple peaks
    positions = np.transpose((sources['x_peak'], sources['y_peak']))
    
    db = DBSCAN(eps=eps, min_samples=1).fit(positions)

    unique_labels = set(db.labels_)
    merged_positions = np.array([ positions[db.labels_ == label].mean(axis=0) for label in unique_labels ])
    merged_fluxes    = np.array([ data_src[int(pos[1]), int(pos[0])] for pos in merged_positions ])

    merged_sources = pd.DataFrame(merged_positions, columns=['x_peak', 'y_peak'])
    merged_sources['peak_value'] = merged_fluxes

    if verbose:
        print(f"Merged to {len(merged_sources)} sources")
    
    return merged_sources


def extract_ROIs(image, sources, box_size=20, max_nan_fraction=0.3):
    torch_flag = False
    if isinstance(image, np.ndarray):
        xp = np
    # elif isinstance(image, cp.array):
    #     xp = cp
    elif isinstance(image, torch.Tensor):
        xp = torch
        torch_flag=True
    else:
        raise TypeError("Unexpected image data type")
        
    ROIs = []
    roi_local_coords  = []  # To store the local image indexes inside NaN-padded ROI
    roi_global_coords = []  # To store the coordinates relative to the original image
    positions = np.transpose((sources['x_peak'], sources['y_peak']))

    D = image.shape[0]  # Depth dimension

    half_box = box_size // 2
    extra_pixel = box_size % 2  # 1 if box_size is odd, 0 if even

    valid_ids = []

    for i,pos in enumerate(positions):
        x, y = int(pos[0]), int(pos[1])
        
        # Calculate the boundaries, ensuring they don't exceed the image size
        x_min = x - half_box
        x_max = x + half_box + extra_pixel
        y_min = y - half_box
        y_max = y + half_box + extra_pixel
        
        # Extract the ROI with NaN-padding if the ROI goes outside the image bounds
        if torch_flag:
            roi = torch.full((D, box_size, box_size), float('nan'), device=image.device)  # Create a blank 3D box filled with NaNs
        else:
            roi = xp.full((D, box_size, box_size), float('nan'))  # Create a blank 3D box filled with NaNs
        
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
        if torch_flag:
            nan_fraction = torch.isnan(roi).sum().item() / roi.numel()
        else:
            nan_fraction = xp.isnan(roi).sum() / roi.size
            
        if nan_fraction <= max_nan_fraction:
            ROIs.append(roi)
            # Store the local coordinates where the actual image data is inside the NaN-padded ROI
            roi_local_coords.append(((y_min_roi, y_max_roi), (x_min_roi, x_max_roi)))
            # Store the global coordinates relative to the original image
            roi_global_coords.append(((y_min_img, y_max_img), (x_min_img, x_max_img)))
    
            valid_ids.append(i)
    
    return ROIs, roi_local_coords, roi_global_coords, valid_ids


def DetectSources(data_cube, threshold, nsigma=3.0, display=False, draw_win_size=None, sort=True):

    data_src = data_cube.sum(dim=0).cpu().numpy() if isinstance(data_cube, torch.Tensor) else data_src.sum(axis=0)
    
    _, median, std = sigma_clipped_stats(np.nan_to_num(data_src), sigma=nsigma)
    
    if threshold is None or threshold == 'auto':
        threshold = median + nsigma * std
        if display: print(f"[auto] using threshold = median + {nsigma}·std = {threshold:.2f}")
    
    # mean, median, std = sigma_clipped_stats(data_src, sigma=3.0)
    sources_df = detect_sources(data_src, threshold=threshold, box_size=11, verbose=True)

    # Draw the detected sources
    if display and draw_win_size is not None:
        # apertures = CircularAperture(srcs_pos, r=5)
        srcs_pos  = np.transpose((sources_df['x_peak'], sources_df['y_peak']))
        # srcs_flux = sources['peak_value'].to_numpy()
        apertures_box = RectangularAperture(srcs_pos, draw_win_size, draw_win_size)
        norm_field = LogNorm(vmin=median*0.9, vmax=threshold*10) # TODO: make it more statistical

        plt.imshow(np.abs(data_src), norm=norm_field, origin='lower', cmap='gray')
        apertures_box.plot(color='gold', lw=2, alpha=0.45)
        plt.show()

    # Sort in descending order of flux (peak value)
    if sort:
        sources_df = sources_df.sort_values(by='peak_value', ascending=False).reset_index(drop=True)

    return sources_df


def ExtractSources(data_cube, srcs_coords, box_size, filter_sources=True, debug_draw=False):
    ROIs, local_coords, global_coords, valid_srcs = extract_ROIs(data_cube, srcs_coords, box_size=box_size)
    # sources_valid = srcs_coords.iloc[valid_srcs].reset_index(drop=True) if filter_sources else srcs_coords
    sources_valid = srcs_coords.iloc[valid_srcs] if filter_sources else srcs_coords

    if debug_draw:
        N_cols = min(8, int(np.ceil(np.sqrt(len(ROIs))))) # Automatically adjusts the number of displayed  columns
        plot_ROIs_as_grid(ROIs, cols=N_cols)

    return {
        "images": ROIs,
        "coords": sources_valid,
        "count": len(sources_valid),
        "img_slices": global_coords,
        "img_crops": local_coords
    }


def extract_ROIs_from_coords(image, roi_local_coords, roi_global_coords, PSF_size):
    torch_flag = False
    if isinstance(image, np.ndarray):
        xp = np
    elif isinstance(image, torch.Tensor):
        xp = torch
        torch_flag = True
    else:
        raise TypeError("Unexpected image data type")
        
    ROIs = []
    D = image.shape[0]  # Depth dimension

    for local_coord, global_coord in zip(roi_local_coords, roi_global_coords):
        (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coord
        (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coord

        # Create a blank 3D box filled with zeros
        if torch_flag:
            roi = torch.zeros((D, PSF_size, PSF_size), dtype=image.dtype, device=image.device)
        else:
            roi = xp.full((D, PSF_size, PSF_size), dtype=image.dtype, device=image.device)
        
        roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi] = image[:, y_min_img:y_max_img, x_min_img:x_max_img]
        ROIs.append(roi)

    return ROIs


def add_ROIs(image, ROIs, local_coords, global_coords):    
    # If ROIs is a torch tensor with ROIs in the 0th dimension
    if isinstance(ROIs, torch.Tensor) and ROIs.ndim == 4:  # [num_rois, channels, height, width]
        for i in range(ROIs.shape[0]):
            roi = ROIs[i]
        
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coords[i]
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coords[i]

            image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:       
        # Original implementation for list of ROIs
        for roi, local_idx, global_idx in zip(ROIs, local_coords, global_coords):
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx
    
            image[:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    return image


def add_ROIs_separately(image, ROIs, local_coords, global_coords):
    # If ROIs is a torch tensor with ROIs in the 0th dimension
    
    # Repeat image for each ROI
    if isinstance(image, torch.Tensor):
        images = [image.clone() for _ in range(len(ROIs))]
    else:
        images = [image.copy() for _ in range(len(ROIs))]
    
    if isinstance(ROIs, torch.Tensor) and ROIs.ndim == 4:  # [num_rois, channels, height, width]
        for i in range(ROIs.shape[0]):
            roi = ROIs[i]
        
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_coords[i]
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_coords[i]

            images[i][:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    else:
        # Original implementation for list of ROIs
        for i, (roi, local_idx, global_idx) in enumerate(zip(ROIs, local_coords, global_coords)):
            (y_min_roi, y_max_roi), (x_min_roi, x_max_roi) = local_idx
            (y_min_img, y_max_img), (x_min_img, x_max_img) = global_idx
    
            images[i][:, y_min_img:y_max_img, x_min_img:x_max_img] += roi[:, y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    return torch.stack(images, dim=0)


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
    
    

def VisualizeSources(data, model, norm=None, mask=1.0, ROI=None, show=True):
    """
    Visualize source data, model, and their difference.

    Parameters:
        data (torch.Tensor or np.ndarray): Source data
        model (torch.Tensor or np.ndarray): Model data
        norm (ImageNormalize, optional): Normalization for visualization
        mask (float or array-like): Mask to apply to the difference
        ROI (slice, optional): Region of interest to visualize
    """
    def process_array(x):
        # Convert to numpy and sum along first dimension if needed
        if isinstance(x, torch.Tensor):
            x = x.sum(dim=0).abs().cpu().numpy()
        else:
            x = np.abs(x.sum(axis=0))

        # Apply ROI if specified
        return np.nan_to_num(x[ROI]) if ROI is not None else np.nan_to_num(x)

    # Process input arrays
    data_vis  = process_array(data)
    model_vis = process_array(model)

    # Handle mask type conversion
    if not isinstance(mask, float):
        mask = torch.as_tensor(mask, dtype=data.dtype, device=data.device) if isinstance(data, torch.Tensor) else \
               mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    # Calculate difference
    diff_vis = process_array((data - model) * mask)

    # Create default normalization if none provided
    if norm is None:
        vmin = min(data_vis.min(), model_vis.min(), diff_vis.min())
        vmax = max(data_vis.max(), model_vis.max(), diff_vis.max())
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    # Plot all three images
    titles = ['Data', 'Model', 'Difference']
    images = [data_vis, model_vis, diff_vis]

    for img, title in zip(images, titles):
        plt.imshow(img, norm=norm, origin='lower')
        plt.title(title)
        plt.axis('off')
        if show: plt.show()

    return diff_vis
    
    
def PlotSourcesProfiles(data, model, sources, radius, title=''):

    ROIs_0, _, _, _ = extract_ROIs(data,  sources, box_size=np.round(radius*2+4).astype('int'))
    ROIs_1, _, _, _ = extract_ROIs(model, sources, box_size=np.round(radius*2+4).astype('int'))
    
    if isinstance(data, torch.Tensor):
        PSFs_0_white = torch.stack(ROIs_0).mean(dim=1).cpu().numpy()
    else:
        PSFs_0_white = np.mean(np.stack(ROIs_0), axis=1)
    
    if isinstance(model, torch.Tensor):
        PSFs_1_white = torch.stack(ROIs_1).mean(dim=1).cpu().numpy()
    else:
        PSFs_1_white = np.mean(np.stack(ROIs_1), axis=1)
    
    plot_radial_profiles(PSFs_0_white, PSFs_1_white, 'Data', 'Model', title=title, cutoff=radius, y_min=5e-2)
    plt.show()


def select_sources(src_dict: dict, selected_ids: Union[list, int]) -> dict:
    if isinstance(selected_ids, int):
        selected_ids = [selected_ids]
    result_dict = {}
    for key, tensor in src_dict.items():
        if hasattr(tensor, 'shape') and tensor.shape[0] > 1:
            result_dict[key] = tensor[selected_ids]
        else:
            result_dict[key] = tensor
    return result_dict
