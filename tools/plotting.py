import numpy as np
import torch
import matplotlib.pyplot as plt
from photutils.centroids import centroid_quadratic, centroid_com, centroid_com, centroid_quadratic
from photutils.profiles import RadialProfile
from matplotlib import cm

def wavelength_to_rgb(wavelength, gamma=0.8, show_invisible=False):
    wavelength = wavelength * 1.0 # Ensure float
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        a = 0.4
        R = a
        G = a
        B = a
        if not show_invisible:
            R = 0.0
            G = 0.0
            B = 0.0
    return (R,G,B)


def render_spectral_PSF(spectral_cube, λs):
    Rs, Gs, Bs = np.zeros_like(λs), np.zeros_like(λs), np.zeros_like(λs)

    for i,λ in enumerate(λs):
        Rs[i], Gs[i], Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

    for id in range(0, len(λs)):
        img = np.log10( 50+np.abs(spectral_cube) )
        img /= img.max()

        spectral_slice_R = img * Rs[None, None, id]
        spectral_slice_G = img * Gs[None, None, id]
        spectral_slice_B = img * Bs[None, None, id]

        aa = np.dstack([spectral_slice_R, spectral_slice_G, spectral_slice_B])
        aa = aa / aa.max()

        plt.imshow(aa)
        plt.axis('off')
        plt.title(f'λ = {λs[id]:.2f} nm')


def plot_wavelength_rgb_log(image, wavelengths=None, min_val=1e-3, max_val=1e1, title=None, show=True):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        
    wavelengths = np.asarray(wavelengths)
    rgb_weights = np.array([wavelength_to_rgb(λ, show_invisible=True) for λ in wavelengths]).T

    weighted = rgb_weights[:, :, None, None] * image[None, :, :, :]
    image_RGB = np.abs(weighted.sum(axis=1))  # shape: (3, height, width)
    image_RGB = np.moveaxis(image_RGB, 0, -1)

    log_min, log_max = np.log10(min_val), np.log10(max_val)
    image_log = np.log10(image_RGB+1e-10)

    image_clipped = np.clip(image_log, log_min, log_max)
    norm_image = (image_clipped - log_min) / (log_max - log_min)

    if show:
        plt.figure()
        plt.imshow(norm_image, origin="lower")
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    return image_RGB


def save_GIF(array, duration=1e3, scale=1, path='test.gif', colormap=cm.viridis):
    from PIL import Image
    from skimage.transform import rescale
    
    gif_anim = []
    for layer in np.rollaxis(array, 2):
        buf = layer/layer.max()
        if scale != 1.0:
            buf = rescale(buf, scale, order=0)
        gif_anim.append( Image.fromarray(np.uint8(colormap(buf)*255)) )
    gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)


def save_GIF_RGB(images_stack, duration=1e3, downscale=4, path='test.gif'):
    from PIL import Image
    from PIL.Image import Resampling
    
    gif_anim = []
    
    def remove_transparency(img, bg_colour=(255, 255, 255)):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, bg_colour + (255,))
        bg.paste(img, mask=alpha)
        return bg
    
    for layer in images_stack:
        im = Image.fromarray(np.uint8(layer*255))
        im.thumbnail((im.size[0]//downscale, im.size[1]//downscale), resample=Resampling.BICUBIC)
        gif_anim.append( remove_transparency(im) )
        gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)



def safe_centroid(data):       
    xycen = centroid_quadratic(np.abs(data))
    
    if np.any(np.isnan(xycen)): xycen = centroid_com(np.abs(data))
    if np.any(np.isnan(xycen)): xycen = np.array(data.shape)//2
    
    return xycen


def render_profile(profile, color, label, linestyle='-', linewidth=1, func=lambda x: x, ax=None):
    x = np.arange(0, profile.shape[-1])
    profile_m = np.median(profile, axis=0)
    
    p_cutoff = 68.2/2
    n_quantiles = 1
    percentiles = np.linspace(p_cutoff/2, 100-p_cutoff, n_quantiles)
    
    alpha_func = lambda x: 0.21436045 * np.exp(-0.11711102 * x)

    for p in percentiles:
        upper_bound = np.percentile(profile-profile_m, 100-p, axis=0)
        lower_bound = np.percentile(profile-profile_m, p,  axis=0)
        if ax is not None:
            ax.fill_between(x, func(profile_m)+lower_bound, func(profile_m)+upper_bound, alpha=alpha_func(n_quantiles), color=color)
        else:
            plt.plot(x, func(profile_m)+lower_bound, color=color, linestyle='--', alpha=alpha_func(n_quantiles))
    
    if not ax is None:
        ax.plot(x, func(profile_m), color=color, label=label, linestyle=linestyle, linewidth=linewidth)
    else:
        plt.plot(x, func(profile_m), color=color, label=label, linestyle=linestyle, linewidth=linewidth)


def calc_profile(data, xycen=None):
    xycen = safe_centroid(data) if xycen is None else xycen
    edge_radii = np.arange(data.shape[-1]//2)
    rp = RadialProfile(data, xycen, edge_radii)
    return rp.profile

    
def plot_radial_profiles(PSF_0,
                         PSF_1,
                         label_0 = 'PSFs #1',
                         label_1 = 'PSFs #2',
                         title   = '',
                         scale   = 'log',
                         colors  = ['tab:blue', 'tab:orange', 'tab:green'],
                         cutoff  = 20,
                         centers = None,
                         return_profiles = False,
                         ax = None,
                         linthresh = 5e-1,
                         y_min = 1e-2,
                         suppress_plot = False):
            
    def _radial_profiles(PSFs, centers=None):
        listify_PSF = lambda PSF_stack: [ x.squeeze() for x in np.split(PSF_stack, PSF_stack.shape[0], axis=0) ]
        PSFs = listify_PSF(PSFs)
        if centers is None:
            centers = [None]*len(PSFs)
        else:
            if type(centers) is not list:
                if centers.size == 2: 
                    centers = [centers] * len(PSFs)
                else:
                    centers = [centers[i,...] for i in range(len(PSFs))]

        profiles = np.vstack( [calc_profile(PSF, center) for PSF, center in zip(PSFs, centers) if not np.all(np.isnan(PSF))] )
        return profiles

    if PSF_0.ndim == 2:
        PSF_0 = PSF_0[np.newaxis, ...]
    
    if PSF_1.ndim == 2:
        PSF_1 = PSF_1[np.newaxis, ...]
    
    if centers is None:
        centers = safe_centroid(np.nanmean(PSF_0, axis=0))
        
    profis_0   = _radial_profiles( PSF_0, centers )
    profis_1   = _radial_profiles( PSF_1, centers )
    profis_err = _radial_profiles( PSF_0-PSF_1, centers )
    
    # center_0 = safe_centroid(np.abs(np.nanmean(PSF_0, axis=0)))
    # center_1 = safe_centroid(np.abs(np.nanmean(PSF_1, axis=0)))
    # center_  = np.mean([center_0, center_1], axis=0)
    
    # profis_0 = _radial_profiles( PSF_0, centers )
    # profis_1 = _radial_profiles( PSF_1, centers )
    # profis_err = _radial_profiles( PSF_0 - PSF_1, center_ )

    if not suppress_plot:
        if ax is None:
            fig = plt.figure(figsize=(6, 4), dpi=300)
            ax  = fig.gca()
    
    y_max = np.median(profis_0, axis=0).max()

    p_0 = profis_0 / y_max * 100.0
    p_1 = profis_1 / y_max * 100.0
    p_err = np.abs(profis_err / y_max * 100.0)

    if not suppress_plot:
        render_profile(p_0,   color=colors[0], label=label_0, linewidth=2, ax=ax)
        render_profile(p_1,   color=colors[1], label=label_1, linewidth=2, ax=ax)
        render_profile(p_err, color=colors[2], label='Error', linestyle='--', ax=ax)

        max_err = np.median(p_err, axis=0).max()
        ax.axhline(max_err, color='green', linestyle='-', alpha=0.5)

        y_lim = max([p_0.max(), p_1.max(), p_err.max()])
        if scale == 'log':
            x_max = cutoff
            ax.set_yscale('symlog', linthresh=linthresh)
            ax.set_ylim(y_min, y_lim)
        else:
            x_max = cutoff
            ax.set_ylim(0, y_lim)

        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, x_max)
        ax.text(x_max-16, max_err+2.5, "Max. err.: {:.1f}%".format(max_err), fontsize=12)
        ax.set_xlabel('Pixels from on-axis, [pix]')
        ax.set_ylabel('Normalized intensity, [%]')
        ax.grid()

    if return_profiles:
        return p_0, p_1, p_err
    

def plot_radial_profiles_relative(PSF_0,
                                  PSF_1,
                                #   label_0 = 'PSFs #1',
                                #   label_1 = 'PSFs #2',
                                  title   = '',
                                  scale   = 'log',
                                  colors  = ['tab:blue', 'tab:orange', 'tab:green'],
                                  cutoff  = 20,
                                  centers = None,
                                  return_profiles = False,
                                  ax = None,
                                  linthresh = 5e-1,
                                  y_min = 1e-2,
                                  suppress_plot = False):
            
    def _radial_profiles(PSFs, centers=None):
        listify_PSF = lambda PSF_stack: [ x.squeeze() for x in np.split(PSF_stack, PSF_stack.shape[0], axis=0) ]
        PSFs = listify_PSF(PSFs)
        if centers is None:
            centers = [None]*len(PSFs)
        else:
            if type(centers) is not list:
                if centers.size == 2: 
                    centers = [centers] * len(PSFs)
                else:
                    centers = [centers[i,...] for i in range(len(PSFs))]

        profiles = np.vstack( [calc_profile(PSF, center) for PSF, center in zip(PSFs, centers) if not np.all(np.isnan(PSF))] )
        return profiles

    if PSF_0.ndim == 2:
        PSF_0 = PSF_0[np.newaxis, ...]
    
    if PSF_1.ndim == 2:
        PSF_1 = PSF_1[np.newaxis, ...]
    
    if centers is None:
        centers = safe_centroid(np.nanmean(PSF_0, axis=0))
        
    # profis_0   = _radial_profiles( PSF_0, centers )
    # profis_1   = _radial_profiles( PSF_1, centers )
    profis_err = _radial_profiles( PSF_1/PSF_0, centers )
    
    # center_0 = safe_centroid(np.abs(np.nanmean(PSF_0, axis=0)))
    # center_1 = safe_centroid(np.abs(np.nanmean(PSF_1, axis=0)))
    # center_  = np.mean([center_0, center_1], axis=0)
    
    # profis_0 = _radial_profiles( PSF_0, centers )
    # profis_1 = _radial_profiles( PSF_1, centers )
    # profis_err = _radial_profiles( PSF_0 - PSF_1, center_ )

    if not suppress_plot:
        if ax is None:
            fig = plt.figure(figsize=(6, 4), dpi=300)
            ax  = fig.gca()
    
    # y_max = np.median(profis_0, axis=0).max()

    # p_0 = profis_0 / y_max * 100.0
    # p_1 = profis_1 / y_max * 100.0
    p_err = np.abs(profis_err) * 100# / y_max * 100.0)

    if not suppress_plot:
        # render_profile(p_0,   color=colors[0], label=label_0, linewidth=2, ax=ax)
        # render_profile(p_1,   color=colors[1], label=label_1, linewidth=2, ax=ax)
        render_profile(p_err, color=colors[2], label='Error', linestyle='--', ax=ax)

        max_err = np.median(p_err, axis=0).max()
        ax.axhline(max_err, color='green', linestyle='-', alpha=0.5)

        y_lim = p_err.max()
        if scale == 'log':
            x_max = cutoff
            ax.set_yscale('symlog', linthresh=linthresh)
            ax.set_ylim(y_min, y_lim)
        else:
            x_max = cutoff
            ax.set_ylim(0, y_lim)

        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, x_max)
        ax.text(x_max-16, max_err+2.5, "Max. err.: {:.1f}%".format(max_err), fontsize=12)
        ax.set_xlabel('Pixels from on-axis, [pix]')
        ax.set_ylabel('Normalized intensity, [%]')
        ax.grid()

    if return_profiles:
        return p_err
    

def draw_PSF_stack(
    PSF_in, PSF_out,
    average=False,
    scale='log',
    min_val=1e-16, max_val=1e16,
    crop=None, ax=None):
    
    from matplotlib.colors import LogNorm
    
    if PSF_in.ndim == 2:  PSF_in  = PSF_in [None, None, ...]
    if PSF_out.ndim == 2: PSF_out = PSF_out[None, None, ...]
    
    if PSF_in.ndim  == 3:  PSF_in = PSF_in [None, ...]
    if PSF_out.ndim == 3: PSF_out = PSF_out[None, ...]
    
    if isinstance(PSF_in,  np.ndarray): PSF_in  = torch.tensor(PSF_in)
    if isinstance(PSF_out, np.ndarray): PSF_out = torch.tensor(PSF_out)
    
    if crop is not None:
        if PSF_in.shape[-2] < crop or PSF_in.shape[-1] < crop:
            raise ValueError('Crop size is larger than the PSF size!')
        ROI_x = slice(PSF_in.shape[-2]//2-crop//2, PSF_in.shape[-2]//2+crop//2)
        ROI_y = slice(PSF_in.shape[-1]//2-crop//2, PSF_in.shape[-1]//2+crop//2)
    else:
        ROI_x, ROI_y = slice(None), slice(None)
    

    dPSF = (PSF_out - PSF_in).abs()
    cut = lambda x: x.abs().detach().cpu().numpy()[..., ROI_x, ROI_y] if crop is not None else x.abs().detach().cpu().numpy()
    
    if average:
        row = []
        if ax is None:
            fig = plt.figure(figsize=(6, 4), dpi=300)
            ax  = fig.gca()
            
        for wvl in range(PSF_in.shape[1]):
            row.append(
                np.hstack([cut(PSF_in[:, wvl,...].mean(dim=0)),
                           cut(PSF_out[:, wvl,...].mean(dim=0)),
                           cut(dPSF[:, wvl,...].mean(dim=0))]) )
        row  = np.vstack(row)
        if scale == 'log':
            norm = LogNorm(vmin=np.maximum(row.min(), min_val), vmax=np.minimum(row.max(), max_val))
        else:
            norm = None
        
        ax.imshow(row, norm=norm)
        # ax.set_title('Sources average')
        ax.axis('off')
        # plt.show()

    else:
        for src in range(PSF_in.shape[0]):
            if ax is None:
                fig = plt.figure(figsize=(6, 4), dpi=300)
                ax  = fig.gca()
            
            row = []
            for wvl in range(PSF_in.shape[1]):
                row.append( np.hstack([cut(PSF_in[src, wvl,...]), cut(PSF_out[src, wvl,...]), cut(dPSF[src, wvl,...])]) )
            row  = np.vstack(row)
            
            if scale == 'log':
                norm = LogNorm(vmin=np.maximum(row.min(), min_val), vmax=np.minimum(row.max(), max_val))
            else:
                norm = None
                
            ax.imshow(row, norm=norm)
            # ax.set_title('Source %d' % src)
            ax.axis('off')
            plt.show()


def hist_thresholded(
    datasets,
    threshold,
    bins=10,
    title="Multiple Histograms with Threshold (Side-by-Side)",
    xlabel="Values",
    ylabel="Percentage",
    labels=None,
    colors=None,
    alpha=0.6
):
    """
    Draws multiple normalized histograms (percentages) for the given datasets,
    grouping values above a threshold into single bars for each dataset.
    Each dataset's bars are placed side-by-side rather than overlaid.

    Parameters:
        datasets (list of array-like): A list of datasets (each dataset is a list 
                                       or array of numeric values).
        threshold (float): The threshold value.
        bins (int): Number of bins for the histograms (default: 10).
        title (str): Title of the plot (default: "Multiple Histograms with Threshold (Side-by-Side)").
        xlabel (str): Label for the x-axis (default: "Values").
        ylabel (str): Label for the y-axis (default: "Percentage").
        labels (list of str): Labels for the datasets (default: None).
        colors (list of str): List of colors for each dataset (default: None).
        alpha (float): Opacity for bars (default: 0.6).
    """
    n_datasets = len(datasets)

    # If labels are not provided, create generic labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n_datasets)]

    # If colors are not provided, pick a colormap or define a simple color cycle
    if colors is None:
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_datasets)]

    # Combine all values below threshold to get consistent bin edges
    all_below_threshold = np.concatenate(
        [np.array(d)[np.array(d) <= threshold] for d in datasets]
    )
    # Calculate reference bin_edges for the histogram
    _, bin_edges = np.histogram(all_below_threshold, bins=bins, range=(0, threshold))

    # The total width of one bin
    total_bin_width = bin_edges[1] - bin_edges[0]
    # We'll split this bin width among the datasets
    bar_width = total_bin_width / n_datasets

    plt.figure(figsize=(8, 6))

    for i, data in enumerate(datasets):
        data_array = np.array(data)

        # Split into below / above threshold
        below_thresh = data_array[data_array <= threshold]
        above_thresh = data_array[data_array > threshold]

        # Compute histogram for the below-threshold portion
        counts, _ = np.histogram(below_thresh, bins=bin_edges)
        total_count = len(data_array)
        normalized_counts = (counts / total_count) * 100

        # Offset for this dataset so bars sit side-by-side
        offset = i * bar_width

        # Plot the side-by-side bars for below-threshold data
        plt.bar(
            bin_edges[:-1] + offset,  # shift the bin edges by offset
            normalized_counts,
            width=bar_width,
            alpha=alpha,
            color=colors[i],
            edgecolor="black",
            label=labels[i] if len(above_thresh) == 0 else None,  # avoid double-labeling
            zorder=3
        )

        # Plot a single bar for values above threshold (outliers), if any
        if len(above_thresh) > 0:
            outlier_percentage = (len(above_thresh) / total_count) * 100
            # Position the outlier bar near threshold plus the offset
            plt.bar(
                threshold + offset,
                outlier_percentage,
                width=bar_width,
                alpha=alpha,
                color=colors[i],
                edgecolor="black",
                label=f"{labels[i]} (> {threshold})",
                zorder=3
            )

    # Add grid below the plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=1)

    # Adjust x limits to show the last bar properly
    plt.xlim(0, threshold + total_bin_width)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    # plt.show()
