#%%
import numpy as np
import os
import torch
import gdown
from torch import nn
from scipy.ndimage import center_of_mass
from math import prod
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import seaborn as sns
from project_settings import xp, use_cupy

try:
    from graphviz import Digraph
except:
    class Digraph:
        def __init__(self, *args, **kwargs):
            print("Warning: graphviz is not installed. Graph visualization will not be available.")
            pass   
        def node(self, *args, **kwargs): pass
        def edge(self, *args, **kwargs): pass


rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
r0 = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]


def DownloadFromDrive(share_url, output_path, overwrite=False, verbose=False):
    """
    Downloads a file from Google Drive using a shareable link.

    Parameters:
        share_url (str): URL to the shared file on Google Drive
        output_path (str): Path where the file should be saved
        overwrite (bool): If True, overwrites the file if it already exists.
                          If False, skips download if file exists. Default is False.
    """
    # Check if the file exists and handle based on overwrite flag
    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists at {output_path}. Set overwrite=True to replace it.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create file's directory if it doesn't exist
    gdown.download(share_url, output_path, quiet=not verbose, fuzzy=True) # Download the file


def check_framework(x):
    # Determine whether an array is NumPy or CuPy.
    
    # Get the module from the array's class
    if hasattr(x, '__module__'):
        module_name = x.__module__.split('.')[0]
        if   module_name == 'numpy': return np
        elif module_name == 'cupy':  return xp

    # Default to NumPy if not using GPU, otherwise CuPy
    return np if not use_cupy else xp


def to_little_endian(array):
    # Determine the current data type and convert to little-endian if necessary
    type_map = {
        '>f8': '<f8',  # Convert from big-endian double to little-endian double
        '>f4': '<f4'   # Convert from big-endian float to little-endian float
    }
    # Check if the current data type needs conversion
    new_type = type_map.get(array.dtype.str)
    if new_type:
        return array.astype(new_type)
    return array


def corr_plot(data, entry_x, entry_y, lims=None, title=None):
    j = sns.jointplot(data=data,   x=entry_x, y=entry_y, kind="kde", space=0, alpha = 0.8, fill=True, colormap='royalblue' )
    i = sns.scatterplot(data=data, x=entry_x, y=entry_y, alpha=0.5, ax=j.ax_joint, color = 'black', s=10)
    sns.set_style("darkgrid")
    
    j.ax_joint.set_aspect('equal')
    
    if lims is None:
        lims = [np.min([j.ax_joint.get_xlim(), j.ax_joint.get_ylim()]),
                np.max([j.ax_joint.get_xlim(), j.ax_joint.get_ylim()])]

    if title is not None:
        j.ax_joint.set_title(title, fontsize=16)
        
        plt.title(title, fontsize=16)
    
    j.ax_joint.set_xlim(lims)
    j.ax_joint.set_ylim(lims)
    j.ax_joint.plot([lims[0], lims[1]], [lims[0], lims[1]], 'gray', linewidth=1.5, linestyle='--')
    
    j.ax_joint.grid(True)


def compare_dicts(d1, d2, path=''):
    for k in d1.keys():
        new_path = f"{path}.{k}" if path else k
        if k in d2.keys():
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                # If both values are dictionaries, compare recursively
                compare_dicts(d1[k], d2[k], new_path)
            elif isinstance(d1[k], torch.Tensor) and isinstance(d2[k], torch.Tensor):
                if not torch.allclose(d1[k], d2[k]):
                    print(f"{new_path} not equal")
            else:
                # Handle non-dict, non-tensor comparisons or mismatched types
                if d1[k] != d2[k]:
                    print(f"{new_path} not equal")
        else:
            print(f"{new_path} not in d2")

    for k in d2.keys():
        new_path = f"{path}.{k}" if path else k
        if k not in d1.keys():
            print(f"{new_path} not in d1")


def pad_lists(input_list, pad_value):
    # Find the length of the longest list
    max_len = max(len(x) if isinstance(x, list) else 1 for x in input_list)

    if max_len == 1:
        return input_list

    # Function to pad a single element (either a number or a list)
    def pad_element(element):
        if isinstance(element, list):
            return element + [pad_value] * (max_len - len(element))
        else:
            return [element] + [pad_value] * (max_len - 1)

    # Apply padding to each element in the input list
    return [pad_element(x) for x in input_list]


def cropper(x, win, center=None):
    if center is None:
        return np.s_[...,
                    x.shape[-2]//2-win//2 : x.shape[-2]//2 + win//2 + win%2,
                    x.shape[-1]//2-win//2 : x.shape[-1]//2 + win//2 + win%2]
    else:        
        return np.s_[...,
                np.round(center[0]).astype(int)-win//2 : np.round(center[0]).astype(int) + win//2 + win%2,
                np.round(center[1]).astype(int)-win//2 : np.round(center[1]).astype(int) + win//2 + win%2]


def gaussian_centroid(img):
    x_, y_ = safe_centroid(img)
    try:
        image_cropped = img[cropper(img, 40, (int(x_), int(y_)))]
        x_0, y_0 = safe_centroid(image_cropped)

        size = image_cropped.shape[0]
        y, x = np.mgrid[:size, :size]
        fitter = fitting.LevMarLSQFitter()

        gaussian_init = models.Gaussian2D(amplitude=np.max(image_cropped), x_mean=x_0, y_mean=y_0, x_stddev=3, y_stddev=3, theta=0)
        fitted_gaussian = fitter(gaussian_init, x, y, image_cropped)
        # gaussian_image  = fitted_gaussian(x, y)
        return fitted_gaussian.x_mean.value - size//2 + x_, fitted_gaussian.y_mean.value - size//2 + y_
    except:
        return x_, y_

    


def mask_circle(N, r, center=(0,0), centered=True):
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


def CroppedROI(im, point, win):
    return ( slice(max(point[0]-win//2, 0), min(point[0]+win//2+win%2, im.shape[0])),
             slice(max(point[1]-win//2, 0), min(point[1]+win//2+win%2, im.shape[1])) )


def GetROIaroundMax(im, win=200):
    im[np.isinf(im)] = np.nan
    # determine the position of maximum intensity, so the image is centered around the brightest star
    max_id = np.unravel_index(np.nanargmax(im), im.shape)
    # make it more correct with the center of mass
    max_crop = CroppedROI(im, max_id, 20)
    CoG_id = np.array(center_of_mass(np.nan_to_num(im[max_crop]))).round().astype(np.int32)
    max_id = (max_crop[0].start + CoG_id[0], max_crop[1].start + CoG_id[1])
    ids = CroppedROI(im, max_id, win)
    return im[ids], ids, max_id


def GetJmag(N_ph):
    J_zero_point = 1.9e12
    return -2.5 * np.log10(368 * N_ph / J_zero_point)


# Adds singleton dimensions to the tensor. If negative, dimensions are added in the beginning, else in the end
def pdims(x, ns):
    expdims = lambda x, n: x.view(*x.shape, *[1 for _ in range(n)]) if n>0 else x.view(*[1 for _ in range(abs(n))], *x.shape)

    if hasattr(ns, "__len__"):
        for n in ns:
            x = expdims(x, n)
        return x
    else:
        return expdims(x, ns)

min_2d = lambda x: x if x.dim() == 2 else x.unsqueeze(1)


# Computes Strehl ratio
def SR(PSF, PSF_DL):
    ratio = torch.amax(PSF.abs(), dim=(-2,-1)) / torch.amax(PSF_DL, dim=(-2,-1)) * PSF_DL.sum(dim=(-2,-1)) / PSF.abs().sum(dim=(-2,-1)) 
    if ratio.squeeze().dim() == 0:
        return ratio.item()
    else:
        return ratio.squeeze()
    

class Photometry:
    def __init__(self):
        self.bands = {
            'U':   (0.360e-6,  0.070e-6, 2.0e12),
            'B':   (0.440e-6,  0.100e-6, 5.4e12),
            'V0':  (0.500e-6,  0.090e-6, 3.3e12),
            'V':   (0.550e-6,  0.090e-6, 3.3e12),
            'R':   (0.640e-6,  0.150e-6, 4.0e12),
            'I':   (0.790e-6,  0.150e-6, 2.7e12),
            'I1':  (0.700e-6,  0.033e-6, 2.7e12),
            'I2':  (0.750e-6,  0.033e-6, 2.7e12),
            'I3':  (0.800e-6,  0.033e-6, 2.7e12),
            'I4':  (0.700e-6,  0.100e-6, 2.7e12),
            'I5':  (0.850e-6,  0.100e-6, 2.7e12),
            'I6':  (1.000e-6,  0.100e-6, 2.7e12),
            'I7':  (0.850e-6,  0.300e-6, 2.7e12),
            'R2':  (0.650e-6,  0.300e-6, 7.92e12),
            'R3':  (0.600e-6,  0.300e-6, 7.92e12),
            'R4':  (0.670e-6,  0.300e-6, 7.92e12),
            'I8':  (0.750e-6,  0.100e-6, 2.7e12),
            'I9':  (0.850e-6,  0.300e-6, 7.36e12),
            'J':   (1.215e-6,  0.260e-6, 1.9e12),
            'H':   (1.654e-6,  0.290e-6, 1.1e12),
            'Kp':  (2.1245e-6, 0.351e-6, 6e11),
            'Ks':  (2.157e-6,  0.320e-6, 5.5e11),
            'K':   (2.179e-6,  0.410e-6, 7.0e11),
            'L':   (3.547e-6,  0.570e-6, 2.5e11),
            'M':   (4.769e-6,  0.450e-6, 8.4e10),
            'Na':  (0.589e-6,  0, 3.3e12),
            'EOS': (1.064e-6,  0, 3.3e12)
        }
        self.__wavelengths = np.array([v[0] for v in self.bands.values()])

    def FluxFromMag(self, mag, band):
        return self.__PhotometricParameters(band)[2] / 368 * 10 ** (-0.4 * mag)

    def PhotonsFromMag(self, area, mag, band, sampling_time):
        return self.FluxFromMag(mag, band) * area * sampling_time

    def MagFromPhotons(self, area, Nph, band, sampling_time):
        photons = Nph / area / sampling_time
        return -2.5 * np.log10(368 * photons / self.__PhotometricParameters(band)[2])

    def __PhotometricParameters(self, inp):
        if isinstance(inp, str):
            if inp in self.bands:
                return self.bands[inp]
            raise ValueError(f'Error: there is no band with the name "{inp}"')

        if isinstance(inp, float):
            if not (self.__wavelengths.min() <= inp <= self.__wavelengths.max()):
                raise ValueError('Error: specified value is outside the defined wavelength range!')
            closest_indices = np.argsort(np.abs(self.__wavelengths - inp))[:2]
            l1, l2 = sorted(self.__wavelengths[closest_indices])
            p1, p2 = (self.bands[k] for k, v in self.bands.items() if v[0] in (l1, l2))
            weight = (inp - l1) / (l2 - l1)
            return weight * (np.array(p2) - np.array(p1)) + np.array(p1)

        raise ValueError(f'Incorrect input: "{inp}"')


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}
    def hook_c_b(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_c_b)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                def grad_ord(x):
                    mins = ""
                    maxs = ""
                    y = [buf for buf in x if buf is not None]
                    for buf in y:
                        min_buf = torch.abs(buf).min().cpu().numpy().item()
                        max_buf = torch.abs(buf).max().cpu().numpy().item()

                        if min_buf < 0.1 or min_buf > 99:
                            mins += "{:.1e}".format(min_buf) + ', '
                        else:
                            mins += str(np.round(min_buf,1)) + ', '
                        if max_buf < 0.1 or max_buf > 99:
                            maxs += "{:.1e}".format(max_buf) + ', '
                        else:
                            maxs += str(np.round(max_buf,1)) + ', '
                    return mins[:-2] + ' | ' + maxs[:-2]

                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__)+'\n'+grad_ord(fn_dict[fn]), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)
        return dot

    return make_dot


class EarlyStopping:
    def __init__(self, patience=2, tolerance=1e-1, relative=False):
        self.__patience  = patience
        self.__tolerance = tolerance
        self.__previous_loss = 1e16
        self.__counter = 0
        self.__relative = relative
        self.stop = False

    def __compare(self, a, b):
        return abs(a/b-1) < self.__tolerance if self.__relative else abs(a-b) < self.__tolerance

    def __call__(self, current_loss):
        if self.__compare(self.__previous_loss, current_loss.item()):
            self.__counter += 1 
            self.stop = True if self.__counter >= self.__patience else False
        else: self.__counter = 0
        self.__previous_loss = current_loss.item()


'''
def FitGauss2D(PSF):
    nPix_crop = 16
    crop = slice(PSF.shape[0]//2-nPix_crop//2, PSF.shape[0]//2+nPix_crop//2)
    PSF_cropped = torch.tensor(PSF[crop,crop], requires_grad=False, device=PSF.device)
    PSF_cropped = PSF_cropped / PSF_cropped.max()

    px, py = torch.meshgrid(
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=PSF.device),
        torch.linspace(-nPix_crop/2, nPix_crop/2-1, nPix_crop, device=PSF.device),
        indexing = 'ij')

    def Gauss2D(X):
        return X[0]*torch.exp( -((px-X[1])/(2*X[3]))**2 - ((py-X[2])/(2*X[4]))**2 )

    X0 = torch.tensor([1.0, 0.0, 0.0, 1.1, 1.1], requires_grad=True, device=PSF.device)

    loss_fn = nn.MSELoss()
    optimizer = optim.LBFGS([X0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(Gauss2D(X0), PSF_cropped)
        loss.backward()
        optimizer.step(lambda: loss_fn(Gauss2D(X0), PSF_cropped))

    FWHM = lambda x: 2*np.sqrt(2*np.log(2)) * np.abs(x)

    return FWHM(X0[3].detach().cpu().numpy()), FWHM(X0[4].detach().cpu().numpy())
'''

def FitMoffat2D_astropy(PSF):
    nPix_crop = 22  # Can be even or odd
    N = PSF.shape[0]
    center = N // 2

    # Determine start and end indices for cropping
    half_crop = nPix_crop // 2
    if nPix_crop % 2 == 0:
        # Even number of pixels
        start = center - half_crop
        end = center + half_crop
    else:
        # Odd number of pixels
        start = center - half_crop
        end = center + half_crop + 1

    # Use np.s_ for slicing
    crop = np.s_[start:end, start:end]

    PSF_cropped = PSF[crop]
    PSF_cropped = PSF_cropped / PSF_cropped.max()

    # Create coordinate grids
    y, x = np.indices(PSF_cropped.shape)

    # Estimate initial parameters from data
    A_init = PSF_cropped.max()
    total = PSF_cropped.sum()
    x0_init = (x * PSF_cropped).sum() / total
    y0_init = (y * PSF_cropped).sum() / total
    sx_init = np.sqrt(((x - x0_init)**2 * PSF_cropped).sum() / total)
    sy_init = np.sqrt(((y - y0_init)**2 * PSF_cropped).sum() / total)
    offset_init = PSF_cropped.min()

    # For Moffat, we need gamma and alpha.
    # gamma ~ a scale radius. We'll pick something related to the average of sx and sy.
    gamma_init = 0.5 * (sx_init + sy_init)
    alpha_init = 2.5  # A reasonable default; adjust as needed.

    # Initial model: Moffat + constant offset
    moffat_init = models.Moffat2D(amplitude=A_init, x_0=x0_init, y_0=y0_init,
                                  gamma=gamma_init, alpha=alpha_init) \
                 + models.Const2D(offset_init)

    # Fit the model
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(moffat_init, x, y, PSF_cropped)

    # Extract fitted parameters for Moffat
    gamma_fit = fitted_model.gamma_0.value
    alpha_fit = fitted_model.alpha_0.value

    # Compute FWHM from Moffat parameters
    # FWHM = 2 * gamma * sqrt(2^(1/alpha) - 1)
    def moffat_FWHM(gamma, alpha):
        return 2 * gamma * np.sqrt(2**(1/alpha) - 1)

    FWHM_x = moffat_FWHM(gamma_fit, alpha_fit)
    # For a perfectly circular Moffat, FWHM doesn't differ in x and y.
    # If needed, one could attempt to get direction-dependent measures
    # but Moffat2D is symmetric in x and y by definition.
    FWHM_y = FWHM_x

    return FWHM_x, FWHM_y, fitted_model(x, y), PSF_cropped


def FitGauss2D_astropy(PSF):
    nPix_crop = 21  # Can be even or odd
    N = PSF.shape[0]
    center = N // 2

    # Determine start and end indices for cropping
    half_crop = nPix_crop // 2
    if nPix_crop % 2 == 0:
        # Even number of pixels
        start = center - half_crop
        end = center + half_crop
    else:
        # Odd number of pixels
        start = center - half_crop
        end = center + half_crop + 1

    # Use np.s_ for slicing
    crop = np.s_[start:end, start:end]

    PSF_cropped = PSF[crop]
    PSF_cropped = PSF_cropped / PSF_cropped.max()

    # Create coordinate grids
    y, x = np.indices(PSF_cropped.shape)

    # Estimate initial parameters from data
    A_init = PSF_cropped.max()
    total = PSF_cropped.sum()
    x0_init = (x * PSF_cropped).sum() / total
    y0_init = (y * PSF_cropped).sum() / total
    sx_init = np.sqrt(((x - x0_init)**2 * PSF_cropped).sum() / total)
    sy_init = np.sqrt(((y - y0_init)**2 * PSF_cropped).sum() / total)
    offset_init = PSF_cropped.min()

    # Initial model: Gaussian + constant offset
    gauss_init = models.Gaussian2D(amplitude=A_init, x_mean=x0_init, y_mean=y0_init,
                                   x_stddev=sx_init, y_stddev=sy_init) \
                 + models.Const2D(offset_init)

    # Fit the model
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(gauss_init, x, y, PSF_cropped)

    # Extract fitted parameters
    sx_fit = fitted_model.x_stddev_0.value
    sy_fit = fitted_model.y_stddev_0.value

    # Compute FWHM from the standard deviations
    FWHM = lambda sigma: 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma)

    return FWHM(sx_fit), FWHM(sy_fit), fitted_model(x, y), PSF_cropped


def FWHM_fitter(PSF_stack, function='Moffat', verbose=False):
    if verbose:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x
        
    if function == 'Moffat':
        func_ = FitMoffat2D_astropy
    else:
        func_ = FitGauss2D_astropy
        
    FWHMs = np.zeros([PSF_stack.shape[0], PSF_stack.shape[1], 2])
    for i in tqdm(range(PSF_stack.shape[0])):
        for l in range(PSF_stack.shape[1]):
            f_x, f_y, _, _ = func_(PSF_stack[i,l,:,:])
            FWHMs[i,l,0] = f_x.item()
            FWHMs[i,l,1] = f_y.item()
    return FWHMs




def PupilVLT(samples, rotation_angle=0, petal_modes=False, vangle=[0,0], one_pixel_pad=True):
    secondary_diameter = 1.12
    pupil_diameter = 8.0
    spider_width = 0.039
    alpha = 101.4

    # Calculate shift of the obscuration
    rad_vangle = np.deg2rad(vangle[0] / 60) # Convert arcmin to radians
    sh_x = np.cos(np.deg2rad(vangle[1])) * 101.4 * np.tan(rad_vangle)
    sh_y = np.sin(np.deg2rad(vangle[1])) * 101.4 * np.tan(rad_vangle)

    grid_spacing = pupil_diameter / (samples-1)  # Calculate grid spacing
    effective_spider_width = max(spider_width, grid_spacing*1.25)  # Ensure at least one pixel width

    # Create coordinate matrices
    scale_factor = 1 - 1/samples * float(not one_pixel_pad)
    grid_range_x = np.linspace(-pupil_diameter/2, pupil_diameter/2, samples)*scale_factor + sh_x # [m]
    grid_range_y = np.linspace(-pupil_diameter/2, pupil_diameter/2, samples)*scale_factor + sh_y # [m]

    x, y = np.meshgrid(grid_range_x, grid_range_y)

    # Mask for pupil and central obstruction
    mask = (np.sqrt((x-sh_x)**2 + (y-sh_y)**2) <= pupil_diameter / 2) & (np.sqrt(x**2+y**2) >= secondary_diameter*1.1 / 2)

    # Rotation function
    def rotate(x, y, angle):
        rad_angle = np.deg2rad(angle)
        x_rot = x * np.cos(rad_angle) - y * np.sin(rad_angle)
        y_rot = x * np.sin(rad_angle) + y * np.cos(rad_angle)
        return x_rot, y_rot

    # Rotate coordinates
    x_rot, y_rot = rotate(x, y, rotation_angle)

    # Function to create spider petals
    def create_petal(condition):
        petal = np.zeros_like(x_rot, dtype=bool)
        petal[condition] = True
        return petal & mask

    # Calculate spider petals with rotation
    alpha_rad = np.deg2rad(alpha)
    slope = np.tan(alpha_rad / 2)

    petal_conditions = [
        np.where(
            ((-y_rot > effective_spider_width / 2 + slope * (-x_rot - secondary_diameter / 2) + effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot < 0) & (y_rot <= 0)) | \
            ((-y_rot > effective_spider_width / 2 + slope * ( x_rot - secondary_diameter / 2) + effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot >= 0) & (y_rot <= 0))
        ),
        np.where(
            ((-y_rot < effective_spider_width / 2 + slope * ( x_rot - secondary_diameter / 2) - effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot > 0) & (y_rot <= 0)) | \
            (( y_rot < effective_spider_width / 2 + slope * ( x_rot - secondary_diameter / 2) - effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot > 0) & (y_rot > 0))
        ),
        np.where(
            (( y_rot > effective_spider_width / 2 + slope * (-x_rot - secondary_diameter / 2) + effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot <= 0) & (y_rot > 0)) | \
            (( y_rot > effective_spider_width / 2 + slope * ( x_rot - secondary_diameter / 2) + effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot > 0) & (y_rot > 0))
        ),
        np.where(
            ((-y_rot < effective_spider_width / 2 + slope * (-x_rot - secondary_diameter / 2) - effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot < 0) & (y_rot < 0)) | \
            (( y_rot < effective_spider_width / 2 + slope * (-x_rot - secondary_diameter / 2) - effective_spider_width / np.sin(alpha_rad / 2) / 2) & (x_rot < 0) & (y_rot >= 0))
        )
    ]
    petals = [create_petal(condition) for condition in petal_conditions]

    mask_spiders = sum(petals)

    if petal_modes:
        x_rot_norm = x_rot / x_rot[np.where(mask_spiders > 0.5)].std()
        y_rot_norm = y_rot / y_rot[np.where(mask_spiders > 0.5)].std()

        def make_island_TT_mode(X, petal, normalize=True):
            mode = X - X[np.where(petal > 0.5)].mean()
            if normalize:
                mode /= mode[np.where(petal > 0.5)].std()
            return mode * petal

        tips  = [make_island_TT_mode(y_rot_norm, petal, False) for petal in petals]
        tilts = [make_island_TT_mode(x_rot_norm, petal, False) for petal in petals]

        petal_modes = np.dstack([*petals, *tips, *tilts])
        return petal_modes
    else:
        return mask_spiders
    

class GradientLoss(nn.Module):
    """
    A gradient-based loss that enforces smoothness by penalizing differences
    between neighboring pixels in both x (horizontal) and y (vertical) directions.
    
    Parameters:
    - p (int or float): The norm degree. Use p=2 for L2-norm (quadratic) or p=1 for L1-norm.
    - reduction (str): Specifies the reduction to apply to the output: 'mean' or 'sum'.
    """
    def __init__(self, p=2, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input):
        """
        Compute the gradient loss on the input phase map.
        
        Args:
            input (torch.Tensor): A tensor of shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: The computed gradient loss.
        """
        # Compute differences along the horizontal (x) direction: shape [B, C, H, W-1]
        diff_x = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        # Compute differences along the vertical (y) direction: shape [B, C, H-1, W]
        diff_y = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        
        # Apply the p-norm to the differences
        if self.p == 1:
            loss_x = diff_x
            loss_y = diff_y
        else:
            loss_x = diff_x ** self.p
            loss_y = diff_y ** self.p
        
        # Sum the losses from both directions to get a scalar loss
        loss = loss_x.sum() + loss_y.sum()
        
        # Optionally, average the loss over the number of elements
        if self.reduction == 'mean':
            num_elements = loss_x.numel() + loss_y.numel()
            loss = loss / num_elements
        
        return loss


class CombinedLoss:
    def __init__(self, data, func, wvl_weights, mae_weight=2500, mse_weight=1120):
        self.data = data
        self.func = func
        self.wvl_weights = wvl_weights
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def __call__(self, x):
        diff = (self.func(x) - self.data) * self.wvl_weights
        mse_loss = (diff * self.mse_weight).pow(2).mean()
        mae_loss = (diff * self.mae_weight).abs().mean()
        return (mse_loss + mae_loss)
    