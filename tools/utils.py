#%%
import time
import numpy as np
import torch
from torch import nn
from photutils.centroids import centroid_quadratic, centroid_com, centroid_com, centroid_quadratic
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from project_settings import xp, use_cupy

try:
    from graphviz import Digraph
except:
    class Digraph:
        def __init__(self, *args, **kwargs):
            warnings.warn("Graphviz package is not installed. Graph visualization will not be available.")
            pass   
        def node(self, *args, **kwargs): pass
        def edge(self, *args, **kwargs): pass


rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
r0     = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]


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


class Timer:
    def __init__(self, device_type='cpu'):
        self.device_type = device_type

    def start(self):
        if self.device_type == 'cuda': 
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return start
        else:
            return time.time()

    def elapsed(self, start):
        if self.device_type == 'cuda':
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        else:
            return (time.time()-start) * 1000.0 # in [ms]


def check_framework(x):
    """Return the array library (numpy or cupy) that matches the input array."""
    if isinstance(x, np.ndarray):
        return np
    if use_cupy and hasattr(x, 'device'):
        return xp
    return np


def cropper(x, win, center=None):
    if center is None:
        return np.s_[...,
                    x.shape[-2]//2-win//2 : x.shape[-2]//2 + win//2 + win%2,
                    x.shape[-1]//2-win//2 : x.shape[-1]//2 + win//2 + win%2]
    else:        
        return np.s_[...,
                np.round(center[0]).astype(int)-win//2 : np.round(center[0]).astype(int) + win//2 + win%2,
                np.round(center[1]).astype(int)-win//2 : np.round(center[1]).astype(int) + win//2 + win%2]


def safe_centroid(data):       
    xycen = centroid_quadratic(np.abs(data))
    
    if np.any(np.isnan(xycen)):
        xycen = centroid_com(np.abs(data))
        
    if np.any(np.isnan(xycen)):
        xycen = np.array(data.shape)//2
    
    return xycen


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


def center_of_mass(arr):
    """
    Calculate the center of mass of an array.

    Works with NumPy or CuPy arrays. If `arr` is a CuPy array, 
    computations run on the GPU.

    Parameters
    ----------
    arr : ndarray (numpy or cupy)
        Input data. Values act as "masses" (can be positive or negative).

    Returns
    -------
    com : tuple of floats
        Coordinates of the center of mass.
    """
    
    _xp = check_framework(arr)
    arr = _xp.asarray(arr)
    m = _xp.sum(arr)
    
    grids = _xp.ogrid[tuple(slice(0, s) for s in arr.shape)]
    grids = [g.astype(_xp.float64, copy=False) for g in grids]
    coords = [_xp.sum(arr * g) / m for g in grids]

    return tuple(coords)


def GetROIaroundMax(image, win=200):
    """
    Get a square ROI centered on the brightest spot (refined by center-of-mass).

    Works with NumPy or CuPy arrays.

    Parameters
    ----------
    im : 2D ndarray (numpy or cupy)
        Image data. Can contain NaN/Inf.
    win : int, optional
        Half-size ("radius") of the ROI to extract. The ROI side length
        is up to (2*win + 1), clipped at image borders.

    Returns
    -------
    roi : ndarray
        Cropped image around the brightest spot (with inf treated as nan).
    ids : tuple(slice, slice)
        Slice objects used to index the ROI.
    max_id : tuple(int, int)
        (row, col) of the final center used for the ROI in the full image.
    """

    _xp = check_framework(image)

    # Work on a safe copy; treat +/−inf as NaN for max/CoM logic
    work = _xp.array(image, copy=True)
    work = _xp.where(_xp.isinf(work), _xp.nan, work)

    # 1) initial brightest pixel (ignoring NaNs)
    max_flat = _xp.nanargmax(work)          # raises if all-NaN; same as numpy/cupy
    
    # max_id = _xp.unravel_index(max_flat, work.shape)
    max_id = tuple(int(i) for i in _xp.unravel_index(max_flat, work.shape))

    # 2) refine with center-of-mass in a small local window (radius 20)
    local_ids = CroppedROI(work, max_id, 20)
    local = work[local_ids]
    CoG_rc = _xp.asarray(center_of_mass(local)).round().astype(_xp.int32)
    max_id = (local_ids[0].start + int(CoG_rc[0]),
              local_ids[1].start + int(CoG_rc[1]))

    # 3) final ROI of size ~ (2*win+1) around refined center
    ids = CroppedROI(work, max_id, int(win))
    roi = work[ids]

    return roi, ids, max_id


'''
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
'''

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


'''
def check_types(obj, prefix=''):
    # Recursively checks and prints the types of elements in a nested dictionary or tensor structure.
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f'{prefix}{key}' if not prefix else f'{prefix}.{key}'
            if isinstance(value, torch.Tensor):
                print(f'{full_key}: torch.Tensor (shape: {value.shape}, device: {value.device})')
            elif isinstance(value, dict):
                print(f'{full_key}: dict')
                check_types(value, full_key)
            else:
                print(f'{full_key}: {type(value).__name__}')
    else:
        print(f'{prefix}: {type(obj).__name__}')
'''

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
    
    
class RadialProfileLossSimple(nn.Module):
    """
    Differentiable radial-profile loss for PSFs of shape (N, N_wvl, H, W).
    Assumes PSFs are already normalized and centered.

    Args:
      n_bins: number of radial bins
      r_max: max radius in pixels (default: half of min(H,W) minus 1)
      sigma_scale: Gaussian bin width factor (sigma = sigma_scale * bin_width)
      loss: 'mse' or 'fvu'
      bin_weight: 'uniform' | 'counts' | 'r'   (optional emphasis on wings/area)
      log_profile: if True, profile over log(PSF+eps) to emphasize wings
      eps: numerical stability
    """
    def __init__(
        self,
        n_bins: int = 64,
        r_max: float | None = None,
        sigma_scale: float = 0.5,
        loss: str = "fvu",
        bin_weight: str = "r",
        log_profile: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert loss in ("mse", "fvu")
        assert bin_weight in ("uniform", "counts", "r")
        self.n_bins = n_bins
        self.r_max = r_max
        self.sigma_scale = sigma_scale
        self.loss = loss
        self.bin_weight = bin_weight
        self.log_profile = log_profile
        self.eps = eps

        # small cache for precomputed weights
        self._cache = {}  # key = (H,W,device,dtype,n_bins,r_max,sigma_scale)

    def _precompute(self, H, W, device, dtype):
        r_max = self.r_max if self.r_max is not None else float(min(H, W) / 2.0 - 1.0)
        key = (H, W, device, dtype, self.n_bins, r_max, self.sigma_scale)
        if key in self._cache:
            return self._cache[key]

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)  # (H,W)

        edges = torch.linspace(0.0, r_max, self.n_bins + 1, device=device, dtype=dtype)
        centers = 0.5 * (edges[:-1] + edges[1:])         # (B,)
        bin_width = r_max / self.n_bins
        sigma = self.sigma_scale * bin_width + self.eps

        # soft (Gaussian) bin weights: W(h,w,b)
        W = torch.exp(-0.5 * ((r.unsqueeze(-1) - centers) / sigma) ** 2)  # (H,W,B)

        self._cache[key] = (W, centers, r_max)
        return W, centers, r_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape and pred.ndim == 4
        N, WVL, H, W = pred.shape
        device, dtype = pred.device, pred.dtype

        # precompute soft radial weights
        W_hwB, centers, _ = self._precompute(H, W, device, dtype)  # (H,W,B),(B,)
        # expand to broadcast over batch/wavelengths
        W_full = W_hwB.view(1, 1, H, W, self.n_bins)               # (1,1,H,W,B)

        # choose image domain (linear or log)
        X = torch.log(pred + self.eps) if self.log_profile else pred
        Y = torch.log(target + self.eps) if self.log_profile else target

        # radial profiles via weighted average over pixels
        num_pred = (X.unsqueeze(-1) * W_full).sum(dim=(-3, -2))    # (N,WVL,B)
        num_targ = (Y.unsqueeze(-1) * W_full).sum(dim=(-3, -2))    # (N,WVL,B)
        den = W_full.sum(dim=(-3, -2)).clamp_min(self.eps)         # (1,1,B) -> broadcasts
        prof_pred = num_pred / den
        prof_targ = num_targ / den

        # per-bin weights
        if self.bin_weight == "uniform":
            w = torch.ones_like(prof_pred)
        elif self.bin_weight == "counts":
            w = den.expand_as(prof_pred)                           # proportional to pixel counts
        else:  # 'r' — emphasize wings
            w = centers.view(1, 1, -1).expand_as(prof_pred)

        # normalize weights per-sample to mean 1 over bins
        w = w / w.mean(dim=-1, keepdim=True).clamp_min(self.eps)

        if self.loss == "mse":
            diff = prof_pred - prof_targ
            loss = (w * diff.pow(2)).mean()
        else:  # 'fvu' across bins, then average over batch & wavelength
            y = prof_targ
            ybar = y.mean(dim=-1, keepdim=True)
            sse = (w * (prof_pred - y).pow(2)).sum(dim=-1)                 # (N,WVL)
            ssy = (w * (y - ybar).pow(2)).sum(dim=-1).clamp_min(self.eps)  # (N,WVL)
            fvu = sse / ssy
            loss = fvu.mean()

        return loss
