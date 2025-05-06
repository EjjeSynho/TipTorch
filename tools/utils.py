#%%
import numpy as np
import torch
from torch import nn
from scipy.ndimage import center_of_mass
from math import prod

import seaborn as sns
from photutils.centroids import centroid_quadratic, centroid_com, centroid_com, centroid_quadratic
from photutils.profiles import RadialProfile
from astropy.modeling import models, fitting

import matplotlib.pyplot as plt
from matplotlib import cm

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
        # return np.s_[...,
        #             np.round(center[0]-win//2,2).astype(int) : np.round(center[0] + win//2 + win%2,2).astype(int),
        #             np.round(center[1]-win//2,2).astype(int) : np.round(center[1] + win//2 + win%2,2).astype(int)]
        
        return np.s_[...,
                np.round(center[0]).astype(int)-win//2 : np.round(center[0]).astype(int) + win//2 + win%2,
                np.round(center[1]).astype(int)-win//2 : np.round(center[1]).astype(int) + win//2 + win%2]


    # else:
    #     return np.s_[...,
    #                 np.floor(center[0]).astype(int)-win//2 : np.ceil(center[0]).astype(int)+win//2+win%2,
    #                 np.floor(center[1]).astype(int)-win//2 : np.ceil(center[1]).astype(int)+win//2+win%2]
        


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



class ParameterReshaper():
    def __init__(self):
        super().__init__()
        self.parameters_list = []
        self.p_shapes = []
        self.fixed_ids = []
        self.device = torch.device('cpu')

    def flatten(self, parameters_list):
        p_list = []
        self.fixed_ids = []
        self.parameters_list = parameters_list
        self.device = self.parameters_list[0].device
        for i,p in enumerate(self.parameters_list):
            if p.requires_grad:
                p_list.append(p.clone().detach().cpu())
            else:
                self.fixed_ids.append(i)
        self.p_shapes = [p.shape for p in p_list]
        return torch.hstack([p.view(-1) for p in p_list])

    def unflatten(self, p):
        N_p = len(self.p_shapes)
        p_shapes_flat = [prod(shp) for shp in self.p_shapes]
        ids = [slice(sum(p_shapes_flat[:i]), sum(p_shapes_flat[:i+1])) for i in range(N_p)]
        p_list = [p[ids[i]].reshape(self.p_shapes[i]) for i in range(N_p)]
        for i in self.fixed_ids:
            p_list.insert(i, self.parameters_list[i])
        for i in range(len(p_list)):
            if type(p_list) != torch.tensor:
                p_list[i] = torch.tensor(p_list[i], device=self.device)
        return p_list


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


# Outputs dictionary, even with subdictionaries
def print_dict(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key) + ':', end='')
      if isinstance(value, dict):
        print()
        print_dict(value, indent+1)
      else:
        # print('\t'*(indent+1) + str(value))
        print('  ' + str(value))


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
        
        # plt.savefig(f'C:/Users/akuznets/Desktop/thesis_results/MUSE/PSFs_examples/{id}.pdf', dpi=300)


class Photometry:
    def __init__(self):
        self.__InitPhotometry()
    
    def FluxFromMag(self, mag, band):
        return self.__PhotometricParameters(band)[2]/368 * 10**(-0.4*mag)
    
    def PhotonsFromMag(self, area, mag, band, sampling_time):
        return self.FluxFromMag(mag,band) * area * sampling_time

    def MagFromPhotons(self, area, Nph, band, sampling_time):
        photons = Nph / area / sampling_time
        return -2.5 * np.log10(368 * photons / self.__PhotometricParameters(band)[2])

    def __InitPhotometry(self):
        # photometry object [wavelength, bandwidth, zeroPoint]
        self.bands = {
            'U'   : [ 0.360e-6 , 0.070e-6 , 2.0e12 ],
            'B'   : [ 0.440e-6 , 0.100e-6 , 5.4e12 ],
            'V0'  : [ 0.500e-6 , 0.090e-6 , 3.3e12 ],
            'V'   : [ 0.550e-6 , 0.090e-6 , 3.3e12 ],
            'R'   : [ 0.640e-6 , 0.150e-6 , 4.0e12 ],
            'I'   : [ 0.790e-6 , 0.150e-6 , 2.7e12 ],
            'I1'  : [ 0.700e-6 , 0.033e-6 , 2.7e12 ],
            'I2'  : [ 0.750e-6 , 0.033e-6 , 2.7e12 ],
            'I3'  : [ 0.800e-6 , 0.033e-6 , 2.7e12 ],
            'I4'  : [ 0.700e-6 , 0.100e-6 , 2.7e12 ],
            'I5'  : [ 0.850e-6 , 0.100e-6 , 2.7e12 ],
            'I6'  : [ 1.000e-6 , 0.100e-6 , 2.7e12 ],
            'I7'  : [ 0.850e-6 , 0.300e-6 , 2.7e12 ],
            'R2'  : [ 0.650e-6 , 0.300e-6 , 7.92e12],
            'R3'  : [ 0.600e-6 , 0.300e-6 , 7.92e12],
            'R4'  : [ 0.670e-6 , 0.300e-6 , 7.92e12],
            'I8'  : [ 0.750e-6 , 0.100e-6 , 2.7e12 ],
            'I9'  : [ 0.850e-6 , 0.300e-6 , 7.36e12],
            'J'   : [ 1.215e-6 , 0.260e-6 , 1.9e12 ],
            'H'   : [ 1.654e-6 , 0.290e-6 , 1.1e12 ],
            'Kp'  : [ 2.1245e-6, 0.351e-6 , 6e11   ],
            'Ks'  : [ 2.157e-6 , 0.320e-6 , 5.5e11 ],
            'K'   : [ 2.179e-6 , 0.410e-6 , 7.0e11 ],
            'L'   : [ 3.547e-6 , 0.570e-6 , 2.5e11 ],
            'M'   : [ 4.769e-6 , 0.450e-6 , 8.4e10 ],
            'Na'  : [ 0.589e-6 , 0        , 3.3e12 ],
            'EOS' : [ 1.064e-6 , 0        , 3.3e12 ]
        }
        self.__wavelengths = np.array( [v[0] for _,v in self.bands.items()] )

    def __PhotometricParameters(self, inp):
        if isinstance(inp, str):
            if inp not in self.bands.keys():
                raise ValueError('Error: there is no band with the name "'+inp+'"')
                return None
            else:
                return self.bands[inp]

        elif isinstance(inp, float):    # perform interpolation of parameters for a current wavelength
            if inp < self.__wavelengths.min() or inp > self.__wavelengths.max():
                print('Error: specified value is outside the defined wavelength range!')
                return None

            difference = np.abs(self.__wavelengths - inp)
            dtype = [('number', int), ('value', float)]

            sorted = np.sort(np.array([(num, val) for num,val in enumerate(difference)], dtype=dtype), order='value')                        

            l_1 = self.__wavelengths[sorted[0][0]]
            l_2 = self.__wavelengths[sorted[1][0]]

            if l_1 > l_2:
                l_1, l_2 = l_2, l_1

            def find_params(input):
                for _,v in self.bands.items():
                    if input == v[0]:
                        return np.array(v)

            p_1 = find_params(l_1)
            p_2 = find_params(l_2)
            weight = ( (np.array([l_1, inp, l_2])-l_1)/(l_2-l_1) )[1]

            return weight*(p_2-p_1) + p_1

        else:
            print('Incorrect input: "'+inp+'"')
            return None             


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


def Center(im, centered=True):
    if type(im) == torch.Tensor: im = im.detach().cpu().numpy()
    im = im.squeeze()
    WoG_ROI = 16
    center = np.array(np.unravel_index(np.argmax(im), im.shape))
    crop = slice(center[0]-WoG_ROI//2, center[1]+WoG_ROI//2)
    crop = (crop, crop)
    buf = im[crop]
    WoG = np.array(center_of_mass(buf)) + im.shape[0]//2-WoG_ROI//2
    return WoG - np.array(im.shape)//2 * int(centered)


def CircularMask(img, center, radius):
    xx,yy = np.meshgrid( np.arange(0,img.shape[1]), np.arange(0,img.shape[0]) )
    mask_PSF = np.ones(img.shape[0:2])
    mask_PSF[np.sqrt((yy-center[0])**2 + (xx-center[1])**2) < radius] = 0.0
    return mask_PSF


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
class OptimizeLBFGS:
    def __init__(self, model, loss_fn, verbous=True):
        self.model = model
        self.loss_fn = loss_fn
        self.verbous = verbous
        

    def Optimize(self, PSF_ref, to_optimize, steps, args=[None]):
        optimizer = optim.LBFGS(to_optimize, lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

        early_stopping = EarlyStopping(patience=2, tolerance=0.01, relative=False)

        for i in range(steps):
            optimizer.zero_grad()
            loss = self.loss_fn( self.model(*args), PSF_ref )

            if np.isnan(loss.item()): return
            early_stopping(loss)

            loss.backward(retain_graph=True)
            optimizer.step( lambda: self.loss_fn(self.model(*args), PSF_ref) )

            if self.verbous:
                print('Loss:', loss.item(), end="\r", flush=True)

            if early_stopping.stop:
                # if self.verbous: print('Stopped at it.', i, 'with loss:', loss.item())
                break


class OptimizeTRF():
    def __init__(self, model, parameters) -> None:
        self.model = model
        self.free_params = np.where(
            np.array([param.requires_grad for param in parameters]))[0].tolist()
        self.parameters = [p.clone().detach() for p in parameters]

    def __unpack_params(self, X):
        to_optimize = torch.tensor(X.reshape(self.init_shape), dtype=torch.float32, device=self.model.device)
        for i,free_param in enumerate(self.free_params):
            self.parameters[free_param] = to_optimize[:,i]

    def __wrapper(self,X):
        self.__unpack_params(X)
        return self.model.PSD2PSF(*self.parameters)

    def Optimize(self, PSF_ref):
        X0 = torch.stack(
            [self.parameters[i] for i in self.free_params]).detach().cpu().numpy().T
        self.init_shape = X0.shape
        X0 = X0.flatten()
        func = lambda x: (PSF_ref-self.__wrapper(x)).detach().cpu().numpy().reshape(-1)

        iterations = 3
        for _ in range(iterations):
            result = least_squares(func, X0, method='trf',
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9,
                                   max_nfev=1000, verbose=1, loss="linear")
            X0 = result.x
        self.__unpack_params(X0)

        for free_param in self.free_params: # restore intial requires_grad from PyTorch
            self.parameters[free_param].requires_grad = True
        return self.parameters
'''


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


'''
# Function to calculate the radial profile of a PSF or of a PSF stack
def radial_profile(data, center=None):
    t2np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def calc_profile(data, center=None):
        if center is None:
            center = (data.shape[0]//2, data.shape[1]//2)
        y, x = np.indices((data.shape))
        r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
        r = r.astype('int')

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile[0:data.shape[0]//2]

    if len(data.shape) > 2:
        raise ValueError('PSF stack of wrong dimensionality is passed!')
    return calc_profile(t2np(data), t2np(center))

def plot_radial_profiles(PSF_refs, PSF_estims, label_refs, label_estim, title='', dpi=300, scale='log', colors=['tab:blue', 'tab:orange'], cutoff=32):
    from tools.utils import radial_profile
    if not isinstance(PSF_refs, list):   PSF_refs   = [PSF_refs.squeeze()]
    if not isinstance(PSF_estims, list): PSF_estims = [PSF_estims.squeeze()]

    n_profiles = len(PSF_refs)

    radial_profiles_0 = []
    radial_profiles_1 = []
    diff_profile      = []

    for i in range(n_profiles):
        if type(PSF_refs[i]) is torch.Tensor:
            PSF_refs[i] = PSF_refs[i].detach().cpu().numpy()
        if type(PSF_estims[i]) is torch.Tensor:
            PSF_estims[i] = PSF_estims[i].detach().cpu().numpy()

        profile_0 = radial_profile(PSF_refs[i].squeeze())[:cutoff+1]
        profile_1 = radial_profile(PSF_estims[i].squeeze())[:cutoff+1]

        radial_profiles_0.append(profile_0)
        radial_profiles_1.append(profile_1)

        diff_profile.append(np.abs(profile_1-profile_0) / profile_0.max() * 100)  # [%]

    mean_profile_0 = np.nanmean(radial_profiles_0, axis=0)
    mean_profile_1 = np.nanmean(radial_profiles_1, axis=0)
    std_profile_0  = np.nanstd (radial_profiles_0, axis=0)
    std_profile_1  = np.nanstd (radial_profiles_1, axis=0)

    mean_profile_diff = np.nanmean(diff_profile, axis=0)
    std_profile_diff  = np.nanstd (diff_profile, axis=0)

    fig = plt.figure(figsize=(6, 4), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Relative intensity')
    if scale == 'log': ax.set_yscale('log')
    ax.set_xlim([0, len(mean_profile_1) - 1])
    ax.grid()
    ax2 = ax.twinx()
    ax2.set_ylim([0, np.nanmax(mean_profile_diff) * 1.5])
    ax2.set_ylabel('Difference [%]')

    l1 = ax.plot(mean_profile_0, label=label_refs,  linewidth=2, color=colors[0])
    l2 = ax.plot(mean_profile_1, label=label_estim, linewidth=2, color=colors[1])
    l3 = ax2.plot(mean_profile_diff, label='Difference', color='green', linewidth=1.5, linestyle='--')

    ax.fill_between(range(len(mean_profile_0)), mean_profile_0-std_profile_0, mean_profile_0+std_profile_0, alpha=0.2, color=colors[0])
    ax.fill_between(range(len(mean_profile_1)), mean_profile_1-std_profile_1, mean_profile_1+std_profile_1, alpha=0.2, color=colors[1])
    ax2.fill_between(range(len(mean_profile_diff)), mean_profile_diff - std_profile_diff, mean_profile_diff + std_profile_diff, alpha=0.2, color='green')

    ls = l1 + l2 + l3
    labs = [l.get_label() for l in ls]
    ax2.legend(ls, labs, loc=0)


def plot_std(x,y, label, color, style): #TODO: deprecated
    y_m = y.mean(axis=0)
    y_s = y.std(axis=0)
    lower_bound = y_m-y_s
    upper_bound = y_m+y_s

    print(label, 'mean:', y_m.max())

    plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.3)
    plt.plot(x, y_m, label=label, color=color, linestyle=style)
    plt.show()
'''

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

    
def plot_radial_profiles_new(PSF_0,
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


def CircPu(samples, D=8.0, centralObstruction=1.12):
    x      = np.linspace(-1/2, 1/2, samples)*D
    xx,yy  = np.meshgrid(x,x)
    circle = np.sqrt(xx**2 + yy**2)
    obs    = circle >= centralObstruction/2
    pupil  = circle < D/2 
    return pupil * obs


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
    
    
'''
class Zernike:
    def __init__(self, modes_num=1):
        global global_gpu_flag
        self.nModes = modes_num
        self.modesFullRes = None
        self.pupil = None

        self.modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X)', 'Astigmatism (+)',
            'Coma vert', 'Coma horiz', 'Trefoil vert', 'Trefoil horiz',
            'Sphere', 'Secondary astig (X)', 'Secondary astig (+)',
            'Quadrofoil vert', 'Quadrofoil horiz',
            'Secondary coma horiz', 'Secondary coma vert',
            'Secondary trefoil horiz', 'Secondary trefoil vert',
            'Pentafoil horiz', 'Pentafoil vert'
        ]
        self.gpu = global_gpu_flag  


    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, 'modesFullRes'):
                if not hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)
        else:
            self.__gpu = False
            if hasattr(self, 'modesFullRes'):
                if hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = self.modesFullRes.get()


    def zernikeRadialFunc(self, n, m, r):
        """
        Fucntion to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """

        R = np.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n-m)/2) + 1):
            R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                            np.math.factorial(n-i)) / (np.math.factorial(i) *
                            np.math.factorial(int(0.5 * (n+m) - i)) *
                            np.math.factorial(int(0.5 * (n-m) - i))),
                            dtype='float')
        return R


    def zernIndex(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0: s = 1
            else:  s = -1
            m *= s

        return [n, m]


    def rotate_coordinates(self, angle, X, Y):
            angle_rad = np.radians(angle)

            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])

            coordinates = np.vstack((X, Y))
            rotated_coordinates = np.dot(rotation_matrix, coordinates)
            rotated_X, rotated_Y = rotated_coordinates[0, :], rotated_coordinates[1, :]

            return rotated_X, rotated_Y
        

    def computeZernike(self, tel, normalize_unit=False, angle=None, transposed=False):
        """
        Function to calculate the Zernike modal basis

        Parameters:
            tel (Telescope): A telescope object, needed mostly to extract pupil data 
            normalize_unit (bool): Sets the regime for normalization of Zernike modes
                                   it's either the telescope's pupil or a unit circle  
        """
 
        resolution = tel.pupil.shape[0]

        self.gpu = self.gpu and tel.gpu
        if normalize_unit:
            self.pupil = mask_circle(N=resolution, r=resolution/2)
        else:
            self.pupil = tel.pupil.get() if self.gpu else tel.pupil

        X, Y = np.where(self.pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        
        if transposed:
            X, Y = Y, X
        
        if angle is not None and angle != 0.0:
            X, Y = self.rotate_coordinates(angle, X, Y)
        
        R = np.sqrt(X**2 + Y**2)
        R /= R.max()
        theta = np.arctan2(Y, X)

        self.modesFullRes = np.zeros([resolution**2, self.nModes])

        for i in range(1, self.nModes+1):
            n, m = self.zernIndex(i+1)
            if m == 0:
                Z = np.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.cos(m*theta)
                else:   #i is odd
                    m = abs(m)
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.sin(m*theta)
            
            Z -= Z.mean()
            Z /= np.std(Z)

            self.modesFullRes[np.where(np.reshape(self.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modesFullRes = np.reshape( self.modesFullRes, [resolution, resolution, self.nModes] )
        
        if self.gpu: # if GPU is used, return a GPU-based array
            self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)


    def modeName(self, index):
        if index < 0:
            return('Incorrent index!')
        elif index >= len(self.modes_names):
            return('Z ' + str(index+2))
        else:
            return(self.modes_names[index])


    # Generate wavefront shape corresponding to given model coefficients and modal basis 
    def wavefrontFromModes(self, tel, coefs_inp):
        xp = cp if self.gpu else np

        coefs = xp.array(coefs_inp).flatten()
        coefs[xp.where(xp.abs(coefs)<1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs))[0]

        if self.modesFullRes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.nModes = xp.max(xp.array([coefs.shape[0], self.nModes]))
            self.computeZernike(tel)

        if self.nModes < coefs.shape[0]:
            self.nModes = coefs.shape[0]
            print('Warning: vector of coefficients is too long. Computiong additional modes...')
            self.computeZernike(tel)

        return self.modesFullRes[:,:,valid_ids] @ coefs[valid_ids] # * tel.pupil


    def Mode(self, coef):
        return self.modesFullRes[:,:,coef]

'''