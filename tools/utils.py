#%%
import numpy as np
import torch
from torch import optim, nn
from astropy.io import fits
from scipy.ndimage import center_of_mass
from scipy.optimize import least_squares
from math import prod
from graphviz import Digraph
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import label
import seaborn as sns
from photutils.centroids import centroid_quadratic, centroid_com
from photutils.profiles import RadialProfile
from astropy.modeling import models, fitting


rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
r0 = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]


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

def separate_islands(binary_image):
    # Label each connected component (each "island") with a unique integer
    labeled_image, num_features = label(binary_image)
    # Create an empty list to store the separated images
    separated_images = []

    # Iterate over each unique label (each island)
    for i in range(1, num_features + 1):
        # Create an image with only the current island
        island_image = (labeled_image == i).astype(int)
        separated_images.append(island_image)

    return separated_images


decompose_WF = lambda WF, basis, pupil: WF[:, pupil > 0] @ basis[:, pupil > 0].T / pupil.sum()
project_WF   = lambda WF, basis, pupil: torch.einsum('mn,nwh->mwh', decompose_WF(WF, basis, pupil), basis)
calc_WFE     = lambda WF, pupil: WF[:, pupil > 0].std()


def BuildPTTBasis(pupil, pytorch=True):
    tip, tilt = np.meshgrid( np.linspace(-1, 1, pupil.shape[-2]), np.linspace(-1, 1, pupil.shape[-1]) )

    tip  = pupil * tip  / np.std( tip [np.where(pupil > 0)] )
    tilt = pupil * tilt / np.std( tilt[np.where(pupil > 0)] )

    PTT_basis = np.stack([pupil, tip, tilt], axis=0)
    
    if pytorch:
        PTT_basis = torch.tensor(PTT_basis)
                
    return PTT_basis


def BuildPetalBasis(segmented_pupil, pytorch=True):
    petals = np.stack( separate_islands(segmented_pupil) )
    x, y = np.meshgrid(np.arange(segmented_pupil.shape[-1]), np.arange(segmented_pupil.shape[-2]))

    tilt = (x[None, ...]*petals).astype(np.float64)
    tip  = (y[None, ...]*petals).astype(np.float64)

    tilt /= tilt.sum(axis=0)[np.where(segmented_pupil)].std()
    tip  /= tip.sum(axis=0) [np.where(segmented_pupil)].std()

    def normalize_TT(x):
        for i in range(petals.shape[0]):
            x[i,...] = (x[i,...] - x[i,...][np.where(petals[i,...])].mean()) * petals[i,...]
        return x

    tilt, tip = normalize_TT(tilt), normalize_TT(tip)
    coefs = [1.]*petals.shape[0] + [0.]*petals.shape[0] + [0.]*petals.shape[0]
    
    basis = np.vstack([petals, tilt, tip])
    
    basis_flatten = basis[:, segmented_pupil > 0]
    modes_STD = np.sqrt( np.diag(basis_flatten @ basis_flatten.T / segmented_pupil.sum()) )
    basis /= modes_STD[:, None, None]
    
    if not pytorch:
        return basis, np.array(coefs)
    else:
        return torch.from_numpy( basis ), torch.tensor(coefs)


class LWE_basis():
    def __init__(self, model) -> None:
        from tools.utils import BuildPetalBasis
        self.model = model
        self.modal_basis, self.__coefs_flat = BuildPetalBasis(self.model.pupil.cpu(), pytorch=True)
        
        # self.modal_basis  = self.modal_basis[1:,...].float().to(model.device)
        # self.__coefs_flat = self.__coefs_flat[1:].float().to(model.device)
       
        self.modal_basis  = self.modal_basis.float().to(model.device)
        self.__coefs_flat = self.__coefs_flat.float().to(model.device)
        
        self.coefs = self.__coefs_flat.repeat(self.model.N_src, 1)
        # if optimizable:
            # self.coefs = nn.Parameter(self.coefs)

    def forward(self, x=None):
        if x is not None:
            self.coefs = x
 
        # self.coefs = self.__coefs_flat.repeat(self.model.N_src, 1)
        OPD = torch.einsum('mn,nwh->mwh', self.coefs, self.modal_basis) * 1e-9  
        return pdims(self.model.pupil * self.model.apodizer, -2) * torch.exp(1j*2*np.pi / pdims(self.model.wvl,2)*OPD.unsqueeze(1))
    
    def __call__(self, *args):
        return self.forward(*args)


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
    gif_anim = []
    
    def remove_transparency(img, bg_colour=(255, 255, 255)):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    
    for layer in images_stack:
        im = Image.fromarray(np.uint8(layer*255))
        im.thumbnail((im.size[0]//downscale, im.size[1]//downscale), Image.ANTIALIAS)
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


def GetROIaroundMax(im, win=100):
    im[np.isinf(im)] = np.nan
    # determine the position of maximum intensity, so the image is centered around the brightest star
    max_id = np.unravel_index(np.nanargmax(im), im.shape)
    # make it more correct with the center of mass
    max_crop = CroppedROI(im, max_id, 20)
    CoG_id = np.array(center_of_mass(np.nan_to_num(im[max_crop]))).round().astype(np.int32)
    max_id = (max_crop[0].start + CoG_id[0], max_crop[1].start + CoG_id[1])
    ids = CroppedROI(im, max_id, win)
    return im[ids], ids, max_id


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
    wavelength = float(wavelength)
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


def draw_PSF_stack(PSF_in, PSF_out, average=False, scale='log', min_val=1e-16, max_val=1e16, crop=None):
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
        
        plt.imshow(row, norm=norm)#, origin='lower')
        plt.title('Sources average')
        # plt.show()

    else:
        for src in range(PSF_in.shape[0]):
            row = []
            for wvl in range(PSF_in.shape[1]):
                row.append( np.hstack([cut(PSF_in[src, wvl,...]), cut(PSF_out[src, wvl,...]), cut(dPSF[src, wvl,...])]) )
            row  = np.vstack(row)
            if scale == 'log':
                norm = LogNorm(vmin=np.maximum(row.min(), min_val), vmax=np.minimum(row.max(), max_val))
            else:
                norm = None
                
            plt.imshow(row, norm=norm)#, origin='lower')
            plt.title('Source %d' % src)
            plt.show()


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
    
    if np.any(np.isnan(xycen)):
        xycen = centroid_com(np.abs(data))
        
    if np.any(np.isnan(xycen)):
        xycen = np.array(data.shape)//2
        
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
                             suppress_plot = False):
            
    def calc_profile(data, xycen=None):
        xycen = safe_centroid(data) if xycen is None else xycen
        edge_radii = np.arange(data.shape[-1]//2)
        rp = RadialProfile(data, xycen, edge_radii)
        return rp.profile

    def _radial_profiles(PSFs, centers=None):
        # if PSFs.ndim == 2: PSFs = PSFs[np.newaxis, ...]
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

    if PSF_0.ndim == 2: PSF_0 = PSF_0[np.newaxis, ...]
    if PSF_1.ndim == 2: PSF_1 = PSF_1[np.newaxis, ...]

    profis_0 = _radial_profiles( PSF_0[:,...], centers )
    profis_1 = _radial_profiles( PSF_1[:,...], centers )

    center_0 = safe_centroid(np.abs(np.nanmean(PSF_0, axis=0)))
    center_1 = safe_centroid(np.abs(np.nanmean(PSF_1, axis=0)))
    center_  = np.mean([center_0, center_1], axis=0)
    profis_err = _radial_profiles( PSF_0[:,...] - PSF_1[:,...], center_ )

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
            ax.set_yscale('symlog', linthresh=5e-1)
            ax.set_ylim(1e-2, y_lim)
        else:
            x_max = cutoff#*0.7
            ax.ylim(0, y_lim)

        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, x_max)
        ax.text(x_max-16, max_err+2.5, "Max. err.: {:.1f}%".format(max_err), fontsize=12)
        ax.set_xlabel('Pixels from on-axis, [pix]')
        ax.set_ylabel('Normalized intensity, [%]')
        ax.grid()

    if return_profiles:
        return p_0, p_1, p_err


'''
def DisplayDataset(samples_list, tiles_in_row, show_labels=True, dpi=300):
    from PIL import Image, ImageDraw
    from matplotlib import cm

    def prepare_single_img(sample, crop_region=32):
        buf_data = sample['input']['image']
        label_buf = sample['file_id']

        crop_region = 32
        ROI_x = slice(buf_data.shape[0]//2-crop_region, buf_data.shape[0]//2+crop_region)
        ROI_y = slice(buf_data.shape[1]//2-crop_region, buf_data.shape[1]//2+crop_region)
        buf_data = np.log(buf_data/buf_data.max())[(ROI_x, ROI_y)]
        buf_data -= buf_data[np.where(np.isfinite(buf_data))].min()
        buf_data[np.where(np.isnan(buf_data))] = 0.0

        im_buf = Image.fromarray( np.uint8(cm.viridis(buf_data/buf_data.max())*255) )
        if show_labels:
            I1 = ImageDraw.Draw(im_buf)
            I1.text((0,0), str(label_buf), fill=(255,255,255))
        return np.asarray(im_buf)

    def prepare_row(row_list):
        row_processed = [prepare_single_img(buf) for buf in row_list]
        hspace = 3
        hgap = np.ones([row_processed[0].shape[0], hspace, 4], dtype=row_processed[0].dtype)

        row = []
        for i in range(len(row_processed)-1):
            row.append(row_processed[i])
            row.append(hgap)
        row.append(row_processed[-1])
        return np.hstack(row)

    def prepare_ids(num_in_row, samples_num):
        A = []
        ids = np.arange(samples_num).tolist()
        c = 0
        if num_in_row > samples_num: num_in_row = samples_num
        for i in range(samples_num//num_in_row+1):
            a = []
            for j in range(num_in_row):
                if c >= samples_num: break
                a.append(ids[c])
                c = c+1
            if len(a) > 0: A.append(a)
        return A

    def match_dimensions(col_list):
        delta_width = col_list[-2].shape[1] - col_list[-1].shape[1]

        if delta_width > 0:
            filler = np.ones([col_list[-2].shape[0], delta_width, 4], dtype=np.uint8)*255
            col_list[-1] = np.hstack([col_list[-1], filler])
        return col_list

    def prepare_cols(samples_list, ids_list):
        col_list = []
        for a in ids_list:
            buf_samples = [samples_list[i] for i in a]
            col_list.append(prepare_row(buf_samples))

        col_list = match_dimensions(col_list)

        col_processed = [col for col in col_list]
        vspace = 3
        vgap = np.ones([vspace, col_processed[0].shape[1], 4], dtype=col_processed[0].dtype)*255

        cols = []
        for i in range(len(col_processed)-1):
            cols.append(col_processed[i])
            cols.append(vgap)
        cols.append(col_processed[-1])
        return np.vstack(cols)

    samples_num = len(samples_list)

    ids  = prepare_ids(tiles_in_row, samples_num)
    tile = prepare_cols(samples_list, ids)

    plt.figure(dpi=dpi)
    plt.axis('off')
    plt.imshow(tile)
'''


def CircPupil(samples, D=8.0, centralObstruction=1.12):
    x      = np.linspace(-1/2, 1/2, samples)*D
    xx,yy  = np.meshgrid(x,x)
    circle = np.sqrt(xx**2 + yy**2)
    obs    = circle >= centralObstruction/2
    pupil  = circle < D/2 
    return pupil * obs


def PupilVLT(samples, vangle=[0,0], petal_modes=False):
    pupil_diameter = 8.0	  # pupil diameter [m]
    secondary_diameter = 1.12 # diameter of central obstruction [m] 1.12
    alpha = 101				  # spider angle [degrees]
    spider_width = 0.039	  # spider width [m] 0.039;  spider is 39 mm
    # wide excepted on small areas where 50 mm width are reached over a length 
    # of 80 mm,near the centre of the spider (before GRAVITY modification?), 
    # see VLT-DWG-AES-11310-101010

    shx = np.cos(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    shx = np.cos(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    shy = np.sin(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    delta = pupil_diameter/samples # distance between samples [m]
    ext = 2*np.max(np.fix(np.abs(np.array([shx, shy]))))+1

    # create coordinate matrices
    x1_min = -(pupil_diameter+ext - 2*delta)/2
    x1_max = (pupil_diameter + ext)/2
    num_grid = int((x1_max-x1_min)/delta)+1

    x1 = np.linspace(x1_min, x1_max, num_grid) #int(1/delta))
    x, y = np.meshgrid(x1, x1)

    #  Member data
    mask = np.ones([num_grid, num_grid], dtype='bool')
    mask[ np.where( np.sqrt( (x-shx)**2 + (y-shy)**2 ) > pupil_diameter/2 ) ] = False
    mask[ np.where( np.sqrt( x**2 + y**2 ) < secondary_diameter/2 ) ] = False

    # Spiders
    alpha_rad = alpha * np.pi / 180
    slope     = np.tan( alpha_rad/2 )

    petal_1 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_2 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_3 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_4 = np.zeros([num_grid, num_grid], dtype='bool')

    #North
    petal_1[ np.where(   
        (( -y > 0.039/2 + slope*(-x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x<0)  & (y<=0)) | \
        (( -y > 0.039/2 + slope*( x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x>=0) & (y<=0)) )] = True
    petal_1 *= mask

    #East 
    petal_2[ np.where(   
        (( -y < 0.039/2 + slope*( x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x>0) & (y<=0)) | \
        ((  y < 0.039/2 + slope*( x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x>0) & (y>0)) )] = True
    petal_2 *= mask
        
    #South
    petal_3[ np.where(   
        ((  y > 0.039/2 + slope*(-x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x<=0) & (y>0)) | \
        ((  y > 0.039/2 + slope*( x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x>0)  & (y>0)) )] = True
    petal_3 *= mask
        
    #West
    petal_4[ np.where(   
        (( -y < 0.039/2 + slope*(-x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x<0) & (y<0)) |\
        ((  y < 0.039/2 + slope*(-x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x<0) & (y>=0)) )] = True
    petal_4 *= mask
        
    lim_x = [ ( np.fix((shy+ext/2)/delta) ).astype('int'), ( -np.fix((-shy+ext/2)/delta) ).astype('int') ]
    lim_y = [ ( np.fix((shx+ext/2)/delta) ).astype('int'), ( -np.fix((-shx+ext/2)/delta) ).astype('int') ]

    petal_1 = resize(petal_1[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_2 = resize(petal_2[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_3 = resize(petal_3[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_4 = resize(petal_4[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)

    if petal_modes:
        xx1, yy1 = np.meshgrid(np.linspace( -0.5, 0.5,  samples), np.linspace(-0.25, 0.75, samples))
        xx2, yy2 = np.meshgrid(np.linspace(-0.75, 0.25, samples), np.linspace( -0.5, 0.5,  samples))
        xx3, yy3 = np.meshgrid(np.linspace( -0.5, 0.5,  samples), np.linspace(-0.75, 0.25, samples))
        xx4, yy4 = np.meshgrid(np.linspace(-0.25, 0.75, samples), np.linspace( -0.5, 0.5,  samples))

        def normalize_petal_mode(petal, coord):
            mode = petal.astype('double') * coord
            mode -= mode.min()
            mode /= (mode.max()+mode.min())
            mode -= 0.5
            mode[np.where(petal==False)] = 0.0
            mode[np.where(petal==True)] -= mode[np.where(petal==True)].mean()
            mode /= mode[np.where(petal==True)].std()
            return mode

        tip_1 = normalize_petal_mode(petal_1, yy1)
        tip_2 = normalize_petal_mode(petal_2, yy2)
        tip_3 = normalize_petal_mode(petal_3, yy3)
        tip_4 = normalize_petal_mode(petal_4, yy4)

        tilt_1 = normalize_petal_mode(petal_1, xx1)
        tilt_2 = normalize_petal_mode(petal_2, xx2)
        tilt_3 = normalize_petal_mode(petal_3, xx3)
        tilt_4 = normalize_petal_mode(petal_4, xx4)

        return np.dstack( [petal_1, petal_2, petal_3, petal_4, tip_1, tip_2, tip_3, tip_4, tilt_1, tilt_2, tilt_3, tilt_4] )

    else:
        return petal_1 + petal_2 + petal_3 + petal_4


def VLTpupilArea(instrument='SPHERE'): # [m2]
    if instrument == 'SPHERE':
        pupil = fits.getdata('C:/Users/akuznets/Projects/TIPTOP/P3/aoSystem/data/VLT_CALIBRATION\VLT_PUPIL/ALC2LyotStop_measured.fits').astype('float')
    else:
        raise NotImplementedError
    relative_area = pupil.sum() / (np.pi*(pupil.shape[0]//2-6.5)**2)
    true_area = np.pi * 8**2 * relative_area
    return true_area

# %%
