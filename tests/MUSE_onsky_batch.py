#%%
%reload_ext autoreload
%autoreload 2

import os
from os import path
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas, mask_circle
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new
from data_processing.MUSE_preproc_utils import GetMUSEonsky
from project_globals import MUSE_DATA_FOLDER, device
from torchmin import minimize
from astropy.stats import sigma_clipped_stats
from managers.parameter_parser import ParameterParser
from managers.config_manager import ConfigManager
from data_processing.normalizers import TransformSequence, Uniform, LineModel, QuadraticModel
from managers.input_manager import InputsTransformer
from data_processing.MUSE_preproc_utils_old import MUSEcube
from tqdm import tqdm
from scipy.ndimage import rotate

from pprint import pprint

from copy import deepcopy

#%%
Moffat_absorber = True
include_sausage = True
derotate_PSF = True


# 411, 410, 409, 405, 146, 296, 276, 395, 254, 281, 343, 335
# ids = [411, 409, 395, 343]
ids = [124, 405, 344, 123] + [411] #, 409, 395, 343] + [276, 395]

PSF_0, _, _, merged_config = GetMUSEonsky(ids, derotate_PSF, device)

N_src = PSF_0.shape[0]
N_wvl = PSF_0.shape[1]

#%%
import configparser
import os

class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr  # Return the option string as is to preserve case

# Function to convert a dictionary to .ini file format
def dict_to_ini(dictionary, ini_filename):
    config = CaseSensitiveConfigParser()

    for section, values in dictionary.items():
        if isinstance(values, dict):
            config[section] = {}
            for key, value in values.items():
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    config[section][key] = ', '.join(map(str, value))
                elif isinstance(value, str):
                    # Add single quotes around string values
                    config[section][key] = f"'{value}'"
                else:
                    config[section][key] = str(value)
        else:
            # Handle scalar values by placing them in a special section
            if isinstance(values, str):
                config["DEFAULT"][section] = f"'{values}'"
            else:
                config["DEFAULT"][section] = str(values)

    # Write the configuration to an .ini file
    with open(ini_filename, 'w') as ini_file:
        config.write(ini_file)

#%%
merged_config_2 = deepcopy(merged_config)

config_manager = ConfigManager()
config_manager.Convert(merged_config_2, framework='list', device=device)

#%%
dict_to_ini(merged_config_2, "C:/Users/akuznets/Projects/TipTorch/data/parameter_files/muse_nfm_multiple.ini")


#%% Initialize the model
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from tools.utils import SausageFeature

toy = TipTorch_new(merged_config, 'sum', device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)

#%%
sausage_absorber = SausageFeature(toy)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

toy.PSD_include['fitting'] = True
toy.PSD_include['WFS noise'] = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing'] = False
toy.PSD_include['chromatism'] = True
toy.PSD_include['diff. refract'] = True
toy.PSD_include['Moffat'] = False #Moffat_absorber

toy.to_float()
# toy.to_double()

inputs_tiptorch = {
    'F':   torch.ones ([N_src, N_wvl], device=toy.device),
    'dx':  torch.zeros([N_src, N_wvl], device=toy.device),
    'dy':  torch.zeros([N_src, N_wvl], device=toy.device),
    'bg':  torch.ones ([N_src, N_wvl], device=toy.device)*1e-6,
    'dn':  torch.ones (N_src, device=toy.device)*1.5,
    'Jx':  torch.ones ([N_src, N_wvl], device=toy.device)*10,
    'Jy':  torch.ones ([N_src, N_wvl], device=toy.device)*10,
    'Jxy': torch.ones ([N_src, 1], device=toy.device)*45
}
#%%
# if Moffat_absorber:
#     inputs_psfao = {
#         'amp':   torch.zeros(toy.N_src, device=toy.device),     # Phase PSD Moffat amplitude [rad²]
#         'b':     torch.zeros(toy.N_src, device=toy.device),     # Phase PSD background [rad² m²]
#         'alpha': torch.ones (toy.N_src, device=toy.device)*0.1, # Phase PSD Moffat alpha [1/m]
#         'beta':  torch.ones (toy.N_src, device=toy.device)*2,   # Phase PSD Moffat beta power law
#         'ratio': torch.ones (toy.N_src, device=toy.device),     # Phase PSD Moffat ellipticity
#         'theta': torch.zeros(toy.N_src, device=toy.device),     # Phase PSD Moffat angle
#     }
# else:
#     inputs_psfao = {}

inputs = inputs_tiptorch #| inputs_psfao
PSF_1 = toy(x=inputs)

#%%

plt.imshow(PSF_1[0,-1,...].abs().log10().cpu().numpy())


#%%
# PSF_1 = toy(x=inputs)
# PSF_1 = toy(x=inputs2)

# angle_correction = angle_search()
# print('Angle correction', angle_correction)
# config_file['telescope']['PupilAngle'] += angle_correction

# toy = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

times = []
N_runs = 10
for i in range(N_runs+1):
    start.record()

    toy.Update(reinit_grids=True, reinit_pupils=True)
    PSF_1 = toy(x=inputs)

    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

times = np.array(times[1:])
print(f'Mean time: {times.mean()} ms')

# PSF_1 = toy(x=inputs, phase_generator=lambda: sausage_absorber(0.25, -20))

#print(toy.EndTimer())
# PSF_DL = toy.DLPSF()

# draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80, scale='log')

aa = 0
plt.imshow(PSF_1[:,aa,...].mean(dim=0).abs().log10().cpu().numpy())
plt.show()
# plt.imshow(PSF_1[0,-1,...].log10().cpu().numpy())
# plt.show()
plt.imshow(PSF_0[:,aa,...].mean(dim=0).abs().log10().cpu().numpy())
plt.show()
# plt.imshow(PSF_0[0,-1,...].abs().log10().cpu().numpy())
# plt.show()

#%%
from tools.utils import safe_centroid, RadialProfile, wavelength_to_rgb

def calc_profile(data, xycen=None):
    xycen = safe_centroid(data) if xycen is None else xycen
    edge_radii = np.arange(data.shape[-1]//2)
    rp = RadialProfile(data, xycen, edge_radii)
    return rp.profile

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



plt.figure(figsize=(10, 8))

profis_0 = _radial_profiles(PSF_0[0, ...].cpu().numpy(), centers=None)
profis_1 = _radial_profiles(PSF_1[0, ...].cpu().numpy(), centers=None)
colors = [wavelength_to_rgb(toy.wvl[0][i]*1e9-100, show_invisible=True) for i in range(N_wvl)]

# colors2 = []
# for i in range(N_wvl):
#     buffo = wavelength_to_rgb(toy.wvl[0][i]*1e9-100, show_invisible=True)
#     buffy = [0,0,0]
#     for j in range(3):
#         buffy[j] = buffo[j]*0.5 + 0.5
#     colors2.append(buffy)

for i in range(PSF_0.shape[1]):
    plt.plot(profis_0[i], color=colors[i], alpha=0.5)
    plt.plot(profis_1[i], color=colors[i], alpha=0.2)
    
    
plt.yscale('symlog', linthresh=1e-5)
plt.xlim(0, 40)
plt.ylim(1e-5, 0.1)
plt.grid()

#%
a = [[51.4, 61.1, 70.3],[47.4,60.5,74.5],[43.6,60.0,78.1]]
a = np.array(a)
b = a[:,-1] - a[:,1]
c = a[:,1] - a[:,0]
r = (b+c)/2 # [pix]

pix_mas = 25

w = np.array([495, 722, 919])*1e-9
r_mas = r*pix_mas / rad2mas
D = 8

l_D = w / D

#%
a = [9.2, 10.7, 12.1, 14, 15.7, 16.5, 17.6]
b = [10.5, 12, 13.4, 15.3, 17.6, 18, 19.6]
w = [495, 560, 622, 722, 820, 853, 919]

# Fit line
w = np.array(w)
a = np.array(a)
b = np.array(b)

w = w.reshape(-1, 1)
a = a.reshape(-1, 1)
b = b.reshape(-1, 1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(w, a)
a_pred = reg.coef_ * w + reg.intercept_

reg = LinearRegression().fit(w, b)
b_pred = reg.coef_ * w + reg.intercept_

# a_pred -= a_pred[0]
# b_pred -= b_pred[0]

# plt.plot(w, a, 'o-', color='blue')
plt.plot(w, a_pred, 'o-', color='blue')
# plt.plot(w, b, 'o-', color='red')
plt.plot(w, b_pred, 'o-', color='red')




#%% PSF fitting (no early-stopping)
norm_F     = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg    = TransformSequence(transforms=[ Uniform(a=-5e-6, b=5e-6)])
norm_r0    = TransformSequence(transforms=[ Uniform(a=0,     b=1)   ])
norm_dxy   = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J     = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_Jxy   = TransformSequence(transforms=[ Uniform(a=-180,  b=180) ])
norm_dn    = TransformSequence(transforms=[ Uniform(a=0,     b=5)   ])

norm_sausage_pow = TransformSequence(transforms=[ Uniform(a=0, b=1)  ])

norm_amp   = TransformSequence(transforms=[ Uniform(a=0,     b=10)  ])
norm_b     = TransformSequence(transforms=[ Uniform(a=0,     b=0.1) ])
norm_alpha = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_beta  = TransformSequence(transforms=[ Uniform(a=0,     b=10)  ])
norm_ratio = TransformSequence(transforms=[ Uniform(a=0,     b=2)   ])
norm_theta = TransformSequence(transforms=[ Uniform(a=-np.pi/2, b=np.pi/2)])

transforms_dump = {
    'F':     norm_F,
    'bg':    norm_bg,
    'r0':    norm_r0,
    'dx':    norm_dxy,
    'dy':    norm_dxy,
    'Jx':    norm_J,
    'Jy':    norm_J,
    'Jxy':   norm_Jxy,
    'dn':    norm_dn,
    's_pow': norm_sausage_pow,
    'amp':   norm_amp,
    'b':     norm_b,
    'alpha': norm_alpha,
    'beta':  norm_beta,
    'ratio': norm_ratio,
    'theta': norm_theta
}

with open('../data/temp/muse_df_fitted_transforms.pickle', 'wb') as handle:
    df_transforms_store = {}
    for entry in transforms_dump:
        df_transforms_store[entry] = transforms_dump[entry].store()
    pickle.dump(df_transforms_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# The order of definition matters here!
transformer_dict = {
    'r0':  norm_r0,
    'F':   norm_F,
    'dx':  norm_dxy,
    'dy':  norm_dxy,
    'bg':  norm_bg,
    'dn':  norm_dn,
    'Jx':  norm_J,
    'Jy':  norm_J,
    'Jxy': norm_Jxy,
    's_pow': norm_sausage_pow,
    # 's_ang': norm_sausage_ang
}

if Moffat_absorber:
    transformer_dict.update({
        'amp'  : norm_amp,
        'b'    : norm_b,
        'alpha': norm_alpha,
        # 'beta' : norm_beta,
        # 'ratio': norm_ratio,
        # 'theta': norm_theta
    })


transformer = InputsTransformer(transformer_dict)

# Loop through the class attributes to get the initial values and their dimensions
if include_sausage:
    toy.s_pow = torch.zeros([toy.N_src,1], device=toy.device).float()

_ = transformer.stack({ attr: getattr(toy, attr) for attr in transformer_dict }) # to create index mapping

#%%
x0 = [\
    *([1.0,]*N_wvl), # F
    *([0.0,]*N_wvl), # dx
    *([0.0,]*N_wvl), # dy
    # 0.0,
    # 0.0,
    *([0.0,]*N_wvl), # bg
    0.7, # dn
    # -0.9,
    # -0.9,
    *([-0.9,]*N_wvl), # Jx
    *([-0.9,]*N_wvl), # Jy
    0.0, # Jxy
]

if Moffat_absorber:
    x0 += [\
        # PSFAO realm
         1.0,
        -0.5,
         0.3,
        # *([ 1.0,]*N_wvl),
        # *([ 0.0,]*N_wvl),
        # *([ 0.3,]*N_wvl)
    ]

if include_sausage:
    x0 += [0.9]

x0 = torch.tensor(x0).float().to(device).repeat(toy.N_src, 1)
x0 = torch.hstack([norm_r0.forward(toy.r0).unsqueeze(-1), x0])

#%%
def func(x_, include_list=None):
    x_torch = transformer.unstack(x_)
    
    if include_sausage and 's_pow' in x_torch:
        phase_func = lambda: sausage_absorber(toy.s_pow.flatten())
    else:
        phase_func = None
    
    if include_list is not None:
        return toy({ key: x_torch[key] for key in include_list }, None, phase_generator=phase_func)
    else:
        return toy(x_torch, None, phase_generator=phase_func)

#%
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1) * 2
mask = torch.tensor(mask_circle(PSF_1.shape[-1], 5)).view(1, 1, *PSF_1.shape[-2:]).to(device)
mask_inv = 1.0 - mask

# plt.imshow(mask.cpu().numpy().squeeze())

#%
# XX, YY = np.meshgrid(np.arange(PSF_0.shape[-1]), np.arange(PSF_0.shape[-2]))
# grad_map = np.sqrt( (XX - PSF_0.shape[-1]//2)**2 + (YY - PSF_0.shape[-2]//2)**2 )
# grad_map = grad_map / grad_map.max()
# grad_map += 0.25
# grad_map = grad_map / grad_map.max() * 2
# grad_map = torch.tensor(grad_map).float().to(device).unsqueeze(0).unsqueeze(0)

def loss_MSE(x_, include_list=None, mask_=1):
    diff = (func(x_, include_list) - PSF_0) * mask_ * wvl_weights ##
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1] #+ MAP*0.25
    # MAP = ((x_-dn_m)*mask_dn).sum()**2 /0.5**2 + ((x_-r0_m)*mask_r0).sum()**2 / 0.5**2
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_MAE(x_, include_list=None, mask_=1):
    diff = (func(x_, include_list) - PSF_0) * wvl_weights * mask_
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]


# include_MAE = ['r0', 'bg', 'dn', 's_pow']
# include_MSE = ['F', 'dx', 'dy', 'Jx', 'Jy', 'Jxy', 'r0', 'bg']
# include_all.remove('dn')

def loss_fn(x_): 
    return loss_MSE(x_) + loss_MAE(x_) * 0.4
    
#%%
# if toy.N_wvl > 1:
_ = func(x0)

result = minimize(loss_MAE, x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0 = result.x
result = minimize(loss_fn, x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0 = result.x

x_torch = transformer.unstack(x0)

phase_func = lambda: sausage_absorber(toy.s_pow.flatten()) if include_sausage else None
PSF_1 = toy(x_torch, None, phase_generator=phase_func)


'''
else:
    _ = func(x0)

    result = minimize(lambda x: loss_MAE(x, mask_=mask_inv), x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
    x0 = result.x

    x_torch = transformer.unstack(x0)
    x_torch['Jx'] = x_torch['Jx']*0 + 10
    x_torch['Jy'] = x_torch['Jy']*0 + 10
    result = minimize(lambda x: loss_MSE(x, include_MSE), x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
    # result = minimize(loss_fn, x0_tiptorch, max_iter=100, tol=1e-3, method='bfgs', disp=2)
    x0 = result.x

    x_torch = transformer.unstack(x0)

    phase_func = lambda: sausage_absorber(toy.s_pow.flatten()) if include_sausage else None
    PSF_1 = toy(x_torch, None, phase_generator=phase_func)
'''


#%%
from tools.utils import plot_radial_profiles_new

center = np.array([PSF_0.shape[-2]//2, PSF_0.shape[-1]//2])

if len(toy.wvl[0]) > 1:
    wvl_select = np.s_[0, 6, 12]

    draw_PSF_stack( PSF_0.cpu().numpy()[:, wvl_select, ...], PSF_1.cpu().numpy()[:, wvl_select, ...], average=True, crop=120 )
    
    PSF_disp = lambda x, w: (x[:,w,...]).cpu().numpy()

    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=60,  ax=ax[i] )
    plt.show()

else:
    draw_PSF_stack( PSF_0.cpu().numpy()[:,0,...], PSF_1.cpu().numpy()[:,0,...], average=True, crop=120 )
    
    plot_radial_profiles_new( PSF_0[0,0,...].cpu().numpy(),  PSF_1[0,0,...].cpu().numpy(),  'Data', 'TipTorch', centers=center, cutoff=60, title='Left PSF')
    plt.show()


'''
PSF_1 = torch.tensor(PSF_1s_).float().to(device).unsqueeze(0)
PSF_0 = PSF_0_.clone()
    
for i in range(N_wvl_):
    draw_PSF_stack(
        PSF_0.cpu().numpy()[0, i, ...],
        PSF_1.cpu().numpy()[0, i, ...],
        average=True, crop=120)
    plt.title(f'{(wvls_[0][i]*1e9):.2f} [nm]')
    plt.tight_layout()
    plt.savefig(f'C:/Users/akuznets/Desktop/MUSE_fits_new/PSF_{(wvls_[0][i]*1e9):.0f}.png')


for i in range(N_wvl_):
    A = PSF_0.cpu().numpy()[0, i, ...]
    B = PSF_1.cpu().numpy()[0, i, ...]
    plot_radial_profiles_new( A, B,  'Data', 'TipTorch', title=f'{(wvls_[0][i]*1e9):.2f} [nm]')
    plt.savefig(f'C:/Users/akuznets/Desktop/MUSE_fits_new/profiles_{(wvls_[0][i]*1e9):.0f}.png')
'''

#%%
# plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, (toy.Jy**2+toy.Jx**2).sqrt().cpu().numpy().flatten())
# plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, toy.Jy.detach().cpu().numpy().flatten())
plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, toy.Jx.detach().cpu().numpy().flatten())
plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, toy.dx.detach().cpu().numpy().flatten())
# plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, toy.F.cpu().numpy().flatten())
# plt.plot(toy.wvl.squeeze().cpu().numpy().flatten()*1e9, toy.Jxy.cpu().numpy().flatten())


#%%
line_norms = {
    'F' : (1e6, 1),
    'bg': (1e6, 1e-6),
    'dx': (1e6, 1e0),
    'dy': (1e6, 1e-1),
    # 'Jx': (1e6, 1e2),
    # 'Jy': (1e6, 1e2),
}

models_dict = {key: QuadraticModel(toy.wvl, line_norms[key]) for key in line_norms.keys() }

inp_dict_2 = {}
for key in transformer.transforms.keys(): 
    if key == 's_pow':
        inp_dict_2[key] = torch.zeros([1,1], device=device).float()
    else:
        param_val = getattr(toy, key).detach().abs() if key in ['F', 'Jx', 'Jy'] else getattr(toy, key).detach().clone()
    if key in line_norms:
        inp_dict_2[key] = torch.tensor(models_dict[key].fit(param_val), device=device)
    else:
        inp_dict_2[key] = param_val

compressor = InputsCompressor(transformer.transforms, models_dict)

x0_compressed = compressor.stack(inp_dict_2).unsqueeze(0) * 0.1
_ = compressor.stack(inp_dict_2).unsqueeze(0)

x0_compressed_backup = x0_compressed.clone()

#%
# test_model = PolyModel(toy.wvl, line_norms['Jx'])
# popt = test_model.fit(toy.Jx)
# 
# pred = test_model(popt)
# 
# plt.plot(toy.wvl.cpu().numpy().flatten(), toy.Jx.cpu().numpy().flatten())
# plt.plot(toy.wvl.cpu().numpy().flatten(), pred.cpu().numpy().flatten())


#%
def func2(x_, include_list=None):
    x_torch = compressor.unstack(x_)
    
    if include_sausage and 's_pow' in x_torch:
        phase_func = lambda: sausage_absorber(toy.s_pow.flatten())
    else:
        phase_func = None
    
    if include_list is not None:
        return toy({ key: x_torch[key] for key in include_list }, None, phase_generator=phase_func)
    else:
        return toy(x_torch, None, phase_generator=phase_func)


x0 = torch.tensor(x0).float().to(device).unsqueeze(0)


def loss_MSE2(x_):
    diff = (func2(x_) - PSF_0) * wvl_weights
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    # return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_MAE2(x_):
    diff = (func2(x_) - PSF_0) * wvl_weights
    # return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_fn2(x_):
    return loss_MSE2(x_) + loss_MAE2(x_) * 0.4


#%%
result = minimize(loss_MAE2, x0_compressed, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0_compressed = result.x

result = minimize(loss_fn2, x0_compressed, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0_compressed = result.x


#%%
PSF_1 = func2(x0_compressed).detach()

if len(toy.wvl[0]) > 1:
    wvl_select = np.s_[0, 6, 12]

    draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )
    
    PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()
    
    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=40,  ax=ax[i] )
    plt.show()

else:
    draw_PSF_stack( PSF_0.cpu().numpy()[:,0,...], PSF_1.cpu().numpy()[:,0,...], average=True, crop=120 )
    
    plot_radial_profiles_new( PSF_0[0,0,...].cpu().numpy(),  PSF_1[0,0,...].cpu().numpy(),  'Data', 'TipTorch', centers=center, cutoff=20, title='Left PSF')
    plt.show()


#%%
# decomposed = depack_line(x0_compressed)

destacked_1 = transformer.unstack(x0.clone())
destacked_2 = compressor.unstack(x0_compressed)

Ffs_2 = destacked_2['F'].cpu().flatten()
bgs_2 = destacked_2['bg'].cpu().flatten()
Jxs_2 = destacked_2['Jx'].cpu().flatten()
Jys_2 = destacked_2['Jy'].cpu().flatten()

Jxs_1 = destacked_1['Jx'].cpu().flatten()
Jys_1 = destacked_1['Jy'].cpu().flatten()

if Jxs_1.numel() == 1:
    Jxs_1 = Jxs_1.repeat(N_wvl)
    Jys_1 = Jys_1.repeat(N_wvl)

plt.plot(toy.wvl.cpu().numpy().flatten(), Ffs_2)
plt.plot(toy.wvl.cpu().numpy().flatten(), destacked_1['F'].cpu().numpy().flatten())
plt.show()

plt.plot(toy.wvl.cpu().numpy().flatten(), bgs_2)
plt.plot(toy.wvl.cpu().numpy().flatten(), destacked_1['bg'].cpu().numpy().flatten())
plt.show()

plt.plot(toy.wvl.cpu().numpy().flatten(), Jxs_2)
plt.plot(toy.wvl.cpu().numpy().flatten(), Jxs_1)
plt.show()

# plt.plot(toy.wvl.cpu().numpy().flatten(), Jys)
# plt.plot(toy.wvl.cpu().numpy().flatten(), destacked_1['Jy'].cpu().numpy().flatten())
# plt.show()


#%%
plt.plot(to_np(toy.wvl), to_np(toy.F))
plt.plot(to_np(toy.wvl), y_pred)
plt.show()

#%%
r0s = []
dns = []
Jxs = []
Jys = []
Fs  = []
bgs = []

for x_ in x0s:
    x_torch = transformer.unstack(x_)
    r0s.append(x_torch['r0'].abs().item())
    dns.append(x_torch['dn'].abs().item())
    Jxs.append(x_torch['Jx'].abs().item())
    Jys.append(x_torch['Jy'].abs().item())
    Fs.append(x_torch['F'].abs().item())
    bgs.append(x_torch['bg'].abs().item())

r0s = np.array(r0s)
dns = np.array(dns)
Jxs = np.array(Jxs) 
Jys = np.array(Jys)
Fs  = np.array(Fs)
bgs = np.array(bgs)


fig, ax = plt.subplots(2, 3, figsize=(12, 7))
wavilo = np.array(wvls_[0]) * 1e9

ax[0,0].plot(wavilo, r0s, label='r0')
ax[0,0].set_ylim([0, r0s.max()*1.2])
ax[0,0].set_title('r0')
ax[0,0].grid(True)

ax[1,0].plot(wavilo, dns, label='dn')
ax[1,0].set_ylim([0, dns.max()*1.2])
ax[1,0].set_title('dn')
ax[1,0].grid(True)
ax[1,0].set_xlabel('Wavelength [nm]')

ax[0,1].plot(wavilo, Jxs, label='Jx')
ax[0,1].set_ylim([0, Jxs.max()*1.2])
ax[0,1].set_title('Jx')
ax[0,1].grid(True)

ax[1,1].plot(wavilo, Jys, label='Jy')
ax[1,1].set_ylim([0, Jys.max()*1.2])
ax[1,1].set_title('Jy')
ax[1,1].grid(True)
ax[1,1].set_xlabel('Wavelength [nm]')

ax[0,2].plot(wavilo, Fs, label='F')
ax[0,2].set_ylim([0, Fs.max()*1.2])
ax[0,2].set_title('F')
ax[0,2].grid(True)

ax[1,2].plot(wavilo, bgs, label='bg')
ax[1,2].set_ylim([0, bgs.max()*1.2])
ax[1,2].set_title('bg')
ax[1,2].grid(True)
ax[1,2].set_xlabel('Wavelength [nm]')

plt.tight_layout()
plt.savefig('C:/Users/akuznets/Desktop/MUSE_fits_new/params.png')

#%%
# for entry in x0_new:
#     item_ = x0_new[entry]
    
#     if len(item_.shape) == 0:
#         print(f'{entry}: {item_.item():.3f}')
#     else:
#         print(entry, end=': ')
#         for x in item_.cpu().numpy().tolist():
#             print(f'{x:.3f}',  end=' ')
#         print('')


#%%
def GetNewPhotons():
    WFS_noise_var = toy.dn + toy.NoiseVariance(toy.r0.abs())

    N_ph_0 = toy.WFS_Nph.clone()

    def func_Nph(x):
        toy.WFS_Nph = x
        var = toy.NoiseVariance(toy.r0.abs())
        return (WFS_noise_var-var).flatten().abs().sum()

    result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
    toy.WFS_Nph = N_ph_0.clone()

    return result_photons.x

Nph_new = GetNewPhotons()

print(toy.WFS_Nph.item(), Nph_new.item())


