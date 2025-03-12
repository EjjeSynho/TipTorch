#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tools.utils import plot_radial_profiles_new, plot_radial_profiles_relative, SR, draw_PSF_stack, rad2mas, mask_circle#, register_hooks
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF
from project_globals import MUSE_DATA_FOLDER, device
from torchmin import minimize
# from astropy.stats import sigma_clipped_stats
# from tools.parameter_parser import ParameterParser
# from tools.config_manager import ConfigManager
from data_processing.normalizers import Uniform, PolyModel, InputsCompressor, InputsManager
from tqdm import tqdm
from tools.utils import OptimizableLO, SausageFeature, PupilVLT, GradientLoss, CombinedLoss
from warnings import warn

#%%
derotate_PSF    = True
Moffat_absorber = True
fit_LO          = True
LO_map_size     = 31

#%%
with open(MUSE_DATA_FOLDER+'/muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

# gave_NaNs = [21, 60, 103, 148, 167, 240, 302, 319, 349]
# 411, 410, 409, 405, 146, 296, 276, 395, 254, 281, 343, 335

# 21 - fits okay
# 240 - fits okay, but super intense background and blurry
# 168 - fits okay, but super intense background
# 349 - fits okay, but blurry
# 302, 319 - wrong rotation, fits okay

# sample = LoadMUSEsampleByID(402)
sample = LoadMUSEsampleByID(394) # -- Blurry, check it later
# sample = LoadMUSEsampleByID(296) # -- sosig
# sample = LoadMUSEsampleByID(395)

PSF_0, PSF_std, norms, bgs = LoadImages(sample)

config_file, PSF_0 = GetConfig(sample, PSF_0)
N_wvl = PSF_0.shape[1]

PSF_0   = PSF_0[...,1:,1:]
PSF_std = PSF_std[...,1:,1:]

# var_mask = np.clip(1 / var_mask, 0, 1e6)
# var_mask = var_mask / var_mask.max(axis=(-1,-2), keepdims=True)

config_file['sensor_science']['FieldOfView'] = PSF_0.shape[-1]
config_file['NumberSources'] = config_file['NumberSources'].int().item()

if derotate_PSF:
    PSF_0   = rotate_PSF(PSF_0, -sample['All data']['Pupil angle'].item())
    PSF_std = rotate_PSF(PSF_std, -sample['All data']['Pupil angle'].item())
    config_file['telescope']['PupilAngle'] = 0 # reset the angle value

max_threshold = 5e6
X_std_inv = torch.nan_to_num(1.0 / PSF_std)
X_std_inv = torch.clip(X_std_inv, 0, max_threshold)**2 / max_threshold**2

#%% Initialize the model
# from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          Moffat_absorber
}
toy = TipTorch_new(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
toy.apodizer = toy.make_tensor(1.0)

LO_basis = OptimizableLO(toy, ignore_pupil=False)

#%% Initialize inputs manager and set up parameters
inputs_manager = InputsManager()

# Initialize normalizers/transforms
norm_F           = Uniform(a=0.0,   b=1.0)
norm_bg          = Uniform(a=-5e-6, b=5e-6)
norm_r0          = Uniform(a=0,     b=1)
norm_dxy         = Uniform(a=-1,    b=1)
norm_J           = Uniform(a=0,     b=50)
norm_Jxy         = Uniform(a=-180,  b=180)
norm_dn          = Uniform(a=0,     b=5)
norm_amp         = Uniform(a=0,     b=10)
norm_b           = Uniform(a=0,     b=0.1)
norm_alpha       = Uniform(a=-1,    b=10)
norm_beta        = Uniform(a=0,     b=2)
norm_ratio       = Uniform(a=0,     b=2)
norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
norm_sausage_pow = Uniform(a=0, b=1)
norm_LO          = Uniform(a=-100, b=100)

# Add base parameters
inputs_manager.add('r0',  torch.tensor([toy.r0.item()]), norm_r0)
inputs_manager.add('F',   torch.tensor([[1.0,]*N_wvl]),  norm_F)
inputs_manager.add('dx',  torch.tensor([[0.0,]*N_wvl]),  norm_dxy)
inputs_manager.add('dy',  torch.tensor([[0.0,]*N_wvl]),  norm_dxy)
inputs_manager.add('bg',  torch.tensor([[0.0,]*N_wvl]),  norm_bg)
inputs_manager.add('dn',  torch.tensor([0.25]),          norm_dn)
inputs_manager.add('Jx',  torch.tensor([[2.5,]*N_wvl]),  norm_J)
inputs_manager.add('Jy',  torch.tensor([[2.5,]*N_wvl]),  norm_J)
inputs_manager.add('Jxy', torch.tensor([[0.0]]),         norm_Jxy)

# Add Moffat parameters if needed
if Moffat_absorber:
    inputs_manager.add('amp',   torch.tensor([0.0]), norm_amp)
    inputs_manager.add('b',     torch.tensor([0.0]), norm_b)
    inputs_manager.add('alpha', torch.tensor([4.5]), norm_alpha)
    inputs_manager.add('beta',  torch.tensor([2.5]), norm_beta)
    inputs_manager.add('ratio', torch.tensor([1.0]), norm_ratio)
    inputs_manager.add('theta', torch.tensor([0.0]), norm_theta)

if fit_LO:
    inputs_manager.add('LO_coefs', torch.zeros([1, LO_map_size**2]), norm_LO)

inputs_manager.to(device)
inputs_manager.to_float()

#%%
def func(x_, include_list=None):
    x_torch = inputs_manager.unstack(x_)
    phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_map_size, LO_map_size))

    if include_list is not None:
        return toy({ key: x_torch[key] for key in include_list }, None, phase_generator=phase_func)
    else:
        return toy(x_torch, None, phase_generator=phase_func)

#%
wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)
wvl_weights = N_wvl / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

mask = torch.tensor(mask_circle(PSF_0.shape[-1], 5)).view(1, 1, *PSF_0.shape[-2:]).to(device)
mask_inv = 1.0 - mask

# plt.imshow(mask_inv.cpu()[0,0,...])
# plt.show()

grad_loss_fn = GradientLoss(p=1, reduction='mean')


def loss_fn1(x_):
    diff1 = (func(x_)-PSF_0) * wvl_weights
    mse_loss = (diff1 * 4000).pow(2).mean()
    mae_loss = (diff1 * 32000).abs().mean()
    grad_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_map_size, LO_map_size)) * 5e-5
    return mse_loss + mae_loss + grad_loss


def loss_fn2(x_):
    diff1 = (func(x_)-PSF_0) * wvl_weights
    mse_loss = diff1.pow(2).sum() / PSF_0.shape[-2] / PSF_0.shape[-1] * 1250
    mae_loss = diff1.abs().sum()  / PSF_0.shape[-2] / PSF_0.shape[-1] * 2500
    grad_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_map_size, LO_map_size)) * 5e-5
    return mse_loss + mae_loss + grad_loss


def loss_fn3(x_):
    diff1 = (func(x_ )-PSF_0) * wvl_weights
    mse_loss = (diff1 * 1500).pow(2).sum()
    mae_loss = (diff1 * 2500).abs().sum()
    return (mse_loss + mae_loss) / PSF_0.shape[-2] / PSF_0.shape[-1]


def loss_fn4(x_):
    diff1 = (func(x_ )-PSF_0) * wvl_weights
    sqrt_loss = (diff1 * 1500).abs().sqrt().sum()
    return sqrt_loss / PSF_0.shape[-2] / PSF_0.shape[-1]

# x_ = inputs_manager.stack()
# print(loss_fn4(x_))

#%%
def minimize_params(loss_fn, include_list, exclude_list, max_iter, verbose=True):
    if len(include_list) > 0:
        inputs_manager.set_optimizable(include_list, True)
    else:
        warn('include_list is empty')
        
    inputs_manager.set_optimizable(exclude_list, False)

    print(inputs_manager)

    result = minimize(loss_fn, inputs_manager.stack(), max_iter=max_iter, tol=1e-4, method='l-bfgs', disp= 2 if verbose else 0)
    OPD_map = inputs_manager['LO_coefs'].view(1, LO_map_size, LO_map_size)[0].detach().cpu().numpy()

    return result.x, func(result.x), OPD_map


include_general = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'LO_coefs'] + (['amp', 'alpha', 'beta'] if Moffat_absorber else [])
exclude_general = ['ratio', 'theta', 'b'] if Moffat_absorber else []

include_LO = ['LO_coefs']
exclude_LO = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy'] + ['ratio', 'theta', 'b', 'amp', 'alpha', 'beta'] if Moffat_absorber else []

#%%
x0, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 200)
x1, PSF_1, OPD_map = minimize_params(loss_fn2, include_LO, exclude_LO, 100)

plt.imshow(OPD_map)
plt.show()

x2, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 100)


#%%
from tools.utils import plot_radial_profiles_new

center = np.array([PSF_0.shape[-2]//2, PSF_0.shape[-1]//2])

if len(toy.wvl[0]) > 1:
    wvl_select = np.s_[0, 6, 12]

    draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )

    PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=30,  y_min=3e-2, linthresh=1e-2, ax=ax[i] )
    plt.show()

else:
    draw_PSF_stack( PSF_0.cpu().numpy()[:,0,...], PSF_1.cpu().numpy()[:,0,...], average=True, crop=120 )

    plot_radial_profiles_new( PSF_0[0,0,...].cpu().numpy(),  PSF_1[0,0,...].cpu().numpy(),  'Data', 'TipTorch', centers=center, cutoff=60, title='Left PSF')
    plt.show()

#%%
line_norms = {
    'F' : (1e6, 1),
    'bg': (1e6, 1e-6),
    'dx': (1e6, 1e0),
    'dy': (1e6, 1e-1),
    # 'Jx': (1e6, 1e2),
    # 'Jy': (1e6, 1e2),
}

models_dict = {key: PolyModel(toy.wvl, line_norms[key]) for key in line_norms.keys() }

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



