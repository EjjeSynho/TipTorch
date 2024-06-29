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
from PSF_models.TipToy_MUSE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky
from project_globals import MUSE_DATA_FOLDER, device
from torchmin import minimize
from astropy.stats import sigma_clipped_stats

from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager

from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer, LineModel, InputsCompressor
    
from data_processing.MUSE_read_preproc_old import MUSEcube
from tqdm import tqdm

#%% My new process

sample_name = '411_M.MUSE.2024-05-06T16-57-21.086.pickle'
# sample_name = '410_M.MUSE.2024-05-06T16-44-36.486.pickle'
# sample_name = '409_M.MUSE.2024-05-02T22-51-44.900.pickle'

with open(MUSE_DATA_FOLDER+'DATA_reduced/'+sample_name, 'rb') as f:
    sample = pickle.load(f)

#%
# %matplotlib qt
PSF_0_ = sample['images']['cube'][:,1:,1:]
PSF_0_ /= PSF_0_.sum(axis=(-1,-2), keepdims=True)
PSF_0_ = torch.tensor(PSF_0_).float().unsqueeze(0).to(device)

# plt.imshow(PSF_0[-1,:,...].mean(dim=0).log().cpu().numpy())
# plt.show()

wvls_ = [(sample['spectral data']['wvls binned']*1e-9).tolist()]
N_wvl_ = len(wvls_[0])
#%%

# 'images'
# 'IRLOS data'
# 'LGS data'
# 'MUSE header data'
# 'Raw Cn2 data'
# 'Raw atm data'
# 'Raw ASM data'
# 'ASM data'
# 'MASS-DIMM data',
# 'DIMM data',
# 'SLODAR data',
# 'All data',
# 'spectral data'

# x0s = []
# PSF_1s = []

# Manage config files
# for wvl_id in tqdm(range(N_wvl_)):

# wvl_id = 0
# wvls = [wvls_[0][wvl_id]]
# N_wvl = 1
# PSF_0 = PSF_0_[:,wvl_id,...].unsqueeze(1)

wvls  = wvls_
PSF_0 = PSF_0_
N_wvl = N_wvl_

config_manager = ConfigManager()
config_file    = ParameterParser('../data/parameter_files/muse_ltao.ini').params
# merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample, *config_loader()) for sample in data_samples])

#%
# For NFM it's save to assume gound layer to be below 2 km, for WFM it's lower than that
h_GL = 2000

Cn2_weights = np.array([sample['Raw Cn2 data'][f'CN2_FRAC_ALT{i}'].item() for i in range(1, 9)])
altitudes   = np.array([sample['Raw Cn2 data'][f'ALT{i}'].item() for i in range(1, 9)])*100 # in meters

Cn2_weights_GL = Cn2_weights[altitudes < h_GL]
altitudes_GL   = altitudes  [altitudes < h_GL]

GL_frac  = Cn2_weights_GL.sum()  # Ground layer fraction
Cn2_w_GL = np.interp(h_GL, altitudes, Cn2_weights)


config_file['NumberSources'] = 1

config_file['telescope']['TelescopeDiameter'] = 8.0
config_file['telescope']['ZenithAngle'] = [90.0 - sample['MUSE header data']['Tel. altitude'].item()]
config_file['telescope']['Azimuth']     = [sample['MUSE header data']['Tel. azimuth'].item()]
config_file['telescope']['PupilAngle']  = 22+5.5

config_file['atmosphere']['Seeing'] = [sample['MUSE header data']['Seeing (header)'].item()]
config_file['atmosphere']['L0'] = [sample['Raw Cn2 data']['L0Tot'].item()]
config_file['atmosphere']['Cn2Weights'] = [[GL_frac, 1-GL_frac]]
config_file['atmosphere']['Cn2Heights'] = [[0, h_GL]]
config_file['atmosphere']['WindSpeed']     = [[sample['MUSE header data']['Wind speed (header)'].item(),]*2]
config_file['atmosphere']['WindDirection'] = [[sample['MUSE header data']['Wind dir (header)'].item(),]*2]
config_file['sources_science']['Wavelength'] = wvls

config_file['sources_LO']['Wavelength'] = (1215+1625)/2.0 * 1e-9

config_file['sensor_science']['PixelScale'] = sample['MUSE header data']['Pixel scale (science)'].item()
config_file['sensor_science']['FieldOfView'] = PSF_0.shape[-1]

LGS_ph = [[sample['All data'][f'LGS{i} photons, [photons/m^2/s]'].item() / 1240e3 for i in range(1,5)]]

# LGS_ph = [[200,]*4]
# config_file['DM']['DmPitchs'] = [config_file['DM']['DmPitchs'][0]*1.25]
    
config_file['sensor_HO']['NumberPhotons'] = LGS_ph
config_file['sensor_HO']['SizeLenslets']  = config_file['sensor_HO']['SizeLenslets'][0]
# config_file['sensor_HO']['NoiseVariance'] = 4.5

IRLOS_ph_per_subap_per_frame = \
    sample['IRLOS data']['IRLOS photons, [photons/s/m^2]'].item() / sample['IRLOS data']['frequency'].item() / 4

config_file['sensor_LO']['PixelScale'] = sample['IRLOS data']['plate scale, [mas/pix]'].item()
config_file['sensor_LO']['NumberPhotons'] = [IRLOS_ph_per_subap_per_frame]
config_file['sensor_LO']['SigmaRON']      = sample['IRLOS data']['RON, [e-]'].item()
config_file['sensor_LO']['Gain']          = [sample['IRLOS data']['gain'].item()]

config_file['RTC']['SensorFrameRate_LO'] = [sample['IRLOS data']['frequency'].item()]
config_file['RTC']['SensorFrameRate_HO'] = [config_file['RTC']['SensorFrameRate_HO']]

config_file['RTC']['LoopDelaySteps_HO'] = [config_file['RTC']['LoopDelaySteps_HO']]
config_file['RTC']['LoopGain_HO'] = [config_file['RTC']['LoopGain_HO']]

config_file['sensor_HO']['ClockRate'] = np.mean([config_file['sensor_HO']['ClockRate']])

config_manager.Convert(config_file, framework='pytorch', device=device)


#% Fernandos's process
'''
# Load image
data_dir = path.normpath('C:/Users/akuznets/Data/MUSE/DATA_Fernando/')
listData = os.listdir(data_dir)
sample_id = 5
sample_name = listData[sample_id]
path_im = path.join(data_dir, sample_name)
angle = np.zeros([len(listData)])
angle[0] = -46
angle[5] = -44
angle = angle[sample_id]

data_cube = MUSEcube(path_im, crop_size=200, angle=angle)
im, _, wvl = data_cube.Layer(5)
obs_info = dict( data_cube.obs_info )

PSF_0 = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(device)
PSF_0 /= PSF_0.sum(dim=(-1,-2), keepdim=True)


config_manager = ConfigManager()
config_file    = ParameterParser('../data/parameter_files/muse_ltao.ini').params

config_file['NumberSources'] = 1

config_file['telescope']['TelescopeDiameter'] = 8.0
config_file['telescope']['ZenithAngle'] = [90.0-obs_info['TELALT']]
config_file['telescope']['Azimuth']     = [obs_info['TELAZ']]

config_file['atmosphere']['Cn2Weights'] = [config_file['atmosphere']['Cn2Weights']]
config_file['atmosphere']['Cn2Heights'] = [config_file['atmosphere']['Cn2Heights']]

config_file['atmosphere']['Seeing']        = [obs_info['SPTSEEIN']]
config_file['atmosphere']['L0']            = [obs_info['SPTL0']]
config_file['atmosphere']['WindSpeed']     = [[obs_info['WINDSP'],] * 2]
config_file['atmosphere']['WindDirection'] = [[obs_info['WINDIR'],] * 2]

config_file['sensor_science']['Zenith']      = [90.0-obs_info['TELALT']]
config_file['sensor_science']['Azimuth']     = [obs_info['TELAZ']]
config_file['sensor_science']['PixelScale']  = 25
config_file['sensor_science']['FieldOfView'] = im.shape[0]

config_file['sources_science']['Wavelength'] = [wvl]

config_file['sources_LO']['Wavelength'] = (1215+1625)/2.0 * 1e-9

config_file['sensor_HO']['NoiseVariance'] = 4.5
config_file['sensor_HO']['SizeLenslets']  = config_file['sensor_HO']['SizeLenslets'][0]
# config_file['sensor_HO']['NumberPhotons'] = [[200,]*4]
config_file['sensor_HO']['ClockRate'] = np.mean([config_file['sensor_HO']['ClockRate']])

config_file['RTC']['SensorFrameRate_HO'] = [config_file['RTC']['SensorFrameRate_HO']]
config_file['RTC']['LoopDelaySteps_HO']  = [config_file['RTC']['LoopDelaySteps_HO']]
config_file['RTC']['LoopGain_HO']        = [config_file['RTC']['LoopGain_HO']]

config_manager.Convert(config_file, framework='pytorch', device=device)
'''

#%% Initialize the model
from PSF_models.TipToy_MUSE_multisrc import TipTorch
# from tools.utils import LWE_basis

toy = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=False, oversampling=1)

toy.PSD_include['fitting'] = True
toy.PSD_include['WFS noise'] = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing'] = True
toy.PSD_include['chromatism'] = True
toy.PSD_include['Moffat'] = False   

toy.to_float()
# toy.to_double()

inputs = {
    # 'r0':  torch.tensor([0.09561153075597545], device=toy.device),
    'F':   torch.tensor([[1.0,]*N_wvl], device=toy.device),
    # 'L0':  torch.tensor([47.93], device=toy.device),
    'dx':  torch.tensor([[0.0,]*N_wvl], device=toy.device),
    'dy':  torch.tensor([[0.0,]*N_wvl], device=toy.device),
    'bg':  torch.tensor([[1e-06,]*N_wvl], device=toy.device),
    # 'dn':  torch.tensor([4.5], device=toy.device),
    'Jx':  torch.tensor([[10,]*N_wvl], device=toy.device),
    'Jy':  torch.tensor([[20,]*N_wvl], device=toy.device),
    'Jxy': torch.tensor([[45]], device=toy.device)
}

# inputs2 = {
#     'amp':   torch.ones (toy.N_src, device=toy.device)*4.0,  # Phase PSD Moffat amplitude [rad²]
#     'b':     torch.ones (toy.N_src, device=toy.device)*0.01, # Phase PSD background [rad² m²]
#     'alpha': torch.ones (toy.N_src, device=toy.device)*0.1,  # Phase PSD Moffat alpha [1/m]
#     'beta':  torch.ones (toy.N_src, device=toy.device)*2,    # Phase PSD Moffat beta power law
#     'ratio': torch.ones (toy.N_src, device=toy.device),      # Phase PSD Moffat ellipticity
#     'theta': torch.zeros(toy.N_src, device=toy.device),      # Phase PSD Moffat angle
# }

PSF_1 = toy(x=inputs)
# PSF_1 = toy(x=inputs2)

#print(toy.EndTimer())
# PSF_DL = toy.DLPSF()

# draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80, scale='log')
plt.imshow(PSF_1[0,-1,...].log10().cpu().numpy())


#%% PSF fitting (no early-stopping)
norm_F     = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg    = TransformSequence(transforms=[ Uniform(a=-5e-6, b=5e-6)])
norm_r0    = TransformSequence(transforms=[ Uniform(a=0,     b=0.5) ])
norm_dxy   = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J     = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_Jxy   = TransformSequence(transforms=[ Uniform(a=-180,  b=180) ])
norm_dn    = TransformSequence(transforms=[ Uniform(a=0,     b=10)  ])

# norm_amp   = TransformSequence(transforms=[ Uniform(a=0,     b=5)   ])
# norm_b     = TransformSequence(transforms=[ Uniform(a=0,     b=1)   ])
# norm_alpha = TransformSequence(transforms=[ Uniform(a=0,     b=5)   ])
# norm_beta  = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
# norm_ratio = TransformSequence(transforms=[ Uniform(a=0,     b=2)   ])
# norm_theta = TransformSequence(transforms=[ Uniform(a=-np.pi/2, b=np.pi/2)])

# The order matters here!
transformer = InputsTransformer({
    'r0':  norm_r0,
    'F':   norm_F,
    'dx':  norm_dxy,
    'dy':  norm_dxy,
    'bg':  norm_bg,
    'dn':  norm_dn,
    'Jx':  norm_J,
    'Jy':  norm_J,
    'Jxy': norm_Jxy,
    
    # 'amp'  : norm_amp,
    # 'b'    : norm_b,
    # 'alpha': norm_alpha,
    # 'beta' : norm_beta,
    # 'ratio': norm_ratio,
    # 'theta': norm_theta
})

# Loop through the class attributes to get the initial values and dimensions
# inp_dict = {}
# ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']:
inp_dict = { attr: getattr(toy, attr) for attr in transformer.transforms.keys() }
_ = transformer.stack(inp_dict) # to create index mapping

#%%
x0 = [\
    # TipTorch realm
    norm_r0.forward(toy.r0).item(),
    *([1.0,]*N_wvl),
    *([0.0,]*N_wvl),
    *([0.0,]*N_wvl),
    *([0.0,]*N_wvl),
    0.5,
    *([-0.9,]*N_wvl),
    *([-0.1,]*N_wvl),
    0.25,

    #PSFAO realm
    # norm_amp.forward(toy.amp).item(),
    # norm_b.forward(toy.b).item(),
    # norm_alpha.forward(toy.alpha).item(),
    # norm_beta.forward(toy.beta).item(),
    # norm_ratio.forward(toy.ratio).item(),
    # norm_theta.forward(toy.theta).item(),
]

# x0 = transformer.stack(inputs)
x0 = torch.tensor(x0).float().to(device).unsqueeze(0)

#%
def func(x_):
    x_torch = transformer.destack(x_)
    return toy(x_torch)


wvl_weights = torch.linspace(1.0, 0.5, N_wvl_).to(device).view(1, N_wvl_, 1, 1) * 2

mask = torch.tensor(mask_circle(PSF_1.shape[-1], 20)).view(1, 1, *PSF_1.shape[-2:]).to(device)
mask_inv = 1.0 - mask
#%
def loss_MSE(x_):
    diff = (func(x_) - PSF_0) * wvl_weights
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    # return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_MAE(x_):
    diff = (func(x_) - PSF_0) * wvl_weights
    # return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_fn(x_):
    diff = (func(x_) - PSF_0)*wvl_weights
    # return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    # return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    return ( diff.pow(2)*200 + mask_inv*diff.abs()*0.5 ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

#%
# PSF_1 = func(x0)
# plt.imshow((PSF_1[0,0,...]).log10().cpu().numpy())

#%%
result = minimize(loss_MSE, x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0 = result.x

# result = minimize(loss_MSE, x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
# x0 = result.x
# result = minimize(loss_MAE, x0, max_iter=20, tol=1e-3, method='bfgs', disp=2)
# result = minimize(loss_MSE, x0, max_iter=20, tol=1e-3, method='bfgs', disp=2)
# result = minimize(loss_fn, x0, max_iter=100, tol=1e-3, method='bfgs', disp=2)
# x0_buf = x0.clone()

# x0s.append(x0_buf)
# PSF_1s.append(func(x0_buf).clone())

# x0s_ = torch.stack(x0s).squeeze().detach().cpu().numpy()
# PSF_1s_ = torch.stack(PSF_1s).squeeze().detach().cpu().numpy()
# np.save('../data/temp/x0s.npy', x0s_)
# np.save('../data/temp/PSF_1s_.npy', x0s_)


#%%
from tools.utils import plot_radial_profiles_new

many_wvls = len(toy.wvl[0]) > 1

PSF_1 = func(x0) 
center = np.array([PSF_0.shape[-2]//2, PSF_0.shape[-1]//2])

if many_wvls:
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
from sklearn.linear_model import LinearRegression

X = toy.wvl.squeeze().cpu().numpy().flatten().reshape(-1, 1) * 1e9
y = toy.bg.detach().cpu().numpy().flatten() * 1e6

reg = LinearRegression().fit(X, y)

y_pred = reg.coef_ * X + reg.intercept_

print(reg.coef_, reg.intercept_)

plt.plot(X, y)
plt.plot(X, y_pred)
plt.show()



#%%
# Fit parabolic function to the background
from scipy.optimize import curve_fit

line = lambda λ, k, y_min, λ_min, A, B: (A*k*(λ-λ_min) + y_min)*B
# parabola = lambda x, a, b, c: a*x**2 + b*x + c

to_np = lambda x: x.flatten().cpu().numpy()

def get_init_params(x, y, norm_A, norm_B):
    x_ = to_np(x)
    y_ = to_np(y)
    x_min = x_.min()
    
    func = lambda x, k, y_min: line(x, k, y_min, x_min, norm_A, norm_B)
    popt, pcov = curve_fit(func, x_, y_, p0=[1e-6, 1e-6])
    y_pred = func(x_, *popt)
    
    return popt, y_pred

line_norms = {
    'F' : (1e6, 1),
    'dx': (1e6, 1e0),
    'dy': (1e6, 1e-1),
    'bg': (1e6, 1e-6),
    'Jx': (1e6, 1e2),
    'Jy': (1e6, 1e2),
}

init_params, y_preds = {}, {}

for key, val in line_norms.items():
    param_val = getattr(toy, key)
    param_val = param_val.abs() if key in ['F', 'Jx', 'Jy'] else getattr(toy, key)
    init_params[key], y_pred = get_init_params(toy.wvl, param_val, *val)
    print(key, init_params[key])
    y_preds[key] = y_pred


slices = {}
current_index = 0

x0_compressed = []
for key in inp_dict.keys():
    if key in init_params:
        x0_compressed.append(init_params[key][0])
        x0_compressed.append(init_params[key][1])
        slices[key] = slice(current_index, current_index+2)
        current_index += 2
    else:
        x0_compressed.append(inp_dict[key].item())
        if isinstance(inp_dict[key], torch.Tensor):
            next_index = current_index + inp_dict[key].numel()
        else:
            next_index = current_index + 1
        slices[key] = slice(current_index, next_index)
        current_index = next_index
        
# x0_compressed = torch.tensor(x0_compressed).float().to(device).unsqueeze(0)

x0_compressed = torch.tensor([
    0.2,
   -0.5,  0.9,
   -1.0,  0.5,
    2.0,  1.0,
    2.0,  1.0,
    0.0, 
    0.8, -0.02,
    0.85, 0.2,
    0.5,
]).float().to(device).unsqueeze(0)


#%
def depack_line(x_compressed):
    decomposed = {}
    for key, sl in slices.items():
        if key in init_params:
            y_pred = line(toy.wvl.squeeze(), x_compressed[:,sl][:,0], x_compressed[:,sl][:,1], toy.wvl.min(), *line_norms[key])
            decomposed[key] = y_pred.squeeze(-1) if sl.stop-sl.start<2 else y_pred
        else:
            val = transformer.transforms[key].backward(x_compressed[:, sl])
            decomposed[key] = val.squeeze(-1) if sl.stop-sl.start<2 else val # expects the TipTorch's conventions about the tensors dimensions
    return decomposed

# decomposed = depack_line(x0_compressed)
#%%
line_norms = {
    'F' : (1e6, 1),
    'dx': (1e6, 1e0),
    'dy': (1e6, 1e-1),
    'bg': (1e6, 1e-6),
    'Jx': (1e6, 1e2),
    'Jy': (1e6, 1e2),
}

models_dict = {key: LineModel(toy.wvl, line_norms[key]) for key in line_norms.keys() }

inp_dict = {}
for key in transformer.transforms.keys(): 
    param_val = getattr(toy, key).abs() if key in ['F', 'Jx', 'Jy'] else getattr(toy, key) 
    if key in line_norms:
        inp_dict[key] = torch.tensor(models_dict[key].fit(param_val), device=device)
    else:
        inp_dict[key] = param_val
        
compressor = InputsCompressor(transformer.transforms, models_dict)

x0_compressed = compressor.stack(inp_dict).unsqueeze(0)
# destacked_2 = compressor.destack(x0_compressed)

#%%
def func2(x_):
    x_torch = compressor.destack(x_)
    return toy(x_torch)

def loss_MSE2(x_):
    diff = (func2(x_) - PSF_0) #* wvl_weights
    return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    # return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]

def loss_MAE2(x_):
    diff = (func2(x_) - PSF_0) #* wvl_weights
    # return diff.pow(2).sum() * 200 / PSF_0.shape[0] / PSF_0.shape[1]
    return diff.abs().sum() / PSF_0.shape[0] / PSF_0.shape[1]
    # return ( mask*diff.pow(2)*200 + diff.abs() ).flatten().sum() / PSF_0.shape[0] / PSF_0.shape[1]


#%%
result = minimize(loss_MSE2, x0_compressed, max_iter=100, tol=1e-3, method='bfgs', disp=2)
# result = minimize(loss_MAE2, x0_compressed, max_iter=100, tol=1e-3, method='bfgs', disp=2)
x0_compressed = result.x

#%%
PSF_1 = func2(x0_compressed)

if many_wvls:
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
decomposed = depack_line(x0_compressed)




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
    x_torch = transformer.destack(x_)
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


