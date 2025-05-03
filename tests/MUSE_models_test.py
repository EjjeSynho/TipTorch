#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.utils import plot_radial_profiles_new, plot_radial_profiles_relative, draw_PSF_stack, mask_circle, PupilVLT
from data_processing.MUSE_preproc_utils import GetConfig, LoadImages, LoadMUSEsampleByID, rotate_PSF
from project_globals import MUSE_DATA_FOLDER, device
from torchmin import minimize
from data_processing.normalizers import TransformSequence, Uniform, LineModel, QuadraticModel
from managers.input_manager import InputsTransformer
from project_globals import MUSE_DATA_FOLDER

#%%
from managers.config_manager import ConfigFromFile
from PSF_models.TipTorch import TipTorch_new

config_file = ConfigFromFile('../data/parameter_files/muse_ltao_ang.ini', device)


#%%
pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )

PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   False,
    'Moffat':          False
}

tiptorch_half = TipTorch_new(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
tiptorch_half.to_float()


W = H = tiptorch_half.N_pix
crop = 20

ROI = np.s_[0, W//2-crop:W//2+crop, H//2-crop:H//2+crop]

with torch.no_grad():
    PSF_1_half  = tiptorch_half()#x=inputs)

plt.imshow( PSF_1_half[:,-1,...][ROI].abs().log10().cpu().numpy())
plt.colorbar()
plt.show()



#%%
with open(MUSE_DATA_FOLDER+'/muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

derotate_PSF    = True
Moffat_absorber = False
include_sausage = True

sample = LoadMUSEsampleByID(405)
PSF_0, var_mask, norms, bgs = LoadImages(sample)
config_file, PSF_0 = GetConfig(sample, PSF_0)
N_wvl = PSF_0.shape[1]

pupil_angle = sample['All data']['Pupil angle'].item()

if derotate_PSF:
    PSF_0 = rotate_PSF(PSF_0, -pupil_angle)
    config_file['telescope']['PupilAngle'] = 0

config_file['NumberSources'] = 1
config_file['sensor_science']['FieldOfView'] = config_file['sensor_science']['FieldOfView'].int().item() - 1


GL_fraction = 0.9
# config_file['atmosphere']['Cn2Weights']   = torch.tensor([[GL_fraction, 1-GL_fraction]], device=device)
# config_file['atmosphere']['Cn2Heights']   = torch.tensor([[0.0, 10000.0]], device=device)
# config_file['sources_science']['Zenith']  = torch.tensor([7.5], device=device)
# config_file['sources_science']['Azimuth'] = torch.tensor([45], device=device)

#%% Initialize the model
from PSF_models.TipToy_MUSE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new


toy = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
toy.PSD_include['fitting']         = True
toy.PSD_include['WFS noise']       = True
toy.PSD_include['spatio-temporal'] = True
toy.PSD_include['aliasing']        = True
toy.PSD_include['chromatism']      = True
toy.PSD_include['Moffat']          = Moffat_absorber
toy.to_float()


with torch.no_grad():
    pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )

    PSD_include = {
        'fitting':         True,
        'WFS noise':       True,
        'spatio-temporal': True,
        'aliasing':        True,
        'chromatism':      True,
        'diff. refract':   False,
        'Moffat':          Moffat_absorber
    }
    
    tiptorch_half = TipTorch_new(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
    tiptorch_half.to_float()
    
    '''
    inputs = {
        'r0':  0.1,
        'F':   [1.0,]*N_wvl,
        # 'dx':  [0.0,]*N_wvl,
        # 'dy':  [0.0,]*N_wvl,
        'dx':  [0.0],
        'dy':  [0.0],
        'bg':  [1e-6,]*N_wvl,
        'dn':  1.5,
        'Jx':  [0,]*N_wvl,
        'Jy':  [0,]*N_wvl,
        # 'Jx':  [0],
        # 'Jy':  [0],
        'Jxy': [0]
    }
    for value, key in zip(inputs.values(), inputs.keys()):
        inputs[key] = torch.tensor([value], device=tiptorch_half.device)
    ''' 
    
    inputs = {
        'r0':    torch.tensor([0.1206], device='cuda:0'),
        'F':     torch.tensor([[0.9876, 1.0038, 1.0193, 0.9823, 0.9953, 1.0135, 1.0284, 1.0016, 0.9713, 0.9705, 0.9454,  0.9150,  0.9032]], device='cuda:0'),
        'dx':    torch.tensor([[0.7711, 0.7151, 0.6626, 0.5700, 0.4754, 0.4131, 0.3473, 0.2099, 0.1137, 0.1353, 0.0684, -0.0509, -0.0493]], device='cuda:0'),
        'dy':    torch.tensor([[0.6729, 0.6363, 0.6054, 0.6064, 0.6106, 0.5911, 0.5819, 0.5877, 0.5911, 0.5854, 0.6026, 0.6167, 0.6000]], device='cuda:0'),
        'bg':    torch.tensor([[-8.8417e-09, -3.5006e-09, -1.0428e-07, -8.2854e-08, -4.9171e-08, -8.3156e-08, -1.7448e-07, -8.5068e-08,  1.4727e-08,  3.1474e-08, 1.1614e-07,  1.6747e-07,  1.6476e-07]], device='cuda:0'),
        'dn':    torch.tensor([2.6343], device='cuda:0'),
        'Jx':    torch.tensor([[ 3.9943,  6.8474,  7.4349,  8.6813, 17.8425, 19.2336, 19.7889, 19.2334, 20.2027, 23.1771, 22.8105, 17.3229, 21.7012]], device='cuda:0'),
        'Jy':    torch.tensor([[ 1.1891,  0.9108,  0.3702,  8.6809,  9.2326,  5.4804, -0.3380, -0.2175, -0.1002, -0.3949, -0.4773, -0.9462,  0.1430]], device='cuda:0'),
        'Jxy':   torch.tensor([0.], device='cuda:0'),
        's_pow': torch.tensor([0.3553], device='cuda:0'),
        'amp':   torch.tensor([1.9407], device='cuda:0'),
        'b':     torch.tensor([0.0050], device='cuda:0'),
        'alpha': torch.tensor([0.1706], device='cuda:0')
    }

#%%
W = H = tiptorch_half.N_pix
crop = 20

ROI = np.s_[0, W//2-crop:W//2+crop, H//2-crop:H//2+crop]

# PSF_1_torch = tiptorch(x=inputs)
PSF_1_toy   = toy(x=inputs)
PSF_1_half  = tiptorch_half(x=inputs)

# differr      = PSF_1_toy  - PSF_1_torch
# differr_half = PSF_1_half - PSF_1_torch

dPSF_1 = PSF_1_half - PSF_1_toy

plt.imshow(  PSF_1_toy[:,-1,...][ROI].abs().log10().cpu().numpy())
plt.colorbar()
plt.show()

plt.imshow( PSF_1_half[:,-1,...][ROI].abs().log10().cpu().numpy())
plt.colorbar()
plt.show()

plt.imshow( dPSF_1[:,-1,...][ROI].abs().log10().cpu().numpy())
plt.colorbar()
plt.show()
# plt.imsho(PSF_1_torch[:,1,...][ROI].abs().log10().cpu().numpy())
# plt.show()

#%%

plt.imshow(tiptorch_half.PSD[0,-1,...].abs().log10().cpu().numpy())



#%%
from tqdm import tqdm

diffs_init = []
diffs_runs = []

N = 25

# for _ in tqdm(range(N)):
#     tiptorch.StartTimer()
#     PSF_1_torch = tiptorch(x=inputs)
#     b = tiptorch.EndTimer()
#     diffs_full.append(b)
for _ in tqdm(range(N)):
    tiptorch_half.StartTimer()
    tiptorch_half.Update(reinit_grids=True, reinit_pupils=True)    
    a = tiptorch_half.EndTimer()
    diffs_init.append(a)

for _ in tqdm(range(N)):
    tiptorch_half.StartTimer()
    PSF_1_torch = tiptorch_half(x=inputs)
    b = tiptorch_half.EndTimer()
    diffs_runs.append(b)

# diffs_full = np.array(diffs_full)
diffs_runs = np.array(diffs_runs[1:])
diffs_init = np.array(diffs_init[1:])

print(f'Initialization time: {diffs_init.mean()}')
print(f'Run time: {diffs_runs.mean()}')


#%%
plt.hist(diffs_full[1:], bins=20, alpha=0.5, label='full')
plt.hist(diffs_runs[1:], bins=20, alpha=0.5, label='half')
plt.xlim([0, diffs_full[1:].max()])

#%%
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler
import torch.nn as nn

writer = SummaryWriter(log_dir='./runs/tiptorch_half_inference_profiling')

# model = nn.Linear(10, 5)
# model.eval()  # Set the model to evaluation mode
# input_tensor = torch.randn(1, 10)

for _ in range(10):
    # tiptorch_half.Update(reinit_grids=True, reinit_pupils=True)
    tiptorch_half(x=inputs)


# Define the profiler
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA],  # Use CUDA if available
    on_trace_ready=profiler.tensorboard_trace_handler('./runs/tiptorch_half_inference_profiling'),
    record_shapes=True,
    with_stack=True  # Optional: Add stack tracing
    
) as prof:
    with torch.no_grad():
        # tiptorch_half.Update(reinit_grids=True, reinit_pupils=True)
        tiptorch_half(x=inputs)
        

# Print profiler summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# prof.export_chrome_trace("trace.json")


#%%

with torch.no_grad():
    PSF_1 = tiptorch_half(x=inputs)

    plt.imshow(PSF_1[0,-1,...].abs().log10().cpu().numpy())


#%%
# tiptorch.OptimalDMProjector()
tiptorch.DMProjector()

#%%
a = tiptorch.DifferentialRefractionPSD()
b = tiptorch.ChromatismPSD()

#%%
from tqdm import tqdm
from PSF_models.TipTorch import TipTorch_new
from matplotlib.colors import LogNorm

sizes = [101 + 25*i for i in range(15)]

OTFs, PSFs = [], []

N_wvl = 7

inputs = {
    'r0':  torch.tensor([0.1], device=toy.device),
    'F':   torch.tensor([[1.0,]*N_wvl], device=toy.device),
    'dx':  torch.tensor([[2.0]], device=toy.device),
    'dy':  torch.tensor([[4.0]], device=toy.device),
    'bg':  torch.tensor([[0e-6,]*N_wvl], device=toy.device),
    'dn':  torch.tensor([1.5], device=toy.device),
    'Jx':  torch.tensor([[10,]*N_wvl], device=toy.device),
    'Jy':  torch.tensor([[10,]*N_wvl], device=toy.device),
    'Jx':  torch.tensor([[33]], device=toy.device),
    'Jy':  torch.tensor([[10]], device=toy.device),
    'Jxy': torch.tensor([[45]], device=toy.device)
}


config_file['sources_science']['Wavelength'] = torch.tensor([[ \
    4.9444e-07, 5.6044e-07, 6.5544e-07, 7.2144e-07, 7.8744e-07, 8.5344e-07, 9.1894e-07
]], device='cuda:0')

#%%
for size_ in tqdm(sizes):
    config_file['sensor_science']['FieldOfView'] = size_
    
    tiptorch = TipTorch_new(config_file, None, device=device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
    PSF_1_torch = tiptorch(x=inputs)
    
    PSFs.append(PSF_1_torch.detach().cpu().numpy())
    OTFs.append(tiptorch.OTF.detach().cpu().numpy())
    
    del tiptorch
    
    torch.cuda.empty_cache()
    
    # sums.append(PSF_1_torch.sum().item())
    # OTF_energy.append(tiptorch.OTF.abs().sum().item())

# tiptorch = TipTorch_new(config_file, None, device=device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
# PSF_1_torch = tiptorch(x=inputs)
# aa = PSF_1_torch.squeeze().sum(dim=(-2,-1))
# wvls = tiptorch.wvl.squeeze().cpu().numpy()

# plt.plot(wvls, aa.cpu().numpy())
#%%
# plt.imshow(np.abs(OTFs[-1][0,0].imag))#, norm=LogNorm())
# plt.colorbar()

for i in range(len(sizes)):
    print(f'{sizes[i]}: {OTFs[i].sum():.2e}, {PSFs[i].sum():.4e}')

#%%

testo = np.array([PSFs[i].sum() for i in range(len(sizes))]) / 7 - 1
testo *= 100


#%%
sums, OTF_energy, OTF_maxes, OTF_mins = [], [], [], []

for i in range(len(sizes)):
    sums.append(PSFs[i].sum())
    OTF_energy.append(OTFs[i].sum())
    OTF_maxes.append(OTFs[i].max())
    OTF_mins.append(OTFs[i].min())

OTF_energy = np.array(OTF_energy)
sums = np.array(sums)


#%%
plt.imshow(np.abs(np.abs(OTFs[0])[0,6,...]))
plt.colorbar()

#%%
# plt.plot(sizes, np.array(sums)/13)
aa = OTF_energy.real / np.array(sizes)**2
bb = sums / np.array(sizes)**2

# plt.plot(sizes, aa)
# plt.plot(sizes, bb)

# plt.ylim(0, 1.3)

plt.plot(sizes, sums)

# for i in range(len(sizes)):
#     print(f'{sizes[i]}: {sums[i]:.2e}, {OTF_energy[i]:.2e}')
    
#%%
x = np.linspace(0, 3, 100)
y = np.exp(-x)
    
plt.plot(x, np.log(y))
    
# tiptorch.OTF.abs().sum()
# plt.imshow(tiptorch.OTF.abs().sum(dim=(0,1)).log10().cpu().numpy())
# print(tiptorch.OTF.abs().sum())

#%%
from PSF_models.TipTorch import TipTorch_new

config_file['sensor_science']['FieldOfView'] = 351

tiptorch = TipTorch_new(config_file, None, device=device, TipTop=True, PSFAO=Moffat_absorber, oversampling=1)
PSF_1_torch = tiptorch(x=inputs)

#%%

# plt.imshow(tiptorch.V.squeeze().cpu().numpy())
tiptorch.OTF.shape

#%%
from tools.utils import PupilVLT

test_pupil = PupilVLT(
    samples = 321,
    vangle = [0,0],
    rotation_angle = 0
)

plt.imshow(test_pupil)


#%%
from torch.nn.functional import interpolate
from torch import fft, nn


phase = tiptorch.pupil

sampling = tiptorch.sampling.min().item()

pupil_size   = phase.shape[-1]
pupil_padder = torch.nn.ZeroPad2d( int(round(pupil_size*sampling/2-pupil_size/2)) )
phase_padded = pupil_padder(phase)

x = phase_padded

OTF = fft.fftshift( fft.ifft2(fft.fft2(x, dim=(-2,-1)).abs()**2, dim=(-2,-1)), dim=(-2,-1) ) / x.shape[-2] / x.shape[-1]

OTF = OTF.view(1, 1, *OTF.shape) if OTF.ndim == 2 else OTF

interp = lambda x: \
    interpolate(x, size=(tiptorch.nOtf, tiptorch.nOtf), mode='bilinear', align_corners=False) * sampling**2
    
# PyTorch doesn't support interpolation of complex tensors yet
OTF_ = interp(OTF.real) + interp(OTF.imag)*1j
# return OTF_ / OTF_.abs().amax(dim=(-2,-1), keepdim=True)

#%%
plt.imshow(OTF_.squeeze().abs().log10().cpu().numpy())

#%%

plt.imshow(tiptorch.pupil.abs().cpu().numpy())

#%%
# aa = tiptorch.OTF.abs()[..., tiptorch.nOtf//2, tiptorch.nOtf//2].cpu().numpy()
bb = tiptorch.OTF.abs().amax(dim=(-2,-1)).cpu().numpy()

#%%
plt.imshow(PSF_1_torch.sum(dim=(0,1)).log10().cpu().numpy())

#%% PSF fitting (no early-stopping)
norm_F     = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg    = TransformSequence(transforms=[ Uniform(a=-5e-6, b=5e-6)])
norm_r0    = TransformSequence(transforms=[ Uniform(a=0,     b=1)  ])
norm_dxy   = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J     = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_Jxy   = TransformSequence(transforms=[ Uniform(a=-180,  b=180) ])
norm_dn    = TransformSequence(transforms=[ Uniform(a=0,     b=5)  ])

norm_sausage_pow = TransformSequence(transforms=[ Uniform(a=0, b=1)  ])
# norm_sausage_ang = TransformSequence(transforms=[ Uniform(a=-30, b=30)  ])

norm_amp   = TransformSequence(transforms=[ Uniform(a=0,     b=10)   ])
norm_b     = TransformSequence(transforms=[ Uniform(a=0,     b=0.1)   ])
norm_alpha = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_beta  = TransformSequence(transforms=[ Uniform(a=0,     b=10)   ])
norm_ratio = TransformSequence(transforms=[ Uniform(a=0,     b=2)   ])
norm_theta = TransformSequence(transforms=[ Uniform(a=-np.pi/2, b=np.pi/2)])


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
}

transformer = InputsTransformer(transformer_dict)

_ = transformer.stack({ attr: getattr(toy, attr) for attr in transformer_dict }) # to create index mapping

#%%
x0 = [\
    norm_r0.forward(toy.r0).item(), # r0
    *([1.0,]*N_wvl), # F
    *([0.0,]*N_wvl), # dx
    *([0.0,]*N_wvl), # dy
    # 0.0,
    # 0.0,
    *([0.0,]*N_wvl), # bg
    -0.9, # dn
    # -0.9,
    # -0.9,
    *([-0.9,]*N_wvl), # Jx
    *([-0.9,]*N_wvl), # Jy
    0.0, # Jxy
]

if Moffat_absorber:
    x0 += [\
        # PSFAO realm
        -1,
        -1,
         0.3,
        # *([ 1.0,]*N_wvl),
        # *([ 0.0,]*N_wvl),
        # *([ 0.3,]*N_wvl)
    ]

if include_sausage:
    x0 += [0.9]

# x0 = transformer.stack(inputs)
x0 = torch.tensor(x0).float().to(device).unsqueeze(0)

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

if include_sausage:
    phase_func = lambda: sausage_absorber(toy.s_pow.flatten())
else:
    phase_func = None

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

    draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )
    
    PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
    plt.show()

else:
    draw_PSF_stack( PSF_0.cpu().numpy()[:,0,...], PSF_1.cpu().numpy()[:,0,...], average=True, crop=120 )
    
    plot_radial_profiles_new( PSF_0[0,0,...].cpu().numpy(),  PSF_1[0,0,...].cpu().numpy(),  'Data', 'TipTorch', centers=center, cutoff=60, title='Left PSF')
    plt.show()
    
#%%

if len(toy.wvl[0]) > 1:
    wvl_select = np.s_[0, 6, 12]

    draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )
    
    PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

    fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
    for i, lmbd in enumerate(wvl_select):
        plot_radial_profiles_relative( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
    plt.show()

else:
    draw_PSF_stack( PSF_0.cpu().numpy()[:,0,...], PSF_1.cpu().numpy()[:,0,...], average=True, crop=120 )
    
    plot_radial_profiles_relative( PSF_0[0,0,...].cpu().numpy(),  PSF_1[0,0,...].cpu().numpy(),  'Data', 'TipTorch', centers=center, cutoff=60, title='Left PSF')
    plt.show()

    

   
#%% ======================================================================================================
   
#% Initialize the model
model = TipTorch(config_file, 'sum', device, TipTop=True, PSFAO=True, oversampling=1)
sausage_absorber = SausageFeature(model)
sausage_absorber.OPD_map = sausage_absorber.OPD_map.flip(dims=(-1,-2))

model.PSD_include['fitting'] = True
model.PSD_include['WFS noise'] = True
model.PSD_include['spatio-temporal'] = True
model.PSD_include['aliasing'] = False
model.PSD_include['chromatism'] = True
model.PSD_include['Moffat'] = True

model.to_float()

F_df     =  toy.F.cpu().numpy()
dx_df    = toy.dx.cpu().numpy()
dy_df    = toy.dy.cpu().numpy()
bg_df    = toy.bg.cpu().numpy()
Jx_df    = toy.Jx.cpu().numpy()
Jy_df    = toy.Jy.cpu().numpy()
r0_df    = toy.r0.cpu().numpy()
dn_df    = toy.dn.cpu().numpy()
Jxy_df   = toy.Jxy.cpu().numpy()
amp_df   = toy.amp.cpu().numpy()
b_df     = toy.b.cpu().numpy()
alpha_df = toy.alpha.cpu().numpy()
beta_df  = toy.beta.cpu().numpy()
ratio_df = toy.ratio.cpu().numpy()
theta_df = toy.theta.cpu().numpy()

s_pow_df = toy.s_pow.cpu().numpy()


inputs_tiptorch = {
    'F':     torch.tensor(F_df    , device=model.device),
    'dx':    torch.tensor(dx_df   , device=model.device),
    'dy':    torch.tensor(dy_df   , device=model.device),
    'bg':    torch.tensor(bg_df   , device=model.device),
    'Jx':    torch.tensor(Jx_df   , device=model.device),
    'Jy':    torch.tensor(Jy_df   , device=model.device),
    'r0':    torch.tensor(r0_df   , device=model.device),
    'dn':    torch.tensor(dn_df   , device=model.device),
    'Jxy':   torch.tensor(Jxy_df  , device=model.device),
    'amp':   torch.tensor(amp_df  , device=model.device),
    'b':     torch.tensor(b_df    , device=model.device),
    'alpha': torch.tensor(alpha_df, device=model.device),
    'beta':  torch.tensor(beta_df , device=model.device),
    'ratio': torch.tensor(ratio_df, device=model.device),
    'theta': torch.tensor(theta_df, device=model.device)
}
setattr(model, 's_pow', torch.tensor(s_pow_df, device=model.device))

PSF_1_ = model(inputs_tiptorch, None, lambda: sausage_absorber(toy.s_pow.flatten()))


#%%
draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1_.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )
draw_PSF_stack( PSF_0.cpu().numpy()[0, wvl_select, ...], PSF_1.cpu().numpy()[0, wvl_select, ...], average=True, crop=120 )

PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1_, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
plt.show()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles_new( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=30,  ax=ax[i] )
plt.show()

#%% ======================================================================================================
from tools.utils import calc_profile

x_torch = transformer.unstack(x0)
_ = toy(x_torch, None, phase_generator=phase_func)

dk = 2*toy.kc / toy.nOtf_AO

PSD_norm = lambda wvl: (dk*wvl*1e9/2/np.pi)**2

PSDs = {entry: (toy.PSDs[entry].clone().squeeze() * PSD_norm(500e-9)) for entry in toy.PSD_entries if toy.PSDs[entry].ndim > 1}
PSDs['chromatism'] = PSDs['chromatism'].mean(dim=0)


plt.figure(figsize=(8, 6))

PSD_map  = toy.PSD[0,...].mean(dim=0).real.cpu().numpy()
k_map    = toy.k[0,...].cpu().numpy()
k_AO_map = toy.k_AO[0,...].cpu().numpy()

center    = [k_map.shape[0]//2, k_map.shape[1]//2]
center_AO = [k_AO_map.shape[0]//2, k_AO_map.shape[1]//2]

PSD_prof = calc_profile(PSD_map,  center)
freqs    = calc_profile(k_map,    center)
freqs_AO = calc_profile(k_AO_map, center_AO)

freq_cutoff = toy.kc.flatten().item()

profiles = {}
for entry in PSDs.keys():
    buf_map = PSDs[entry].abs().cpu().numpy()
    if entry == 'fitting':
        buf_prof = calc_profile(buf_map, center)
    else:
        buf_prof = calc_profile(buf_map, center_AO)
    profiles[entry] = buf_prof


plt.plot(freqs, PSD_prof, label='Total', linewidth=2, linestyle='--', color='black')

for entry, value in profiles.items():
    if entry == 'fitting':
        plt.plot(freqs, value, label=entry)
    else:
        plt.plot(freqs_AO, value, label=entry)

plt.legend()
plt.yscale('symlog', linthresh=5e-5)
plt.xscale('log')
# plt.vlines(freq_cutoff, PSD_prof.min(), PSD_prof.max(), color='red', linestyle='--')
plt.grid()
plt.xlim(freqs.min(), freqs.max())

plt.savefig(f"C:/Users/akuznets/Desktop/thesis_results/MUSE/PSDs/profiles.pdf", dpi=300)


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
from matplotlib.colors import LogNorm
import matplotlib as mpl

vmins = {
    'fitting': 1e-4,
    'WFS noise': 0.1,
    'spatio-temporal': 0.02,
    'chromatism': 1e-5,
    'Moffat': 1e-3
}

cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
cmap.set_bad(color=(0,0,0,0))
    
for entry in PSDs.keys():
    fig = plt.figure(figsize=(8,)*2)
    A = PSDs[entry].cpu().numpy().real
    A -= np.nanmin(A)
    A = np.abs(A)
    
    if entry != 'fitting':
        A += 1e-7
        A = A * toy.mask_corrected_AO.squeeze().cpu().numpy() * toy.piston_filter.squeeze().cpu().numpy()
    
    norm = LogNorm(vmin=vmins[entry], vmax=np.nanmax(A)*2)
    plt.imshow(A, norm=norm, cmap=cmap)
    plt.title(entry)
    plt.axis('off')
    plt.savefig(f"C:/Users/akuznets/Desktop/thesis_results/MUSE/PSDs/{entry}.pdf", dpi=300)
    plt.show()
    
#%%
x = x0.clone().detach().requires_grad_(True)

x_torch = transformer.unstack(x)

Q = ( toy(x_torch)-PSF_0 ).abs().sum()
get_dot = register_hooks(Q)
Q.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable


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

#%% ==================================================================================
import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas, cropper, EarlyStopping, mask_circle
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds, process_mask
from managers.config_manager import GetSPHEREonsky
from project_globals import SPHERE_DATA_FOLDER, device
from torchmin import minimize
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm


#%% 
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

sample_id = 768
# sample_id = 2649

PSF_data, _, merged_config = SPHERE_preprocess(
    sample_ids    = [sample_id],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

PSF_0    = PSF_data[0]['PSF (mean)'].unsqueeze(0)
PSF_var  = PSF_data[0]['PSF (var)'].unsqueeze(0)
PSF_mask = PSF_data[0]['mask (mean)'].unsqueeze(0)
norms    = PSF_data[0]['norm (mean)']
del PSF_data


merged_config['sensor_science']['FieldOfView'] = PSF_0.shape[-1]
# --New below
merged_config['NumberSources'] = merged_config['NumberSources'].int().item()
merged_config['DM']['DmHeights'] = torch.tensor(merged_config['DM']['DmHeights'], device=device)
merged_config['sources_HO']['Wavelength'] = merged_config['sources_HO']['Wavelength']
merged_config['sources_HO']['Height'] = torch.inf
# --New end


# if psf_df.loc[sample_id]['Nph WFS'] < 10:
# PSF_mask   = PSF_mask * 0 + 1
PSF_mask = process_mask(PSF_mask) * 0 + 1
# LWE_flag   = psf_df.loc[sample_id]['LWE']
LWE_flag = True
wings_flag = True#psf_df.loc[sample_id]['Wings']
# wings_flag = False

if psf_df.loc[sample_id]['Central hole'] == True:
    circ_mask = 1-mask_circle(PSF_0.shape[-1], 3, centered=True)
    PSF_mask *= torch.tensor(circ_mask[None, None, ...]).to(device)


# plt.imshow(circ_mask * np.squeeze(PSF_0[0,0,...].cpu().numpy()), norm=LogNorm())

#%% Initialize model
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new
# from tools.utils import LWE_basis

over = 1

tiptorch_old = TipTorch(merged_config, None, device, oversampling=over)

tiptorch_old.PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   False,
    'Moffat':          False
}

_ = tiptorch_old()

DL_old = tiptorch_old.DLPSF() * over**2


PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   False,
    'Moffat':          False
}
tiptorch_new = TipTorch_new(merged_config, 'SCAO', None, PSD_include, 'sum', device, oversampling=over)
tiptorch_new.to_float()

PSF_1_new = tiptorch_new()

DL_new = tiptorch_new.DLPSF()

#%
'''
x = {
    'r0':  torch.tensor([0.0747], device=device),
    'F':   torch.tensor([[1.1569, 1.1547]], device=device) * 0 + 1,
    # 'dx':  torch.tensor([[0.8032, 0.7462]], device=device),
    'dx':  torch.tensor([[40., 39.]], device=device) * 0,
    'dy':  torch.tensor([[-0.0599, -0.1582]], device=device)*0,
    'bg':  torch.tensor([[-6.9435e-06, -6.3148e-06]], device=device) * 0,
    'dn':  torch.tensor([0.0070], device=device) * 0,
    'Jx':  torch.tensor([12.3*3], device=device) ,
    'Jy':  torch.tensor([12.3*3], device=device) ,
    'Jxy': torch.tensor([471.3628], device=device) * 0,
    'wind_dir': torch.tensor([397.8669], device=device) * 0,
    'basis_coefs': torch.tensor([[
        -30.3420,  33.2847,  34.2236, -31.0020,  -6.7741,   4.3112,
        11.0884, 2.2478,  -0.5152,  -3.5624,  -2.766,   9.7230
    ]], device=device)
}
'''

x = {
    'r0': torch.tensor([0.0437], device='cuda:0'),
    'F':  torch.tensor([[1.2656, 1.2267]], device='cuda:0')* 0 + 1,
    'dx': torch.tensor([[0.5865, 0.8096]], device='cuda:0') * 0,
    'dy': torch.tensor([[-0.5381, -0.9207]], device='cuda:0') * 0,
    'bg': torch.tensor([[-2.4762e-05, -1.0387e-05]], device='cuda:0')* 0,
    'dn': torch.tensor([0.0245], device='cuda:0')* 0,
    'Jx': torch.tensor([36.1686], device='cuda:0')*0,
    'Jy': torch.tensor([37.7516], device='cuda:0')*0,
    'Jxy': torch.tensor([-0.2651], device='cuda:0')* 0,
    'wind_dir': torch.tensor([-33.6068], device='cuda:0')* 0,
    'basis_coefs': torch.tensor([[
        22.5876, -15.5937, -37.1299,  25.9760, -41.4860,   0.7404,
        -1.0915,  -4.5618,   1.3280,  16.4002, -23.5714,  -1.0974
    ]], device='cuda:0') * 0
}

#%%
PSF_1_old = tiptorch_old(x)
PSF_1_new = tiptorch_new(x)

# PSF_1_old = DL_old
# PSF_1_new = DL_new


#%
# with torch.no_grad():
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
plot_radial_profiles_new( (PSF_1_old*PSF_mask)[:,0,...].cpu().numpy(), (PSF_1_new*PSF_mask)[:,0,...].cpu().numpy(), 'Old', 'New', title='Left PSF',  ax=ax[0] )
plot_radial_profiles_new( (PSF_1_old*PSF_mask)[:,1,...].cpu().numpy(), (PSF_1_new*PSF_mask)[:,1,...].cpu().numpy(), 'Old', 'New', title='Right PSF', ax=ax[1] )
plt.show()

draw_PSF_stack(PSF_1_old*PSF_mask, PSF_1_new*PSF_mask, min_val=1e-6, average=True, crop=80)#, scale=None)
plt.show()
#%%
# PSD_old = tiptorch_old.PSDs['WFS noise']#.squeeze()
# PSD_new = tiptorch_new.half_PSD_to_full(tiptorch_new.PSDs['WFS noise'])

# PSD_old = tiptorch_old.Rx.unsqueeze(0).imag
# PSD_new = tiptorch_new.half_PSD_to_full(tiptorch_new.Rx.unsqueeze(0)).real

# PSD_old = tiptorch_old.piston_filter.unsqueeze(0)
# PSD_new = tiptorch_new.half_PSD_to_full(tiptorch_new.piston_filter.unsqueeze(0))


# PSD_old = tiptorch_old.VonKarmanSpectrum(tiptorch_old.r0.abs(), tiptorch_old.L0.abs(), tiptorch_old.k2_AO) #* self.piston_filter
# PSD_new = tiptorch_new.VonKarmanSpectrum(tiptorch_new.r0.abs(), tiptorch_new.L0.abs(), tiptorch_new.k2_AO) #* self.piston_filter

PSD_old = tiptorch_old.kx
PSD_new = tiptorch_new.kx
PSD_new = tiptorch_new.half_PSD_to_full(PSD_new)

plot_radial_profiles_new( PSD_old.cpu().numpy(), PSD_new.cpu().numpy(), 'Old', 'New', title='Left PSF')
plt.show()

#%%


PSD_old


#%%

print( tiptorch_old.noise_gain )
print( tiptorch_new.noise_gain )


print( tiptorch_old.NoiseVariance(tiptorch_old.r0) )
print( tiptorch_new.NoiseVariance() )

#%%

plt.imshow(tiptorch_new.piston_filter.squeeze().cpu())



#%%
# Gaussian blur
from scipy.ndimage import gaussian_filter

x['Jx'] *= 0
x['Jy'] *= 0

# PSF_1_old_blur = tiptorch_old(x)
PSF_1_new_blur = tiptorch_new(x)

PSF_1_new_blur[0,0,...] = torch.tensor(gaussian_filter(PSF_1_new_blur[0,0,...].cpu().numpy(), sigma=2), device=device)
PSF_1_new_blur[0,1,...] = torch.tensor(gaussian_filter(PSF_1_new_blur[0,1,...].cpu().numpy(), sigma=2), device=device)

draw_PSF_stack(PSF_1_new, PSF_1_new_blur, min_val=1e-6, average=True, crop=80)#, scale=None)
plt.show()