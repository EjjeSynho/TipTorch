#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from torch import nn
from torch import optim
from tools.utils import ParameterReshaper, plot_radial_profiles, SR, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager, GetSPHEREonsky, GetSPHEREsynth
from tools.utils import rad2mas, SR, pdims, mask_circle, cropper
from project_globals import SPHERE_DATA_FOLDER, device


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
psf_df = psf_df[psf_df['Low quality'] == False]
psf_df = psf_df[psf_df['Medium quality'] == False]
psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]

good_ids = psf_df.index.values.tolist()

#%%
# 448, 452, 465, 552, 554, 556, 564, 576, 578, 580, 581, 578, 576, 992
# 1209 # high noise
# 1452 # high noise
# 1660 # LWE
# 456
# 465
# 1393 #50 DITs
# 1408
# 898

PSF_data, _, merged_config = SPHERE_preprocess(
    sample_ids    = [1660],
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

PSF_0   = PSF_data[0]['PSF (mean)'].unsqueeze(0)
PSF_var = PSF_data[0]['PSF (var)'].unsqueeze(0)
bg      = PSF_data[0]['bg (mean)']
norms   = PSF_data[0]['norm (mean)']
del PSF_data

#%% Initialize model
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

toy = TipTorch(merged_config, None, device)

_ = toy()

# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']
toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']

_ = toy({
    'F':   torch.tensor([1.0, 1.0]*toy.N_src, device=toy.device).flatten(),
    'Jx':  torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'Jy':  torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'Jxy': torch.tensor([1.0]*toy.N_src, device=toy.device).flatten(),
    'bg':  bg.to(device)
})

PSF_1 = toy()
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)
# mask_in  = toy.mask_rim_in.unsqueeze(1).float()
# mask_out = toy.mask_rim_out.unsqueeze(1).float()


#%%

# Create a sample signal: a sine wave
t = torch.linspace(0, 2*np.pi, steps=400)
signal = torch.sin(2.0 * np.pi * 5.0 * t)

# Perform the real FFT
spectrum = torch.fft.rfft(signal)

# Visualize the magnitude of the original frequency spectrum
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.plot(torch.abs(spectrum))
plt.title('Original Frequency Spectrum Magnitude')

# Modify the spectrum (e.g., zeroing out some frequencies)
spectrum[20:] = 0

# Perform the inverse real FFT
modified_signal = torch.fft.irfft(spectrum, n=signal.numel())

# Visualize the modified signal
plt.subplot(1, 2, 2)
plt.plot(t, modified_signal)
plt.title('Modified Signal in Time Domain')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import fft

with torch.no_grad():
    PSD = toy.ComputePSD()[0,0,...]
    PSD[..., toy.nOtf_AO//2, toy.nOtf_AO//2] = 0.0
    cov = fft.fftshift(fft.fft2(fft.fftshift(PSD, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
    
    PSD_2 = PSD[121//2:, :]
    # PSD_2 = PSD[:, 121//2:]
    # cov2 = fft.fftshift(fft.rfft2(fft.fftshift(PSD, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
    # cov2 = fft.fftshift(fft.rfft2(PSD), dim=(-2,-1)) # FFT axes are -2,-1 #TODO: real FFT?
    cov2 = fft.fftshift(fft.rfft2(fft.fftshift(PSD, dim=(-2,-1))), dim=-2) # FFT axes are -2,-1 #TODO: real FFT?
    
    SF = 2*(cov2.abs().amax(dim=(-2,-1), keepdim=True) - cov2.abs())

plt.imshow(SF.real.log().cpu().numpy())


#%%
N = 127
# def gen_grid(N):
factor = 0.5*(1-N%2)
ggrid = torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N)]*2, indexing = 'ij')

ggrid[0][:,]


#%% PSF fitting (no early-stopping)
from tools.utils import OptimizeLBFGS

class LWE_basis():
    def __init__(self) -> None:
        from tools.utils import BuildPetalBasis
        self.modal_basis, self.coefs = BuildPetalBasis(toy.pupil.cpu(), pytorch=True)
        self.modal_basis = self.modal_basis[1:,...].to(device)
        self.coefs = self.coefs[1:].to(device)
        self.coefs.requires_grad = True
        
    def forward(self, wvl):
        OPD = torch.sum(pdims(self.coefs, 2) * self.modal_basis, dim=0) * 1e-9 
        return toy.pupil  * toy.apodizer * torch.exp(1j*2*np.pi/wvl*OPD)
    
    def __call__(self, *args):
        return self.forward(*args)


basis = LWE_basis()

crop_LWE = cropper(PSF_0, 20)
crop_all = cropper(PSF_0, 100)

def loss_fn(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all])
    # return nn.HuberLoss(reduction='sum', delta=0.1)(A[crop_all], B[crop_all])

def loss_LWE(A, B):
    return nn.MSELoss(reduction='sum')(A[crop_LWE], B[crop_LWE])*10000

optimizer_lbfgs_all = OptimizeLBFGS(toy, loss_fn)
optimizer_lbfgs_LWE = OptimizeLBFGS(toy, loss_LWE)

optimizer_lbfgs_all.Optimize(PSF_0, [toy.bg], 3)
for i in range(15):
    optimizer_lbfgs_LWE.Optimize(PSF_0, [basis.coefs], 10, [None, None, lambda x: basis(x)])
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.F], 4)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.dx, toy.dy], 4)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.r0], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.dn], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.wind_dir, toy.wind_speed], 3)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.Jx, toy.Jy], 4)

with torch.no_grad():
    PSF_1 = toy(None, None, lambda x: basis(x))

#%%
with torch.no_grad():
    testo = toy.ComputeStaticOTF(basis)[0,0,...]

plt.imshow(testo.real.abs().log().cpu().numpy())



#%%
from torch.autograd.functional import hessian, jacobian

'''
if x is not None:
for name, value in x.items():
    if isinstance(getattr(self, name), nn.Parameter):
        setattr(self, name, nn.Parameter(value))
    else:
        setattr(self, name, value)
'''

def func(x):
    X_ = {
        'r0':  x[0],
        'F':   x[1],
        'dx':  x[2],
        'dy':  x[3],
        'bg':  x[4],
        'Jx':  x[5],
        'Jy':  x[6],
        'Jxy': x[7]
    }
    return toy(x=X_, PSD=None, pupil_function=lambda x: basis(x))

x_0 =

FIM = 1.0/(PSF_0 - PSF_1).var() * hessian(func, )

plt.imshow(FIM.cpu().numpy())



#%%
r0_stats  = torch.tensor(0.3  + 0.3j).to(device)
F_stats   = torch.tensor(1.0  + 0.2j).to(device)
bg_stats  = torch.tensor(0.0  + 1e-5j).to(device)
dxy_stats = torch.tensor(0.0  + 0.5j).to(device)
Jxy_stats = torch.tensor(10.0 + 10.0j).to(device)
LWE_stats = torch.tensor(0.0  + 100.0j).to(device)

MAP = lambda x, x0: (x - x0.real)**2 / x0.imag**2

param_loss_all = lambda:\
    MAP(toy.r0,  r0_stats) + \
    MAP(toy.F,   F_stats) + \
    MAP(toy.dx,  dxy_stats) + \
    MAP(toy.dy,  dxy_stats) #+ \
    # MAP(toy.bg,  bg_stats) + \
    # MAP(toy.Jxy, Jxy_stats)
    
param_loss_LWE = lambda: MAP(basis.coefs, LWE_stats).sum()

basis = LWE_basis()

crop_LWE = cropper(PSF_0, 20)
crop_all = cropper(PSF_0, 100)

def loss_fn(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all]) + param_loss_all().sum() * 0.5

def loss_LWE(A, B):
    return nn.MSELoss(reduction='sum')(A[crop_LWE], B[crop_LWE])*10000

optimizer_lbfgs_all = OptimizeLBFGS(toy, loss_fn)
optimizer_lbfgs_LWE = OptimizeLBFGS(toy, loss_LWE)

optimizer_lbfgs_all.Optimize(PSF_0, [toy.bg], 3)
for i in range(15):
    optimizer_lbfgs_LWE.Optimize(PSF_0, [basis.coefs], 10, [None, None, lambda x: basis(x)])
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.F], 4)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.dx, toy.dy], 4)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.r0], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.dn], 3)
    # optimizer_lbfgs.Optimize(PSF_0, [toy.wind_dir, toy.wind_speed], 3)
    optimizer_lbfgs_all.Optimize(PSF_0, [toy.Jx, toy.Jy], 4)


with torch.no_grad():
    PSF_1 = toy(None, None, lambda x: basis(x))

#%%
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))
draw_PSF_stack(PSF_0, PSF_1, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)

#%%
with torch.no_grad():
    dd = (PSF_0[crop_LWE].cpu().numpy() - PSF_1[crop_LWE].cpu().numpy())[0,0,...]

plt.imshow(dd)
plt.colorbar()
plt.show()

#%%
with torch.no_grad():
    windy = torch.sum(pdims(basis.coefs, 2) * basis.modal_basis, dim=0) * 1e-9 
    
plt.imshow(windy.cpu().numpy())

#%%
WFE = torch.mean(toy.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = toy.D/4 * 1e9*(toy.Jx+toy.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / toy.wvl.flatten()[0]

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')