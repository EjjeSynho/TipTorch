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
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']
toy.optimizables = []

_ = toy({ 'bg': bg.unsqueeze(0).to(device) })

PSF_1 = toy()
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)
# mask_in  = toy.mask_rim_in.unsqueeze(1).float()
# mask_out = toy.mask_rim_out.unsqueeze(1).float()
#%%
r0  = toy.r0.clone()
F   = toy.F.clone()
Jx  = toy.Jx.clone()
Jy  = toy.Jy.clone()
Jxy = toy.Jxy.clone()
bg  = toy.bg.clone()
dx  = toy.dx.clone()
dy  = toy.dy.clone()


r0.requires_grad  = True
F.requires_grad   = True
Jx.requires_grad  = True
Jy.requires_grad  = True
Jxy.requires_grad = True
bg.requires_grad  = True
dx.requires_grad  = True
dy.requires_grad  = True


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
from tools.utils import pdims

class LWE_basis():
    def __init__(self, model, optimizable=True) -> None:
        from tools.utils import BuildPetalBasis
        self.model = model
        self.modal_basis, self.coefs = BuildPetalBasis(self.model.pupil.cpu(), pytorch=True)
        self.modal_basis = self.modal_basis[1:,...].float().to(device)
        self.coefs = self.coefs[1:].to(device).repeat(model.N_src, 1)
        if optimizable:
            self.coefs = nn.Parameter(self.coefs)

    def forward(self, x=None):
        if x is not None:
            self.coefs = x
            
        OPD = torch.einsum('mn,nwh->mwh', self.coefs, self.modal_basis) * 1e-9  
        return pdims(self.model.pupil * self.model.apodizer, -2) * torch.exp(1j*2*np.pi / pdims(self.model.wvl,2)*OPD.unsqueeze(1))
    
    def __call__(self, *args):
        return self.forward(*args)

basis = LWE_basis(toy)

#%%
crop_LWE = cropper(PSF_0, 20)
crop_all = cropper(PSF_0, 100)

def loss_fn(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all])
    # return nn.HuberLoss(reduction='sum', delta=0.1)(A[crop_all], B[crop_all])

def loss_LWE(A, B):
    return nn.MSELoss(reduction='sum')(A[crop_LWE], B[crop_LWE])*10000

optimizer_lbfgs_all = OptimizeLBFGS(toy, loss_fn)
optimizer_lbfgs_LWE = OptimizeLBFGS(toy, loss_LWE)

x = {
    'r0':  r0,
    'F':   F,
    'dx':  dx,
    'dy':  dy,
    'bg':  bg,
    'Jx':  Jx,
    'Jy':  Jy,
    'Jxy': Jxy
}

optimizer_lbfgs_LWE.Optimize(PSF_0, [basis.coefs], 10, [None, toy.PSD, lambda: basis()])
optimizer_lbfgs_all.Optimize(PSF_0, [bg], 3, [x, None, None])
for i in range(15):
    optimizer_lbfgs_LWE.Optimize(PSF_0, [basis.coefs], 10, [None, toy.PSD, lambda: basis()])
    optimizer_lbfgs_all.Optimize(PSF_0, [F], 4, [x, None, None])
    optimizer_lbfgs_all.Optimize(PSF_0, [dx, dy], 4, [x, None, None])
    optimizer_lbfgs_all.Optimize(PSF_0, [r0], 3, [x, None, None])
    # optimizer_lbfgs.Optimize(PSF_0, [dn], 3, [x, None, None])
    # optimizer_lbfgs.Optimize(PSF_0, [wind_dir, wind_speed], 3, [x, None, None])
    optimizer_lbfgs_all.Optimize(PSF_0, [Jx, Jy], 4, [x, None, None])

with torch.no_grad():
    PSF_1 = toy()


#%%
from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer

norm_F   = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg  = TransformSequence(transforms=[ Uniform(a=-1e-5, b=1e-5)])
norm_r0  = TransformSequence(transforms=[ Uniform(a=0,     b=0.5) ])
norm_dxy = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J   = TransformSequence(transforms=[ Uniform(a=0,     b=30)  ])
norm_Jxy = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_LWE = TransformSequence(transforms=[ Uniform(a=-200,  b=200) ])

transformer = InputsTransformer({
    'F':   norm_F,
    'bg':  norm_bg,
    'r0':  norm_r0,
    'dx':  norm_dxy,
    'dy':  norm_dxy,
    'Jx':  norm_J,
    'Jy':  norm_J,
    'Jxy': norm_Jxy,
    'basis_coefs': norm_LWE
})


inp_dict = {
    'r0':  r0,
    'F':   F,
    'dx':  dx,
    'dy':  dy,
    'bg':  bg,
    'Jx':  Jx,
    'Jy':  Jy,
    'Jxy': Jxy,
    'basis_coefs': basis.coefs
}

_ = transformer.stack(inp_dict) # to create index mapping

#%%
x0 = [1.0,
      1.0, 1.0,
      0.0,
      0.0,
      0.0, 0.0,
      0.5,
      0.5,
      0.1] + [0,]*3 + [0,]*8

x0 = torch.tensor(x0).float().to(device).unsqueeze(0)
x0.requires_grad = True

if basis.coefs.requires_grad:
    buf = basis.coefs.detach().clone()
    basis.coefs = buf


#%%
from tools.utils import EarlyStopping

optimizer = optim.LBFGS([x0], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")

def loss_fn_all(A, B):
    return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all])

def func(x_):
    x_torch = transformer.destack(x_)
    return toy(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))

early_stopping = EarlyStopping(patience=2, tolerance=1e-4, relative=False)

for i in range(100):
    optimizer.zero_grad()

    loss = loss_fn_all(func(x0), PSF_0)

    if np.isnan(loss.item()):
        break
    
    early_stopping(loss)

    loss.backward(retain_graph=True)
    optimizer.step( lambda: loss_fn_all(func(x0), PSF_0) )

    print('Loss:', loss.item(), end="\r", flush=True)

    if early_stopping.stop:
        print('Stopped at it.', i, 'with loss:', loss.item())
        break
    

#%%
decomposed_variables = transformer.destack(x0)

for key, value in decomposed_variables.items():
    print(f"{key}: {value}")

draw_PSF_stack(PSF_0, PSF_1_joint:=func(x0), average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_0), destack(PSF_1_joint), 'Data', 'TipToy', title='IRDIS PSF', dpi=200)


#%%
from torch.autograd.functional import hessian, jacobian

hessian_mat = hessian(lambda x_: loss_fn_all(func(x_), PSF_0).log(), x0).squeeze()
hessian_mat_ = hessian_mat.clone()
hessian_mat_[1:,0] = hessian_mat_[0,1:]
hessian_mat_[0,0] = 0.0
hessian_mat_[hessian_mat.abs() < 1e-11] = 1e-11


#%%
# plt.figure(dpi=200)
plt.imshow(hessian_mat_.cpu().detach().numpy())

# Change labels of x ticks
xticks = np.arange(0, hessian_mat_.shape[-1])
xticklabels = \
    list(decomposed_variables.keys())[:-1] + \
    [f"Piston {i}" for i in range(2, 5)] + \
    [f"Tip {i}" for i in range(1, 5)] + \
    [f"Tilt {i}" for i in range(1, 5)]

# Insert value after 'bg'
xticklabels.insert(xticklabels.index('bg')+1, 'bg R')
xticklabels.insert(xticklabels.index('F')+1,  'F R')

xticklabels[xticklabels.index('bg')] = 'bg L'
xticklabels[xticklabels.index('F')]  = 'F L'
plt.xticks(xticks, xticklabels, rotation=45, ha="right")
plt.yticks(xticks, xticklabels, rotation=45, ha="right")

#%%
hessian_inv = torch.inverse(hessian_mat_)
variance_estimates = torch.diag(hessian_inv).abs().cpu().numpy()

plt.bar(np.arange(0, len(variance_estimates)), variance_estimates)
plt.xticks(np.arange(0, len(variance_estimates)), xticklabels, rotation=45, ha="right")
plt.yscale('log')
plt.ylabel('Variance of the estimated paraemters')
plt.title('Fisher information per parameter')

#%%
L_complex, V_complex = torch.linalg.eig(hessian_mat_)

L = L_complex.real.cpu().numpy()
V = V_complex.real.cpu().numpy()
V = V / np.linalg.norm(V, axis=0)

for eigenvalue_id in range(0, L.shape[0]):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].set_title('Eigenvalues')
    ax[0].bar(np.arange(0, L.shape[0]), np.abs(L))
    ax[0].bar(eigenvalue_id, np.abs(L[eigenvalue_id]), color='red')

    ax[1].set_title(f'Eigenvector #{eigenvalue_id}')
    ax[1].bar(np.arange(0, V.shape[0]), np.abs(V[:, eigenvalue_id]), color='green')
    ax[1].set_xticks(xticks, xticklabels, rotation=45, ha="right")

    plt.show()

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
    LWE_phase = torch.einsum('mn,nwh->mwh', basis.coefs, basis.modal_basis).cpu().numpy()[0,...]
    plt.imshow(LWE_phase)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('LWE OPD, [nm] RMS')

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