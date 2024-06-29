#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas, cropper
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky
from project_globals import SPHERE_DATA_FOLDER, device


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted']   == False]
psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
psf_df = psf_df[psf_df['Central hole'] == False]
psf_df = psf_df[psf_df['Multiples']    == False]
psf_df = psf_df[psf_df['λ left (nm)']  == 1625]
psf_df = psf_df[ ~np.isnan(psf_df['Rate']) ]
# psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]

good_ids = psf_df.index.values.tolist()

# Randomly select 8 samples from the dataset
# ids = np.random.choice(good_ids, 8, replace=False)
# ids = np.array([ 487, 2803, 1053, 1653,  845, 1332, 3962,  183]) # one with one NaN
ids = np.array([ 1115, 2818, 637, 869, 1370,  159, 2719, 1588 ])

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
    sample_ids    = ids,
    norm_regime   = 'sum',
    split_cube    = False,
    PSF_loader    = lambda x: SamplesByIds(x, synth=False),
    config_loader = GetSPHEREonsky,
    framework     = 'pytorch',
    device        = device)

PSF_0   = torch.stack([PSF_data[i]['PSF (mean)' ] for i in range(len(ids))])
PSF_var = torch.stack([PSF_data[i]['PSF (var)'  ] for i in range(len(ids))])
bg      = torch.stack([PSF_data[i]['bg (mean)'  ] for i in range(len(ids))])
norms   = torch.stack([PSF_data[i]['norm (mean)'] for i in range(len(ids))])
del PSF_data

#%% Initialize model
from PSF_models.TipToy_SPHERE_multisrc import TipTorch

toy = TipTorch(merged_config, None, device)

_ = toy()

# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']
# toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'Jx', 'Jy', 'Jxy']
# toy.optimizables = []

# _ = toy({ 'bg': bg.to(device) })

PSF_1 = toy()
#print(toy.EndTimer())
PSF_DL = toy.DLPSF()

draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)
# mask_in  = toy.mask_rim_in.unsqueeze(1).float()
# mask_out = toy.mask_rim_out.unsqueeze(1).float()

'''
from tools.config_manager import ConfigManager
import json
from copy import deepcopy

config_manager = ConfigManager(GetSPHEREonsky())
config_manager.Convert(merged_config_2 := deepcopy(merged_config), framework='list')

with open('C:/Users/akuznets/Desktop/presa_buf/config.json', 'w') as f:
    json.dump(merged_config_2, f, indent=4)
'''

#%% PSF fitting (no early-stopping)
from tools.utils import LWE_basis
from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer

basis = LWE_basis(toy)

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
    'r0':  toy.r0,
    'F':   toy.F,
    'dx':  toy.dx,
    'dy':  toy.dy,
    'bg':  toy.bg,
    'Jx':  toy.Jx,
    'Jy':  toy.Jy,
    'Jxy': toy.Jxy,
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
x0 = x0.repeat(PSF_0.shape[0], 1)

x0.requires_grad = True

if basis.coefs.requires_grad:
    buf = basis.coefs.detach().clone()
    basis.coefs = buf

#%%
from tools.utils import EarlyStopping

optimizer = optim.LBFGS([x0], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
crop_all = cropper(PSF_0, 100)

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

    print('Loss:', loss.item()/PSF_0.shape[0], end="\r", flush=True)

    if early_stopping.stop:
        print('Stopped at it.', i, 'with loss:', loss.item()/PSF_0.shape[0])
        break

torch.cuda.empty_cache()

#%%
decomposed_variables = transformer.destack(x0)
# for key, value in decomposed_variables.items(): print(f"{key}: {value}")

with torch.no_grad():
    PSF_1_joint = func(x0)
    plot_radial_profiles_new(
        PSF_0[:,0,...].cpu().numpy(),
        PSF_1_joint[:,0,...].cpu().numpy(),
        'Data', 'TipToy', title='IRDIS PSF', dpi=200
    )
    # draw_PSF_stack(PSF_0, PSF_1_joint, average=True)#, scale=None)


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
    [f"Tip {i}"    for i in range(1, 5)] + \
    [f"Tilt {i}"   for i in range(1, 5)]

# Insert value after 'bg'
xticklabels.insert(xticklabels.index('bg')+1, 'bg R')
xticklabels.insert(xticklabels.index('F') +1, 'F R')

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

    # plt.show()
    numeration = str(eigenvalue_id)
    if eigenvalue_id < 10:
        numeration = '0' + numeration
    plt.savefig(f'C:/Users/akuznets/Desktop/buf/couplings/eigen_{numeration}.png', dpi=200)


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
WFE = torch.mean(toy.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = toy.D/4 * 1e9*(toy.Jx+toy.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / toy.wvl.flatten()[0]

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')