#%%
%reload_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tools.utils import plot_radial_profiles_new, SR, draw_PSF_stack, rad2mas, cropper, EarlyStopping
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from tools.config_manager import GetSPHEREonsky
from project_globals import SPHERE_DATA_FOLDER, device
from torchmin import minimize
from astropy.stats import sigma_clipped_stats

'''
files_1 = os.listdir(SPHERE_DATA_FOLDER+'IRDIS_fitted')
files_2 = os.listdir(SPHERE_DATA_FOLDER+'IRDIS_fitted_prev_BFBS-2')

files_1 = set(files_1)
files_2 = set(files_2)

files_diff = list(files_2-files_1)

for file in files_diff:
    with open(SPHERE_DATA_FOLDER+'IRDIS_fitted_prev_BFBS-2/'+file, 'rb') as handle:
        data_1 = pickle.load(handle)
        PSF_ = data_1['Img. fit']
        if np.isfinite(PSF_.sum()):
            print('Juan.', file)
            break
'''

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
psf_df = psf_df[psf_df['Multiples'] == False]
# psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
# psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]
#%
subset_df = psf_df[psf_df['High quality'] == True]
subset_df = subset_df[subset_df['High SNR'] == True]
# subset_df = subset_df[subset_df['LWE'] == False]
subset_df = subset_df[subset_df['LWE'] == True]
subset_df = subset_df[subset_df['Central hole'] == False]

print(f"Total samples: {len(subset_df)}")

#%%
from matplotlib.colors import LogNorm
from data_processing.SPHERE_preproc_utils import LoadSPHEREsampleByID

# samples_ids = np.array(subset_df.index.tolist())
sample_id = 768 #np.random.choice(samples_ids)

print("Chosen sample:", sample_id)

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

merged_config['NumberSources'] = merged_config['NumberSources'].int().item()
merged_config['sensor_science']['FieldOfView'] = PSF_0.shape[-1]
merged_config['DM']['DmHeights'] = torch.tensor(merged_config['DM']['DmHeights'], device=device)
merged_config['sources_HO']['Wavelength'] = merged_config['sources_HO']['Wavelength']
merged_config['sources_HO']['Height'] = torch.inf

# if psf_df.loc[sample_id]['Nph WFS'] < 10:
PSF_mask   = PSF_mask * 0 + 1
# LWE_flag   = psf_df.loc[sample_id]['LWE']
LWE_flag = True
wings_flag = True #psf_df.loc[sample_id]['Wings']
# wings_flag = False

#%%
# plt.imshow(PSF_0[0,0,...].abs().log10().cpu().numpy())
# plt.show()
# plt.imshow(PSF_0[0,1,...].abs().log10().cpu().numpy())
# plt.show()

#%% Initialize model
# from PSF_models.Tiptiptorch_SPHERE_multisrc import TipTorch
from PSF_models.TipTorch import TipTorch_new
from tools.utils import LWE_basis

# tiptorch = TipTorch(merged_config, None, device, oversampling=1)
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        True,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          False
}

tiptorch = TipTorch_new(merged_config, 'SCAO', None, PSD_include, 'sum', device, oversampling=1)
tiptorch.to_float()

# _ = tiptorch()
basis = LWE_basis(tiptorch)

# print(tiptorch.WFS_Nph.item())
# _ = tiptorch({ 'bg': bg.unsqueeze(0).to(device) })
# _ = tiptorch({ 'dx': torch.tensor([[0.0, 0.0]]).to(device) })
# _ = tiptorch({ 'dy': torch.tensor([[0.0, 0.0]]).to(device) })

PSF_1 = tiptorch()
#print(tiptorch.EndTimer())
# PSF_DL = tiptorch.DLPSF()


#%%
plt.imshow(PSF_1[0,0,...].log10().cpu().numpy())
plt.show()
plt.imshow(PSF_1[0,1,...].log10().cpu().numpy())
plt.show()

plt.imshow(PSF_0[0,0,...].abs().log10().cpu().numpy())
plt.show()
plt.imshow(PSF_0[0,1,...].abs().log10().cpu().numpy())
plt.show()

#%%
from tools.utils import safe_centroid, RadialProfile

def calc_profile(data, xycen=None):
    xycen = safe_centroid(data) if xycen is None else xycen
    edge_radii = np.arange(data.shape[-1]//2)
    rp = RadialProfile(data, xycen, edge_radii)
    return rp.profile


dk = 2*tiptorch.kc / tiptorch.nOtf_AO

PSD_norm = lambda wvl: (dk*wvl*1e9/2/np.pi)**2

PSDs = {entry: (tiptorch.PSDs[entry].clone().squeeze() * PSD_norm(500e-9)) for entry in tiptorch.PSD_entries if tiptorch.PSDs[entry].ndim > 1}
PSDs['chromatism'] = PSDs['chromatism'].mean(dim=0)

plt.figure(figsize=(8, 6))

PSD_map  = tiptorch.PSD[0,...].mean(dim=0).real.cpu().numpy()
k_map    = tiptorch.k[0,...].cpu().numpy()
k_AO_map = tiptorch.k_AO[0,...].cpu().numpy()

center    = [k_map.shape[0]//2, k_map.shape[1]//2]
center_AO = [k_AO_map.shape[0]//2, k_AO_map.shape[1]//2]

PSD_prof = calc_profile(PSD_map,  center)
freqs    = calc_profile(k_map,    center)
freqs_AO = calc_profile(k_AO_map, center_AO)

freq_cutoff = tiptorch.kc

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

# plt.savefig(f"C:/Users/akuznets/Desktop/thesis_results/SPHERE/PSDs/profiles.pdf", dpi=300)

# C:\Users\akuznets\Desktop\thesis_results\MUSE\PSDs



#%%
draw_PSF_stack(PSF_0*PSF_mask, PSF_1, average=True, min_val=2e-6, crop=80, scale='log')
# mask_in  = tiptorch.mask_rim_in.unsqueeze(1).float()
# mask_out = tiptorch.mask_rim_out.unsqueeze(1).float()

#%% PSF fitting (no early-stopping)
from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer
from tools.utils import mask_circle

# np.save('../data/LWE_basis_SPHERE.npy', basis.modal_basis.cpu().numpy())

norm_F   = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0)  ])
norm_bg  = TransformSequence(transforms=[ Uniform(a=-1e-6, b=1e-6) ])
norm_r0  = TransformSequence(transforms=[ Uniform(a=0,     b=0.5)  ])
norm_dxy = TransformSequence(transforms=[ Uniform(a=-1,    b=1)    ])
norm_J   = TransformSequence(transforms=[ Uniform(a=0,     b=30)   ])
norm_Jxy = TransformSequence(transforms=[ Uniform(a=0,     b=50)   ])
norm_LWE = TransformSequence(transforms=[ Uniform(a=-20,   b=20)   ])
norm_dn  = TransformSequence(transforms=[ Uniform(a=-0.02, b=0.02) ])
norm_wind_spd = TransformSequence(transforms=[ Uniform(a=0, b=20)  ])
norm_wind_dir = TransformSequence(transforms=[ Uniform(a=0, b=360) ])


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
    'wind_speed':  norm_wind_spd,
    'wind_dir':    norm_wind_dir,
    'basis_coefs': norm_LWE
})


inp_dict = {}

# Loop through the class attributes
for attr in ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy']:
    inp_dict[attr] = getattr(tiptorch, attr)

if wings_flag:
    inp_dict['wind_dir'] = tiptorch.wind_dir
    # inp_dict['wind_speed'] = tiptorch.wind_speed
    
if LWE_flag:
    inp_dict['basis_coefs'] = basis.coefs

_ = transformer.stack(inp_dict) # to create index mapping

#%%
mask_core = 1-mask_circle(PSF_0.shape[-1], 5, center=(0,0), centered=True)
mask_core *= mask_circle(PSF_0.shape[-1], 10, center=(0,0), centered=True)                    
mask_core = torch.tensor(mask_core[None,None,...]).to(device)

x0 = [norm_r0.forward(tiptorch.r0).item(),
      1.0, 1.0,
      0.0, 0.0,
      0.0, 0.0,
      0.0, 0.0,
      0.0,
      0.5,
      0.5,
      0.1]

if wings_flag:
    x0 = x0 + [
        norm_wind_dir.forward(tiptorch.wind_dir).item(),
        # norm_wind_spd.forward(tiptorch.wind_speed).item()
    ]   

if LWE_flag: x0 = x0 + [0,]*4 + [0,]*8

# x0 += (np.array([0,0,0,0,  0,-1,1,0,  1,0,0,-1])*10).tolist()
# x0 += (np.array([0,0,0,0,  0,1,-1,0, -1,0,0, 1])*10).tolist()
# x0 += (np.array([0,0,0,0,  0,-1,1,0, -1,0,0, 1])*10).tolist()
# x0 += (np.array([0,0,0,0,  0,1,-1,0,  1,0,0,-1])*10).tolist()
# x0 += (np.array([0,0,0,0,  1,0,0,-1,  0,1,-1,0])*10).tolist()
# x0 += (np.array([0,0,0,0,  -1,0,0,1,  0,-1,1,0])*10).tolist()
# x0 += (np.array([1,0,0,1,  0,0,0,0,  0,0,0,0])*10).tolist()

x0 = torch.tensor(x0).float().to(device).unsqueeze(0)

def func(x_):
    x_torch = transformer.unstack(x_)
    if 'basis_coefs' in x_torch:
        return tiptorch(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))
    else:
        return tiptorch(x_torch)

# testo = func(x0)
# draw_PSF_stack(PSF_0, testo, average=True, crop=80, scale='log')

#%%
# pattern_3 = torch.tensor().float().to(device).unsqueeze(0)
# pattern_4 = torch.tensor().float().to(device).unsqueeze(0)

# # testo1 = torch.einsum('mn,nwh->mwh', pattern_pos, basis.modal_basis).squeeze().cpu()
# # testo2 = torch.einsum('mn,nwh->mwh', pattern_neg, basis.modal_basis).squeeze().cpu()
# # testo3 = torch.einsum('mn,nwh->mwh', pattern_1, basis.modal_basis).squeeze().cpu()
# # testo4 = torch.einsum('mn,nwh->mwh', pattern_2, basis.modal_basis).squeeze().cpu()
# # testo4 = torch.einsum('mn,nwh->mwh', pattern_4, basis.modal_basis).squeeze().cpu()
# # testo4 = torch.einsum('mn,nwh->mwh', pattern_4, basis.modal_basis).squeeze().cpu()

# plt.imshow(testo4.cpu().squeeze(0))
# plt.colorbar()
# plt.show()

#%
if LWE_flag:

    A = 50.0

    pattern_pos = torch.tensor([[0,0,0,0,  0,-1,1,0,  1,0,0,-1]]).to(device).float() * A
    pattern_neg = torch.tensor([[0,0,0,0,  0,1,-1,0, -1,0,0, 1]]).to(device).float() * A
    pattern_1   = torch.tensor([[0,0,0,0,  0,-1,1,0, -1,0,0, 1]]).to(device).float() * A
    pattern_2   = torch.tensor([[0,0,0,0,  0,1,-1,0,  1,0,0,-1]]).to(device).float() * A
    pattern_3   = torch.tensor([[0,0,0,0,  1,0,0,-1,  0,1,-1,0]]).to(device).float() * A
    pattern_4   = torch.tensor([[0,0,0,0,  -1,0,0,1,  0,-1,1,0]]).to(device).float() * A

    gauss_penalty = lambda A, x, x_0, sigma: A * torch.exp(-torch.sum((x - x_0) ** 2) / (2 * sigma ** 2))
    img_punish = lambda x: ( (func(x)-PSF_0) * PSF_mask ).flatten().abs().sum()
    Gauss_err  = lambda pattern, coefs: (pattern * gauss_penalty(5, coefs, pattern, A/2)).flatten().abs().sum()
            
    LWE_regularizer = lambda c: \
        Gauss_err(pattern_pos, c) + Gauss_err(pattern_neg, c) + \
        Gauss_err(pattern_1, c)   + Gauss_err(pattern_2, c) + \
        Gauss_err(pattern_3, c)   + Gauss_err(pattern_4, c)
    
    def loss_fn(x_):
        coefs_ = transformer.unstack(x_)['basis_coefs']
        loss = img_punish(x_)  + LWE_regularizer(coefs_) + (coefs_**2).mean()*1e-4
        return loss
    
else:
    def loss_fn(x_):
        loss = (func(x_)-PSF_0)*PSF_mask
        return loss.flatten().abs().sum()

result = minimize(loss_fn, x0, max_iter=200, tol=1e-4, method='bfgs', disp=2)

x0 = result.x
# x0_buf = x0.clone()

#%%
from tools.utils import BuildPTTBasis, decompose_WF, project_WF, calc_WFE

LWE_coefs = transformer.unstack(x0)['basis_coefs'].clone()
PTT_basis = BuildPTTBasis(tiptorch.pupil.cpu().numpy(), True).to(device).float()

TT_max = PTT_basis.abs()[1,...].max().item()
pixel_shift = lambda coef: 4 * TT_max * rad2mas / tiptorch.psInMas / tiptorch.D * 1e-9 * coef

LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, basis.modal_basis)
PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, tiptorch.pupil)
PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, tiptorch.pupil)

x0_new = transformer.unstack(x0)
x0_new['basis_coefs'] = decompose_WF(LWE_OPD-PPT_OPD, basis.modal_basis, tiptorch.pupil) 
x0_new['dx'] -= pixel_shift(PTT_coefs[:, 2])
x0_new['dy'] -= pixel_shift(PTT_coefs[:, 1])
x0 = transformer.stack(x0_new)

plt.imshow((LWE_OPD-PPT_OPD).cpu().numpy()[0,...])
plt.colorbar()


'''
for i in range(basis.modal_basis.shape[0]):
    a = basis.modal_basis[i,...].cpu()
    plt.imshow(a, vmax=4.25, vmin=-4.25)
    # plt.colorbar()
    plt.axis('off')
    plt.savefig(f'C:/Users/akuznets/Desktop/thesis_results/SPHERE/modal_basis/LWE_mode_{i}.pdf', dpi=300)
'''
#%%
with torch.no_grad():
    PSF_1 = func(x0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_radial_profiles_new( PSF_0[:,0,...].cpu().numpy(), PSF_1[:,0,...].cpu().numpy(), 'Data', 'TipTorch', title='Left PSF',  ax=ax[0] )
    plot_radial_profiles_new( PSF_0[:,1,...].cpu().numpy(), PSF_1[:,1,...].cpu().numpy(), 'Data', 'TipTorch', title='Right PSF', ax=ax[1] )
    plt.show()
  
    draw_PSF_stack(PSF_0, PSF_1, average=True, crop=80)#, scale=None)

#%
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
    WFS_noise_var = tiptorch.dn + tiptorch.NoiseVariance(tiptorch.r0.abs())

    N_ph_0 = tiptorch.WFS_Nph.clone()

    def func_Nph(x):
        tiptorch.WFS_Nph = x
        var = tiptorch.NoiseVariance(tiptorch.r0.abs())
        return (WFS_noise_var-var).flatten().abs().sum()

    result_photons = minimize(func_Nph, N_ph_0, method='bfgs', disp=0)
    tiptorch.WFS_Nph = N_ph_0.clone()

    return result_photons.x

Nph_new = GetNewPhotons()

print(tiptorch.WFS_Nph.item(), Nph_new.item())


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

    # plt.show()
    numeration = str(eigenvalue_id)
    if eigenvalue_id < 10:
        numeration = '0' + numeration
    plt.savefig(f'C:/Users/akuznets/Desktop/buf/couplings/eigen_{numeration}.png', dpi=200)


#%%
print('\nStrehl ratio: ', SR(PSF_1, PSF_DL))
draw_PSF_stack(PSF_0, PSF_1, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]
plot_radial_profiles(destack(PSF_0), destack(PSF_1), 'Data', 'Tiptiptorch', title='IRDIS PSF', dpi=200)

#%%
with torch.no_grad():
    LWE_phase = torch.einsum('mn,nwh->mwh', basis.coefs, basis.modal_basis).cpu().numpy()[0,...]
    plt.imshow(LWE_phase)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('LWE OPD, [nm] RMS')

#%%
WFE = torch.mean(tiptorch.PSD.sum(axis=(-2,-1))**0.5)
WFE_jitter = tiptorch.D/4 * 1e9*(tiptorch.Jx+tiptorch.Jy)*0.5/rad2mas
WFE_total  = torch.sqrt(WFE**2 + WFE_jitter**2).item()

rads = 2*np.pi*WFE_total*1e-9 / tiptorch.wvl.flatten()[0]

S_0 = SR(PSF_0, PSF_DL).detach().cpu().numpy()
S = torch.exp(-rads**2).detach().cpu().numpy()

print(f'WFE: {WFE_total:.2f} nm (no LWE)')