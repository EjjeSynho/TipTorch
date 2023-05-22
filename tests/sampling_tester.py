#%%
%reload_ext autoreload
%autoreload 2

import sys
import os
sys.path.append('..')

from tools.config_manager import ConfigManager, GetSPHEREonsky
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
from tools.utils import draw_PSF_stack, plot_radial_profiles
from project_globals import device
import pickle
import torch
from torch.nn.functional import interpolate
import numpy as np
from torch import fft

from pprint import pprint
import torch.autograd.profiler as profiler

#%%
# %matplotlib qt
# %matplotlib inline

nPix = 256

oversampling = 1.#255
nOtf = nPix * oversampling

to_odd  = lambda f: int( np.ceil(f)//2 * 2 + 1 )
to_even = lambda f: int( np.ceil(f)//2 * 2 )
align_oddity = lambda x, y: to_odd(x) if y % 2 else to_even(x)

eps = 0.
dk = 1.

nOtf = align_oddity(nOtf, nPix)

def gen_grid(N):
    factor = 0.5*(1-N%2)
    return torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N)]*2, indexing = 'ij')

kx, ky = gen_grid(nOtf)
kx, ky = kx*dk+eps, ky*dk+eps

U, V = gen_grid(nOtf)
U = U / nOtf, V / nOtf


kc = 40

k2 = kx**2 + ky**2
k = torch.sqrt(k2)
mask = torch.zeros_like(k2).int()
mask[k2 <= kc**2] = 1

plt.imshow(mask)

mask_slice = mask[mask.shape[0]//2, :].tolist()
first_one = mask_slice.index(1)
last_one = len(mask_slice) - mask_slice[::-1].index(1) - 1

nOtf_AO = last_one - first_one + 1

corrected_ROI = slice(first_one, last_one+1)
mask_AO = mask[corrected_ROI, corrected_ROI]

PSD_padder = torch.nn.ZeroPad2d(first_one)

center_aligner = torch.exp(-1j*np.pi/nOtf*(kx+ky)*(1-nOtf%2))

plt.imshow(PSD_padder(mask_AO) + mask)

#%
from astropy.io import fits

pupil_path = 'C:/Users/akuznets/Projects/TipToy/data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits'
pupil = torch.tensor( fits.getdata(pupil_path).byteswap().newbyteorder() )

'''
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

pupil = torch.tensor( mask_circle(N=pupla_size, r=pupla_size//2) )
'''
# plt.imshow(pupil)

#%
def fftAutoCorr(x):
    x_fft = fft.fft2(x)
    return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft)) / x.size(0)*x.size(1) )


sampling = 4.13


pupil_padded = torch.nn.ZeroPad2d(int(pupil.shape[0]*sampling/2-pupil.shape[0]/2))(pupil)
fl_even = nOtf%2 == 0 and pupil_padded.shape[0]%2 == 0
pupil_padded = pupil_padded[:-1, :-1] if fl_even else pupil_padded


OTF_static_1 = interpolate(torch.real(fftAutoCorr(pupil_padded))[None,None,...], size=(nOtf,nOtf), mode='bicubic', align_corners=False).squeeze()
# OTF_static_2 = interpolate(torch.real(fftAutoCorr(pupil_padded))[None,None,...], size=(nOtf,nOtf), mode='bicubic', align_corners=False).squeeze()

plt.imshow(OTF_static_1)
# plt.imshow(OTF_static_2)



#%%
#%% ============================================================================
norm_regime = 'sum'

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

selected_files = [fitted_files[20]] #, fitted_files[40] ]
sample_ids = [ int(file.split('.')[0]) for file in selected_files ]

regime = '1P21I'

tensy = lambda x: torch.tensor(x).to(device)

PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess(sample_ids, regime, norm_regime)
norms = norms[:, None, None].cpu().numpy()

with open(fitted_folder + selected_files[0], 'rb') as handle:
    data = pickle.load(handle)

init_config['sensor_science']['FieldOfView'] = 255

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)


toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True)#, oversampling=1)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    toy.Update(reinit_grids=True, reinit_pupils=False)
    
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

toy.optimizables = []

data['dx']  = 0
data['dy']  = 0
data['Jx']  = 0
data['Jy']  = 0
data['Jxy'] = 0
data['F']   = data['F'] * 0 + 1
data['bg']  = data['bg'] * 0

toy.F     = tensy( data['F']     ).squeeze()
toy.bg    = tensy( data['bg']    ).squeeze()
toy.Jy    = tensy( data['Jy']    ).flatten()
toy.Jxy   = tensy( data['Jxy']   ).flatten()
toy.Jx    = tensy( data['Jx']    ).flatten()
toy.dx    = tensy( data['dx']    ).flatten()
toy.dy    = tensy( data['dy']    ).flatten()
toy.b     = tensy( data['b']     ).flatten()
toy.r0    = tensy( data['r0']    ).flatten()
toy.amp   = tensy( data['amp']   ).flatten()
toy.beta  = tensy( data['beta']  ).flatten()
toy.theta = tensy( data['theta'] ).flatten()
toy.alpha = tensy( data['alpha'] ).flatten()
toy.ratio = tensy( data['ratio'] ).flatten()

#%%
# start.record()
PSFs_pred_1 = toy().detach().clone()
# PSFs_test_1 = torch.tensor( data['Img. fit'][0,...] / norms).to(device)
# a_1 = toy.PSD.detach().cpu().clone()

# PSD_1 = toy.PSD.detach().cpu().numpy()
#%
# end.record()
# torch.cuda.synchronize()
# print(start.elapsed_time(end))

# plt.imshow(torch.log10(PSFs_pred_1[0,0,...].abs()).cpu().numpy())
plt.imshow(torch.abs(PSFs_pred_1[0,0,...].abs()).cpu().numpy())



#%%

#%%

test = toy.OTF.detach().cpu().numpy()

with open('../data/test.pkl', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('../data/test.pkl', 'rb') as handle:
    test = pickle.load(handle)

#%%
toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True, oversampling=4)
toy.optimizables = []

toy.F     = tensy( data['F']     ).squeeze()
toy.bg    = tensy( data['bg']    ).squeeze()
toy.Jy    = tensy( data['Jy']    ).flatten()
toy.Jxy   = tensy( data['Jxy']   ).flatten()
toy.Jx    = tensy( data['Jx']    ).flatten()
toy.dx    = tensy( data['dx']    ).flatten()
toy.dy    = tensy( data['dy']    ).flatten()
toy.b     = tensy( data['b']     ).flatten()
toy.r0    = tensy( data['r0']    ).flatten()
toy.amp   = tensy( data['amp']   ).flatten()
toy.beta  = tensy( data['beta']  ).flatten()
toy.theta = tensy( data['theta'] ).flatten()
toy.alpha = tensy( data['alpha'] ).flatten()
toy.ratio = tensy( data['ratio'] ).flatten()

PSFs_pred_2 = toy().detach().clone()

a_2 = toy.PSD.detach().cpu().clone()

plot_radial_profiles(PSFs_pred_1[:,0,...], PSFs_pred_2[:,0,...], 'Before Update', 'After', title='PSFs', dpi=200, cutoff=32, scale='log')
plt.show()

#%%

%matplotlib qt
draw_PSF_stack(PSFs_pred_1, PSFs_pred_2)

#%%
plt.figure(1)
plt.imshow(PSFs_pred_1[0,0,...].cpu().numpy())
plt.show()

plt.figure(2)
plt.imshow(PSFs_pred_2[0,0,...].cpu().numpy())
plt.show()

# %%

a_2_ = interpolate(a_2, size=(a_1.shape[-2:]), mode='bilinear')

#%
# draw_PSF_stack(a_1/a_1.sum(), a_2_/a_2_.sum())

plot_radial_profiles((a_1/a_1.sum())[:,0,...], (a_2_/a_2_.sum())[:,0,...], 'Before Update', 'After', title='PSFs', dpi=200, cutoff=32, scale='log')


#%%
# %matplotlib inline
%matplotlib qt

with open('C:/Users/akuznets/Projects/TipToy/data/temp/PSD_spatio-temporal_new.pkl', 'rb') as handle:
    cov_new = pickle.load(handle).squeeze()
    
with open('C:/Users/akuznets/Projects/TipToy/data/temp/PSD_spatio-temporal_old.pkl', 'rb') as handle:
    cov_old = pickle.load(handle).squeeze()


cov_new = 2*fft.fftshift(fft.fft2(fft.fftshift(cov_new, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1
cov_old = 2*fft.fftshift(fft.fft2(fft.fftshift(cov_old, dim=(-2,-1))), dim=(-2,-1)) # FFT axes are -2,-1


plt.figure(1)
plt.imshow(torch.log10(cov_new.real).squeeze().cpu().numpy())
plt.show()

plt.figure(2)
plt.imshow(torch.log10(cov_old.real).squeeze().cpu().numpy())
plt.show()

a = torch.log10(cov_new+1e-6)
print(torch.where(torch.isnan(a)))
# [0, 0]
# [0, 0]
# [83, 93]
# [ 73, 103]

b = torch.log10(cov_old+1e-6)
print(torch.where(torch.isnan(b)))


# %%
from torch import fft
from tools.utils import pdims

PSD = toy.PSD[0,0,...]

align_even = torch.exp(-1j*np.pi*(toy.U+toy.V)*(1-toy.nOtf%2)).squeeze()

cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD*align_even))) # FFT axes are -2,-1

SF = cov.abs().amax(dim=(-2,-1), keepdim=True) - cov.abs()

OTF_turb  = torch.exp(-0.5*SF*pdims(2*np.pi*1e-9/toy.wvl,2)**2)

OTF = OTF_turb[0,0,...]

test = fft.fftshift( fft.fft2( fft.fftshift(OTF*align_even) ) )

plt.imshow(np.log10(OTF.abs().cpu().numpy()))

#%%
OTF = toy.OTF.clone()[0,0,...]

test = fft.fftshift( fft.fft2(OTF) )

plt.imshow(np.log10(OTF.abs().cpu().numpy()))

plt.imshow(np.log10(test.abs().cpu().numpy()))

#%%

test0 = torch.zeros([50]*2)

N = test0.shape[0]

center = test0.shape[0]//2
test0[center,center] = 1
test0[center,center-1] = 1
test0[center-1,center] = 1
test0[center-1,center-1] = 1

for i in range(2, 26):
    test0[center-i:center+i, center-i:center+i] += 1

test0 = test0**3

def gen_grid(N):
    factor = 0.5*(1-N%2)
    return torch.meshgrid(*[torch.linspace(-N//2+N%2+factor, N//2-factor, N)]*2, indexing = 'ij')

test0 = OTF[127-10:127+12, 127-10:127+12].cpu().real
al = align_even[127-10:127+12, 127-10:127+12].cpu()

N = test0.shape[0]
U,V = gen_grid(N)
U, V = U/N, V/N

align_even = torch.exp(-1j*np.pi*(U+V)*(1-N%2))

plt.imshow(test0)

#%%


# OTF_padded = torch.nn.functional.pad(test0*align_even, (0, 1, 0, 1))  # Pad by 1 pixel


test1 = fft.fftshift( fft.ifft2( fft.ifftshift( test0 ) ) )
# test1 = fft.fft2( fft.fftshift(test0 ) )

plt.imshow(test0)
plt.show()
plt.imshow(torch.log10(test1.real))
plt.show()


