#%%
%reload_ext autoreload
%autoreload 2

import sys
import os
import numpy as np
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
from torch import fft
from torch.nn.functional import interpolate

from pprint import pprint
# import torch.autograd.profiler as profiler

#%% ============================================================================
norm_regime = 'sum'

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

selected_files = [fitted_files[20]] #, fitted_files[40] ]
sample_ids = [ int(file.split('.')[0]) for file in selected_files ]

regime = '1P21I'

tensy = lambda x: torch.tensor(x).to(device)

PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess(sample_ids, regime, norm_regime, device)
norms = norms[:, None, None].cpu().numpy()

with open(fitted_folder + selected_files[0], 'rb') as handle:
    data = pickle.load(handle)

init_config['sensor_science']['FieldOfView'] = 255

toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True, oversampling=2)
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

PSFs_pred_1 = toy().detach().clone()

#%%
toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True, oversampling=4)
toy.optimizables = []

torch.cuda.empty_cache()

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

plot_radial_profiles(PSFs_pred_1[:,0,...], PSFs_pred_2[:,0,...], 'Before Update', 'After', title='PSFs', dpi=200, cutoff=32, scale='log')
plt.show()

#%% ================================================================================================================
%reload_ext autoreload
%autoreload 2

import sys
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
from pprint import pprint

norm_regime = 'sum'

fitted_folder = 'E:/ESO/Data/SPHERE/IRDIS_fitted_PAO_1P21I/'
fitted_files = os.listdir(fitted_folder)

# selected_files = [fitted_files[20], fitted_files[30], fitted_files[50], fitted_files[40] ]
selected_files = [fitted_files[50], fitted_files[30]] #, fitted_files[40] ]
# selected_files = [ fitted_files[20] ]
sample_ids = [ int(file.split('.')[0]) for file in selected_files ]

regime = 'different'

psdsaves = ['PSD_1', 'PSD_2', 'PSD_double']

#%%============================================================================
# PSFs_test_1 = []
PSFs_pred_1 = []
PSDec_1 = []
OTFs_1 = []

tensy = lambda x: torch.tensor(x).to(device)

c = 0

for file, sample in zip(selected_files,sample_ids):

    PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess([sample], regime, norm_regime, device)
    PSF_0 = PSF_0[...,1:,1:]
    init_config['sensor_science']['FieldOfView'] = 255
    
    norms = norms[:, None, None].cpu().numpy()

    with open(fitted_folder + file, 'rb') as handle:
        data = pickle.load(handle)

    toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True)
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
    toy.dx   = tensy( [60]    ).flatten()

    c += 1

    PSFs_pred_1.append( toy().detach().clone()[0,...] )
    PSDec_1.append( toy.PSD.detach().clone() )
    OTFs_1.append( toy.OTF.detach().clone() )
    # PSFs_test_1.append( torch.tensor( data['Img. fit'][0,...] / norms).to(device) )

# PSFs_test_1 = torch.stack(PSFs_test_1)
PSFs_pred_1 = torch.stack(PSFs_pred_1)
PSDec_1 = torch.stack(PSDec_1).squeeze()
OTFs_1 = torch.stack(OTFs_1).squeeze()

# draw_PSF_stack(PSFs_test_1, PSFs_pred_1)

#%%============================================================================
PSF_0, bg, norms, data_samples, init_config = SPHERE_preprocess(sample_ids, regime, norm_regime, device)
PSF_0 = PSF_0[...,1:,1:]
init_config['sensor_science']['FieldOfView'] = 255
    
if len(sample_ids) == 1:
    norms = norms[None,...]

datas = []
for file in selected_files:
    with open(fitted_folder + file, 'rb') as handle:
        datas.append( pickle.load(handle) )

toy = TipTorch(init_config, norm_regime, device, TipTop=False, PSFAO=True)
toy.optimizables = []

tensy = lambda x: torch.tensor(x).to(device)

toy.F     = tensy( [data['F']     for data in datas] ).squeeze()
toy.bg    = tensy( [data['bg']    for data in datas] ).squeeze()
toy.Jy    = tensy( [data['Jy']    for data in datas] ).flatten()
toy.Jxy   = tensy( [data['Jxy']   for data in datas] ).flatten()
toy.Jx    = tensy( [data['Jx']    for data in datas] ).flatten()
toy.dx    = tensy( [data['dx']    for data in datas] ).flatten()
toy.dy    = tensy( [data['dy']    for data in datas] ).flatten()
toy.b     = tensy( [data['b']     for data in datas] ).flatten()
toy.r0    = tensy( [data['r0']    for data in datas] ).flatten()
toy.amp   = tensy( [data['amp']   for data in datas] ).flatten()
toy.beta  = tensy( [data['beta']  for data in datas] ).flatten()
toy.theta = tensy( [data['theta'] for data in datas] ).flatten()
toy.alpha = tensy( [data['alpha'] for data in datas] ).flatten()
toy.ratio = tensy( [data['ratio'] for data in datas] ).flatten()
toy.dx   = tensy(  [60]*len(sample_ids)              ).flatten()


# PSFs_test = []
# for i in range(len(datas)):
#     PSFs_test.append( torch.tensor( datas[i]['Img. fit'][0,...] / norms[i,:,None,None].cpu().numpy() ).to(device) )
# PSFs_test = torch.stack(PSFs_test)

# if len(sample_ids) != 1:
    # PSFs_test = PSFs_test.squeeze()

PSF_pred = toy()

PSDec = toy.PSD.detach().clone().squeeze()
OTFs = toy.OTF.detach().clone().squeeze()

# draw_PSF_stack(PSFs_test, PSF_pred)
# draw_PSF_stack(PSFs_pred_1, PSF_pred, scale='log', min_val=1e-7, max_val=1e16)

# plot_radial_profiles(PSFs_pred_1[0,0,...].unsqueeze(0), PSF_pred[0,0,...].unsqueeze(0), 'L_1', 'L', title='PSFs', dpi=200, cutoff=14, scale='linear')


#%%

plt.imshow(torch.log10(PSDec_1[0,...]).cpu().numpy())
plt.show()
plt.imshow(torch.log10(PSDec[0,...]).cpu().numpy())
plt.show()

# %%

# plt.imshow(torch.log10(OTFs_1[0,0,...]).cpu().numpy())
# plt.show()
plt.imshow(torch.log10(OTFs[0,0,...]).cpu().numpy())
plt.show()
