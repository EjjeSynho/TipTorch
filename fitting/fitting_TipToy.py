#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

from os import path
import pickle5 as pickle
import torch
from torch import nn
import re
import os
import numpy as np

from PSF_models.TipToy_SPHERE_multisrc import TipToy
from data_processing.SPHERE_data import LoadSPHEREsampleByID
from tools.parameter_parser import ParameterParser
from tools.utils import Center, BackgroundEstimate
from tools.utils import OptimizeLBFGS, FitGauss2D, SR

path_base = 'C:/Users/akuznets/Data/SPHERE/test/'
path_save = 'C:/Users/akuznets/Data/SPHERE/fitted/'

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')


path_ini  = '../parameter_files/irdis.ini'
config_file = ParameterParser(path_ini).params

#%%
def FitDataSample(data_sample, save_fitted_path=None):
    data_samples = [data_sample]
    norm_regime = 'max'
    toy = TipToy(config_file, data_samples, norm_regime, device)

    r_0_val  = data_samples[0]['r0']
    n_ph_val = data_samples[0]['WFS']['Nph vis']

    if np.isnan(r_0_val) or r_0_val is None:
        r_0_val = 0.19 #[m] median value for seeing

    if np.isnan(n_ph_val):
        n_subap = 1240
        rate = data_samples[0]['WFS']['rate']
        if np.isnan(rate): rate = 1380
        n_ph_val = 10**7.2/(rate*n_subap)

    ims = []
    for sample in data_samples:
        im = sample['image']
        if    toy.norm_regime == 'max': norma = im.max()
        elif  toy.norm_regime == 'sum': norma = im.sum()
        else: norma = 1.0
        ims.append(torch.tensor(im/norma, device=toy.device))

    PSF_0 = torch.tensor(torch.dstack(ims)).permute([2,0,1])
    dx_0, dy_0 = Center(PSF_0.squeeze(0))
    bg_0 = torch.tensor([BackgroundEstimate(PSF_, radius=90).item() for PSF_ in ims]).to(toy.device)

    init_torch_param = lambda x, N=None: \
        torch.tensor([x]*N, requires_grad=True, device=device).flatten() \
        if N is not None and N != 0 \
        else torch.tensor(x, device=device, requires_grad=True).flatten()

    r0  = init_torch_param(r_0_val, 1)
    L0  = init_torch_param(25.0, 1)
    F   = init_torch_param(1.0,  1)
    dx  = init_torch_param(dx_0, 1)
    dy  = init_torch_param(dy_0, 1)
    bg  = init_torch_param(bg_0, 1)
    dn  = init_torch_param(0.05, 1)
    Jx  = init_torch_param(5.0,  1)
    Jy  = init_torch_param(5.0,  1)
    Jxy = init_torch_param(2.0,  1)
    toy.WFS_Nph = init_torch_param(n_ph_val, 1)
    toy.WFS_Nph.requires_grad = True

    parameters = [r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy]
    loss = nn.L1Loss(reduction='sum')

    # Confines a value between 0 and the specified value
    window_loss = lambda x, x_max: \
        torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2

    # TODO: specify loss weights
    def loss_fn(a,b):
        z = loss(a,b) + \
            window_loss(r0, 0.5).sum() * 5.0 + \
            window_loss(Jx, 50).sum()*0.5 + \
            window_loss(Jy, 50).sum()*0.5 + \
            window_loss(Jxy, 400).sum()*0.5 + \
            window_loss(dn + toy.NoiseVariance(r0), 1.5).sum()
        return z

    optimizer_lbfgs = OptimizeLBFGS(toy, parameters, loss_fn)
    for _ in range(15):
        optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy, r0, dn, toy.WFS_Nph], 5)
        optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
        optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

    PSF_1  = toy.PSD2PSF(*parameters)
    PSF_DL = toy.DLPSF()

    save_data = {
        'F':   F.item(),
        'dx':  dx.item(),
        'dy':  dy.item(),
        'r0':  r0.item(),
        'n':   toy.NoiseVariance(r0.abs()).item(),
        'dn':  dn.item(),
        'bg':  bg.item(),
        'Jx':  Jx.item(),
        'Jy':  Jy.item(),
        'Jxy': Jxy.item(),
        'Nph WFS': toy.WFS_Nph.item(),
        'SR data': SR(PSF_0, PSF_DL), # data PSF
        'SR fit':  SR(PSF_1, PSF_DL), # fitted PSF
        'FWHM fit':  FitGauss2D(PSF_1.squeeze(0).float()), 
        'FWHM data': FitGauss2D(PSF_0.squeeze(0).float()),
        'Img. data': PSF_0.detach().cpu().numpy()*norma,
        'Img. fit':  PSF_1.detach().cpu().numpy()*norma
    }

    if save_fitted_path is not None:
        with open(save_fitted_path, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved at:', save_fitted_path)


files = os.listdir(path_base)
for file in files:
    path_data = path.join(path_base,file)
    index = int(re.findall(r'[0-9]+', file)[0])
    with open(path_data, 'rb') as handle:
        data_sample = pickle.load(handle)
    print('Fitting sample #'+str(index)+':')
    try:
        FitDataSample(data_sample, path_save+str(index)+'.pickle')
    except RuntimeError:
        print("Oops! Anyway...")
    
#la chignon et tarte
# %%
