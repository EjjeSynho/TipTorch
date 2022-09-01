#%%
from os import path
import pickle
import torch
from torch import nn
#from PSFAO_SPHERE import PSFAO
import re
import os
from SPHERE_data import LoadSPHEREsampleByID
from parameterParser import parameterParser
from utils import r0_new, Center, BackgroundEstimate
from utils import OptimizeLBFGS, FitGauss2D

path_base = 'C:/Users/akuznets/Data/SPHERE/test/'
path_save = 'C:/Users/akuznets/Data/SPHERE/fitted/'


triggers = ['# end of import list', 'class TipToy(torch.nn.Module):', '# end of class defenition']
with open(r'TipToy_SPHERE.py', 'r') as script:
    lines = script.readlines()

line_ids = [0,0,0]

for i,line in enumerate(lines):
    if triggers[0] in line: line_ids[0] = i
    if triggers[1] in line: line_ids[1] = i
    if triggers[2] in line: line_ids[2] = i

reduced_script = lines[0:line_ids[0]] + lines[line_ids[1]:line_ids[2]]
code = "".join(reduced_script)

exec(code)

#%%
#num = 450
#num_id, data_sample = LoadSPHEREsampleByID('C:/Users/akuznets/Data/SPHERE/test/', num)

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP_old/P3')
path_ini  = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))
config_file = parameterParser(path_root, path_ini).params

def FitDataSample(data_test, save_fitted_path=None):
    toy = TipToy(config_file, data_test, 'CUDA')
    toy.norm_regime = 'max'

    im = data_test['image']
    if    toy.norm_regime == 'max': norma = im.max()
    elif  toy.norm_regime == 'sum': norma = im.sum()
    else: norma = 1.0
    PSF_0 = torch.tensor(im/norma, device=toy.device)

    dx_0, dy_0 = Center(PSF_0)
    bg_0 = BackgroundEstimate(PSF_0, radius=90)
    WFS_n = toy.NoiseVariance(r0_new(data_test['r0'], toy.GS_wvl, 0.5e-6))

    r0  = torch.tensor(r0_new(data_test['r0'], toy.wvl, 0.5e-6), requires_grad=True, device=toy.device)
    L0  = torch.tensor(25.0,  requires_grad=True,  device=toy.device)
    F   = torch.tensor(1.0,   requires_grad=True,  device=toy.device)
    dx  = torch.tensor(dx_0,  requires_grad=True,  device=toy.device)
    dy  = torch.tensor(dy_0,  requires_grad=True,  device=toy.device)
    bg  = torch.tensor(0.0,   requires_grad=True,  device=toy.device)
    n   = torch.tensor(WFS_n, requires_grad=True,  device=toy.device)
    Jx  = torch.tensor(10.0,  requires_grad=True,  device=toy.device)
    Jy  = torch.tensor(10.0,  requires_grad=True,  device=toy.device)
    Jxy = torch.tensor(2.0,   requires_grad=True,  device=toy.device)

    parameters = [r0, L0, F, dx, dy, bg, n, Jx, Jy, Jxy]

    loss = nn.L1Loss(reduction='sum')
    window_loss = lambda x, x_max: (x>0).float()*(0.01/x)**2 + (x<0).float()*100 + 100*(x>x_max).float()*(x-x_max)**2

    def loss_fn(a,b):
        z = loss(a,b) + \
            window_loss(r0_new(r0, 0.5e-6, toy.wvl), 1.5) + \
            window_loss(Jx, 50) + \
            window_loss(Jy, 50) + \
            window_loss(Jxy, 400) + \
            window_loss(n+toy.NoiseVariance(r0_new(r0, toy.GS_wvl, toy.wvl)), 1.5)
        return z

    optimizer_lbfgs = OptimizeLBFGS(toy, parameters, loss_fn)
    for _ in range(15):
        optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy, r0, n], 5)
        optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
        optimizer_lbfgs.Optimize(PSF_0, [Jx, Jy, Jxy], 3)

    PSF_1 = toy(*parameters)
    PSF_DL = toy.DLPSF()

    SR = lambda PSF: (PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()).item()

    #la chignon et tarte
    save_data = {
        'F':   F.item(),
        'dx':  dx.item(),
        'dy':  dy.item(),
        'r0':  r0.item(),
        'n':   toy.NoiseVariance(r0_new(r0, toy.GS_wvl, toy.wvl)).item(),
        'dn':  n.item(),
        'bg':  bg.item(),
        'Jx':  Jx.item(),
        'Jy':  Jy.item(),
        'Jxy': Jxy.item(),
        'SR data': SR(PSF_0), # data PSF
        'SR fit':  SR(PSF_1), # fitted PSF
        'FWHM fit':  FitGauss2D(PSF_1), 
        'FWHM data': FitGauss2D(PSF_0.float()),
        'Img. data': PSF_0.detach().cpu().numpy()*norma,
        'Img. fit':  PSF_1.detach().cpu().numpy()*norma
    }

    if save_fitted_path is not None:
        #with open(path_save+str(index)+'.pickle', 'wb') as handle:
        #    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #print('Saved at:', path_save+str(index)+'.pickle')
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
    
# %%
