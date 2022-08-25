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


triggers = ['# end of import list', 'class PSFAO(torch.nn.Module):', '# end of class defenition']
with open(r'PSFAO_SPHERE.py', 'r') as script:
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

path_root = path.normpath('C:/Users/akuznets/Projects/TIPTOP/P3')
path_ini  = path.join(path_root, path.normpath('aoSystem/parFiles/irdis.ini'))
config_file = parameterParser(path_root, path_ini).params

def FitDataSample(data_test, save_fitted_path=None):
    psfao = PSFAO(config_file, data_test, 'CUDA')
    
    im = data_test['image']
    norma = im.sum()
    PSF_0 = torch.tensor(im/norma, device=psfao.device)

    dx_0, dy_0 = Center(PSF_0)
    bg_0 = BackgroundEstimate(PSF_0, radius=90)

    r0    = torch.tensor(r0_new(data_test['r0'], psfao.wvl, 0.5e-6), requires_grad=True, device=psfao.device)
    L0    = torch.tensor(25.0,  requires_grad=False, device=psfao.device) # Outer scale [m]
    F     = torch.tensor(1.0,   requires_grad=True,  device=psfao.device)
    bg    = torch.tensor(bg_0,  requires_grad=True,  device=psfao.device)
    b     = torch.tensor(1e-4,  requires_grad=True,  device=psfao.device) # Phase PSD background [rad² m²]
    amp   = torch.tensor(3.0,   requires_grad=True,  device=psfao.device) # Phase PSD Moffat amplitude [rad²]
    dx    = torch.tensor(dx_0,  requires_grad=True,  device=psfao.device)
    dy    = torch.tensor(dy_0,  requires_grad=True,  device=psfao.device)
    alpha = torch.tensor(0.1,   requires_grad=True,  device=psfao.device) # Phase PSD Moffat alpha [1/m]
    ratio = torch.tensor(1.0,   requires_grad=True,  device=psfao.device) # Phase PSD Moffat ellipticity
    theta = torch.tensor(0.0,   requires_grad=True,  device=psfao.device) # Phase PSD Moffat angle
    beta  = torch.tensor(1.6,   requires_grad=True,  device=psfao.device) # Phase PSD Moffat beta power law

    parameters = [r0, L0, F, dx, dy, bg, amp, b, alpha, beta, ratio, theta]


    loss_fn = nn.L1Loss(reduction='sum')
    optimizer_lbfgs = OptimizeLBFGS(psfao, parameters, loss_fn)

    for i in range(20):
        optimizer_lbfgs.Optimize(PSF_0, [F, dx, dy], 2)
        optimizer_lbfgs.Optimize(PSF_0, [bg], 2)
        optimizer_lbfgs.Optimize(PSF_0, [b], 2)
        optimizer_lbfgs.Optimize(PSF_0, [r0, amp, alpha, beta], 5)
        optimizer_lbfgs.Optimize(PSF_0, [ratio, theta], 5)

    PSF_1 = psfao(*parameters)
    PSF_DL = psfao.DLPSF()

    SR = lambda PSF: (PSF.max()/PSF_DL.max() * PSF_DL.sum()/PSF.sum()).item()

    #la chignon et tarte
    save_data = {
        'r0':        r0.item(),
        'F':         F.item(),
        'dx':        dx.item(),
        'dy':        dy.item(),
        'bg':        bg.item(),
        'amp':       amp.item(),
        'b':         b.item(),
        'alpha':     alpha.item(),
        'beta':      beta.item(),
        'ratio':     ratio.item(),
        'theta':     theta.item(),
        'SR data':   SR(PSF_0), # data PSF
        'SR fit':    SR(PSF_1), # fitted PSF
        'FWHM fit':  FitGauss2D(PSF_1), 
        'FWHM data': FitGauss2D(PSF_0.float()),
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
    
# %%
