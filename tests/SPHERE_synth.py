#%%
# %reload_ext autoreload
# %autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
from torch import nn, optim
import numpy as np
from tools.utils import OptimizeLBFGS, ParameterReshaper, plot_radial_profiles, SR, draw_PSF_stack
from PSF_models.TipToy_SPHERE_multisrc import TipToy
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess
import matplotlib.pyplot as plt
from tools.utils import rad2mas, rad2arc

from globals import SPHERE_DATA_FOLDER, device


#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]
good_ids = psf_df.index.values.tolist()

#%% ================================= Read OOPAO sample =================================

regime = 'NP2NI'
# regime = '1P21I'
norm_regime = 'sum'

sample_ids = [992]
# sample_ids = [1209]

#% Initialize model
PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True)
toy = TipToy(synth_config, norm_regime, device)


#%%
def clone_members(toy, member_names):
    cloned_members = {}
    for name in member_names:
        member = getattr(toy, name, None)
        if member is not None and torch.is_tensor(member):
            cloned_members[name] = member.detach().clone()
        else:
            print(f"{name} is not a tensor or not found in the toy instance.")
    return cloned_members


def GetJitter():
    TT_res = synth_samples[0]['WFS']['tip/tilt residuals']
    D = synth_config['telescope']['TelescopeDiameter']
    ang_pix = synth_samples[0]['Detector']['psInMas'] / rad2mas
    jitter = lambda a: 2*2*a/D/ang_pix
    TT_jitter = jitter(TT_res)
    Jx = TT_jitter[:,0].std() * ang_pix * rad2mas * 2.355
    Jy = TT_jitter[:,1].std() * ang_pix * rad2mas * 2.355
    return Jx, Jy


Jx, Jy = GetJitter()
wind_dir   = synth_config['atmosphere']['WindSpeed'].clone().detach()
wind_speed = synth_config['atmosphere']['WindDirection'].clone().detach()
r0         = rad2arc*0.976*synth_config['atmosphere']['Wavelength'] / synth_config['atmosphere']['Seeing'].clone().detach()
n          = toy.NoiseVariance(toy.r0).clone().detach()


loss = nn.L1Loss(reduction='sum')
# Confines a value between 0 and the specified value
window_loss = lambda x, x_max: \
    torch.gt(x,0)*(0.01/x)**2 + torch.lt(x,0)*100 + 100*torch.gt(x,x_max)*(x-x_max)**2

def loss_fn(a,b):
    z = loss(a,b) + \
        window_loss(toy.r0, 0.5).sum() * 5.0 + \
        window_loss(toy.Jx, 50).sum() * 0.5 + \
        window_loss(toy.Jy, 50).sum() * 0.5 + \
        window_loss(toy.Jxy, 400).sum() * 0.5 + \
        window_loss(toy.dn + toy.NoiseVariance(toy.r0), 1.5).sum()
    return z

delta = 0.2
cloned_members = []

for i in range(5):
    PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True)
    toy = TipToy(synth_config, norm_regime, device)

    ddy = np.random.uniform(-2.0, 2.0)
    ddx = np.random.uniform(-2.0, 2.0)
    dJx = np.random.uniform(1-delta, 1+delta)
    dJy = np.random.uniform(1-delta, 1+delta)
    dr0 = np.random.uniform(1-delta, 1+delta)
    dwd = np.random.uniform(1-delta, 1+delta)
    dws = np.random.uniform(1-delta, 1+delta)
    ddn = np.random.uniform( -delta,   delta)

    toy.optimizables = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy', 'wind_dir', 'wind_speed']
    _ = toy({ 
        'Jxy': torch.tensor([0.1]*toy.N_src, device=toy.device).flatten(),
        'Jx':  torch.tensor([Jx*dJx]*toy.N_src, device=toy.device).flatten(),
        'Jy':  torch.tensor([Jy*dJy]*toy.N_src, device=toy.device).flatten(),
        'dx':  torch.tensor([0.0+ddx]*toy.N_src, device=toy.device).flatten(),
        'dy':  torch.tensor([0.0+ddy]*toy.N_src, device=toy.device).flatten(),
        'dn':  torch.tensor([ddn]*toy.N_src, device=toy.device).flatten(),
        'bg':  bg.to(device)
    })

    toy.wind_speed.data = synth_config['atmosphere']['WindSpeed'].clone() * dws
    toy.wind_dir.data = synth_config['atmosphere']['WindDirection'].clone() * dwd
    toy.r0.data = rad2arc*0.976*synth_config['atmosphere']['Wavelength'] / synth_config['atmosphere']['Seeing'].clone().detach() * dr0
    toy.dn.data = torch.tensor([ddn]*toy.N_src, device=toy.device).flatten()

    PSF_3  = toy()
    PSF_DL = toy.DLPSF()

    optimizer_lbfgs = OptimizeLBFGS(toy, loss_fn)

    for i in range(10):
        optimizer_lbfgs.Optimize(PSF_2, [toy.F], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.bg], 2)
        optimizer_lbfgs.Optimize(PSF_2, [toy.dx, toy.dy], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.r0, toy.dn], 5)
        optimizer_lbfgs.Optimize(PSF_2, [toy.wind_dir, toy.wind_speed], 3)
        optimizer_lbfgs.Optimize(PSF_2, [toy.Jx, toy.Jy, toy.Jxy], 3)

    PSF_3 = toy()
    print('\nStrehl ratio: ', SR(PSF_3, PSF_DL))

    cloned_members.append( clone_members(toy, toy.optimizables) )

#%%
PSF_2, bg, norms, synth_samples, synth_config = SPHERE_preprocess(sample_ids, regime, norm_regime, synth=True)

r0         = rad2arc*0.976*synth_config['atmosphere']['Wavelength'] / synth_config['atmosphere']['Seeing'].clone().detach()
wind_dir   = synth_config['atmosphere']['WindSpeed'].clone().detach()
wind_speed = synth_config['atmosphere']['WindDirection'].clone().detach()
Jx, Jy     = GetJitter()

Jx = torch.tensor([[Jx]]).to(device)
Jy = torch.tensor([[Jy]]).to(device)
r0 = torch.tensor([[r0]]).to(device) if regime == '1P21I' else r0.unsqueeze(0)

wind_dir = wind_dir.to(device).unsqueeze(0)
wind_speed = wind_speed.to(device).unsqueeze(0)

wds  = torch.stack([x['wind_dir'] for x in cloned_members]).to(device)
wss  = torch.stack([x['wind_speed'] for x in cloned_members]).to(device)
Jxys = torch.stack([x['Jxy'] for x in cloned_members]).to(device)
Jxs  = torch.stack([x['Jx'] for x in cloned_members]).to(device)
Jys  = torch.stack([x['Jy'] for x in cloned_members]).to(device)
r0s  = torch.stack([x['r0'] for x in cloned_members]).to(device)
dxs  = torch.stack([x['dx'] for x in cloned_members]).to(device)
dys  = torch.stack([x['dy'] for x in cloned_members]).to(device)
dns  = torch.stack([x['dn'] for x in cloned_members]).to(device)

if regime == 'NP2NI':
    Jxys = Jxys.mean(dim=1)
    wds  = wds.mean(dim=1)
    wss  = wss.mean(dim=1)
    r0s  = r0s.mean(dim=1)
    dns  = dns.mean(dim=1)
    Jxs  = Jxs.mean(dim=1)
    Jys  = Jys.mean(dim=1)
    dxs  = dxs.mean(dim=1)
    dys  = dys.mean(dim=1)
      
d_wds_mean  = torch.mean(wds, dim=0).squeeze()[0].item()
d_wss_mean  = torch.mean(wss, dim=0).squeeze()[0].item()
d_Jxys_mean = torch.mean(Jxys, dim=0).item()
d_Jxs_mean  = torch.mean(Jxs, dim=0).item()
d_Jys_mean  = torch.mean(Jys, dim=0).item()
d_r0s_mean  = torch.mean(r0s, dim=0).item()
d_n_mean    = torch.mean(dns, dim=0).item()

d_wds_std   = torch.std(wds, dim=0).squeeze()[0].item()
d_wss_std   = torch.std(wss, dim=0).squeeze()[0].item()
d_Jxys_std  = torch.std(r0s, dim=0).item()
d_Jxs_std   = torch.std(Jxs, dim=0).item()
d_Jys_std   = torch.std(Jys, dim=0).item()
d_r0s_std   = torch.std(r0s, dim=0).item()
d_n_std     = torch.std(dns, dim=0).item()


# Define the parameters and their values
params = ['$r_0$,\n[cm]', '$J_x$,\n[mas]', '$J_y$,\n[mas]', r'$J_{x,y},$'+'\n[mas]', 'Wind dir.\n (ground)', 'Wind spd.\n (ground)', 'WFS \n noise \n (x10)']

values_modified = [d_r0s_mean*100,
                   d_Jxs_mean,
                   d_Jys_mean,
                   d_Jxys_mean,
                   d_wds_mean,
                   d_wss_mean,
                   (d_n_mean+n.mean().item())*10]

stds_modified = [d_r0s_std*100,
                 d_Jxs_std,
                 d_Jys_std,
                 d_Jxys_std,
                 d_wds_std,
                 d_wss_std,
                 d_n_std*10]

values_original = [r0.mean().item()*100,
                   Jx.item(),
                   Jy.item(),
                   0.1,
                   wind_dir.squeeze()[...,0].mean().item(),
                   wind_speed.squeeze()[...,0].mean().item(),
                   n.mean().item()*10]

fig, ax = plt.subplots()

width = 0.35

ind = np.arange(len(params))

plt.grid()

ax.bar(ind, values_original, width, label='Original')
ax.bar(ind + width, values_modified, width, yerr=stds_modified, label='Fitted', color='skyblue')

# Add some labels and formatting to the plot
ax.set_xlabel('Parameters')
ax.set_ylabel('Values')
ax.set_ylim([0, 50])
ax.set_axisbelow(True)
ax.set_title('Original and fitted values, '+regime+', '+str(sample_ids[0]))
ax.set_xticks(ind + width/2)
ax.set_xticklabels(params)
ax.legend()

# Display the plot
plt.show()


#%%
colors = [['tab:blue', 'tab:red'],
          ['tab:blue', 'tab:orange'],
          ['tab:red',  'tab:orange']]

# flux_ratio = (PSF_3.amax(dim=(-2,-1)) * PSF_2.amax(dim=(-2,-1)))
# flux_ratio = PSF_3.amax(dim=(-2,-1), keepdim=True) / PSF_2.amax(dim=(-2,-1), keepdim=True)
# toy.F.data /= flux_ratio
# PSF_3 = PSF_3 / flux_ratio

print('\nStrehl ratio: ', SR(PSF_3, PSF_DL))
draw_PSF_stack(PSF_2, PSF_3, average=True)

destack = lambda PSF_stack: [ x for x in torch.split(PSF_stack[:,0,...].cpu(), 1, dim=0) ]

plot_radial_profiles(destack(PSF_2), destack(PSF_3), 'OOPAO', 'TipToy', title='IRDIS PSF', dpi=200, colors=colors[2], cutoff=64)

pval = lambda x,y: 100*(1-x/y).mean().detach().cpu().numpy()

print( np.round(pval(r0,toy.r0), 1) )
print( np.round(pval(wind_dir,toy.wind_dir), 1) )
print( np.round(pval(wind_speed,toy.wind_speed), 1) )