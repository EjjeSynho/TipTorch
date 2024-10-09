#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt


import numpy as np	
from tools.utils import radial_profile, Center

PSF_0 = np.load('C:/Users/akuznets/Data/SPHERE/PSF_init.npy').squeeze()
PSF_1 = np.load('C:/Users/akuznets/Data/SPHERE/PSF_TipToy.npy').squeeze()
PSF_2 = np.load('C:/Users/akuznets/Data/SPHERE/PSF_OOPAO.npy')


def plot_radial_profile_2(PSFs, labels, title='', colors=None, dpi=300, scale='log'):
    center = Center(PSFs[0], centered=False)

    profile_0 = radial_profile(PSFs[0], center)[:32+1]
    profile_1 = radial_profile(PSFs[1], center)[:32+1]
    profile_diff = np.abs(profile_1-profile_0) / profile_0.max() * 100 #[%]

    fig = plt.figure(figsize=(6,4), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Relative intensity')
    if scale == 'log': ax.set_yscale('log')
    ax.set_xlim([0, len(profile_1)-1])
    ax.grid()
    ax2 = ax.twinx()
    ax2.set_ylim([0, profile_diff.max()*1.5])
    ax2.set_ylabel('Difference [%]')
    
    if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green']

    l3 = ax2.plot(profile_diff, label='Difference', color=colors[2], linewidth=1.5, linestyle='--')
    l2 = ax.plot(profile_0, label=labels[0], linewidth=2, color=colors[0])
    l1 = ax.plot(profile_1, label=labels[1], linewidth=2, color=colors[1])

    ls = l1+l2+l3
    labs = [l.get_label() for l in ls]
    ax2.legend(ls, labs, loc=0)
    
#%%
plot_radial_profile_2([PSF_0, PSF_1], ['Real', 'TipToy'], title='TipToy vs. Real',  colors=('tab:blue', 'tab:orange','tab:green'), dpi=100)
plot_radial_profile_2([PSF_0, PSF_2], ['Real',  'OOPAO'], title='OOPAO  vs. Real',  colors=('tab:blue', 'tab:red',   'tab:green'), dpi=100)
plot_radial_profile_2([PSF_1, PSF_2], ['TipToy','OOPAO'], title='OOPAO vs. TipToy', colors=('tab:orange', 'tab:red', 'tab:green'), dpi=100)