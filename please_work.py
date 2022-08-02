#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path
import os
import seaborn as sns
import pandas as pd
from matplotlib import colors
import matplotlib
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%
path_fitted = 'C:\\Users\\akuznets\\Data\\SPHERE\\fitted 2\\'
path_input  = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\'

files_input  = os.listdir(path_input)
files_fitted = os.listdir(path_fitted)

ids_fitted = set( [int(file[:-7]) for file in files_fitted] )

data = []
for file in files_input:
    id_file = int(file.split('_')[0])
    if id_file in ids_fitted:
        with open(os.path.join(path_input, file), 'rb') as handle:
            data_input = pickle.load(handle)
        with open(os.path.join(path_fitted, str(id_file)+'.pickle'), 'rb') as handle:
            data_fitted = pickle.load(handle)
        data.append((data_input, data_fitted, id_file))

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

images_data = []
images_fit  = []

r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]

dataframe = []

for id in range(len(data)):
    wvl = data[id][0]['spectrum']['lambda']
    images_data.append(data[id][1]['Img. data'])
    images_fit.append(data[id][1]['Img. fit'])

    dataframe_buf = {}
    dataframe_buf['FWHM fit']  = np.hypot(data[id][1]['FWHM fit'][0], data[id][1]['FWHM fit'][1])
    dataframe_buf['FWHM data'] = np.hypot(data[id][1]['FWHM data'][0], data[id][1]['FWHM data'][1])
    dataframe_buf['index']     = data[id][2]
    dataframe_buf['SR fit']    = data[id][1]['SR fit']
    dataframe_buf['SR data']   = data[id][1]['SR data']
    dataframe_buf['SR SPARTA'] = data[id][0]['Strehl']
    dataframe_buf['Jx']  = data[id][1]['Jx']
    dataframe_buf['Jy']  = data[id][1]['Jy']
    dataframe_buf['Jxy'] = data[id][1]['Jxy']
    dataframe_buf['WFS noise'] = data[id][1]['n']
    dataframe_buf['$r_0$ SPARTA @ 500 [nm]'] = data[id][0]['r0']
    dataframe_buf['$r_0$ fitted @ 500 [nm]'] = r0_new(data[id][1]['r0'], 500e-9, wvl)
    dataframe_buf['Seeing fitted [asec]'] = seeing(r0_new(data[id][1]['r0'], 500e-9, wvl), 500e-9)
    dataframe_buf['Seeing SPARTA [asec]'] = data[id][0]['seeing']['SPARTA']
    dataframe_buf['Wavelength ($\mu m$)'] = np.round(wvl*1e6,3)
    dataframe.append(dataframe_buf)

dataframe = pd.DataFrame(dataframe)
dataframe['Img data'] = images_data
dataframe['Img fit']  = images_fit

#dataframe = dataframe[:100]

#plt.scatter(r0_data, r0_fit1)
#plt.xlim([-0.1,1.])
#plt.ylim([-0.1,1.])
#plt.grid()
#%%

for index, row in dataframe.iterrows():
    print(index, row['index'])


#%% --------------------------------------------------------------

def radial_profile(data, center=None):
    if center is None:
        center = (data.shape[0]//2, data.shape[1]//2)
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile[0:data.shape[0]//2]


%matplotlib qt
#%matplotlib inline

current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='darkslateblue')
fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot2grid((5, 10), (0, 0), colspan=4, rowspan=4)
plt.grid()

def set_lim(axx, lim):
    if hasattr(lim, '__len__()'):
        axx.set_ylim([lim[0], lim[1]])
        axx.set_xlim([lim[0], lim[1]])   
    else:
        axx.set_ylim([0, lim])
        axx.set_xlim([0, lim])

s = pd.Series([0,1,2,3,4,5,6])
s.plot.line(linewidth=1., color='gray', linestyle='--')
set_lim(ax1, 1.0)
ax1.grid()

#flag = 'SR SPARTA'
flag = 'r0'

choice = {
    'SR': ('SR fit', 'SR data', 1.0),
    'SR SPARTA': ('SR fit', 'SR SPARTA', 1.0),
    'seeing': ('Seeing fitted [asec]', 'Seeing SPARTA [asec]', 2.0),
    'r0': ('$r_0$ SPARTA @ 500 [nm]', '$r_0$ fitted @ 500 [nm]', 1.0),
    'FWHM': ('FWHM fit', 'FWHM data', [2.5, 6.0])
}

sns.scatterplot(x=choice[flag][0], y=choice[flag][1], hue='Wavelength ($\mu m$)', data=dataframe, picker=4)
set_lim(ax1, choice[flag][2])

ax2 = plt.subplot2grid((5, 10), (0, 4), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((5, 10), (0, 6), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((5, 10), (0, 8), colspan=2, rowspan=2)
ax5 = plt.subplot2grid((5, 10), (2, 4), colspan=6, rowspan=3)
ax6 = ax5.twinx()
ax7 = plt.subplot2grid((5, 10), (4, 0), colspan=4, rowspan=1)
ax7.axis('off')

for axx in [ax2, ax3, ax4, ax7]:
    axx.axes.get_xaxis().set_visible(False)
    axx.axes.get_yaxis().set_visible(False)

fig.subplots_adjust(hspace=0.5, wspace=0.6)

# Fancy picker ------------------------------------------
prev_ind   = np.array([0])
img_switch = True
im_sky     = None
im_fit     = None
im_diff    = None
sample_cnt = 0

evento = None

def onpick(event):
    global sample_cnt, prev_ind, img_switch, im_sky, im_fit, im_diff, evento

    if len(event.ind) > 0:
        #if prev_ind.all() == event.ind.all():
        #    sample_cnt += 1
        #    sample_cnt = sample_cnt % len(event.ind)
        #else:
        #    prev_ind = np.copy(event.ind)
        #    sample_cnt = 0
        
        sample_cnt = 0

        ax5.clear()
        ax6.clear()
        ax7.clear()
        ax7.axis('off')

        sample_id = dataframe['index'][event.ind[sample_cnt]]
        selected_id = event.ind[sample_cnt]
        im_sky  = np.copy(dataframe['Img data'][selected_id])
        im_fit  = np.copy(dataframe['Img fit' ][selected_id])
        im_diff = np.abs(im_sky-im_fit)

        #ax1.scatter(choice[flag][0][selected_id], choice[flag][1][selected_id], s=10, c='red')

        profile_0 = radial_profile(im_sky)[:32]
        profile_1 = radial_profile(im_fit)[:32]
        profile_diff = np.abs(profile_1-profile_0)/im_sky.max()*100 #[%]

        ax5.set_title('TipToy fitting')
        l2 = ax5.plot(profile_0, label='Data')
        l1 = ax5.plot(profile_1, label='TipToy')
        ax5.set_xlabel('Pixels')
        ax5.set_yscale('log')
        ax5.set_xlim([0, len(profile_1)])
        ax5.grid()

        l3 = ax6.plot(profile_diff, label='Difference', color='green')
        ax6.set_ylim([0, profile_diff.max()*1.5])
        ax6.set_ylabel('Intensity [ADU]  /  Difference [%]')

        ls = l1+l2+l3
        labs = [l.get_label() for l in ls]
        ax6.legend(ls, labs, loc=0)

        minval = np.nanmin([im_sky.min(), im_fit.min()]) * 1e-2
        im_sky -= minval
        im_fit -= minval

        vmax = np.max([np.nanmax(im_sky), np.nanmax(im_fit)])
        vmin = np.max([np.nanmin(im_sky), np.nanmin(im_fit)]) 
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        crop = slice(im_sky.shape[0]//2-64, im_sky.shape[0]//2+64)
        crop = (crop,crop)

        ax2.set_title('Data')
        ax2.imshow(im_sky[crop],  norm=norm)
        ax3.set_title('Fit')
        ax3.imshow(im_fit[crop],  norm=norm)
        ax4.set_title('Difference')
        ax4.imshow(im_diff[crop], norm=norm)

        Jx_tmp      = str(np.round(dataframe['Jx'][selected_id],1))
        Jy_tmp      = str(np.round(dataframe['Jy'][selected_id],1))
        Jxy_tmp     = str(np.round(dataframe['Jxy'][selected_id],1))
        noise_tmp   = str(np.round(np.abs(dataframe['WFS noise'][selected_id]),2))
        r0_buf_fit  = str(np.round(dataframe['$r_0$ fitted @ 500 [nm]'][selected_id],2))
        r0_buf_data = str(np.round(dataframe['$r_0$ SPARTA @ 500 [nm]'][selected_id],2))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = 'Jx, Jy, Jxy: '+Jx_tmp+', '+Jy_tmp+', '+Jxy_tmp+'\nWFS noise: '+noise_tmp+ \
            '\nr$_{0 fit}$, r$_{0 data}$:'+r0_buf_fit+', '+r0_buf_data
        ax7.text(0.0, 0.7, textstr, transform=ax7.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        ax1.set_title(str(sample_id))
        print("Picked: "+str(sample_id))
        
        plt.pause(0.01)
        evento = event

ax1.figure.canvas.mpl_connect("pick_event", onpick)

A = evento

print(A)


#%%
'''
def on_press(event):
    global img_switch
    global im_sky
    global im_fit
    
    #print('press', event.key)
    sys.stdout.flush()
    if event.key == 'up' or event.key == 'down':
        if im_sky is not None and im_fit is not None:
            vmax = np.max([np.max(im_sky), np.max(im_fit)])
            vmin = np.max([np.min(im_sky), np.min(im_fit)]) 
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            if img_switch:
                ax2.imshow(im_sky, norm=norm)
                ax3.imshow(im_fit, norm=norm)
            else:
                ax3.imshow(im_sky, norm=norm)
                ax2.imshow(im_fit, norm=norm)
            img_switch = not img_switch
    plt.pause(0.1)

fig.canvas.mpl_connect('key_press_event', on_press)
ax1.figure.canvas.mpl_connect("pick_event", onpick)

fig.canvas.mpl_connect('key_press_event', on_press)
'''
# %%



