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
path_fitted = 'C:\\Users\\akuznets\\Data\\SPHERE\\fitted\\'
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

# %%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

images_data = []
images_fit  = []

FWHM_fit    = np.zeros([len(data),2])
FWHM_data   = np.zeros([len(data),2])
SR_fit      = np.zeros(len(data))
SR_data     = np.zeros(len(data))
SR_SPARTA   = np.zeros(len(data))
r0_data     = np.zeros(len(data))
r0_fit      = np.zeros(len(data))
jitters     = np.zeros([len(data),3])
noise       = np.zeros(len(data))
indexes     = np.zeros(len(data))
wvls        = np.zeros(len(data))
seeing_data = np.zeros(len(data))

r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]

for id in range(len(data)):
    images_data.append(data[id][1]['Img. data'])
    images_fit.append(data[id][1]['Img. fit'])
    FWHM_fit [id][:] = np.array(data[id][1]['FWHM fit'])
    FWHM_data[id][:] = np.array(data[id][1]['FWHM data'])
    noise[id]        = data[id][1]['n']
    SR_fit[id]       = data[id][1]['SR fit']
    SR_data[id]      = data[id][1]['SR data']
    SR_SPARTA[id]    = data[id][0]['Strehl']
    r0_data[id]      = data[id][0]['r0']
    r0_fit[id]       = data[id][1]['r0']
    jitters[id,:]    = np.array([ data[id][1]['Jx'], data[id][1]['Jy'], data[id][1]['Jxy'] ])
    wvls[id]         = data[id][0]['spectrum']['lambda']
    seeing_data[id]  = data[id][0]['seeing']['SPARTA']
    indexes[id]      = data[id][2]

#r0_data1 = r0_new(r0_data, wvls, 500e-9)
r0_fit1  = r0_new(r0_fit, 500e-9, wvls)

dataframe = pd.DataFrame(data={
    'FWHM fit':  np.hypot(FWHM_fit[:,0], FWHM_fit[:,1]),
    'FWHM data': np.hypot(FWHM_data[:,0], FWHM_data[:,1]),
    'index': indexes,
    'SR fit': SR_fit,
    'SR data': SR_data,
    'SR SPARTA': SR_SPARTA,
    'Jx':  jitters[:,0],
    'Jy':  jitters[:,1],
    'Jxy': jitters[:,2],
    'WFS noise': noise,
    '$r_0$ SPARTA @ 500 [nm]':  r0_data,
    '$r_0$ fitted @ 500 [nm]': r0_fit1,
    'Seeing fitted [asec]': seeing(r0_fit1, 500e-9),
    'Seeing SPARTA [asec]': seeing_data,
    'Wavelength ($\mu m$)': np.round(wvls*1e6,2),
    'Img data': images_data,
    'Img fit': images_fit
})

#dataframe = dataframe[:100]

#plt.scatter(r0_data, r0_fit1)
#plt.xlim([-0.1,1.])
#plt.ylim([-0.1,1.])
#plt.grid()

#%% --------------------------------------------------------------
#fig = plt.figure(figsize=(5, 5), dpi=200)
#sns.scatterplot(x='Seeing SPARTA [asec]', y='Seeing fitted [asec]', hue='Wavelength ($\mu m$)', data=dataframe)
#s = pd.Series([0,1,2])
#s.plot.line(linewidth=1., color='gray', linestyle='--')
#plt.title(r"$r_0$ fitted vs SPARTA")
#plt.ylim([0, 0.6])
#plt.xlim([0, 0.6])
#plt.grid()


%matplotlib qt
#%matplotlib inline

current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='darkslateblue')

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot2grid((5, 10), (0, 0), colspan=4, rowspan=4)
plt.grid()


flag = 'r0'
if flag == 'SR':
    s = pd.Series([0,1,2])
    s.plot.line(linewidth=1., color='gray', linestyle='--')
    sns.scatterplot(x='SR fit', y='SR data', hue='Wavelength ($\mu m$)', data=dataframe, picker=4)
    ax1.set_ylim([0, 1.0])
    ax1.set_xlim([0, 1.0])

elif flag == 'SR SPARTA':
    s = pd.Series([0,1,2])
    s.plot.line(linewidth=1., color='gray', linestyle='--')
    sns.scatterplot(x='SR fit', y='SR SPARTA', hue='Wavelength ($\mu m$)', data=dataframe, picker=4)
    ax1.set_ylim([0, 1.0])
    ax1.set_xlim([0, 1.0])

elif flag == 'r0':
    sns.scatterplot(x='$r_0$ SPARTA @ 500 [nm]', y='$r_0$ fitted @ 500 [nm]', hue='Wavelength ($\mu m$)', data=dataframe, picker=4)
    #sns.scatterplot(x='Seeing SPARTA [asec]', y='Seeing fitted [asec]', hue='Wavelength ($\mu m$)', data=dataframe, picker=4)
    s = pd.Series([0,1,2])
    s.plot.line(linewidth=1., color='gray', linestyle='--')
    ax1.set_ylim([0, 0.7])
    ax1.set_xlim([0, 0.7])
    ax1.grid()

elif flag == 'FWHM':
    sns.scatterplot(x='FWHM fit', y='FWHM data', hue='Wavelength ($\mu m$)', data=dataframe)
    s = pd.Series([0,1,2,3,4,5,6])
    s.plot.line(linewidth=1., color='gray', linestyle='--')
    plt.title('FWHM TIPTOP vs. data')
    plt.ylim([2.5, 6.0])
    plt.xlim([2.5, 6.0])
    plt.grid()


ax2 = plt.subplot2grid((5, 10), (0, 4), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((5, 10), (0, 6), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((5, 10), (0, 8), colspan=2, rowspan=2)
ax5 = plt.subplot2grid((5, 10), (2, 4), colspan=6, rowspan=3)
ax6 = ax5.twinx()
ax7 = plt.subplot2grid((5, 10), (4, 0), colspan=4, rowspan=1)
ax7.axis('off')

ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax3.axes.get_xaxis().set_visible(False)
ax3.axes.get_yaxis().set_visible(False)
ax4.axes.get_xaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
ax7.axes.get_xaxis().set_visible(False)
ax7.axes.get_yaxis().set_visible(False)
fig.subplots_adjust(hspace=0.5, wspace=0.6)


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

# Fancy picker ------------------------------------------
prev_ind   = np.array([0])
img_switch = True
im_sky     = None
im_fit     = None
im_diff    = None
sample_cnt = 0


def onpick(event):
    global sample_cnt
    global prev_ind
    global img_switch
    global im_sky
    global im_fit
    global im_diff

    if len(event.ind) > 0:
        if prev_ind.all() == event.ind.all():
            sample_cnt += 1
            sample_cnt = sample_cnt % len(event.ind)
        else:
            prev_ind = np.copy(event.ind)
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

        #flag = True
        #if flag == 'SR':
        #    ax1.scatter(x=dataframe['SR fit'][selected_id], y=dataframe['SR data'][selected_id], s=10, c='red')
        #if flag == 'r0':
        #    ax1.scatter(x=dataframe['$r_0$ SPARTA @ 500 [nm]'][selected_id], y=dataframe['$r_0$ fitted @ 500 [nm]'][selected_id], s=10, c='red')
        #elif flag == 'FWHM':
        #    ax1.scatter(x=dataframe['FWHM fit'][selected_id], y=dataframe['FWHM data'][selected_id], s=10, c='red')
        #else:
        #    pass

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

ax1.figure.canvas.mpl_connect("pick_event", onpick)

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
