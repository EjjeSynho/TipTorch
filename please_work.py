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
    SR_fit[id]       = data[id][1]['SR fit']
    SR_data[id]      = data[id][1]['SR data']
    SR_SPARTA[id]    = data[id][0]['Strehl']
    r0_data[id]      = data[id][0]['r0']
    r0_fit[id]       = data[id][1]['r0']
    wvls[id]         = data[id][0]['spectrum']['lambda']
    seeing_data[id]  = data[id][0]['seeing']['SPARTA']
    indexes[id]      = data[id][2]

r0_data1 = r0_new(r0_data, wvls, 500e-9)
r0_fit1  = r0_new(r0_fit, 500e-9, wvls)

dataframe = pd.DataFrame(data={
    'FWHM fit':  np.hypot(FWHM_fit[:,0], FWHM_fit[:,1]),
    'FWHM data': np.hypot(FWHM_data[:,0], FWHM_data[:,1]),
    'index': indexes,
    'SR fit': SR_fit,
    'SR data': SR_data,
    'SR SPARTA': SR_SPARTA,
    '$r_0$ SPARTA @ 500 [nm]':  r0_data,
    '$r_0$ fitted @ 500 [nm]': r0_fit1,
    'Seeing fitted [asec]': seeing(r0_fit1, 500e-9),
    'Seeing SPARTA [asec]': seeing_data,
    'Wavelength ($\mu m$)': np.round(wvls*1e6,2),
    'Img data': images_data,
    'Img fit': images_fit
})


#%% --------------------------------------------------------------
#fig = plt.figure(figsize=(5, 5), dpi=200)
#sns.scatterplot(x='Seeing SPARTA [asec]', y='Seeing fitted [asec]', hue='Wavelength ($\mu m$)', data=dataframe)
#s = pd.Series([0,1,2])
#s.plot.line(linewidth=1., color='gray', linestyle='--')
#plt.title(r"$r_0$ fitted vs SPARTA")
#plt.ylim([0, 0.6])
#plt.xlim([0, 0.6])
#plt.grid()

#current_cmap = matplotlib.cm.get_cmap()
#current_cmap.set_bad(color='darkslateblue')

%matplotlib qt
#%matplotlib inline

fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=4, rowspan=4)
plt.grid()
'''
flag = 'SR'
#if flag == 'SR':
s = pd.Series([0,1,2])
s.plot.line(linewidth=1., color='gray', linestyle='--')
sns.scatterplot(x='SR fit', y='SR data', hue='Wavelength ($\mu m$)', data=dataframe)
#plt.title('Strehl ratio TIPTOP vs. data')
ax1.set_ylim([0, 1.0])
ax1.set_xlim([0, 1.0])
'''
#sns.scatterplot(x='SR fit', y='SR SPARTA', hue='Wavelength ($\mu m$)', data=dataframe)
#s = pd.Series([0,1,2])
#s.plot.line(linewidth=1., color='gray', linestyle='--')
#ax1.set_ylim([0, 1.0])
#ax1.set_xlim([0, 1.0])


#elif flag == 'r0':
sns.scatterplot(x='$r_0$ SPARTA @ 500 [nm]', y='$r_0$ fitted @ 500 [nm]', hue='Wavelength ($\mu m$)', data=dataframe)
#sns.scatterplot(x='Seeing SPARTA [asec]', y='Seeing fitted [asec]', hue='Wavelength ($\mu m$)', data=dataframe)
s = pd.Series([0,1,2])
s.plot.line(linewidth=1., color='gray', linestyle='--')
ax1.set_ylim([0, 0.7])
ax1.set_xlim([0, 0.7])


'''
elif flag == 'FWHM':
    sns.scatterplot(x='FWHM fit', y='FWHM data', hue='Wavelength ($\mu m$)', data=dataframe)
    s = pd.Series([0,1,2,3,4,5,6])
    s.plot.line(linewidth=1., color='gray', linestyle='--')
    plt.title('FWHM TIPTOP vs. data')
    plt.ylim([2.5, 6.0])
    plt.xlim([2.5, 6.0])
    plt.grid()

'''
ax2 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)

ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax3.axes.get_xaxis().set_visible(False)
ax3.axes.get_yaxis().set_visible(False)


# Fancy picker ------------------------------------------
#sample_cnt = 0
#prev_ind = np.array([0])
#img_switch = True
#im_sky = None
#im_fit = None

def onpick(event):
    if len(event.ind) > 0:
        ax1.set_title(str(event.ind[0]))
        print('Aaaaa')
        plt.pause(0.1)

ax1.figure.canvas.mpl_connect("pick_event", onpick)

'''    
    global sample_cnt
    global prev_ind
    global img_switch
    global im_sky
    global im_fit

    if len(event.ind) > 0:
        if prev_ind.all() == event.ind.all():
            sample_cnt += 1
            sample_cnt = sample_cnt % len(event.ind)
        else:
            prev_ind = np.copy(event.ind)
            sample_cnt = 0
        
        sample_id = dataframe['index'][event.ind[sample_cnt]]
        selected_id = event.ind[sample_cnt]
        im_sky = dataframe['Img data'][selected_id]
        im_fit = dataframe['Img fit' ][selected_id]

        flag = True
        if flag == 'SR':
            ax1.scatter(x=dataframe['SR fit'][selected_id], y=dataframe['SR data'][selected_id], s=10, c='red')
        if flag == 'r0':
            ax1.scatter(x=dataframe['$r_0$ SPARTA @ 500 [nm]'][selected_id], y=dataframe['$r_0$ fitted @ 500 [nm]'][selected_id], s=10, c='red')
        elif flag == 'FWHM':
            ax1.scatter(x=dataframe['FWHM fit'][selected_id], y=dataframe['FWHM data'][selected_id], s=10, c='red')

        vmax = np.max([np.max(im_sky), np.max(im_fit)])
        vmin = np.max([np.min(im_sky), np.min(im_fit)]) 
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        if img_switch:
            ax2.imshow(im_sky, norm=norm)
            ax3.imshow(im_fit, norm=norm)
        else:
            ax3.imshow(im_sky, norm=norm)
            ax2.imshow(im_fit, norm=norm)

        ax1.set_title(str(sample_id))
        print("Picked: "+str(sample_id))
        #plt.pause(0.1)
'''
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
