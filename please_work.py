#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button
import numpy as np
import pickle
from os import path
import os
import pandas as pd
from astropy.io import fits
import scipy

import sys
from utils import Photometry, seeing, r0_new, r0
from scipy.stats import gaussian_kde

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

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
images_data = []
images_fit  = []
dataframe = []

for id in range(len(data)):
    wvl = data[id][0]['spectrum']['lambda']
    images_data.append(data[id][1]['Img. data'])
    images_fit.append(data[id][1]['Img. fit'])

    dataframe_buf = {}
    dataframe_buf['FWHM fit']  = np.hypot(data[id][1]['FWHM fit'][0], data[id][1]['FWHM fit'][1])
    dataframe_buf['FWHM data'] = np.hypot(data[id][1]['FWHM data'][0], data[id][1]['FWHM data'][1])
    dataframe_buf['sample number']     = data[id][2]
    dataframe_buf['SR fit']    = data[id][1]['SR fit']
    dataframe_buf['SR data']   = data[id][1]['SR data']
    dataframe_buf['SR SPARTA'] = data[id][0]['Strehl']
    dataframe_buf['Jx']  = data[id][1]['Jx']
    dataframe_buf['Jy']  = data[id][1]['Jy']
    dataframe_buf['Jxy'] = data[id][1]['Jxy']
    dataframe_buf['WFS noise'] = data[id][1]['n']
    dataframe_buf['$r_0$ SPARTA @ 500 [nm]'] = data[id][0]['r0']
    dataframe_buf['$r_0$ fitted @ 500 [nm]'] = r0_new(data[id][1]['r0'], 500e-9, wvl)
    dataframe_buf['$r_0$ MASSDIMM @ 500 [nm]'] = r0(data[id][0]['seeing']['MASSDIMM'], 500e-9, )
    dataframe_buf['Seeing fitted [asec]'] = seeing(r0_new(data[id][1]['r0'], 500e-9, wvl), 500e-9)
    dataframe_buf['Seeing SPARTA [asec]'] = data[id][0]['seeing']['SPARTA']
    dataframe_buf['Seeing MASSDIMM [asec]'] = data[id][0]['seeing']['MASSDIMM']
    dataframe_buf['Wavelength ($\mu m$)'] = np.round(wvl*1e6,3)
    dataframe.append(dataframe_buf)

dataframe = pd.DataFrame(dataframe)
dataframe['Img data'] = images_data
dataframe['Img fit']  = images_fit
#dataframe.set_index('index')
#dataframe = dataframe[:100]

#%% --------------------------------------------------------------

def CleanNans(X,Y):
    xy = np.vstack([X, Y])
    xy = xy[:,np.isfinite(xy[0,:])]
    xy = xy[:,np.isfinite(xy[1,:])]
    return xy

class Gauss2DFitter():

    def __init__(self) -> None:
        pass

    def gaussian(self, height, center_x, center_y, width_x, width_y, rotation):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x,y):
            xp = x*np.cos(rotation) - y*np.sin(rotation)
            yp = x*np.sin(rotation) + y*np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2+
                    ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        height = data[0,:].max() 
        total = data[0,:].sum()
        X = data[1,:]
        Y = data[2,:]
        x = (X*data).sum()/total
        y = (Y*data).sum()/total

        width_x = np.sqrt( np.mean((X-x)**2) )
        width_y = np.sqrt( np.mean((Y-y)**2) )
        return height, x, y, width_x, width_y, 0.0

    def fit(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(data[1,:],data[2,:]) - data[0,:])
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p

fitter = Gauss2DFitter()

'''
X = dataframe['$r_0$ SPARTA @ 500 [nm]']
Y = dataframe['$r_0$ MASSDIMM @ 500 [nm]']
xy = CleanNans(X,Y)
density = gaussian_kde(xy)(xy)
data = np.vstack([density, xy[0,:], xy[1,:]])

#moments = fitter.moments(data)
params = fitter.fit(data)

xlim = np.median(data[1,:])*2
ylim = np.median(data[2,:])*2
x = np.linspace(0., xlim, 50)
y = np.linspace(0., ylim, 50)

xx, yy = np.meshgrid(x, y)

z = fitter.gaussian(*params)(xx,yy)
plt.scatter(X, Y, s=5)
plt.contour(xx, yy, z, levels=5, colors=['black'], linewidths=[0.5])
plt.xlim([0,xlim])
plt.ylim([0,ylim])
'''


%matplotlib qt
#%matplotlib inline

# Function to draw the radial profile of a PSF
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

# Fields that appear in dropdown menu
choice_list = list(dataframe.columns)
choice_list.remove('Img data')
choice_list.remove('Img fit')
choice_list.remove('sample number')

# Preapre the plot for correlation line
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='darkslateblue')
fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot2grid((5, 10), (0, 0), colspan=4, rowspan=4)
plt.grid()

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


class PlotScatter():
    def __init__(self) -> None:
        self.show_density = False
        self.axx = ax1
        self.xlim = [0.0, 1.0]
        self.ylim = [0.0, 1.0]
        self.choice = ['$r_0$ SPARTA @ 500 [nm]', '$r_0$ fitted @ 500 [nm]']

    def __set_lim(self):
        self.axx.set_ylim(self.ylim)
        self.axx.set_xlim(self.xlim)   

    def set_bb_lims(self):
        self.xlim = [0, dataframe[self.choice[0]].median()*2.25]
        self.ylim = [0, dataframe[self.choice[1]].median()*2.25]
        self.__set_lim()

    def plot(self):
        self.axx.clear()
        self.axx.plot([0,10], [0,10], linewidth=1., color='gray', linestyle='--', label = '1-to-1')
        self.__set_lim()
        self.axx.grid()

        if self.show_density:
            xy = CleanNans(dataframe[self.choice[0]], dataframe[self.choice[1]])
            try:
                density = gaussian_kde(xy)(xy)
                data = np.vstack([density, xy[0,:], xy[1,:]])
                params = fitter.fit(data)

                x = np.linspace(self.xlim[0], self.xlim[1], 50)
                y = np.linspace(self.ylim[0], self.ylim[1], 50)
                xx, yy = np.meshgrid(x, y)
                z = fitter.gaussian(*params)(xx,yy)
                self.axx.contour(xx, yy, z, levels=5, colors=['black'], linewidths=[0.5])

            except np.linalg.LinAlgError:
                print('It\'s the same parameter, come on')
                density = np.ones_like(xy[0,:])
            
            self.axx.scatter(x=xy[0,:], y=xy[1,:], c=density, picker=4)
            self.axx.set_xlabel(self.choice[0])
            self.axx.set_ylabel(self.choice[1])
        else:
            sns.scatterplot(x=self.choice[0], y=self.choice[1], hue='Wavelength ($\mu m$)', data=dataframe, picker=4, ax=self.axx)

        self.__set_lim()
        plt.draw()

# Initialize interactive elements
data_plot = PlotScatter()

def lim(xmin, xmax, ymin, ymax):
    try:
        data_plot.xlim = [float(xmin), float(xmax)]
        data_plot.ylim = [float(ymin), float(ymax)]
    except ValueError:
        print("Enter the number, please, I beg you")
    data_plot.plot()

def select(axisx, axisy):
    data_plot.choice = [axisx, axisy]
    data_plot.set_bb_lims()
    data_plot.plot()

interact(lim, xmin='0', xmax='1', ymin='0', ymax='1')
interact(select, axisx=choice_list, axisy=choice_list)

# Draw correlation line for the first time
data_plot.set_bb_lims()
data_plot.plot()


# Fancy picker ========================================================================
prev_ind   = np.array([0])
img_switch = True
im_sky     = None
im_fit     = None
im_diff    = None
plot_dens  = False
sample_cnt = 0

evento = None

def onpick(event):
    global sample_cnt, prev_ind, img_switch, im_sky, im_fit, im_diff, evento, plot_dens

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

        sample_id = dataframe['sample number'][event.ind[sample_cnt]]
        selected_id = event.ind[sample_cnt]
        im_sky  = np.copy(dataframe['Img data'][selected_id])
        im_fit  = np.copy(dataframe['Img fit' ][selected_id])
        im_diff = np.abs(im_sky-im_fit)

        # Plot a red dot for the selected sample
        data_plot.plot()
        ax1.scatter(
            dataframe[data_plot.choice[0]][selected_id],
            dataframe[data_plot.choice[1]][selected_id],
            s=20, c='red')

        # Calculate radial profiles of PSFs
        profile_0 = radial_profile(im_sky)[:64]
        profile_1 = radial_profile(im_fit)[:64]
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

        # Initialize legend for radial profiles plot
        ls = l1+l2+l3
        labs = [l.get_label() for l in ls]
        ax6.legend(ls, labs, loc=0)

        # Plot PSFs
        minval = np.nanmin([im_sky.min(), im_fit.min()]) * 1e-2
        im_sky -= minval
        im_fit -= minval

        vmax = np.abs(np.max([np.nanmax(im_sky), np.nanmax(im_fit)]))
        vmin = np.abs(np.max([np.nanmin(im_sky), np.nanmin(im_fit)]))

        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        crop = slice(im_sky.shape[0]//2-64, im_sky.shape[0]//2+64)
        crop = (crop,crop)

        ax2.set_title('Data')
        ax2.imshow(im_sky[crop],  norm=norm)
        ax3.set_title('Fit')
        ax3.imshow(im_fit[crop],  norm=norm)
        ax4.set_title('Difference')
        ax4.imshow(im_diff[crop], norm=norm)

        # Draw  parameters information window
        Jx_tmp      = str( np.round(dataframe['Jx'][selected_id],1) )
        Jy_tmp      = str( np.round(dataframe['Jy'][selected_id],1) )
        Jxy_tmp     = str( np.round(dataframe['Jxy'][selected_id],1) )
        noise_tmp   = str( np.round(np.abs(dataframe['WFS noise'][selected_id]),2) )
        r0_buf_fit  = str( np.round(dataframe['$r_0$ fitted @ 500 [nm]'][selected_id],2) )
        r0_buf_data = str( np.round(dataframe['$r_0$ SPARTA @ 500 [nm]'][selected_id],2) )

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = 'Jx, Jy, Jxy: '+Jx_tmp+', '+Jy_tmp+', '+Jxy_tmp+'\nWFS noise: '+noise_tmp+ \
            '\nr$_{0 fit}$, r$_{0 data}$:'+r0_buf_fit+', '+r0_buf_data
        ax7.text(0.0, 0.7, textstr, transform=ax7.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        ax1.set_title(str(sample_id))
        #print("Picked: " + str(sample_id))
        
        plt.pause(0.01)
        evento = event

ax1.figure.canvas.mpl_connect("pick_event", onpick)

# Display buttons for the regime selection =============================================
class Index:
    ind = 0
    global plot_dens

    def dens(self, event):
        data_plot.show_density = True
        data_plot.plot()
    def wvls(self, event):
        data_plot.show_density = False
        data_plot.plot()

callback = Index()
axdens = plt.axes([0.025, 0.90, 0.05, 0.05])
axwvls = plt.axes([0.025, 0.85, 0.05, 0.05])
bdens = Button(axdens, 'Density')
bdens.on_clicked(callback.dens)
bwvls = Button(axwvls, "$\lambda$")
bwvls.on_clicked(callback.wvls)
# %%
