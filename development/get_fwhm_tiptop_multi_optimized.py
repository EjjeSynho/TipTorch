import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

import matplotlib
# matplotlib.use('agg')  # OTTIMIZZAZIONE: usa backend non-interattivo per velocizzare
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
from mpdaf.obj import Cube
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

from maoppy.psfmodel import Psfao
from maoppy.psffit import psffit
from maoppy.instrument import muse_nfm

import os
import warnings
from scipy.ndimage import zoom
from multiprocessing import Pool

warnings.simplefilter('ignore')

def compute_strehl(data, wave, Npix=128):

    # Calcola la trasformata di Fourier bidimensionale della PSF normalizzata
    # image_oversampled = zoom(data, 10, order=3)
    # otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data)))

    # wvl = wave * 1e-9
    # samp = muse_nfm.samp(wvl)
    # Pmodel = Psfao((Npix, Npix), system=muse_nfm, samp=samp*10)

    # otf_difraction = Pmodel.otfDiffraction
    # strehl = np.real(np.sum(otf)/np.sum(otf_difraction))

    wvl = wave * 1e-9
    samp = muse_nfm.samp(wvl)

    r0 = 0.15 # Fried parameter [m]
    bck = 1e-5 # background [rad2 m2]
    amp = 5.0 # Moffat amplitude [rad2]
    alpha = 0.1 # Moffat alpha [1/m]
    ratio = 1.2 # alpha_x/alpha_y
    theta = 0 # rotation angle
    beta = 1.6 # Moffat beta power law, 1.6 good for MUSE NFM
    param = [r0, bck, amp, alpha, ratio, theta, beta]
    psfmodel = Psfao(data.shape, system=muse_nfm, samp=samp)

    out = psffit(data, psfmodel, param)

    strehl = psfmodel.strehlOTF(out.x)

    return strehl



def compute_fwhm(image, oversampling=5):  # OTTIMIZZAZIONE: ridotto da 10 a 5 (4x più veloce)
    # Sovracampionamento dell'immagine
    image_oversampled = zoom(image, oversampling, order=1)  # OTTIMIZZAZIONE: order=1 invece di 3

    # Trova il valore massimo nell'immagine sovracampionata
    max_value = np.nanmax(image_oversampled)

    # Trova le coordinate del valore massimo
    max_coords = np.unravel_index(np.nanargmax(image_oversampled), image_oversampled.shape)

    # Crea un profilo radiale
    y, x = np.indices(image_oversampled.shape)
    r = np.sqrt((x - max_coords[1])**2 + (y - max_coords[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), image_oversampled.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    # Trova il valore a metà massimo
    half_max = max_value / 2.0

    # Trova il raggio dove il profilo radiale incrocia il valore a metà massimo
    fwhm = 2 * np.interp(half_max, radialprofile[::-1], np.arange(len(radialprofile))[::-1])

    return fwhm / oversampling * 25  # Converti da pixel sovracampionati a pixel originali e poi a mas

def normalize_psf(data):

    return data/np.nansum(data)

def get_image(name, wave=7000, dw=10, size=128):

    cube = Cube(name)
    image = cube.get_image((wave-dw, wave+dw))

    return image.data.data

def process_psf(params):

    airmass, seeing, ngs, tc, suffix, wave, type_strehl = params
    try:
        print(f'./PSF_files{suffix}/muse_z{airmass:0.2f}_s{seeing:0.2f}_n{ngs:0.2f}.fits')
        data, head = fits.getdata(f'./PSF_files{suffix}/muse_z{airmass:0.2f}_s{seeing:0.2f}_n{ngs:0.2f}.fits',
                                ext=1, header=True)
    except FileNotFoundError:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if data.shape[0] != 1:
        # OTTIMIZZAZIONE: ricerca diretta invece di iterare tutto l'header
        idx = None
        wave_str = str(wave)
        for key, value in head.items():
            if isinstance(value, str) and value.strip() == wave_str:
                idx = int(key[-1])
                break

        data = data[idx][0]
        strehl = head[f'SR00w0{idx}']
        ensqar = head[f'EE00W0{idx}']

    else:
        data = data[0]
        strehl = head['SR0000']
        ensqar = head[f'EE500000']

    data = data/np.sum(data)

    fwhm = compute_fwhm(data)

    if type_strehl == 'strehl':
        strehl = compute_strehl(data, 700)

    return airmass, seeing, fwhm, tc, strehl, ensqar

if __name__ == '__main__':

    airmasses = np.arange(1.0, 2.0, 0.1)
    seeings = np.arange(0.4, 1.5, 0.1)

    suffix= '_SPIE_fix'
    multi = False
    type_strehl = 'strehl'
    rerun = True
    show = False

    for wave in [900]:
        real = Table.read(f'./All_PSF_{wave}.fits')
        print(len(real))
        mask_date = Time(real['date']) >= Time('2022-04-01T18:00:00')
        real = real[mask_date]
        print(len(real))
        if multi:
            suffix_w = f'{suffix}'
        else:
            suffix_w = f'{suffix}_{wave}'

        airmass_out, seeing_out, fwhm_out, tc_out, strehl_out, ensq_out = [], [], [], [], [], []

        stars = Table.read(f'stars_30.fits')

        params = [(row['airmass'], row['seeing'], row['ngs'], row['tc'], suffix_w, wave, type_strehl) for row in stars]


        if rerun:
            with Pool(processes=8) as pool:
                results = pool.map(process_psf, params)
                print(results)

            for airmass, seeing, fwhm, tc, strehl, ensq in results:
                airmass_out.append(airmass)
                seeing_out.append(seeing)
                fwhm_out.append(fwhm)
                tc_out.append(tc)
                strehl_out.append(strehl)
                ensq_out.append(ensq)

            table = Table([airmass_out, seeing_out, fwhm_out, tc_out, strehl_out, ensq_out],
                          names=['airmass', 'seeing', 'fwhm', 'tc', 'strehl', 'ensquared'])
            table.add_column(stars['ngs'], name='ngs')
            table.write(f'fwhm_table_{wave}_{type_strehl}.fits', overwrite=True)
        else:
            table = Table.read(f'fwhm_table_{wave}_{type_strehl}.fits')





        tc_to_seeing = {
            10:0.6,
            20:0.7,
            30:0.8,
            50:1.0,
            100:1.4
        }


        quantiles = Table.read('./quantiles_table.fits')
        quantiles = quantiles[quantiles['wave']==wave]

        seeing_tc = np.array([tc_to_seeing[tc] for tc in quantiles['tc']])

        # OTTIMIZZAZIONE: Pre-calcolo delle maschere (riutilizzate in tutti i plot)
        print("Pre-calculating masks...")
        tc_list = list(tc_to_seeing.keys())

        # Maschere per 'real'
        real_masks = {tc: real['tc'] == tc for tc in tc_list}

        # Maschere per 'table'
        table_masks_tc = {tc: table['tc'] == tc for tc in tc_list}
        mask_ngs_low = table['ngs'] > 16
        mask_ngs_high = table['ngs'] < 9
        table_masks_main = {tc: table_masks_tc[tc] & (~mask_ngs_low) & (~mask_ngs_high) for tc in tc_list}
        table_masks_low = {tc: table_masks_tc[tc] & mask_ngs_low for tc in tc_list}
        table_masks_high = {tc: table_masks_tc[tc] & mask_ngs_high for tc in tc_list}

        # Maschere per 'quantiles'
        quantiles_masks = {tc: quantiles['tc'] == tc for tc in tc_list}

        # OTTIMIZZAZIONE: Crea norm e cmap una sola volta
        norm = plt.Normalize(vmin=0.4, vmax=1)
        cmap = plt.cm.viridis

        # ==================== PLOT FWHM ====================
        fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        ax = ax.flatten()

        # Remove the last subplot (subplot 6)
        fig.delaxes(ax[-1])
        ax = ax[:-1]

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i].scatter(
                real['airmass'][real_masks[tc]], real['fwhm'][real_masks[tc]],
                c='grey', label='Observations', s=50, alpha=0.4
            )

            scatter = ax[i].scatter(
                table['airmass'][table_masks_main[tc]], table['fwhm'][table_masks_main[tc]],
                c=cmap(norm(table['seeing'][table_masks_main[tc]])),
                label='Simulated Data', s=50, edgecolor='k', alpha=0.8
            )

            mask1 = real['airmass'][real_masks[tc]] <1.2
            mask2 = (real['airmass'][real_masks[tc]]>= 1.2) * (real['airmass'][real_masks[tc]] <1.5)
            mask3 = (real['airmass'][real_masks[tc]]>= 1.5) * (real['airmass'][real_masks[tc]] <2)

            for_boxplot = [
                real['fwhm'][real_masks[tc]][mask1],
                real['fwhm'][real_masks[tc]][mask2],
                real['fwhm'][real_masks[tc]][mask3],
            ]

            # Boxplot for quantiles con posizioni specifiche
            # Centri degli intervalli di airmass
            positions = [1.1, 1.35, 1.75]
            ax[i].boxplot(for_boxplot, positions=positions, widths=0.08, manage_ticks=False,
                          medianprops={'linewidth': 4}, boxprops={'linewidth': 1.5},
                          whis=(0, 100), whiskerprops={'color': 'grey'}, capprops={'color': 'grey'},
                          label='Median')

            medians = [np.nanmedian(np.array(item)) for item in for_boxplot]

            ax[i].errorbar(positions, medians, xerr=[0.1, 0.15, 0.25], ls='', c='grey',
                           capsize=8, lw=1)


            # Add turbulence category as text in the top-left corner of each subplot
            ax[i].text(
                0.95, 0.95, f'TC: {tc}%', transform=ax[i].transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )

        # Add a colorbar for the scatter plot
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.02, aspect=30)
        cbar.ax.set_position([0.92, 0.08, 0.02, 0.85])
        cbar.set_label('Seeing', fontsize=12, rotation=-90, labelpad=13)
        cbar.ax.tick_params(labelsize=10)
        cbar.mappable.set_clim(0.4, 1)

        # Set axis labels with larger font size
        ax[2].set_xlabel('Airmass', fontsize=14)
        ax[3].set_xlabel('Airmass', fontsize=14)
        ax[4].set_xlabel('Airmass', fontsize=14)
        ax[0].set_ylabel('FWHM (mas)', fontsize=14)
        ax[3].set_ylabel('FWHM (mas)', fontsize=14)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_ylim(30, 200)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.90, bottom=0.08, top=0.93)

        # Show the plot
        plt.suptitle(f'FWHM vs Airmass at {wave} nm', fontsize=16)
        plt.savefig(f'./plots/TT_vs_STD{suffix_w}_{wave}_{type_strehl}.png')
        if show:
            plt.show()
        else:
            plt.close()

        # ==================== PLOT STREHL ====================
        fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        ax = ax.flatten()

        # Remove the last subplot (subplot 6)
        fig.delaxes(ax[-1])
        ax = ax[:-1]

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i].scatter(
                real['airmass'][real_masks[tc]], real[type_strehl][real_masks[tc]] * 100,
                c='grey', label='Observations', s=50, alpha=0.4
            )

            scatter = ax[i].scatter(
                table['airmass'][table_masks_tc[tc]], table['strehl'][table_masks_tc[tc]]*100,
                c=cmap(norm(table['seeing'][table_masks_tc[tc]])),
                label='Simulated Data', s=50, edgecolor='k', alpha=0.8
            )

            mask1 = real['airmass'][real_masks[tc]] <1.2
            mask2 = (real['airmass'][real_masks[tc]]>= 1.2) * (real['airmass'][real_masks[tc]] <1.5)
            mask3 = (real['airmass'][real_masks[tc]]>= 1.5) * (real['airmass'][real_masks[tc]] <2)

            for_boxplot = [
                real[type_strehl][real_masks[tc]][mask1] * 100,
                real[type_strehl][real_masks[tc]][mask2] * 100,
                real[type_strehl][real_masks[tc]][mask3] * 100,
            ]

            # Boxplot for quantiles con posizioni specifiche
            positions = [1.1, 1.35, 1.75]
            ax[i].boxplot(for_boxplot, positions=positions, widths=0.08, manage_ticks=False,
                          medianprops={'linewidth': 4}, boxprops={'linewidth': 1.5},
                          whis=(0, 100), whiskerprops={'color': 'grey'}, capprops={'color': 'grey'},
                          label='Median')

            medians = [np.nanmedian(np.array(item)) for item in for_boxplot]

            ax[i].errorbar(positions, medians, xerr=[0.1, 0.15, 0.25], ls='', c='grey',
                           capsize=8, lw=1)

            # Add turbulence category as text in the top-left corner of each subplot
            ax[i].text(
                0.95, 0.95, f'TC: {tc}%', transform=ax[i].transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )

        # Add a colorbar for the scatter plot
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.02, aspect=30)
        cbar.ax.set_position([0.92, 0.08, 0.02, 0.85])
        cbar.set_label('Seeing', fontsize=12, rotation=-90, labelpad=13)
        cbar.ax.tick_params(labelsize=10)
        cbar.mappable.set_clim(0.4, 1)

        # Set axis labels with larger font size
        ax[2].set_xlabel('Airmass', fontsize=14)
        ax[3].set_xlabel('Airmass', fontsize=14)
        ax[4].set_xlabel('Airmass', fontsize=14)
        ax[0].set_ylabel('Strehl (%)', fontsize=14)
        ax[3].set_ylabel('Strehl (%)', fontsize=14)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_ylim(0, 26)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.90, bottom=0.08, top=0.93)

        # Show the plot
        plt.suptitle(f'Strehl vs Airmass at {wave} nm ({type_strehl})', fontsize=16)
        plt.savefig(f'./plots/TT_vs_STD_strehl{suffix_w}_{wave}_{type_strehl}.png')
        if show:
            plt.show()
        else:
            plt.close()

        # ==================== PLOT ENSQUARED ENERGY ====================
        fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        ax = ax.flatten()

        # Remove the last subplot (subplot 6)
        fig.delaxes(ax[-1])
        ax = ax[:-1]

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i].scatter(
                real['airmass'][real_masks[tc]], real['ensquared_tt'][real_masks[tc]],
                c='grey', label='Observations', s=50, alpha=0.4
            )

            scatter = ax[i].scatter(
                table['airmass'][table_masks_tc[tc]], table['ensquared'][table_masks_tc[tc]],
                c=cmap(norm(table['seeing'][table_masks_tc[tc]])),
                label='Simulated Data', s=50, edgecolor='k', alpha=0.8
            )

            mask1 = real['airmass'][real_masks[tc]] <1.2
            mask2 = (real['airmass'][real_masks[tc]]>= 1.2) * (real['airmass'][real_masks[tc]] <1.5)
            mask3 = (real['airmass'][real_masks[tc]]>= 1.5) * (real['airmass'][real_masks[tc]] <2)

            for_boxplot = [
                real['ensquared_tt'][real_masks[tc]][mask1],
                real['ensquared_tt'][real_masks[tc]][mask2],
                real['ensquared_tt'][real_masks[tc]][mask3],
            ]

            # Boxplot for quantiles con posizioni specifiche
            positions = [1.1, 1.35, 1.75]
            ax[i].boxplot(for_boxplot, positions=positions, widths=0.08, manage_ticks=False,
                          medianprops={'linewidth': 4}, boxprops={'linewidth': 1.5},
                          whis=(0, 100), whiskerprops={'color': 'grey'}, capprops={'color': 'grey'},
                          label='Median')

            medians = [np.nanmedian(np.array(item)) for item in for_boxplot]

            ax[i].errorbar(positions, medians, xerr=[0.1, 0.15, 0.25], ls='', c='grey',
                           capsize=8, lw=1)

            # Add turbulence category as text in the top-left corner of each subplot
            ax[i].text(
                0.95, 0.95, f'TC: {tc}%', transform=ax[i].transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )

        # Add a colorbar for the scatter plot
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.02, aspect=30)
        cbar.ax.set_position([0.92, 0.08, 0.02, 0.85])
        cbar.set_label('Seeing', fontsize=12, rotation=-90, labelpad=13)
        cbar.ax.tick_params(labelsize=10)
        cbar.mappable.set_clim(0.4, 1)

        # Set axis labels with larger font size
        ax[2].set_xlabel('Airmass', fontsize=14)
        ax[3].set_xlabel('Airmass', fontsize=14)
        ax[4].set_xlabel('Airmass', fontsize=14)
        ax[0].set_ylabel('Ensquared Energy', fontsize=14)
        ax[3].set_ylabel('Ensquared Energy', fontsize=14)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_ylim(0, 0.5)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.90, bottom=0.08, top=0.93)

        # Show the plot
        plt.suptitle(f'Ensquared Energy vs Airmass at {wave} nm', fontsize=16)
        plt.savefig(f'./plots/TT_vs_STD_ensquared{suffix_w}_{wave}.png')
        if show:
            plt.show()
        else:
            plt.close()

        # ==================== HISTOGRAMS FWHM ====================
        fig, ax = plt.subplots(1, 6, figsize=(25, 6), sharex=True, sharey=True)
        ax = ax.flatten()

        bins = np.arange(0, 200 + 5, 5)
        ax[0].hist(real['fwhm'], bins=bins, color='grey', alpha=0.5, label='Observations')
        ax[0].hist(table['fwhm'], bins=bins, color='blue', alpha=0.5, label='Simulated Data')

        # Add turbulence category as text in the top-left corner of each subplot
        ax[0].set_title(f'All', fontsize=15)

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i+1].hist(real['fwhm'][real_masks[tc]], bins=bins, color='grey', label='Observations', alpha=0.4)
            ax[i+1].hist(table['fwhm'][table_masks_tc[tc]], bins=bins, color='blue', label='Simulated Data', alpha=0.4)

            # Add turbulence category as text in the top-left corner of each subplot
            ax[i+1].set_title(f'TC: {tc}%', fontsize=15)

        # Set a single x label for all plots
        fig.text(0.5, 0.04, 'FWHM (mas)', ha='center', fontsize=16)

        # Set y-axis labels for the leftmost plots
        ax[0].set_ylabel('Frequency', fontsize=16)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_xlim(0, 149)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95)

        # Show the plot
        plt.suptitle(f'FWHM at {wave} nm', fontsize=16)
        plt.savefig(f'./plots/Hist_FWHM{suffix_w}_{wave}.png')
        if show:
            plt.show()
        else:
            plt.close()


        # ==================== HISTOGRAMS STREHL ====================
        fig, ax = plt.subplots(1, 6, figsize=(25, 6), sharex=True, sharey=True)
        ax = ax.flatten()

        bins = np.arange(0, 21, 1)
        ax[0].hist(real[type_strehl]*100, bins=bins, color='grey', alpha=0.5, label='Observations')
        ax[0].hist(table['strehl']*100, bins=bins, color='blue', alpha=0.5, label='Simulated Data')

        # Add turbulence category as text in the top-left corner of each subplot
        ax[0].set_title(f'All', fontsize=15)

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i+1].hist(real[type_strehl][real_masks[tc]]*100, bins=bins, color='grey', label='Observations', alpha=0.4)
            ax[i+1].hist(table['strehl'][table_masks_tc[tc]]*100, bins=bins, color='blue', label='Simulated Data', alpha=0.4)

            # Add turbulence category as text in the top-left corner of each subplot
            ax[i+1].set_title(f'TC: {tc}%', fontsize=15)

        # Set a single x label for all plots
        fig.text(0.5, 0.04, 'Strehl (%)', ha='center', fontsize=16)

        # Set y-axis labels for the leftmost plots
        ax[0].set_ylabel('Frequency', fontsize=16)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_xlim(0, 19.5)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95)

        # Show the plot
        plt.suptitle(f'Strehl at {wave} nm', fontsize=16)
        plt.savefig(f'./plots/Hist_Strehl{suffix_w}_{wave}.png')
        if show:
            plt.show()
        else:
            plt.close()


        # ==================== HISTOGRAMS ENSQUARED ENERGY ====================
        fig, ax = plt.subplots(1, 6, figsize=(25, 6), sharex=True, sharey=True)
        ax = ax.flatten()

        bins = np.arange(0, 1.01, 0.02)
        ax[0].hist(real['ensquared_tt'], bins=bins, color='grey', alpha=0.5, label='Observations')
        ax[0].hist(table['ensquared'], bins=bins, color='blue', alpha=0.5, label='Simulated Data')

        # Add turbulence category as text in the top-left corner of each subplot
        ax[0].set_title(f'All', fontsize=15)

        # Scatter plot for the main data
        for i, tc in enumerate(tc_list):
            print('Working on TC:', tc)

            ax[i+1].hist(real['ensquared_tt'][real_masks[tc]], bins=bins, color='grey', label='Observations', alpha=0.4)
            ax[i+1].hist(table['ensquared'][table_masks_tc[tc]], bins=bins, color='blue', label='Simulated Data', alpha=0.4)

            # Add turbulence category as text in the top-left corner of each subplot
            ax[i+1].set_title(f'TC: {tc}%', fontsize=15)

        # Set a single x label for all plots
        fig.text(0.5, 0.04, 'Ensquared Energy', ha='center', fontsize=16)

        # Set y-axis labels for the leftmost plots
        ax[0].set_ylabel('Frequency', fontsize=16)

        # Set axis ticks with larger font size
        for item in ax:
            item.tick_params(axis='both', which='major', labelsize=12)
            item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
            item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

        # Set y-axis limits
        ax[0].set_xlim(0, 1)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95)

        # Show the plot
        plt.suptitle(f'Ensquared Energy at {wave} nm', fontsize=16)
        plt.savefig(f'./plots/Hist_Ensquared{suffix_w}_{wave}.png')
        if show:
            plt.show()
        else:
            plt.close()

        print("All plots saved successfully!")
