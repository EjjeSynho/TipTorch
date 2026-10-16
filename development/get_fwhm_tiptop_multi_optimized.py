# Original code developed by Enrico Congiu (ESO). Adapted for the TipTorch repository: comments translated from Italian, paths
# made repository-relative, the PSF measurements moved to `tools/ETC_tools.py` / `ETC_compute_PSF_params.py` so that the TIPTOP
# and TipTorch PSFs are measured by the same code.
"""
PSF parameters of the TIPTOP Monte Carlo PSFs (FWHM, Strehl, encircled energy) and Enrico's comparison plots against the on-sky
MUSE NFM PSFs, per turbulence category (TC).

Inputs: `data/Enrico_ETC/generated_random_PSFs/stars.fits` and `{wave}nm/*.fits` (from `run_tiptop_random_carlo.py`), the on-sky
tables `data/Enrico_ETC/onsky_PSFs_params/All_PSF_{wave}.fits`. Outputs: `generated_random_PSFs/psf_params_{wave}.fits` (also
written by `ETC_compute_PSF_params.py --source tiptop`) and the figures in `data/Enrico_ETC/plots/`.

`--strehl-type` selects which on-sky Strehl column is compared with the simulated one: `strehl` (PSFAO fit, the simulated PSFs
are fitted the same way) or `strehl_tt` (TIPTOP-style peak ratio, taken from the TIPTOP header for the simulated PSFs).
Run from the repository root in the `TipTorch` conda env:

    python development/get_fwhm_tiptop_multi_optimized.py [--wave 900] [--strehl-type strehl] [--no-rerun] [--show]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ETC_compute_PSF_params import compute_params_table, SOURCES, WAVELENGTHS

REPO_ROOT    = Path(__file__).resolve().parents[1]
ETC_FOLDER   = REPO_ROOT / 'data' / 'Enrico_ETC'
ONSKY_TABLE  = ETC_FOLDER / 'onsky_PSFs_params' / 'All_PSF_{wave}.fits'
PLOTS_FOLDER = ETC_FOLDER / 'plots'
PSF_FOLDER   = SOURCES['tiptop']

TC_TO_SEEING = {
    10:  0.6,
    20:  0.7,
    30:  0.8,
    50:  1.0,
    100: 1.4
}


def airmass_boxplot(ax, airmass, values, positions=(1.1, 1.35, 1.75)):
    """ Box plots of the on-sky values in three airmass bins (<1.2, 1.2-1.5, 1.5-2) at specific positions """
    mask1 = airmass < 1.2
    mask2 = (airmass >= 1.2) * (airmass < 1.5)
    mask3 = (airmass >= 1.5) * (airmass < 2)
    for_boxplot = [values[mask1], values[mask2], values[mask3]]
    for_boxplot = [np.asarray(item)[np.isfinite(np.asarray(item, dtype=float))] for item in for_boxplot]

    # Box plot of the on-sky values with specific positions: the centers of the airmass intervals
    ax.boxplot(for_boxplot, positions=list(positions), widths=0.08, manage_ticks=False,
               medianprops={'linewidth': 4}, boxprops={'linewidth': 1.5},
               whis=(0, 100), whiskerprops={'color': 'grey'}, capprops={'color': 'grey'},
               label='Median')
    medians = [np.nanmedian(np.array(item)) if len(item) else np.nan for item in for_boxplot]
    ax.errorbar(list(positions), medians, xerr=[0.1, 0.15, 0.25], ls='', c='grey', capsize=8, lw=1)


def scatter_vs_airmass(real, table, real_col, table_col, real_masks, table_masks, tc_list, ylabel, ylim, title, path, scale=1.0, show=False):
    norm = plt.Normalize(vmin=0.4, vmax=1)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    ax = ax.flatten()

    # Remove the last subplot (subplot 6)
    fig.delaxes(ax[-1])
    ax = ax[:-1]

    for i, tc in enumerate(tc_list):
        ax[i].scatter(
            real['airmass'][real_masks[tc]], real[real_col][real_masks[tc]] * scale,
            c='grey', label='Observations', s=50, alpha=0.4
        )
        scatter = ax[i].scatter(
            table['airmass'][table_masks[tc]], table[table_col][table_masks[tc]] * scale,
            c=cmap(norm(table['seeing'][table_masks[tc]])),
            label='Simulated Data', s=50, edgecolor='k', alpha=0.8
        )
        airmass_boxplot(ax[i], np.asarray(real['airmass'][real_masks[tc]]), np.asarray(real[real_col][real_masks[tc]], dtype=float) * scale)

        # Turbulence category in the top-right corner of each subplot
        ax[i].text(
            0.95, 0.95, f'TC: {tc}%', transform=ax[i].transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
        )

    # Colorbar for the scatter plot
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.02, aspect=30)
    cbar.ax.set_position([0.92, 0.08, 0.02, 0.85])
    cbar.set_label('Seeing', fontsize=12, rotation=-90, labelpad=13)
    cbar.ax.tick_params(labelsize=10)
    cbar.mappable.set_clim(0.4, 1)

    for k in (2, 3, 4):
        ax[k].set_xlabel('Airmass', fontsize=14)
    for k in (0, 3):
        ax[k].set_ylabel(ylabel, fontsize=14)

    for item in ax:
        item.tick_params(axis='both', which='major', labelsize=12)
        item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

    ax[0].set_ylim(*ylim)
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.90, bottom=0.08, top=0.93)
    plt.suptitle(title, fontsize=16)
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


def histograms(real, table, real_col, table_col, real_masks, table_masks, tc_list, bins, xlabel, xlim, title, path, scale=1.0, show=False):
    fig, ax = plt.subplots(1, 6, figsize=(25, 6), sharex=True, sharey=True)
    ax = ax.flatten()

    ax[0].hist(np.asarray(real[real_col], dtype=float) * scale, bins=bins, color='grey', alpha=0.5, label='Observations')
    ax[0].hist(np.asarray(table[table_col], dtype=float) * scale, bins=bins, color='blue', alpha=0.5, label='Simulated Data')
    ax[0].set_title('All', fontsize=15)

    for i, tc in enumerate(tc_list):
        ax[i+1].hist(np.asarray(real[real_col][real_masks[tc]], dtype=float) * scale, bins=bins, color='grey', label='Observations', alpha=0.4)
        ax[i+1].hist(np.asarray(table[table_col][table_masks[tc]], dtype=float) * scale, bins=bins, color='blue', label='Simulated Data', alpha=0.4)
        ax[i+1].set_title(f'TC: {tc}%', fontsize=15)

    # Single x label for all plots
    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=16)
    ax[0].set_ylabel('Frequency', fontsize=16)

    for item in ax:
        item.tick_params(axis='both', which='major', labelsize=12)
        item.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        item.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')

    ax[0].set_xlim(*xlim)
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95)
    plt.suptitle(title, fontsize=16)
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wave', type=int, nargs='*', default=WAVELENGTHS, choices=WAVELENGTHS)
    parser.add_argument('--strehl-type', choices=['strehl', 'strehl_tt'], default='strehl')
    parser.add_argument('--since', default='2022-04-01T18:00:00', help="Keep on-sky PSFs observed after this date (Enrico's IRLOS-upgrade cut); '' for all")
    parser.add_argument('--no-rerun', action='store_true', help='Reuse an existing psf_params_{wave}.fits instead of measuring the PSFs')
    parser.add_argument('--processes', type=int, default=16)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if not args.show:
        matplotlib.use('agg')  # OPTIMIZATION: non-interactive backend to speed things up

    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    type_strehl = args.strehl_type
    suffix_w = ''
    stars = Table.read(PSF_FOLDER / 'stars.fits')

    for wave in args.wave:
        real = Table.read(str(ONSKY_TABLE).format(wave=wave))
        print(len(real))
        if args.since:
            mask_date = Time(real['date']) >= Time(args.since)
            real = real[mask_date]
        print(len(real))

        table_path = PSF_FOLDER / f'psf_params_{wave}.fits'
        if args.no_rerun and table_path.exists():
            table = Table.read(table_path)
        else:
            table = compute_params_table(PSF_FOLDER, stars, wave, processes=args.processes, do_psfao=True)
            table.write(table_path, overwrite=True)
        table['ensquared'] = table['ensquared_tt']
        sim_strehl = 'strehl' if type_strehl == 'strehl' else 'strehl_tt'

        # OPTIMIZATION: pre-compute the masks (reused in all plots)
        print("Pre-calculating masks...")
        tc_list = list(TC_TO_SEEING.keys())

        # Masks for 'real'
        real_masks = {tc: real['tc'] == tc for tc in tc_list}

        # Masks for 'table'
        table_masks_tc = {tc: table['tc'] == tc for tc in tc_list}
        mask_ngs_low = table['ngs'] > 16
        mask_ngs_high = table['ngs'] < 9
        table_masks_main = {tc: table_masks_tc[tc] & (~mask_ngs_low) & (~mask_ngs_high) for tc in tc_list}

        # ==================== PLOT FWHM ====================
        scatter_vs_airmass(real, table, 'fwhm', 'fwhm', real_masks, table_masks_main, tc_list, 'FWHM (mas)', (30, 200),
                           f'FWHM vs Airmass at {wave} nm', PLOTS_FOLDER / f'TT_vs_STD{suffix_w}_{wave}_{type_strehl}.png', show=args.show)

        # ==================== PLOT STREHL ====================
        scatter_vs_airmass(real, table, type_strehl, sim_strehl, real_masks, table_masks_tc, tc_list, 'Strehl (%)', (0, 26),
                           f'Strehl vs Airmass at {wave} nm ({type_strehl})', PLOTS_FOLDER / f'TT_vs_STD_strehl{suffix_w}_{wave}_{type_strehl}.png',
                           scale=100, show=args.show)

        # ==================== PLOT ENSQUARED ENERGY ====================
        scatter_vs_airmass(real, table, 'ensquared_tt', 'ensquared', real_masks, table_masks_tc, tc_list, 'Ensquared Energy', (0, 0.5),
                           f'Ensquared Energy vs Airmass at {wave} nm', PLOTS_FOLDER / f'TT_vs_STD_ensquared{suffix_w}_{wave}.png', show=args.show)

        # ==================== HISTOGRAMS FWHM ====================
        histograms(real, table, 'fwhm', 'fwhm', real_masks, table_masks_tc, tc_list, np.arange(0, 200 + 5, 5), 'FWHM (mas)', (0, 149),
                   f'FWHM at {wave} nm', PLOTS_FOLDER / f'Hist_FWHM{suffix_w}_{wave}.png', show=args.show)

        # ==================== HISTOGRAMS STREHL ====================
        histograms(real, table, type_strehl, sim_strehl, real_masks, table_masks_tc, tc_list, np.arange(0, 21, 1), 'Strehl (%)', (0, 19.5),
                   f'Strehl at {wave} nm', PLOTS_FOLDER / f'Hist_Strehl{suffix_w}_{wave}.png', scale=100, show=args.show)

        # ==================== HISTOGRAMS ENSQUARED ENERGY ====================
        histograms(real, table, 'ensquared_tt', 'ensquared', real_masks, table_masks_tc, tc_list, np.arange(0, 1.01, 0.02), 'Ensquared Energy', (0, 1),
                   f'Ensquared Energy at {wave} nm', PLOTS_FOLDER / f'Hist_Ensquared{suffix_w}_{wave}.png', show=args.show)

        print("All plots saved successfully!")
