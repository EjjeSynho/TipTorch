"""
Comparison of the PSF parameter distributions, per turbulence category (TC) and wavelength, between the on-sky MUSE NFM PSFs,
the vanilla-TIPTOP Monte Carlo and the calibrated-TipTorch Monte Carlo.

Inputs (all under `data/Enrico_ETC/`):
- `onsky_PSFs_params/All_PSF_{wave}.fits`      on-sky parameters (Enrico); `tc` from the header seeing and tau0
- `generated_random_PSFs/psf_params_{wave}.fits` TIPTOP Monte Carlo   (`ETC_compute_PSF_params.py --source tiptop`)
- `tiptorch_random_PSFs/psf_params_{wave}.fits`  TipTorch Monte Carlo (`ETC_compute_PSF_params.py --source tiptorch`)

Parameters compared: `fwhm` [mas] (oversampled radial profile), `strehl` [%] (PSFAO fit), `strehl_tt` [%] (TIPTOP-style peak
ratio), `ensquared_tt` (TIPTOP-style encircled energy within 50 mas). Both simulators are measured by the same code.

Outputs in `data/Enrico_ETC/plots/`:
- `ETC_hist_{param}_{wave}nm.png`   density histograms (All + one panel per TC) of on-sky vs TIPTOP vs TipTorch, medians marked
- `ETC_summary_medians.png`         median and interquartile range of every parameter vs TC, one column per wavelength
- `ETC_distance_heatmap.png`        Wasserstein distance to the on-sky distribution (TIPTOP and TipTorch) per TC and wavelength
- `ETC_comparison_summary.csv/.md`  medians, IQRs, Wasserstein-1 distances and KS statistics per wavelength, TC and parameter

Options: `--validation-only` keeps only the on-sky PSFs belonging to the validation split of the calibrator (matched to the
dataset by observation time), `--since 2022-04-01T18:00:00` applies Enrico's IRLOS-upgrade date cut to the on-sky sample.
Run from the repository root in the `TipTorch` conda env:

    python development/ETC_compare_distributions.py [--wave 500 700 900] [--validation-only] [--since 2022-04-01T18:00:00]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from scipy.stats import wasserstein_distance, ks_2samp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

REPO_ROOT    = Path(__file__).resolve().parents[1]
ETC_FOLDER   = REPO_ROOT / 'data' / 'Enrico_ETC'
ONSKY_TABLE  = ETC_FOLDER / 'onsky_PSFs_params' / 'All_PSF_{wave}.fits'
ONSKY_IDS    = ETC_FOLDER / 'onsky_PSFs_params' / 'onsky_to_dataset_ids.csv'
TIPTOP_DIR   = ETC_FOLDER / 'generated_random_PSFs'
TIPTORCH_DIR = ETC_FOLDER / 'tiptorch_random_PSFs'
PLOTS_FOLDER = ETC_FOLDER / 'plots'

WAVELENGTHS = [500, 600, 700, 800, 900]
TC_LIST     = [10, 20, 30, 50, 100]

# parameter -> (label, scale, bins)
PARAMS = {
    'fwhm':         ('FWHM [mas]',                 1.0,   np.arange(0, 200 + 5, 5)),
    'strehl':       ('Strehl, PSFAO fit [%]',      100.0, np.arange(0, 70 + 2.5, 2.5)),
    'strehl_tt':    ('Strehl, peak ratio [%]',     100.0, np.arange(0, 70 + 2.5, 2.5)),
    'ensquared_tt': ('Encircled energy (r=50 mas)', 1.0,   np.arange(0, 1.0 + 0.025, 0.025)),
}

# fixed series colours (dataviz reference palette: neutral ink for the observations, categorical slots 1 and 2 for the simulators)
SERIES = {
    'On-sky':   {'color': '#6b6a66', 'fill': True},
    'TIPTOP':   {'color': '#2a78d6', 'fill': False},
    'TipTorch': {'color': '#eb6834', 'fill': False},
}


def onsky_to_dataset_ids(onsky_dates):
    """
    Dataset sample ids of the on-sky PSFs (matched on observation time within 2 s to `muse_df['time']`); -1 when not in the
    dataset. Cached in `onsky_to_dataset_ids.csv`, together with the validation-split membership.
    """
    if ONSKY_IDS.exists():
        cache = pd.read_csv(ONSKY_IDS)
        if len(cache) == len(onsky_dates) and (cache['date'].to_numpy() == np.asarray(onsky_dates).astype(str)).all():
            return cache['dataset_id'].to_numpy(), cache['in_validation'].to_numpy().astype(bool)

    import pickle
    import torch
    sys.path.insert(0, str(REPO_ROOT / 'machine_learning'))
    from tiptorch._config import project_settings, WEIGHTS_FOLDER
    std_folder = Path(project_settings['MUSE_STD_data_folder'])
    with open(std_folder / 'muse_df.pickle', 'rb') as f:
        muse_df = pickle.load(f)
    data = torch.load(std_folder / 'dataset_cache' / 'muse_STD_stars_dataset.pt', weights_only=False, mmap=True)
    sample_ids = np.asarray(data['sample_ids'])
    with open(REPO_ROOT / 'machine_learning' / 'NFM_calibrator_config.json') as f:
        cfg = json.load(f)
    with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
        split = np.load(json.load(f)['split_path'])
    val_ids = set(sample_ids[split['val_idx']].tolist())

    df_times = pd.to_datetime(muse_df['time'], utc=True).sort_values()
    dates = pd.to_datetime(np.asarray(onsky_dates).astype(str), utc=True)
    pos = np.searchsorted(df_times.values, dates.values)
    dataset_id = np.full(len(dates), -1)
    for i, ts in enumerate(dates):
        best = [(abs((df_times.iloc[j] - ts).total_seconds()), df_times.index[j]) for j in (pos[i] - 1, pos[i]) if 0 <= j < len(df_times)]
        best = [b for b in best if b[0] <= 2.0]
        if best and min(best)[1] in sample_ids:
            dataset_id[i] = min(best)[1]
    in_validation = np.array([i in val_ids for i in dataset_id])
    pd.DataFrame({'date': np.asarray(onsky_dates).astype(str), 'dataset_id': dataset_id, 'in_validation': in_validation}).to_csv(ONSKY_IDS, index=False)
    print(f'{(dataset_id >= 0).sum()}/{len(dates)} on-sky PSFs matched to the dataset, {in_validation.sum()} in the validation split -> {ONSKY_IDS}')
    return dataset_id, in_validation


def load_onsky(wave, validation_only=False, since=''):
    t = Table.read(str(ONSKY_TABLE).format(wave=wave))
    keep = np.ones(len(t), dtype=bool)
    if since:
        keep &= np.asarray(Time(t['date']) >= Time(since))
    if validation_only:
        _, in_val = onsky_to_dataset_ids(np.asarray(t['date']))
        keep &= in_val
    t = t[keep]
    df = pd.DataFrame({p: np.asarray(t[p], dtype=float) for p in PARAMS if p in t.colnames})
    df['tc'] = np.asarray(t['tc'], dtype=int)
    return df


def load_simulated(folder, wave):
    path = Path(folder) / f'psf_params_{wave}.fits'
    if not path.exists():
        return None
    t = Table.read(path)
    df = pd.DataFrame({p: np.asarray(t[p], dtype=float) for p in PARAMS})
    df['tc'] = np.asarray(t['tc'], dtype=int)
    return df


def clean(values, bins, scale):
    v = np.asarray(values, dtype=float) * scale
    return v[np.isfinite(v) & (v >= bins[0]) & (v <= bins[-1])]


def plot_histograms(groups, param, wave, out_path):
    label, scale, bins = PARAMS[param]
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.6), sharex=True, sharey=True)
    for ax, tc in zip(axes, ['All'] + TC_LIST):
        for name, df in groups.items():
            if df is None:
                continue
            sel = df if tc == 'All' else df[df['tc'] == tc]
            v = clean(sel[param], bins, scale)
            if len(v) == 0:
                continue
            style = SERIES[name]
            if style['fill']:
                ax.hist(v, bins=bins, density=True, color=style['color'], alpha=0.35, label=f'{name} (N={len(v)})')
            else:
                ax.hist(v, bins=bins, density=True, histtype='step', color=style['color'], linewidth=2, label=f'{name} (N={len(v)})')
            ax.axvline(np.median(v), color=style['color'], linestyle='--', linewidth=1.5)
        ax.set_title('All TCs' if tc == 'All' else f'TC {tc}%', fontsize=13)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8.5, frameon=False, loc='upper right')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel('Density', fontsize=12)
    fig.supxlabel(label, fontsize=12)
    fig.suptitle(f'{label} at {wave} nm: on-sky vs TIPTOP vs calibrated TipTorch (dashed: medians)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summarize(groups, wave):
    """ Rows (wave, tc, param, source, N, median, q25, q75, W1 to on-sky, KS to on-sky) """
    rows = []
    onsky = groups['On-sky']
    for tc in ['All'] + TC_LIST:
        for param, (label, scale, bins) in PARAMS.items():
            ref = clean((onsky if tc == 'All' else onsky[onsky['tc'] == tc])[param], bins, scale)
            for name, df in groups.items():
                if df is None:
                    continue
                v = clean((df if tc == 'All' else df[df['tc'] == tc])[param], bins, scale)
                row = {'wave': wave, 'tc': tc, 'param': param, 'source': name, 'N': len(v),
                       'median': np.median(v) if len(v) else np.nan,
                       'q25': np.percentile(v, 25) if len(v) else np.nan,
                       'q75': np.percentile(v, 75) if len(v) else np.nan}
                if name != 'On-sky' and len(v) > 1 and len(ref) > 1:
                    row['W1_to_onsky'] = wasserstein_distance(ref, v)
                    row['KS_to_onsky'] = ks_2samp(ref, v).statistic
                else:
                    row['W1_to_onsky'] = np.nan
                    row['KS_to_onsky'] = np.nan
                rows.append(row)
    return rows


def plot_summary_medians(summary, waves, out_path):
    fig, axes = plt.subplots(len(PARAMS), len(waves), figsize=(4.2 * len(waves), 3.2 * len(PARAMS)), sharex=True, squeeze=False)
    x = np.arange(len(TC_LIST))
    offsets = {'On-sky': -0.22, 'TIPTOP': 0.0, 'TipTorch': 0.22}
    for i, (param, (label, _, _)) in enumerate(PARAMS.items()):
        for j, wave in enumerate(waves):
            ax = axes[i, j]
            for name, style in SERIES.items():
                s = summary[(summary['wave'] == wave) & (summary['param'] == param) & (summary['source'] == name) & (summary['tc'] != 'All')]
                s = s.set_index('tc').reindex(TC_LIST)
                if s['median'].isna().all():
                    continue
                err = np.vstack([s['median'] - s['q25'], s['q75'] - s['median']])
                ax.errorbar(x + offsets[name], s['median'], yerr=err, fmt='o', color=style['color'], markersize=5, capsize=3, linewidth=1.5, label=name)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{tc}%' for tc in TC_LIST])
            ax.grid(alpha=0.25, linewidth=0.5)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            if i == 0:
                ax.set_title(f'{wave} nm', fontsize=13)
            if j == 0:
                ax.set_ylabel(label, fontsize=11)
            if i == len(PARAMS) - 1:
                ax.set_xlabel('Turbulence category', fontsize=11)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle('Median and IQR per TC: on-sky vs TIPTOP vs calibrated TipTorch', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_distance_heatmap(summary, waves, out_path):
    """ Wasserstein-1 distance to the on-sky distribution, normalized by the on-sky IQR of the parameter at that TC and wavelength """
    onsky = summary[summary['source'] == 'On-sky'].set_index(['wave', 'tc', 'param'])
    sims  = [n for n in ('TIPTOP', 'TipTorch') if (summary['source'] == n).any()]
    fig, axes = plt.subplots(len(PARAMS), len(sims), figsize=(1.2 + 1.1 * len(waves) * len(sims), 3.2 * len(PARAMS)), squeeze=False,
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.45})
    tcs = TC_LIST + ['All']
    for i, (param, (label, _, _)) in enumerate(PARAMS.items()):
        grids, vmax = [], 0.0
        for name in sims:
            grid = np.full((len(tcs), len(waves)), np.nan)
            for r, tc in enumerate(tcs):
                for c, wave in enumerate(waves):
                    s = summary[(summary['wave'] == wave) & (summary['tc'] == tc) & (summary['param'] == param) & (summary['source'] == name)]
                    if len(s) and (wave, tc, param) in onsky.index:
                        iqr = onsky.loc[(wave, tc, param), 'q75'] - onsky.loc[(wave, tc, param), 'q25']
                        grid[r, c] = s['W1_to_onsky'].iloc[0] / iqr if iqr > 0 else np.nan
            grids.append(grid)
            vmax = max(vmax, np.nanmax(grid) if np.isfinite(grid).any() else 0.0)
        for j, (name, grid) in enumerate(zip(sims, grids)):
            ax = axes[i, j]
            im = ax.imshow(grid, cmap='Blues', vmin=0, vmax=max(vmax, 1e-6), aspect='auto')
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if np.isfinite(grid[r, c]):
                        ax.text(c, r, f'{grid[r, c]:.2f}', ha='center', va='center', fontsize=9, color='white' if grid[r, c] > 0.6 * vmax else '#1a1a19')
            ax.set_xticks(range(len(waves)))
            ax.set_xticklabels([f'{w} nm' for w in waves], fontsize=9)
            ax.set_yticks(range(len(tcs)))
            ax.set_yticklabels([f'TC {tc}%' if tc != 'All' else 'All' for tc in tcs], fontsize=9)
            ax.set_title(f'{name}: {label}', fontsize=11)
        fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02, pad=0.02, label='W1 / on-sky IQR')
    fig.suptitle('Wasserstein distance of the simulated distributions to the on-sky ones (lower is better)', fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def write_markdown(summary, waves, out_path, onsky_note):
    lines = [f'# ETC comparison summary', '', onsky_note, '',
             'Wasserstein-1 distance (W1) of each simulator to the on-sky distribution and medians, per wavelength and TC.',
             'Units: FWHM in mas, Strehl in %, EE within r = 50 mas as a fraction. `closer` names the simulator with the smaller W1.', '']
    for wave in waves:
        lines += [f'## {wave} nm', '', '| TC | parameter | N on-sky / TIPTOP / TipTorch | median on-sky | median TIPTOP | median TipTorch | W1 TIPTOP | W1 TipTorch | closer |', '|---|---|---|---|---|---|---|---|---|']
        for tc in ['All'] + TC_LIST:
            for param in PARAMS:
                s = summary[(summary['wave'] == wave) & (summary['tc'] == tc) & (summary['param'] == param)].set_index('source')
                if 'On-sky' not in s.index:
                    continue
                g = lambda src, col: s.loc[src, col] if src in s.index else np.nan
                w_tt, w_tx = g('TIPTOP', 'W1_to_onsky'), g('TipTorch', 'W1_to_onsky')
                closer = 'TipTorch' if np.nan_to_num(w_tx, nan=np.inf) < np.nan_to_num(w_tt, nan=np.inf) else 'TIPTOP' if np.isfinite(w_tt) else '-'
                lines.append(f"| {tc if tc == 'All' else str(tc) + '%'} | {param} | {int(g('On-sky', 'N'))} / {int(np.nan_to_num(g('TIPTOP', 'N')))} / {int(np.nan_to_num(g('TipTorch', 'N')))} "
                             f"| {g('On-sky', 'median'):.3g} | {g('TIPTOP', 'median'):.3g} | {g('TipTorch', 'median'):.3g} | {w_tt:.3g} | {w_tx:.3g} | {closer} |")
        lines.append('')
    # overall score: how often each simulator is closer, per parameter
    lines += ['## Overall', '', '| parameter | TIPTOP closer (TC x wavelength cells) | TipTorch closer | mean W1 TIPTOP | mean W1 TipTorch |', '|---|---|---|---|---|']
    for param in PARAMS:
        s = summary[(summary['param'] == param) & (summary['tc'] != 'All')]
        tt = s[s['source'] == 'TIPTOP'].set_index(['wave', 'tc'])['W1_to_onsky']
        tx = s[s['source'] == 'TipTorch'].set_index(['wave', 'tc'])['W1_to_onsky']
        both = pd.concat([tt.rename('tt'), tx.rename('tx')], axis=1).dropna()
        lines.append(f"| {param} | {(both['tt'] < both['tx']).sum()} | {(both['tx'] < both['tt']).sum()} | {both['tt'].mean():.3g} | {both['tx'].mean():.3g} |")
    out_path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wave', type=int, nargs='*', default=WAVELENGTHS, choices=WAVELENGTHS)
    parser.add_argument('--validation-only', action='store_true', help='Only on-sky PSFs from the validation split of the calibrator dataset')
    parser.add_argument('--since', default='', help="Keep on-sky PSFs observed after this date (e.g. 2022-04-01T18:00:00, Enrico's IRLOS-upgrade cut)")
    parser.add_argument('--tiptop-dir', type=Path, default=TIPTOP_DIR)
    parser.add_argument('--tiptorch-dir', type=Path, default=TIPTORCH_DIR)
    parser.add_argument('--out-dir', type=Path, default=PLOTS_FOLDER)
    parser.add_argument('--tag', default='', help='Suffix for the output file names (e.g. "_validation")')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ('_validation' if args.validation_only else '')
    rows = []
    for wave in args.wave:
        groups = {
            'On-sky':   load_onsky(wave, args.validation_only, args.since),
            'TIPTOP':   load_simulated(args.tiptop_dir, wave),
            'TipTorch': load_simulated(args.tiptorch_dir, wave),
        }
        print(f'{wave} nm: ' + ', '.join(f'{k} N={len(v) if v is not None else 0}' for k, v in groups.items()))
        for param in PARAMS:
            plot_histograms(groups, param, wave, args.out_dir / f'ETC_hist_{param}_{wave}nm{tag}.png')
        rows += summarize(groups, wave)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / f'ETC_comparison_summary{tag}.csv', index=False)
    plot_summary_medians(summary, args.wave, args.out_dir / f'ETC_summary_medians{tag}.png')
    plot_distance_heatmap(summary, args.wave, args.out_dir / f'ETC_distance_heatmap{tag}.png')
    note = f"On-sky sample: {'validation split only' if args.validation_only else 'all PSFs'}" + (f', observed since {args.since}' if args.since else '') + '.'
    write_markdown(summary, args.wave, args.out_dir / f'ETC_comparison_summary{tag}.md', note)
    print((args.out_dir / f'ETC_comparison_summary{tag}.md').read_text().split('## Overall')[-1])
    print(f'Figures and tables -> {args.out_dir}')
