"""
Shared code of the ETC-regime validation (calibrators/validate_ETC_calibrator.py, analysis/ETC_feature_analysis.py): the validation split replayed in the
ETC regime, PSF measurements with the ETC metrics of tools/ETC_tools.py (FWHM, PSFAO Strehl, peak-ratio Strehl, encircled
energy within 50 mas), the on-sky / TIPTOP / predictor tables, the distribution and per-sample accuracy metrics and the figures.

Tables: one row per (sample, wavelength[, draw]) with the parameter columns PARAM_COLUMNS, the ETC conditions and `source`;
cached in data/Enrico_ETC/ETC_calibrator/params_<source>.csv.
"""
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from astropy.table import Table
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr, wilcoxon
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tiptorch._config import WEIGHTS_FOLDER, default_device, project_settings
from calibrators.ETC_calibrator import ETCCalibrator, ETC_table_from_telemetry, ETC_INPUTS, PREDICTORS, TC_LIST
from tools import tiptop_env  # noqa: F401, the astro-tiptop checkouts
from tools.ETC_tools import compute_FWHM, compute_strehl_psfao, tiptop_metrics, load_pupil, crop_center, normalize_psf, read_psf_file, PIXEL_SCALE_MAS

ETC_FOLDER    = REPO_ROOT / 'data' / 'Enrico_ETC'
OUT_DIR       = ETC_FOLDER / 'ETC_calibrator'
PLOTS_DIR     = ETC_FOLDER / 'plots' / 'ETC_calibrator'
DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache' / 'muse_STD_stars_dataset.pt'
ETC_CONFIG    = REPO_ROOT / 'machine_learning' / 'calibrators' / 'configs' / 'ETC_calibrator_config.json' # the config the ETC bundle was trained with

NOMINAL_WAVES = [500, 600, 700, 800, 900] # [nm], the dataset bins nearest to them are used
PARAMS = { # parameter -> (label, scale, histogram bins)
    'fwhm':         ('FWHM [mas]',                  1.0,   np.arange(0, 200 + 5, 5)),
    'strehl':       ('Strehl, PSFAO fit [%]',       100.0, np.arange(0, 70 + 2.5, 2.5)),
    'strehl_tt':    ('Strehl, peak ratio [%]',      100.0, np.arange(0, 70 + 2.5, 2.5)),
    'ensquared_tt': ('Encircled energy (r=50 mas)', 1.0,   np.arange(0, 1.0 + 0.025, 0.025)),
}
PARAM_COLUMNS = ['fwhm', 'strehl', 'strehl_tt', 'fwhm_tt', 'ensquared_tt']
CONDITIONS    = ['sample_id', 'seeing', 'tau0', 'airmass', 'ngs_mag', 'tc', 'date']
SOURCES       = ['tiptop'] + PREDICTORS
# neutral ink for the observations, categorical palette slots for the simulators
COLORS = {'onsky': '#6b6a66', 'enrico': '#b8b6b0', 'tiptop': '#2a78d6', 'donor': '#eb6834', 'donor-nearest': '#eda100', 'direct-net': '#1baf7a'}
LABELS = {'onsky': 'on-sky', 'enrico': 'on-sky (Enrico)', 'tiptop': "TIPTOP (Enrico's ETC)", 'donor': 'TipTorch donor',
          'donor-nearest': 'TipTorch nearest donor', 'direct-net': 'TipTorch direct net'}


# ------------------------------------------------------------- setup -------------------------------------------------------------
def load_bundle(bundle=None):
    """ETC calibrator bundle, by default the one of the training config's metadata."""
    if bundle is None:
        with open(ETC_CONFIG) as f:
            cfg = json.load(f)
        with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
            bundle = json.load(f)['bundle_path']
    return ETCCalibrator(bundle, device=default_device)


def nearest_bins(wavelengths_nm, nominal=NOMINAL_WAVES):
    """Indices of the dataset wavelength bins nearest to the nominal wavelengths, and their wavelengths [nm]."""
    bins = [int(np.argmin(np.abs(np.asarray(wavelengths_nm) - w))) for w in nominal]
    return bins, [float(wavelengths_nm[b]) for b in bins]


def load_validation(etc_cal, dataset):
    """Validation split as an ETC table of replayed real conditions, with sample ids, dataset indices and dates."""
    val_idx = np.asarray(dataset.val_idx, int)
    feat    = [dataset.features.index(f) for f in ETC_INPUTS]
    scaled  = np.array([list(dataset.telemetry[i].values()) for i in val_idx], dtype=np.float64)[:, feat]
    etc     = ETC_table_from_telemetry(pd.DataFrame(scaled * etc_cal.scale + etc_cal.mean, columns=ETC_INPUTS))
    etc.insert(0, 'sample_id', np.asarray(dataset.sample_ids)[val_idx])
    etc.insert(1, 'dataset_idx', val_idx)
    dates = torch.load(DATASET_CACHE, weights_only=False, mmap=True)['acquisition_times'] # not kept by NFMDataset
    etc['date'] = pd.to_datetime(np.asarray(dates)[val_idx], utc=True, format='ISO8601').strftime('%Y-%m-%dT%H:%M:%S')
    return etc


# ----------------------------------------------------------- measuring -----------------------------------------------------------
def _measure_worker(args):
    """FWHM and PSFAO Strehl of one unit-sum PSF (numpy / scipy / maoppy only: runs in the worker processes)."""
    psf, wave_nm, do_psfao = args
    out = {'fwhm': compute_FWHM(psf, pixel_scale_mas=PIXEL_SCALE_MAS), 'strehl': np.nan}
    if do_psfao:
        try:
            out['strehl'] = compute_strehl_psfao(psf, wave_nm)[0]
        except Exception as e:
            print(f'PSFAO fit failed: {type(e).__name__}: {e}')
    return out


def measure_PSFs(PSFs, waves_nm, pupil, pool=None, do_psfao=True):
    """
    PSFs [N, N_wvl, H, W] (cropped to the 100 px field and normalized here) -> table with one row per (sample, wavelength):
    `sample`, `wave` (nominal), `wave_nm` and PARAM_COLUMNS (the TIPTOP-style metrics run in this process, P3 may use the GPU).
    """
    N, N_wvl = PSFs.shape[:2]
    cropped  = [[normalize_psf(crop_center(np.asarray(PSFs[i, w], dtype=float))) for w in range(N_wvl)] for i in range(N)]
    pairs    = [(i, w) for i in range(N) for w in range(N_wvl)]
    jobs     = [(cropped[i][w], waves_nm[w], do_psfao) for i, w in pairs]
    res      = pool.map(_measure_worker, jobs, chunksize=4) if pool is not None else [_measure_worker(j) for j in jobs]
    rows = []
    for (i, w), r in zip(pairs, res):
        sr, fwhm_tt, ee = tiptop_metrics(cropped[i][w], waves_nm[w], pupil)
        rows.append({'sample': i, 'wave': NOMINAL_WAVES[w], 'wave_nm': waves_nm[w], **r, 'strehl_tt': sr, 'fwhm_tt': fwhm_tt, 'ensquared_tt': ee})
    return pd.DataFrame(rows)


def attach_conditions(table, etc, source, draw=0):
    """Adds the ETC conditions of `etc.iloc[sample]`, the source name and the draw index."""
    out = table.copy()
    for c in CONDITIONS:
        out[c] = etc[c].to_numpy()[out['sample'].to_numpy()]
    return out.assign(source=source, draw=draw)


def clean(values, bins, scale):
    """Finite values in the histogram range, in display units."""
    v = np.asarray(values, dtype=float) * scale
    return v[np.isfinite(v) & (v >= bins[0]) & (v <= bins[-1])]


def select(df, wave, tc=None):
    """Rows of one wavelength and, unless None, one turbulence category."""
    return df[(df['wave'] == wave) & ((df['tc'] == tc) if tc is not None else True)]


# ------------------------------------------------------------ sources ------------------------------------------------------------
def compute_onsky(dataset, etc, bins, waves_nm, pupil, pool=None, do_psfao=True):
    """Metrics of the on-sky validation PSFs (+ Enrico's values of the matched observations as the source 'enrico')."""
    PSFs  = np.stack([np.asarray(dataset.PSF_cubes[i])[..., bins].transpose(2, 0, 1) for i in etc['dataset_idx']])
    onsky = attach_conditions(measure_PSFs(PSFs, waves_nm, pupil, pool, do_psfao), etc, 'onsky')
    onsky.to_csv(OUT_DIR / 'params_onsky.csv', index=False)

    ids  = pd.read_csv(ETC_FOLDER / 'onsky_PSFs_params' / 'onsky_to_dataset_ids.csv')
    rows = []
    for wave in NOMINAL_WAVES:
        t = Table.read(ETC_FOLDER / 'onsky_PSFs_params' / f'All_PSF_{wave}.fits').to_pandas()
        t['dataset_id'] = ids['dataset_id'].to_numpy()
        t = t[t['dataset_id'] >= 0].drop_duplicates('dataset_id').set_index('dataset_id')
        for i, sid in enumerate(etc['sample_id']):
            if sid in t.index:
                r = t.loc[sid]
                rows.append({'sample': i, 'wave': wave, 'wave_nm': float(wave), 'fwhm': r['fwhm'], 'strehl': r['strehl'],
                             'strehl_tt': r['strehl_tt'], 'fwhm_tt': np.nan, 'ensquared_tt': r['ensquared_tt']})
    enrico = attach_conditions(pd.DataFrame(rows), etc, 'enrico')
    enrico.to_csv(OUT_DIR / 'params_enrico.csv', index=False)
    print(f"on-sky: {len(etc)} validation PSFs measured, {enrico['sample'].nunique()} with Enrico's values")
    return onsky


def compute_tiptop(etc, waves_nm, pupil, pool=None, do_psfao=True, workers=4, overwrite=False):
    """TIPTOP (Enrico's tuned configuration) for the validation conditions, simulated at the dataset bin wavelengths."""
    import run_tiptop_random_carlo as rc
    stars = Table([np.arange(len(etc)), etc['seeing'].to_numpy(), etc['tau0'].to_numpy() * 1e3, etc['ngs_mag'].to_numpy(), etc['airmass'].to_numpy(),
                   etc['tc'].to_numpy(), etc['ngs_flux'].to_numpy()], names=['index', 'seeing', 'tau', 'ngs', 'airmass', 'tc', 'ngs_flux'])
    root = OUT_DIR / 'tiptop_validation'
    root.mkdir(parents=True, exist_ok=True)
    stars.write(root / 'stars.fits', overwrite=True)

    PSFs = np.full((len(etc), len(waves_nm), 100, 100), np.nan)
    for w, (nominal, wave_nm) in enumerate(zip(NOMINAL_WAVES, waves_nm)):
        wave_dir, config_dir = root / f'{nominal}nm', root / f'{nominal}nm' / 'configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        names = []
        for row in stars:
            config = rc.build_tiptop_config(rc.read_base_config(), row, nominal) # Enrico's tuning of the nominal wavelength
            config['sources_science']['Wavelength'] = str([wave_nm * 1e-9])        # simulated at the dataset bin wavelength
            names.append(rc.sample_name(row, row['index']))
            with open(config_dir / f'{names[-1]}.ini', 'w') as f:
                config.write(f)
        rc.run_simulations(names, config_dir, wave_dir, max_workers=workers, overwrite=overwrite)
        for i, name in enumerate(names):
            if (wave_dir / f'{name}.fits').exists():
                PSFs[i, w] = read_psf_file(wave_dir / f'{name}.fits')['psf']

    ok    = np.isfinite(PSFs).all(axis=(1, 2, 3))
    table = measure_PSFs(np.nan_to_num(PSFs), waves_nm, pupil, pool, do_psfao)
    table.loc[~ok[table['sample'].to_numpy()], PARAM_COLUMNS] = np.nan
    table = attach_conditions(table, etc, 'tiptop')
    table.to_csv(OUT_DIR / 'params_tiptop.csv', index=False)
    print(f'TIPTOP: {ok.sum()}/{len(etc)} validation conditions simulated at {len(waves_nm)} wavelengths')
    return table


def compute_predictor(name, etc_cal, etc, renderer, pupil, pool=None, do_psfao=True, n_draws=3, seed=0):
    """PSFs of one TipTorch predictor for the validation conditions (stochastic ones drawn n_draws times), measured."""
    PSFs, params = renderer.render(etc_cal.predictor(name), etc, np.random.default_rng(seed), n_draws=n_draws, verbose=True)
    draws = params['draw'].to_numpy()
    table = pd.concat([attach_conditions(measure_PSFs(PSFs[draws == d].numpy(), renderer.wavelengths_nm, pupil, pool, do_psfao), etc, name, int(d))
                       for d in np.unique(draws)], ignore_index=True)
    params.to_csv(OUT_DIR / f'predicted_params_{name}.csv', index=False)
    table.to_csv(OUT_DIR / f'params_{name}.csv', index=False)
    t = table[table.wave == 700]
    print(f'{name}: {len(np.unique(draws))} draw(s), median FWHM@700 {t.fwhm.median():.1f} mas, peak Strehl {100 * t.strehl_tt.median():.1f} %')
    return table


# ------------------------------------------------------------- metrics -------------------------------------------------------------
def distribution_summary(onsky, tables):
    """Rows (wave, tc, param, source): N, median, quartiles, W1 and KS to on-sky, paired MAE / bias / rho of the per-sample mean."""
    rows = []
    for wave in NOMINAL_WAVES:
        for tc in [None] + TC_LIST:
            for p, (label, scale, bins) in PARAMS.items():
                ref = clean(select(onsky, wave, tc)[p], bins, scale)
                for name, df in [('onsky', onsky)] + list(tables.items()):
                    v   = clean(select(df, wave, tc)[p], bins, scale)
                    row = {'wave': wave, 'tc': 'All' if tc is None else tc, 'param': p, 'source': name, 'N': len(v),
                           'median': np.median(v) if len(v) else np.nan, 'q25': np.percentile(v, 25) if len(v) else np.nan,
                           'q75': np.percentile(v, 75) if len(v) else np.nan, 'W1': np.nan, 'KS': np.nan, 'MAE': np.nan, 'bias': np.nan, 'rho': np.nan}
                    if name != 'onsky' and len(v) > 1 and len(ref) > 1:
                        row['W1'], row['KS'] = wasserstein_distance(ref, v), ks_2samp(ref, v).statistic
                        both = pd.concat([select(df, wave, tc).groupby('sample')[p].mean().rename('a'),
                                          select(onsky, wave, tc).set_index('sample')[p].rename('b')], axis=1).dropna() * scale
                        if len(both) > 2:
                            row['MAE'], row['bias'], row['rho'] = (both.a - both.b).abs().mean(), (both.a - both.b).mean(), spearmanr(both.a, both.b).statistic
                    rows.append(row)
    return pd.DataFrame(rows)


def accuracy_metrics(onsky, tables, reference='tiptop', n_boot=500, seed=0):
    """
    Per (wavelength, parameter) cell, all TCs pooled: W1 to on-sky / on-sky IQR of every source, KS p-value, relative error of the
    median, 5-95 % coverage of the on-sky values, and the paired bootstrap 90 % CI of W1(reference) - W1(source).
    """
    rng  = np.random.default_rng(seed)
    rows = []
    for wave in NOMINAL_WAVES:
        for p, (label, scale, _) in PARAMS.items():
            ref    = (onsky[onsky.wave == wave].set_index('sample')[p] * scale).dropna()
            iqr    = np.percentile(ref, 75) - np.percentile(ref, 25)
            pooled = {n: df[df.wave == wave].set_index('sample')[p] * scale for n, df in tables.items()}
            boots  = {n: [] for n in tables}
            for _ in range(n_boot):
                b = rng.choice(ref.index.to_numpy(), size=len(ref), replace=True)
                for n in tables:
                    boots[n].append(wasserstein_distance(ref.loc[b], pooled[n].loc[b].dropna()))
            for n, v in pooled.items():
                v = v.dropna()
                lo, hi = np.percentile(np.array(boots[reference]) - np.array(boots[n]), [5, 95]) / iqr if n != reference else (np.nan, np.nan)
                lo5, hi95 = np.percentile(v, [5, 95])
                rows.append({'wave': wave, 'param': p, 'source': n, 'W1_over_IQR': wasserstein_distance(ref, v) / iqr, 'KS_p': ks_2samp(ref, v).pvalue,
                             'median_rel_error': (np.median(v) - np.median(ref)) / np.median(ref), 'coverage_5_95': np.mean((ref >= lo5) & (ref <= hi95)),
                             f'dW1_vs_{reference}_lo': lo, f'dW1_vs_{reference}_hi': hi, f'better_than_{reference}': bool(lo > 0)})
    return pd.DataFrame(rows)


def accuracy_summary(acc, reference='tiptop'):
    """One row per source: mean W1 / IQR, cells indistinguishable from on-sky, |median error|, coverage, cells won, Wilcoxon vs the reference."""
    g   = acc.groupby('source')
    out = pd.DataFrame({'mean W1 / IQR': g['W1_over_IQR'].mean(), 'cells indistinguishable (KS p > 0.05)': g['KS_p'].apply(lambda s: int((s > 0.05).sum())),
                        'mean |median error| [%]': 100 * g['median_rel_error'].apply(lambda s: s.abs().mean()), 'mean 5-95 % coverage': g['coverage_5_95'].mean()})
    best = acc.loc[acc.groupby(['wave', 'param'])['W1_over_IQR'].idxmin(), 'source'].value_counts()
    out['cells closest to on-sky'] = best.reindex(out.index).fillna(0).astype(int)
    out[f'cells better than {reference} (90 %)'] = g[f'better_than_{reference}'].sum().astype(int)
    ref_w1 = acc[acc.source == reference].set_index(['wave', 'param'])['W1_over_IQR']
    for n in out.index:
        if n != reference:
            d = ref_w1 - acc[acc.source == n].set_index(['wave', 'param'])['W1_over_IQR']
            out.loc[n, f'Wilcoxon p (W1 {reference} > W1 source)'] = wilcoxon(d, alternative='greater').pvalue
    return out


def per_sample_metrics(onsky, tables):
    """Per (wavelength, parameter, source): paired MAE, bias, Spearman rho, relative MAE and fraction within 20 % of on-sky."""
    rows = []
    for wave in NOMINAL_WAVES:
        for p, (label, scale, _) in PARAMS.items():
            ref = onsky[onsky.wave == wave].set_index('sample')[p] * scale
            for name, df in tables.items():
                pred = df[df.wave == wave].groupby('sample')[p].mean() * scale # mean over draws for stochastic predictors
                both = pd.concat([ref.rename('o'), pred.rename('p')], axis=1).dropna()
                err  = both.p - both.o
                rows.append({'wave': wave, 'param': p, 'source': name, 'N': len(both), 'MAE': err.abs().mean(), 'bias': err.mean(),
                             'rho': spearmanr(both.o, both.p).statistic, 'rel_MAE': err.abs().mean() / both.o.median(), 'within_20pct': np.mean(err.abs() <= 0.2 * both.o)})
    return pd.DataFrame(rows)


def per_sample_summary(ps):
    out = ps.groupby('source')[['rel_MAE', 'rho', 'within_20pct']].mean()
    out['cells with lowest MAE'] = ps.loc[ps.groupby(['wave', 'param'])['MAE'].idxmin(), 'source'].value_counts().reindex(out.index).fillna(0).astype(int)
    return out


def write_report(etc, summary, acc_sum, ps_sum, path):
    """Markdown summary of the comparison (distribution accuracy, per-sample accuracy, all-TC medians per source and parameter)."""
    lines = ['# ETC validation: on-sky vs TIPTOP vs ETC predictors', '',
             f'Validation split: {etc["sample_id"].nunique()} observations (per TC: ' + ', '.join(f'{tc}% {int((etc.tc == tc).sum())}' for tc in TC_LIST) + '), '
             f'wavelength bins nearest to {NOMINAL_WAVES} nm, all TCs pooled per cell.', '',
             '## Distribution accuracy (20 wavelength x parameter cells)', '', acc_sum.round(3).to_markdown(), '',
             '## Per-sample accuracy (per-sample mean over draws for the sampler)', '', ps_sum.round(3).to_markdown(), '',
             '## All TCs, mean over wavelengths', '', '| source | param | median on-sky | median source | W1 | KS | MAE | bias | rho |', '|---|---|---|---|---|---|---|---|---|']
    all_tc = summary[summary.tc == 'All']
    for name in [n for n in summary.source.unique() if n != 'onsky']:
        for p in PARAMS:
            s, o = all_tc[(all_tc.source == name) & (all_tc.param == p)], all_tc[(all_tc.source == 'onsky') & (all_tc.param == p)]
            lines.append(f"| {name} | {p} | {o['median'].mean():.3g} | {s['median'].mean():.3g} | {s['W1'].mean():.3g} | {s['KS'].mean():.2f} | "
                         f"{s['MAE'].mean():.3g} | {s['bias'].mean():+.3g} | {s['rho'].mean():.2f} |")
    Path(path).write_text('\n'.join(lines) + '\n')
    return lines


# ------------------------------------------------------------- figures -------------------------------------------------------------
def _style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def _hist(ax, name, v, bins, lw=1.8, label=None):
    """On-sky as a filled histogram, simulators as step histograms, medians dashed."""
    if name == 'onsky':
        ax.hist(v, bins=bins, density=True, color=COLORS[name], alpha=0.3, label=label)
    else:
        ax.hist(v, bins=bins, density=True, histtype='step', color=COLORS[name], lw=lw, label=label)
    ax.axvline(np.median(v), color=COLORS[name], ls='--', lw=1.2)


def plot_summary_medians(summary, out_path=None):
    """Median and IQR per TC: rows = parameters, columns = wavelengths, one marker per source."""
    names = ['onsky'] + [n for n in summary.source.unique() if n != 'onsky']
    fig, axes = plt.subplots(len(PARAMS), len(NOMINAL_WAVES), figsize=(4.4 * len(NOMINAL_WAVES), 3.3 * len(PARAMS)), sharex=True, squeeze=False)
    x, width = np.arange(len(TC_LIST)), 0.8 / len(names)
    for i, (p, (label, _, _)) in enumerate(PARAMS.items()):
        for j, wave in enumerate(NOMINAL_WAVES):
            ax = axes[i, j]
            for k, name in enumerate(names):
                s = summary[(summary.wave == wave) & (summary.param == p) & (summary.source == name) & (summary.tc != 'All')].set_index('tc').reindex(TC_LIST)
                ax.errorbar(x + (k - len(names) / 2 + 0.5) * width, s['median'], yerr=np.vstack([s['median'] - s['q25'], s['q75'] - s['median']]),
                            fmt='o', ms=3.5, capsize=2, lw=1.2, color=COLORS.get(name, '#999'), label=LABELS.get(name, name))
            ax.set_xticks(x, [f'{tc}%' for tc in TC_LIST])
            ax.set(title=f'{wave} nm' if i == 0 else None, ylabel=label if j == 0 else None, xlabel='Turbulence category' if i == len(PARAMS) - 1 else None)
            _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle('Median and IQR per TC (validation split)', fontsize=14)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
    return fig


def plot_accuracy(acc, out_path=None, reference='tiptop'):
    """W1 / on-sky IQR per wavelength and parameter, one bar per source; * when the 90 % bootstrap CI beats the reference."""
    names = list(acc.source.unique())
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4.2 * len(PARAMS), 4.2), sharey=True)
    x, width = np.arange(len(NOMINAL_WAVES)), 0.8 / len(names)
    for ax, (p, (label, _, _)) in zip(axes, PARAMS.items()):
        for k, name in enumerate(names):
            a   = acc[(acc.param == p) & (acc.source == name)].set_index('wave').reindex(NOMINAL_WAVES)
            pos = x + (k - len(names) / 2 + 0.5) * width
            ax.bar(pos, a['W1_over_IQR'], width=width, color=COLORS[name], label=LABELS[name])
            for xi, yi, better in zip(pos, a['W1_over_IQR'], a[f'better_than_{reference}']):
                if better:
                    ax.text(xi, yi + 0.02, '*', ha='center', fontsize=12, color='#52514e')
        ax.set_xticks(x, [f'{w} nm' for w in NOMINAL_WAVES], fontsize=9)
        ax.set_title(label, fontsize=12)
        _style(ax)
    axes[0].set_ylabel('W1 / on-sky IQR')
    axes[0].legend(frameon=False, fontsize=9)
    means = acc.groupby('source')['W1_over_IQR'].mean()
    fig.suptitle(f'Distribution distance to the on-sky PSF parameters (lower is better; *: better than {LABELS[reference]} at 90 % bootstrap confidence)\n'
                 + 'mean over all cells: ' + ', '.join(f'{LABELS[n]} {means[n]:.2f}' for n in names), fontsize=11)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    return fig


def plot_per_sample(onsky, tables, ps, params=(('fwhm', (25, 160)), ('ensquared_tt', (0, 0.7))), wave=700, out_path=None):
    """Predicted vs on-sky value of every observation at one wavelength (left), relative MAE per wavelength (right)."""
    names = list(tables)
    fig, axes = plt.subplots(len(params), len(names) + 1, figsize=(4.0 * (len(names) + 1), 3.8 * len(params)), gridspec_kw={'width_ratios': [1] * len(names) + [1.35]})
    ref = onsky[onsky.wave == wave].set_index('sample')
    for i, (p, lim) in enumerate(params):
        label, scale, _ = PARAMS[p]
        for j, name in enumerate(names):
            ax   = axes[i, j]
            pred = tables[name][tables[name].wave == wave].groupby('sample')[p].mean() * scale
            both = pd.concat([(ref[p] * scale).rename('o'), pred.rename('p')], axis=1).dropna()
            ax.scatter(both.o, both.p, s=16, alpha=0.75, color=COLORS[name], edgecolor='white', linewidth=0.4)
            ax.plot(lim, lim, color='#6b6a66', lw=1, ls='--')
            ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel=f'on-sky {label}', ylabel=f'predicted {label}' if j == 0 else None)
            m = ps[(ps.wave == wave) & (ps.param == p) & (ps.source == name)].iloc[0]
            ax.set_title(f'{LABELS[name]}\nMAE {m.MAE:.2g}, bias {m.bias:+.2g}, rho {m.rho:.2f}', fontsize=9.5)
            _style(ax)
        ax = axes[i, -1]
        x, w = np.arange(len(NOMINAL_WAVES)), 0.8 / len(names)
        for k, name in enumerate(names):
            a = ps[(ps.param == p) & (ps.source == name)].set_index('wave').reindex(NOMINAL_WAVES)
            ax.bar(x + (k - len(names) / 2 + 0.5) * w, 100 * a['rel_MAE'], width=w, color=COLORS[name], label=LABELS[name])
        ax.set_xticks(x, NOMINAL_WAVES, fontsize=9)
        ax.set(xlabel='wavelength [nm]', ylabel='per-sample MAE / on-sky median [%]', title=f'{label}: relative MAE')
        _style(ax)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'Per-sample accuracy on the validation observations (left: {wave} nm; right: all wavelengths)', fontsize=12)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    return fig


def plot_distributions(series, tc=None, out_path=None):
    """Density histograms per wavelength (columns) and parameter (rows) with the W1 distance to on-sky; `tc`: one TC or None for all."""
    fig, axes = plt.subplots(len(PARAMS), len(NOMINAL_WAVES), figsize=(3.6 * len(NOMINAL_WAVES), 2.9 * len(PARAMS)), sharex='row', sharey='row')
    for i, (p, (label, scale, bins)) in enumerate(PARAMS.items()):
        for j, wave in enumerate(NOMINAL_WAVES):
            ax, values = axes[i, j], {}
            for name, df in series:
                values[name] = v = clean(select(df, wave, tc)[p], bins, scale)
                _hist(ax, name, v, bins, lw=2.0, label=LABELS[name])
            ax.text(0.97, 0.95, '\n'.join(f'{LABELS[n].split(" ")[0]}: W1 {wasserstein_distance(values["onsky"], v):.2g}' for n, v in values.items() if n != 'onsky'),
                    transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#52514e')
            ax.tick_params(labelsize=8)
            ax.set_title(f'{wave} nm' if i == 0 else '', fontsize=12)
            ax.set_xlabel(label if i == len(PARAMS) - 1 else '', fontsize=10)
            _style(ax)
        hi = max(np.percentile(clean(df[p], bins, scale), 99.5) for _, df in series)
        axes[i, 0].set_xlim(bins[0], min(bins[-1], 1.1 * hi))
        axes[i, 0].set_ylabel('density', fontsize=9)
        axes[i, 0].text(-0.28, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, va='center', ha='center', fontsize=10, color='#52514e')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'ETC PSF parameters of the {series[0][1]["sample_id"].nunique()} validation observations ({"all TCs" if tc is None else f"TC {tc}%"}; dashed: medians)', fontsize=12, y=1.06)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig


def plot_pooled_distributions(series, params=('fwhm', 'ensquared_tt'), out_path=None):
    """Presentation figure: all TCs and wavelengths pooled, one panel per parameter (png + pdf)."""
    bins = {'fwhm': np.arange(20, 160 + 4, 4), 'ensquared_tt': np.arange(0, 0.7 + 0.02, 0.02), 'strehl_tt': np.arange(0, 60 + 2, 2), 'strehl': np.arange(0, 30 + 1, 1)}
    fig, axes = plt.subplots(1, len(params), figsize=(6.2 * len(params), 5.0), squeeze=False)
    for ax, p in zip(axes[0], params):
        label, scale, _ = PARAMS[p]
        values = {}
        for name, df in series:
            values[name] = v = clean(df[p], bins[p], scale)
            _hist(ax, name, v, bins[p], lw=2.5, label=LABELS[name])
        text = [f'median on-sky {np.median(values["onsky"]):.3g}'] + \
               [f'{LABELS[n].split(" ")[0]}: median {np.median(v):.3g}, W1 {wasserstein_distance(values["onsky"], v):.2g}' for n, v in values.items() if n != 'onsky']
        ax.text(0.97, 0.95, '\n'.join(text), transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#52514e', linespacing=1.5)
        ax.set(yticks=[])
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel('density', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2, lw=0.5, axis='x')
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series), frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.905, f'{series[0][1]["sample_id"].nunique()} validation observations, all turbulence categories and wavelengths (500-900 nm) pooled; dashed: medians',
             ha='center', fontsize=11, color='#52514e')
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        fig.savefig(Path(out_path).with_suffix('.pdf'), bbox_inches='tight')
    return fig
