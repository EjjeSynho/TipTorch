#%%
"""
Cell-by-cell ETC validation of the best ETC predictor (`donor-params`, see `agents/execplans/ETC_calibrator.md`).

Replays the validation split of the calibrator dataset in the ETC regime and compares, per wavelength, the distributions of the
ETC PSF parameters (FWHM, PSFAO Strehl, peak-ratio Strehl, EE50) of three sources:
  - on-sky:   the dataset PSFs of the validation observations (same metric code, 100 px field),
  - TIPTOP:   Enrico's tuned vanilla TIPTOP run on the same conditions,
  - TipTorch: calibrated TipTorch with the best ETC predictor.
Every slow stage is cached in `data/Enrico_ETC/ETC_calibrator/` by `validate_ETC_calibrator.py`; set RECOMPUTE to redo it.
Run cell by cell in IPython / VS Code (GPU, the astro-tiptop checkouts, ~15 min when nothing is cached).
"""
try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in dir() else Path.cwd().resolve()
for p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    sys.path.insert(0, str(p))

from tiptorch._config import WEIGHTS_FOLDER, default_device
from calibrators.NFM_calibrator import NFMDataset
from calibrators.ETC_calibrator import ETCCalibrator, PREDICTOR_PRESETS, TC_LIST
import validate_ETC_calibrator as V
from ETC_psf_metrics import load_pupil

# ── Settings ──────────────────────────────────────────────────────────────────
BEST       = 'donor-params'   # best ETC predictor of the validation (see comparison_summary.md)
N_DRAWS    = 3                # draws per observation for the stochastic predictor
RECOMPUTE  = {'onsky': False, 'tiptop': False, 'best': False}
PROCESSES  = 32               # workers for the FWHM / PSFAO metrics
TC_SELECT  = None             # None: all TCs in the final figure, or one of TC_LIST
PSFAO      = True             # the (slow) PSFAO Strehl fit

#%%
# ── Bundle, dataset, validation conditions ────────────────────────────────────
with open(V.ETC_CONFIG) as f:
    cfg = json.load(f)
with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
    bundle_path = Path(json.load(f)['bundle_path'])

etc_cal = ETCCalibrator(bundle_path, device=default_device)
dataset = NFMDataset(V.DATASET_CACHE)
etc, tel_scaled = V.load_validation(etc_cal, dataset)
bins, waves_nm = V.nearest_bins(np.array(dataset.configs[0]['sources_science']['Wavelength']) * 1e9, V.NOMINAL_WAVES)
pupil = load_pupil()
V.OUT_DIR.mkdir(parents=True, exist_ok=True)
V.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f'ETC bundle {bundle_path.name} (PSF-R: {etc_cal.psfr_bundle_path.name})')
print(f'Validation: {len(etc)} observations, per TC ' + ', '.join(f'{tc}%: {(etc.tc == tc).sum()}' for tc in TC_LIST))
print(f'Wavelength bins: {np.round(waves_nm, 1).tolist()} nm (nominal {V.NOMINAL_WAVES})')
etc.head()

#%%
# ── Metrics pool (FWHM + PSFAO fits in worker processes), created on first use ─
# The spawned workers import `validate_ETC_calibrator`, not this cell script, so the pool is safe in IPython; when this file
# runs as a plain script, recomputation falls back to the serial path (pool=None) to avoid re-importing the script in the workers.
_pool = None

def get_pool():
    global _pool
    if _pool is None and ('get_ipython' in dir() or __name__ != '__main__'):
        _pool = mp.get_context('spawn').Pool(PROCESSES)
    return _pool

#%%
# ── On-sky reference: the dataset PSFs of the validation observations ─────────
cache = V.OUT_DIR / 'params_onsky.csv'
if cache.exists() and not RECOMPUTE['onsky']:
    onsky = pd.read_csv(cache)
    print(f'on-sky parameters loaded from {cache.name}')
else:
    onsky = V.stage_onsky(dataset, etc, bins, waves_nm, pupil, get_pool(), PSFAO)
    
onsky[onsky.wave == 700][['fwhm', 'strehl', 'strehl_tt', 'ensquared_tt']].describe().round(3)

#%%
# ── Enrico's TIPTOP on the same conditions ────────────────────────────────────
cache = V.OUT_DIR / 'params_tiptop.csv'
if cache.exists() and not RECOMPUTE['tiptop']:
    tiptop = pd.read_csv(cache)
    print(f'TIPTOP parameters loaded from {cache.name}')
else:
    tiptop = V.stage_tiptop(etc, waves_nm, pupil, get_pool(), PSFAO)
    
tiptop[tiptop.wave == 700][['fwhm', 'strehl', 'strehl_tt', 'ensquared_tt']].describe().round(3)

#%%
# ── Best ETC predictor: calibrated TipTorch ───────────────────────────────────
cache = V.OUT_DIR / f'params_{BEST}.csv'
if cache.exists() and not RECOMPUTE['best']:
    best = pd.read_csv(cache)
    print(f'{BEST} parameters loaded from {cache.name} ({best.draw.nunique()} draw(s))')
else:
    kind, kw, heights_policy, stochastic = PREDICTOR_PRESETS[BEST]
    predictor = etc_cal.predictor(kind, **kw)
    renderer  = etc_cal.renderer(waves_nm, heights_policy=heights_policy, batch_size=16)
    rng = np.random.default_rng(0)
    PSFs, params = renderer.render(predictor, etc, rng, n_draws=N_DRAWS if stochastic else 1, verbose=True)
    parts = []
    
    for d in np.unique(params['draw']):
        sel = np.where(params['draw'].to_numpy() == d)[0]
        parts.append(V.attach_conditions(V.measure_PSFs(PSFs[sel].numpy(), waves_nm, pupil, do_psfao=PSFAO, pool=get_pool()), etc, BEST, draw=int(d)))
   
    best = pd.concat(parts, ignore_index=True)
    params.to_csv(V.OUT_DIR / f'predicted_params_{BEST}.csv', index=False)
    best.to_csv(cache, index=False)
best[best.wave == 700][['fwhm', 'strehl', 'strehl_tt', 'ensquared_tt']].describe().round(3)

#%%
if _pool is not None:
    _pool.close()
    _pool.join()

#%%
# ── Distribution distances per wavelength (all TCs) ───────────────────────────
rows = []
for wave in V.NOMINAL_WAVES:
    for p, (label, scale, hbins) in V.PARAMS.items():
        ref = V.clean(onsky[onsky.wave == wave][p], hbins, scale)
        for name, df in (('TIPTOP', tiptop), ('TipTorch', best)):
            v = V.clean(df[df.wave == wave][p], hbins, scale)
            rows.append({'wave': wave, 'param': p, 'source': name, 'median on-sky': np.median(ref), 'median': np.median(v),
                         'W1': wasserstein_distance(ref, v), 'W1 / IQR': wasserstein_distance(ref, v) / (np.percentile(ref, 75) - np.percentile(ref, 25))})
distances = pd.DataFrame(rows)
print(distances.pivot_table(index=['param', 'wave'], columns='source', values='W1 / IQR').round(2))
print('\nmean W1 / IQR over wavelengths and parameters:')
print(distances.groupby('source')['W1 / IQR'].mean().round(3))

#%%
# ── Final figure: distributions per wavelength, on-sky vs TIPTOP vs calibrated TipTorch ──
SERIES = [  # (label, table, colour, filled)   neutral ink for the observations, categorical slots 1 and 2 for the simulators
    ('on-sky (validation PSFs)', onsky,  '#6b6a66', True),
    ("TIPTOP (Enrico's ETC)",    tiptop, '#2a78d6', False),
    (f'TipTorch ETC ({BEST})',   best,   '#eb6834', False),
]

def plot_distributions_per_wavelength(series, tc=None, out_path=None):
    """ Rows: parameters, columns: wavelengths; density histograms with medians (dashed) and the W1 distance to on-sky """
    n_rows, n_cols = len(V.PARAMS), len(V.NOMINAL_WAVES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.9 * n_rows), sharex='row', sharey='row')
    for i, (p, (label, scale, hbins)) in enumerate(V.PARAMS.items()):
        for j, wave in enumerate(V.NOMINAL_WAVES):
            ax = axes[i, j]
            values = {}
            for name, df, color, filled in series:
                sel = df[(df.wave == wave) & ((df.tc == tc) if tc is not None else True)]
                v = V.clean(sel[p], hbins, scale)
                
                if len(v) == 0:
                    continue
                values[name] = v
                
                if filled:
                    ax.hist(v, bins=hbins, density=True, color=color, alpha=0.3, label=name)
                else:
                    ax.hist(v, bins=hbins, density=True, histtype='step', color=color, lw=2.0, label=name)
                    
                ax.axvline(np.median(v), color=color, ls='--', lw=1.2)
                
            ref = values.get(series[0][0])
           
            if ref is not None:
                texts = [f'{name.split(" ")[0]}: W1 {wasserstein_distance(ref, v):.2g}' for name, v in values.items() if name != series[0][0]]
                ax.text(0.97, 0.95, '\n'.join(texts), transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#52514e')
            ax.grid(alpha=0.2, lw=0.5)
            
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
                
            ax.tick_params(labelsize=8)
            
            if i == 0:
                ax.set_title(f'{wave} nm', fontsize=12)
                
            if j == 0:
                ax.set_ylabel('density', fontsize=9)
                
            if i == n_rows - 1:
                ax.set_xlabel(label, fontsize=10)
                
            else:
                ax.set_xlabel('')
            
        hi = max(np.percentile(V.clean(df[p], hbins, scale), 99.5) for _, df, _, _ in series)
        axes[i, 0].set_xlim(hbins[0], min(hbins[-1], 1.1 * hi))
        axes[i, 0].text(-0.28, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, va='center', ha='center', fontsize=10, color='#52514e')
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    scope = 'all TCs' if tc is None else f'TC {tc}%'
    n = series[0][1]['sample_id'].nunique()
    fig.suptitle(f'ETC PSF parameters of the {n} validation observations ({scope}): on-sky vs TIPTOP vs calibrated TipTorch (dashed: medians)',
                 fontsize=12, y=1.06)
    
    plt.tight_layout()
    
    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        print(f'figure -> {out_path}')
        
    return fig

tag = '' if TC_SELECT is None else f'_TC{TC_SELECT}'
fig = plot_distributions_per_wavelength(SERIES, tc=TC_SELECT, out_path=V.PLOTS_DIR / f'ETC_distributions_{BEST}_vs_TIPTOP{tag}.png')
plt.show()

#%%
# ── Same figure for every TC ──────────────────────────────────────────────────
for tc in TC_LIST:
    plot_distributions_per_wavelength(SERIES, tc=tc, out_path=V.PLOTS_DIR / f'ETC_distributions_{BEST}_vs_TIPTOP_TC{tc}.png')
    plt.close()

#%%
# ── Presentation figure: FWHM and EE50 distributions, all TCs and wavelengths pooled ──
def plot_pooled_distributions(series, params=('fwhm', 'ensquared_tt'), out_path=None):
    """ One panel per parameter, all TCs and wavelengths pooled; density histograms, medians (dashed) and W1 to on-sky """
    labels = {'fwhm': 'FWHM [mas]', 'ensquared_tt': 'Encircled energy within r = 50 mas', 'strehl_tt': 'Strehl, peak ratio [%]', 'strehl': 'Strehl, PSFAO fit [%]'}
    hbins  = {'fwhm': np.arange(20, 160 + 4, 4), 'ensquared_tt': np.arange(0, 0.7 + 0.02, 0.02), 'strehl_tt': np.arange(0, 60 + 2, 2), 'strehl': np.arange(0, 30 + 1, 1)}
    fig, axes = plt.subplots(1, len(params), figsize=(6.2 * len(params), 5.0))
    for ax, p in zip(np.atleast_1d(axes), params):
        scale = V.PARAMS[p][1]
        values = {}
        for name, df, color, filled in series:
            v = V.clean(df[p], hbins[p], scale)
            values[name] = v
            if filled:
                ax.hist(v, bins=hbins[p], density=True, color=color, alpha=0.3, label=name)
            else:
                ax.hist(v, bins=hbins[p], density=True, histtype='step', color=color, lw=2.5, label=name)
            ax.axvline(np.median(v), color=color, ls='--', lw=1.6)
        ref = values[series[0][0]]
        text = [f'median on-sky {np.median(ref):.3g}'] + [f'{name.split(" ")[0]}: median {np.median(v):.3g}, W1 {wasserstein_distance(ref, v):.2g}'
                                                          for name, v in values.items() if name != series[0][0]]
        ax.text(0.97, 0.95, '\n'.join(text), transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#52514e', linespacing=1.5)
        ax.set_xlabel(labels[p], fontsize=14)
        ax.set_ylabel('density', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_yticks([])
        ax.grid(alpha=0.2, lw=0.5, axis='x')
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    handles, lbls = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper center', ncol=len(series), frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.02))
    n = series[0][1]['sample_id'].nunique()
    fig.text(0.5, 0.905, f'{n} validation observations, all turbulence categories and wavelengths (500-900 nm) pooled; dashed: medians',
             ha='center', fontsize=11, color='#52514e')
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        fig.savefig(Path(out_path).with_suffix('.pdf'), bbox_inches='tight')
        print(f'figure -> {out_path} (+ .pdf)')
    return fig

fig = plot_pooled_distributions(SERIES, out_path=V.PLOTS_DIR / f'ETC_pooled_FWHM_EE_{BEST}_vs_TIPTOP.png')
plt.show()

#%%
# ── Accuracy metrics: is calibrated TipTorch closer to the on-sky distributions than TIPTOP? ──
# Per (wavelength, parameter) cell: W1 distance to on-sky normalized by the on-sky IQR for both simulators, its paired bootstrap
# difference (observations resampled, 90 % CI), the KS test against on-sky, the error of the median and the 5-95 % coverage of the
# on-sky values by the simulated distribution. Aggregates: mean normalized W1, cells won, Wilcoxon test of the paired W1 differences.
from scipy.stats import ks_2samp, wilcoxon

def cell_values(df, wave, p, scale):
    """ per-observation values (mean over draws for stochastic predictors) and the pooled draws, as arrays indexed by sample """
    sel = df[df.wave == wave]
    per_sample = sel.groupby('sample')[p].mean() * scale
    return per_sample, sel.set_index('sample')[p] * scale

def accuracy_metrics(onsky, sims, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for wave in V.NOMINAL_WAVES:
        for p, (label, scale, _) in V.PARAMS.items():
            ref_ps, _ = cell_values(onsky, wave, p, scale)
            ref = ref_ps.dropna()
            iqr = np.percentile(ref, 75) - np.percentile(ref, 25)
            samples = ref.index.to_numpy()
            pooled = {name: cell_values(df, wave, p, scale)[1] for name, df in sims.items()}
            w1 = {name: wasserstein_distance(ref, v.dropna()) for name, v in pooled.items()}
            # paired bootstrap over observations of the W1 difference TIPTOP - TipTorch
            names = list(sims)
            diffs = []
            for _ in range(n_boot):
                b = rng.choice(samples, size=len(samples), replace=True)
                r = ref.loc[b]
                d = [wasserstein_distance(r, pooled[n].loc[b].dropna()) for n in names]
                diffs.append(d[0] - d[1])
            lo, hi = np.percentile(diffs, [5, 95])
            row = {'wave': wave, 'param': p, 'IQR_onsky': iqr, 'median_onsky': np.median(ref),
                   'dW1_90CI': f'[{lo / iqr:+.2f}, {hi / iqr:+.2f}]', 'better': names[1] if w1[names[0]] > w1[names[1]] else names[0],
                   'significant': bool(lo > 0 or hi < 0)}
            for n in names:
                v = pooled[n].dropna()
                row[f'W1/IQR {n}']   = w1[n] / iqr
                row[f'KS p {n}']     = ks_2samp(ref, v).pvalue
                row[f'dmedian {n}']  = (np.median(v) - np.median(ref)) / np.median(ref)
                lo5, hi95 = np.percentile(v, [5, 95])
                row[f'coverage {n}'] = np.mean((ref >= lo5) & (ref <= hi95))
            rows.append(row)
    return pd.DataFrame(rows)

sims = {'TIPTOP': tiptop, 'TipTorch': best}
acc = accuracy_metrics(onsky, sims)
acc.to_csv(V.OUT_DIR / f'accuracy_metrics_{BEST}_vs_TIPTOP.csv', index=False)

cols = ['wave', 'param', 'W1/IQR TIPTOP', 'W1/IQR TipTorch', 'dW1_90CI', 'significant', 'KS p TIPTOP', 'KS p TipTorch', 'dmedian TIPTOP', 'dmedian TipTorch', 'coverage TIPTOP', 'coverage TipTorch']
print(acc[cols].round(3).to_string(index=False))

summary = pd.DataFrame({
    'mean W1 / IQR':                 {n: acc[f'W1/IQR {n}'].mean() for n in sims},
    'cells closer to on-sky (of 20)': {n: int((acc['better'] == n).sum()) for n in sims},
    'cells closer, 90 % significant': {n: int(((acc['better'] == n) & acc['significant']).sum()) for n in sims},
    'cells not distinguishable (KS p > 0.05)': {n: int((acc[f'KS p {n}'] > 0.05).sum()) for n in sims},
    'mean |median error| [%]':       {n: 100 * acc[f'dmedian {n}'].abs().mean() for n in sims},
    'mean 5-95 % coverage':          {n: acc[f'coverage {n}'].mean() for n in sims},
})
w = wilcoxon(acc['W1/IQR TIPTOP'] - acc['W1/IQR TipTorch'], alternative='greater')
print('\n' + summary.round(3).T.to_string())
print(f"\nWilcoxon signed-rank test over the 20 cells, H1: W1(TIPTOP) > W1(TipTorch): p = {w.pvalue:.2g}; "
      f"median ratio W1(TipTorch) / W1(TIPTOP) = {np.median(acc['W1/IQR TipTorch'] / acc['W1/IQR TIPTOP']):.2f}")
summary.T.to_csv(V.OUT_DIR / f'accuracy_summary_{BEST}_vs_TIPTOP.csv')

#%%
# ── Figure: normalized distance to on-sky per wavelength and parameter, TIPTOP vs TipTorch ──
fig, axes = plt.subplots(1, len(V.PARAMS), figsize=(4.2 * len(V.PARAMS), 4.2), sharey=True)
x = np.arange(len(V.NOMINAL_WAVES))
for ax, (p, (label, _, _)) in zip(axes, V.PARAMS.items()):
    a = acc[acc.param == p].set_index('wave').reindex(V.NOMINAL_WAVES)
    ax.bar(x - 0.2, a['W1/IQR TIPTOP'],   width=0.38, color='#2a78d6', label='TIPTOP (Enrico\'s ETC)')
    ax.bar(x + 0.2, a['W1/IQR TipTorch'], width=0.38, color='#eb6834', label=f'TipTorch ETC ({BEST})')
    for xi, sig, better in zip(x, a['significant'], a['better']):
        if sig:
            ax.text(xi, max(a['W1/IQR TIPTOP'].iloc[xi], a['W1/IQR TipTorch'].iloc[xi]) + 0.03, '*', ha='center', fontsize=14, color='#52514e')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w} nm' for w in V.NOMINAL_WAVES], fontsize=10)
    ax.set_title(label, fontsize=12)
    ax.grid(alpha=0.2, axis='y', lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel('distance to on-sky  (W1 / on-sky IQR)', fontsize=11)
axes[0].legend(frameon=False, fontsize=10, loc='upper left')
fig.suptitle(f'Distribution distance to the on-sky PSF parameters per wavelength (lower is better; * : 90 % bootstrap CI excludes 0)\n'
             f'mean over all cells: TIPTOP {summary.loc["TIPTOP", "mean W1 / IQR"]:.2f}, TipTorch {summary.loc["TipTorch", "mean W1 / IQR"]:.2f}', fontsize=12)
plt.tight_layout()
fig.savefig(V.PLOTS_DIR / f'ETC_accuracy_{BEST}_vs_TIPTOP.png', dpi=200, bbox_inches='tight')
fig.savefig(V.PLOTS_DIR / f'ETC_accuracy_{BEST}_vs_TIPTOP.pdf', bbox_inches='tight')
plt.show()

#%%
# ── Per-sample accuracy: paired errors against the on-sky value of each observation ──
# TIPTOP and the direct network are deterministic; donor-params is a sampler, so its per-sample prediction is the mean over the
# draws (which pays a variance penalty by construction). Metrics per (wavelength, parameter): MAE, bias, Spearman rho,
# relative MAE (MAE / on-sky median) and the fraction of observations predicted within 20 % of the on-sky value.
from scipy.stats import spearmanr

cache = V.OUT_DIR / 'params_direct-net.csv'
direct = pd.read_csv(cache) if cache.exists() else None
if direct is None:
    print('direct-net table not found; run validate_ETC_calibrator.py --stage predict --predictors direct-net')
PER_SAMPLE = {'TIPTOP': tiptop, 'TipTorch direct-net': direct, f'TipTorch {BEST} (mean of draws)': best}
PER_SAMPLE = {k: v for k, v in PER_SAMPLE.items() if v is not None}
PS_COLORS  = {'TIPTOP': '#2a78d6', 'TipTorch direct-net': '#1baf7a', f'TipTorch {BEST} (mean of draws)': '#eb6834'}

def per_sample_metrics(onsky, sims):
    rows = []
    for wave in V.NOMINAL_WAVES:
        for p, (label, scale, _) in V.PARAMS.items():
            ref = onsky[onsky.wave == wave].set_index('sample')[p] * scale
            for name, df in sims.items():
                pred = df[df.wave == wave].groupby('sample')[p].mean() * scale
                both = pd.concat([ref.rename('o'), pred.rename('p')], axis=1).dropna()
                err = both.p - both.o
                rows.append({'wave': wave, 'param': p, 'source': name, 'N': len(both), 'MAE': err.abs().mean(), 'bias': err.mean(),
                             'rho': spearmanr(both.o, both.p).statistic, 'rel_MAE': err.abs().mean() / both.o.median(),
                             'within_20pct': np.mean(err.abs() <= 0.2 * both.o)})
    return pd.DataFrame(rows)

ps = per_sample_metrics(onsky, PER_SAMPLE)
ps.to_csv(V.OUT_DIR / f'per_sample_metrics_{BEST}_vs_TIPTOP.csv', index=False)
print(ps[ps.wave == 700].round(3).to_string(index=False))
ps_summary = ps.groupby('source')[['rel_MAE', 'rho', 'within_20pct']].mean().round(3)
ps_summary['cells with lowest MAE (of 20)'] = ps.loc[ps.groupby(['wave', 'param'])['MAE'].idxmin(), 'source'].value_counts().reindex(ps_summary.index).fillna(0).astype(int)
print('\nmean over all 20 cells:')
print(ps_summary.to_string())

#%%
# ── Figure: per-sample scatter at 700 nm (FWHM, EE50) and relative MAE per wavelength ──
fig, axes = plt.subplots(2, len(PER_SAMPLE) + 1, figsize=(4.0 * (len(PER_SAMPLE) + 1), 7.6), gridspec_kw={'width_ratios': [1] * len(PER_SAMPLE) + [1.35]})
o7 = onsky[onsky.wave == 700].set_index('sample')
for i, (p, lim) in enumerate((('fwhm', (25, 160)), ('ensquared_tt', (0, 0.7)))):
    label, scale, _ = V.PARAMS[p]
    for j, (name, df) in enumerate(PER_SAMPLE.items()):
        ax = axes[i, j]
        pred = df[df.wave == 700].groupby('sample')[p].mean() * scale
        both = pd.concat([(o7[p] * scale).rename('o'), pred.rename('p')], axis=1).dropna()
        ax.scatter(both.o, both.p, s=16, alpha=0.75, color=PS_COLORS[name], edgecolor='white', linewidth=0.4)
        ax.plot(lim, lim, color='#6b6a66', lw=1, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        m = ps[(ps.wave == 700) & (ps.param == p) & (ps.source == name)].iloc[0]
        ax.set_title(f'{name}\nMAE {m.MAE:.2g}, bias {m.bias:+.2g}, rho {m.rho:.2f}', fontsize=9.5)
        ax.set_xlabel(f'on-sky {label}', fontsize=9.5)
        if j == 0:
            ax.set_ylabel(f'predicted {label}', fontsize=9.5)
        ax.grid(alpha=0.2, lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    ax = axes[i, -1]
    x = np.arange(len(V.NOMINAL_WAVES))
    w = 0.8 / len(PER_SAMPLE)
    for k, name in enumerate(PER_SAMPLE):
        a = ps[(ps.param == p) & (ps.source == name)].set_index('wave').reindex(V.NOMINAL_WAVES)
        ax.bar(x + (k - len(PER_SAMPLE) / 2 + 0.5) * w, 100 * a['rel_MAE'], width=w, color=PS_COLORS[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{wv}' for wv in V.NOMINAL_WAVES], fontsize=9)
    ax.set_xlabel('wavelength [nm]', fontsize=9.5)
    ax.set_ylabel('per-sample MAE / on-sky median [%]', fontsize=9.5)
    ax.set_title(f'{label}: relative MAE per wavelength', fontsize=9.5)
    ax.grid(alpha=0.2, axis='y', lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0, -1].legend(frameon=False, fontsize=8)
fig.suptitle('Per-sample accuracy on the validation observations (left: 700 nm, predicted vs on-sky; right: all wavelengths)', fontsize=12)
plt.tight_layout()
fig.savefig(V.PLOTS_DIR / f'ETC_per_sample_{BEST}_vs_TIPTOP.png', dpi=200, bbox_inches='tight')
fig.savefig(V.PLOTS_DIR / f'ETC_per_sample_{BEST}_vs_TIPTOP.pdf', bbox_inches='tight')
plt.show()

# %%
