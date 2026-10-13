#%%
"""
Cell-by-cell ETC validation: the on-sky PSF parameters of the validation observations against Enrico's TIPTOP and the TipTorch
ETC predictors (donor sampler and direct network), replayed on the same conditions. The slow stages are cached by
`validate_ETC_calibrator.py` in `data/Enrico_ETC/ETC_calibrator/params_<source>.csv`; set RECOMPUTE to redo one of them.
Run cell by cell in IPython / VS Code (GPU, the astro-tiptop checkouts; ~15 min when nothing is cached).
"""
try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in dir() else Path.cwd().resolve()
for p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    sys.path.insert(0, str(p))

from calibrators.NFM_calibrator import NFMDataset
from calibrators.ETC_calibrator import TC_LIST
import validate_ETC_calibrator as V
from ETC_psf_metrics import load_pupil

# ── Settings ──────────────────────────────────────────────────────────────────
SOURCES   = ['tiptop', 'donor', 'direct-net']     # compared with the on-sky PSFs
BEST      = 'donor'                               # the sampler, shown in the two-panel figure
N_DRAWS   = 3                                     # draws per observation for the sampler
RECOMPUTE = {'onsky': False, 'tiptop': False, 'donor': False, 'direct-net': False}
PROCESSES = 32
PSFAO     = True                                  # the (slow) PSFAO Strehl fit

#%%
# ── Bundle, dataset, validation conditions ────────────────────────────────────
etc_cal = V.load_bundle()
dataset = NFMDataset(V.DATASET_CACHE)
etc = V.load_validation(etc_cal, dataset)
bins, waves_nm = V.nearest_bins(np.array(dataset.configs[0]['sources_science']['Wavelength']) * 1e9)
renderer, pupil = etc_cal.renderer(waves_nm), load_pupil()
V.OUT_DIR.mkdir(parents=True, exist_ok=True)
V.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f'ETC bundle {etc_cal.path.name}; validation: {len(etc)} observations, per TC ' + ', '.join(f'{tc}%: {(etc.tc == tc).sum()}' for tc in TC_LIST))
print(f'wavelength bins {np.round(waves_nm, 1).tolist()} nm')

# The worker processes import `validate_ETC_calibrator`, so the pool is safe in IPython; as a plain script, recomputation runs serially
_pool = None
def get_pool():
    global _pool
    if _pool is None and 'get_ipython' in dir():
        _pool = mp.get_context('spawn').Pool(PROCESSES)
    return _pool

#%%
# ── Tables: on-sky reference, TIPTOP and the TipTorch predictors (cached or recomputed) ──
tables = {}
for name in ['onsky'] + SOURCES:
    cache = V.OUT_DIR / f'params_{name}.csv'
    if cache.exists() and not RECOMPUTE.get(name, False):
        tables[name] = pd.read_csv(cache)
        print(f'{name}: loaded {cache.name}')
    elif name == 'onsky':
        tables[name] = V.stage_onsky(dataset, etc, bins, waves_nm, pupil, get_pool(), PSFAO)
    elif name == 'tiptop':
        tables[name] = V.stage_tiptop(etc, waves_nm, pupil, get_pool(), PSFAO)
    else:
        tables[name] = V.stage_predict(etc_cal, etc, renderer, pupil, get_pool(), PSFAO, [name], N_DRAWS)[name]
onsky = tables.pop('onsky')
if _pool is not None:
    _pool.close()
onsky[onsky.wave == 700][['fwhm', 'strehl', 'strehl_tt', 'ensquared_tt']].describe().round(3)

#%%
# ── Distribution accuracy: W1 / on-sky IQR per wavelength and parameter, bootstrap against TIPTOP ──
acc = V.accuracy_metrics(onsky, tables)
print(acc.pivot_table(index=['param', 'wave'], columns='source', values='W1_over_IQR').round(2))
print()
print(V.accuracy_summary(acc).round(3).T)
V.plot_accuracy(acc, V.PLOTS_DIR / 'ETC_accuracy.png')

#%%
# ── Per-sample accuracy: paired errors against the on-sky value of every observation ──
ps = V.per_sample_metrics(onsky, tables)
print(ps[ps.wave == 700].round(3).to_string(index=False))
print(ps.groupby('source')[['rel_MAE', 'rho', 'within_20pct']].mean().round(3))

fig, axes = plt.subplots(2, len(SOURCES) + 1, figsize=(4.0 * (len(SOURCES) + 1), 7.6), gridspec_kw={'width_ratios': [1] * len(SOURCES) + [1.35]})
o7 = onsky[onsky.wave == 700].set_index('sample')
for i, (p, lim) in enumerate((('fwhm', (25, 160)), ('ensquared_tt', (0, 0.7)))):
    label, scale, _ = V.PARAMS[p]
    for j, name in enumerate(SOURCES):
        ax = axes[i, j]
        pred = tables[name][tables[name].wave == 700].groupby('sample')[p].mean() * scale
        both = pd.concat([(o7[p] * scale).rename('o'), pred.rename('p')], axis=1).dropna()
        ax.scatter(both.o, both.p, s=16, alpha=0.75, color=V.COLORS[name], edgecolor='white', linewidth=0.4)
        ax.plot(lim, lim, color='#6b6a66', lw=1, ls='--')
        ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel=f'on-sky {label}')
        m = ps[(ps.wave == 700) & (ps.param == p) & (ps.source == name)].iloc[0]
        ax.set_title(f'{V.LABELS[name]}\nMAE {m.MAE:.2g}, bias {m.bias:+.2g}, rho {m.rho:.2f}', fontsize=9.5)
        if j == 0:
            ax.set_ylabel(f'predicted {label}', fontsize=9.5)
        V._style(ax)
    ax = axes[i, -1]
    x, w = np.arange(len(V.NOMINAL_WAVES)), 0.8 / len(SOURCES)
    for k, name in enumerate(SOURCES):
        a = ps[(ps.param == p) & (ps.source == name)].set_index('wave').reindex(V.NOMINAL_WAVES)
        ax.bar(x + (k - len(SOURCES) / 2 + 0.5) * w, 100 * a['rel_MAE'], width=w, color=V.COLORS[name], label=V.LABELS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(V.NOMINAL_WAVES, fontsize=9)
    ax.set(xlabel='wavelength [nm]', ylabel='per-sample MAE / on-sky median [%]', title=f'{label}: relative MAE')
    V._style(ax)
axes[0, -1].legend(frameon=False, fontsize=8)
fig.suptitle('Per-sample accuracy on the validation observations (left: 700 nm; right: all wavelengths)', fontsize=12)
plt.tight_layout()
fig.savefig(V.PLOTS_DIR / 'ETC_per_sample.png', dpi=200, bbox_inches='tight')
plt.show()

#%%
# ── Distributions per wavelength: rows = parameters, columns = wavelengths ──
SERIES = [('onsky', onsky), ('tiptop', tables['tiptop']), (BEST, tables[BEST])]

def plot_distributions_per_wavelength(series, tc=None, out_path=None):
    """ Density histograms with medians (dashed) and the W1 distance to on-sky; `tc`: one TC or None for all """
    fig, axes = plt.subplots(len(V.PARAMS), len(V.NOMINAL_WAVES), figsize=(3.6 * len(V.NOMINAL_WAVES), 2.9 * len(V.PARAMS)), sharex='row', sharey='row')
    for i, (p, (label, scale, hbins)) in enumerate(V.PARAMS.items()):
        for j, wave in enumerate(V.NOMINAL_WAVES):
            ax, values = axes[i, j], {}
            for name, df in series:
                sel = df[(df.wave == wave) & ((df.tc == tc) if tc is not None else True)]
                values[name] = v = V.clean(sel[p], hbins, scale)
                if name == 'onsky':
                    ax.hist(v, bins=hbins, density=True, color=V.COLORS[name], alpha=0.3, label=V.LABELS[name])
                else:
                    ax.hist(v, bins=hbins, density=True, histtype='step', color=V.COLORS[name], lw=2.0, label=V.LABELS[name])
                ax.axvline(np.median(v), color=V.COLORS[name], ls='--', lw=1.2)
            ax.text(0.97, 0.95, '\n'.join(f'{V.LABELS[n].split(" ")[0]}: W1 {wasserstein_distance(values["onsky"], v):.2g}' for n, v in values.items() if n != 'onsky'),
                    transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#52514e')
            V._style(ax)
            ax.tick_params(labelsize=8)
            ax.set_title(f'{wave} nm' if i == 0 else '', fontsize=12)
            ax.set_xlabel(label if i == len(V.PARAMS) - 1 else '', fontsize=10)
        hi = max(np.percentile(V.clean(df[p], hbins, scale), 99.5) for _, df in series)
        axes[i, 0].set_xlim(hbins[0], min(hbins[-1], 1.1 * hi))
        axes[i, 0].set_ylabel('density', fontsize=9)
        axes[i, 0].text(-0.28, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, va='center', ha='center', fontsize=10, color='#52514e')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'ETC PSF parameters of the {series[0][1]["sample_id"].nunique()} validation observations ({"all TCs" if tc is None else f"TC {tc}%"}; dashed: medians)', fontsize=12, y=1.06)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig

plot_distributions_per_wavelength(SERIES, out_path=V.PLOTS_DIR / f'ETC_distributions_{BEST}_vs_TIPTOP.png')
plt.show()
for tc in TC_LIST:
    plot_distributions_per_wavelength(SERIES, tc=tc, out_path=V.PLOTS_DIR / f'ETC_distributions_{BEST}_vs_TIPTOP_TC{tc}.png')
    plt.close()

#%%
# ── Presentation figure: FWHM and EE50 distributions, all TCs and wavelengths pooled ──
def plot_pooled_distributions(series, params=('fwhm', 'ensquared_tt'), out_path=None):
    hbins = {'fwhm': np.arange(20, 160 + 4, 4), 'ensquared_tt': np.arange(0, 0.7 + 0.02, 0.02), 'strehl_tt': np.arange(0, 60 + 2, 2), 'strehl': np.arange(0, 30 + 1, 1)}
    fig, axes = plt.subplots(1, len(params), figsize=(6.2 * len(params), 5.0))
    for ax, p in zip(np.atleast_1d(axes), params):
        label, scale, _ = V.PARAMS[p]
        values = {}
        for name, df in series:
            values[name] = v = V.clean(df[p], hbins[p], scale)
            if name == 'onsky':
                ax.hist(v, bins=hbins[p], density=True, color=V.COLORS[name], alpha=0.3, label=V.LABELS[name])
            else:
                ax.hist(v, bins=hbins[p], density=True, histtype='step', color=V.COLORS[name], lw=2.5, label=V.LABELS[name])
            ax.axvline(np.median(v), color=V.COLORS[name], ls='--', lw=1.6)
        text = [f'median on-sky {np.median(values["onsky"]):.3g}'] + [f'{V.LABELS[n].split(" ")[0]}: median {np.median(v):.3g}, W1 {wasserstein_distance(values["onsky"], v):.2g}' for n, v in values.items() if n != 'onsky']
        ax.text(0.97, 0.95, '\n'.join(text), transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#52514e', linespacing=1.5)
        ax.set(xlabel=label, yticks=[])
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel('density', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2, lw=0.5, axis='x')
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(series), frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.905, f'{series[0][1]["sample_id"].nunique()} validation observations, all turbulence categories and wavelengths (500-900 nm) pooled; dashed: medians', ha='center', fontsize=11, color='#52514e')
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        fig.savefig(Path(out_path).with_suffix('.pdf'), bbox_inches='tight')
    return fig

plot_pooled_distributions(SERIES, out_path=V.PLOTS_DIR / f'ETC_pooled_FWHM_EE_{BEST}_vs_TIPTOP.png')
plt.show()

# %%
