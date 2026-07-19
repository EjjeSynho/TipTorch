#%%
"""
Ablation evaluation script for the NFM calibrator.

Scans all completed ablation runs under WEIGHTS_FOLDER / 'NFM_data_ablation',
loads their stored val_cubes.pt and meta.json, and computes per-run metrics:

  • val_loss     - stored final validation loss
  • peak_err     - median relative peak-pixel error (data vs. predicted)
  • fwhm_err     - median relative FWHM error via Moffat2D fits (same logic as
                   MUSE_omega_cen.py); reported per-wavelength and spectrally averaged
  • profile_err  - median max-ΔSR across radii (from radial profiles)

Results are aggregated over repeats (median ± 1-sigma quantiles) and plotted
as a function of the number of training samples.
"""

try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from tiptorch._config import *
from tiptorch.tools.utils import FitMoffat2D_astropy
from tools.plotting import plot_radial_PSF_profiles

# ── Configuration ─────────────────────────────────────────────────────────────
ABLATION_DIR = WEIGHTS_FOLDER / 'NFM_data_ablation'

# ── Helpers ───────────────────────────────────────────────────────────────────
def _moffat_fwhm(psf: np.ndarray) -> float:
    """Return scalar FWHM (pixels) from a 2-D Moffat fit, or NaN on failure."""
    try:
        fx, fy, _, _ = FitMoffat2D_astropy(psf)
        return float(np.sqrt(fx**2 + fy**2))
    except Exception:
        return float('nan')


def _radial_profile_max_err(psf_data: np.ndarray, psf_pred: np.ndarray) -> float:
    """Max ΔSR across radii for a single (H,W) pair.  Returns value in [0,1]."""
    result = plot_radial_PSF_profiles(
        psf_data[None], psf_pred[None],
        suppress_plot=True, return_profiles=True, cutoff=40,
    )
    # result is (p_0, p_1, p_err, centers) - p_err is already in %, shape [N_PSFs, R]
    err_profile = result[2]              # [1, R]
    return float(err_profile.max())      # [%]


def _compute_metrics(PSFs_data: torch.Tensor,
                     PSFs_pred: torch.Tensor,
                     val_loss:  float) -> dict:
    """
    Compute all metrics for one run.

    Parameters
    ----------
    PSFs_data, PSFs_pred : Tensor [N_val, N_wvl, H, W]  (on CPU)
    val_loss : float

    Returns
    -------
    dict with scalar metric values
    """
    N, C, H, W = PSFs_data.shape
    data_np = PSFs_data.numpy()   # [N, C, H, W]
    pred_np = PSFs_pred.numpy()

    # ── peak relative error ──────────────────────────────────────────────────
    data_peak = data_np.max(axis=(-2, -1))   # [N, C]
    pred_peak = pred_np.max(axis=(-2, -1))
    peak_errs = np.abs(data_peak - pred_peak) / (data_peak + 1e-12)  # [N, C]
    peak_err_per_wvl = np.nanmedian(peak_errs, axis=0)   # [C]

    # ── spectrally-averaged PSF  ──────────────────────────────────────────────
    # Average over wavelengths → [N, H, W]
    data_white = data_np.mean(axis=1)
    pred_white = pred_np.mean(axis=1)

    # ── FWHM relative error ───────────────────────────────────────────────────
    # Fit Moffat2D on every (sample, wavelength) pair individually,
    # then report the spectral mean per sample and the overall sample median.
    fwhm_errs = np.full((N, C), np.nan)   # [N, C]
    for n in range(N):
        for i in range(C):
            fd = _moffat_fwhm(data_np[n, i])
            fp = _moffat_fwhm(pred_np[n, i])
            fwhm_errs[n, i] = np.abs(fd - fp) / (fd + 1e-12)

    fwhm_err_per_wvl    = np.nanmedian(fwhm_errs, axis=0)   # [C]  — spectral profile
    fwhm_err_per_sample = np.nanmean  (fwhm_errs, axis=1)   # [N]  — spectrally averaged, per sample

    # ── radial profile max-ΔSR on spectrally-averaged PSF ────────────────────
    # Mirror the validate script: average over wavelengths per sample, then
    # compute max profile error across radii; report median over samples.
    prof_errs = [
        _radial_profile_max_err(data_white[n], pred_white[n])
        for n in range(N)
    ]
    prof_errs = np.array(prof_errs)   # [N]

    return {
        "val_loss":               float(val_loss),
        "peak_err_per_wvl":       peak_err_per_wvl.tolist(),          # [C]
        "peak_err_mean":          float(np.nanmean(peak_err_per_wvl)),
        "fwhm_err_per_wvl":       fwhm_err_per_wvl.tolist(),          # [C]
        "fwhm_err_mean":          float(np.nanmean(fwhm_err_per_wvl)),
        "fwhm_err_per_sample":    fwhm_err_per_sample.tolist(),        # [N]
        "fwhm_err_sample_median": float(np.nanmedian(fwhm_err_per_sample)),
        "fwhm_err_sample_p16":    float(np.nanpercentile(fwhm_err_per_sample, 16)),
        "fwhm_err_sample_p84":    float(np.nanpercentile(fwhm_err_per_sample, 84)),
        "profile_err_per_sample": prof_errs.tolist(),                  # [N]
        "profile_err_median":     float(np.nanmedian(prof_errs)),
        "profile_err_p16":        float(np.nanpercentile(prof_errs, 16)),
        "profile_err_p84":        float(np.nanpercentile(prof_errs, 84)),
    }


#%% ── 1. Discover runs ────────────────────────────────────────────────────────
run_dirs = sorted([d for d in ABLATION_DIR.iterdir()
                   if d.is_dir() and (d / 'meta.json').exists() and (d / 'val_cubes.pt').exists()])

print(f"Found {len(run_dirs)} completed runs in {ABLATION_DIR}")

#%% ── 2. Compute metrics for each run ────────────────────────────────────────
records = []   # list of dicts, one per run

for run_dir in tqdm(run_dirs, desc="Evaluating runs"):
    with open(run_dir / 'meta.json') as _f:
        meta = json.load(_f)

    cubes = torch.load(run_dir / 'val_cubes.pt', map_location='cpu', weights_only=False)
    PSFs_data = cubes['PSFs_data']   # [N, C, H, W]
    PSFs_pred = cubes['PSFs_pred']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metrics = _compute_metrics(PSFs_data, PSFs_pred, cubes['final_val_loss'])

    records.append({
        "run_name":       meta["run_name"],
        "n_samples":      meta["n_samples"],       # None = full dataset
        "n_train_actual": meta["n_train_actual"],
        "repeat_idx":     meta["repeat_idx"],
        "run_seed":       meta["run_seed"],
        **metrics,
    })
    print(
        f"  {meta['run_name']:<22}  n={meta['n_train_actual']:>4}  "
        f"loss={metrics['val_loss']:.5f}  "
        f"FWHM_err={metrics['fwhm_err_mean']*100:.1f}%  "
        f"peak_err={metrics['peak_err_mean']*100:.1f}%  "
        f"prof_err={metrics['profile_err_median']:.1f}%"
    )

print(f"\nEvaluation done - {len(records)} runs processed.")

#%% ── 3. Aggregate over repeats ──────────────────────────────────────────────
# Group by n_train_actual
from collections import defaultdict

groups = defaultdict(list)
for r in records:
    groups[r["n_train_actual"]].append(r)

# Unique n_train values in ascending order (None/full → largest)
n_train_vals = sorted(groups.keys())

def _agg(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return {
        "median": float(np.nanmedian(vals)),
        "p16":    float(np.nanpercentile(vals, 16)),
        "p84":    float(np.nanpercentile(vals, 84)),
        "mean":   float(np.nanmean(vals)),
        "std":    float(np.nanstd(vals)),
        "all":    vals.tolist(),
    }

agg_results = {}
for n in n_train_vals:
    rows = groups[n]
    agg_results[n] = {
        key: _agg(rows, key) for key in ("val_loss", "peak_err_mean", "fwhm_err_mean", "fwhm_err_sample_median", "profile_err_median")
    }

# Print summary table
print(f"\n{'n_train':>8}  {'reps':>4}  {'val_loss':>10}  {'FWHM_err%':>10}  {'peak_err%':>10}  {'prof_err%':>10}")
print("-" * 62)
for n in n_train_vals:
    a = agg_results[n]
    n_lbl = "full" if groups[n][0]["n_samples"] is None else str(n)
    reps  = len(groups[n])
    print(
        f"{n_lbl:>8}  {reps:>4}  "
        f"{a['val_loss']['median']:>10.5f}  "
        f"{a['fwhm_err_mean']['median']*100:>10.2f}  "
        f"{a['peak_err_mean']['median']*100:>10.2f}  "
        f"{a['profile_err_median']['median']:>10.2f}"
    )

#%% ── 4. Plots ────────────────────────────────────────────────────────────────
x      = np.array(n_train_vals)
x_lbl  = [("full" if groups[n][0]["n_samples"] is None else str(n)) for n in n_train_vals]

METRICS = [
    ("val_loss",          "Validation loss",                   None),
    ("fwhm_err_mean",     "FWHM relative error [%]",           100.0),
    ("peak_err_mean",     "Peak-pixel relative error [%]",     100.0),
    ("profile_err_median","Max ΔSR profile error [%]",         None),
]

fig, axes = plt.subplots(1, len(METRICS), figsize=(5.5 * len(METRICS), 4.5))

for ax, (key, ylabel, scale) in zip(axes, METRICS):
    med = np.array([agg_results[n][key]["median"] for n in n_train_vals])
    p16 = np.array([agg_results[n][key]["p16"]    for n in n_train_vals])
    p84 = np.array([agg_results[n][key]["p84"]    for n in n_train_vals])
    if scale is not None:
        med, p16, p84 = med * scale, p16 * scale, p84 * scale

    ax.plot(range(len(x)), med, 'o-', color='tab:blue', linewidth=1.8, markersize=6)
    ax.fill_between(range(len(x)), p16, p84, alpha=0.25, color='tab:blue', label='1-σ quantile range')
    # Individual run dots
    for i, n in enumerate(n_train_vals):
        vals = np.array(agg_results[n][key]["all"])
        if scale is not None:
            vals = vals * scale
        ax.scatter([i] * len(vals), vals, s=22, color='tab:blue', alpha=0.5, zorder=4)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x_lbl, rotation=30, ha='right')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

fig.suptitle('NFM Calibrator - Data Ablation Study', fontsize=13)
plt.tight_layout()
# plt.show()
plt.savefig(ABLATION_DIR / 'ablation_eval_summary.pdf', dpi=200)

#%% ── 5. FWHM error per wavelength for each n_train ─────────────────────────
# Use the mean-PSF FWHM profile across all runs in each group
N_wvl = len(records[0]["fwhm_err_per_wvl"])
wvl_x = np.arange(N_wvl)
cmap  = plt.cm.viridis(np.linspace(0, 1, len(n_train_vals)))

fig, ax = plt.subplots(figsize=(10, 4))
for color, n in zip(cmap, n_train_vals):
    rows = groups[n]
    fwhm_per_wvl = np.array([r["fwhm_err_per_wvl"] for r in rows])   # [reps, C]
    med_curve = np.nanmedian(fwhm_per_wvl, axis=0) * 100
    n_lbl = "full" if groups[n][0]["n_samples"] is None else str(n)
    ax.plot(wvl_x, med_curve, color=color, linewidth=1.5, label=f"N={n_lbl}")

ax.set_xlabel('Wavelength index')
ax.set_ylabel('Median FWHM relative error [%]')
ax.set_title('FWHM error vs. wavelength - by training-set size')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

#%% ── 6. Save aggregated results ─────────────────────────────────────────────
_out_path = ABLATION_DIR / 'ablation_eval_results.json'
with open(_out_path, 'w') as _f:
    # Convert int keys to strings for JSON
    json.dump(
        {str(k): v for k, v in agg_results.items()},
        _f, indent=2
    )
print(f"Aggregated results saved → {_out_path}")

# %%
