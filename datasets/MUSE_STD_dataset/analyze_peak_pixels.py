#%%
"""
Peak-pixel diagnostic for the fitted MUSE NFM STD stars (ExecPlan agents/execplans/NFM_bright_pixels_filter.md).

Hypothesis under test: the large (10-30 %) peak errors of the PSF fits come from a few over-exposed / charge-overfilled
central pixels whose values flicker between spectral slices, so the spectrally binned data holds a biased peak. The fits
(`fitted/{id}.pickle`, DEROTATED frame) tell how big and how localized the core residual is; the cached cubes
(`cached_cubes/`, RAW detector frame, binned sum + spectral std within each bin) tell whether the raw peak pixel behaves
anomalously, and the box spectrum of the star (3681 slices) whether single slices spike.

Per sample and wavelength bin:
  peak_err          (data peak - model peak) / data peak, 7x7 central crop (same definition as NFM_validation_utils)
  top1_frac         |residual| of the worst pixel / sum |residual| over the central 7x7 (1 = the whole core error sits in one pixel)
  ring_err          relative error of the 8 neighbours of the data peak (small ring_err + large peak_err = the pixel, not the core)
  spike_ratio       (data peak / data ring) / (model peak / model ring): how much sharper the data peak is than the model's
  cv_excess         raw frame: coefficient of variation (spectral std / mean within the bin) at the peak pixel relative to its 8 neighbours
  peak_exposure     raw frame: mean peak flux per slice x exposure time (proxy for the detector counts at the peak)
  spectrum_spikes   number of slices of the box spectrum deviating > 6 sigma from a running median (hot-pixel-like events)

    python datasets/MUSE_STD_dataset/analyze_peak_pixels.py

Outputs: data/analysis/NFM_STD_peaks/peak_pixels_per_wvl.csv, peak_pixels.csv (per sample), peak_pixels_*.png.
Needs fitted/ and cached_cubes/ of the STD-star folder, CPU only, ~3 min.
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter
from tqdm import tqdm

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent.parent))
from tiptorch._config import project_settings, DATA_FOLDER

STD_FOLDER    = Path(project_settings['MUSE_STD_data_folder'])
FITTED_FOLDER = Path(project_settings['MUSE_fitted_STD_folder'])
CUBES_CACHE   = STD_FOLDER / 'cached_cubes'
OUT_FOLDER    = DATA_FOLDER / 'analysis' / 'NFM_STD_peaks'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

WVL_IDS = np.clip(np.arange(0, 32, 2), 0, 30) # the 16 bins used by STD_fitter.py out of the 31 cached ones
NA_GAP  = (824, 1041)                         # full-spectrum slice indices of the Na-notch (as in MUSEObservation._detect_bad_exposures)
C_HALF  = 3                                   # 7x7 central crop for the peak

muse_df = pickle.load(open(STD_FOLDER / 'muse_df.pickle', 'rb'))
FLAGS   = ['Good quality', 'Medium quality', 'Bad quality', 'Corrupted', 'Non-point', 'Has streaks', 'Tomography error', 'k_c mismatch']


def peak_and_ring(img, c=None):
    """ Peak pixel (within the 7x7 central crop) and the mean of its 8 neighbours. `img`: [..., H, W]. Returns (peak, ring, iy, ix). """
    H, W = img.shape[-2:]
    cy, cx = (H // 2, W // 2) if c is None else c
    crop = img[..., cy-C_HALF:cy+C_HALF+1, cx-C_HALF:cx+C_HALF+1]
    flat = crop.reshape(*crop.shape[:-2], -1)
    k    = flat.argmax(-1)
    iy, ix = np.unravel_index(k, crop.shape[-2:])
    iy, ix = iy + cy - C_HALF, ix + cx - C_HALF
    peak = np.take_along_axis(img.reshape(*img.shape[:-2], -1), (iy * W + ix)[..., None], -1)[..., 0]
    ring = np.zeros_like(peak)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            ring += np.take_along_axis(img.reshape(*img.shape[:-2], -1), ((iy + dy) * W + ix + dx)[..., None], -1)[..., 0]
    return peak, ring / 8.0, iy, ix


def fitted_core_metrics(D, P):
    """ D, P: [N_wvl, H, W] data and model (derotated frame, unit sum). """
    H, W = D.shape[-2:]
    cy, cx = H // 2, W // 2
    d_peak, d_ring, iy, ix = peak_and_ring(D)
    p_peak, p_ring, _, _   = peak_and_ring(P)
    # the model evaluated at the data's peak pixel (the model peak may sit on a neighbouring pixel)
    p_at = P.reshape(D.shape[0], -1)[np.arange(D.shape[0]), iy * W + ix]
    R  = (D - P)[:, cy-C_HALF:cy+C_HALF+1, cx-C_HALF:cx+C_HALF+1]
    aR = np.abs(R).reshape(D.shape[0], -1)
    srt = np.sort(aR, axis=-1)[:, ::-1]
    tot = aR.sum(-1).clip(1e-30)
    # ring of the data peak in the model vs data
    yy, xx = np.mgrid[:H, :W]
    out = dict(
        peak_err    = (d_peak - p_peak) / d_peak,
        peak_err_at = (d_peak - p_at) / d_peak,
        top1_frac   = srt[:, 0] / tot,
        top3_frac   = srt[:, :3].sum(-1) / tot,
        ring_err    = (d_ring - p_ring) / d_ring,
        spike_ratio = (d_peak / d_ring) / (p_peak / p_ring),
        peak_iy = iy, peak_ix = ix,
    )
    # how well the rest of the core / AO area fits: median |rel. err| in 3 < r <= 10 px, flux-weighted
    r = np.hypot(yy - cy, xx - cx)
    band = (r > 3) & (r <= 10)
    out['band_err'] = np.abs(D[:, band] - P[:, band]).sum(-1) / D[:, band].sum(-1)
    return out


def raw_cube_metrics(sample):
    """ Spectral-variability metrics of the raw (unrotated) cached cube at its peak pixel. """
    cube, std = sample['images']['cube'], sample['images']['std']
    sp   = sample['spectral data']
    λ_edges, λ_bins = np.asarray(sp['wvl bins'], float), np.asarray(sp['wvls binned'], float)
    Δλ = float(sp['wvl range'][2])
    # slices per bin: bins of the edge list matched to the surviving bin centres (the Na bin was dropped)
    centres = 0.5 * (λ_edges[1:] + λ_edges[:-1])
    n_all   = np.round(np.diff(λ_edges) / Δλ).astype(int)
    n_slices = np.array([n_all[np.argmin(np.abs(centres - λ))] for λ in λ_bins])
    cube, std = cube[WVL_IDS], std[WVL_IDS]
    n = n_slices[WVL_IDS].astype(float)

    mean = cube / n[:, None, None]
    m_peak, m_ring, iy, ix = peak_and_ring(mean)
    cv = std / np.abs(mean).clip(1e-12)
    H, W = cube.shape[-2:]
    cv_peak = cv.reshape(cv.shape[0], -1)[np.arange(cv.shape[0]), iy * W + ix]
    cv_ring = np.stack([cv.reshape(cv.shape[0], -1)[np.arange(cv.shape[0]), (iy + dy) * W + ix + dx]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], -1)
    cv_ring = np.median(cv_ring, -1)
    exptime = float(sample['All data']['Exp. time'].iloc[0])

    # box spectrum of the star (3681 slices): single-slice spikes against a running median
    λs, s = sp['spectrum (full)']
    s = np.asarray(s, float)
    idx = np.arange(len(s))
    valid = np.isfinite(s) & ((idx < NA_GAP[0]) | (idx >= NA_GAP[1]))
    hp = s - median_filter(np.nan_to_num(s), size=15, mode='nearest')
    hp_v = hp[valid]
    mad = 1.4826 * np.median(np.abs(hp_v - np.median(hp_v))).clip(1e-12)
    z = np.where(valid, hp / mad, 0.0)
    return dict(
        cv_peak = cv_peak, cv_ring = cv_ring, cv_excess = cv_peak / cv_ring.clip(1e-12),
        raw_spike = m_peak / m_ring, peak_flux_slice = m_peak, peak_exposure = m_peak * exptime, exptime = exptime,
        n_slices = n, raw_peak_iy = iy, raw_peak_ix = ix,
        spectrum_spikes = int((np.abs(z) > 6).sum()), spectrum_max_z = float(np.abs(z).max()),
        spectrum_spikes_pos = int((z > 6).sum()), spectrum_spikes_neg = int((z < -6).sum()),
    )


#%%
rows = []
fitted_files = sorted(FITTED_FOLDER.glob('*.pickle'), key=lambda p: int(p.stem))
cache_index  = {int(p.name.split('_', 1)[0]): p for p in CUBES_CACHE.glob('*.pickle')}

for f in tqdm(fitted_files):
    id = int(f.stem)
    if id not in cache_index:
        continue
    fit = pickle.load(open(f, 'rb'))
    D, P = fit['Img. data'][0], fit['Img. fit'][0]
    λ_nm = np.asarray(fit['wavelengths'], float).ravel() * 1e9
    core = fitted_core_metrics(D, P)
    sample, _ = pickle.load(open(cache_index[id], 'rb'))
    raw = raw_cube_metrics(sample)
    for l in range(D.shape[0]):
        row = {'ID': id, 'wvl_id': l, 'λ_nm': λ_nm[l], 'SR_data': float(fit['SR data'][l]), 'SR_fit': float(fit['SR fit'][l])}
        row.update({k: (float(v[l]) if np.ndim(v) else v) for k, v in core.items()})
        row.update({k: (float(v[l]) if np.ndim(v) else v) for k, v in raw.items()})
        rows.append(row)

per_wvl = pd.DataFrame(rows)
per_wvl.to_csv(OUT_FOLDER / 'peak_pixels_per_wvl.csv', index=False)

agg = per_wvl.groupby('ID').agg(
    peak_err_med     = ('peak_err', 'median'),
    peak_err_absmed  = ('peak_err', lambda x: np.median(np.abs(x))),
    peak_err_min     = ('peak_err', 'min'),
    peak_err_max     = ('peak_err', 'max'),
    top1_frac_med    = ('top1_frac', 'median'),
    top3_frac_med    = ('top3_frac', 'median'),
    ring_err_med     = ('ring_err', 'median'),
    spike_ratio_med  = ('spike_ratio', 'median'),
    band_err_med     = ('band_err', 'median'),
    cv_excess_med    = ('cv_excess', 'median'),
    cv_excess_max    = ('cv_excess', 'max'),
    cv_peak_med      = ('cv_peak', 'median'),
    raw_spike_med    = ('raw_spike', 'median'),
    peak_exposure_max= ('peak_exposure', 'max'),
    peak_flux_max    = ('peak_flux_slice', 'max'),
    exptime          = ('exptime', 'first'),
    spectrum_spikes  = ('spectrum_spikes', 'first'),
    spectrum_spikes_pos = ('spectrum_spikes_pos', 'first'),
    spectrum_max_z   = ('spectrum_max_z', 'first'),
    SR_data_med      = ('SR_data', 'median'),
)
for c in FLAGS:
    agg[c] = muse_df.loc[agg.index, c].astype(bool).values
agg['Filename'] = muse_df.loc[agg.index, 'Filename'].values
agg = agg.sort_values('peak_err_absmed', ascending=False)
agg.to_csv(OUT_FOLDER / 'peak_pixels.csv')

print(f'{len(agg)} samples, {len(per_wvl)} (sample, bin) pairs')
print('median |peak err| over samples: %.1f %%,  sign: %.0f %% of bins model too faint (peak_err > 0)' %
      (100 * agg['peak_err_absmed'].median(), 100 * (per_wvl['peak_err'] > 0).mean()))
print('\nWorst 25 samples by median |peak err| (flags: G good, M medium, B bad, C corrupted, N non-point, S streaks, T tomography, K k_c):')
flag_str = lambda r: ''.join(ch if r[c] else '.' for ch, c in zip('GMBCNSTK', FLAGS))
hdr = f"{'ID':>4} {'flags':>8} {'|pk|%':>6} {'pk%':>6} {'top1':>5} {'ring%':>6} {'spike':>6} {'band%':>6} {'cvX':>5} {'pkExp':>8} {'spk':>4} {'maxz':>5}"
print(hdr)
for id, r in agg.head(25).iterrows():
    print(f"{id:>4} {flag_str(r):>8} {100*r.peak_err_absmed:6.1f} {100*r.peak_err_med:6.1f} {r.top1_frac_med:5.2f} {100*r.ring_err_med:6.1f} "
          f"{r.spike_ratio_med:6.2f} {100*r.band_err_med:6.1f} {r.cv_excess_med:5.2f} {r.peak_exposure_max:8.0f} {r.spectrum_spikes:4d} {r.spectrum_max_z:5.1f}")

#%% Correlations: does the peak error follow any raw-pixel anomaly indicator?
print('\nSpearman correlation of median |peak err| with:')
for c in ['top1_frac_med', 'ring_err_med', 'spike_ratio_med', 'cv_excess_med', 'cv_excess_max', 'raw_spike_med', 'peak_exposure_max',
          'peak_flux_max', 'exptime', 'spectrum_spikes', 'spectrum_max_z', 'SR_data_med', 'band_err_med']:
    print(f'  {c:>18}: {agg["peak_err_absmed"].corr(agg[c], method="spearman"):+.2f}')
print('\nSpearman correlation of signed median peak err with:')
for c in ['spike_ratio_med', 'cv_excess_med', 'peak_exposure_max', 'SR_data_med', 'ring_err_med']:
    print(f'  {c:>18}: {agg["peak_err_med"].corr(agg[c], method="spearman"):+.2f}')

print('\nBy quality flag: median |peak err| / top1 / ring err / spike ratio')
for c in FLAGS:
    for v in (True, False):
        sub = agg[agg[c] == v]
        if len(sub):
            print(f'  {c:>17}={str(v):5}: n={len(sub):3d}  |pk| {100*sub.peak_err_absmed.median():5.1f} %  pk {100*sub.peak_err_med.median():+5.1f} %  '
                  f'top1 {sub.top1_frac_med.median():.2f}  ring {100*sub.ring_err_med.median():+5.1f} %  spike {sub.spike_ratio_med.median():.2f}  cvX {sub.cv_excess_med.median():.2f}')

#%% Figures
C_MAIN, C_ALT, C_MUTED = '#2A6FDB', '#D9822B', '#9AA0A6'
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ax = axes[0, 0]
ax.hist(100 * per_wvl['peak_err'], bins=80, color=C_MAIN, alpha=0.85)
ax.axvline(0, color=C_MUTED, lw=1); ax.set_xlabel('peak error (data - model) / data [%]'); ax.set_ylabel('(sample, bin) pairs'); ax.set_title('Peak error, all bins')
ax = axes[0, 1]
ax.scatter(per_wvl['ring_err'] * 100, per_wvl['peak_err'] * 100, s=4, alpha=0.3, color=C_MAIN)
ax.plot([-40, 40], [-40, 40], color=C_MUTED, lw=1); ax.set_xlim(-40, 40); ax.set_ylim(-40, 40)
ax.set_xlabel('8-neighbour ring error [%]'); ax.set_ylabel('peak error [%]'); ax.set_title('Peak vs ring error (per bin)')
ax = axes[0, 2]
ax.scatter(per_wvl['top1_frac'], per_wvl['peak_err'] * 100, s=4, alpha=0.3, color=C_MAIN)
ax.set_xlabel('worst pixel share of the 7x7 |residual|'); ax.set_ylabel('peak error [%]'); ax.set_title('Residual localization')
ax = axes[1, 0]
ax.scatter(agg['peak_exposure_max'], 100 * agg['peak_err_absmed'], s=10, alpha=0.6, color=C_MAIN)
ax.set_xscale('log'); ax.set_xlabel('max peak flux per slice x exposure time (counts proxy)'); ax.set_ylabel('median |peak error| [%]'); ax.set_title('Over-exposure test')
ax = axes[1, 1]
ax.scatter(per_wvl['cv_excess'], per_wvl['peak_err'] * 100, s=4, alpha=0.3, color=C_MAIN)
ax.set_xscale('log'); ax.axvline(1, color=C_MUTED, lw=1); ax.set_xlabel('spectral CV at raw peak / CV of its ring'); ax.set_ylabel('peak error [%]'); ax.set_title('Peak-pixel spectral variability excess')
ax = axes[1, 2]
for l in range(16):
    sub = per_wvl[per_wvl['wvl_id'] == l]
    ax.errorbar(sub['λ_nm'].iloc[0], 100 * sub['peak_err'].median(), yerr=100 * np.abs(sub['peak_err'].quantile([0.16, 0.84]).values - sub['peak_err'].median())[:, None], fmt='o', color=C_MAIN, ms=4)
ax.axhline(0, color=C_MUTED, lw=1); ax.set_xlabel('wavelength [nm]'); ax.set_ylabel('peak error [%] (median, 16-84 %)'); ax.set_title('Peak error vs wavelength')
plt.tight_layout(); plt.savefig(OUT_FOLDER / 'peak_pixels_overview.png', dpi=130); plt.close()

# Gallery of the worst cores: data, model, residual (derotated frame) at the bin of the largest |peak err|, plus the raw CV map
N_show = 12
fig, axes = plt.subplots(N_show, 4, figsize=(10, 2.5 * N_show))
for k, (id, r) in enumerate(agg.head(N_show).iterrows()):
    fit = pickle.load(open(FITTED_FOLDER / f'{id}.pickle', 'rb'))
    D, P = fit['Img. data'][0], fit['Img. fit'][0]
    sub = per_wvl[per_wvl['ID'] == id]
    l = int(sub.iloc[np.argmax(np.abs(sub['peak_err'].values))]['wvl_id'])
    H = D.shape[-1]; c = H // 2; s = np.s_[c-7:c+8, c-7:c+8]
    vmax = D[l][s].max()
    axes[k, 0].imshow(D[l][s], vmin=0, vmax=vmax, cmap='viridis'); axes[k, 0].set_title(f'ID {id}  data {sub.iloc[l]["λ_nm"]:.0f} nm', fontsize=9)
    axes[k, 1].imshow(P[l][s], vmin=0, vmax=vmax, cmap='viridis'); axes[k, 1].set_title(f'model  peak err {100*sub.iloc[l]["peak_err"]:+.0f} %', fontsize=9)
    res = (D[l] - P[l])[s] / vmax * 100
    lim = np.abs(res).max()
    axes[k, 2].imshow(res, vmin=-lim, vmax=lim, cmap='RdBu_r'); axes[k, 2].set_title(f'residual [% of peak], top1 {sub.iloc[l]["top1_frac"]:.2f}', fontsize=9)
    sample, _ = pickle.load(open(cache_index[id], 'rb'))
    cube, std = sample['images']['cube'][WVL_IDS[l]], sample['images']['std'][WVL_IDS[l]]
    iy, ix = int(sub.iloc[l]['raw_peak_iy']), int(sub.iloc[l]['raw_peak_ix'])
    sr = np.s_[iy-7:iy+8, ix-7:ix+8]
    cv = std[sr] / np.abs(cube[sr] / sub.iloc[l]['n_slices']).clip(1e-12)
    axes[k, 3].imshow(cv, vmin=0, vmax=np.nanpercentile(cv, 95), cmap='magma'); axes[k, 3].set_title(f'raw spectral CV, excess {sub.iloc[l]["cv_excess"]:.2f}', fontsize=9)
    for a in axes[k]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT_FOLDER / 'peak_pixels_worst_gallery.png', dpi=110); plt.close()
print(f'\nSaved tables and figures to {OUT_FOLDER}')
