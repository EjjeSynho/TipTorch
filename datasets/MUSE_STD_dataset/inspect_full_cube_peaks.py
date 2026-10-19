#%%
"""
Full-spectral-resolution check of the peak pixel of MUSE NFM STD stars (ExecPlan agents/execplans/NFM_bright_pixels_filter.md).

Runs on the reduced MUSE cubes (DATA + STAT extensions) of the suspicious samples selected by analyze_peak_pixels.py and answers,
per cube, whether the central pixel flickers between spectral slices (over-exposure / charge overflow / cosmic rays) and how much
such slices can bias the spectrally binned peak used by the dataset:

  1. detect_flickering_peaks (the MUSEObservation._detect_bad_exposures logic): ring/peak ratio along the spectrum, adaptive threshold;
  2. spectral_outlier_scores (the spectral part of MUSEObservation.FilterHotPixels): STAT-whitened leave-one-out residual z-scores
     and curvature at the peak pixel and its 8 neighbours;
  3. the peak / ring ratio spectrum itself: a smooth, slowly varying ratio means a sharp but well-behaved core, scatter above the
     STAT-propagated noise means an unstable pixel;
  4. the binned peak pixel (the dataset's wavelength bins) with and without the flagged slices: the bias the flagged slices cause.

    python datasets/MUSE_STD_dataset/inspect_full_cube_peaks.py --cubes <folder with the FITS cubes or FITS files...> [--ids 469 412 ...]
        [--out data/analysis/NFM_STD_peaks/full_cubes] [--roi 64] [--z 6] [--write-masks]

The cube files are matched to the dataset IDs by their name (muse_df 'Filename', ':' -> '-'). Outputs per cube: a PNG with the peak and ring
spectra, the ratio and the flags, a row in full_cube_peaks.csv (counts of flagged slices, max bias of the binned peak) and, with
--write-masks, <ID>_peak_mask.npz: a boolean [N_wvl, H, W] mask of the ROI (True = bad voxel) plus the ROI slices in the cube, to be
applied before the spectral binning in GetSpectralCubeAndHeaderData. CPU only, ~1 min per cube.
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent.parent))
from tiptorch._config import project_settings, DATA_FOLDER
from tiptorch.tools.utils import GetROIaroundMax
from data_processing.MUSE_data_utils import detect_flickering_peaks, spectral_outlier_scores, NA_GAP_SLICES

STD_FOLDER  = Path(project_settings['MUSE_STD_data_folder'])
CUBES_CACHE = STD_FOLDER / 'cached_cubes'

_parser = argparse.ArgumentParser(description='Peak-pixel check on full MUSE NFM cubes')
_parser.add_argument('--cubes', nargs='+', required=True, help='folder(s) with FITS cubes and/or FITS files')
_parser.add_argument('--ids',   nargs='*', type=int, default=None, help='restrict to these dataset IDs')
_parser.add_argument('--out',   type=str, default=str(DATA_FOLDER / 'analysis' / 'NFM_STD_peaks' / 'full_cubes'))
_parser.add_argument('--roi',   type=int, default=64, help='ROI side around the star [px]')
_parser.add_argument('--z',     type=float, default=6.0, help='z-score threshold of the hot-pixel-like slices')
_parser.add_argument('--write-masks', action='store_true')
try:
    _args = _parser.parse_args()
except SystemExit:
    _args = _parser.parse_args(['--cubes', str(STD_FOLDER / 'cubes')])

OUT = Path(_args.out); OUT.mkdir(parents=True, exist_ok=True)
muse_df = pickle.load(open(STD_FOLDER / 'muse_df.pickle', 'rb'))
name_to_id = {str(n).replace(':', '-').replace('.fits', ''): int(i) for i, n in muse_df['Filename'].items()}

cube_files = []
for c in _args.cubes:
    c = Path(c)
    cube_files += sorted(c.glob('*.fits')) + sorted(c.glob('*.fits.fz')) + sorted(c.glob('*.fits.gz')) if c.is_dir() else [c]

def file_to_id(f):
    stem = f.name.replace(':', '-')
    for suffix in ('.fits.fz', '.fits.gz', '.fits'):
        stem = stem.replace(suffix, '')
    if stem in name_to_id:
        return name_to_id[stem]
    hits = [i for n, i in name_to_id.items() if n in stem or stem in n]
    return hits[0] if len(hits) == 1 else None


def bin_edges_of(id):
    """ Full-spectrum slice indices of the dataset's wavelength bins (bins_to_ignore of the Na notch are already dropped). """
    cache = next(CUBES_CACHE.glob(f'{id}_*.pickle'), None)
    if cache is None:
        return None, None
    sample, _ = pickle.load(open(cache, 'rb'))
    sp = sample['spectral data']
    λs = sp['spectrum (full)'][0]
    edges = np.array([np.argmin(np.abs(λs - w)) for w in np.asarray(sp['wvl bins']).ravel()])
    centres = 0.5 * (np.asarray(sp['wvl bins'])[1:] + np.asarray(sp['wvl bins'])[:-1])
    keep = [np.argmin(np.abs(centres - w)) for w in np.asarray(sp['wvls binned'])]
    return edges, keep


rows = []
for f in cube_files:
    id = file_to_id(f)
    if _args.ids is not None and id not in _args.ids:
        continue
    print(f'\n=== {f.name}  (ID {id})')
    with fits.open(f, memmap=True) as hdul:
        data = np.asarray(hdul['DATA'].data if 'DATA' in hdul else hdul[1].data, dtype=np.float32)
        stat = np.asarray(hdul['STAT'].data if 'STAT' in hdul else hdul[2].data, dtype=np.float32)
        hdr  = hdul['DATA'].header if 'DATA' in hdul else hdul[1].header
    N_wvl = data.shape[0]
    λs = (np.arange(N_wvl) * hdr['CD3_3'] + hdr['CRVAL3']) / 10.0 # [nm]

    white = np.nan_to_num(data).sum(0)
    _, roi_ids, max_id = GetROIaroundMax(white, _args.roi // 2)
    roi  = torch.tensor(np.nan_to_num(data[:, roi_ids[0], roi_ids[1]]))
    var  = torch.tensor(np.nan_to_num(stat[:, roi_ids[0], roi_ids[1]], nan=np.inf))
    n_nan_peak = int(np.isnan(data[:, max_id[0], max_id[1]]).sum())
    cy, cx = max_id[0] - roi_ids[0].start, max_id[1] - roi_ids[1].start

    # 1. flicker detection (ring / peak ratio with the adaptive threshold)
    anomalous, overexposed, low_SNR, Na, diag = detect_flickering_peaks(roi[None], torch.tensor([[cy, cx]]), return_diagnostics=True)
    anomalous, overexposed, low_SNR = anomalous[0].numpy(), overexposed[0].numpy(), low_SNR[0].numpy()
    peak, ring = diag['peaks'][0].numpy(), diag['ring_mean'][0].numpy()

    # 2. hot-pixel-like slices at the peak and its 8 neighbours (STAT-whitened)
    pos = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    spectra  = torch.stack([roi[:, y, x] for y, x in pos])
    variance = torch.stack([var[:, y, x] for y, x in pos])
    sc = spectral_outlier_scores(spectra, variance, half_window=11, exclude_radius=1)
    z, zc = sc['z_residual'].numpy(), sc['z_curvature'].numpy()
    hot_peak = (np.abs(z[4]) > _args.z) & (np.abs(zc[4]) > 5) & sc['valid_range'].numpy()
    hot_ring = ((np.abs(z) > _args.z) & (np.abs(zc) > 5)).any(0) & ~hot_peak & sc['valid_range'].numpy()

    # 3. ratio scatter vs the STAT-propagated noise
    ratio = ring / np.where(peak != 0, peak, np.nan)
    good  = ~Na[0].numpy() & ~low_SNR & np.isfinite(ratio)
    from scipy.ndimage import median_filter
    hp = ratio - median_filter(np.nan_to_num(ratio), 31, mode='nearest')
    sigma_ratio = np.sqrt(variance[1:].mean(0).numpy() / 64.0 + ring**2 / np.clip(peak, 1e-6, None)**2 * variance[4].numpy()) / np.clip(np.abs(peak), 1e-6, None)
    chi = hp / np.clip(sigma_ratio, 1e-12, None)
    ratio_chi_rms = float(np.sqrt(np.nanmean(chi[good]**2)))

    # 4. bias of the binned peak due to the flagged slices
    flagged = anomalous | hot_peak
    edges, keep = bin_edges_of(id) if id is not None else (None, None)
    bias = []
    if edges is not None:
        for b in keep:
            a, e = edges[b], edges[b + 1]
            sl = np.arange(a, e); ok = ~flagged[sl]
            if ok.sum() == 0 or np.nansum(peak[sl]) == 0: bias.append(np.nan); continue
            full = np.nanmean(peak[sl]); clean = np.nanmean(peak[sl][ok])
            bias.append((full - clean) / clean)
    bias = np.array(bias)

    row = dict(ID=id, file=f.name, N_wvl=N_wvl, peak_y=max_id[0], peak_x=max_id[1], nan_slices_at_peak=n_nan_peak,
               flicker_slices=int(anomalous.sum()), overexposed_slices=int(overexposed.sum()), low_SNR_slices=int(low_SNR.sum()),
               hot_peak_slices=int(hot_peak.sum()), hot_ring_slices=int(hot_ring.sum()), ratio_chi_rms=ratio_chi_rms,
               peak_max=float(np.nanmax(peak)), bias_max_pct=float(100 * np.nanmax(np.abs(bias))) if bias.size else np.nan,
               bias_med_pct=float(100 * np.nanmedian(np.abs(bias))) if bias.size else np.nan)
    rows.append(row)
    print('  ' + ', '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items() if k not in ('file',)))

    # Figure
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    ax[0].plot(λs, peak, lw=0.6, color='#2A6FDB', label='peak pixel'); ax[0].plot(λs, ring, lw=0.6, color='#D9822B', label='8-neighbour ring mean')
    ax[0].axhline(1500, color='#9AA0A6', lw=0.8, ls='--', label='overexposure threshold'); ax[0].set_ylabel('flux / slice'); ax[0].legend(fontsize=8)
    ax[0].set_title(f'ID {id}  {f.name}: {int(anomalous.sum())} flicker slices, {int(hot_peak.sum())} hot-pixel slices at the peak, ratio chi rms {ratio_chi_rms:.1f}', fontsize=10)
    ax[1].plot(λs, ratio, lw=0.6, color='#2A6FDB'); ax[1].set_ylabel('ring / peak')
    ax[1].scatter(λs[anomalous], ratio[anomalous], s=12, color='#D9822B', label='flicker (ring/peak threshold)', zorder=3)
    ax[1].scatter(λs[hot_peak], ratio[hot_peak], s=12, color='#C0392B', marker='x', label=f'hot pixel |z| > {_args.z:g}', zorder=3); ax[1].legend(fontsize=8)
    ax[2].plot(λs, z[4], lw=0.6, color='#2A6FDB', label='peak'); ax[2].plot(λs, np.nanmax(np.abs(np.delete(z, 4, 0)), 0), lw=0.5, color='#D9822B', alpha=0.7, label='max |z| of the ring')
    ax[2].axhline(_args.z, color='#9AA0A6', lw=0.8, ls='--'); ax[2].axhline(-_args.z, color='#9AA0A6', lw=0.8, ls='--'); ax[2].set_ylabel('STAT-whitened residual z'); ax[2].legend(fontsize=8)
    if bias.size:
        cache = next(CUBES_CACHE.glob(f'{id}_*.pickle')); λb = pickle.load(open(cache, 'rb'))[0]['spectral data']['wvls binned']
        ax[3].bar(λb, 100 * bias, width=12, color='#2A6FDB'); ax[3].set_ylabel('binned peak bias\nfrom flagged slices [%]')
    ax[3].set_xlabel('wavelength [nm]')
    for a in ax: a.axvspan(λs[NA_GAP_SLICES[0]], λs[min(NA_GAP_SLICES[1], N_wvl - 1)], color='#9AA0A6', alpha=0.15)
    plt.tight_layout(); plt.savefig(OUT / f'{id}_{f.stem}.png', dpi=120); plt.close()

    if _args.write_masks and flagged.any():
        mask = np.zeros(roi.shape, dtype=bool)
        mask[flagged, cy, cx] = True
        mask[hot_ring, :, :] = False
        np.savez_compressed(OUT / f'{id}_peak_mask.npz', mask=mask, y0=roi_ids[0].start, y1=roi_ids[0].stop, x0=roi_ids[1].start, x1=roi_ids[1].stop,
                            flagged_slices=np.flatnonzero(flagged))

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'full_cube_peaks.csv', index=False)
print(f'\nSaved {len(summary)} rows to {OUT / "full_cube_peaks.csv"}')
if len(summary):
    print(summary[['ID', 'nan_slices_at_peak', 'flicker_slices', 'hot_peak_slices', 'ratio_chi_rms', 'bias_max_pct', 'bias_med_pct']].to_string(index=False))
