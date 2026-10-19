"""
Checks of the shared bright-pixel spectral QC functions in data_processing/MUSE_data_utils.py on synthetic cubes (CPU, a few seconds):

  - detect_flickering_peaks: recovers slices where the peak pixel of a bright source drops (charge overflow) and ignores a clean source;
  - spectral_outlier_scores: a single-slice spike at a pixel gets a high z-score / curvature and compact support;
  - peak_variability_excess: a flickering peak stands out in the binned sum/std representation of the STD-star cache.

    python tests/test_bright_pixel_qc.py
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_processing.MUSE_data_utils import detect_flickering_peaks, spectral_outlier_scores, peak_variability_excess, NA_GAP_SLICES

rng = np.random.default_rng(1)
N_wvl, H, W = 1500, 36, 36
yy, xx = np.mgrid[:H, :W]
psf = np.exp(-((yy - 18)**2 + (xx - 18)**2) / (2 * 1.2**2))
spectrum = 1.0 + 0.3 * np.sin(np.linspace(0, 6, N_wvl))

def make_roi(amp, noise=20.0):
    return amp * spectrum[:, None, None] * psf[None] + rng.normal(0, noise, (N_wvl, H, W))

results = []

# ---- detect_flickering_peaks: source 0 flickers (peak drops by 40 % in 25 slices), source 1 is clean
roi_bad, roi_ok = make_roi(6000.0), make_roi(6000.0)
bad = np.sort(rng.choice(np.arange(100, NA_GAP_SLICES[0] - 50), 25, replace=False))
roi_bad[bad, 18, 18] *= 0.6
ROIs = torch.tensor(np.stack([roi_bad, roi_ok]), dtype=torch.float32)
centers = torch.tensor([[18, 18], [18, 18]])
anomalous, overexposed, low_SNR, Na = detect_flickering_peaks(ROIs, centers)
recovered = int(anomalous[0, bad].sum())
false_bad = int(anomalous[0].sum()) - recovered
false_ok  = int(anomalous[1].sum())
# The adaptive quantile threshold always flags the upper tail of the ring/peak ratio, so a clean source gets ~1 % of its slices flagged
# (noise); the injected drops must be recovered well above that rate
results.append((recovered >= 10 and false_bad <= 15 and false_ok <= 30,
                f'detect_flickering_peaks: {recovered}/{len(bad)} injected slices flagged, {false_bad} extra on the flickering source, {false_ok} (noise) on the clean one'))
results.append((bool(Na[:, NA_GAP_SLICES[0]:NA_GAP_SLICES[1]].all()) and not bool(Na[:, :NA_GAP_SLICES[0]].any()), 'Na-gap mask covers exactly the gap'))

# ---- spectral_outlier_scores: a spike in one slice of the peak-pixel spectrum
spectra = torch.tensor(np.stack([roi_ok[:, 18, 18], roi_ok[:, 10, 10]]), dtype=torch.float32)
spectra[0, 700] += 3000.0
variance = torch.full_like(spectra, 20.0**2)
scores = spectral_outlier_scores(spectra, variance, half_window=11, exclude_radius=1)
z, c, s = scores['z_residual'], scores['z_curvature'], scores['support']
results.append((float(z[0, 700]) > 12 and float(c[0, 700]) > 8 and int(s[0, 700]) <= 2,
                f'spectral_outlier_scores: injected spike z={float(z[0, 700]):.0f}, curvature z={float(c[0, 700]):.0f}, support={int(s[0, 700])}'))
clean = z[1, scores['valid_range']]
results.append((float(clean.abs().nanmedian()) < 1.5 and float(clean.abs().max()) < 6,
                f'spectral_outlier_scores: clean pixel |z| median {float(clean.abs().nanmedian()):.2f}, max {float(clean.abs().max()):.1f}'))

# ---- peak_variability_excess on the binned representation of the two sources
edges = np.arange(0, N_wvl + 1, 100)
n_slices = np.diff(edges)
def binned(roi):
    cube = np.stack([roi[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    std  = np.stack([roi[a:b].std(0) for a, b in zip(edges[:-1], edges[1:])])
    return cube, std
ex_bad = peak_variability_excess(*binned(roi_bad), n_slices)['excess']
ex_ok  = peak_variability_excess(*binned(roi_ok),  n_slices)['excess']
results.append((ex_bad.max() > 2.0 and ex_ok.max() < 1.6,
                f'peak_variability_excess: flickering source max excess {ex_bad.max():.1f}, clean source {ex_ok.max():.2f}'))

for ok, msg in results:
    print(f'[{"PASS" if ok else "FAIL"}] {msg}')
print('[PASS] all checks passed' if all(ok for ok, _ in results) else '[FAIL] some checks failed')
