#%%
"""
Detect MUSE NFM STD samples recorded with a DM correction radius that differs from the model's k_c.

Some STD stars were observed with fewer DM modes corrected, which moves the cut-off frequency k_c, while the PSF
model uses the nominal k_c = 1 / (2 * DmPitch) for every sample.  In the residual (data - fit) this shows up as an
azimuthally coherent ring just inside the nominal correction radius r_c(λ) = λ k_c / pixel_scale, located at the
same r / r_c(λ) for every wavelength bin. The ring position is a proxy of the real cut-off: k_c(real) ≈ u_ring * k_c.

For each fitted sample (``STD_FOLDER / 'fitted' / '<id>.pickle'``) the routine:
  1. builds the relative residual image (data - fit) / <fit>_azimuthal(r) for every wavelength bin,
  2. takes the azimuthal *median* in bins of u = r / r_c(λ) (the median suppresses streaks and the tomographic cross),
  3. stacks the profiles over wavelength (median) and measures the bump above the baseline in u ∈ [0.76, 1.08],
  4. measures the angular coherence (fraction of 16 sectors with positive residual at the ring radius) and the
     fraction of wavelength bins reproducing the bump,
  5. classifies the sample as 'strong', 'weak' or 'none' and refines the ring position (k_c fraction) to sub-bin precision.

Dataset preparation entry point: ``update_kc_mismatch_labels(muse_df)`` (called from ``create_STD_dataset.py`` once the fits exist).
It scans only the fitted samples that are not yet in ``STD_FOLDER / 'kc_ring_scores.csv'`` (keyed by MUSE filename), appends their
scores, and appends the newly detected 'strong' / 'weak' samples to ``STD_FOLDER / 'kc_mismatch_labels.txt'`` together with their
k_c percentage. Entries already in the labels file (including the manual ones) are never removed automatically; their k_c
percentage is refreshed when the sample is re-scanned. ``add_kc_mismatch_flag`` (STD_dataset_utils) turns the file into the
``'k_c mismatch'`` / ``'k_c fraction'`` columns of ``muse_df``.

Run as a script to (re)scan everything:  python datasets/MUSE_STD_dataset/detect_kc_mismatch.py [--force]
"""
import sys
sys.path.append('..')
sys.path.append('../..')

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import map_coordinates

from STD_dataset_utils import STD_FOLDER, KC_MISMATCH_FLAG, read_kc_mismatch_labels, write_kc_mismatch_labels, add_kc_mismatch_flag

FITTED_FOLDER = STD_FOLDER / 'fitted'
PLOTS_FOLDER  = STD_FOLDER / 'plots'
SCORES_FILE   = STD_FOLDER / 'kc_ring_scores.csv'
LABELS_FILE   = STD_FOLDER / 'kc_mismatch_labels.txt'

PIX_MAS    = 25.0    # MUSE NFM pixel scale [mas]
DM_PITCH   = 0.22    # [m], must match the PSF model's DmPitchs
RAD_PER_PX = PIX_MAS * 1e-3 / 206265.0

U_EDGES   = np.arange(0.30, 1.62, 0.04)
U         = 0.5 * (U_EDGES[1:] + U_EDGES[:-1])
DU        = U[1] - U[0]
RING_BAND = (U > 0.76) & (U < 1.08)
BASE_BAND = ((U > 0.42) & (U < 0.68)) | ((U > 1.16) & (U < 1.56))
N_SECTORS = 16
STACK_RC  = 20       # r_c in pixels of the wavelength-stacked ring map
STACK_HALF = 40

# Classification thresholds (set by visual inspection of the galleries of the 2026-10 scan)
STRONG = dict(amp=0.12, coherence=0.85, lambda_frac=0.70, snr=5.0)
WEAK   = dict(amp=0.06, coherence=0.80, lambda_frac=0.70, snr=3.0, u_min=0.90)

SCORE_COLUMNS = ['id', 'time', 'Filename', 'ring_class', 'kc_fraction', 'ring_amp', 'ring_u', 'coherence', 'lambda_frac', 'snr', 'fit loss']


def azimuthal_mean(img, rr, r_max=70, step=1.0):
    bins = np.arange(0, r_max + step, step)
    idx = np.digitize(rr.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    s = np.bincount(idx[ok], weights=img.ravel()[ok], minlength=len(bins) - 1)
    n = np.bincount(idx[ok], minlength=len(bins) - 1)
    return 0.5 * (bins[1:] + bins[:-1]), s / np.maximum(n, 1)


def ring_profiles(data, fit, wavelengths):
    """data, fit: [N_wvl, H, W]. Returns azimuthal-median relative residual profiles [N_wvl, N_u],
    sector coherence [N_wvl, N_u] and the wavelength-stacked ring map [2*STACK_HALF+1]^2 (r_c = STACK_RC px)."""
    L, H, W = data.shape
    r_c = wavelengths / (2 * DM_PITCH) / RAD_PER_PX   # [px]
    prof = np.full((L, len(U)), np.nan)
    coh  = np.full((L, len(U)), np.nan)
    stack = np.zeros((2 * STACK_HALF + 1, 2 * STACK_HALF + 1))
    g = np.arange(-STACK_HALF, STACK_HALF + 1)
    gy, gx = np.meshgrid(g, g, indexing='ij')
    y, x = np.indices((H, W))

    for l in range(L):
        cy, cx = np.unravel_index(np.argmax(fit[l]), fit[l].shape)
        rr = np.hypot(y - cy, x - cx)
        th = np.arctan2(y - cy, x - cx)
        r_ax, fit_prof = azimuthal_mean(fit[l], rr)
        rel = (data[l] - fit[l]) / np.maximum(np.interp(rr, r_ax, fit_prof), 1e-12)
        rel = np.clip(rel, -3, 3)

        ub = np.digitize((rr / r_c[l]).ravel(), U_EDGES) - 1
        sb = ((th.ravel() + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
        relv = rel.ravel()
        for b in range(len(U)):
            m = ub == b
            if m.sum() < 4:
                continue
            prof[l, b] = np.median(relv[m])
            sect = [np.mean(relv[m & (sb == s)]) for s in range(N_SECTORS) if np.any(m & (sb == s))]
            coh[l, b] = np.mean(np.array(sect) > 0)

        sc = r_c[l] / STACK_RC
        stack += map_coordinates(rel, [cy + gy * sc, cx + gx * sc], order=1, mode='nearest')

    return prof, coh, stack / L


def ring_score(prof, coh):
    L = prof.shape[0]
    P = np.nanmedian(prof, axis=0)
    base = np.nanmedian(P[BASE_BAND])
    sigma = np.nanstd(P[BASE_BAND]) + 1e-6
    Pb = P - base
    k = np.flatnonzero(RING_BAND)[np.nanargmax(Pb[RING_BAND])]
    win = slice(max(k - 1, 0), k + 2)
    per_l = np.array([np.nanmax(prof[l, win]) - np.nanmedian(prof[l, BASE_BAND]) for l in range(L)])
    A = Pb[k]

    # Sub-bin ring position (parabolic interpolation of the peak): proxy of k_c(real) / k_c(model)
    u_ring = U[k]
    if 0 < k < len(U) - 1 and np.all(np.isfinite(Pb[k-1:k+2])):
        denom = Pb[k-1] - 2 * Pb[k] + Pb[k+1]
        if denom < 0:
            u_ring = U[k] + 0.5 * (Pb[k-1] - Pb[k+1]) / denom * DU

    return dict(
        ring_amp    = A,
        ring_u      = U[k],
        kc_fraction = float(u_ring),
        coherence   = np.nanmedian(coh[:, win]),
        lambda_frac = float(np.mean(per_l > 0.5 * A)) if A > 0 else 0.0,
        snr         = A / sigma,
    )


def classify(tab):
    s = (tab.ring_amp >= STRONG['amp']) & (tab.coherence >= STRONG['coherence']) & \
        (tab.lambda_frac >= STRONG['lambda_frac']) & (tab.snr >= STRONG['snr'])
    w = ~s & (tab.ring_amp >= WEAK['amp']) & (tab.coherence >= WEAK['coherence']) & \
        (tab.lambda_frac >= WEAK['lambda_frac']) & (tab.snr >= WEAK['snr']) & (tab.ring_u >= WEAK['u_min'])
    return np.where(s, 'strong', np.where(w, 'weak', 'none'))


def scan(ids, fitted_folder=FITTED_FOLDER, verbose=True):
    """Scores the fitted samples with the given ids. Returns a DataFrame indexed by id and a dict of ring maps."""
    rows, stacks = [], {}
    for i in ids:
        p = Path(fitted_folder) / f'{i}.pickle'
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            d = pickle.load(f)
        data, fit = d['Img. data'][0], d['Img. fit'][0]
        if np.isnan(data).any() or np.isnan(fit).any():
            if verbose: print(f'Sample {i}: NaNs in the images, skipped')
            continue
        prof, coh, stack = ring_profiles(data, fit, d['wavelengths'])
        s = ring_score(prof, coh)
        s['id'] = int(i)
        s['fit loss'] = d['loss']
        rows.append(s)
        stacks[int(i)] = stack.astype(np.float32)
        if verbose:
            print(f"Sample {i}: amp={s['ring_amp']:.3f} k_c={100*s['kc_fraction']:.1f}% coh={s['coherence']:.2f} snr={s['snr']:.1f}", flush=True)

    if len(rows) == 0:
        return pd.DataFrame(columns=[c for c in SCORE_COLUMNS if c not in ('time', 'Filename')]).set_index('id'), stacks

    tab = pd.DataFrame(rows).set_index('id')
    tab['ring_class'] = classify(tab)
    return tab, stacks


def fitted_ids(fitted_folder=FITTED_FOLDER):
    return sorted(int(p.stem) for p in Path(fitted_folder).glob('*.pickle') if p.stem.isdigit())


def load_scores(scores_file=SCORES_FILE):
    if Path(scores_file).exists():
        return pd.read_csv(scores_file, index_col='id')
    return pd.DataFrame(columns=[c for c in SCORE_COLUMNS if c != 'id']).rename_axis('id')


def update_kc_mismatch_labels(muse_df, fitted_folder=FITTED_FOLDER, scores_file=SCORES_FILE, labels_file=LABELS_FILE,
                              force=False, plot=True, verbose=True):
    """
    Incremental k_c-mismatch detection for the STD-stars dataset preparation.

    Scans the fitted samples whose MUSE filename is not yet in `scores_file` (all of them if `force=True`), appends their
    scores there and appends the newly detected 'strong' / 'weak' samples to `labels_file` with their k_c percentage.
    Existing label entries are kept (manual entries included); their k_c percentage is refreshed if re-scanned, the class too unless
    it was set by hand ('manual' = mismatch, 'none' = checked and vetoed, never re-added and excluded from the flag).
    Returns the updated scores DataFrame (indexed by id) and the labels dict {Filename: entry}.
    """
    filenames = muse_df['Filename']
    scores = load_scores(scores_file)
    labels = read_kc_mismatch_labels(labels_file) if Path(labels_file).exists() else {}

    ids_all = [i for i in fitted_ids(fitted_folder) if i in filenames.index]
    if force:
        ids_new = ids_all
    else:
        scanned = set(scores['Filename'].dropna()) if len(scores) else set()
        ids_new = [i for i in ids_all if filenames.loc[i] not in scanned]

    if verbose:
        print(f'k_c mismatch detection: {len(ids_all)} fitted samples, {len(ids_new)} to scan, {len(labels)} already labeled')
    if len(ids_new) == 0:
        return scores, labels

    new, stacks = scan(ids_new, fitted_folder, verbose=verbose)
    new = new.join(muse_df[['time', 'Filename']], how='left')
    new = new[[c for c in SCORE_COLUMNS if c != 'id']]

    # Merge by filename (robust to re-indexing), the new scan wins
    if len(scores):
        scores = scores[~scores['Filename'].isin(new['Filename'])]
    scores = pd.concat([scores, new]).sort_values('time')
    scores.to_csv(scores_file)

    # Append / refresh the labels
    n_added = 0
    for i, r in new.iterrows():
        entry = dict(date=str(r['time'])[:10], kc_percent=100 * r['kc_fraction'], kc_source='ring', ring_class=r['ring_class'], amp=r['ring_amp'])
        if r['Filename'] in labels:
            labels[r['Filename']]['amp'] = entry['amp']
            if labels[r['Filename']].get('kc_source') != 'fit': # a fitted k_c (scan_kc_fraction.py) is kept
                labels[r['Filename']].update(kc_percent=entry['kc_percent'], kc_source='ring')
            if labels[r['Filename']].get('ring_class') not in ('manual', 'none'): # hand-set classes are kept
                labels[r['Filename']]['ring_class'] = entry['ring_class']
        elif r['ring_class'] in ('strong', 'weak'):
            labels[r['Filename']] = entry
            n_added += 1
            if verbose: print(f"  New k_c-mismatch sample: id {i} {r['Filename']} ({r['ring_class']}, k_c = {entry['kc_percent']:.1f} %)")
    write_kc_mismatch_labels(labels, labels_file)
    if verbose:
        print(f'k_c mismatch labels: {n_added} added, {len(labels)} total -> {labels_file}')

    if plot and len(stacks):
        PLOTS_FOLDER.mkdir(exist_ok=True)
        gallery(new.join(muse_df[[KC_MISMATCH_FLAG]], how='left') if KC_MISMATCH_FLAG in muse_df else new, stacks,
                PLOTS_FOLDER / ('kc_ring_gallery.png' if force else 'kc_ring_gallery_new.png'))
    return scores, labels


def gallery(tab, stacks, path, ncol=10, flagged_only=True):
    """Wavelength-stacked ring maps (dashed circle = nominal r_c) of the flagged (or all) scanned samples, sorted by time."""
    import matplotlib.pyplot as plt
    tab = tab.sort_values('time') if 'time' in tab else tab
    sel = [i for i in tab.index if i in stacks and (not flagged_only or tab.loc[i, 'ring_class'] != 'none' or bool(tab.loc[i].get(KC_MISMATCH_FLAG, False)))]
    if len(sel) == 0:
        return
    nrow = max(1, int(np.ceil(len(sel) / ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for k, i in enumerate(sel):
        r = tab.loc[i]
        axs[k].imshow(stacks[i], vmin=-0.5, vmax=0.5, cmap='RdBu_r')
        axs[k].add_patch(plt.Circle((STACK_HALF, STACK_HALF), STACK_RC, fill=False, color='k', ls='--', lw=0.7))
        axs[k].add_patch(plt.Circle((STACK_HALF, STACK_HALF), STACK_RC * r.kc_fraction, fill=False, color='g', ls=':', lw=0.7))
        axs[k].axis('off')
        axs[k].set_title(f"{i} {str(r.ring_class)[0].upper()} A={r.ring_amp:.2f} k_c={100*r.kc_fraction:.0f}%\n{str(r.get('time', ''))[:10]}",
                         fontsize=7, color='red' if r.ring_class == 'strong' else 'black')
    for a in axs[len(sel):]:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


#%%
if __name__ == '__main__':
    force = '--force' in sys.argv
    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)

    scores, labels = update_kc_mismatch_labels(muse_df, force=force)
    muse_df = add_kc_mismatch_flag(muse_df)
    with open(STD_FOLDER / 'muse_df.pickle', 'wb') as handle:
        pickle.dump(muse_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nClass counts (all scanned):'); print(scores.ring_class.value_counts().to_string())
    flagged = scores[scores['Filename'].isin(labels)]
    print(f'\nLabeled k_c-mismatch samples ({len(flagged)}):')
    print(flagged[['time', 'Filename', 'ring_class', 'kc_fraction', 'ring_amp', 'coherence', 'snr']].to_string())
