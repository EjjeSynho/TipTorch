#%%
"""
MUSE NFM STD stars recorded with a DM correction radius (k_c) different from the model's: detection, labels and k_c refinement.

Some STD stars were observed with fewer DM modes corrected, so their cut-off frequency k_c is below the nominal k_c = 1 / (2 DmPitch)
the PSF model uses. In the residuals (data - fit) this is an azimuthally coherent ring just inside the nominal correction radius
r_c(λ) = λ k_c / pixel scale, at the same r / r_c(λ) for every wavelength bin.

Cells (run them one by one, or the whole file as a script):
  1. settings                 paths, thresholds, scan factors; CLI: python kc_mismatch.py [--force] [--scan] [--ids ...] [--devices ...]
  2. ring diagnostics         ring_profiles / ring_score: relative residual (data - fit) / <fit>(r), azimuthal median in u = r / r_c(λ)
                              (suppresses streaks and the tomographic cross), bump above baseline, angular coherence, λ consistency,
                              sub-bin ring position = first k_c estimate
  3. detection                update_kc_mismatch_labels(muse_df): scores the fits not yet in kc_ring_scores.csv (keyed by MUSE filename),
                              appends the 'strong' / 'weak' detections to kc_mismatch_labels.txt as "k_c <p> % (ring)"; called from
                              create_STD_dataset.py once the fits exist
  4. gallery                  wavelength-stacked ring maps of the flagged samples
  5. k_c fit scan             scan_kc_by_fits(ids): the ring position is biased high by a few % (the fit compensates with r0 / dn), so the
                              labeled samples are re-fitted (STD_fitter.py subprocesses, kc_fraction=<v>) at ring estimate x factors; the
                              parabolic loss minimum goes to the labels as "k_c <p> % (fit)" (never overwritten by the detector); a fitted
                              k_c at the nominal value sets the 'none' class (not a mismatch)
  6. run                      detection -> labels -> [scan] -> 'k_c mismatch' / 'k_c fraction' / 'k_c group' columns of muse_df.pickle

Labels file: one "<Filename>: k_c <p> % (ring|fit), <class>, amp <v>, <date>" per line; classes strong / weak (automatic), manual (added by
hand), none (vetoed by hand or by the fit: excluded, never re-added). Entries are never removed automatically. See read_kc_mismatch_labels /
add_kc_mismatch_flag in STD_dataset_utils.py for the parsing, the grouping by k_c and the muse_df columns.
"""

import sys
import pickle
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from STD_dataset_utils import (
    STD_FOLDER,
    KC_MISMATCH_FLAG,
    KC_FRACTION_FEATURE,
    read_kc_mismatch_labels,
    write_kc_mismatch_labels,
    add_kc_mismatch_flag
)

SCRIPT_DIR    = Path(__file__).resolve().parent
FITTED_FOLDER = STD_FOLDER / 'fitted'
SCAN_FOLDER   = STD_FOLDER / 'temp/fitted_kc_scan'      # fits of the k_c scan: <id>/kc_<fraction>/<id>.pickle ('_raw' suffix for raw-frame fits, see scan_folder)
PLOTS_FOLDER  = STD_FOLDER / 'plots'
SCORES_FILE   = STD_FOLDER / 'temp/kc_ring_scores.csv'  # ring diagnostics of every scanned fit (the ring estimate of k_c lives here)
SCAN_FILE     = STD_FOLDER / 'temp/kc_scan_results.csv' # loss per scanned k_c, fitted k_c
LABELS_FILE   = STD_FOLDER / 'kc_mismatch_labels.txt'

PIX_MAS, DM_PITCH = 25.0, 0.22                       # MUSE NFM pixel scale [mas], nominal DM pitch [m] (InitNFMConfig)
RAD_PER_PX = PIX_MAS * 1e-3 / 206265.0

U_EDGES   = np.arange(0.30, 1.62, 0.04)              # bins of u = r / r_c(λ)
U         = 0.5 * (U_EDGES[1:] + U_EDGES[:-1])
RING_BAND = (U > 0.76) & (U < 1.08)
BASE_BAND = ((U > 0.42) & (U < 0.68)) | ((U > 1.16) & (U < 1.56))
N_SECTORS = 16
STACK_RC, STACK_HALF = 20, 40                        # wavelength-stacked ring map: r_c in px, half size in px

STRONG = dict(amp=0.12, coherence=0.85, lambda_frac=0.70, snr=5.0)             # thresholds set by eye on the 2026-10 galleries
WEAK   = dict(amp=0.06, coherence=0.80, lambda_frac=0.70, snr=3.0, u_min=0.90) # u_min rejects the cross / streak bumps at u <= 0.88

SCAN_FACTORS      = (0.86, 0.90, 0.94, 0.98, 1.02)   # scanned k_c = ring estimate x factor
NO_MISMATCH_ABOVE = 0.99                             # fitted k_c >= 99 % of the nominal: not a mismatch -> 'none' class


#%% ------------------------------------------------ ring diagnostics ------------------------------------------------

def ring_profiles(data, fit, wavelengths):    
    """data, fit: [N_wvl, H, W]. Returns the azimuthal-median relative residual profiles [N_wvl, N_u], the fraction of positive
    angular sectors [N_wvl, N_u] and the wavelength-stacked ring map [2 STACK_HALF + 1]^2 with r_c = STACK_RC px."""
    
    def _azimuthal_mean(img, rr, r_max=70):
        idx = np.clip(rr.ravel().astype(int), 0, r_max)
        return np.arange(r_max + 1) + 0.5, np.bincount(idx, weights=img.ravel(), minlength=r_max + 1) / np.maximum(np.bincount(idx, minlength=r_max + 1), 1)
    
    L, H, W = data.shape
    r_c = wavelengths / (2 * DM_PITCH) / RAD_PER_PX # [px]
    prof, coh = np.full((L, len(U)), np.nan), np.full((L, len(U)), np.nan)
    stack = np.zeros((2 * STACK_HALF + 1,) * 2)
    gy, gx = np.meshgrid(*(np.arange(-STACK_HALF, STACK_HALF + 1),) * 2, indexing='ij')
    y, x = np.indices((H, W))
    
    for l in range(L):
        cy, cx = np.unravel_index(np.argmax(fit[l]), fit[l].shape)
        rr, th = np.hypot(y - cy, x - cx), np.arctan2(y - cy, x - cx)
        rel = np.clip((data[l] - fit[l]) / np.maximum(np.interp(rr, *_azimuthal_mean(fit[l], rr)), 1e-12), -3, 3).ravel()
        ub = np.digitize((rr / r_c[l]).ravel(), U_EDGES) - 1
        sb = ((th.ravel() + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
        for b in range(len(U)):
            m = ub == b
            if m.sum() >= 4:
                prof[l, b] = np.median(rel[m])
                coh[l, b]  = np.mean([np.mean(rel[m & (sb == s)]) > 0 for s in range(N_SECTORS) if np.any(m & (sb == s))])
        sc = r_c[l] / STACK_RC
        stack += map_coordinates(rel.reshape(H, W), [cy + gy * sc, cx + gx * sc], order=1, mode='nearest')
    
    return prof, coh, stack / L


def ring_score(prof, coh):
    """Ring amplitude above the baseline, its position (bin and sub-bin -> kc_fraction), angular coherence, λ consistency and SNR."""
    P = np.nanmedian(prof, axis=0)
    Pb, sigma = P - np.nanmedian(P[BASE_BAND]), np.nanstd(P[BASE_BAND]) + 1e-6
    k = np.flatnonzero(RING_BAND)[np.nanargmax(Pb[RING_BAND])]
    win = slice(max(k - 1, 0), k + 2)
    
    A = Pb[k]
    per_l = np.nanmax(prof[:, win], axis=1) - np.nanmedian(prof[:, BASE_BAND], axis=1)
    u_ring, d2 = U[k], Pb[k-1] - 2 * Pb[k] + Pb[k+1] if 0 < k < len(U) - 1 else np.nan
    
    if np.isfinite(d2) and d2 < 0:
        u_ring += 0.5 * (Pb[k-1] - Pb[k+1]) / d2 * (U[1] - U[0])
    
    return dict(ring_amp=A, ring_u=U[k], kc_fraction=float(u_ring), coherence=np.nanmedian(coh[:, win]),
                lambda_frac=float(np.mean(per_l > 0.5 * A)) if A > 0 else 0.0, snr=A / sigma)


def load_fit(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def score_fits(paths, verbose=True):
    """Ring diagnostics of fit pickles {id: path}. Returns a DataFrame indexed by id and the ring maps {id: map}."""
    
    def _classify(tab):
        s = (tab.ring_amp >= STRONG['amp']) & (tab.coherence >= STRONG['coherence']) & (tab.lambda_frac >= STRONG['lambda_frac']) & (tab.snr >= STRONG['snr'])
        w = ~s & (tab.ring_amp >= WEAK['amp']) & (tab.coherence >= WEAK['coherence']) & (tab.lambda_frac >= WEAK['lambda_frac']) & (tab.snr >= WEAK['snr']) & (tab.ring_u >= WEAK['u_min'])
        return np.where(s, 'strong', np.where(w, 'weak', 'none'))
      
    rows, stacks = [], {}
    
    for i, p in paths.items():
        d = load_fit(p)
        data, fit = d['Img. data'][0], d['Img. fit'][0]
        
        if np.isnan(data).any() or np.isnan(fit).any():
            continue
        
        prof, coh, stacks[i] = ring_profiles(data, fit, d['wavelengths'])
        rows.append(dict(id=i, **ring_score(prof, coh), fit_loss=d['loss']))
       
        if verbose:
            print(f"Sample {i}: ring amp {rows[-1]['ring_amp']:+.3f} at k_c {100 * rows[-1]['kc_fraction']:.1f} %, coherence {rows[-1]['coherence']:.2f}", flush=True)
    
    tab = pd.DataFrame(rows).set_index('id') if rows else pd.DataFrame(columns=['ring_amp', 'ring_u', 'kc_fraction', 'coherence', 'lambda_frac', 'snr', 'fit_loss']).rename_axis('id')
    tab['ring_class'] = _classify(tab) if len(tab) else []
    
    return tab, stacks


#%% --------------------------------------------------- detection ----------------------------------------------------

def save_muse_df(muse_df):
    with open(STD_FOLDER / 'muse_df.pickle', 'wb') as f:
        pickle.dump(muse_df, f, protocol=pickle.HIGHEST_PROTOCOL)


def update_labels(labels, entries, source):
    """entries: DataFrame indexed by id with Filename, time, kc_fraction (+ ring_class, ring_amp for the ring source).
    Ring source: new strong / weak detections are added, existing entries get the new ring estimate unless they hold a fitted one;
    hand-set classes (manual / none) are kept. Fit source: the k_c of every entry is set, a k_c at the nominal value vetoes it."""
    added, vetoed = [], []
   
    for i, r in entries.iterrows():
        e = labels.get(r['Filename'])
        if source == 'ring':
            if e is None:
                if r['ring_class'] in ('strong', 'weak'):
                    labels[r['Filename']] = dict(date=str(r['time'])[:10], kc_percent=100 * r['kc_fraction'], kc_source='ring', ring_class=r['ring_class'], amp=r['ring_amp'])
                    added.append(i)
            else:
                e['amp'] = r['ring_amp']
                
                if e.get('kc_source') != 'fit':
                    e.update(kc_percent=100 * r['kc_fraction'], kc_source='ring')
            
                if e.get('ring_class') not in ('manual', 'none'):
                    e['ring_class'] = r['ring_class']
        else:
            if e is None:
                e = labels[r['Filename']] = dict(date=str(r['time'])[:10], ring_class='manual', amp=None)
                
            e.update(kc_percent=100 * min(r['kc_fraction'], 1.0), kc_source='fit')
            
            if r['kc_fraction'] >= NO_MISMATCH_ABOVE:
                e['ring_class'] = 'none'; vetoed.append(i)
   
    return added, vetoed


def update_kc_mismatch_labels(muse_df, force=False, plot=True, verbose=True):
    """Incremental detection: scores the fits whose MUSE filename is not in SCORES_FILE (all with force), updates the labels file.
    Returns (scores of all scanned fits, labels dict)."""
    scores = pd.read_csv(SCORES_FILE, index_col='id') if SCORES_FILE.exists() else pd.DataFrame(columns=['Filename'])
    labels = read_kc_mismatch_labels(LABELS_FILE) if LABELS_FILE.exists() else {}
    ids = [int(p.stem) for p in FITTED_FOLDER.glob('*.pickle') if p.stem.isdigit() and int(p.stem) in muse_df.index]
    scanned = set() if force else set(scores['Filename'])
    ids = sorted(i for i in ids if muse_df.loc[i, 'Filename'] not in scanned)
    
    if verbose:
        print(f'k_c mismatch detection: {len(ids)} fits to score, {len(labels)} samples labeled')
    
    if not ids:
        return scores, labels

    new, stacks = score_fits({i: FITTED_FOLDER / f'{i}.pickle' for i in ids}, verbose)
    new = muse_df.loc[new.index, ['time', 'Filename']].join(new)
    scores = pd.concat([scores[~scores['Filename'].isin(new['Filename'])], new]).sort_values('time')
    scores.to_csv(SCORES_FILE)

    added, _ = update_labels(labels, new, 'ring')
    write_kc_mismatch_labels(labels, LABELS_FILE)
    
    if verbose:
        for i in added:
            print(f"  new k_c-mismatch sample: id {i} {new.loc[i, 'Filename']} ({new.loc[i, 'ring_class']}, ring at k_c {100 * new.loc[i, 'kc_fraction']:.1f} %)")
        print(f'k_c mismatch labels: {len(added)} added, {len(labels)} total -> {LABELS_FILE}')
   
    if plot:
        sel = [i for i in new.index if new.loc[i, 'Filename'] in labels and labels[new.loc[i, 'Filename']].get('ring_class') != 'none']
        gallery(new.loc[sel], stacks, PLOTS_FOLDER / ('kc_ring_gallery.png' if force else 'kc_ring_gallery_new.png'))
    
    return scores, labels


#%% ---------------------------------------------------- gallery -----------------------------------------------------
def gallery(tab, stacks, path, ncol=10):
    """Wavelength-stacked ring maps (dashed: nominal r_c, dotted: ring estimate) of the samples of tab, sorted by time."""
    import matplotlib.pyplot as plt
    tab = tab.sort_values('time')
    
    if len(tab) == 0:
        return
    
    nrow = int(np.ceil(len(tab) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow))
    axs = np.atleast_1d(axs).ravel()
    
    for ax, (i, r) in zip(axs, tab.iterrows()):
        ax.imshow(stacks[i], vmin=-0.5, vmax=0.5, cmap='RdBu_r')
        ax.add_patch(plt.Circle((STACK_HALF, STACK_HALF), STACK_RC, fill=False, color='k', ls='--', lw=0.7))
        ax.add_patch(plt.Circle((STACK_HALF, STACK_HALF), STACK_RC * r.kc_fraction, fill=False, color='g', ls=':', lw=0.7))
        ax.set_title(f"{i} {str(r.ring_class)[0].upper()} A={r.ring_amp:.2f} k_c={100 * r.kc_fraction:.0f}%\n{str(r.time)[:10]}", fontsize=7, color='red' if r.ring_class == 'strong' else 'black')
   
    for ax in axs:
        ax.axis('off')
        
    path.parent.mkdir(exist_ok=True)
   
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


#%% -------------------------------------------------- k_c fit scan --------------------------------------------------
def scan_folder():
    """The scan fits follow the fitter's regime (STD_fitter.DEROTATE_PSF), like the dataset cache: derotated PSFs or raw frames."""
    from STD_fitter import DEROTATE_PSF
    return SCAN_FOLDER if DEROTATE_PSF else SCAN_FOLDER.with_name(SCAN_FOLDER.name + '_raw')


def fit_with_kc(id_, kc_fraction, device):
    """STD_fitter.load_and_fit_sample(id, device, kc_fraction) in a subprocess (one GPU context per fit, several run in parallel);
    returns the fit pickle path, reused if it exists."""
    out = scan_folder() / str(id_) / f'kc_{kc_fraction:.4f}' / f'{id_}.pickle'
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        code = chr(10).join([
            f"import sys, pickle; sys.path[:0] = [{str(SCRIPT_DIR)!r}, {str(SCRIPT_DIR.parent.parent)!r}]",
            "from STD_fitter import load_and_fit_sample",
            f"pickle.dump(load_and_fit_sample({id_}, {device!r}, {kc_fraction}), open({str(out)!r}, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)",
        ])
        with open(out.parent / f'fit_{id_}.log', 'w') as log:
            subprocess.run([sys.executable, '-c', code], stdout=log, stderr=subprocess.STDOUT)
        print(f'  fit id {id_} at k_c {100 * kc_fraction:.1f} %: {"ok" if out.exists() else "FAILED"}', flush=True)
    return out


def parabolic_min(x, y):
    """Minimum of the parabola through the three points around the lowest y; the lowest point itself at the edges (flagged)."""
    k = int(np.argmin(y))
    if 0 < k < len(x) - 1 and (a := np.polyfit(x[k-1:k+2], y[k-1:k+2], 2))[0] > 0:
        return float(np.clip(-a[1] / (2 * a[0]), x[k-1], x[k+1])), ''
    
    return float(x[k]), 'low' if k == 0 else 'high'


def scan_kc_by_fits(muse_df, ids=None, factors=SCAN_FACTORS, devices=('cuda:0',), n_jobs=6, apply=True, verbose=True):
    """Fits the labeled samples (ids: default all flagged, fittable) at k_c = ring estimate x factors in parallel, takes the loss minimum
    as the sample's k_c, stores the results in SCAN_FILE and (apply) writes the fitted k_c to the labels and refreshes muse_df."""
    labels = read_kc_mismatch_labels(LABELS_FILE)
    scores = pd.read_csv(SCORES_FILE).set_index('Filename') if SCORES_FILE.exists() else pd.DataFrame()
    
    if ids is None:
        ids = muse_df.index[muse_df[KC_MISMATCH_FLAG] & ~muse_df['Corrupted'] & ~muse_df['Bad quality']].tolist()
    
    # The scan base is the ring estimate (SCORES_FILE); samples labeled by hand without one use their muse_df (group) k_c
    base = {i: float(scores.loc[muse_df.loc[i, 'Filename'], 'kc_fraction']) if muse_df.loc[i, 'Filename'] in scores.index else float(muse_df.loc[i, KC_FRACTION_FEATURE]) for i in ids}
    jobs = [(i, round(base[i] * f, 4), devices[n % len(devices)]) for n, (i, f) in enumerate((i, f) for i in ids for f in factors)]
   
    if verbose:
        print(f'k_c scan: {len(ids)} samples x {len(factors)} factors on {list(devices)}, {n_jobs} parallel fits (~2 min each, existing ones reused)')
   
    with ThreadPoolExecutor(n_jobs) as pool:
        list(pool.map(lambda j: fit_with_kc(*j), jobs))

    rows = []
    for i in ids: # all the scanned fractions of the sample, also from previous scans
        fits = {float(d.name[3:]): d / f'{i}.pickle' for d in sorted((scan_folder() / str(i)).glob('kc_*')) if (d / f'{i}.pickle').exists()}
        if len(fits) < 2:
            continue
        
        fracs = np.array(sorted(fits)); losses = np.array([load_fit(fits[f])['loss'] for f in fracs])
        best, edge = parabolic_min(fracs, losses)
        nominal = FITTED_FOLDER / f'{i}.pickle'
        
        rows.append(
            dict(
                id=i,
                Filename=muse_df.loc[i, 'Filename'],
                time=str(muse_df.loc[i, 'time'])[:10],
                ring_estimate=base[i],
                kc_fraction=best,
                at_scan_edge=edge,
                loss_min=losses.min(),
                loss_nominal_kc=load_fit(nominal)['loss'] if nominal.exists() else np.nan,
                scanned=' '.join(f'{f:.3f}:{l:.4f}' for f, l in zip(fracs, losses))
            )
        )

        if verbose:
            print(f"  id {i}: ring estimate {100 * base[i]:.1f} % -> fitted k_c {100 * best:.1f} %{f' ({edge} scan edge!)' if edge else ''}, loss {losses.min():.4f}")
    
    res = pd.DataFrame(rows).set_index('id')
    
    if SCAN_FILE.exists(): # keep the samples not scanned this time
        prev = pd.read_csv(SCAN_FILE, index_col='id')
        res = pd.concat([prev[~prev.index.isin(res.index)], res]).sort_index()
    
    res.to_csv(SCAN_FILE)

    if apply and len(res):
        _, vetoed = update_labels(labels, res.loc[[i for i in ids if i in res.index]], 'fit')
        write_kc_mismatch_labels(labels, LABELS_FILE)
        if vetoed:
            print(f"Fitted k_c >= {100 * NO_MISMATCH_ABOVE:.0f} % of the nominal: ids {vetoed} set to the 'none' class (not a k_c mismatch; edit the labels file to override)")
        save_muse_df(add_kc_mismatch_flag(muse_df, LABELS_FILE, verbose=verbose))
    return res


#%% ------------------------------------------------------ run -------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='k_c mismatch detection and labels of the MUSE NFM STD stars')
    parser.add_argument('--force',    action='store_true', help='re-score every fit (default: only the fits not scored yet)')
    parser.add_argument('--scan',     action='store_true', help='refine the k_c of the labeled samples by fits after the detection')
    parser.add_argument('--ids',      type=int, nargs='*', default=None, help='samples of the fit scan (default: all labeled)')
    parser.add_argument('--factors',  type=float, nargs='+', default=list(SCAN_FACTORS), help='scanned k_c = ring estimate x factor')
    parser.add_argument('--devices',  type=str, nargs='+', default=['cuda:0'])
    parser.add_argument('--n_jobs',   type=int, default=6)
    parser.add_argument('--no_apply', action='store_true', help='scan without touching the labels / muse_df')
    args, _ = parser.parse_known_args()

    #%% detection: scores, labels, muse_df columns
    muse_df =  pd.read_pickle(STD_FOLDER / 'muse_df.pickle')

    scores, labels = update_kc_mismatch_labels(muse_df, force=args.force)
    muse_df = add_kc_mismatch_flag(muse_df)
    save_muse_df(muse_df)
    print(scores.ring_class.value_counts().to_string())
    print(scores[scores['Filename'].isin(labels)][['time', 'Filename', 'ring_class', 'kc_fraction', 'ring_amp', 'coherence', 'snr']].to_string())

    #%% k_c refinement by fits (optional, ~2 min per fit, 6 in parallel)
    if args.scan:
        res = scan_kc_by_fits(muse_df, ids=args.ids, factors=args.factors, devices=args.devices, n_jobs=args.n_jobs, apply=not args.no_apply)
        print(res[['time', 'ring_estimate', 'kc_fraction', 'at_scan_edge', 'loss_min', 'loss_nominal_kc']].to_string())
