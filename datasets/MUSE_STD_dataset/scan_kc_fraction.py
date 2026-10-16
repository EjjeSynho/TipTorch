"""
Refines the k_c of the MUSE NFM STD samples flagged 'k_c mismatch' by scanning the model k_c with PSF fits.

The ring position in the data - fit residuals (detect_kc_mismatch.py) overestimates the real k_c by a few percent because the fit
compensates part of the mismatch with r0 / dn. For every flagged sample this script fits the PSF (STD_fitter.py, the production fitting
routine) with the model k_c forced to  <ring estimate> x <factor>  for a few factors, takes the fit loss minimum (parabolic interpolation
of the three lowest points) as the sample's k_c and writes it to STD_FOLDER / 'kc_mismatch_labels.txt' as "k_c <p> % (fit)", which
the detector never overwrites. The 'k_c fraction' / 'k_c group' columns of muse_df.pickle are refreshed afterwards (the groups are
re-derived from the fitted values).

Fits run as parallel STD_fitter.py subprocesses (~2 min each on a GPU); existing results in the scan folder are reused, so an
interrupted scan can be resumed. The scan results (loss per factor, best k_c, ring amplitude at the best k_c) go to
STD_FOLDER / 'kc_scan_results.csv'.

Usage:
    python scan_kc_fraction.py                                  # all flagged (fittable) samples, 6 jobs on cuda:0
    python scan_kc_fraction.py --ids 19 451 51 --devices cuda:0 cuda:1 --n_jobs 6
    python scan_kc_fraction.py --factors 0.86 0.90 0.94 0.98 1.0 --no_apply   # scan only, do not touch the labels
"""
import sys
import os
import time
import pickle
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from STD_dataset_utils import (STD_FOLDER, read_kc_mismatch_labels, write_kc_mismatch_labels, add_kc_mismatch_flag,
                               KC_MISMATCH_FLAG, KC_FRACTION_FEATURE)

SCRIPT_DIR  = Path(__file__).resolve().parent
LABELS_FILE = STD_FOLDER / 'kc_mismatch_labels.txt'
RESULTS_CSV = STD_FOLDER / 'kc_scan_results.csv'
DEFAULT_SCAN_DIR = STD_FOLDER / 'fitted_kc_scan'
DEFAULT_FACTORS  = (0.86, 0.90, 0.94, 0.98) # x the ring estimate (the real k_c is below it)
NO_MISMATCH_ABOVE = 0.99 # fitted k_c >= 99 % of the nominal: not a mismatch, the label is set to the 'none' class (vetoed, by the fit)


def run_fits(jobs, devices, n_jobs, verbose=True):
    """ jobs: list of (id, kc_fraction, out_dir). Runs STD_fitter.py subprocesses, at most n_jobs at a time, round-robin over devices. """
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(SCRIPT_DIR), str(SCRIPT_DIR.parent.parent)]) + os.pathsep + env.get('PYTHONPATH', '')
    pending, running, done, failed = list(jobs), [], 0, []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < n_jobs:
            id_, frac, out_dir = pending.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            device = devices[(done + len(running) + len(failed)) % len(devices)]
            cmd = [sys.executable, str(SCRIPT_DIR / 'STD_fitter.py'), device, str(id_), str(id_), f'kc_fraction={frac:.4f}', str(out_dir)]
            log = open(out_dir / f'fit_{id_}.log', 'w')
            running.append((id_, frac, out_dir, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        time.sleep(5)
        still = []
        for job in running:
            id_, frac, out_dir, proc, log = job
            if proc.poll() is None:
                still.append(job); continue
            log.close()
            if proc.returncode == 0 and (out_dir / f'{id_}.pickle').exists():
                done += 1
            else:
                failed.append((id_, frac))
            if verbose:
                print(f'  [{done + len(failed)}/{len(jobs)}] id {id_} k_c {100 * frac:.1f} %: {"ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"}  '
                      f'({(time.time() - t0) / 60:.1f} min elapsed)', flush=True)
        running = still
    return failed


def parabolic_min(x, y):
    """ Minimum of the parabola through the three points around the lowest y; falls back to the lowest point at the edges. """
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = int(np.argmin(y))
    if 0 < k < len(x) - 1:
        a, b, c = np.polyfit(x[k-1:k+2], y[k-1:k+2], 2)
        if a > 0:
            return float(np.clip(-b / (2 * a), x[k-1], x[k+1])), False
    return float(x[k]), True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ids',      type=int, nargs='*', default=None, help='sample ids (default: all flagged, not Corrupted / Bad quality)')
    parser.add_argument('--devices',  type=str, nargs='+', default=['cuda:0'])
    parser.add_argument('--n_jobs',   type=int, default=6)
    parser.add_argument('--factors',  type=float, nargs='+', default=list(DEFAULT_FACTORS), help='scanned k_c = ring estimate x factor')
    parser.add_argument('--scan_dir', type=str, default=str(DEFAULT_SCAN_DIR))
    parser.add_argument('--no_apply', action='store_true', help='do not update the labels file / muse_df')
    args = parser.parse_args()

    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
        muse_df = pickle.load(f)
    labels = read_kc_mismatch_labels(LABELS_FILE)
    name_to_id = {v: k for k, v in muse_df['Filename'].items()}

    if args.ids is None:
        sel = muse_df[muse_df[KC_MISMATCH_FLAG] & ~muse_df['Corrupted'] & ~muse_df['Bad quality']]
        ids = sel.index.tolist()
    else:
        ids = args.ids
    scan_dir = Path(args.scan_dir)

    # Jobs: ring estimate x factors. Labels already refined by a previous scan keep the ring estimate stored in the scan results as the
    # base (so that the factors always refer to the same value); samples labeled by hand without a measurement use the group value of muse_df
    previous = pd.read_csv(RESULTS_CSV, index_col='id') if RESULTS_CSV.exists() else None
    ring_scores = pd.read_csv(STD_FOLDER / 'kc_ring_scores.csv') if (STD_FOLDER / 'kc_ring_scores.csv').exists() else None
    jobs, base = [], {}
    for id_ in ids:
        name = muse_df.loc[id_, 'Filename']
        e = labels.get(name, {})
        if e.get('kc_percent') is not None and e.get('kc_source') != 'fit':
            ring = e['kc_percent'] / 100.0
        elif previous is not None and id_ in previous.index and np.isfinite(previous.loc[id_, 'ring_estimate']):
            ring = float(previous.loc[id_, 'ring_estimate'])
        elif ring_scores is not None and name in set(ring_scores['Filename']):
            ring = float(ring_scores.set_index('Filename').loc[name, 'kc_fraction'])
        elif e.get('kc_percent') is not None:
            ring = e['kc_percent'] / 100.0 # re-scan around the previous fit
        else:
            ring = None
        base[id_] = ring if ring is not None else float(muse_df.loc[id_, KC_FRACTION_FEATURE])
        for fct in args.factors:
            frac = round(base[id_] * fct, 4)
            out_dir = scan_dir / str(id_) / f'kc_{frac:.4f}'
            if not (out_dir / f'{id_}.pickle').exists():
                jobs.append((id_, frac, out_dir))
    print(f'k_c scan: {len(ids)} samples x {len(args.factors)} factors, {len(jobs)} fits to run ({len(ids) * len(args.factors) - len(jobs)} reused) on {args.devices}, {args.n_jobs} jobs')

    failed = run_fits(jobs, args.devices, args.n_jobs)
    if failed:
        print(f'{len(failed)} fits failed: {failed}')

    # Collect: loss per fraction, best k_c, ring amplitude at the best k_c
    from detect_kc_mismatch import ring_profiles, ring_score
    rows = []
    for id_ in ids:
        fracs, losses, pickles = [], [], {}
        for d in sorted((scan_dir / str(id_)).glob('kc_*')):
            p = d / f'{id_}.pickle'
            if not p.exists(): continue
            with open(p, 'rb') as f: fit = pickle.load(f)
            fracs.append(float(d.name[3:])); losses.append(float(fit['loss'])); pickles[fracs[-1]] = fit
        if len(fracs) < 2:
            print(f'  id {id_}: not enough fits ({len(fracs)}), skipped'); continue
        order = np.argsort(fracs); fracs, losses = np.array(fracs)[order], np.array(losses)[order]
        best, at_edge = parabolic_min(fracs, losses)
        at_edge = ('low' if np.argmin(losses) == 0 else 'high') if at_edge else ''
        fit_best = pickles[fracs[np.argmin(np.abs(fracs - best))]]
        prof, coh, _ = ring_profiles(fit_best['Img. data'][0], fit_best['Img. fit'][0], fit_best['wavelengths'])
        s = ring_score(prof, coh)
        rows.append(dict(id=id_, Filename=muse_df.loc[id_, 'Filename'], time=str(muse_df.loc[id_, 'time'])[:10], ring_estimate=base[id_],
                         kc_fit=best, at_scan_edge=at_edge, loss_min=float(losses.min()), loss_nominal_kc=np.nan,
                         ring_amp_at_fit=s['ring_amp'], scanned=' '.join(f'{f:.3f}:{l:.4f}' for f, l in zip(fracs, losses))))
        print(f"  id {id_}: ring estimate {100 * base[id_]:.1f} % -> fitted k_c {100 * best:.1f} %{f' ({at_edge} scan edge!)' if at_edge else ''}, "
              f"loss {losses.min():.4f}, ring amp at the fit {s['ring_amp']:+.2f}")
    res = pd.DataFrame(rows).set_index('id')
    if previous is not None: # keep the rows of the samples not scanned this time
        res = pd.concat([previous.loc[[i for i in previous.index if i not in res.index], res.columns.intersection(previous.columns)], res]).sort_index()
    res['kc_fit'] = res['kc_fit'].clip(upper=1.0)
    res['mismatch'] = res['kc_fit'] < NO_MISMATCH_ABOVE # a fitted k_c at the nominal one: the ring was not a k_c mismatch
    nominal_fits = Path(STD_FOLDER / 'fitted')
    for id_ in res.index:
        p = nominal_fits / f'{id_}.pickle'
        if p.exists():
            with open(p, 'rb') as f: res.loc[id_, 'loss_nominal_kc'] = float(pickle.load(f)['loss'])
    res.to_csv(RESULTS_CSV)
    print(f'Scan results -> {RESULTS_CSV}')

    if args.no_apply or len(res) == 0:
        return
    vetoed = []
    for id_, r in res.iterrows():
        name = r['Filename']
        if name not in labels:
            labels[name] = dict(date=r['time'], ring_class='manual', amp=None)
        labels[name].update(kc_percent=100 * r['kc_fit'], kc_source='fit')
        if not r['mismatch']:
            labels[name]['ring_class'] = 'none'; vetoed.append(id_)
    if vetoed:
        print(f"Fitted k_c at the nominal value (>= {100 * NO_MISMATCH_ABOVE:.0f} %): ids {vetoed} set to the 'none' class (not a k_c mismatch; edit the labels file to override)")
    write_kc_mismatch_labels(labels, LABELS_FILE)
    muse_df = add_kc_mismatch_flag(muse_df, LABELS_FILE)
    with open(STD_FOLDER / 'muse_df.pickle', 'wb') as f:
        pickle.dump(muse_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{len(res)} fitted k_c values written to {LABELS_FILE}; muse_df.pickle refreshed')


if __name__ == '__main__':
    main()
