#%%
"""
Balanced vs legacy PSF loss: what each fit set gets right (STD-star fits in two folders, same fitter, same regularizers, only the PSF loss
differs; see PSF_LOSS_CFG in STD_fitter.py). Per sample and wavelength bin, for both fit sets:

  peak_err           (data peak - model peak) / data peak, central 7x7 (NFM_validation_utils definition)
  ring_err           relative error of the 8 neighbours of the data peak
  band_err_<a>_<b>   relative flux error of the model in the annulus a <= r < b px (model - data) / data, signed
  band_abs_<a>_<b>   mean |model - data| / mean data in the annulus (shape error)
  EE2_err, EE8_err   encircled-energy ratio errors within 2 and 8 px
  FWHM_err           relative FWHM error (from the stored FWHM fits)
  loss_balanced      BalancedPSFLoss of the fit (the fitter's parameters), and its decomposition by radial band
  loss_legacy        the legacy MSE + MAE loss of the fit

    python datasets/MUSE_STD_dataset/compare_fit_losses.py [--a fitted_balanced --b fitted_legacy]

Outputs: data/analysis/NFM_STD_loss/compare_<a>_<b>.csv (per sample, bin, fit set), summary printed, compare_<a>_<b>.png. CPU, ~3 min.
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here.parent.parent))
from tiptorch._config import project_settings, DATA_FOLDER
from tiptorch.tools.utils import BalancedPSFLoss

_parser = argparse.ArgumentParser()
_parser.add_argument('--a', default='fitted_balanced'); _parser.add_argument('--b', default='fitted_legacy')
_parser.add_argument('--n', type=int, default=0, help='limit the number of samples (0 = all)')
try: _args = _parser.parse_args()
except SystemExit: _args = _parser.parse_args([])

STD_FOLDER = Path(project_settings['MUSE_STD_data_folder'])
OUT = DATA_FOLDER / 'analysis' / 'NFM_STD_loss'; OUT.mkdir(parents=True, exist_ok=True)
FOLDERS = {_args.a: STD_FOLDER / _args.a, _args.b: STD_FOLDER / _args.b}
BANDS = [(0, 1.5), (1.5, 3.5), (3.5, 10.5), (10.5, 20.5), (20.5, 40.5), (40.5, 80.5)]
LEGACY = dict(w_MSE=900.0, w_MAE=1.6, w=2e4)
muse_df = pickle.load(open(STD_FOLDER / 'muse_df.pickle', 'rb'))

ids = sorted(set(int(p.stem) for p in FOLDERS[_args.a].glob('*.pickle')) & set(int(p.stem) for p in FOLDERS[_args.b].glob('*.pickle')))
if _args.n: ids = ids[::max(1, len(ids) // _args.n)][:_args.n]
print(f'{len(ids)} samples in both {_args.a} and {_args.b}')

loss_obj, R = None, None
rows = []
for id in tqdm(ids):
    for name, folder in FOLDERS.items():
        d = pickle.load(open(folder / f'{id}.pickle', 'rb'))
        D, P = d['Img. data'][0], d['Img. fit'][0]
        N_wvl, H, W = D.shape; c = H // 2
        if loss_obj is None:
            cfg = {k: v for k, v in d['loss config'].items() if k in ('radial_power', 'r_max', 'asinh_k', 'huber_delta', 'scale', 'match_background')}
            loss_obj = BalancedPSFLoss(H, W, 'cpu', **cfg)
            yy, xx = np.mgrid[:H, :W]; R = np.hypot(yy - c, xx - c)
        Dt, Pt = torch.tensor(D)[None], torch.tensor(P)[None]
        pix = loss_obj.pixel_map(Dt, Pt)[0].numpy()                     # [N_wvl, H, W]
        diff = Pt - Dt
        loss_leg = ((diff.pow(2).mean(dim=(-2, -1)) * LEGACY['w_MSE'] + diff.abs().mean(dim=(-2, -1)) * LEGACY['w_MAE']) * LEGACY['w'])[0].numpy()
        s = np.s_[:, c-3:c+4, c-3:c+4]
        d_peak, p_peak = D[s].max(axis=(-2, -1)), P[s].max(axis=(-2, -1))
        λ = np.asarray(d['wavelengths']).ravel() * 1e9
        for l in range(N_wvl):
            cr = D[l, c-3:c+4, c-3:c+4]; k = np.unravel_index(cr.argmax(), cr.shape); iy, ix = k[0] + c - 3, k[1] + c - 3
            rd = (D[l, iy-1:iy+2, ix-1:ix+2].sum() - D[l, iy, ix]) / 8; rp = (P[l, iy-1:iy+2, ix-1:ix+2].sum() - P[l, iy, ix]) / 8
            row = dict(fit=name, ID=id, wvl_id=l, λ_nm=λ[l], SR=float(d['SR data'][l]),
                       peak_err=(d_peak[l] - p_peak[l]) / d_peak[l], ring_err=(rd - rp) / rd,
                       loss_balanced=float(pix[l].sum()), loss_legacy=float(loss_leg[l]),
                       FWHM_err=float(np.mean(d['FWHM fit'][l]) / np.mean(d['FWHM data'][l]) - 1))
            for a, b in BANDS:
                m = (R >= a) & (R < b)
                row[f'band_err_{a:g}_{b:g}'] = (P[l][m].sum() - D[l][m].sum()) / D[l][m].sum()
                row[f'band_abs_{a:g}_{b:g}'] = np.abs(P[l][m] - D[l][m]).mean() / np.abs(D[l][m]).mean()
                row[f'lossb_{a:g}_{b:g}'] = float(pix[l][m].sum())
            for rr in (2, 8):
                m = R <= rr; row[f'EE{rr}_err'] = P[l][m].sum() / D[l][m].sum() - 1
            rows.append(row)
        # fitted parameters
        rows[-1].update(r0=float(d['r0'][0]), dn=float(d['dn'][0]), J_med=float(np.median(np.abs(d['J_ctrl']))), bump=float(d['LO_coefs'][0, 0]),
                        LO_rms=float(np.sqrt((d['LO_coefs'][0, 1:]**2).sum())), F_med=float(np.median(d['F_ctrl'])), fit_loss=float(d['loss']))

t = pd.DataFrame(rows)
t.to_csv(OUT / f'compare_{_args.a}_{_args.b}.csv', index=False)

#%% Summary
def med_abs(x): return 100 * np.median(np.abs(x))
cols = ['peak_err', 'ring_err', 'EE2_err', 'EE8_err', 'FWHM_err'] + [f'band_err_{a:g}_{b:g}' for a, b in BANDS]
print('\nmedian |error| [%] per (sample, bin):')
print(t.groupby('fit')[cols].agg(med_abs).round(1).T.to_string())
print('\nmedian signed error [%] (model - data; peak_err and ring_err are data - model):')
print(t.groupby('fit')[cols].agg(lambda x: 100 * np.median(x)).round(1).T.to_string())
print('\nshape error: mean |model - data| / mean data per band [%]:')
print(t.groupby('fit')[[f'band_abs_{a:g}_{b:g}' for a, b in BANDS]].agg(med_abs).round(1).T.to_string())
print('\ncross-evaluated losses (median per bin):')
print(t.groupby('fit')[['loss_balanced', 'loss_legacy']].median().round(4).to_string())
print('\nbalanced-loss budget by band (median share of the per-bin loss, %):')
share = t[[f'lossb_{a:g}_{b:g}' for a, b in BANDS]].div(t['loss_balanced'], axis=0) * 100
print(share.groupby(t['fit']).median().round(1).T.to_string())
print('\nper-wavelength median |peak err| [%]:')
print(t.groupby(['λ_nm', 'fit'])['peak_err'].agg(med_abs).unstack().round(1).to_string())
print('\nfitted parameters (median over samples):')
pp = t.dropna(subset=['r0']).groupby('fit')[['r0', 'dn', 'J_med', 'bump', 'LO_rms', 'F_med', 'fit_loss']].median().round(3)
print(pp.to_string())
print('\nby quality flag, median |peak err| [%]:')
for flag in ['Good quality', 'Medium quality', 'Non-point', 'Has streaks']:
    ok = muse_df.loc[t.ID, flag].values.astype(bool)
    print(f"  {flag:>15}=True : " + '  '.join(f'{f} {med_abs(t[ok & (t.fit == f).values].peak_err):.1f}' for f in FOLDERS) +
          f"  | False: " + '  '.join(f'{f} {med_abs(t[~ok & (t.fit == f).values].peak_err):.1f}' for f in FOLDERS))

#%% Figure
C = {_args.a: '#2A6FDB', _args.b: '#D9822B'}
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for name in FOLDERS:
    s = t[t.fit == name]
    ax[0, 0].hist(100 * s.peak_err, bins=np.linspace(-40, 40, 81), histtype='step', lw=1.5, color=C[name], label=name)
    g = s.groupby('λ_nm').peak_err
    ax[0, 1].plot(g.median().index, 100 * g.median().values, 'o-', color=C[name], label=name, ms=4)
    ax[0, 1].fill_between(g.median().index, 100 * g.quantile(0.16).values, 100 * g.quantile(0.84).values, color=C[name], alpha=0.15)
    prof = [100 * np.median(s[f'band_err_{a:g}_{b:g}']) for a, b in BANDS]
    ax[0, 2].plot([0.5 * (a + b) for a, b in BANDS], prof, 'o-', color=C[name], label=name)
    absprof = [100 * np.median(s[f'band_abs_{a:g}_{b:g}']) for a, b in BANDS]
    ax[1, 0].plot([0.5 * (a + b) for a, b in BANDS], absprof, 'o-', color=C[name], label=name)
    sh = [100 * np.median(s[f'lossb_{a:g}_{b:g}'] / s.loss_balanced) for a, b in BANDS]
    ax[1, 1].plot([0.5 * (a + b) for a, b in BANDS], sh, 'o-', color=C[name], label=name)
ax[0, 0].axvline(0, color='#9AA0A6', lw=1); ax[0, 0].set_xlabel('peak error (data - model) / data [%]'); ax[0, 0].set_ylabel('(sample, bin) pairs'); ax[0, 0].legend(); ax[0, 0].set_title('Peak error')
ax[0, 1].axhline(0, color='#9AA0A6', lw=1); ax[0, 1].set_xlabel('wavelength [nm]'); ax[0, 1].set_ylabel('peak error [%] (median, 16-84 %)'); ax[0, 1].legend(); ax[0, 1].set_title('Peak error vs wavelength')
ax[0, 2].axhline(0, color='#9AA0A6', lw=1); ax[0, 2].set_xscale('log'); ax[0, 2].set_xlabel('radius [px]'); ax[0, 2].set_ylabel('flux error (model - data) / data [%]'); ax[0, 2].legend(); ax[0, 2].set_title('Radial flux bias')
ax[1, 0].set_xscale('log'); ax[1, 0].set_xlabel('radius [px]'); ax[1, 0].set_ylabel('mean |model - data| / mean data [%]'); ax[1, 0].legend(); ax[1, 0].set_title('Radial shape error')
ax[1, 1].set_xscale('log'); ax[1, 1].set_xlabel('radius [px]'); ax[1, 1].set_ylabel('share of the balanced loss [%]'); ax[1, 1].legend(); ax[1, 1].set_title('Where the balanced loss lives')
a_ = t[t.fit == _args.a].set_index(['ID', 'wvl_id']); b_ = t[t.fit == _args.b].set_index(['ID', 'wvl_id'])
ax[1, 2].scatter(100 * b_.peak_err, 100 * a_.loc[b_.index].peak_err, s=3, alpha=0.3, color='#2A6FDB')
ax[1, 2].plot([-40, 40], [-40, 40], color='#9AA0A6', lw=1); ax[1, 2].set_xlim(-40, 40); ax[1, 2].set_ylim(-40, 40)
ax[1, 2].set_xlabel(f'peak error, {_args.b} [%]'); ax[1, 2].set_ylabel(f'peak error, {_args.a} [%]'); ax[1, 2].set_title('Per-bin peak error, one fit set vs the other')
plt.tight_layout(); plt.savefig(OUT / f'compare_{_args.a}_{_args.b}.png', dpi=130); plt.close()
print(f'\nSaved {OUT}')
