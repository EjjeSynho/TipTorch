#%%
"""
Does the model's seeing halo match the data beyond r = 40 px (the r_max of the balanced loss)? Diagnostic on the fitted validation
stars with all fitted parameters frozen. The dataset PSFs are background-subtracted by the minimum of their radial profile and
renormalized (LoadSTDStarCache), so the model PSFs get the identical processing before any comparison (`process_like_data`).

  1. the data / model ratio of the azimuthal profile in radial rings and per wavelength (the deficit itself);
  2. a single multiplicative excess (1 + eps) of the fitting-error PSD (`TipTorch.fitting_excess`) fitted per star on the halo
     (40-85 px): eps ~ 0.2 everywhere means the atmospheric normalization is off (r0 compromise between core and halo);
  3. eps fitted separately in an inner (40-60 px) and an outer (60-85 px) ring: the same value means a multiplicative deficit, a
     larger outer value an additional, flatter component (scattering);
  4. eps fitted together with a free constant background per wavelength: eps dropping to ~0 means a data-reduction offset.

    python machine_learning/analysis/analyze_halo_deficit.py [--device cuda:1] [--n-samples 48] [--batch 8] [--steps 80]

Outputs: data/analysis/NFM_calibrator/halo_deficit.{csv,png}. Needs the dataset cache and a GPU (~15 min for 48 stars).
"""
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from calibrators.NFM_calibrator import NFMDataset, release_gpu_memory

_parser = argparse.ArgumentParser(description="Halo deficit diagnostic")
_parser.add_argument('--device',    type=str, default=None)
_parser.add_argument('--n-samples', type=int, default=48)
_parser.add_argument('--batch',     type=int, default=8)
_parser.add_argument('--steps',     type=int, default=80)
try:
    _args = _parser.parse_args()
except SystemExit:
    _args = _parser.parse_args([])

device     = torch.device(_args.device) if _args.device else default_device
STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])
OUT_FOLDER = DATA_FOLDER / 'analysis' / 'NFM_calibrator'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_CFG = dict(LO_NCPAs=True, chrom_defocus=False, use_Moffat=False, retain_PSDs=False, N_spline_nodes=5, Z_mode_max=9)
RINGS     = {'core 0-3': (0, 3.5), 'AO 3-10': (3.5, 10.5), 'edge 10-20': (10.5, 20.5), 'halo 20-40': (20.5, 40.5), 'halo 40-60': (40.5, 60.5), 'halo 60-85': (60.5, 85.5)}
R_HALO    = (40.5, 85.5) # the region ignored by the balanced loss (r_max = 40)

#%%
dataset = NFMDataset(STD_FOLDER / 'dataset_cache' / 'muse_STD_stars_dataset.pt')
val_idx = np.asarray(dataset.val_idx if dataset.val_idx is not None else np.arange(len(dataset)), dtype=int)
rng     = np.random.default_rng(0)
samples = np.sort(rng.choice(val_idx, size=min(_args.n_samples, len(val_idx)), replace=False))
batches = [samples[i:i+_args.batch] for i in range(0, len(samples), _args.batch)]
H, W    = dataset.H, dataset.W
wvl_nm  = (np.asarray(dataset.configs[samples[0]]['sources_science']['Wavelength'], dtype=float).ravel() * 1e9).round().astype(int)
print(f'{len(samples)} validation stars, {len(wvl_nm)} wavelengths {wvl_nm[0]}-{wvl_nm[-1]} nm, field {H} px')

yy, xx = np.mgrid[:H, :W]
R      = torch.as_tensor(np.hypot(yy - H // 2, xx - W // 2), dtype=torch.float32, device=device)
r_edges = np.arange(0, 86)
ring_masks = torch.stack([(R >= r_edges[k]) & (R < r_edges[k+1]) for k in range(len(r_edges) - 1)]).float() # [N_r, H, W]
r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
_edges_full = np.arange(0, H // 2 + 1) # RadialProfile(edge_radii = arange(H // 2)) of the dataset preprocessing
ring_masks_full = torch.stack([(R >= _edges_full[k]) & (R < _edges_full[k+1]) for k in range(len(_edges_full) - 1)]).float()
DELTA = 1e-6 # [PSF sum units] floor of the log-ratio halo loss (the far wing of the processed profiles tends to zero)


def profiles(P, r_max=None):
    """Azimuthal mean profiles [B, N_wvl, N_r] of PSF cubes [B, N_wvl, H, W]."""
    masks = ring_masks if r_max is None else ring_masks_full
    return torch.einsum('bwhx,rhx->bwr', P, masks) / masks.sum(dim=(-2, -1))


def process_like_data(P):
    """The dataset's preprocessing applied to a model cube: subtract the minimum of the radial profile (out to the field edge) per
    wavelength, renormalize to a unit sum (LoadSTDStarCache with subtract_background and normalize)."""
    bg = profiles(P, r_max='full').min(dim=-1).values # [B, N_wvl]
    P  = P - bg[..., None, None]
    return P / P.sum(dim=(-2, -1), keepdim=True)


def fitted_x_dict(ids):
    f = {k: torch.stack([torch.as_tensor(np.asarray(dataset.fitted_vals[i][k]), dtype=torch.float32) for i in ids]).to(device) for k in dataset.fitted_vals[ids[0]]}
    cn2 = f['Cn2_weights'].clamp(min=0.0)
    x = {k: v for k, v in f.items() if k in ('F_ctrl', 'J_ctrl', 'bg_ctrl', 'dx_ctrl', 'dy_ctrl', 'LO_coefs', 'dn', 'L0', 'r0')}
    x['wind_speed_single'] = f['wind_speed_single'].view(-1, 1)
    x['Cn2_weights']       = torch.hstack([(1.0 - cn2.sum(-1, keepdim=True)).clamp(min=0.0), cn2])
    return x


def make_model(ids):
    _, _, _, config, _ = dataset.collate_batch(tuple(dataset[int(i)] for i in ids), device=device)
    model = PSFModelNFM(config, multiple_obs=True, device=device, **MODEL_CFG)
    return model


def run(model, x, eps=None):
    model.model.fitting_excess = eps
    return process_like_data(model(x, update_params=False)) # [B, N_wvl, H, W], processed like the dataset PSFs


def halo_loss(prof_model, prof_data, r_range):
    """Mean squared log ratio of the profiles inside r_range, where the data profile is positive."""
    sel  = torch.as_tensor((r_mid >= r_range[0]) & (r_mid < r_range[1]), device=device)
    ok   = (prof_data > -DELTA) & sel
    lr   = torch.log((prof_model + DELTA).clamp(min=1e-12) / (prof_data + DELTA).clamp(min=1e-12))
    return ((lr ** 2) * ok).sum(dim=(1, 2)) / ok.sum(dim=(1, 2)).clamp(min=1) # [B]


def fit_eps(model, x, prof_data, r_range, with_bg=False, steps=80, lr=0.05):
    """eps [B] (and bg [B, N_wvl] in units of 1e-6 of the normalized PSF) minimizing the halo loss, the fitted parameters frozen."""
    B, N_wvl = prof_data.shape[:2]
    eta = torch.zeros(B, device=device, requires_grad=True)                  # 1 + eps = exp(eta)
    bg  = torch.zeros(B, N_wvl, device=device, requires_grad=with_bg)
    opt = torch.optim.Adam([eta, bg] if with_bg else [eta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P    = run(model, x, torch.exp(eta) - 1.0)
        prof = profiles(P) + 1e-6 * bg.unsqueeze(-1)
        loss = halo_loss(prof, prof_data, r_range).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        eps = torch.exp(eta) - 1.0
        P   = run(model, x, eps)
        prof = profiles(P) + 1e-6 * bg.unsqueeze(-1)
    model.model.fitting_excess = None
    return eps.detach(), bg.detach(), prof, halo_loss(prof, prof_data, r_range)


#%%
rows, ratio_rows = [], []
for ids in batches:
    PSF_data = torch.stack([dataset[int(i)][0] for i in ids]).to(device)
    model    = make_model(ids)
    x        = fitted_x_dict(ids)
    with torch.no_grad():
        prof_data  = profiles(PSF_data)
        prof_model = profiles(run(model, x))
    ratio0 = (prof_model / prof_data.clamp(min=1e-12)).cpu().numpy() # [B, N_wvl, N_r]

    eps_all, bg0, prof_all, loss_all = fit_eps(model, x, prof_data, R_HALO,          steps=_args.steps)
    eps_in,  _,   _,        _        = fit_eps(model, x, prof_data, (40.5, 60.5),    steps=_args.steps)
    eps_out, _,   _,        _        = fit_eps(model, x, prof_data, (60.5, 85.5),    steps=_args.steps)
    eps_bg,  bg,  prof_bg,  loss_bg  = fit_eps(model, x, prof_data, R_HALO, with_bg=True, steps=_args.steps)
    ratio_fit = (prof_all / prof_data.clamp(min=1e-12)).cpu().numpy()

    for n, i in enumerate(ids):
        rows.append({'sample_id': int(dataset.sample_ids[i]), 'r0_fit': float(x['r0'][n]), 'seeing': float(np.asarray(dataset.configs[i]['atmosphere']['Seeing']).ravel()[0]),
                     'eps_halo': float(eps_all[n]), 'eps_40_60': float(eps_in[n]), 'eps_60_85': float(eps_out[n]), 'eps_with_bg': float(eps_bg[n]),
                     'bg_median_1e-6': float(bg[n].median()), 'halo_loss_0': float(halo_loss(prof_model, prof_data, R_HALO)[n]),
                     'halo_loss_eps': float(loss_all[n]), 'halo_loss_eps_bg': float(loss_bg[n]),
                     **{f'ratio {name}': float(np.median(ratio0[n][:, (r_mid >= a) & (r_mid < b)])) for name, (a, b) in RINGS.items()},
                     **{f'ratio_fit {name}': float(np.median(ratio_fit[n][:, (r_mid >= a) & (r_mid < b)])) for name, (a, b) in RINGS.items()}})
        for w in (0, len(wvl_nm) // 2, len(wvl_nm) - 1):
            ratio_rows.append({'sample_id': int(dataset.sample_ids[i]), 'wvl_nm': int(wvl_nm[w]), **{f'r{int(r)}': float(ratio0[n, w, k]) for k, r in enumerate(r_mid)}})
    print(f'batch {ids[0]}-{ids[-1]}: model/data halo 40-85 px {np.median([r["ratio halo 40-60"] for r in rows[-len(ids):]]):.3f} | '
          f'eps halo {eps_all.median():+.3f}  40-60 {eps_in.median():+.3f}  60-85 {eps_out.median():+.3f}  with bg {eps_bg.median():+.3f}', flush=True)
    model.cleanup(); del model
    release_gpu_memory()

table  = pd.DataFrame(rows).set_index('sample_id')
ratios = pd.DataFrame(ratio_rows)
table.to_csv(OUT_FOLDER / 'halo_deficit.csv')
ratios.to_csv(OUT_FOLDER / 'halo_deficit_ratios.csv', index=False)

#%%
q = lambda s: f'{s.median():+.3f} (p16 {s.quantile(0.16):+.3f}, p84 {s.quantile(0.84):+.3f})'
print('\nmodel / data profile ratio (medians over stars and wavelengths), fitted parameters as they are:')
for name in RINGS:
    print(f"  {name:12s} {table[f'ratio {name}'].median():.3f}   after the halo eps fit: {table[f'ratio_fit {name}'].median():.3f}")
print('\nfitting-error excess eps:')
print(f'  halo 40-85 px          {q(table.eps_halo)}')
print(f'  inner ring 40-60 px    {q(table.eps_40_60)}')
print(f'  outer ring 60-85 px    {q(table.eps_60_85)}')
print(f'  with free background   {q(table.eps_with_bg)}   bg median {table["bg_median_1e-6"].median():+.2f} x 1e-6 of the PSF sum')
print(f'  halo loss: as fitted {table.halo_loss_0.median():.4f}, with eps {table.halo_loss_eps.median():.4f}, with eps + bg {table.halo_loss_eps_bg.median():.4f}')
print(f'  corr(eps, r0): {np.corrcoef(table.eps_halo, table.r0_fit)[0, 1]:+.2f}, corr(eps, seeing): {np.corrcoef(table.eps_halo, table.seeing)[0, 1]:+.2f}')

#%%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
for wv, c in zip(sorted(ratios.wvl_nm.unique()), ('C0', 'C1', 'C3')):
    g = ratios[ratios.wvl_nm == wv][[f'r{int(r)}' for r in r_mid]].to_numpy()
    ax.plot(r_mid, np.median(g, axis=0), color=c, label=f'{wv} nm (median of {len(g)} stars)')
    ax.fill_between(r_mid, np.percentile(g, 16, axis=0), np.percentile(g, 84, axis=0), color=c, alpha=0.15)
ax.axhline(1, color='k', lw=0.8); ax.axvline(40, color='gray', ls='--', lw=0.8)
ax.set(xlabel='radius [px]', ylabel='model / data azimuthal profile', ylim=(0.5, 1.5), title='Profile ratio with the fitted parameters')
ax.legend(fontsize=8)
ax = axes[1]
bins = np.linspace(-0.5, 1.0, 31)
for col, lab in (('eps_halo', 'halo 40-85 px'), ('eps_40_60', 'ring 40-60 px'), ('eps_60_85', 'ring 60-85 px'), ('eps_with_bg', 'halo, free background')):
    ax.hist(table[col].clip(bins[0], bins[-1]), bins=bins, histtype='step', lw=1.8, label=f'{lab}: median {table[col].median():+.2f}')
ax.axvline(0, color='k', lw=0.8)
ax.set(xlabel='fitting-error excess eps', ylabel='stars', title='Excess of the fitting PSD needed by the halo'); ax.legend(fontsize=8)
ax = axes[2]
ax.scatter(table.r0_fit, table.eps_halo, s=18, alpha=0.8)
ax.axhline(0, color='k', lw=0.8)
ax.set(xlabel='fitted r0 [m]', ylabel='eps (halo 40-85 px)', title=f'eps vs fitted r0 (corr {np.corrcoef(table.eps_halo, table.r0_fit)[0, 1]:+.2f})')
plt.tight_layout()
plt.savefig(OUT_FOLDER / 'halo_deficit.png', dpi=130)
print(f'Saved -> {OUT_FOLDER / "halo_deficit.csv"} / .png')
