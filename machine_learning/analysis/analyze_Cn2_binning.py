#%%
"""
Cn2 binning analysis for the MUSE NFM PSF model: how well 3-layer reductions of the AOF Cn2 profile reproduce the PSF computed with
the full (repaired) 8-layer profile, for the fitted parameters of validation samples with a recorded profile. Compared reductions:
the weighted K-means with free per-sample altitudes ('kmeans' binning), the fixed 3-layer grid ('grid' binning: the dataset medians of
the K-means altitudes, the same for all samples, with the (5/3)-moment reprojection of Cn2Grid), the same grid with the linear
(mean-altitude) reprojection and with the other projection of the AOF ground layer, and the median profile (no information).
The grid is fitted from muse_df as create_STD_dataset.py does (store_Cn2_grid), without storing it.

    python machine_learning/analysis/analyze_Cn2_binning.py [--device cuda:1] [--n-samples 64] [--batch 8]

Outputs: data/analysis/NFM_calibrator/Cn2_binning.csv / .png. Needs the dataset cache and muse_df.pickle; ~5 min on a GPU.
"""
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from tiptorch.tools.utils import BalancedPSFLoss
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from calibrators.NFM_calibrator import NFMDataset, release_gpu_memory, load_STD_dataset # first: adds the repo root to sys.path
from data_processing.Cn2_profile import Cn2Grid, ALT_COLS, FRAC_COLS
from data_processing.MUSE_data_utils import raw_Cn2_profiles, recompute_simplified_Cn2
sys.path.insert(0, str(next(p for p in (_here, *_here.parents) if (p / 'datasets' / 'MUSE_STD_dataset').is_dir()) / 'datasets' / 'MUSE_STD_dataset'))
from STD_dataset_utils import store_Cn2_grid, AOF_Cn2_profiles_stats

_parser = argparse.ArgumentParser(description="Cn2 binning analysis")
_parser.add_argument('--device',    type=str, default=None)
_parser.add_argument('--n-samples', type=int, default=64)
_parser.add_argument('--batch',     type=int, default=8)
_parser.add_argument('--derotated', action='store_true', help='use the derotated dataset cache (default: raw frames, as the calibrator configs)')
try:
    _args = _parser.parse_args()
except SystemExit:
    _args = _parser.parse_args([])

device     = torch.device(_args.device) if _args.device else default_device
STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])
OUT_FOLDER = DATA_FOLDER / 'analysis' / 'NFM_calibrator'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_CFG = dict(LO_NCPAs=True, chrom_defocus=False, use_Moffat=False, retain_PSDs=False, N_spline_nodes=5, Z_mode_max=9)
LOSS_CFG  = dict(radial_power=2.0, r_max=40.0, asinh_k=3.0, huber_delta=0.1, scale=3.0, match_background=False)

#%%
dataset = load_STD_dataset(_args.derotated)
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
    muse_df = pickle.load(f)
    muse_df = pickle.load(f)

(median_alts, median_fracs), _, _ = AOF_Cn2_profiles_stats(muse_df)
median_profile = {'median_Cn2_alts': median_alts, 'median_Cn2_fracs': median_fracs}
grid = store_Cn2_grid(muse_df, median_Cn2_profile=(median_alts, median_fracs), store=False) # as create_STD_dataset.py, not stored

alts_raw, fracs_raw = raw_Cn2_profiles(muse_df.loc[dataset.sample_ids])
alts, fracs, n_repaired, valid = grid.repair(alts_raw, fracs_raw)

val_idx = np.asarray(dataset.val_idx if dataset.val_idx is not None else np.arange(len(dataset)), dtype=int)
val_idx = val_idx[valid[val_idx]]
rng     = np.random.default_rng(0)
samples = np.sort(rng.choice(val_idx, size=min(_args.n_samples, len(val_idx)), replace=False))
print(f'{grid} | {valid.sum()} / {len(dataset)} samples with a profile, {(n_repaired > 0).sum()} repaired | {len(samples)} validation samples analysed')

#%%
def linear_split(grid, alts, fracs):
    """The reprojection of Cn2Grid.to_grid with the linear split (mass and mean altitude preserved) instead of the (5/3)-moment one."""
    a = alts.copy()
    if grid.pin_ground_layer:
        a[:, 0] = 0.0
    h = np.clip(a, grid.nodes[0], grid.nodes[-1])
    k = np.clip(np.searchsorted(grid.nodes, h, side='right') - 1, 0, grid.N_layers - 2)
    t = (h - grid.nodes[k]) / (grid.nodes[k+1] - grid.nodes[k])
    w = np.zeros((len(a), grid.N_layers))
    rows = np.repeat(np.arange(len(a)), a.shape[1])
    np.add.at(w, (rows, k.ravel()),   ((1.0 - t) * fracs).ravel())
    np.add.at(w, (rows, k.ravel()+1), (t * fracs).ravel())
    return w


# Layer schemes: name -> (heights [N, L] in km, weights [N, L]) for the analysed samples
fixed_h = lambda: np.tile(grid.nodes, (len(samples), 1))
km = recompute_simplified_Cn2(muse_df.loc[dataset.sample_ids[samples], ALT_COLS + FRAC_COLS].copy(), n_layers=3, median_Cn2_profile=median_profile)
f2, f3 = km['Cn2_frac_binned_2'].to_numpy(float), km['Cn2_frac_binned_3'].to_numpy(float)
kmeans_h = np.column_stack([np.zeros(len(samples)), km['Cn2_alt_binned_2'].to_numpy(float), km['Cn2_alt_binned_3'].to_numpy(float)])
kmeans_w = np.column_stack([1.0 - f2 - f3, f2, f3])

_alt_grid = Cn2Grid(grid.nodes, grid.layer_altitudes, median_fracs=grid.median_fracs, pin_ground_layer=not grid.pin_ground_layer)
_alt_name = 'GL pinned' if _alt_grid.pin_ground_layer else 'GL split'

schemes = {
    'reference (8 AOF layers)':         (alts[samples], fracs[samples]),
    'K-means (per-sample altitudes)':   (kmeans_h, kmeans_w),
    'fixed grid':                       (fixed_h(), grid.to_grid(alts[samples], fracs[samples])),
    'fixed grid, linear split':         (fixed_h(), linear_split(grid, alts[samples], fracs[samples])),
    f'fixed grid, {_alt_name}':         (fixed_h(), _alt_grid.to_grid(alts[samples], fracs[samples])),
    'median profile':                   (fixed_h(), np.tile(grid.median_fracs, (len(samples), 1))),
}

#%%
def fitted_x_dict(ids):
    """PSF-model inputs of the fitted values (phase bump included, Cn2 profile set per scheme)."""
    f = {k: torch.stack([torch.as_tensor(np.asarray(dataset.fitted_vals[i][k]), dtype=torch.float32) for i in ids]).to(device) for k in dataset.fitted_vals[ids[0]]}
    return {k: v for k, v in f.items() if k in ('F_ctrl', 'J_ctrl', 'bg_ctrl', 'dx_ctrl', 'dy_ctrl', 'LO_coefs', 'dn', 'L0', 'r0')} | {'wind_speed_single': f['wind_speed_single'].view(-1, 1)}


def scheme_configs(ids, heights, weights):
    configs = []
    for n, i in enumerate(ids):
        cfg = deepcopy(dataset.configs[i])
        atm = cfg['atmosphere']
        ws, wd = float(np.asarray(atm['WindSpeed']).ravel()[0]), float(np.asarray(atm['WindDirection']).ravel()[0])
        L = len(heights[n])
        atm['Cn2Heights']    = [(np.asarray(heights[n], dtype=float) * 1e3).tolist()]
        atm['Cn2Weights']    = [np.asarray(weights[n], dtype=float).tolist()]
        atm['WindSpeed']     = [[ws] + [0.0] * (L - 1)]
        atm['WindDirection'] = [[wd] + [0.0] * (L - 1)]
        configs.append(cfg)
    config = MultipleTargetsInDifferentObservations(configs, device=device)
    config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
    config['telescope']['PupilAngle'] = 0.0
    config['DM']['NumberReconstructedLayers'] = torch.tensor(config['atmosphere']['Cn2Heights'].shape[-1], device=device)
    return config


models = {} # one PSF model per number of layers, the configs are swapped per batch as in NFMCalibratorTrainer.run_model

@torch.no_grad()
def render(ids, heights, weights):
    config = scheme_configs(ids, heights, weights)
    L = heights.shape[1]
    if L not in models:
        models[L] = PSFModelNFM(config, multiple_obs=True, device=device, **MODEL_CFG)
    model = models[L]
    model.model.N_obs = len(ids)
    model.model.config = config
    model.model.Update(grids=False, pupils=False, tomography=True)
    x = model.inputs_manager.to_dict()
    x.update(fitted_x_dict(ids))
    x['Cn2_weights'] = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
    return model(x, update_params=False)


#%%
_, _, _, cfg0, _ = dataset.collate_batch(tuple(dataset[i] for i in samples[:1]), device=device)
H, W = dataset.H, dataset.W
balanced = BalancedPSFLoss(H, W, device, **LOSS_CFG)
yy, xx = np.mgrid[:H, :W]
R = np.hypot(yy - H // 2, xx - W // 2)
r_bins = np.arange(0, 41)
ring   = [(R >= r_bins[k]) & (R < r_bins[k+1]) for k in range(len(r_bins) - 1)]
c = H // 2
wvl_nm = (np.asarray(dataset.configs[samples[0]]['sources_science']['Wavelength'], dtype=float).ravel() * 1e9).round().astype(int)
wvl_sel = [0, len(wvl_nm) // 2, len(wvl_nm) - 1]


def profile_err(P, Rf):
    """Max |azimuthal profile difference| in % of the reference peak, [N, N_wvl], r <= 40 px."""
    prof = lambda X: np.stack([X[..., m].mean(axis=-1) for m in ring], axis=-1)
    return np.abs(prof(P) - prof(Rf)).max(axis=-1) / Rf[..., c-3:c+4, c-3:c+4].max(axis=(-2, -1)) * 100


rows = []
ref_cubes = {}
batches = [samples[i:i+_args.batch] for i in range(0, len(samples), _args.batch)]
for name, (heights, weights) in schemes.items():
    losses, peak_err, prof_err, EE8_err = [], [], [], []
    for b, ids in enumerate(batches):
        sl = slice(b * _args.batch, b * _args.batch + len(ids))
        P = render(ids, heights[sl], weights[sl])
        if name.startswith('reference'):
            ref_cubes[b] = P.clone()
            continue
        Rf = ref_cubes[b]
        losses.append(balanced(Rf, P).cpu().numpy())
        Pn, Rn = P.cpu().numpy(), Rf.cpu().numpy()
        peak_err.append(np.abs(Pn[..., c-3:c+4, c-3:c+4].max(axis=(-2, -1)) / Rn[..., c-3:c+4, c-3:c+4].max(axis=(-2, -1)) - 1) * 100)
        prof_err.append(profile_err(Pn, Rn))
        m8 = R <= 8.5
        EE8_err.append(np.abs(Pn[..., m8].sum(axis=-1) / Rn[..., m8].sum(axis=-1) - 1) * 100)
        release_gpu_memory()
    if name.startswith('reference'):
        print(f'{name}: rendered')
        continue
    losses, peak_err, prof_err, EE8_err = (np.concatenate(x) for x in (losses, peak_err, prof_err, EE8_err))
    row = {
        'scheme':                name,
        'balanced_loss_median':  float(np.median(losses)),
        'balanced_loss_p84':     float(np.percentile(losses, 84)),
        'peak_err_pct_median':   float(np.median(np.abs(peak_err).mean(axis=1))),
        'peak_err_pct_p84':      float(np.percentile(np.abs(peak_err).mean(axis=1), 84)),
        'EE8_err_pct_median':    float(np.median(EE8_err.mean(axis=1))),
        'profile_err_pct_median': float(np.median(prof_err.mean(axis=1))),
        **{f'profile_err_pct_{wvl_nm[l]}nm': float(np.median(prof_err[:, l])) for l in wvl_sel},
    }
    rows.append(row)
    print(f"{name:32s} balanced loss {row['balanced_loss_median']:.5f}  |peak err| {row['peak_err_pct_median']:.2f} %  "
          f"EE8 err {row['EE8_err_pct_median']:.2f} %  profile err {row['profile_err_pct_median']:.2f} %", flush=True)

table = pd.DataFrame(rows).set_index('scheme')
table.to_csv(OUT_FOLDER / 'Cn2_binning.csv')
print('\n' + table.round(4).to_string())

#%%
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.6]})
ax = axes[0]
hg, wg = schemes['fixed grid']
for n, i in enumerate(samples[:4]):
    color = f'C{n}'
    ax.plot(fracs[i], alts[i], 'o-', color=color, alpha=0.8, label='AOF profile' if n == 0 else None)
    ax.plot(kmeans_w[n], kmeans_h[n], 's:', color=color, alpha=0.6, label='K-means' if n == 0 else None)
    ax.plot(wg[n], hg[n], 'D--', color=color, alpha=0.9, markersize=8, label='fixed grid' if n == 0 else None)
for h in grid.nodes:
    ax.axhline(h, color='gray', alpha=0.4)
ax.set_xlabel('Cn2 fraction'); ax.set_ylabel('altitude [km]'); ax.set_ylim(-0.5, 30); ax.legend(); ax.set_title('Profiles of 4 samples')

ax = axes[1]
order = table.sort_values('balanced_loss_median').index
y = np.arange(len(order))
ax.barh(y, table.loc[order, 'balanced_loss_median'], xerr=table.loc[order, 'balanced_loss_p84'] - table.loc[order, 'balanced_loss_median'], color='C0', alpha=0.7)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlabel('balanced loss vs 8-layer reference (median, +p84)')
ax2 = ax.twiny()
ax2.plot(table.loc[order, 'peak_err_pct_median'], y, 'rD', label='|peak err| % (median)')
ax2.set_xlabel('|peak error| [%]', color='r'); ax2.legend(loc='lower right')
ax.set_title(f'3-layer reductions vs the full profile ({len(samples)} validation samples)')
plt.tight_layout()
plt.savefig(OUT_FOLDER / 'Cn2_binning.png', dpi=130)
print(f'Saved -> {OUT_FOLDER / "Cn2_binning.csv"} / .png')
