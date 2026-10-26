#%%
"""
Cn2 binning analysis for the MUSE NFM PSF model: how well 3-layer reductions of the AOF Cn2 profile reproduce the PSF computed with
the full (repaired) 8-layer profile, for the fitted parameters of validation samples with a recorded profile. Compared reductions:
the current weighted K-means with free altitudes (the configs of the dataset cache), the profile projected on the fixed 8-node grid
(projection error alone), the median profile (no information) and fixed altitude boundaries between the grid nodes, with per-sample
(5/3)-moment layer heights or dataset-level heights. The best boundaries go into the working copy of the grid in the STD folder
(NFM_bundle.STD_RESOURCES_FOLDER / AOF_Cn2_grid.json), from which create_STD_dataset.py packs it into the calibrator bundle.

    python machine_learning/analysis/analyze_Cn2_binning.py [--device cuda:1] [--n-samples 64] [--batch 8] [--store]

Outputs: data/analysis/NFM_calibrator/Cn2_binning.csv / .png. Needs the dataset cache and muse_df.pickle; ~5 min on a GPU.
"""
import sys
import json
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
from data_processing.Cn2_profile import Cn2Grid
from data_processing.MUSE_data_utils import raw_Cn2_profiles
from data_processing.NFM_bundle import STD_RESOURCES_FOLDER, CN2_GRID_FILE

_parser = argparse.ArgumentParser(description="Cn2 binning analysis")
_parser.add_argument('--device',    type=str, default=None)
_parser.add_argument('--n-samples', type=int, default=64)
_parser.add_argument('--batch',     type=int, default=8)
_parser.add_argument('--store',     action='store_true', help='store the best boundaries in AOF_Cn2_grid.json')
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
CANDIDATE_BOUNDARIES = [(1.0, 13.0), (4.0, 9.0), (4.0, 13.0), (4.0, 16.5), (8.0, 16.5)] # between the grid nodes (2.1 / 6.6 / 11.3 / 14.9 / 18.6 km)

#%%
dataset = load_STD_dataset(_args.derotated)
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
    muse_df = pickle.load(f)

grid = Cn2Grid.load(STD_RESOURCES_FOLDER / CN2_GRID_FILE) # the working copy (store_Cn2_grid), also with the K-means binning
alts_raw, fracs_raw = raw_Cn2_profiles(muse_df.loc[dataset.sample_ids])
alts, fracs, n_repaired, valid = grid.repair(alts_raw, fracs_raw)
grid_fracs = grid.to_grid(alts, fracs)

val_idx = np.asarray(dataset.val_idx if dataset.val_idx is not None else np.arange(len(dataset)), dtype=int)
val_idx = val_idx[valid[val_idx]]
rng     = np.random.default_rng(0)
samples = np.sort(rng.choice(val_idx, size=min(_args.n_samples, len(val_idx)), replace=False))
print(f'{grid} | {valid.sum()} / {len(dataset)} samples with a profile, {(n_repaired > 0).sum()} repaired | {len(samples)} validation samples analysed')

#%%
# Layer schemes: name -> (heights [N, L] in km, weights [N, L]) for the analysed samples
def bins_scheme(boundaries, fixed_heights=False):
    g = Cn2Grid(grid.nodes, boundaries, median_fracs=grid.median_fracs, layer_altitudes_km=grid.layer_altitudes)
    w, h = g.to_layers(grid_fracs[samples])
    if fixed_heights:
        h = np.tile(g._median_heights, (len(samples), 1))
    return h, w

kmeans_h = np.array([np.asarray(dataset.configs[i]['atmosphere']['Cn2Heights'], dtype=float).ravel() / 1e3 for i in samples])
kmeans_w = np.array([np.asarray(dataset.configs[i]['atmosphere']['Cn2Weights'], dtype=float).ravel() for i in samples])
w_med, h_med = grid.default_layers()

schemes = {
    'reference (8 AOF layers)': (alts[samples], fracs[samples]),
    'grid (8 fixed nodes)':     (np.tile(grid.nodes, (len(samples), 1)), grid_fracs[samples]),
    'K-means (current)':        (kmeans_h, kmeans_w),
    'median profile':           (np.tile(h_med, (len(samples), 1)), np.tile(w_med, (len(samples), 1))),
}
for b in CANDIDATE_BOUNDARIES:
    schemes[f'bins {b[0]:g}-{b[1]:g} km']         = bins_scheme(b)
    schemes[f'bins {b[0]:g}-{b[1]:g} km, fixed h'] = bins_scheme(b, fixed_heights=True)

# The other projection of the AOF ground layer onto the grid (pinned to the 0 km node vs. split linearly with the 2 km node)
_alt_grid = Cn2Grid(grid.nodes, grid.boundaries, median_fracs=grid.median_fracs, layer_altitudes_km=grid.layer_altitudes, pin_ground_layer=not grid.pin_ground_layer)
_alt_fracs = _alt_grid.to_grid(alts, fracs)
_alt_name  = 'GL pinned' if _alt_grid.pin_ground_layer else 'GL split'
schemes[f'grid (8 fixed nodes, {_alt_name})'] = (np.tile(grid.nodes, (len(samples), 1)), _alt_fracs[samples])
for b in CANDIDATE_BOUNDARIES:
    _g = Cn2Grid(grid.nodes, b, median_fracs=grid.median_fracs, layer_altitudes_km=grid.layer_altitudes, pin_ground_layer=_alt_grid.pin_ground_layer)
    _w, _h = _g.to_layers(_alt_fracs[samples])
    schemes[f'bins {b[0]:g}-{b[1]:g} km, {_alt_name}'] = (_h, _w)

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
# Best fixed boundaries (per-sample heights) by the median balanced loss
bins_rows = table[[s.startswith('bins') and not s.endswith('fixed h') for s in table.index]]
best_name = bins_rows['balanced_loss_median'].idxmin()
best_b    = CANDIDATE_BOUNDARIES[[f'bins {b[0]:g}-{b[1]:g} km' in best_name for b in CANDIDATE_BOUNDARIES].index(True)]
best_pin  = (_alt_grid.pin_ground_layer if best_name.endswith(_alt_name) else grid.pin_ground_layer)
print(f'\nBest boundaries: {best_name} (K-means: {table.loc["K-means (current)", "balanced_loss_median"]:.5f}, '
      f'grid: {table.loc["grid (8 fixed nodes)", "balanced_loss_median"]:.5f}, median profile: {table.loc["median profile", "balanced_loss_median"]:.5f})')

#%%
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.6]})
ax = axes[0]
for n, i in enumerate(samples[:4]):
    color = f'C{n}'
    ax.plot(fracs[i], alts[i], 'o-', color=color, alpha=0.8, label='AOF profile' if n == 0 else None)
    ax.plot(grid_fracs[i], grid.nodes, 's:', color=color, alpha=0.6, label='fixed grid' if n == 0 else None)
    hb, wb = schemes[best_name]
    ax.plot(wb[n], hb[n], 'D--', color=color, alpha=0.9, markersize=8, label=f'{best_name}' if n == 0 else None)
for b in best_b:
    ax.axhline(b, color='gray', alpha=0.4)
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

#%%
if _args.store:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'datasets' / 'MUSE_STD_dataset'))
    from STD_dataset_utils import store_Cn2_grid
    store_Cn2_grid(muse_df, boundaries_km=best_b, pin_ground_layer=best_pin)
    print(f'Stored the grid with the boundaries {best_b} km, ground layer {"pinned" if best_pin else "split"}')
