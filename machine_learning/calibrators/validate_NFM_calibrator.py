#%%
import json
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import sys
# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from tiptorch.tools.utils import r0
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from calibrators.NFM_calibrator import (build_PSF_model,
    NFMDataset,
    load_STD_dataset,
    build_NFM_trainer,
    load_NFM_calibrator,
    load_telemetry_preprocessor,
    release_gpu_memory,
    telemetry_base_params,
    Cn2_grid_features_from_GLF,
    CONFIGS_FOLDER
)
from data_processing.Cn2_profile import Cn2Grid
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack
from data_processing.telemetry_preprocessor import TelemetryPreprocessor # after calibrators (adds the repo root to sys.path)

logger = logging.getLogger(__name__)

STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE = STD_FOLDER / 'dataset_cache'

#%%
# ── Which run to validate ─────────────────────────────────────────────────────
# A run is given by its bundle <name>_bundle.pth, which train_NFM_calibrator.py writes next to <name>_checkpoint.pth and <name>_meta.json:
# --bundle <path>, or by default the most recently written NFM calibrator bundle under WEIGHTS_FOLDER. The trainer is rebuilt from the
# training config stored in the bundle, never from a config file, which may have been edited since.
def _load_bundle(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def _latest_bundle():
    """ (path, state) of the most recently written NFM calibrator bundle under WEIGHTS_FOLDER (not an ETC one), (None, None) if there is none. """
    for path in sorted(WEIGHTS_FOLDER.glob('*/*_bundle.pth'), key=lambda p: p.stat().st_mtime, reverse=True):
        state = _load_bundle(path)
        if 'calibrator_type' in state:
            return path, state
    return None, None

_vparser = argparse.ArgumentParser(description="Validate NFM Calibrator")
_vparser.add_argument('--bundle', help='calibrator bundle of the run to validate (default: the most recent one under the weights folder)')
try:
    _bundle_path = _vparser.parse_known_args()[0].bundle # known args only: IPython kernels add their own (--f=...)
except SystemExit:
    _bundle_path = None

if _bundle_path:
    _bundle_path = Path(_bundle_path)
    if not _bundle_path.exists():
        raise FileNotFoundError(f'Calibrator bundle not found: {_bundle_path}')
    _bundle_state = _load_bundle(_bundle_path)
else:
    _bundle_path, _bundle_state = _latest_bundle()
    if _bundle_path is None:
        raise RuntimeError(f'No NFM calibrator bundle under {WEIGHTS_FOLDER}: run train_NFM_calibrator.py first, or pass --bundle <path>')

cfg = _bundle_state['training_config']
logger.info(f"Validating the run of the bundle {_bundle_path}: {cfg['weights_subdir']} / {cfg['name']} ({_bundle_state['calibrator_type']} calibrator)")

#%%
# ── Run metadata, next to the bundle: split, output names, training losses ───
_meta_path = _bundle_path.with_name(f'{cfg["name"]}_meta.json')
if not _meta_path.exists():
    raise FileNotFoundError(f"Training metadata not found: {_meta_path}. Run train_NFM_calibrator.py first.")

with open(_meta_path) as _f:
    train_meta = json.load(_f)

output_names    = train_meta['output_names']
N_wvl_total     = train_meta['N_wvl_total']
N_outputs       = train_meta['N_outputs']
input_features  = train_meta['input_feature_names']

logger.info(f"Metadata loaded: name={train_meta['name']} | best_val={train_meta['best_val_loss']:.6f} | completed={train_meta['training_completed_at']}")

#%%
# ── Ensure consistent dataset split ──────────────────────────────────────────
_split_path = Path(train_meta['split_path'])

# ── Initialize trainer, unless the session already has one built for this very run ──────────────────────────
# BEST_CALIB_PATH, set by train_NFM_calibrator.py or by a previous validation in this session, tells which run the session's trainer is for
_checkpoint_path = _bundle_path.with_name(f'{cfg["name"]}_checkpoint.pth')
if 'trainer' not in dir() or 'BEST_CALIB_PATH' not in dir() or Path(BEST_CALIB_PATH).resolve() != _checkpoint_path.resolve():
    # The regime the calibrator was trained in: derotated PSFs or raw frames (per-OB pupil rotation), see train_NFM_calibrator.py
    dataset = load_STD_dataset(cfg['derotated_PSFs'])
    logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W}), {'derotated PSFs' if dataset.derotated else 'raw frames'}")

    PSF_model, calibrator, outputs_transformer, trainer = build_NFM_trainer(cfg, dataset, default_device, weights_dir=WEIGHTS_FOLDER / cfg['weights_subdir'])
    release_gpu_memory()
    logger.info("Trainer initialized from scratch (standalone mode).")

BEST_CALIB_PATH = _checkpoint_path

# The validation must use exactly the split the calibrator was trained with, matched by sample IDs (the dataset cache may have been
# regenerated since training)
with np.load(_split_path) as _f:
    _split = dict(_f) # read and close, otherwise the file stays locked on Windows

_id_to_pos = {sample_id: pos for pos, sample_id in enumerate(dataset.sample_ids)}
_lost_ids  = [i for i in np.concatenate([_split['train_ids'], _split['val_ids']]) if i not in _id_to_pos]
if _lost_ids:
    raise ValueError(f"{len(_lost_ids)} samples of the training split are missing in the current dataset cache "
                     f"(e.g. {_lost_ids[:5]}). The calibrator was trained on a different dataset.")
_train_idx = np.array([_id_to_pos[i] for i in _split['train_ids']], dtype=int)
_val_idx   = np.array([_id_to_pos[i] for i in _split['val_ids']],   dtype=int)

if set(_val_idx) != set(dataset.val_idx):
    logger.warning("The calibrator was trained with a split different from the one stored in the dataset cache.")

if not (np.array_equal(_train_idx, trainer.train_idx) and np.array_equal(_val_idx, trainer.val_idx)):
    logger.info("Restoring the training split.")
    trainer.train_idx    = _train_idx
    trainer.val_idx      = _val_idx
    trainer.train_loader = trainer._make_loader(trainer.train_idx, shuffle=True)
    trainer.val_loader   = trainer._make_loader(trainer.val_idx,   shuffle=False)
logger.info(f"Split: {len(trainer.train_idx)} train / {len(trainer.val_idx)} val")


#%% ========================== Evaluation & diagnostics ===============================================
if not BEST_CALIB_PATH.exists():
    raise FileNotFoundError(f"Checkpoint of the validated run not found: {BEST_CALIB_PATH}")
trainer.load_checkpoint(BEST_CALIB_PATH, load_optimizer=False) # the optimizer state is not needed for the validation
calibrator.eval()

PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, telemetry_vecs, final_val_loss = trainer.validate(return_cubes=True)

PSFs_pred_cube = PSFs_pred_cube.cpu()
PSFs_data_cube = PSFs_data_cube.cpu()
validation_ids = validation_ids.cpu()

# Network outputs -> PSF-model parameters -> normalized parameter space (the direct calibrator's output space). For the
# direct calibrator NN_params == NN_predictions; for the factor one NN_predictions are corrections, see FactorParameterization
with torch.no_grad():
    x_dict_pred = trainer._params_from_pred(NN_predictions.to(default_device), telemetry_vecs.to(default_device))
    NN_params   = trainer._params_to_normalized(x_dict_pred).cpu()
    x_dict_pred = {k: v.cpu() for k, v in x_dict_pred.items()}

release_gpu_memory()
print(f"Validation loss: {final_val_loss:.6f} | PSF cubes {PSFs_pred_cube.shape}")


#%%
stds  = NN_params.std(dim=0)
means = NN_params.mean(dim=0)

plt.bar(np.arange(len(means)), means.cpu().numpy(), yerr=stds.cpu().numpy(), alpha=0.7, capsize=3)
plt.xticks(ticks=np.arange(len(means)), labels=output_names, rotation=45, ha='right')

selected_mean_features = []
feature_keywords = {'J_ctrl', 'J_aniso', 'wing_frac_ctrl', 'F_ctrl', 'LO_coefs', 'wind_speed_single', 'L0'}

for i, name in enumerate(output_names):
    if any(keyword in name for keyword in feature_keywords):
        selected_mean_features.append((name, means[i].item(), stds[i].item()))

#%%
from tiptorch.tools.utils import r0

# Telemetry scaler + imputer the validated calibrator was trained with. It is stored in the bundle (the checkpoint holds only the
# network weights); without a bundle, the one of the dataset cache is the same by construction (see NFMCalibratorTrainer.save_calibrator)
telemetry_preprocessor = load_telemetry_preprocessor(_bundle_state) # the bundle of the validated run (see above)

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as fh:
    muse_df = pickle.load(fh)

x_vec = telemetry_preprocessor.inverse_transform(telemetry_vecs.cpu().numpy())

# Telemetry-derived Cn2 profile (the base of the factor calibrator: AOF profile on the fixed grid binned into the model layers,
# or the K-means fractions)
_Cn2_base = telemetry_base_params(x_vec, input_features, Cn2_grid=Cn2Grid.from_state_dict(dataset.Cn2_grid) if dataset.Cn2_grid is not None else None)['Cn2_weights'].numpy()
X = {
    'r0': r0(muse_df['Seeing (header)'].loc[validation_ids].to_numpy(), 500e-9),
    'L0': muse_df['L0Tot'].loc[validation_ids].to_numpy(),
    'wind_speed_single': muse_df['Wind speed (header)'].loc[validation_ids].to_numpy(),
    'Cn2_weights_1': _Cn2_base[:, 0],
    'Cn2_weights_2': _Cn2_base[:, 1],
    'Cn2_weights_3': _Cn2_base[:, 2],
}

features_of_interest = list(X.keys()) + ['Cn2_weights']

Y = {k: v.numpy() for k, v in x_dict_pred.items() if k in features_of_interest}
Y['Cn2_weights_1'] = Y['Cn2_weights'][:, 0]
Y['Cn2_weights_2'] = Y['Cn2_weights'][:, 1]
Y['Cn2_weights_3'] = Y['Cn2_weights'][:, 2]

Y['wind_speed_single'] = np.abs(Y['wind_speed_single'])

_ = Y.pop('Cn2_weights', None)

#%%
common_keys = [k for k in X if k in Y]
n_keys = len(common_keys)
ncols  = min(3, n_keys)
nrows  = int(np.ceil(n_keys / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
axes_flat = np.array(axes).ravel()

for i, key in enumerate(common_keys):
    ax = axes_flat[i]
    x_vals = np.asarray(X[key]).ravel()
    y_vals = np.asarray(Y[key]).ravel()

    bins = np.linspace(min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max()), 30)
    ax.hist(x_vals, bins=bins, alpha=0.6, label='Input (X)', color='tab:blue',   density=True)
    ax.hist(y_vals, bins=bins, alpha=0.6, label='Predicted (Y)', color='tab:orange', density=True)
    ax.set_title(key)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for ax in axes_flat[n_keys:]:
    ax.set_visible(False)

fig.suptitle('Input vs. Predicted distributions', fontsize=13)
plt.tight_layout()
plt.show()


#%% ======================================================================================================================================================================
#%  ======================================================================================================================================================================
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as fh:
    muse_df = pickle.load(fh)

lambda_full = trainer.lambda_full
id_src      = np.random.randint(0, PSFs_data_cube.shape[0])
true_id     = validation_ids[id_src].item()

# true_id = 382
# id_src = torch.where(validation_ids == true_id)[0].item()

wvl_select  = np.s_[0, N_wvl_total // 2, -1]

PSF_0 = PSFs_data_cube[id_src]
PSF_1 = PSFs_pred_cube[id_src]

vmin  = np.percentile(PSF_0[PSF_0 > 0].numpy(), 10)
vmax  = np.percentile(PSF_0[PSF_0 > 0].numpy(), 99.995)

fig, axes = plt.subplots(1, len(wvl_select), figsize=(15, 1.35 * len(wvl_select)))
p_errs = []

for i, lmbd in enumerate(wvl_select):
    err = plot_radial_PSF_profiles(
        PSF_0[lmbd].numpy(),
        PSF_1[lmbd].numpy(),
        'Data',
        'TipTorch',
        cutoff = 40,
        y_min = 3e-2,
        linthresh = 1e-2,
        return_profiles = True,
        ax = axes[i],
    )[2].squeeze().max().item()
    
    axes[i].set_title(f"lambda = {int((lambda_full[lmbd] * 1e9).round().item())} nm")
    p_errs.append(err)
    
plt.tight_layout()
plt.show()
print(f"ΔSR per wavelength: {np.array(p_errs)}")

draw_PSF_stack(
    PSF_0.numpy()[wvl_select, ...],
    PSF_1.numpy()[wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=80,
    cmap='inferno',
)
plt.title(muse_df.loc[true_id]['Filename']); plt.show()

#%%
fig, axes = plt.subplots(1, len(wvl_select), figsize=(15, 1.35 * len(wvl_select)))
p_errs = []

for i, lmbd in enumerate(wvl_select):
    errs = plot_radial_PSF_profiles(
        PSFs_data_cube[:, lmbd, ...].cpu().numpy(),
        PSFs_pred_cube[:, lmbd, ...].cpu().numpy(),
        'Data',
        'TipTorch',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        ax=axes[i],
        return_profiles=True,
    )
    axes[i].set_title(f"λ = {int((lambda_full[lmbd] * 1e9).round().item())} nm")
    p_errs.append(errs)
    
plt.tight_layout()
plt.show()

#%%
# Spectrally-averaged radial profiles across the validation set
fig = plt.figure(figsize=(10, 6))
PSF_avg = lambda x: x.cpu().numpy().mean(axis=1)

plot_radial_PSF_profiles(
    PSF_avg(PSFs_data_cube),
    PSF_avg(PSFs_pred_cube),
    'Data',
    'TipTorch',
    title='Spectrally averaged PSF',
    cutoff=40,
    ax=fig.add_subplot(111),
)
plt.tight_layout()
plt.show()

#%%
# Per-sample loss histogram
# Use the trainer's loss computation for consistency
with torch.no_grad(): # in chunks on the trainer's device (the balanced loss keeps its weight maps there)
    loss_SR = torch.cat([
        trainer._loss_per_sample(
            PSFs_data_cube[i:i+8].to(default_device),
            PSFs_pred_cube[i:i+8].to(default_device),
            {k: v[i:i+8].to(default_device) for k, v in x_dict_pred.items()}
        ).cpu()
        for i in range(0, len(PSFs_data_cube), 8)
    ])

num_best_samples  = 50
num_worst_samples = 15

best_sample_ids  = torch.topk(loss_SR, k=num_best_samples,  largest=False).indices
worst_sample_ids = torch.topk(loss_SR, k=num_worst_samples, largest=True).indices

true_best_ids  = validation_ids[best_sample_ids].cpu().numpy()
true_worst_ids = validation_ids[worst_sample_ids].cpu().numpy()

bad_border_start  = loss_SR[worst_sample_ids].min().item()
mid_boarder_start = loss_SR[best_sample_ids].max().item()

sus_ids = torch.where((loss_SR >= mid_boarder_start) & (loss_SR < bad_border_start))[0]
true_sus_ids = validation_ids[sus_ids].cpu().numpy()

# Sort true IDs by loss for better interpretability
true_best_ids  = true_best_ids.tolist()
true_worst_ids = true_worst_ids.tolist()
true_sus_ids   = true_sus_ids.tolist()

true_worst_ids.sort()
true_best_ids.sort()
true_sus_ids.sort()

_ = plt.hist(loss_SR.cpu().numpy(), bins=50)#, log=True)
plt.axvspan(0, loss_SR[best_sample_ids].max().item(), color='g', alpha=0.2, label='Best samples (low loss)')
plt.axvspan(bad_border_start, loss_SR.max().item(), color='r', alpha=0.2, label='Bad samples (high loss)')
plt.axvspan(mid_boarder_start, bad_border_start, color='y', alpha=0.2, label='Middle samples (medium loss)')
plt.xlabel('Per-sample loss')
plt.ylabel('Count')
plt.legend(); plt.grid(True); plt.show()

print(f"Best predicted sample IDs (lowest loss): {true_best_ids}")
print(f"Worst predicted sample IDs (highest loss): {true_worst_ids}")
print(f"Middle sample IDs (medium loss): {true_sus_ids}")

#%% ===================================== Residual diagnostics =====================================
# Where does the residual live? Radial profiles of the residuals (linear and log), the share of the loss coming from
# each radial band, 2D residual maps (linear and log), and the core accuracy (peak, encircled energy) vs. wavelength.
D_np = PSFs_data_cube.numpy()
P_np = PSFs_pred_cube.numpy()
_, _, H_, W_ = D_np.shape
yy, xx = np.mgrid[:H_, :W_]
R_pix = np.hypot(yy - H_//2, xx - W_//2)
R_int = np.round(R_pix).astype(int).ravel()
R_cnt = np.bincount(R_int)
λ_nm  = (lambda_full.cpu().numpy() * 1e9).round().astype(int)

def radial_mean(cube):
    ''' Azimuthal average around the frame center: [..., H, W] -> [..., N_r] '''
    flat = cube.reshape(-1, H_*W_)
    prof = np.stack([np.bincount(R_int, weights=f, minlength=len(R_cnt)) for f in flat]) / R_cnt
    return prof.reshape(*cube.shape[:-2], -1)

prof_D, prof_P = radial_mean(D_np), radial_mean(P_np)
prof_abs = radial_mean(np.abs(P_np - D_np))
r_max_plot = 60
r_ax = np.arange(r_max_plot)
wvl_diag = [0, N_wvl_total//2, N_wvl_total-1]

fig, axes = plt.subplots(2, len(wvl_diag), figsize=(15, 8), sharex=True)
for j, li in enumerate(wvl_diag):
    ax = axes[0, j]
    ax.semilogy(r_ax, np.median(prof_D[:, li, :r_max_plot], 0), 'k', label='Data')
    ax.semilogy(r_ax, np.median(prof_P[:, li, :r_max_plot], 0), 'tab:orange', label='TipTorch')
    ax.semilogy(r_ax, np.median(prof_abs[:, li, :r_max_plot], 0), 'tab:orange', ls=':', label='|TipTorch - Data|')
    ax.set_title(f'λ = {λ_nm[li]} nm'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, j]
    rel_bias = (prof_P - prof_D)[:, li, :r_max_plot] / np.abs(prof_D[:, li, :r_max_plot]).clip(1e-12)
    ax.plot(r_ax, np.median(rel_bias, 0), 'tab:orange', label='median (model-data)/data')
    ax.fill_between(r_ax, *np.percentile(rel_bias, [16, 84], 0), color='tab:orange', alpha=0.2, label='16-84%')
    ax.plot(r_ax, np.median(prof_abs[:, li, :r_max_plot] / np.abs(prof_D[:, li, :r_max_plot]).clip(1e-12), 0), 'tab:red', ls=':', label='median <|Δ|>/data')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-1, 1.5); ax.set_xlabel('r [pix]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes[0, 0].set_ylabel('Normalized intensity'); axes[1, 0].set_ylabel('Relative residual')
plt.tight_layout(); plt.show()

#%%
# Share of the loss contributed by each radial band, compared to the number of pixels and to the PSF flux in it
bands = [0, 1.5, 3.5, 6.5, 10.5, 20.5, 40.5, 70.5, R_pix.max() + 1]
band_names = [f'{a:g}-{b:g}' for a, b in zip(bands[:-1], bands[1:-1] + [np.inf])]

diff_np = P_np - D_np
shares = {
    'legacy MSE term': (1200 * diff_np**2).sum(axis=(0, 1)),
    'legacy MAE term': (1.6 * np.abs(diff_np)).sum(axis=(0, 1)),
}

if trainer.balanced_loss is not None:
    with torch.no_grad():
        shares['balanced loss'] = sum(
            trainer.balanced_loss.pixel_map(PSFs_data_cube[i:i+8].to(default_device), PSFs_pred_cube[i:i+8].to(default_device)).sum(dim=(0, 1)).cpu().numpy()
            for i in range(0, len(PSFs_data_cube), 8)
        )
shares['data flux'] = D_np.clip(0).sum(axis=(0, 1))

fig, ax = plt.subplots(figsize=(10, 4))
x_b = np.arange(len(band_names))
for k, (name, pix_map) in enumerate(shares.items()):
    band_share = np.array([pix_map[(R_pix >= a) & (R_pix < b)].sum() for a, b in zip(bands[:-1], bands[1:])]) / pix_map.sum()
    ax.bar(x_b + (k - len(shares)/2 + 0.5) * 0.8/len(shares), band_share, width=0.8/len(shares), label=name)
ax.set_xticks(x_b); ax.set_xticklabels(band_names)
ax.set_xlabel('Radial band [pix]'); ax.set_ylabel('Share'); ax.set_title('Share of the loss / flux per radial band (validation set)')
ax.legend(); ax.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.show()

#%%
# 2D residual maps: linear (mean residual normalized by the data peak) and log (ratio of the median model to the median data)
crop_diag = 25
wvl_maps = [0, N_wvl_total//3, 2*N_wvl_total//3, N_wvl_total-1]
c_ = H_ // 2
sl_ = np.s_[:, c_-crop_diag:c_+crop_diag+1, c_-crop_diag:c_+crop_diag+1]

fig, axes = plt.subplots(2, len(wvl_maps), figsize=(4*len(wvl_maps), 7.5))
for j, li in enumerate(wvl_maps):
    d, p = D_np[:, li][sl_], P_np[:, li][sl_]
    lin_map = ((p - d) / d.max(axis=(-2, -1), keepdims=True)).mean(0)
    v = np.abs(lin_map).max()
    im = axes[0, j].imshow(lin_map, cmap='RdBu_r', vmin=-v, vmax=v)
    plt.colorbar(im, ax=axes[0, j], fraction=0.046)
    axes[0, j].set_title(f'<(model-data)/peak>, {λ_nm[li]} nm', fontsize=9)

    log_map = np.log10(np.median(p, 0).clip(1e-9) / np.median(d, 0).clip(1e-9))
    im = axes[1, j].imshow(log_map, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=axes[1, j], fraction=0.046)
    axes[1, j].set_title(f'log10(median model / median data), {λ_nm[li]} nm', fontsize=9)
plt.tight_layout(); plt.show()

#%%
# Core accuracy vs. wavelength: peak and encircled energy ratios (model / data)
c3 = np.s_[..., c_-3:c_+4, c_-3:c_+4]
peak_ratio = P_np[c3].max(axis=(-2, -1)) / D_np[c3].max(axis=(-2, -1))
fig, ax = plt.subplots(figsize=(10, 4))
for name, ratio, col in [('peak', peak_ratio, 'tab:red')] + \
        [(f'EE(r≤{rr})', P_np[..., R_pix <= rr].sum(-1) / D_np[..., R_pix <= rr].sum(-1), c) for rr, c in ((2, 'tab:blue'), (8, 'tab:green'))]:
    ax.plot(λ_nm, np.median(ratio, 0), color=col, label=name)
    ax.fill_between(λ_nm, *np.percentile(ratio, [16, 84], 0), color=col, alpha=0.15)
ax.axhline(1, color='k', lw=0.5)
ax.set_xlabel('λ [nm]'); ax.set_ylabel('model / data'); ax.set_title('Core accuracy: median and 16-84% range over the validation set')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

#%%
# Worst offenders: per-sample loss, core error and the conditions of the observation
worst_table = pd.DataFrame({
    'ID':         validation_ids.numpy(),
    'loss':       loss_SR.numpy(),
    'peak ratio': peak_ratio.mean(axis=1),
    'peak data':  D_np[c3].max(axis=(-2, -1)).mean(axis=1),
}).set_index('ID')
_cols = [c for c in ('Seeing (header)', 'Tau0 (header)', 'NGS mag (from ph.)', 'Exp. time', 'Filename') if c in muse_df.columns]
worst_table = worst_table.join(muse_df[_cols]).sort_values('loss', ascending=False)
print(worst_table.head(15).to_string(float_format=lambda v: f'{v:.4f}'))
print(f"Top-10 samples hold {worst_table['loss'].head(10).sum() / worst_table['loss'].sum():.1%} of the total validation loss")

#%%
peak_err = (PSFs_data_cube - PSFs_pred_cube).amax(dim=(-2,-1)) / PSFs_data_cube.amax(dim=(-2,-1))
peak_err = peak_err.cpu().numpy()[:, wvl_select,...]

profile_errs = np.stack([p[2].max(axis=-1) for p in p_errs])

loss_np = loss_SR.cpu().numpy()
x_fit = np.linspace(loss_SR.min().item(), loss_SR.max().item(), 100)

# Calculate correlations between loss and profile errors
print("\nCorrelation between loss function and profile errors:")
print(f"{'Wavelength [nm]':<20} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
print("-" * 70)

for wvl in range(len(wvl_select)):
    wvl_nm = int((lambda_full[wvl_select[wvl]] * 1e9).round().item())
    errs = profile_errs[wvl] / 100.0
    
    pearson_r,  pearson_p  = pearsonr (loss_np, errs)
    spearman_r, spearman_p = spearmanr(loss_np, errs)
    
    print(f"{wvl_nm:<20} {pearson_r:>12.4f} {pearson_p:>12.2e} {spearman_r:>12.4f} {spearman_p:>12.2e}")
    
    plt.scatter(loss_SR, errs, s=5, alpha=0.6, label=f"λ = {wvl_nm} nm (r={pearson_r:.3f})")
    
    # Add best fitting line per wavelength
    coeffs = np.polyfit(loss_np, errs, 1)
    fit_line = np.poly1d(coeffs)
    plt.plot(x_fit, fit_line(x_fit), '--', linewidth=1.5, alpha=0.8)

# Overall correlation across all wavelengths
all_loss = np.tile(loss_np, len(wvl_select))
all_errs = profile_errs.flatten() / 100
pearson_all,  pearson_p_all  = pearsonr (all_loss, all_errs)
spearman_all, spearman_p_all = spearmanr(all_loss, all_errs)
print(f"{'All wavelengths':<20} {pearson_all:>12.4f} {pearson_p_all:>12.2e} {spearman_all:>12.4f} {spearman_p_all:>12.2e}")

plt.xlabel('Per-sample loss')
plt.ylabel('Max ΔSR across radii')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.ylim(0, None)
plt.tight_layout()
plt.show()

#%%
print("\nCorrelation between peak pixel-wise error and profile errors:")

for wvl in range(len(wvl_select)):
    wvl_nm = int((lambda_full[wvl_select[wvl]] * 1e9).round().item())
    errs = profile_errs[wvl]/100.0
    
    peak_errs_wvl = peak_err[:, wvl]
    
    pearson_r,  pearson_p  = pearsonr (peak_errs_wvl, errs)
    spearman_r, spearman_p = spearmanr(peak_errs_wvl, errs)
    
    print(f"{wvl_nm:<20} {pearson_r:>12.4f} {pearson_p:>12.2e} {spearman_r:>12.4f} {spearman_p:>12.2e}")
    
    plt.scatter(peak_errs_wvl, errs, s=5, alpha=0.6, label=f"λ = {wvl_nm} nm (r={pearson_r:.3f})")
    
    # Add best fitting line per wavelength
    coeffs = np.polyfit(peak_errs_wvl, errs, 1)
    fit_line = np.poly1d(coeffs)
    plt.plot(peak_errs_wvl, fit_line(peak_errs_wvl), '--', linewidth=1.5, alpha=0.8)

# Overall correlation across all wavelengths
# all_loss = np.tile(loss_np, len(wvl_select))
# all_errs = profile_errs.flatten() / 100
# pearson_all,  pearson_p_all  = pearsonr(all_loss, all_errs)
# spearman_all, spearman_p_all = spearmanr(all_loss, all_errs)
# print(f"{'All wavelengths':<20} {pearson_all:>12.4f} {pearson_p_all:>12.2e} {spearman_all:>12.4f} {spearman_p_all:>12.2e}")

plt.xlabel('Peak relative error (max pixel-wise ΔSR)')
plt.ylabel('Max ΔSR across radii')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.ylim(0, None)
plt.tight_layout()
plt.show()


#%%
# Radial profiles at selected wavelengths for best, middle, and worst samples
wvl_select = [0, N_wvl_total//2, -1]

# Plot for each group
groups = [
    (PSFs_data_cube[best_sample_ids,...],  PSFs_pred_cube[best_sample_ids,...],  'Best samples (low loss)',      'tab:green'),
    (PSFs_data_cube[worst_sample_ids,...], PSFs_pred_cube[worst_sample_ids,...], 'Worst samples (high loss)',    'tab:red'),
    (PSFs_data_cube[sus_ids,...],          PSFs_pred_cube[sus_ids,...],          'Middle samples (medium loss)', 'tab:orange'),
]

for PSF_data_avg, PSF_pred_avg, group_name, color in groups:
    fig, axes = plt.subplots(1, len(wvl_select), figsize=(15, 4.5))
    
    for i, lmbd in enumerate(wvl_select):
        plot_radial_PSF_profiles(
            PSF_data_avg[:,lmbd,...].cpu().numpy(),
            PSF_pred_avg[:,lmbd,...].cpu().numpy(),
            'Data',
            'TipTorch',
            cutoff=40,
            y_min=3e-2,
            linthresh=1e-2,
            ax=axes[i],
        )
        axes[i].set_title(f"λ = {int((lambda_full[lmbd] * 1e9).round().item())} nm")
    
    fig.suptitle(f'{group_name}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    # plt.savefig(STD_FOLDER / f'plots/calibrated_prediction_{group_name.split()[0].lower()}_wavelengths.pdf', dpi=300)


#%%
for PSF_data_avg, PSF_pred_avg, group_name, color in groups:
    fig = plt.figure(figsize=(10, 6))
    plot_radial_PSF_profiles(
        PSF_avg(PSF_data_avg),
        PSF_avg(PSF_pred_avg),
        'Data',
        'TipTorch',
        title=f'Spectrally averaged PSF ({group_name})',
        cutoff=40,
        ax=fig.add_subplot(111),
    )
    plt.tight_layout()
    plt.show()
    
    
#%% ======================================= ETC Performance Analysis =======================================
# ETC-like validation: for every validation-set sample, the atmospheric/HO-WFS entries of its config
# are replaced by ESO Paranal ambient-condition bins (turbulence categories, TC), keyed by percentile.
# The TC of a sample is picked as the one whose reference seeing is closest to the sample's own seeing.
from collections import Counter
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations

WIND_SPEED = { 10: 10.4,  20: 10.5,  30: 9.9,   50: 10.1,  70: 12.8,   85: 14.5,   100: 15    }
WIND_DIR   = { 10: 12.5,  20: 28.8,  30: 40.2,  50: 37.2,  70: 40.0,   85: 37.0,   100: 37.0  }
L0_TC      = { 10: 12.48, 20: 17.1,  30: 22.1,  50: 23.5,  70: 24,     85: 24.8,   100: 25.5  }
GLF_TC     = { 10: 0.971, 20: 0.973, 30: 0.980, 50: 0.972, 70: 0.976,  85: 0.977,  100: 0.977 }
SEEING_TC  = { 10: 0.6,   20: 0.7,   30: 0.8,   50: 1.0,   70: 1.15,   85: 1.4,    100: 1.6   }

ETC_HO_PHOTONS  = 1740.2396627862904
ETC_LGS_PHOTON_RATE_SCALE = 1240e3  # subapertures * HO frames per second
ETC_REFERENCE_WAVELENGTH  = 500e-9  # seeing/r0 reference wavelength [m]
ETC_TC_BINS     = sorted(SEEING_TC.keys())
ETC_SEEING_VALS = np.array([SEEING_TC[tc] for tc in ETC_TC_BINS])


def seeing_to_TC(seeing_value):
    """Map a (header) seeing value [arcsec] onto the nearest ETC turbulence category (TC) bin."""
    return ETC_TC_BINS[int(np.argmin(np.abs(ETC_SEEING_VALS - seeing_value)))]


def build_ETC_config(base_config, TC):
    """Return a copy of a per-sample config with atmosphere/HO-WFS entries replaced by ETC 'white profile' values for TC."""
    etc_cfg = deepcopy(base_config)
    glf = GLF_TC[TC]

    etc_cfg['atmosphere']['Seeing']        = [SEEING_TC[TC]]
    etc_cfg['atmosphere']['L0']            = [L0_TC[TC]]
    # The calibrator predicts three Cn2 weights. Keep the ETC profile's two
    # active layers and add a zero-weight placeholder for shape compatibility.
    etc_cfg['atmosphere']['Cn2Weights']    = [[glf, 1.0 - glf, 0.0]]
    etc_cfg['atmosphere']['Cn2Heights']    = [[0, 2000, 20000]]
    etc_cfg['atmosphere']['WindSpeed']     = [[WIND_SPEED[TC]] * 3]
    etc_cfg['atmosphere']['WindDirection'] = [[WIND_DIR[TC]] * 3]
    etc_cfg['sensor_HO']['NumberPhotons']  = [[ETC_HO_PHOTONS] * 4]
    etc_cfg['DM']['NumberReconstructedLayers'] = 3

    return etc_cfg


def build_ETC_telemetry(telemetry_vectors, TCs, calibrator):
    """
    Build physical-unit ETC telemetry from normalized validation telemetry.

    Observation-specific quantities for which ETC has no replacement (target, pointing, IRLOS setup, etc.) are retained. Atmospheric and HO-WFS features
    are replaced by ETC values or quantities derived from them. AO diagnostics unavailable to ETC are set to NaN so the fitted telemetry imputer estimates
    them from the ETC conditions instead of leaking the observed values.
    """
    if len(telemetry_vectors) != len(TCs):
        raise ValueError(f"Expected one ETC category per telemetry row, got {len(TCs)} categories for {len(telemetry_vectors)} rows.")

    telemetry_physical = pd.DataFrame(
        calibrator.telemetry_preprocessor.inverse_transform(telemetry_vectors.cpu().numpy()),
        columns=calibrator.features,
    )

    seeing   = np.asarray([SEEING_TC[tc]  for tc in TCs], dtype=np.float64)
    GLF      = np.asarray([GLF_TC[tc]     for tc in TCs], dtype=np.float64)
    wind     = np.asarray([WIND_SPEED[tc] for tc in TCs], dtype=np.float64)
    wind_dir = np.asarray([WIND_DIR[tc]   for tc in TCs], dtype=np.float64)

    seeing_rad = np.deg2rad(seeing / 3600.0)
    r0 = 0.98 * ETC_REFERENCE_WAVELENGTH / seeing_rad
    k = 2.0 * np.pi / ETC_REFERENCE_WAVELENGTH
    Cn2_integral = r0 ** (-5.0 / 3.0) / (0.423 * k**2)
    tau0 = 0.314 * r0 / wind
    free_atmosphere_seeing = seeing * (1.0 - GLF) ** (3.0 / 5.0)

    replacements = {
        'Seeing (header)':               seeing,
        'Tau0 (header)':                 tau0,
        'Wind dir (header)':             wind_dir,
        'Wind speed (header)':           wind,
        'Free Atmosphere Seeing ["]':    free_atmosphere_seeing,
        'MASS Tau0 [s]':                 tau0,
        'MASS-DIMM Turb Velocity [m/s]': wind,
        'MASS_FRACGL':                   GLF,
        'IA_FWHMLINOBS':                 seeing,
        'LGS_TUR_ALT':                   1.0 - GLF,
        'LGS photons, [photons/m^2/s]':  ETC_HO_PHOTONS * ETC_LGS_PHOTON_RATE_SCALE,
        'MASS_TURB total':               Cn2_integral,
        'Cn2 fraction below 2000m':      GLF,
        'Cn2_alt_binned_2':              np.full_like(seeing, 2.0),
        'Cn2_frac_binned_2':             1.0 - GLF,
        'Cn2_alt_binned_3':              np.full_like(seeing, 20.0),
        'Cn2_frac_binned_3':             np.zeros_like(seeing),
    }
    if calibrator.Cn2_grid is not None: # grid profile features: GLF on the ground node, the median profile above
        replacements.update(Cn2_grid_features_from_GLF(calibrator.Cn2_grid, GLF))

    for feature, values in replacements.items():
        if feature in telemetry_physical:
            telemetry_physical[feature] = values

    # These are measured closed-loop performance diagnostics, not ETC inputs.
    # Impute them from the ETC conditions rather than retaining information from
    # the actual observation in an otherwise synthetic ETC prediction.
    for feature in ('LGS_STREHL', 'LGS_TURVAR_RES', 'LGS_FWHM_GAIN', 'LGS_SLOPERMS', 'theta0'):
        if feature in telemetry_physical:
            telemetry_physical[feature] = np.nan

    return telemetry_physical


# ── Build the ETC ('white profile') config for every validation-set sample ──
true_val_ids = validation_ids.cpu().numpy().tolist()

ETC_configs, ETC_TCs = [], []
for true_id in true_val_ids:
    seeing_value = dataset.configs[true_id]['atmosphere']['Seeing'][0]
    TC = seeing_to_TC(seeing_value)
    ETC_TCs.append(TC)
    ETC_configs.append(build_ETC_config(dataset.configs[true_id], TC))

print(f"TC distribution across the validation set: {dict(sorted(Counter(ETC_TCs).items()))}")

#%%
# ── Forward-run the calibrated physics-based model with the ETC atmosphere ──
# Batched over samples (like the DataLoader batches) and over wavelength subsets (like
# NFMCalibratorTrainer.run_model/validate) to keep peak VRAM usage bounded and avoid OOM.
ETC_CALIB_PATH = _bundle_path # the bundle of the validated run
ETC_calibrator = load_NFM_calibrator(ETC_CALIB_PATH, device=default_device) # direct or factor bundle

# NFMDataset stores telemetry after the same scaling/imputation pipeline used by
# NFMCalibrator. Guard the feature ordering before passing those prepared vectors
# directly to the calibrator network.
if list(dataset.features) != list(ETC_calibrator.features):
    raise ValueError("ETC telemetry features do not match the calibrator bundle. Rebuild the dataset and calibrator with the same telemetry pipeline.")

N_val          = len(ETC_configs)
ETC_batch_size = cfg['batch_size']
PSFs_ETC_cube  = torch.zeros((N_val, N_wvl_total, dataset.H, dataset.W))

# Build a separate ETC telemetry set in physical units, then apply exactly the
# scaler and imputer used to train the calibrator. telemetry_vecs is ordered in
# the same validation-local order as validation_ids / ETC_TCs.
ETC_telemetry_features = build_ETC_telemetry(telemetry_vecs, ETC_TCs, ETC_calibrator)
ETC_telemetry_vecs = ETC_calibrator.prepare_features(ETC_telemetry_features)

for start in range(0, N_val, ETC_batch_size):
    end           = min(start + ETC_batch_size, N_val)
    chunk_configs = ETC_configs[start:end]
    chunk_config  = MultipleTargetsInDifferentObservations(chunk_configs, device=default_device)
    chunk_telemetry = ETC_telemetry_vecs[start:end]

    with torch.no_grad(): # the jitter / wing regime of the training (see build_PSF_model); calibrate sets the bundle's fitted shared values (e.g. the wing shape)
        ETC_PSF_model = build_PSF_model(chunk_config, cfg['model'], default_device)
        _ = ETC_calibrator.check_compatibility(ETC_PSF_model)
        ETC_calibrator.calibrate_prepared(chunk_telemetry, ETC_PSF_model)

        for lambda_ids in trainer.lambda_id_sets:
            wvl = trainer.lambda_full[lambda_ids].to(device=default_device)
            chunk_config['sources_science']['Wavelength'] = wvl.view(1, -1)
            ETC_PSF_model.model.config = chunk_config

            if len(trainer.lambda_id_sets) == 1:
                ETC_PSF_model.model.Update(grids=False, pupils=False, tomography=True)
            else:
                ETC_PSF_model.SetWavelengths(wvl)

            PSF_chunk = ETC_PSF_model().detach().cpu()
            for wi, li in enumerate(lambda_ids):
                PSFs_ETC_cube[start:end, li] = PSF_chunk[:, wi]

        ETC_PSF_model.cleanup()
        del chunk_telemetry
    release_gpu_memory()

print(f"Calibrated ETC ('white profile') PSF cube: {PSFs_ETC_cube.shape}")

#%%
# Radial profile validation of the calibrated ETC 'white profile' prediction against the true validation-set data,
# following the same approach as the profile validation performed above for the calibrated predictions.
wvl_select = [0, N_wvl_total // 2, -1]

fig, axes = plt.subplots(1, len(wvl_select), figsize=(15, 4.5))
p_errs_ETC = []

for i, lmbd in enumerate(wvl_select):
    err = plot_radial_PSF_profiles(
        PSFs_data_cube[:, lmbd, ...].cpu().numpy(),
        PSFs_ETC_cube [:, lmbd, ...].cpu().numpy(),
        'Data',
        'Calibrated ETC (white profile)',
        cutoff=40,
        y_min=3e-2,
        linthresh=1e-2,
        return_profiles=True,
        ax=axes[i],
    )[2].squeeze().max().item()

    axes[i].set_title(f"λ = {int((lambda_full[lmbd] * 1e9).round().item())} nm")
    p_errs_ETC.append(err)

fig.suptitle('Calibrated ETC white-profile validation', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
print(f"ΔSR per wavelength (calibrated ETC white profile): {np.array(p_errs_ETC)}")

# TODO: best_calibrator_bundle vs NFM_calibrator_bundle

#%%
# Spectrally-averaged radial profile for the calibrated ETC white profile, across the whole validation set
fig = plt.figure(figsize=(10, 6))
plot_radial_PSF_profiles(
    PSF_avg(PSFs_data_cube),
    PSF_avg(PSFs_ETC_cube),
    'Data',
    'Calibrated ETC (white profile)',
    title='Spectrally averaged PSF — calibrated ETC white profile validation',
    cutoff=40,
    ax=fig.add_subplot(111),
)
plt.tight_layout()
plt.show()


#%% ======================================= Features correlations =======================================
# Correlation of each feature with loss
input_features = dataset.features if 'dataset' in dir() else input_features
loss_vals = loss_SR.cpu().numpy()
tel_np    = telemetry_vecs.cpu().numpy()

pearson_r, spearman_r = [], []

for i in range(len(input_features)):
    pr, _ = pearsonr(tel_np[:, i],  loss_vals)
    sr, _ = spearmanr(tel_np[:, i], loss_vals)
    pearson_r.append(pr)
    spearman_r.append(sr)

pearson_r  = np.array(pearson_r)
spearman_r = np.array(spearman_r)

sort_idx = np.argsort(np.abs(spearman_r))[::-1]
x_corr   = np.arange(len(input_features))

fig, ax = plt.subplots(figsize=(max(12, len(input_features) * 0.9), 5))
ax.bar(x_corr - 0.2, pearson_r[sort_idx],  width=0.4, label='Pearson r',  alpha=0.8, color='tab:blue')
ax.bar(x_corr + 0.2, spearman_r[sort_idx], width=0.4, label='Spearman ρ', alpha=0.8, color='tab:orange')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x_corr)
ax.set_xticklabels([input_features[i] for i in sort_idx], rotation=45, ha='right')
ax.set_ylabel('Correlation with loss')
ax.set_title('Feature-loss correlation (sorted by |Spearman ρ|)')
ax.legend()
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# Print ranked table
print(f"{'Feature':<30} {'Pearson r':>10} {'Spearman ρ':>12}")
print("-" * 54)
for i in sort_idx:
    print(f"{input_features[i]:<30} {pearson_r[i]:>10.3f} {spearman_r[i]:>12.3f}")


#%%
tel_vecs_best  = telemetry_vecs[best_sample_ids]
tel_vecs_worst = telemetry_vecs[worst_sample_ids]
tel_vecs_sus   = telemetry_vecs[sus_ids]

# Single summary plot: median ± 1-sigma quantiles per feature for best / sus / worst samples
def feature_stats(vecs):
    arr = vecs.cpu().numpy()
    med = np.median(arr, axis=0)
    q16 = np.percentile(arr, 16, axis=0)
    q84 = np.percentile(arr, 84, axis=0)
    return med, q16, q84

med_best,  q16_best,  q84_best  = feature_stats(tel_vecs_best)
med_worst, q16_worst, q84_worst = feature_stats(tel_vecs_worst)
med_sus,   q16_sus,   q84_sus   = feature_stats(tel_vecs_sus)

x = np.arange(len(input_features))
width = 0.125

groups = [
    (med_best,  q16_best,  q84_best,  'Best (low loss)',   'tab:green'),
    (med_sus,   q16_sus,   q84_sus,   'Middle',            'tab:orange'),
    (med_worst, q16_worst, q84_worst, 'Worst (high loss)', 'tab:red'),
]

fig, ax = plt.subplots(figsize=(max(12, len(input_features) * 0.5), 5))
for offset, (med, q16, q84, label, color) in enumerate(groups):
    pos = x + (offset-1) * width
    ax.errorbar(pos, med, yerr=[med-q16, q84-med], fmt='o', capsize=4, markersize=5, color=color, label=label, linestyle='none')

ax.set_xticks(x)
ax.set_xticklabels(input_features, rotation=45, ha='right')
ax.set_ylabel('Feature value')
ax.set_title('Feature distributions: median ± 1σ quantile (best / middle / worst samples)')
ax.legend()
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

#%%
# Similar plot for telemetry OUTPUTS (calibrator predictions) based on category
# Extract output vectors for each category
out_vecs_best  = NN_params[best_sample_ids]
out_vecs_worst = NN_params[worst_sample_ids]
out_vecs_sus   = NN_params[sus_ids]

# Compute statistics for outputs
med_out_best,  q16_out_best,  q84_out_best  = feature_stats(out_vecs_best)
med_out_worst, q16_out_worst, q84_out_worst = feature_stats(out_vecs_worst)
med_out_sus,   q16_out_sus,   q84_out_sus   = feature_stats(out_vecs_sus)

x_out = np.arange(len(output_names))
width_out = 0.125

groups_out = [
    (med_out_best,  q16_out_best,  q84_out_best,  'Best (low loss)',   'tab:green'),
    (med_out_sus,   q16_out_sus,   q84_out_sus,   'Middle',            'tab:orange'),
    (med_out_worst, q16_out_worst, q84_out_worst, 'Worst (high loss)', 'tab:red'),
]

fig, ax = plt.subplots(figsize=(max(12, len(output_names) * 0.5), 5))
for offset, (med, q16, q84, label, color) in enumerate(groups_out):
    pos = x_out + (offset-1) * width_out
    ax.errorbar(pos, med, yerr=[med-q16, q84-med], fmt='o', capsize=4, markersize=5, color=color, label=label, linestyle='none')

ax.set_xticks(x_out)
ax.set_xticklabels(output_names, rotation=45, ha='right')
ax.set_ylabel('Normalized parameter')
ax.set_title('Predicted parameter distributions: median ± 1σ quantile (best / middle / worst samples)')
ax.legend()
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.ylim(-2.5, 2.5)
plt.show()

#%%
# Scatter plot: selected feature value vs loss
# feature_to_plot = 'Derot. angle'  # Change to any feature name in input_features
feature_to_plot = 'theta0'  # Change to any feature name in input_features

if feature_to_plot in input_features:
    feat_idx = input_features.index(feature_to_plot)
    feat_vals = telemetry_vecs[:, feat_idx].cpu().numpy()
    loss_vals = loss_SR.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(feat_vals, loss_vals, s=12, alpha=0.6, c=loss_vals, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Loss')
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel('Loss')
    ax.set_title(f"Loss vs '{feature_to_plot}'")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print(f"Feature '{feature_to_plot}' not found. Available features: {input_features}")



#%%
# Example diagnostic
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# X_tel: [N, D] reduced telemetry
# losses: [N] validation/per-sample loss from current model
reg = RandomForestRegressor(n_estimators=300, random_state=0)
scores = cross_val_score(reg, telemetry_vecs.cpu().numpy(), np.log10(loss_SR.cpu().numpy() + 1e-12), cv=5, scoring="r2")

print(f"CV R² predicting log-loss from telemetry: {scores.mean():.2f} ± {scores.std():.2f}")


#%%
# t-SNE diagnostic
from sklearn.manifold import TSNE

X_tsne = TSNE(n_components=2, metric="cosine", perplexity=50, init="pca", learning_rate="auto", random_state=0).fit_transform(telemetry_vecs)

sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=12, c=loss_SR.log().numpy(), cmap='viridis')
plt.colorbar(sc, label='log(Loss)')
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
plt.title('t-SNE of telemetry coloured by log-loss')
plt.grid(True, alpha=0.3); plt.show()

#%%
# =============================================================================
# SHAP Analysis
# -----------------------------------------------------------------------------
# Uses DeepExplainer (gradient-based, exact for torch nn.Module) to compute
# SHAP values: shap_values[i] is the contribution of each input feature to
# each output neuron i.  We then aggregate across outputs to answer:
#   - Which input features matter most globally?
#   - Which outputs are controlled by which inputs?
#   - Which features are essentially irrelevant for all outputs?
# =============================================================================
try:
    import shap
except ImportError as _e: # e.g. numba incompatible with the installed numpy
    raise ImportError(f"shap is not usable in this environment ({_e}); the SHAP cells below are skipped") from _e

# ── 1.  Prepare background and test sets ─────────────────────────────────────
# Rebuild the telemetry matrix directly from the dataset so that `tel_np` and
# `input_features` are always consistent, regardless of what is stored in the
# outer-scope `telemetry_vecs` variable (which may be stale from a previous
# interactive run or from a different analysis in the same session).
shap_input_features = dataset.features         # D feature names — SHAP-local copy
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)  # [N_dataset, D]

N_background = min(100, len(shap_tel_np))
rng          = np.random.default_rng(42)
bg_idx       = rng.choice(len(shap_tel_np), size=N_background, replace=False)
background   = torch.tensor(shap_tel_np[bg_idx], dtype=torch.float32, device=default_device)
X_explain    = torch.tensor(shap_tel_np,         dtype=torch.float32, device=default_device)

#%%
# ── 2.  Build explainer and compute SHAP values ───────────────────────────────
# DeepExplainer does not support LayerNorm (used in SmallCalibratorNet), so we
# use GradientExplainer instead — it computes integrated-gradient-based SHAP
# values and supports any differentiable PyTorch model.
# shap_values shape: [N_explain, N_features, N_outputs]
calibrator.eval()
explainer = shap.GradientExplainer(calibrator, background)
sv_list   = explainer.shap_values(X_explain)    # list of N_outputs arrays [N, N_feat]
shap_values = np.stack(sv_list)        # [N, N_feat, N_out]
print(f"SHAP values computed: {shap_values.shape}  (samples x features x outputs)")
# NOTE: for the factor calibrator the outputs are corrections to the telemetry-derived priors, not the parameters themselves

# Mean |SHAP| across samples: [N_feat, N_out]
mean_abs_shap = np.abs(shap_values).mean(axis=0)

#%%
# ── 3.  Build readable output names from the transformer ──────────────────────
# Each output neuron corresponds to one element of the stacked parameter vector (direct calibrator) or of the predicted keys only
# (factor calibrator, its constants are not network outputs): net_slices gives the layout of the network output.
# We expand the parameter names to match individual neurons for clear labelling.
net_slices = trainer.net_slices()
output_neuron_names = []
for param_name, slc in net_slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_neuron_names.append(param_name)
    else:
        output_neuron_names.extend([f"{param_name}[{i}]" for i in range(width)])

assert len(output_neuron_names) == shap_values.shape[-1], f"Name count mismatch: {len(output_neuron_names)} vs {shap_values.shape[-1]}"

#%%
# ── 4.  Global feature importance  (mean |SHAP| summed over all outputs) ─────
# Re-derive shap_input_features from mean_abs_shap's leading dimension so this
# cell is self-consistent even when run independently in an interactive session.
shap_input_features = dataset.features[:mean_abs_shap.shape[0]]
# Replace certain names in the features list:
# shap_input_features[shap_input_features.index('gain')] = 'LO WFS gain'
# shap_input_features[shap_input_features.index('window')] = 'LO WFS window'
shap_input_features[shap_input_features.index('theta0')] = 'Θ₀'
# shap_input_features[shap_input_features.index('frequency')] = 'LO WFS frequency'
shap_input_features[shap_input_features.index('NGS mag (from ph.)')] = 'NGS mag'
shap_input_features[shap_input_features.index('LGS photons, [photons/m^2/s]')] = 'LGS flux, [ph./m²/s]'

shap_input_features = [feat.replace('_binned_', '_') for feat in shap_input_features]
shap_input_features = [feat.replace('Tau0', 'τ₀') for feat in shap_input_features]
shap_input_features = [feat.replace('Cn2', 'Cₙ²') for feat in shap_input_features]

global_importance   = mean_abs_shap.sum(axis=1)    # [N_feat]
feat_order          = np.argsort(global_importance)[::-1]

fig, ax = plt.subplots(figsize=(max(10, len(shap_input_features) * 0.55), 5))
bars = ax.bar(range(len(shap_input_features)), global_importance[feat_order], color=plt.cm.RdYlGn(global_importance[feat_order] / global_importance.max()))

ax.set_xticks(range(len(shap_input_features)))
ax.set_xticklabels([shap_input_features[i] for i in feat_order], rotation=45, ha='right')
ax.set_ylabel('Mean |SHAP| summed over all outputs')
ax.set_title('Global feature importance across all calibrator outputs')
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout(); plt.show()

# Print ranked table
print(f"\n{'Feature':<35} {'Global |SHAP|':>14}")
print("-" * 51)

for i in feat_order:
    print(f"{shap_input_features[i]:<35} {global_importance[i]:>14.4f}")

#%%
# ── 5.  Identify useless features ─────────────────────────────────────────────
# A feature is "useless" if its global importance is below a relative threshold
# (here: less than 1 % of the most important feature).
useless_threshold  = 0.01 * global_importance.max()
useless_feat_mask  = global_importance < useless_threshold
useless_features   = [shap_input_features[i] for i in range(len(shap_input_features)) if useless_feat_mask[i]]
useful_features    = [shap_input_features[i] for i in range(len(shap_input_features)) if not useless_feat_mask[i]]

print(f"\nUseless features (< 1% of max global importance): {useless_features}")
print(f"Useful features ({len(useful_features)}): {useful_features}")

# ── 6.  Heatmap: input features vs. output parameters ─────────────────────────
#%%
# Aggregate per-parameter-group (not per-neuron) for readability:
# collapse the neuron dimension by averaging within each parameter group.
param_names  = list(net_slices.keys())
mean_abs_per_param = np.zeros((len(shap_input_features), len(param_names)))

for j, (pname, slc) in enumerate(net_slices.items()):
    mean_abs_per_param[:, j] = mean_abs_shap[:, slc].mean(axis=1)

import seaborn as sns

# Normalise each output column to [0,1] so that small-magnitude outputs are still visible
norm_heatmap = mean_abs_per_param / (mean_abs_per_param.max(axis=0, keepdims=True) + 1e-12)

# Order features by global importance (most important at top)
feat_order_hm = np.argsort(global_importance)[::-1]
fig, ax = plt.subplots(figsize=(max(10, len(param_names) * 0.65), max(6,  len(shap_input_features) * 0.35)))

sns.heatmap(
    norm_heatmap[feat_order_hm, :],
    xticklabels=param_names,
    yticklabels=[shap_input_features[i] for i in feat_order_hm],
    cmap='magma',
    linewidths=0.3,
    linecolor='#333',
    vmin=0, vmax=1,
    ax=ax,
    cbar_kws={'label': 'Normalised mean |SHAP| (per output column)'},
)
ax.set_title('Input feature → output parameter importance (column-normalised)')
ax.set_xlabel('Calibrator output (predicted parameter)')
ax.set_ylabel('Input feature (telemetry, sorted by global importance)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.show()

# ── 7.  SHAP beeswarm summary plot (all outputs combined) ────────────────────
#%%
# Flatten shap_values to [N, N_feat] by summing |SHAP| over outputs per sample,
# then feed to SHAP's summary_plot in "violin" style for a cleaner view.
shap_summary = shap_values.sum(axis=-1)    # [N, N_feat]  — signed sum over outputs

# Re-derive locally so this cell works even when run independently.
_n_feat = shap_summary.shape[1]
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)
assert shap_tel_np.shape[1] == _n_feat, (
    f"Dataset has {shap_tel_np.shape[1]} features but shap_values has {_n_feat}. "
    "Re-run Steps 1-2 to recompute shap_values with the current dataset."
)
shap_input_features = np.array(dataset.features[:_n_feat])

shap.summary_plot(
    shap_summary,
    features=shap_tel_np,
    feature_names=shap_input_features,
    plot_type='violin',
    max_display=len(shap_input_features),
    show=True,
)

# ── 8.  Per-output SHAP bar charts for the most important parameters ──────────
#%%
# Show the top-k most important features for each predicted parameter individually.
TOP_K_FEATURES = 10   # how many features to display per parameter

n_params  = len(param_names)
ncols     = min(4, n_params)
nrows     = int(np.ceil(n_params / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), constrained_layout=True)
axes_flat = np.array(axes).ravel()

for j, pname in enumerate(param_names):
    ax   = axes_flat[j]
    slc  = net_slices[pname]
    imp  = mean_abs_shap[:, slc].mean(axis=1)          # mean over neurons in this param
    top  = np.argsort(imp)[::-1][:TOP_K_FEATURES]
    ax.barh([shap_input_features[i] for i in top[::-1]], imp[top[::-1]],
            color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_title(pname, fontsize=9)
    ax.set_xlabel('Mean |SHAP|', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='x', alpha=0.3)

# Hide any unused axes
for ax in axes_flat[n_params:]:
    ax.set_visible(False)

fig.suptitle(f'Top-{TOP_K_FEATURES} input features per predicted parameter', fontsize=11)
plt.show()

# ── 9.  SHAP dependence plot for the single most predictive feature per output ─
#%%
print("\nMost predictive input feature per predicted parameter:")
print(f"{'Parameter':<30} {'Feature':<35} {'Mean |SHAP|':>12}")
print("-" * 80)
for j, pname in enumerate(param_names):
    slc  = net_slices[pname]
    imp  = mean_abs_shap[:, slc].mean(axis=1)
    best = int(np.argmax(imp))
    print(f"{pname:<30} {shap_input_features[best]:<35} {imp[best]:>12.4f}")

#%%
# Dependence plots: for each parameter group show the dominant feature's
# SHAP value vs. its raw value, coloured by the second-strongest feature.
# Limit to the first 6 parameter groups for visual clarity.
# Re-derive shap_tel_np locally for the same session-isolation reason as Step 7.
_n_feat = shap_values.shape[1]
shap_tel_np = np.array([list(dataset.telemetry[i].values()) for i in range(len(dataset))], dtype=np.float32)

assert shap_tel_np.shape[1] == _n_feat, (
    f"Dataset has {shap_tel_np.shape[1]} features but shap_values has {_n_feat}. "
    "Re-run Steps 1-2 to recompute shap_values with the current dataset."
)
shap_input_features = np.array(dataset.features[:_n_feat])

shap_per_param = {
    pname: shap_values[:, :, slc].mean(axis=-1)   # [N, N_feat]  mean SHAP over neurons
    for pname, slc in net_slices.items()
}

N_PARAM_PLOTS = min(6, n_params)
fig, axes = plt.subplots(2, (N_PARAM_PLOTS + 1) // 2, figsize=((N_PARAM_PLOTS + 1) // 2 * 5, 8), constrained_layout=True)
axes_flat = np.array(axes).ravel()

for j, pname in enumerate(list(shap_per_param.keys())[:N_PARAM_PLOTS]):
    sv  = shap_per_param[pname]                     # [N, N_feat]
    imp = np.abs(sv).mean(axis=0)
    top2 = np.argsort(imp)[::-1][:2]
    feat_x   = shap_input_features[top2[0]]
    feat_col = shap_input_features[top2[1]] if len(top2) > 1 else feat_x

    ax    = axes_flat[j]
    x_val = shap_tel_np[:, top2[0]]
    y_val = sv[:, top2[0]]
    c_val = shap_tel_np[:, top2[1]]

    sc = ax.scatter(x_val, y_val, c=c_val, cmap='coolwarm', s=10, alpha=0.7)
    ax.axhline(0, color='black', lw=0.7, linestyle='--')
    ax.set_xlabel(feat_x, fontsize=8)
    ax.set_ylabel(f'SHAP for {pname}', fontsize=8)
    ax.set_title(pname, fontsize=9)
    fig.colorbar(sc, ax=ax, label=feat_col, fraction=0.03, pad=0.02)
    ax.grid(True, alpha=0.3)

for ax in axes_flat[N_PARAM_PLOTS:]:
    ax.set_visible(False)

fig.suptitle('SHAP dependence: dominant feature per parameter\n(colour = second-strongest feature)', fontsize=11)
plt.show()

# %%
