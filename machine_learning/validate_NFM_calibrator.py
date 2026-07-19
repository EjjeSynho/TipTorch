
#%%
# ── Imports ───────────────────────────────────────────────────────────────────
import json
import logging
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from tiptorch._config import *
from tiptorch.tools.utils import r0
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from calibrators.NFM_calibrator import (
    SmallCalibratorNet,
    NFMCalibratorTrainer,
    NFMDataset,
    release_gpu_memory
)
from tools.plotting import plot_radial_PSF_profiles, draw_PSF_stack

logger = logging.getLogger(__name__)

STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE = STD_FOLDER / 'dataset_cache'

#%%
# ── Load training config ──────────────────────────────────────────────────────
_vparser = argparse.ArgumentParser(description="Validate NFM Calibrator")
_vparser.add_argument('--config', type=str, default=None,
                      help='Path to the JSON config used for training')
try:
    _vargs       = _vparser.parse_args()
    _config_path = Path(_vargs.config) if _vargs.config else None
except SystemExit:
    _config_path = None

if _config_path and _config_path.exists():
    with open(_config_path) as _f:
        cfg = json.load(_f)
elif 'cfg' not in dir():
    # Fall back to NFM_calibrator_config.json in the same directory as this script.
    # Works for both CLI (uses __file__) and plain IPython (uses cwd).
    _script_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
    _default_cfg_path = _script_dir / 'NFM_calibrator_config.json'
    if _default_cfg_path.exists():
        with open(_default_cfg_path) as _f:
            cfg = json.load(_f)
        logger.info(f"No --config given; loaded default config from {_default_cfg_path}")
    else:
        raise RuntimeError(
            "cfg is not defined. Either pass --config path/to/config.json, "
            "run this file after train_NFM_calibrator.py in the same IPython session, "
            f"or place NFM_calibrator_config.json at {_default_cfg_path}."
        )

#%%
# ── Load post-training metadata ───────────────────────────────────────────────
_meta_path = WEIGHTS_FOLDER / cfg["weights_subdir"] / f'{cfg["name"]}_meta.json'

if not _meta_path.exists():
    raise FileNotFoundError(f"Training metadata not found: {_meta_path}. Run train_NFM_calibrator.py first.")

with open(_meta_path) as _f:
    train_meta = json.load(_f)

BEST_CALIB_PATH = Path(train_meta['checkpoint_path'])
output_names    = train_meta['output_names']
N_wvl_total     = train_meta['N_wvl_total']
N_outputs       = train_meta['N_outputs']
input_features  = train_meta['input_feature_names']

logger.info(f"Metadata loaded: name={train_meta['name']} | best_val={train_meta['best_val_loss']:.6f} | completed={train_meta['training_completed_at']}")

#%%
# ── Ensure consistent dataset split ──────────────────────────────────────────
_split_path = Path(train_meta['split_path'])

#%%
# ── Initialize trainer (when running standalone, outside a training session) ──
if 'trainer' not in dir():
    dataset = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
    logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W})")

    _tmp_batch  = tuple([dataset[i] for i in np.random.randint(0, len(dataset), size=cfg['batch_size'])])
    _, _, _, _tmp_config, _ = NFMDataset.collate_batch(_tmp_batch, default_device)

    with torch.no_grad():
        PSF_model = PSFModelNFM(
            _tmp_config,
            multiple_obs   = True,
            LO_NCPAs       = cfg['model']['LO_NCPAs'],
            chrom_defocus  = cfg['model']['chrom_defocus'],
            use_Moffat     = cfg['model']['use_Moffat'],
            retain_PSDs    = cfg['model']['retain_PSDs'],
            N_spline_nodes = cfg['model']['N_spline_nodes'],
            Z_mode_max     = cfg['model']['Z_mode_max'],
            device         = default_device,
        )
    del _tmp_batch, _tmp_config

    for _key in cfg['fixed_params']:
        try:
            PSF_model.inputs_manager.delete(_key)
        except Exception:
            pass
    if PSF_model.use_Moffat:
        PSF_model.inputs_manager.delete('theta')
        PSF_model.inputs_manager.delete('ratio')

    PSF_model.inputs_manager.set_optimizable(['LO_coefs'],          cfg['trainer']['predict_LO_NCPAs'])
    PSF_model.inputs_manager.set_optimizable(['Cn2_weights'],       cfg['trainer']['predict_Cn2_profile'])
    PSF_model.inputs_manager.set_optimizable(['wind_speed_single'], cfg['trainer']['predict_wind_speed'])

    outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
    _buf = outputs_transformer.unstack(PSF_model.inputs_manager.stack())
    if 'LO_coefs' in _buf:
        _buf['LO_coefs'] = _buf['LO_coefs'][:, 1:]
    _ = outputs_transformer.stack(_buf)
    del _buf

    calibrator = SmallCalibratorNet(
        n_features   = train_meta['N_features'],
        n_outputs    = train_meta['N_outputs'],
        hidden_dim   = cfg['calibrator']['hidden_dim'],
        dropout_rate = cfg['calibrator']['dropout_rate'],
    ).to(default_device)

    trainer = NFMCalibratorTrainer(
        PSF_model           = PSF_model,
        calibrator          = calibrator,
        dataset             = dataset,
        outputs_transformer = outputs_transformer,
        device              = default_device,
        batch_size          = cfg['batch_size'],
        lr                  = cfg['trainer']['lr'],
        lambda_step         = cfg['trainer']['lambda_step'],
        predict_LO_NCPAs    = cfg['trainer']['predict_LO_NCPAs'],
        predict_Cn2_profile = cfg['trainer']['predict_Cn2_profile'],
        random_state        = cfg['trainer']['random_state'],
        pre_init_astrometry = cfg['trainer']['pre_init_astrometry'],
        optimize_astrometry = cfg['trainer']['optimize_astrometry'],
        collate_fn          = NFMDataset.collate_batch,
    )
    release_gpu_memory()
    logger.info("Trainer initialized from scratch (standalone mode).")

if _split_path.exists():
    _split = np.load(_split_path)
    if not (np.array_equal(_split['train_idx'], trainer.train_idx) and
            np.array_equal(_split['val_idx'],   trainer.val_idx)):
        logger.warning("Trainer split does not match saved split — restoring.")
        trainer.train_idx    = _split['train_idx']
        trainer.val_idx      = _split['val_idx']
        trainer.train_loader = trainer._make_loader(trainer.train_idx, shuffle=True)
        trainer.val_loader   = trainer._make_loader(trainer.val_idx,   shuffle=False)
        logger.info(f"Restored split: {len(trainer.train_idx)} train / {len(trainer.val_idx)} val")
else:
    logger.warning(f"Split file not found at {_split_path}; using current trainer split.")


#%% ========================== Evaluation & diagnostics ===============================================
trainer.load_checkpoint(BEST_CALIB_PATH)
calibrator.eval()

PSFs_pred_cube, PSFs_data_cube, validation_ids, NN_predictions, telemetry_vecs, final_val_loss = trainer.validate(return_cubes=True)

PSFs_pred_cube = PSFs_pred_cube.cpu()
PSFs_data_cube = PSFs_data_cube.cpu()
validation_ids = validation_ids.cpu()

release_gpu_memory()
print(f"Validation loss: {final_val_loss:.6f} | PSF cubes {PSFs_pred_cube.shape}")


#%%
stds  = NN_predictions.std(dim=0)
means = NN_predictions.mean(dim=0)

plt.bar(np.arange(len(means)), means.cpu().numpy(), yerr=stds.cpu().numpy(), alpha=0.7, capsize=3)
plt.xticks(ticks=np.arange(len(means)), labels=output_names, rotation=45, ha='right')

# Select features with low varibility relative to their mean (e.g., < 5% std/mean)
# low_var_features = (stds / means) < 0.05
# print(f"Low variability features: {low_var_features.sum().item()} / {len(low_var_features)}:")
# for i, is_low_var in enumerate(low_var_features):
#     if is_low_var:
#         print(f"  {output_names[i]}: mean={means[i].item():.4f}, std={stds[i].item():.4f}, std/mean={stds[i].item()/means[i].item():.4f}")

selected_mean_features = []
feature_keywords = {'J_ctrl', 'F_ctrl', 'LO_coefs', 'wind_speed_single', 'L0'}

for i, name in enumerate(output_names):
    if any(keyword in name for keyword in feature_keywords):
        selected_mean_features.append((name, means[i].item(), stds[i].item()))


#%%
from tiptorch.tools.utils import r0

# TODO: something is very wrong with the binned Cn2_weights

with open(TELEMETRY_CACHE / 'MUSE/muse_telemetry_scaler.pickle', 'rb') as handle:
    telemetry_scaler = pickle.load(handle)
    
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as fh:
    muse_df = pickle.load(fh)

x_vec = telemetry_scaler.inverse_transform(telemetry_vecs.cpu().numpy())

X = {
    'r0': r0(muse_df['Seeing (header)'].loc[validation_ids].to_numpy(), 500e-9),
    'L0': muse_df['L0Tot'].loc[validation_ids].to_numpy(),
    'wind_speed_single': muse_df['Wind speed (header)'].loc[validation_ids].to_numpy(),
    'Cn2_weights_2': x_vec[input_features.index('Cn2_frac_binned_2')],
    'Cn2_weights_3': x_vec[input_features.index('Cn2_frac_binned_3')]
}
X['Cn2_weights_1'] = 1.0 - X['Cn2_weights_2'] - X['Cn2_weights_3']

features_of_interest = list(X.keys()) + ['Cn2_weights']

Y = {k: v.cpu().numpy() for k, v in trainer.outputs_transformer.unstack(NN_predictions).items() if k in features_of_interest}
Y['Cn2_weights_1'] = Y['Cn2_weights'][:, 0]
Y['Cn2_weights_2'] = Y['Cn2_weights'][:, 1]
Y['Cn2_weights_3'] = Y['Cn2_weights'][:, 2]

Y['wind_speed_single'] = np.abs(Y['wind_speed_single'])

_ = Y.pop('Cn2_weights', None)

#%%
common_keys = [k for k in X if k in Y]
n_keys = len(common_keys)
ncols = min(3, n_keys)
nrows = int(np.ceil(n_keys / ncols))

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

#%%
trainer.save_calibrator(Path(train_meta['bundle_path']))

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
x_dict_pred = outputs_transformer.unstack(NN_predictions)
loss_SR = trainer._loss_per_sample(PSFs_data_cube, PSFs_pred_cube, x_dict_pred).cpu()

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
    errs = profile_errs[wvl]/100
    
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
pearson_all,  pearson_p_all  = pearsonr(all_loss, all_errs)
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


#%%
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
out_vecs_best  = NN_predictions[best_sample_ids]
out_vecs_worst = NN_predictions[worst_sample_ids]
out_vecs_sus   = NN_predictions[sus_ids]

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
ax.set_ylabel('Output value')
ax.set_title('Calibrator output distributions: median ± 1σ quantile (best / middle / worst samples)')
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
import shap

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

# Mean |SHAP| across samples: [N_feat, N_out]
mean_abs_shap = np.abs(shap_values).mean(axis=0)

#%%
# ── 3.  Build readable output names from the transformer ──────────────────────
# Each output neuron corresponds to one element of the stacked parameter vector.
# We expand the parameter names to match individual neurons for clear labelling.
output_neuron_names = []
for param_name, slc in outputs_transformer.slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_neuron_names.append(param_name)
    else:
        output_neuron_names.extend([f"{param_name}[{i}]" for i in range(width)])

assert len(output_neuron_names) == N_outputs, f"Name count mismatch: {len(output_neuron_names)} vs {N_outputs}"

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
param_names  = list(outputs_transformer.slices.keys())
mean_abs_per_param = np.zeros((len(shap_input_features), len(param_names)))
for j, (pname, slc) in enumerate(outputs_transformer.slices.items()):
    mean_abs_per_param[:, j] = mean_abs_shap[:, slc].mean(axis=1)

import seaborn as sns

# Normalise each output column to [0,1] so that small-magnitude outputs are still visible
norm_heatmap = mean_abs_per_param / (mean_abs_per_param.max(axis=0, keepdims=True) + 1e-12)

# Order features by global importance (most important at top)
feat_order_hm = np.argsort(global_importance)[::-1]

fig, ax = plt.subplots(figsize=(max(10, len(param_names) * 0.65),
                                max(6,  len(shap_input_features) * 0.35)))
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

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 4.5, nrows * 3.5),
                         constrained_layout=True)
axes_flat = np.array(axes).ravel()

for j, pname in enumerate(param_names):
    ax   = axes_flat[j]
    slc  = outputs_transformer.slices[pname]
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
    slc  = outputs_transformer.slices[pname]
    imp  = mean_abs_shap[:, slc].mean(axis=1)
    best = int(np.argmax(imp))
    print(f"{pname:<30} {shap_input_features[best]:<35} {imp[best]:>12.4f}")

#%%
# Dependence plots: for each parameter group show the dominant feature's
# SHAP value vs. its raw value, coloured by the second-strongest feature.
# Limit to the first 6 parameter groups for visual clarity.
# Re-derive shap_tel_np locally for the same session-isolation reason as Step 7.
_n_feat = shap_values.shape[1]
shap_tel_np = np.array(
    [list(dataset.telemetry[i].values()) for i in range(len(dataset))],
    dtype=np.float32,
)
assert shap_tel_np.shape[1] == _n_feat, (
    f"Dataset has {shap_tel_np.shape[1]} features but shap_values has {_n_feat}. "
    "Re-run Steps 1-2 to recompute shap_values with the current dataset."
)
shap_input_features = np.array(dataset.features[:_n_feat])

shap_per_param = {
    pname: shap_values[:, :, slc].mean(axis=-1)   # [N, N_feat]  mean SHAP over neurons
    for pname, slc in outputs_transformer.slices.items()
}

N_PARAM_PLOTS = min(6, n_params)
fig, axes = plt.subplots(2, (N_PARAM_PLOTS + 1) // 2,
                         figsize=((N_PARAM_PLOTS + 1) // 2 * 5, 8),
                         constrained_layout=True)
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
