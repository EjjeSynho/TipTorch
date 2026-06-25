#%%
try:
    ipy = get_ipython()
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import os
import sys
import logging
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from datetime import datetime

from torch.utils.data import Dataset

from tiptorch._config import *
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from calibrators.NFM_calibrator import SmallCalibratorNet, NFMCalibratorTrainer, release_gpu_memory

# ── Paths ─────────────────────────────────────────────────────────────────────
MUSE_DATA_FOLDER = Path(project_settings["MUSE_data_folder"])
STD_FOLDER       = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE    = STD_FOLDER / 'dataset_cache'
BEST_CALIB_PATH  = WEIGHTS_FOLDER / 'NFM_calibrator/best_calibrator_checkpoint.pth'

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train NFM Calibrator")
parser.add_argument('--continue-training',    action='store_true',  help='Resume training from last checkpoint')
parser.add_argument('--run-kfold-difficulty', action='store_true',  help='Run K-fold difficulty weighting before final training')
parser.add_argument('--kfolds',               type=int,   default=5)
parser.add_argument('--kfold-epochs',         type=int,   default=120)
parser.add_argument('--kfold-patience',       type=int,   default=12)
parser.add_argument('--difficulty-alpha',     type=float, default=0.30)
parser.add_argument('--difficulty-clip-min',  type=float, default=0.50)
parser.add_argument('--difficulty-clip-max',  type=float, default=2.00)
parser.add_argument('--debug',                action='store_true')
parser.add_argument('--batch-size',           type=int,   default=16)

try:
    args = parser.parse_args()
except SystemExit:
    args = argparse.Namespace(
        continue_training=True,
        run_kfold_difficulty=True,
        kfolds=5,
        kfold_epochs=120,
        kfold_patience=12,
        difficulty_alpha=0.30,
        difficulty_clip_min=0.50,
        difficulty_clip_max=2.00,
        debug=True,
        batch_size=16,
    )

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")


# Dataset
class NFMDataset(Dataset):
    """
    Dataset of MUSE NFM PSF cubes with paired telemetry and fitted parameter values.
    Each sample: (PSF_cube [C,H,W], telemetry [D], fitted_params dict, config, dataset_idx).
    """
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.PSF_cubes   = data['PSF_cubes']
        self.telemetry   = data['telemetry']
        self.sample_ids  = data['sample_ids']
        self.configs     = data['model_configs']
        self.fitted_vals = data['fitted_param_values']
        self.features    = list(data['telemetry'][0].keys())

        _rename = {p: p + '_ctrl' for p in ['J', 'F', 'bg', 'dx', 'dy']}
        for i in range(len(self.sample_ids)):
            if isinstance(self.configs[i], list):
                self.configs[i] = self.configs[i][0]
                
            for old_k, new_k in _rename.items():
                if old_k in self.fitted_vals[i] and new_k not in self.fitted_vals[i]:
                    self.fitted_vals[i][new_k] = self.fitted_vals[i].pop(old_k)

        self.N, self.H, self.W, self.C = self.PSF_cubes.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        PSFs = torch.from_numpy(self.PSF_cubes[idx].astype(np.float32)).permute(2, 0, 1)
        tel  = torch.from_numpy(np.array(list(self.telemetry[idx].values()), dtype=np.float32))
        fit  = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.fitted_vals[idx].items()}
        return PSFs, tel, fit, self.configs[idx], idx

    @staticmethod
    def collate_batch(batch, device):
        PSF_cubes, telemetry_vecs, fitted_vals, configs, idxs = zip(*batch)
        PSF_cubes      = torch.stack(PSF_cubes,      0).to(device=device, non_blocking=True)
        telemetry_vecs = torch.stack(telemetry_vecs, 0).to(device=device, non_blocking=True)
        idxs           = torch.tensor(idxs, dtype=torch.long, device=device)
        fitted_vals    = {k: torch.stack([fv[k] for fv in fitted_vals], 0).to(device=device, non_blocking=True) for k in fitted_vals[0]}
        batch_config = MultipleTargetsInDifferentObservations(configs, device=device)
        batch_config['PathPupil'] = str(DATA_FOLDER / 'calibrations/VLT_CALIBRATION/VLT_PUPIL/ut4pupil320.fits')
        batch_config['telescope']['PupilAngle'] = 0.0
        batch_config['DM']['NumberReconstructedLayers'] = torch.tensor(batch_config['atmosphere']['Cn2Heights'].shape[-1], device=device)
        return PSF_cubes, telemetry_vecs, fitted_vals, batch_config, idxs


#%%
dataset     = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_wvl_total = dataset.C
logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W})")

#%%
# Initialize the PSF model with a representative batch config
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

_tmp_batch  = tuple([dataset[i] for i in np.random.randint(0, len(dataset), size=args.batch_size)])
_, _, _, _tmp_config, _ = dataset.collate_batch(_tmp_batch, device=default_device)


if 'PSF_model' in locals():
    del PSF_model

with torch.no_grad():
    PSF_model = PSFModelNFM(
        _tmp_config,
        multiple_obs   = True,
        LO_NCPAs       = True,
        chrom_defocus  = False,
        use_Moffat     = False,
        retain_PSDs    = False,
        N_spline_nodes = 5,
        Z_mode_max     = 9,
        device         = default_device,
    )

del _tmp_batch, _tmp_config
release_gpu_memory(sync=True)

# Remove parameters that are fixed externally and do not need to be predicted
for _key in ['Jxy', 'bg_ctrl', 'dx_ctrl', 'dy_ctrl', 'L0',
             'F_norm', 'F_norm_lambda_ctrl', 'src_dirs_x', 'src_dirs_y',
             'wind_speed_single', 'wind_dir_single', 'F_norm_λ_ctrl']:
    try:
        PSF_model.inputs_manager.delete(_key)

    except Exception:
        pass   # key may have an alternate name; delete what is present

if PSF_model.use_Moffat:
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

predict_LO_NCPAs    = True
predict_Cn2_profile = True
PSF_model.inputs_manager.set_optimizable(['LO_coefs'],    predict_LO_NCPAs)
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'], predict_Cn2_profile)
print(PSF_model.inputs_manager)

#%%
# Build the outputs transformer: maps calibrator predictions <-> PSF model inputs
outputs_transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
_buf = outputs_transformer.unstack(PSF_model.inputs_manager.stack())
if 'LO_coefs' in _buf:
    _buf['LO_coefs'] = _buf['LO_coefs'][:, 1:] # strip phase-bump column; not predicted
_ = outputs_transformer.stack(_buf)
del _buf

N_outputs  = outputs_transformer.get_stacked_size()
N_features = len(dataset.features)
release_gpu_memory()
print(f"Calibrator: {N_features} inputs -> {N_outputs} outputs")

#%%
# Calibrator network  (can be swapped for any nn.Module with the same I/O contract)
calibrator = SmallCalibratorNet(
    n_features   = N_features,
    n_outputs    = N_outputs,
    hidden_dim   = 48,
    dropout_rate = 0.2,
).to(default_device)


# Instantiate the trainer
trainer = NFMCalibratorTrainer(
    PSF_model           = PSF_model,
    calibrator          = calibrator,
    dataset             = dataset,
    outputs_transformer = outputs_transformer,
    device              = default_device,
    batch_size          = args.batch_size,
    lr                  = 1e-2,
    lambda_step         = 1,
    predict_LO_NCPAs    = predict_LO_NCPAs,
    predict_Cn2_profile = predict_Cn2_profile,
    random_state        = 43,
    pre_init_astrometry = True,
    optimize_astrometry = False,
    collate_fn          = NFMDataset.collate_batch,
)

#%% ===========================  Pretraining ==================================================
pretrain_path = trainer.weights_dir / 'pretrain_weights.pth'
if (not args.continue_training) or not pretrain_path.exists():
    trainer.pretrain(num_epochs=100, patience=15, save_path=pretrain_path)
else:
    calibrator.load_state_dict(torch.load(pretrain_path, map_location=default_device))
    logger.info(f"Loaded pretrain weights from {pretrain_path}")

release_gpu_memory(sync=True)

val_loss_pretrain = trainer.validate()
logger.info(f"Post-pretrain validation loss: {val_loss_pretrain:.6f}")

#%% ======================== Main training ========================================
if args.continue_training and BEST_CALIB_PATH.exists():
    trainer.load_checkpoint(BEST_CALIB_PATH)

BASE_STATE = deepcopy(calibrator.state_dict())
sample_weights = None

if args.run_kfold_difficulty:
    sample_weights = trainer.run_kfold_difficulty_weighting(
        base_indices       = trainer.train_idx,
        initial_state_dict = BASE_STATE,
        n_splits           = args.kfolds,
        fold_epochs        = args.kfold_epochs,
        fold_patience      = args.kfold_patience,
        alpha              = args.difficulty_alpha,
        clip               = (args.difficulty_clip_min, args.difficulty_clip_max),
    )

#%% ======================== Main training ========================================
# Back up previous best checkpoint before overwriting
if BEST_CALIB_PATH.exists():
    os.rename(BEST_CALIB_PATH, BEST_CALIB_PATH.with_name(BEST_CALIB_PATH.stem + '_backup.pth'))

train_losses, val_losses = trainer.train(
    num_epochs         = 500,
    patience           = 20,
    nan_recovery       = True,
    max_nan_recoveries = 10,
    sample_weights     = sample_weights,
    checkpoint_path    = BEST_CALIB_PATH,
)

release_gpu_memory()
logger.info(f"Training done. Final train={train_losses[-1]:.6f} val={val_losses[-1]:.6f} best={min(val_losses):.6f}")

# Quit here if not run with IPython (e.g. from command line)
if not (hasattr(sys, 'ps1') or sys.flags.interactive):
    logger.info("Non-interactive environment detected. Exiting after training.")
    sys.exit(0)
    
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label='Train')
ax.plot(val_losses,   label='Val')
ax.set(xlabel='Epoch', ylabel='Loss', title='Training Losses')
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.show()

#%%
# Optional: fine-tune selected parameters after main training
trainer.tune_calibrator(tuned_params=['F_ctrl', 'dn'], num_epochs=20, lr=1e-3)

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
trainer.save_calibrator(trainer.weights_dir / 'NFM_calibrator_bundle.pth')

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

true_best_ids = validation_ids[best_sample_ids].cpu().numpy()
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
from scipy.stats import pearsonr, spearmanr

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
from scipy.stats import pearsonr, spearmanr

input_features = dataset.features
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
    (med_sus,   q16_sus,   q84_sus,   'Middle',        'tab:orange'),
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
# Build readable output names from the transformer
output_names = []
for param_name, slc in outputs_transformer.slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_names.append(param_name)
    else:
        output_names.extend([f"{param_name}[{i}]" for i in range(width)])

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
plt.ylim(-1, 1.5)
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
X_explain    = torch.tensor(shap_tel_np,          dtype=torch.float32, device=default_device)

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
shap_input_features[shap_input_features.index('gain')] = 'LO WFS gain'
shap_input_features[shap_input_features.index('window')] = 'LO WFS window'
shap_input_features[shap_input_features.index('theta0')] = 'Θ₀'
shap_input_features[shap_input_features.index('frequency')] = 'LO WFS frequency'
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

