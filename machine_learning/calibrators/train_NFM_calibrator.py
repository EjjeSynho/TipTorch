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
import json
import random
import logging
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from calibrators.NFM_calibrator import NFMDataset, build_NFM_trainer, release_gpu_memory, CONFIGS_FOLDER

# ── JSON config ───────────────────────────────────────────────────────────────
# All training settings live in an external JSON file; pass it via --config.
# Anything not present in the file falls back to DEFAULT_CONFIG.
DEFAULT_CONFIG = {
    "name":                 "best_calibrator",
    "debug":                True,
    "seed":                 43,
    "batch_size":           16,
    "continue_training":    False,
    "reuse_pretrain":       False,
    "run_kfold_difficulty": False,
    "kfolds":               5,
    "kfold_epochs":         120,
    "kfold_patience":       12,
    "difficulty_alpha":     0.30,
    "difficulty_clip_min":  0.50,
    "difficulty_clip_max":  2.00,
    "tune": {
        "enabled":    False,
        "params":     ["F_ctrl", "dn"],
        "num_epochs": 20,
        "lr":         1e-3
    },
    "model": {
        "LO_NCPAs":       True,
        "chrom_defocus":  False,
        "use_Moffat":     False,
        "retain_PSDs":    False,
        "N_spline_nodes": 5,
        "Z_mode_max":     9
    },
    "calibrator": {
        "type":         "direct", # "direct" (absolute parameters) or "factor" (corrections to telemetry-derived priors, configs/NFM_factor_calibrator_config.json)
        "hidden_dim":   48,
        "dropout_rate": 0.2,
        "factor_keys":  ["r0", "L0", "wind_speed_single", "Cn2_weights"], # factor type only: parameters predicted as log-factors on telemetry priors
        "const_keys":   ["F_ctrl", "LO_coefs"],                            # factor type only: dataset-level constants, the rest is predicted directly
        "Cn2_mix":      "fixed",   # factor type only, grid Cn2 features: reduction of the 8-node profile to the model layers, "learned" (soft assignment trained end-to-end) or "fixed" (altitude bins)
        "Cn2_targets":  "fits",    # factor type only: pretraining target of the Cn2 corrections, "fits" (log-ratio to the fitted profile) or "prior" (zero: the telemetry profile itself)
        "Cn2_offset":   False      # factor type only: learned dataset-level log-offset of the Cn2 layer weights
    },
    "trainer": {
        "lr":                  0.01,
        "lambda_step":         1,
        "predict_LO_NCPAs":    True,
        "predict_Cn2_profile": True,
        "predict_wind_speed":  True,
        "pre_init_astrometry": True,
        "optimize_astrometry": False,
        "nan_recovery":        True,
        "random_state":        43,
        "num_epochs":          250,
        "patience":            20,
        "max_nan_recoveries":  10,
        "pretrain_epochs":     100,
        "pretrain_patience":   15,
        "pretrain_standardize": True
    },
    # PSF loss, see NFMCalibratorTrainer / BalancedPSFLoss. "legacy" is the linear MSE + MAE loss used before.
    # r_max = 40: beyond it, the model's seeing halo is systematically ~20% fainter than the data (a missing
    # scattered-light component), and fitting it there degrades the AO-corrected halo and the core.
    "loss": {
        "type":             "balanced",
        "radial_power":     2.0,
        "r_max":            40.0,
        "asinh_k":          3.0,
        "huber_delta":      0.1,
        "scale":            3.0,
        "match_background": False,
        "legacy_weight":    0.0,
        "LO_sign_penalty":  False,
        "fit_reg_weight":   0.0,  # parameter-space distance to the per-sample fits, see NFMCalibratorTrainer._regularization
        "shrink_weight":    0.0   # factor calibrator only: shrinkage of the corrections towards the prior
    },
    "fixed_params": [
        "Jxy", "bg_ctrl",
        "dx_ctrl", "dy_ctrl",
        "F_norm", "F_norm_lambda_ctrl", "F_norm_λ_ctrl",
        "src_dirs_x", "src_dirs_y",
        "wind_dir_single"
    ],
    "weights_subdir":     "NFM_calibrator",
    "max_train_samples":  None
}

# The training configs live in calibrators/configs/ (NFM_calibrator_config.json: direct calibrator, NFM_factor_calibrator_config.json:
# factor calibrator). The JSON given with --config overrides DEFAULT_CONFIG above section by section; without --config the direct one is used.
# The validation / ablation / comparison scripts take the same file to rebuild the identical setup and find the run's weights.
_parser = argparse.ArgumentParser(description="Train NFM Calibrator")
_parser.add_argument('--config', type=str, default=str(CONFIGS_FOLDER / 'NFM_calibrator_config.json'), help='JSON training config (default: calibrators/configs/NFM_calibrator_config.json)')
_parser.add_argument('--device', type=str, default=None, help="Torch device, e.g. 'cuda:1' (default: the one of project_config.json)")

try:
    _cli        = _parser.parse_args()
    _cfg_path   = Path(_cli.config) if _cli.config else None
    if _cli.device:
        default_device = torch.device(_cli.device) # overrides the project setting for this run only
except SystemExit:
    _cfg_path   = None


if _cfg_path and _cfg_path.exists():
    with open(_cfg_path) as _f:
        _loaded = json.load(_f)
        
    cfg = deepcopy(DEFAULT_CONFIG)
    
    for _k, _v in _loaded.items():
        if isinstance(_v, dict) and _k in cfg and isinstance(cfg[_k], dict):
            cfg[_k] = {**cfg[_k], **_v}   # shallow-merge nested dicts
        else:
            cfg[_k] = _v
else:
    cfg = deepcopy(DEFAULT_CONFIG)
    if _cfg_path:
        print(f"Warning: config file not found: {_cfg_path}. Using defaults.")


# ── Paths ─────────────────────────────────────────────────────────────────────
MUSE_DATA_FOLDER  = Path(project_settings["MUSE_data_folder"])
STD_FOLDER        = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE     = STD_FOLDER / 'dataset_cache'
WEIGHTS_DIR_CALIB = WEIGHTS_FOLDER / cfg["weights_subdir"]
BEST_CALIB_PATH   = WEIGHTS_DIR_CALIB / f'{cfg["name"]}_checkpoint.pth'

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = DATA_FOLDER / 'logs' # the data folder of project_config.json, independent of the working directory
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'training_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.DEBUG if cfg["debug"] else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")

if _cfg_path:
    logger.info(f"Config loaded from: {_cfg_path}")

# ── Reproducibility ───────────────────────────────────────────────────────────
# Seeds the network initialization, the batch order and the dropout masks, so that runs differing only in the config (e.g. the
# loss) are comparable. The train/val split does not depend on it: it is stored in the dataset cache (see trainer.random_state).
random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
logger.info(f"Seed: {cfg['seed']}")


#%%
dataset     = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_wvl_total = dataset.C
logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W})")

#%%
# PSF model, calibrator network (direct or factor, see cfg["calibrator"]["type"]), outputs transformer and trainer
# When continuing training from existing weights, start with a smaller learning rate than a fresh run to avoid disrupting
# the already-converged calibrator.
trainer_lr = 2.5e-3 if cfg["continue_training"] else cfg["trainer"]["lr"]

if 'PSF_model' in locals():
    del PSF_model

PSF_model, calibrator, outputs_transformer, trainer = build_NFM_trainer(
    cfg         = {**cfg, "trainer": {**cfg["trainer"], "lr": trainer_lr}},
    dataset     = dataset,
    device      = default_device,
    weights_dir = WEIGHTS_DIR_CALIB, # keep split / pretrain / meta / bundle next to the checkpoint
    seed        = cfg["seed"],
)
release_gpu_memory(sync=True)
print(PSF_model.inputs_manager)

N_outputs  = outputs_transformer.get_stacked_size()
N_features = len(dataset.features)
print(f"Calibrator ({trainer.calibrator_type}): {N_features} inputs -> {N_outputs} outputs")

# Build readable output names from the transformer
output_names = []
for param_name, slc in outputs_transformer.slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_names.append(param_name)
    else:
        output_names.extend([f"{param_name}[{i}]" for i in range(width)])

# The trainer takes the train/val split stored in the dataset cache (grouped by acquisition time)
logger.info(f"Split: {len(trainer.train_idx)} train / {len(trainer.val_idx)} val "
            f"({'stored in the dataset cache' if dataset.train_idx is not None else 'random, dataset cache has no split'})")

# Persist the train/val split so the validation script can reproduce it exactly
_split_path = trainer.weights_dir / f'{cfg["name"]}_split.npz'

# ── Optional training-set subsampling (data-ablation experiments) ─────────────
_max_n = cfg["max_train_samples"]
if _max_n is not None and _max_n < len(trainer.train_idx):
    _rng_sub = np.random.default_rng(cfg["trainer"]["random_state"])
    _n_train_full = len(trainer.train_idx)
    trainer.train_idx    = _rng_sub.choice(trainer.train_idx, size=int(_max_n), replace=False)
    trainer.train_loader = trainer._make_loader(trainer.train_idx, shuffle=True)
    logger.info(f"Training subset: {len(trainer.train_idx)} / {_n_train_full} training samples (max_train_samples={_max_n})")

# Sample IDs are stored as well, so that the validation can detect if the dataset cache was regenerated since training
np.savez(
    _split_path,
    train_idx = trainer.train_idx,
    val_idx   = trainer.val_idx,
    train_ids = dataset.sample_ids[trainer.train_idx],
    val_ids   = dataset.sample_ids[trainer.val_idx],
)
logger.info(f"Dataset split saved -> {_split_path}")

#%% ===========================  Pretraining ==================================================
# The pretraining does not depend on the PSF loss, so runs that compare losses can start from the same warm start (reuse_pretrain)
pretrain_path = trainer.weights_dir / 'pretrain_weights.pth'
if pretrain_path.exists() and (cfg["continue_training"] or cfg["reuse_pretrain"]):
    calibrator.load_state_dict(torch.load(pretrain_path, map_location=default_device))
    logger.info(f"Loaded pretrain weights from {pretrain_path}")
else:
    trainer.pretrain(
        num_epochs  = cfg["trainer"]["pretrain_epochs"],
        patience    = cfg["trainer"]["pretrain_patience"],
        save_path   = pretrain_path,
        standardize = cfg["trainer"]["pretrain_standardize"],
    )

release_gpu_memory(sync=True)

val_loss_pretrain = trainer.validate()
logger.info(f"Post-pretrain validation loss: {val_loss_pretrain:.6f}")

#%% ======================== Main training ========================================
if cfg["continue_training"] and BEST_CALIB_PATH.exists():
    trainer.load_checkpoint(BEST_CALIB_PATH)
    # The loaded optimizer state carries the learning rate of the checkpointed epoch; restore the intended one
    for _pg in trainer.optimizer.param_groups[:1]:
        _pg['lr'] = trainer_lr

BASE_STATE     = deepcopy(calibrator.state_dict())
sample_weights = None

if cfg["run_kfold_difficulty"]:
    sample_weights = trainer.run_kfold_difficulty_weighting(
        base_indices       = trainer.train_idx,
        initial_state_dict = BASE_STATE,
        n_splits           = cfg["kfolds"],
        fold_epochs        = cfg["kfold_epochs"],
        fold_patience      = cfg["kfold_patience"],
        alpha              = cfg["difficulty_alpha"],
        clip               = (cfg["difficulty_clip_min"], cfg["difficulty_clip_max"]),
    )

#%% ======================== Main training ========================================
# Back up previous best checkpoint before overwriting
if BEST_CALIB_PATH.exists():
    os.rename(BEST_CALIB_PATH, BEST_CALIB_PATH.with_name(BEST_CALIB_PATH.stem + '_backup.pth'))

train_losses, val_losses = trainer.train(
    num_epochs         = cfg["trainer"]["num_epochs"],
    patience           = cfg["trainer"]["patience"],
    nan_recovery       = cfg["trainer"]["nan_recovery"],
    max_nan_recoveries = cfg["trainer"]["max_nan_recoveries"],
    sample_weights     = sample_weights,
    checkpoint_path    = BEST_CALIB_PATH,
)

release_gpu_memory()
logger.info(f"Training done. Final train={train_losses[-1]:.6f} val={val_losses[-1]:.6f} best={min(val_losses):.6f}")

# Remove per-epoch checkpoint files to keep the directory tidy
def _cleanup_epoch_checkpoints(directory: Path) -> None:
    files = sorted(directory.glob('checkpoint_epoch_*.pth'))
    for f in files:
        f.unlink()
    if files:
        logger.info(f"Cleaned up {len(files)} epoch checkpoint(s) from {directory.name}/")

_cleanup_epoch_checkpoints(trainer.weights_dir)
_kfold_dir = trainer.weights_dir / 'kfold_difficulty'
if _kfold_dir.exists():
    _cleanup_epoch_checkpoints(_kfold_dir)

# ── Save post-training metadata ───────────────────────────────────────────────
_meta = {
    "name":                  cfg["name"],
    "calibrator_type":       trainer.calibrator_type,
    "checkpoint_path":       str(BEST_CALIB_PATH),
    "split_path":            str(_split_path),
    "bundle_path":           str(trainer.weights_dir / f'{cfg["name"]}_bundle.pth'),
    "output_names":          output_names,
    "input_feature_names":   dataset.features,
    "N_outputs":             N_outputs,
    "N_features":            N_features,
    "N_wvl_total":           N_wvl_total,
    "training_completed_at": datetime.now().isoformat(timespec='seconds'),
    "final_train_loss":      float(train_losses[-1]),
    "final_val_loss":        float(val_losses[-1]),
    "best_val_loss":         float(min(val_losses)),
    "tuned_params":          None,
}
_meta_path = trainer.weights_dir / f'{cfg["name"]}_meta.json'
with open(_meta_path, 'w') as _f:
    json.dump(_meta, _f, indent=2)

logger.info(f"Training metadata saved -> {_meta_path}")

# Export the bundle here, before the non-interactive exit below: otherwise CLI runs leave a stale bundle from a
# previous run, which NFMCalibrator (e.g. the ETC validation) would then load instead of the new checkpoint.
trainer.save_calibrator(Path(_meta['bundle_path']))

# Quit here if not run with IPython (e.g. from command line)
if not (hasattr(sys, 'ps1') or sys.flags.interactive):
    logger.info("Non-interactive environment detected. Exiting after training.")
    sys.exit(0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label='Train')
ax.plot(val_losses, label='Val')
ax.set(xlabel='Epoch', ylabel='Loss', title='Training Losses')
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.show()

#%%
# Optional fine-tuning — controlled by cfg["tune"]["enabled"]
if cfg["tune"]["enabled"]:
    trainer.tune_calibrator(
        tuned_params = cfg["tune"]["params"],
        num_epochs   = cfg["tune"]["num_epochs"],
        lr           = cfg["tune"]["lr"],
    )
    _meta["tuned_params"] = cfg["tune"]["params"]

    with open(_meta_path, 'w') as _f:
        json.dump(_meta, _f, indent=2)

    logger.info(f"Post-tune metadata updated -> {_meta_path}")
    trainer.save_calibrator(Path(_meta['bundle_path']))

# %%
