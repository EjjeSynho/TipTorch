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
import logging
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from datetime import datetime

from tiptorch._config import *
from calibrators.NFM_calibrator import SmallCalibratorNet, NFMCalibratorTrainer, NFMDataset, release_gpu_memory

# ── JSON config ───────────────────────────────────────────────────────────────
# All training settings live in an external JSON file; pass it via --config.
# Anything not present in the file falls back to DEFAULT_CONFIG.
DEFAULT_CONFIG = {
    "name":                 "best_calibrator",
    "debug":                False,
    "batch_size":           16,
    "continue_training":    False,
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
        "hidden_dim":   48,
        "dropout_rate": 0.2
    },
    "trainer": {
        "lr":                  0.01,
        "lambda_step":         1,
        "predict_LO_NCPAs":    True,
        "predict_Cn2_profile": True,
        "predict_wind_speed":  True,
        "pre_init_astrometry": True,
        "optimize_astrometry": False,
        "random_state":        43,
        "num_epochs":          250,
        "patience":            20,
        "nan_recovery":        True,
        "max_nan_recoveries":  10,
        "pretrain_epochs":     100,
        "pretrain_patience":   15
    },
    "fixed_params": [
        "Jxy", "bg_ctrl", "dx_ctrl", "dy_ctrl",
        "F_norm", "F_norm_lambda_ctrl", "src_dirs_x", "src_dirs_y",
        "wind_dir_single", "F_norm_λ_ctrl"
    ],
    "weights_subdir":     "NFM_calibrator",
    "max_train_samples":  null
}

_parser = argparse.ArgumentParser(description="Train NFM Calibrator")
_parser.add_argument('--config', type=str, default=None, help='Path to JSON training config file')
try:
    _cli        = _parser.parse_args()
    _cfg_path   = Path(_cli.config) if _cli.config else None
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
log_dir = Path('../data/logs')
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


#%%
dataset     = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_wvl_total = dataset.C
logger.info(f"Dataset loaded: {len(dataset)} samples, PSF cubes ({dataset.C}, {dataset.H}, {dataset.W})")

#%%
# Initialize the PSF model with a representative batch config
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

_tmp_batch  = tuple([dataset[i] for i in np.random.randint(0, len(dataset), size=cfg["batch_size"])])
_, _, _, _tmp_config, _ = dataset.collate_batch(_tmp_batch, device=default_device)


if 'PSF_model' in locals():
    del PSF_model

with torch.no_grad():
    PSF_model = PSFModelNFM(
        _tmp_config,
        multiple_obs   = True,
        LO_NCPAs       = cfg["model"]["LO_NCPAs"],
        chrom_defocus  = cfg["model"]["chrom_defocus"],
        use_Moffat     = cfg["model"]["use_Moffat"],
        retain_PSDs    = cfg["model"]["retain_PSDs"],
        N_spline_nodes = cfg["model"]["N_spline_nodes"],
        Z_mode_max     = cfg["model"]["Z_mode_max"],
        device         = default_device,
    )

del _tmp_batch, _tmp_config
release_gpu_memory(sync=True)

# fixed_params = [
#     'Jxy', 'bg_ctrl', 'dx_ctrl', 'dy_ctrl', 'L0',
#     'F_norm', 'F_norm_lambda_ctrl', 'src_dirs_x', 'src_dirs_y',
#     'wind_speed_single', 'wind_dir_single', 'F_norm_λ_ctrl'
# ]

fixed_params = cfg["fixed_params"]

# Remove parameters that are fixed externally and do not need to be predicted
for _key in fixed_params:
    try:
        PSF_model.inputs_manager.delete(_key)
    except Exception:
        pass   # key may have an alternate name; delete what is present

if PSF_model.use_Moffat:
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

predict_LO_NCPAs    = cfg["trainer"]["predict_LO_NCPAs"]
predict_Cn2_profile = cfg["trainer"]["predict_Cn2_profile"]
predict_wind_speed  = cfg["trainer"]["predict_wind_speed"]

PSF_model.inputs_manager.set_optimizable(['LO_coefs'],          predict_LO_NCPAs)
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'],       predict_Cn2_profile)
PSF_model.inputs_manager.set_optimizable(['wind_speed_single'], predict_wind_speed)

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

# Build readable output names from the transformer
output_names = []
for param_name, slc in outputs_transformer.slices.items():
    width = slc.stop - slc.start
    if width == 1:
        output_names.append(param_name)
    else:
        output_names.extend([f"{param_name}[{i}]" for i in range(width)])

#%%
# Calibrator network  (can be swapped for any nn.Module with the same I/O contract)
calibrator = SmallCalibratorNet(
    n_features   = N_features,
    n_outputs    = N_outputs,
    hidden_dim   = cfg["calibrator"]["hidden_dim"],
    dropout_rate = cfg["calibrator"]["dropout_rate"],
).to(default_device)


# Instantiate the trainer
trainer = NFMCalibratorTrainer(
    PSF_model           = PSF_model,
    calibrator          = calibrator,
    dataset             = dataset,
    outputs_transformer = outputs_transformer,
    device              = default_device,
    batch_size          = cfg["batch_size"],
    lr                  = cfg["trainer"]["lr"],
    lambda_step         = cfg["trainer"]["lambda_step"],
    predict_LO_NCPAs    = predict_LO_NCPAs,
    predict_Cn2_profile = predict_Cn2_profile,
    random_state        = cfg["trainer"]["random_state"],
    pre_init_astrometry = cfg["trainer"]["pre_init_astrometry"],
    optimize_astrometry = cfg["trainer"]["optimize_astrometry"],
    collate_fn          = NFMDataset.collate_batch,
)

# Persist the train/val split so the validation script can reproduce it exactly
_split_path = trainer.weights_dir / f'{cfg["name"]}_split.npz'

# ── Optional training-set subsampling (data-ablation experiments) ─────────────
_max_n = cfg["max_train_samples"]
if _max_n is not None and _max_n < len(trainer.train_idx):
    _rng_sub = np.random.default_rng(cfg["trainer"]["random_state"])
    trainer.train_idx    = _rng_sub.choice(trainer.train_idx, size=int(_max_n), replace=False)
    trainer.train_loader = trainer._make_loader(trainer.train_idx, shuffle=True)
    logger.info(
        f"Training subset: {len(trainer.train_idx)} / "
        f"{len(trainer.train_idx) + len(trainer.val_idx)} samples "
        f"(max_train_samples={_max_n})"
    )

np.savez(_split_path, train_idx=trainer.train_idx, val_idx=trainer.val_idx)
logger.info(f"Dataset split saved -> {_split_path}")

#%% ===========================  Pretraining ==================================================
pretrain_path = trainer.weights_dir / 'pretrain_weights.pth'
if (not cfg["continue_training"]) or not pretrain_path.exists():
    trainer.pretrain(
        num_epochs = cfg["trainer"]["pretrain_epochs"],
        patience   = cfg["trainer"]["pretrain_patience"],
        save_path  = pretrain_path,
    )
else:
    calibrator.load_state_dict(torch.load(pretrain_path, map_location=default_device))
    logger.info(f"Loaded pretrain weights from {pretrain_path}")

release_gpu_memory(sync=True)

val_loss_pretrain = trainer.validate()
logger.info(f"Post-pretrain validation loss: {val_loss_pretrain:.6f}")

#%% ======================== Main training ========================================
if cfg["continue_training"] and BEST_CALIB_PATH.exists():
    trainer.load_checkpoint(BEST_CALIB_PATH)

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
# Optional fine-tuning — controlled by cfg["tune"]["enabled"]
if cfg["tune"]["enabled"]:
    trainer.tune_calibrator(
        tuned_params = cfg["tune"]["params"],
        num_epochs   = cfg["tune"]["num_epochs"],
        lr           = cfg["tune"]["lr"],
    )
    _meta["tuned_params"] = cfg["tune"]["params"]

#%%
with open(_meta_path, 'w') as _f:
    json.dump(_meta, _f, indent=2)
logger.info(f"Post-tune metadata updated -> {_meta_path}")


# %%
trainer.save_calibrator(Path(_meta['bundle_path']))
