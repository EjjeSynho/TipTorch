"""
Data-ablation study for the NFM calibrator.

Trains the calibrator repeatedly with progressively fewer training samples.
Each sample-count level is repeated N_REPEATS times with different randomly
selected subsets (different seeds) to accumulate statistics.

The validation set is FIXED across all runs so results are directly comparable.

Artefacts saved under WEIGHTS_FOLDER / <ablation.output_subdir>:

    NFM_data_ablation/
        ablation_N100_r0/
            checkpoint.pth       ← best calibrator weights
            meta.json            ← all metadata (config, indices, losses, …)
            val_cubes.pt         ← dict with PSF cubes + predictions
        ablation_N100_r1/
            ...
        ablation_full_r0/
            ...
        ablation_summary.json    ← list of completed runs + best_val_loss table

Usage:
    python run_ablation_NFM_calibrator.py --config NFM_calibrator_config.json
"""

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
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

from tiptorch._config import *
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from calibrators.NFM_calibrator import SmallCalibratorNet, NFMCalibratorTrainer, NFMDataset, release_gpu_memory

# ── Base training config (mirrors DEFAULT_CONFIG in train_NFM_calibrator.py) ──
_BASE_DEFAULT = {
    "name":                 "best_calibrator",
    "debug":                False,
    "batch_size":           16,
    "continue_training":    False,
    "run_kfold_difficulty": False,
    
    "tune": {
        "enabled": False,
        "params": [],
        "num_epochs": 20,
        "lr": 1e-3
    },
    
    "model": {
        "LO_NCPAs": True,
        "chrom_defocus": False,
        "use_Moffat": False,
        "retain_PSDs": False,
        "N_spline_nodes": 5,
        "Z_mode_max": 9
    },
    
    "calibrator": {
        "hidden_dim": 48,
        "dropout_rate": 0.2
    },
    
    "trainer": {
        "lr": 0.01,
        "lambda_step": 1,
        "predict_LO_NCPAs": True,
        "predict_Cn2_profile": True,
        "predict_wind_speed": True,
        "pre_init_astrometry": True,
        "optimize_astrometry": False,
        "random_state": 43,
        "num_epochs": 250,
        "patience": 20,
        "nan_recovery": True,
        "max_nan_recoveries": 10,
        "pretrain_epochs": 100,
        "pretrain_patience": 15,
    },
    
    "fixed_params": [
        "Jxy", "bg_ctrl", "dx_ctrl", "dy_ctrl",
        "F_norm", "F_norm_lambda_ctrl", "src_dirs_x", "src_dirs_y",
        "wind_dir_single", "F_norm_λ_ctrl",
    ],
    
    "weights_subdir":    "NFM_calibrator",
    "max_train_samples": None,
    # ── ablation-specific ──────────────────────────────────────────────────────
    "ablation": {
        "sample_counts":        [50, 100, 150, 200, 300, 400],
        "n_repeats":            2,
        "base_seed":            200,   # repeat k uses seed base_seed + k*1000
        "output_subdir":        "NFM_data_ablation",
        "include_full_dataset": True,  # also benchmark the full training set
    },
}

# ── Parse CLI ─────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="NFM Calibrator – Data Ablation Study")
_parser.add_argument('--config', type=str, default=None,
                     help='Path to the same JSON config used for training')
try:
    _cli      = _parser.parse_args()
    _cfg_path = Path(_cli.config) if _cli.config else None
except SystemExit:
    _cfg_path = None

if _cfg_path and _cfg_path.exists():
    with open(_cfg_path) as _f:
        _loaded = json.load(_f)
    cfg = deepcopy(_BASE_DEFAULT)
    for _k, _v in _loaded.items():
        if isinstance(_v, dict) and _k in cfg and isinstance(cfg[_k], dict):
            cfg[_k] = {**cfg[_k], **_v}
        else:
            cfg[_k] = _v
else:
    cfg = deepcopy(_BASE_DEFAULT)
    if _cfg_path:
        print(f"Warning: config not found at {_cfg_path}. Using defaults.")

# Fall back to default config JSON next to this script when running interactively
if 'cfg' not in dir():
    _script_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
    _default_cfg_path = _script_dir / 'NFM_calibrator_config.json'
    if _default_cfg_path.exists():
        with open(_default_cfg_path) as _f:
            _loaded = json.load(_f)
        cfg = deepcopy(_BASE_DEFAULT)
        for _k, _v in _loaded.items():
            if isinstance(_v, dict) and _k in cfg and isinstance(cfg[_k], dict):
                cfg[_k] = {**cfg[_k], **_v}
            else:
                cfg[_k] = _v

abl = cfg["ablation"]    # shorthand

# ── Paths & logging ───────────────────────────────────────────────────────────
STD_FOLDER    = Path(project_settings["MUSE_STD_data_folder"])
DATASET_CACHE = STD_FOLDER / 'dataset_cache'
ABLATION_DIR  = WEIGHTS_FOLDER / abl["output_subdir"]
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

log_dir = Path('../data/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = log_dir / f'ablation_NFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG if cfg["debug"] else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(f"Ablation artefacts → {ABLATION_DIR}")
logger.info(f"Log → {log_filename}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cleanup_epoch_checkpoints(directory: Path) -> None:
    files = sorted(directory.glob('checkpoint_epoch_*.pth'))
    for f in files:
        f.unlink()
    if files:
        logger.info(f"  Cleaned {len(files)} epoch checkpoint(s) from {directory.name}/")


def _build_outputs_transformer(PSF_model):
    from copy import deepcopy
    ot = deepcopy(PSF_model.inputs_manager.get_transformer())
    _buf = ot.unstack(PSF_model.inputs_manager.stack())
    if 'LO_coefs' in _buf:
        _buf['LO_coefs'] = _buf['LO_coefs'][:, 1:]   # strip phase-bump column
    _ = ot.stack(_buf)
    return ot


#%% ── 1. Load dataset (shared across all runs) ──────────────────────────────
dataset     = NFMDataset(DATASET_CACHE / 'muse_STD_stars_dataset.pt')
N_wvl_total = dataset.C
logger.info(f"Dataset: {len(dataset)} samples | cubes ({dataset.C}, {dataset.H}, {dataset.W})")

#%% ── 2. Initialise PSF model (shared across all runs) ──────────────────────
_tmp_batch  = tuple([dataset[i] for i in np.random.randint(0, len(dataset), size=cfg["batch_size"])])
_, _, _, _tmp_config, _ = NFMDataset.collate_batch(_tmp_batch, default_device)

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

for _key in cfg["fixed_params"]:
    try:
        PSF_model.inputs_manager.delete(_key)
    except Exception:
        pass
if PSF_model.use_Moffat:
    PSF_model.inputs_manager.delete('theta')
    PSF_model.inputs_manager.delete('ratio')

PSF_model.inputs_manager.set_optimizable(['LO_coefs'],          cfg["trainer"]["predict_LO_NCPAs"])
PSF_model.inputs_manager.set_optimizable(['Cn2_weights'],       cfg["trainer"]["predict_Cn2_profile"])
PSF_model.inputs_manager.set_optimizable(['wind_speed_single'], cfg["trainer"]["predict_wind_speed"])

outputs_transformer = _build_outputs_transformer(PSF_model)
N_outputs  = outputs_transformer.get_stacked_size()
N_features = len(dataset.features)

output_names = []
for _pname, _slc in outputs_transformer.slices.items():
    _w = _slc.stop - _slc.start
    output_names += [_pname] if _w == 1 else [f"{_pname}[{i}]" for i in range(_w)]

logger.info(f"PSF model ready | calibrator: {N_features} → {N_outputs}")

#%% ── 3. Compute fixed train / val split ─────────────────────────────────────
# All ablation runs share the same validation set for fair comparison.
_ref_split_seed = cfg["trainer"]["random_state"]
full_train_idx, val_idx = train_test_split(
    np.arange(len(dataset)),
    test_size    = 0.2,
    random_state = _ref_split_seed,
)
logger.info(
    f"Fixed split (seed={_ref_split_seed}): "
    f"{len(full_train_idx)} train / {len(val_idx)} val"
)

# Save the fixed split so the evaluation script can verify it
np.savez(ABLATION_DIR / 'fixed_split.npz', train_idx=full_train_idx, val_idx=val_idx)

#%% ── 4. Ablation loop ───────────────────────────────────────────────────────
sample_counts = list(abl["sample_counts"])
if abl["include_full_dataset"]:
    sample_counts = [None] + sample_counts   # None = full training set (baseline)

summary_rows = []  # accumulated across all runs

for n_samples in sample_counts:
    label = "full" if n_samples is None else str(n_samples)
    _n_actual = len(full_train_idx) if n_samples is None else min(int(n_samples), len(full_train_idx))

    for repeat_idx in range(abl["n_repeats"]):
        run_seed = abl["base_seed"] + repeat_idx * 1000
        run_name = f"ablation_N{label}_r{repeat_idx}"
        run_dir  = ABLATION_DIR / run_name
        run_dir.mkdir(exist_ok=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN  {run_name}  | n_train={_n_actual}  seed={run_seed}")
        logger.info(f"{'='*60}")

        # ── Sub-sample training indices ──────────────────────────────────────
        rng_sub = np.random.default_rng(run_seed)
        if n_samples is not None and n_samples < len(full_train_idx):
            run_train_idx = rng_sub.choice(full_train_idx, size=_n_actual, replace=False)
        else:
            run_train_idx = full_train_idx.copy()

        # ── Fresh calibrator ─────────────────────────────────────────────────
        calibrator = SmallCalibratorNet(
            n_features   = N_features,
            n_outputs    = N_outputs,
            hidden_dim   = cfg["calibrator"]["hidden_dim"],
            dropout_rate = cfg["calibrator"]["dropout_rate"],
        ).to(default_device)

        # ── Fresh trainer (pre-loaded with fixed split) ──────────────────────
        trainer = NFMCalibratorTrainer(
            PSF_model           = PSF_model,
            calibrator          = calibrator,
            dataset             = dataset,
            outputs_transformer = deepcopy(outputs_transformer),
            device              = default_device,
            batch_size          = cfg["batch_size"],
            lr                  = cfg["trainer"]["lr"],
            lambda_step         = cfg["trainer"]["lambda_step"],
            predict_LO_NCPAs    = cfg["trainer"]["predict_LO_NCPAs"],
            predict_Cn2_profile = cfg["trainer"]["predict_Cn2_profile"],
            random_state        = run_seed,
            pre_init_astrometry = cfg["trainer"]["pre_init_astrometry"],
            optimize_astrometry = cfg["trainer"]["optimize_astrometry"],
            collate_fn          = NFMDataset.collate_batch,
            train_idx           = run_train_idx,
            val_idx             = val_idx,
        )

        # ── Pretrain ─────────────────────────────────────────────────────────
        pretrain_path = run_dir / 'pretrain_weights.pth'
        trainer.pretrain(
            num_epochs = cfg["trainer"]["pretrain_epochs"],
            patience   = cfg["trainer"]["pretrain_patience"],
            save_path  = pretrain_path,
        )
        release_gpu_memory(sync=True)
        logger.info(f"  Pretrain done | post-pretrain val = {trainer.validate():.6f}")

        # ── Main training ────────────────────────────────────────────────────
        checkpoint_path = run_dir / 'checkpoint.pth'
        train_losses, val_losses = trainer.train(
            num_epochs         = cfg["trainer"]["num_epochs"],
            patience           = cfg["trainer"]["patience"],
            nan_recovery       = cfg["trainer"]["nan_recovery"],
            max_nan_recoveries = cfg["trainer"]["max_nan_recoveries"],
            checkpoint_path    = checkpoint_path,
        )
        release_gpu_memory()
        _cleanup_epoch_checkpoints(run_dir)
        logger.info(
            f"  Training done | "
            f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}  "
            f"best={min(val_losses):.6f}"
        )

        # ── Validate & collect PSF cubes ──────────────────────────────────────
        trainer.load_checkpoint(checkpoint_path)
        (PSFs_pred_cube, PSFs_data_cube,
         validation_ids, NN_predictions,
         telemetry_vecs, final_val_loss) = trainer.validate(return_cubes=True)

        val_cubes_path = run_dir / 'val_cubes.pt'
        torch.save({
            'PSFs_pred':      PSFs_pred_cube.cpu(),
            'PSFs_data':      PSFs_data_cube.cpu(),
            'validation_ids': validation_ids.cpu(),
            'NN_predictions': NN_predictions.cpu(),
            'telemetry_vecs': telemetry_vecs.cpu(),
            'final_val_loss': final_val_loss,
        }, val_cubes_path)
        logger.info(f"  Val cubes saved → {val_cubes_path.name}  (loss={final_val_loss:.6f})")

        # ── Per-run metadata ─────────────────────────────────────────────────
        run_meta = {
            "run_name":        run_name,
            "n_samples":       None if n_samples is None else int(n_samples),
            "n_train_actual":  int(_n_actual),
            "n_val":           int(len(val_idx)),
            "repeat_idx":      repeat_idx,
            "run_seed":        run_seed,
            "split_seed":      _ref_split_seed,
            "train_idx":       run_train_idx.tolist(),
            "val_idx":         val_idx.tolist(),
            "checkpoint_path": str(checkpoint_path),
            "val_cubes_path":  str(val_cubes_path),
            "output_names":    output_names,
            "input_feature_names": dataset.features,
            "N_outputs":       N_outputs,
            "N_features":      N_features,
            "N_wvl_total":     N_wvl_total,
            "train_losses":    [float(x) for x in train_losses],
            "val_losses":      [float(x) for x in val_losses],
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss":   float(val_losses[-1]),
            "best_val_loss":    float(min(val_losses)),
            "completed_at":     datetime.now().isoformat(timespec='seconds'),
            "config":           cfg,
        }
        with open(run_dir / 'meta.json', 'w') as _f:
            json.dump(run_meta, _f, indent=2)

        summary_rows.append({
            "run_name":        run_name,
            "n_samples":       run_meta["n_samples"],
            "n_train_actual":  run_meta["n_train_actual"],
            "repeat_idx":      repeat_idx,
            "run_seed":        run_seed,
            "best_val_loss":   run_meta["best_val_loss"],
            "final_val_loss":  run_meta["final_val_loss"],
            "n_epochs":        len(train_losses),
            "completed_at":    run_meta["completed_at"],
        })

        # ── Update rolling summary after every run ───────────────────────────
        with open(ABLATION_DIR / 'ablation_summary.json', 'w') as _f:
            json.dump(summary_rows, _f, indent=2)

        del calibrator, trainer, PSFs_pred_cube, PSFs_data_cube
        del NN_predictions, telemetry_vecs, validation_ids
        release_gpu_memory(sync=True)

# ── Final summary ─────────────────────────────────────────────────────────────
logger.info("\n" + "="*60)
logger.info("ABLATION COMPLETE")
logger.info(f"{'n_train':>10}  {'repeat':>6}  {'best_val':>10}")
logger.info("-"*30)
for row in summary_rows:
    n_lbl = "full" if row["n_samples"] is None else str(row["n_samples"])
    logger.info(f"{n_lbl:>10}  {row['repeat_idx']:>6}  {row['best_val_loss']:>10.6f}")
logger.info("="*60)
logger.info(f"Summary → {ABLATION_DIR / 'ablation_summary.json'}")
