#%%
"""
Side-by-side validation metrics of NFM calibrators (direct or factor) trained with the same config family on the same split:
the core / halo metrics of calibrators/NFM_validation_utils.py for every checkpoint or bundle given on the command line, plus the
per-sample fits (the floor of the PSF model) and, optionally, the pretrain-only network.

    python machine_learning/analysis/compare_NFM_calibrators.py [--config <training config>] [--device cuda:1] [--out table.json] \\
        name=path/to/checkpoint_or_bundle.pth [name=path ...]

Every run must have the same outputs (key groups of the calibrator section, Z_mode_max, N_spline_nodes) as the config: the PSF model and the trainer
are built once from the config, only the network weights (and the factor constants / parameterization) are swapped.
A factor checkpoint can only be evaluated with a factor config, and vice versa.
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from calibrators.NFM_calibrator import NFMDataset, build_NFM_trainer, release_gpu_memory, load_STD_dataset, CONFIGS_FOLDER
from calibrators.NFM_validation_utils import predict_cubes, core_halo_metrics, metrics_table

_parser = argparse.ArgumentParser(description="Compare NFM calibrators on the validation set")
_parser.add_argument('runs', nargs='*', help='name=path (checkpoint with calibrator_state_dict, or bundle with net_state_dict)')
_parser.add_argument('--config', type=str, default=None, help='training config (default: calibrators/configs/NFM_calibrator_config.json)')
_parser.add_argument('--device', type=str, default=None)
_parser.add_argument('--out',    type=str, default=None, help='JSON file for the metrics')
_parser.add_argument('--split',  type=str, default=None, help='split file of a training run (default: the split of the dataset cache)')
try:
    _args = _parser.parse_args()
except SystemExit:
    _args = _parser.parse_args([])

with open(_args.config or CONFIGS_FOLDER / 'NFM_calibrator_config.json') as f:
    cfg = json.load(f)

device     = torch.device(_args.device) if _args.device else default_device
STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])

#%%
dataset = load_STD_dataset(cfg['derotated_PSFs'])
PSF_model, calibrator, outputs_transformer, trainer = build_NFM_trainer(cfg, dataset, device, weights_dir=DATA_FOLDER / 'temp')

if _args.split:
    with np.load(_args.split) as f:
        split = dict(f)
    
    id_to_pos = {sample_id: pos for pos, sample_id in enumerate(dataset.sample_ids)}
    trainer.val_idx    = np.array([id_to_pos[i] for i in split['val_ids']], dtype=int)
    trainer.val_loader = trainer._make_loader(trainer.val_idx, shuffle=False)

lambda_nm = trainer.lambda_full.cpu().numpy().ravel() * 1e9
print(f'{len(trainer.val_idx)} validation samples, calibrator type {trainer.calibrator_type}')

#%%
results = {}

def evaluate(name, predict):
    D, P, _ = predict_cubes(trainer, predict)
    results[name] = core_halo_metrics(D, P, trainer, lambda_nm)
    release_gpu_memory()
    print(f"{name:28s} balanced loss {results[name].get('balanced_loss_mean', float('nan')):.4f}  "
          f"|peak err| {results[name]['peak_err_pct_median']:.1f} %  EE2 err {results[name]['EE2_err_pct_median']:.1f} %", flush=True)

for run in _args.runs:
    name, path = run.split('=', 1)
    state = torch.load(path, map_location=device, weights_only=False)
    
    if 'calibrator_state_dict' in state:   # checkpoint
        trainer.calibrator.load_state_dict(state['calibrator_state_dict'])
        trainer._load_extra_state(state)
    else:                                  # bundle or pretrain weights
        trainer.calibrator.load_state_dict(state.get('net_state_dict', state))
        trainer._load_extra_state(state)
        
    evaluate(name, lambda tel_b, fit_b, idxs_b: trainer._predict(tel_b)[1])

# The per-sample fits: directly (the floor of the PSF model), and through the parameterization (direct: normalized fits, factor: the
# corrections reproducing them)
evaluate('fits (oracle)', lambda tel_b, fit_b, idxs_b: trainer._calibrator_friendly_fit_dict(fit_b))
evaluate('fits via parameterization', lambda tel_b, fit_b, idxs_b: trainer._params_from_pred(trainer._pretrain_target_vector(fit_b, tel=tel_b), tel_b))

print('\n' + metrics_table(results))

if _args.out:
    with open(_args.out, 'w') as f:
        json.dump(results, f, indent=1)
        
    print(f'Saved -> {_args.out}')
