"""
Build the ETC calibrator bundle for MUSE NFM (`calibrators/ETC_calibrator.py`, ExecPlan `agents/execplans/ETC_calibrator.md`).

1. Donor pool: ETC inputs, fitted PSF-model parameters and Cn2 layer heights of the training split of the calibrator dataset. The
   samples flagged 'k_c mismatch' are left out: their fits compensate a reduced k_c, while the ETC renders the nominal one.
2. Direct network: `SmallCalibratorNet` from the six ETC inputs to the PSF-model parameters, pretrained on the fitted values and
   trained end-to-end with the PSF model (`build_NFM_trainer` on `ETCDataset`, configs with the training-median Cn2 heights). All
   training samples are used: their configs carry their own DM pitch, and the trainer batches them per k_c group.
3. Bundle `data/weights/ETC_calibrator/ETC_calibrator_bundle.pth` (+ `_meta.json`).

The PSF-model regime (anisotropic jitter, image-frame wing; J_angle and the wing shape at the fitted medians), the loss and the dataset cache (raw
frames or derotated PSFs) come from the config, laid out like the NFM calibrator ones (configs/NFM_*calibrator_config.json): keep the
"model", "loss" and "fixed_params" sections in sync with them and with STD_fitter.py.

Requirements: the dataset cache (with its telemetry preprocessor and stored split) and a CUDA device (~20 s per epoch).
Run from the repository root in the `TipTorch` conda env:

    python machine_learning/calibrators/train_ETC_calibrator.py [--config machine_learning/calibrators/configs/ETC_calibrator_config.json] [--skip-direct-net]
"""

import sys
import json
import random
import logging
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'machine_learning'))

from tiptorch._config import WEIGHTS_FOLDER, default_device
from calibrators.NFM_calibrator import NFMDataset, build_NFM_trainer, fixed_inputs, fits_median, model_flags, release_gpu_memory, load_STD_dataset
from calibrators.ETC_calibrator import ETCDataset, ETC_INPUTS, dataset_cache_path, donor_keys

# The ETC calibrator config lives with the other calibrator configs in machine_learning/calibrators/configs/
DEFAULT_CONFIG = REPO_ROOT / 'machine_learning' / 'calibrators' / 'configs' / 'ETC_calibrator_config.json'

logger = logging.getLogger('train_ETC_calibrator')


def build_pool(dataset, idx, preprocessor, keys):
    """ ETC inputs [N, 6] (physical units), fitted parameters `keys` (PSF-model inputs), Cn2 heights [N, 3] and sample ids of the observations `idx` """
    feat    = [dataset.features.index(f) for f in ETC_INPUTS]
    scaled  = np.array([list(dataset.telemetry[i].values()) for i in idx], dtype=np.float64)[:, feat]
    missing = [k for k in keys if k not in dataset.fitted_vals[int(idx[0])]]

    if missing:
        raise KeyError(f'The fits of the dataset carry no {missing}: the "model" section of the config does not match the regime of the fits')
    fitted = {k: np.array([np.asarray(dataset.fitted_vals[i][k], dtype=np.float32) for i in idx]) for k in keys}

    return {
        'sample_ids': np.asarray(dataset.sample_ids)[idx],
        'etc_inputs': scaled * preprocessor['scale'][feat] + preprocessor['mean'][feat],
        'fitted':     fitted,
        'heights':    np.array([dataset.configs[i]['atmosphere']['Cn2Heights'][0] for i in idx], dtype=float),
    }


def fits_median_inputs(dataset, idx, model_cfg):
    """ Kept fixed inputs set from the fits ("fixed_from_fits" of the model section: J_angle, wing shape): medians over `idx` as
    {key: tensor [1, ...]}, as build_NFM_trainer sets them (used when the direct network is not retrained) """
    out = {}
    for key in model_cfg.get('fixed_from_fits', []):
        median   = fits_median(dataset, idx, key) # [n_values]
        out[key] = median if median.numel() == 1 else median.view(1, -1)
    return out


def train_direct_net(dataset, cfg, weights_dir, median_heights, train_idx, val_idx):
    """ End-to-end training of the direct network; returns the state stored in the bundle and the kept fixed inputs / regime of its PSF model """
    etc_dataset = ETCDataset(dataset, heights=median_heights)
    PSF_model, net, transformer, trainer = build_NFM_trainer(cfg, etc_dataset, default_device, weights_dir=weights_dir, seed=cfg['seed'],
                                                             train_idx=train_idx, val_idx=val_idx)
    logger.info(f'Direct network: {len(ETC_INPUTS)} ETC inputs -> {transformer.get_stacked_size()} outputs {list(transformer.slices)}')
    t = cfg['trainer']
    trainer.pretrain(num_epochs=t['pretrain_epochs'], patience=t['pretrain_patience'], standardize=t['pretrain_standardize'],
                     save_path=weights_dir / 'direct_net_pretrain.pth')
    train_losses, val_losses = trainer.train(num_epochs=t['num_epochs'], patience=t['patience'], nan_recovery=t['nan_recovery'],
                                             max_nan_recoveries=t['max_nan_recoveries'], checkpoint_path=weights_dir / 'direct_net_checkpoint.pth')

    for f in weights_dir.glob('checkpoint_epoch_*.pth'):
        f.unlink()

    feat = [dataset.features.index(f) for f in ETC_INPUTS]
    state = {
        'arch': {'n_features': len(ETC_INPUTS), 'n_outputs': transformer.get_stacked_size(), 'hidden_dim': cfg['calibrator']['hidden_dim'],
                 'dropout_rate': cfg['calibrator']['dropout_rate']},
        'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()},
        'outputs_transformer': transformer.save(),
        'feature_mean': dataset.telemetry_preprocessor['mean'][feat], 'feature_scale': dataset.telemetry_preprocessor['scale'][feat],
        'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': float(min(val_losses)),
        'model_cfg': deepcopy(cfg['model']),
    }
    kept, flags = fixed_inputs(PSF_model), model_flags(PSF_model)
    PSF_model.cleanup()

    release_gpu_memory(sync=True)
    return state, kept, flags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--skip-direct-net', action='store_true', help='Only rebuild the pool (keeps the network of an existing bundle)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    weights_dir = WEIGHTS_FOLDER / cfg['weights_subdir']
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / 'data' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', force=True,
                        handlers=[logging.FileHandler(log_dir / f'training_ETC_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), logging.StreamHandler(sys.stdout)])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])

    # Raw frames with the per-sample pupil angles or derotated PSFs, as the NFM calibrator (see train_NFM_calibrator.py)
    dataset_cache = dataset_cache_path()
    dataset = load_STD_dataset(cfg['derotated_PSFs'])

    pre = dataset.telemetry_preprocessor
    if pre is None:
        raise ValueError('The dataset cache has no telemetry preprocessor; regenerate it')

    train_idx, val_idx = np.asarray(dataset.train_idx, int), np.asarray(dataset.val_idx, int)
    feat = [dataset.features.index(f) for f in ETC_INPUTS]
    logger.info(f"Dataset {dataset_cache.name}: {len(dataset)} samples ({len(train_idx)} train / {len(val_idx)} val), "
                f"{'derotated PSFs' if dataset.derotated else 'raw frames'}")

    # Donor pool and template config from the nominal-k_c training samples (k_c group 0)
    nominal  = np.asarray(dataset.kc_group, int) == 0
    pool_idx = train_idx[nominal[train_idx]]
    pool = build_pool(dataset, pool_idx, pre, donor_keys(cfg['model']))
    median_heights = np.median(pool['heights'], axis=0).tolist()
    template_config = deepcopy(dataset.configs[int(pool_idx[0])])
    template_config['atmosphere']['Cn2Heights'] = [median_heights]
    logger.info(f"Pool: {len(pool_idx)} observations ({len(train_idx) - len(pool_idx)} 'k_c mismatch' training samples left out), "
                f"keys {list(pool['fitted'])}; median Cn2 heights {np.round(median_heights).tolist()} m")

    bundle_path = weights_dir / f'{cfg["name"]}_bundle.pth'
    direct_net, kept, flags = None, None, None

    if not args.skip_direct_net:
        direct_net, kept, flags = train_direct_net(dataset, cfg, weights_dir, median_heights, train_idx, val_idx)
        logger.info(f"Direct network: best validation PSF loss {direct_net['best_val_loss']:.5f}")
    elif bundle_path.exists():
        old = torch.load(bundle_path, map_location='cpu', weights_only=False)
        direct_net = old.get('direct_net')
        if direct_net is not None and direct_net.get('model_cfg', old.get('model_cfg')) != cfg['model']:
            logger.warning('The kept direct network was trained with another "model" section than the config: retrain it (no --skip-direct-net)')
        logger.info('Kept the direct network of the existing bundle')

    # Kept fixed inputs of the renderer (J_angle and the wing shape at the fitted training medians)
    kept = kept if kept is not None else fits_median_inputs(dataset, train_idx, cfg['model'])
    logger.info(f"Fixed inputs: { {k: np.round(v.numpy(), 3).tolist() for k, v in kept.items()} }")

    state = {
        'name': cfg['name'],
        'etc_inputs': list(ETC_INPUTS),
        'etc_mean': pre['mean'][feat],
        'etc_scale': pre['scale'][feat],
        'pool': pool,
        'val_sample_ids': np.asarray(dataset.sample_ids)[val_idx],
        'median_heights': median_heights,
        'phase_bump_median': float(np.median(pool['fitted']['LO_coefs'][:, 0])),
        'template_config': template_config,
        'model_cfg': cfg['model'],
        'model_flags': flags,
        'fixed_inputs': kept,
        'direct_net': direct_net,
        'dataset_cache': str(dataset_cache),
        'derotated': dataset.derotated,
        'training_config': deepcopy(cfg),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }

    torch.save(state, bundle_path)
    meta = { k: state[k] for k in ('name', 'etc_inputs', 'median_heights', 'phase_bump_median', 'dataset_cache', 'derotated', 'created_at') }
    meta.update({'bundle_path': str(bundle_path), 'N_pool': int(len(pool_idx)), 'N_val': int(len(val_idx)), 'model_cfg': cfg['model'],
                 'fixed_inputs': {k: v.numpy().tolist() for k, v in kept.items()},
                 'direct_net_best_val_loss': None if direct_net is None else direct_net['best_val_loss']})

    with open(weights_dir / f'{cfg["name"]}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f'ETC calibrator bundle -> {bundle_path}')
