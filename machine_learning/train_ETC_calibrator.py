"""
Build the ETC calibrator bundle for MUSE NFM (`calibrators/ETC_calibrator.py`, ExecPlan `agents/execplans/ETC_calibrator.md`).

1. Donor pool: ETC inputs, fitted PSF-model parameters and Cn2 layer heights of the training split of the calibrator dataset.
2. Direct network: `SmallCalibratorNet` from the six ETC inputs to the PSF-model parameters, pretrained on the fitted values and
   trained end-to-end with the PSF model (`NFMCalibratorTrainer` on `ETCDataset`, configs with the training-median Cn2 heights).
3. Bundle `data/weights/ETC_calibrator/ETC_calibrator_bundle.pth` (+ `_meta.json`).

Requirements: the dataset cache (with its telemetry preprocessor and stored split) and a CUDA device (~20 s per epoch).
Run from the repository root in the `TipTorch` conda env:

    python machine_learning/train_ETC_calibrator.py [--config machine_learning/ETC_calibrator_config.json] [--skip-direct-net]
"""

import sys
import json
import logging
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'machine_learning'))

from tiptorch._config import WEIGHTS_FOLDER, default_device, project_settings
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from calibrators.NFM_calibrator import NFMDataset, NFMCalibratorTrainer, SmallCalibratorNet, release_gpu_memory
from calibrators.ETC_calibrator import ETCDataset, ETC_INPUTS

DATASET_CACHE  = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache' / 'muse_STD_stars_dataset.pt'
DEFAULT_CONFIG = REPO_ROOT / 'machine_learning' / 'ETC_calibrator_config.json'
FITTED_KEYS    = ['F', 'J', 'r0', 'L0', 'dn', 'wind_speed_single', 'Cn2_weights', 'LO_coefs']

logger = logging.getLogger('train_ETC_calibrator')


def build_pool(dataset, idx, preprocessor):
    """ ETC inputs [N, 6] (physical units), fitted parameters, Cn2 heights [N, 3] and sample ids of the observations `idx` """
    feat = [dataset.features.index(f) for f in ETC_INPUTS]
    scaled = np.array([list(dataset.telemetry[i].values()) for i in idx], dtype=np.float64)[:, feat]
    fitted = {k: np.array([np.asarray(dataset.fitted_vals[i][k + '_ctrl' if k in ('F', 'J') else k], dtype=np.float32) for i in idx]) for k in FITTED_KEYS}
    return {
        'sample_ids': np.asarray(dataset.sample_ids)[idx],
        'etc_inputs': scaled * preprocessor['scale'][feat] + preprocessor['mean'][feat],
        'fitted':     fitted,
        'heights':    np.array([dataset.configs[i]['atmosphere']['Cn2Heights'][0] for i in idx], dtype=float),
    }


def train_direct_net(dataset, cfg, weights_dir, median_heights):
    """ End-to-end training of the direct network; returns the state stored in the bundle """
    d = cfg['direct_net']
    etc_dataset = ETCDataset(dataset, heights=median_heights)
    batch = tuple(etc_dataset[i] for i in np.random.default_rng(0).integers(0, len(dataset), size=cfg['batch_size']))
    _, _, _, config, _ = etc_dataset.collate_batch(batch, device=default_device)
    with torch.no_grad():
        PSF_model = PSFModelNFM(config, multiple_obs=True, device=default_device, **cfg['model'])
    for key in cfg['fixed_params']:
        try:
            PSF_model.inputs_manager.delete(key)
        except Exception:
            pass
    PSF_model.inputs_manager.set_optimizable(['LO_coefs', 'Cn2_weights', 'wind_speed_single'], True)

    transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
    buf = transformer.unstack(PSF_model.inputs_manager.stack())
    buf['LO_coefs'] = buf['LO_coefs'][:, 1:]  # the phase bump is not predicted
    transformer.stack(buf)

    net = SmallCalibratorNet(len(ETC_INPUTS), transformer.get_stacked_size(), d['hidden_dim'], d['dropout_rate']).to(default_device)
    logger.info(f'Direct network: {len(ETC_INPUTS)} ETC inputs -> {transformer.get_stacked_size()} outputs {list(transformer.slices)}')
    trainer = NFMCalibratorTrainer(
        PSF_model=PSF_model, calibrator=net, dataset=etc_dataset, outputs_transformer=transformer, device=default_device,
        batch_size=cfg['batch_size'], lr=d['lr'], lambda_step=1, random_state=d['random_state'], collate_fn=NFMDataset.collate_batch,
        loss_cfg=cfg['loss'], weights_dir=weights_dir,
    )
    trainer.pretrain(num_epochs=d['pretrain_epochs'], patience=d['pretrain_patience'], save_path=weights_dir / 'direct_net_pretrain.pth')
    train_losses, val_losses = trainer.train(num_epochs=d['num_epochs'], patience=d['patience'], checkpoint_path=weights_dir / 'direct_net_checkpoint.pth')
    for f in weights_dir.glob('checkpoint_epoch_*.pth'):
        f.unlink()
    feat = [dataset.features.index(f) for f in ETC_INPUTS]
    state = {
        'arch': {'n_features': len(ETC_INPUTS), 'n_outputs': transformer.get_stacked_size(), 'hidden_dim': d['hidden_dim'], 'dropout_rate': d['dropout_rate']},
        'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()},
        'outputs_transformer': transformer.save(),
        'feature_mean': dataset.telemetry_preprocessor['mean'][feat], 'feature_scale': dataset.telemetry_preprocessor['scale'][feat],
        'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': float(min(val_losses)),
    }
    PSF_model.cleanup()
    release_gpu_memory(sync=True)
    return state


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

    dataset = NFMDataset(DATASET_CACHE)
    pre = dataset.telemetry_preprocessor
    if pre is None:
        raise ValueError('The dataset cache has no telemetry preprocessor; regenerate it')
    train_idx, val_idx = np.asarray(dataset.train_idx, int), np.asarray(dataset.val_idx, int)
    feat = [dataset.features.index(f) for f in ETC_INPUTS]
    logger.info(f'Dataset: {len(dataset)} samples ({len(train_idx)} train / {len(val_idx)} val)')

    pool = build_pool(dataset, train_idx, pre)
    median_heights = np.median(pool['heights'], axis=0).tolist()
    template_config = deepcopy(dataset.configs[int(train_idx[0])])
    template_config['atmosphere']['Cn2Heights'] = [median_heights]
    logger.info(f'Pool: {len(train_idx)} observations; median Cn2 heights {np.round(median_heights).tolist()} m')

    bundle_path = weights_dir / f'{cfg["name"]}_bundle.pth'
    direct_net = None
    if not args.skip_direct_net:
        direct_net = train_direct_net(dataset, cfg, weights_dir, median_heights)
        logger.info(f"Direct network: best validation PSF loss {direct_net['best_val_loss']:.5f}")
    elif bundle_path.exists():
        direct_net = torch.load(bundle_path, map_location='cpu', weights_only=False).get('direct_net')
        logger.info('Kept the direct network of the existing bundle')

    state = {
        'name': cfg['name'], 'etc_inputs': list(ETC_INPUTS), 'etc_mean': pre['mean'][feat], 'etc_scale': pre['scale'][feat],
        'pool': pool, 'val_sample_ids': np.asarray(dataset.sample_ids)[val_idx], 'median_heights': median_heights,
        'phase_bump_median': float(np.median(pool['fitted']['LO_coefs'][:, 0])), 'template_config': template_config,
        'model_cfg': cfg['model'], 'direct_net': direct_net, 'dataset_cache': str(DATASET_CACHE),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    torch.save(state, bundle_path)
    meta = {k: state[k] for k in ('name', 'etc_inputs', 'median_heights', 'phase_bump_median', 'dataset_cache', 'created_at')}
    meta.update({'bundle_path': str(bundle_path), 'N_pool': int(len(train_idx)), 'N_val': int(len(val_idx)),
                 'direct_net_best_val_loss': None if direct_net is None else direct_net['best_val_loss']})
    with open(weights_dir / f'{cfg["name"]}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f'ETC calibrator bundle -> {bundle_path}')
