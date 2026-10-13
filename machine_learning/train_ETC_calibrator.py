"""
Build the ETC calibrator bundle for MUSE NFM (see `calibrators/ETC_calibrator.py` and `agents/execplans/ETC_calibrator.md`).

1. The pool of training observations: standardized telemetry, fitted PSF-model parameters and Cn2 layer heights of the training
   split of the calibrator dataset (`muse_STD_stars_dataset.pt`), used by the donor-based and Gaussian predictors.
2. The direct ETC network: `SmallCalibratorNet` from the six ETC inputs (header seeing, tau0, altitude, NGS magnitude, IRLOS
   frequency and gain) to the same PSF-model parameters as the PSF-R calibrator, pretrained on the fitted values and trained
   end-to-end with the PSF model in the loop (`NFMCalibratorTrainer` on `ETCDataset`, configs with the dataset median Cn2 heights).
3. The bundle `data/weights/ETC_calibrator/ETC_calibrator_bundle.pth` (+ `_meta.json`) with the pool, the network, the dataset
   median heights, the median fitted phase bump, a template config and the path of the PSF-R calibrator bundle (from the
   `*_meta.json` of `NFM_calibrator_config.json`).

Requirements: the dataset cache, a CUDA device (~20 s per epoch for the direct network), the trained PSF-R calibrator bundle.
Run from the repository root in the `TipTorch` conda env:

    python machine_learning/train_ETC_calibrator.py [--config machine_learning/ETC_calibrator_config.json] [--skip-direct-net]
                                                     [--psfr-bundle PATH]
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
from calibrators.ETC_calibrator import ETCDataset, ETC_DIRECT_FEATURES

DATASET_CACHE = Path(project_settings['MUSE_STD_data_folder']) / 'dataset_cache' / 'muse_STD_stars_dataset.pt'
DEFAULT_CONFIG = REPO_ROOT / 'machine_learning' / 'ETC_calibrator_config.json'

logger = logging.getLogger('train_ETC_calibrator')


def psfr_bundle_path(cfg):
    """ PSF-R calibrator bundle referenced by the training metadata of `NFM_calibrator_config.json` """
    with open(REPO_ROOT / cfg['psfr_config']) as f:
        psfr_cfg = json.load(f)
    meta_path = WEIGHTS_FOLDER / psfr_cfg['weights_subdir'] / f'{psfr_cfg["name"]}_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            return Path(json.load(f)['bundle_path'])
    return WEIGHTS_FOLDER / psfr_cfg['weights_subdir'] / f'{psfr_cfg["name"]}_bundle.pth'


def build_pool(dataset, idx):
    """ Pool of training observations: standardized telemetry [N, 31], fitted parameters, Cn2 heights [N, 3], sample ids """
    telemetry = np.array([list(dataset.telemetry[i].values()) for i in idx], dtype=np.float32)
    keys = ['F', 'J', 'r0', 'L0', 'dn', 'wind_speed_single', 'Cn2_weights', 'LO_coefs']
    fitted = {}
    for k in keys:
        src = k + '_ctrl' if k in ('F', 'J') and (k + '_ctrl') in dataset.fitted_vals[0] else k
        fitted[k] = np.array([np.asarray(dataset.fitted_vals[i][src], dtype=np.float32) for i in idx])
    heights = np.array([dataset.configs[i]['atmosphere']['Cn2Heights'][0] for i in idx], dtype=float)
    return {'sample_ids': np.asarray(dataset.sample_ids)[idx], 'telemetry_scaled': telemetry, 'fitted': fitted, 'heights': heights}


def make_PSF_model(dataset, cfg, batch_size):
    """ PSF model on a representative batch config, fixed parameters pruned, as in `train_NFM_calibrator.py` """
    batch = tuple(dataset[i] for i in np.random.default_rng(0).integers(0, len(dataset), size=batch_size))
    _, _, _, config, _ = dataset.collate_batch(batch, device=default_device)
    with torch.no_grad():
        PSF_model = PSFModelNFM(config, multiple_obs=True, device=default_device, **cfg['model'])
    for key in cfg['fixed_params']:
        try:
            PSF_model.inputs_manager.delete(key)
        except Exception:
            pass
    PSF_model.inputs_manager.set_optimizable(['LO_coefs', 'Cn2_weights', 'wind_speed_single'], True)
    return PSF_model


def outputs_transformer_of(PSF_model):
    transformer = deepcopy(PSF_model.inputs_manager.get_transformer())
    buf = transformer.unstack(PSF_model.inputs_manager.stack())
    if 'LO_coefs' in buf:
        buf['LO_coefs'] = buf['LO_coefs'][:, 1:]  # the phase bump is not predicted
    transformer.stack(buf)
    return transformer


def train_direct_net(dataset, cfg, weights_dir, median_heights):
    """ End-to-end training of the direct ETC network; returns the state to store in the bundle and the loss curves """
    d = cfg['direct_net']
    etc_dataset = ETCDataset(dataset, ETC_DIRECT_FEATURES, heights=median_heights if d.get('train_heights', 'median') == 'median' else None)
    PSF_model   = make_PSF_model(etc_dataset, cfg, cfg['batch_size'])
    transformer = outputs_transformer_of(PSF_model)
    net = SmallCalibratorNet(len(ETC_DIRECT_FEATURES), transformer.get_stacked_size(), d['hidden_dim'], d['dropout_rate']).to(default_device)
    logger.info(f'Direct network: {len(ETC_DIRECT_FEATURES)} ETC inputs -> {transformer.get_stacked_size()} outputs ({list(transformer.slices)})')

    trainer = NFMCalibratorTrainer(
        PSF_model=PSF_model, calibrator=net, dataset=etc_dataset, outputs_transformer=transformer, device=default_device,
        batch_size=cfg['batch_size'], lr=d['lr'], lambda_step=d['lambda_step'], predict_LO_NCPAs=True, predict_Cn2_profile=True,
        pre_init_astrometry=True, optimize_astrometry=False, random_state=d['random_state'], collate_fn=NFMDataset.collate_batch,
        loss_cfg=cfg['loss'], weights_dir=weights_dir,
    )
    trainer.pretrain(num_epochs=d['pretrain_epochs'], patience=d['pretrain_patience'], save_path=weights_dir / 'direct_net_pretrain.pth',
                     standardize=d['pretrain_standardize'])
    logger.info(f'Post-pretrain validation PSF loss: {trainer.validate():.5f}')
    train_losses, val_losses = trainer.train(num_epochs=d['num_epochs'], patience=d['patience'], checkpoint_path=weights_dir / 'direct_net_checkpoint.pth')
    for f in weights_dir.glob('checkpoint_epoch_*.pth'):
        f.unlink()
    mean, scale = trainer.dataset.base.telemetry_preprocessor['mean'], trainer.dataset.base.telemetry_preprocessor['scale']
    idx = [dataset.features.index(f) for f in ETC_DIRECT_FEATURES]
    state = {
        'arch': {'n_features': len(ETC_DIRECT_FEATURES), 'n_outputs': transformer.get_stacked_size(), 'hidden_dim': d['hidden_dim'], 'dropout_rate': d['dropout_rate']},
        'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()},
        'outputs_transformer': transformer.save(),
        'output_feature_names': list(transformer.slices.keys()),
        'feature_names': list(ETC_DIRECT_FEATURES),
        'feature_mean': np.asarray(mean)[idx], 'feature_scale': np.asarray(scale)[idx],
        'train_heights': d.get('train_heights', 'median'),
        'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': float(min(val_losses)),
    }
    PSF_model.cleanup()
    release_gpu_memory(sync=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--skip-direct-net', action='store_true', help='Only build the pool (no network training)')
    parser.add_argument('--psfr-bundle', type=Path, default=None, help='PSF-R calibrator bundle to reference (default: from NFM_calibrator_config.json metadata)')
    parser.add_argument('--name', default=None, help='Bundle name (default: config "name")')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
        
    name        = args.name or cfg['name']
    weights_dir = WEIGHTS_FOLDER / cfg['weights_subdir']
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / 'data' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', force=True,
                        handlers=[logging.FileHandler(log_dir / f'training_ETC_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), logging.StreamHandler(sys.stdout)])

    dataset = NFMDataset(DATASET_CACHE)
    if dataset.telemetry_preprocessor is None:
        raise ValueError('The dataset cache has no telemetry preprocessor; regenerate it')
    train_idx, val_idx = np.asarray(dataset.train_idx, int), np.asarray(dataset.val_idx, int)
    logger.info(f'Dataset: {len(dataset)} samples ({len(train_idx)} train / {len(val_idx)} val), {len(dataset.features)} telemetry features')

    pool = build_pool(dataset, train_idx)
    all_heights = np.array([dataset.configs[i]['atmosphere']['Cn2Heights'][0] for i in train_idx], dtype=float)  # training split only
    median_heights = np.median(all_heights, axis=0)
    phase_bump_median = float(np.median(pool['fitted']['LO_coefs'][:, 0]))
    template_config = deepcopy(dataset.configs[int(train_idx[0])])
    template_config['atmosphere']['Cn2Heights'] = [median_heights.tolist()]
    logger.info(f'Pool: {len(train_idx)} observations; median Cn2 heights {median_heights.round(0).tolist()} m; median phase bump {phase_bump_median:.1f}')

    psfr_bundle = args.psfr_bundle or psfr_bundle_path(cfg)
    if not Path(psfr_bundle).exists():
        logger.warning(f'PSF-R calibrator bundle not found: {psfr_bundle} (the calibrator-based predictors will fail to load until it exists)')

    direct_net = None
    if cfg['direct_net'].get('enabled', True) and not args.skip_direct_net:
        direct_net = train_direct_net(dataset, cfg, weights_dir, median_heights.tolist())
        logger.info(f"Direct network trained: best validation PSF loss {direct_net['best_val_loss']:.5f}")

    bundle_path = weights_dir / f'{name}_bundle.pth'
    if bundle_path.exists() and direct_net is None:  # keep a previously trained network when only the pool is rebuilt
        old = torch.load(bundle_path, map_location='cpu', weights_only=False)
        direct_net = old.get('direct_net')
        if direct_net is not None:
            logger.info('Kept the direct network of the existing bundle')

    state = {
        'name': name,
        'etc_features': list(ETC_DIRECT_FEATURES),
        'telemetry_preprocessor': dataset.telemetry_preprocessor,
        'pool': pool,
        'val_sample_ids': np.asarray(dataset.sample_ids)[val_idx],
        'median_heights': median_heights.tolist(),
        'phase_bump_median': phase_bump_median,
        'template_config': template_config,
        'model_cfg': cfg['model'],
        'pool_k': cfg['pool']['k'],
        'direct_net': direct_net,
        'psfr_calibrator_bundle': str(psfr_bundle),
        'dataset_cache': str(DATASET_CACHE),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    torch.save(state, bundle_path)
    
    meta = {k: state[k] for k in ('name', 'etc_features', 'median_heights', 'phase_bump_median', 'psfr_calibrator_bundle', 'dataset_cache', 'created_at')}
    meta.update({'bundle_path': str(bundle_path), 'N_pool': int(len(train_idx)), 'N_val': int(len(val_idx)),
                 'direct_net_best_val_loss': None if direct_net is None else direct_net['best_val_loss']})
    
    with open(weights_dir / f'{name}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    logger.info(f'ETC calibrator bundle -> {bundle_path}')
