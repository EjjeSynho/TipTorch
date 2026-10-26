"""
Round-trip test of the NFM calibrator bundle (`NFMCalibratorTrainer.save_calibrator` -> `NFMCalibrator`): the telemetry
scaler + imputer stored in the bundle must be the one of the dataset cache, reproduce the training telemetry vectors from the raw
MUSE telemetry (`muse_df.pickle` -> filter / reduce / scale / impute), and the loaded network must reproduce the checkpoint.

1. The bundle's `telemetry_preprocessor` state (and Cn2 grid) equals the dataset cache's one, feature names match the calibrator inputs;
   the bundle carries the telemetry resources (LGS / IRLOS archive tables, median Cn2 profile).
2. `NFMCalibrator.prepare_telemetry(raw telemetry)` reproduces the dataset telemetry vectors of every sample (atol 1e-5).
3. `NFMCalibrator.forward_prepared` reproduces the best checkpoint's network, with the phase-bump column prepended to LO_coefs.
4. `NFMCalibrator.save` -> reload gives the same outputs and the same preprocessor; `inverse_transform` inverts the scaling.

Needs the dataset cache built in the run's regime (`dataset_cache/muse_STD_stars_dataset.pt`, raw frames or derotated, ~3.6 GB), `muse_df.pickle` of the STD folder and the
training outputs `best_calibrator_{bundle,checkpoint}.pth` + `best_calibrator_meta.json` in `data/weights/NFM_calibrator/`.
CPU only, ~1 min. Run from the repository root in the `TipTorch` conda env:

    python tests/test_NFM_calibrator_bundle.py [--weights-subdir NFM_calibrator] [--name best_calibrator] [--n-samples 64]
    python tests/test_NFM_calibrator_bundle.py --weights-subdir NFM_factor_calibrator --name factor_calibrator   (factor bundle)
"""

import sys
import json
import pickle
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# sys.path.insert(0, str(REPO_ROOT / 'machine_learning'))

from tiptorch._config import project_settings, WEIGHTS_FOLDER
from data_processing.telemetry_preprocessor import TelemetryPreprocessor
from machine_learning.calibrators.NFM_calibrator import NFMCalibrator, NFMFactorCalibrator, NFMDataset, SmallCalibratorNet, load_NFM_calibrator, load_STD_dataset


def check(condition, message):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {message}')
    return bool(condition)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--weights-subdir', default='NFM_calibrator')
    parser.add_argument('--name', default='best_calibrator')
    parser.add_argument('--n-samples', type=int, default=64, help='samples checked in the network / raw-telemetry tests')
    args = parser.parse_args()

    device = torch.device('cpu')
    weights_dir = WEIGHTS_FOLDER / args.weights_subdir
    std_folder = Path(project_settings['MUSE_STD_data_folder'])
    meta = json.load(open(weights_dir / f'{args.name}_meta.json'))
    bundle_path, checkpoint_path = Path(meta['bundle_path']), Path(meta['checkpoint_path'])

    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    # The dataset cache of the regime the calibrator was trained in (configs without the key predate the raw frames: derotated)
    train_cfg = bundle.get('training_config') or meta.get('config') or {}
    dataset = load_STD_dataset(train_cfg.get('derotated_PSFs', True))
    calibrator = load_NFM_calibrator(bundle_path, device=device)
    print(f'bundle type: {type(calibrator).__name__}')
    results = []

    # 1. Preprocessor of the bundle vs. the dataset cache
    cache_state = dataset.telemetry_preprocessor
    results.append(check(cache_state is not None, 'dataset cache carries a telemetry preprocessor'))
    bundle_state = bundle.get('telemetry_preprocessor')
    results.append(check(bundle_state is not None, 'bundle carries a telemetry preprocessor'))
    if cache_state is not None and bundle_state is not None:
        same = all(
            (np.array_equal(np.asarray(cache_state[k]), np.asarray(bundle_state[k])) if k != 'meta' else cache_state[k] == bundle_state[k])
            for k in cache_state
        )
        results.append(check(same, 'bundle preprocessor state == dataset cache preprocessor state'))
    results.append(check(list(calibrator.features) == list(dataset.features) == list(bundle['input_feature_names']),
                         f'feature names of the preprocessor, the dataset and the bundle match ({len(dataset.features)} features)'))
    if dataset.Cn2_grid is not None: # caches since 2026-10-13: the Cn2 grid of the features goes into the bundle
        same_grid = bundle.get('Cn2_grid') is not None and all(np.allclose(np.asarray(dataset.Cn2_grid[k]), np.asarray(bundle['Cn2_grid'][k])) for k in dataset.Cn2_grid)
        results.append(check(same_grid and calibrator.Cn2_binning == 'grid', f'bundle carries the Cn2 grid of the dataset cache ({calibrator.Cn2_grid})'))
    else:
        results.append(check(calibrator.Cn2_binning == 'kmeans', 'old cache without Cn2 grid: the bundle uses the K-means Cn2 features'))
    # The telemetry resources of the distributed bundle (data_processing/NFM_bundle.py, stage 1): the bundle alone processes new data
    results.append(check(sorted(bundle.get('archive_telemetry', {})) == ['IRLOS_regimes', 'LGS_flux', 'LGS_slopes'] and bundle.get('median_Cn2_profile') is not None,
                         'bundle carries the LGS / IRLOS archive tables and the median Cn2 profile'))

    # 2. Raw telemetry -> prepared vectors reproduces the dataset
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False))
    ids = dataset.sample_ids[pos]
    
    with open(std_folder / 'muse_df.pickle', 'rb') as fh:
        muse_df = pickle.load(fh)
        
    prepared = calibrator.prepare_telemetry(muse_df.loc[ids].copy())
    stored = torch.stack([dataset[int(i)][1] for i in pos]).to(device)
    err = (prepared - stored).abs().max().item()
    results.append(check(err < 1e-5, f'prepare_telemetry(raw muse_df) reproduces the dataset telemetry vectors (max |diff| = {err:.2e})'))

    # 3. Network of the bundle vs. the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = SmallCalibratorNet(**bundle['net_arch']).to(device)
    net.load_state_dict(ckpt['calibrator_state_dict'])
    net.eval()
    with torch.no_grad():
        y_ckpt = net(stored)
        x_pred = calibrator.forward_prepared(stored)
        y_bundle = calibrator.net(stored)
    err = (y_ckpt - y_bundle).abs().max().item()
    results.append(check(err == 0.0, f'bundle network == best checkpoint network (max |diff| = {err:.2e}, epoch {ckpt.get("epoch")})'))
    with torch.no_grad():
        x_pred_ckpt = calibrator._params_from_pred(y_ckpt, stored)
    err = max((x_pred[k][:, 1:] if k == 'LO_coefs' else x_pred[k]).sub(x_pred_ckpt[k]).abs().max().item() for k in x_pred_ckpt)
    results.append(check(err == 0.0, f'forward_prepared reproduces the parameters of the checkpoint network (max |diff| = {err:.2e})'))
    n_LO = bundle['LO_modes_max'] - 1
    results.append(check('LO_coefs' in x_pred and x_pred['LO_coefs'].shape == (len(pos), n_LO) and torch.all(x_pred['LO_coefs'][:, 0] == 0),
                         f'forward_prepared prepends the zero phase-bump column: LO_coefs {tuple(x_pred["LO_coefs"].shape)}'))
    results.append(check(list(bundle['output_feature_names']) == list(calibrator.outputs_transformer.slices.keys()),
                         'outputs transformer slices match the stored output names'))
    if isinstance(calibrator, NFMFactorCalibrator):
        P = calibrator.parameterization
        results.append(check(set(P.const_keys) == set(calibrator.consts) and all(k in P.slices for k in P.factor_keys + P.direct_keys),
                             f'factor bundle: parameterization and constants ({P.factor_keys} / {P.direct_keys} / {P.const_keys})'))
        base = P.base_values(stored)
        results.append(check(all(torch.isfinite(v).all() for v in base.values()) and bool((base['Cn2_weights'].sum(-1) - 1).abs().max() < 1e-5),
                             'factor bundle: telemetry base values are finite, Cn2 profile normalized'))

    # 4. save -> reload round trip and the inverse scaling
    with tempfile.TemporaryDirectory() as tmp:
        rt_path = calibrator.save(Path(tmp) / 'bundle_roundtrip.pth', note='round-trip test')
        reloaded = NFMCalibrator(rt_path, device=device)
        with torch.no_grad():
            err = (reloaded.net(stored) - y_bundle).abs().max().item()
        rt_state = torch.load(rt_path, map_location=device, weights_only=False)
        same_pre = all(
            (np.array_equal(np.asarray(bundle_state[k]), np.asarray(rt_state['telemetry_preprocessor'][k])) if k != 'meta' else True)
            for k in bundle_state
        )
        results.append(check(err == 0.0 and same_pre and rt_state.get('fine_tuning') == {'note': 'round-trip test'},
                             'NFMCalibrator.save -> reload keeps the network, the preprocessor and the metadata'))
    physical = calibrator.telemetry_preprocessor.inverse_transform(stored.numpy())
    rescaled = calibrator.telemetry_preprocessor.scale_features(physical)
    err = np.abs(rescaled - stored.numpy()).max()
    results.append(check(err < 1e-4, f'inverse_transform -> scale_features round trip (max |diff| = {err:.2e})'))

    n_fail = results.count(False)
    print(f'\n{len(results) - n_fail} / {len(results)} checks passed')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
