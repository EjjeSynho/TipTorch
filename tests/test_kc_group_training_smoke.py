"""
Smoke test of the k_c-group handling in the NFM calibrator trainer on the real dataset cache.

The dataset cache carries the samples flagged 'k_c mismatch' (k_c group / fraction and the scaled DM pitch of their configs, stored by
build_dataset_cache); a trainer is built from the factor calibrator config and evaluated on a mixed subset:
  - every batch holds a single k_c group (KcGroupBatchSampler) and the configs merge without error,
  - run_model rebuilds the model grids (Update(grids=True)) exactly when the k_c of the batch changes, and the model's k_c follows
    the batch's group,
  - the validation loss is finite and a backward pass through a mismatched batch gives finite gradients.

Requires the dataset cache of the config's regime (muse_STD_stars_dataset.pt for raw frames, ~3.6 GB) and a GPU (~3 min).
Run:  python tests/test_kc_group_training_smoke.py [cuda:N]
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'datasets' / 'MUSE_STD_dataset'))

import numpy as np
import torch

from STD_dataset_utils import kc_fraction_from_config
from machine_learning.calibrators.NFM_calibrator import NFMDataset, build_NFM_trainer, release_gpu_memory, load_STD_dataset

device = torch.device(sys.argv[1]) if len(sys.argv) > 1 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
results = []
def check(name, ok, info=''):
    results.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({info})' if info else ''), flush=True)

cfg = json.load(open(ROOT / 'machine_learning/calibrators/configs/NFM_factor_calibrator_config.json'))
cfg['weights_subdir'] = 'kc_group_smoke_test'

dataset = load_STD_dataset(cfg['derotated_PSFs'])
groups = dataset.kc_group
print('k_c groups in the cache:', {int(g): int(n) for g, n in zip(*np.unique(groups, return_counts=True))})
check('the cache holds mismatched samples of several k_c groups', len(np.unique(groups)) >= 3)
check('configs of the flagged samples carry the scaled pitch',
      all(np.isclose(kc_fraction_from_config(dataset.configs[i]), dataset.kc_fraction[i], atol=1e-6) for i in np.flatnonzero(groups > 0)))

weights_dir = Path(__file__).resolve().parent / '_kc_group_smoke_weights'
PSF_model, calibrator, outputs_transformer, trainer = build_NFM_trainer(cfg, dataset, device, weights_dir=weights_dir, seed=cfg['seed'])
release_gpu_memory()
kc_nominal = float(PSF_model.model.kc)
check('trainer built from a single-k_c representative batch', np.isclose(kc_nominal, 1.0 / (2 * 0.22), rtol=1e-6), f'k_c = {kc_nominal:.4f} 1/m')
print(trainer._kc_groups_summary())

# Mixed subset: a few nominal samples and all the mismatched ones; count the grid rebuilds and the model k_c per batch
rng = np.random.default_rng(0)
subset = np.concatenate([rng.choice(np.flatnonzero(groups == 0), size=20, replace=False), np.flatnonzero(groups > 0)])
loader = trainer._make_loader(subset, shuffle=True)

model = trainer.PSF_model.model
n_grid_updates, kc_per_batch, groups_per_batch = [0], [], []
_orig_update = model.Update
def _counting_update(*args, **kwargs):
    if kwargs.get('grids', False): n_grid_updates[0] += 1
    return _orig_update(*args, **kwargs)
model.Update = _counting_update
_orig_run_model = trainer.run_model
def _recording_run_model(x_dict, config, idx, lambda_ids):
    out = _orig_run_model(x_dict, config, idx, lambda_ids)
    kc_per_batch.append(float(model.kc)); groups_per_batch.append(set(groups[idx.cpu().numpy()].tolist()))
    return out
trainer.run_model = _recording_run_model

with torch.no_grad():
    val_loss = trainer.validate(loader=loader)
check('validation over the mixed subset runs and gives a finite loss', np.isfinite(val_loss), f'loss {val_loss:.4f}, {len(loader)} batches')
check('every batch holds one k_c group', all(len(g) == 1 for g in groups_per_batch), f'{len(groups_per_batch)} run_model calls')
expected_kc = [kc_nominal * float(np.mean(dataset.kc_fraction[groups == next(iter(g))])) for g in groups_per_batch]
check('the model k_c follows the k_c group of every batch', np.allclose(kc_per_batch, expected_kc, rtol=1e-6),
      f'k_c per batch {np.round(np.unique(kc_per_batch), 4).tolist()}')
switches = sum(1 for a, b in zip(kc_per_batch[:-1], kc_per_batch[1:]) if abs(a - b) > 1e-9) + (1 if abs(kc_per_batch[0] - kc_nominal) > 1e-9 else 0)
n_sets = len(trainer.lambda_id_sets)
check('the grids are rebuilt exactly when the k_c changes' if n_sets == 1 else 'the grids are rebuilt at least when the k_c changes',
      n_grid_updates[0] == switches if n_sets == 1 else n_grid_updates[0] >= switches, f'{n_grid_updates[0]} rebuilds for {switches} k_c switches, {n_sets} wavelength set(s)')

# Backward pass through a mismatched batch
trainer.calibrator.train()
for PSF_data_b, tel_b, fit_b, config_b, idxs_b in trainer._make_loader(np.flatnonzero(groups == groups.max()), shuffle=False):
    x_pred, x_dict = trainer._predict(tel_b)
    lambda_ids = trainer.lambda_id_sets[0]
    PSF_pred_b = trainer.run_model(x_dict, config_b, idxs_b, lambda_ids)
    loss = trainer.loss_fn(PSF_data_b[:, lambda_ids, ...], PSF_pred_b, x_dict)
    loss = loss.mean() if loss.dim() > 0 else loss
    trainer.calibrator.zero_grad(); loss.backward()
    grads_ok = all(torch.isfinite(p.grad).all().item() for p in trainer.calibrator.parameters() if p.grad is not None)
    check('backward pass through a mismatched batch: finite loss and gradients', np.isfinite(loss.item()) and grads_ok, f'loss {loss.item():.4f}, batch of {len(idxs_b)}')
    break

print(f"\n{sum(results)} / {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
