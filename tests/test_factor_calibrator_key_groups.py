"""
Synthetic end-to-end test of the key groups of the factor NFM calibrator (KEY_GROUPS in NFM_calibrator.py): predicted_keys and factor_keys
(the network outputs, per sample), optimized_shared_keys (one value for all samples, trained with the network), fitted_keys (the per-sample
fits) and fitted_shared_keys (the medians of the fits, set in the PSF model); every other input is left to the PSF model. No STD-star data
needed: the "data" PSFs are rendered by the PSF model itself with a jitter stretch shared by all stars (J_aniso = 1.4), while the per-sample
"fits" put it at ~1.1; its angle J_angle is fitted right (~25 deg). The key groups are those of the NFM factor config (J_aniso optimized
shared, J_angle fitted shared), with wing_frac_ctrl moved to fitted_shared_keys.

1. Key groups: the calibrator outputs are the predicted, factor and optimized shared keys, the network outputs the predicted and factor ones;
   the trainer gets the fitted and fitted shared keys; overlapping groups, a factor key without a telemetry baseline and a direct calibrator
   with optimized shared keys raise; every training config has the five groups and none of the former settings.
2. The optimized shared keys start at the medians of the training fits (normalized storage), the fitted shared keys (J_angle, wing) are the
   medians in the PSF model, the fitted keys the per-sample fits; run_model passes the per-sample fits, the broadcast shared values and the
   per-sample phase bump, and none of the inputs left to the model except the ones its forward needs.
3. Training (pretraining + a short end-to-end run, the optimized shared keys at a raised learning rate): J_aniso moves towards the truth and,
   with the network fixed, lowers the training loss; the fitted (shared) keys, J_angle included, do not move.
4. Checkpoint round trip, validation cubes, tune_calibrator skipping the optimized shared keys, bundle -> NFMFactorCalibrator reproducing the
   trainer, calibrate_prepared setting the learned J_aniso and the fitted shared values (J_angle, wing) in a fresh PSF model, the fitted keys
   left alone.

Needs parameter_files/muse_ltao.ini and the VLT pupil of the TipTorch data folder and a GPU (~40 s). The loss is restricted to the 61 px
frames (r_max / noise_r_min). Run from the repository root:

    python tests/test_factor_calibrator_key_groups.py [cuda:N]
"""
import sys, json, warnings, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tiptorch._config import DATA_FOLDER
from tiptorch.managers.config_manager import ConfigManager
from tiptorch.tools.utils import r0 as r0_from_seeing
from machine_learning.calibrators.NFM_calibrator import (NFMDataset, build_NFM_trainer, build_PSF_model, load_NFM_calibrator, NFMFactorCalibrator,
                                                         calibrator_key_groups, release_gpu_memory, KEY_GROUPS, MODEL_REQUIRED_INPUTS)

device  = torch.device(sys.argv[1]) if len(sys.argv) > 1 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_tmp    = tempfile.TemporaryDirectory()
scratch = Path(_tmp.name) # weights of the run: pretraining, checkpoint, bundle
sys.stdout.reconfigure(errors='backslashreplace') # input names such as F_norm_λ_ctrl on a non-UTF-8 console
results = []
def check(name, ok, info=''):
    results.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({info})' if info else ''), flush=True)

torch.manual_seed(0)
rng   = np.random.default_rng(0)
N, FoV, WVLS = 24, 61, [500e-9, 700e-9, 900e-9]
TRUE_ANISO, TRUE_ANGLE = 1.4, 25.0


def make_config(seeing, wind, Cn2_w, L0):
    cm = ConfigManager()
    c  = cm.Load(DATA_FOLDER / 'parameter_files/muse_ltao.ini')
    c['atmosphere']['Cn2Heights']    = [[0.0, 8500.0, 17400.0]]
    c['atmosphere']['Cn2Weights']    = [list(Cn2_w)]
    c['atmosphere']['WindSpeed']     = [[wind, 0.0, 0.0]]
    c['atmosphere']['WindDirection'] = [[0.0, 0.0, 0.0]]
    c['atmosphere']['L0']            = [L0]
    c['atmosphere']['Seeing']        = [seeing]
    c['telescope']['TelescopeDiameter'] = 8.0
    c['telescope']['ZenithAngle'] = [30.0]
    c['telescope']['Azimuth']     = [0.0]
    c['telescope']['PupilAngle']  = 0.0
    c['sources_science']['Wavelength'] = WVLS
    c['sources_LO']['Wavelength'] = (1215 + 1625) / 2.0 * 1e-9
    c['sources_HO']['Zenith']     = [[10.0] * 4]
    c['sensor_science']['PixelScale']  = 25.0
    c['sensor_science']['FieldOfView'] = FoV
    c['sensor_HO']['NumberPhotons'] = [[2000.0] * 4]
    c['sensor_HO']['ClockRate']     = np.mean([c['sensor_HO']['ClockRate']])
    c['sensor_LO']['PixelScale']    = 78.0
    c['sensor_LO']['NumberPhotons'] = [300.0]
    c['sensor_LO']['SigmaRON']      = 1.0
    c['sensor_LO']['Gain']          = [1.0]
    c['RTC']['SensorFrameRate_LO']  = [500.0]
    c['RTC']['SensorFrameRate_HO']  = [c['RTC']['SensorFrameRate_HO']]
    c['RTC']['LoopDelaySteps_HO']   = [c['RTC']['LoopDelaySteps_HO']]
    c['RTC']['LoopGain_HO']         = [c['RTC']['LoopGain_HO']]
    c['DM']['DmPitchs'][0] = 0.22
    c = cm.wrap_scalars_to_lists(c)
    c = cm.ensure_dimensions(c, N_src=1)
    cm.Convert(c, framework='list')
    return c


class SyntheticNFMDataset(NFMDataset):
    def __init__(self, configs, telemetry, fitted_vals, PSF_cubes, preprocessor_state, train_idx, val_idx):
        self.PSF_cubes, self.telemetry, self.configs, self.fitted_vals = PSF_cubes, telemetry, configs, fitted_vals
        self.sample_ids  = np.arange(len(configs))
        self.features    = list(telemetry[0].keys())
        self.telemetry_preprocessor = preprocessor_state
        self.Cn2_grid, self.derotated, self.time_groups = None, True, None
        self.train_idx, self.val_idx = train_idx, val_idx
        self.N, self.H, self.W, self.C = PSF_cubes.shape
        self.kc_group, self.kc_fraction = np.zeros(self.N, dtype=int), np.ones(self.N)


# ── Synthetic telemetry, fits and configs ─────────────────────────────────────
seeing = rng.uniform(0.5, 1.2, N)
wind   = rng.uniform(3.0, 12.0, N)
f2, f3 = rng.uniform(0.05, 0.25, N), rng.uniform(0.02, 0.15, N)
tau0   = rng.uniform(2.0, 6.0, N)
phys   = np.stack([seeing, wind, f2, f3, tau0], axis=1)
names  = ['Seeing (header)', 'Wind speed (header)', 'Cn2_frac_binned_2', 'Cn2_frac_binned_3', 'Tau0 (header)']
mean, scale = phys.mean(0), phys.std(0)
D = len(names)
pre_state = {'features': names, 'mean': mean, 'scale': scale, 'initial_fill': np.zeros(D), 'step_feature': np.zeros(0, dtype=np.int64),
             'step_coef': np.zeros((0, D)), 'step_intercept': np.zeros(0), 'min_value': -10 * np.ones(D), 'max_value': 10 * np.ones(D), 'meta': {}}
telemetry = [dict(zip(names, ((phys[i] - mean) / scale).tolist())) for i in range(N)]

fits, configs = [], []
for i in range(N):
    J0 = rng.uniform(10.0, 20.0)
    fits.append({
        'r0':                float(r0_from_seeing(torch.tensor(seeing[i]), 500e-9)) * float(np.exp(rng.normal(0, 0.1))),
        'L0':                25.0 + rng.normal(0, 2.0),
        'dn':                30.0 + rng.normal(0, 2.0),
        'wind_speed_single': wind[i] * (1 + rng.normal(0, 0.05)),
        'Cn2_weights':       np.array([f2[i], f3[i]]) * np.exp(rng.normal(0, 0.1, 2)),
        'J_ctrl':            J0 + np.linspace(2.0, -2.0, 5),
        'F_ctrl':            1.0 + rng.normal(0, 0.01, 5),
        'wing_frac_ctrl':    0.005 + rng.normal(0, 3e-4, 5),
        'J_aniso':           1.1 + abs(rng.normal(0, 0.03)), # the "fits" are off: the truth shared by all stars is 1.4
        'J_angle':           TRUE_ANGLE + rng.normal(0, 3.0), # the angle is fitted right
        'LO_coefs':          np.concatenate([[rng.normal(0, 20.0)], rng.normal(0, 8.0, 7)]), # phase bump first
        'dx_ctrl':           rng.normal(0, 0.05, 5), 'dy_ctrl': rng.normal(0, 0.05, 5),
        'wing_width': 5.0 + rng.normal(0, 0.2), 'wing_width_t': 1.5, 'wing_angle': rng.normal(0, 2.0),
    })
    Cn2 = [1 - f2[i] - f3[i], f2[i], f3[i]]
    configs.append(make_config(seeing[i], wind[i], Cn2, 25.0))

dataset = SyntheticNFMDataset(configs, telemetry, fits, np.zeros((N, FoV, FoV, len(WVLS)), dtype=np.float32), pre_state,
                              train_idx=np.arange(0, 18), val_idx=np.arange(18, N))

cfg = json.load(open(ROOT / 'machine_learning/calibrators/configs/NFM_factor_calibrator_config.json'))
cfg['batch_size'] = 6
cfg['weights_subdir'] = 'synthetic'
cfg['loss'].update(r_max=25.0, noise_r_min=27.0) # the 61 px frames have no background annulus at 85 px
calib_cfg = cfg['calibrator']
calib_cfg['optimized_shared_keys'] = [k for k in calib_cfg['optimized_shared_keys'] if k != 'wing_frac_ctrl']
calib_cfg['fitted_shared_keys']    = ['wing_frac_ctrl'] + calib_cfg['fitted_shared_keys'] # a calibrator output made a fitted shared key
assert 'J_aniso' in calib_cfg['optimized_shared_keys'] and 'J_angle' in calib_cfg['fitted_shared_keys'], \
    'the test scenario needs J_aniso among the optimized shared keys and J_angle among the fitted shared keys of NFM_factor_calibrator_config.json'

# ── Key groups ────────────────────────────────────────────────────────────────
PSF_model, calibrator, transformer, trainer = build_NFM_trainer(cfg, dataset, device, weights_dir=scratch, seed=0)
release_gpu_memory()
P       = trainer.parameterization
manager = PSF_model.inputs_manager
names   = manager.get_names(optimizable_only=False)
groups  = {g: calib_cfg[g] for g in KEY_GROUPS}
outputs = groups['predicted_keys'] + groups['factor_keys'] + groups['optimized_shared_keys']
check('the calibrator outputs (optimizable inputs) are the predicted, factor and optimized shared keys',
      set(transformer.slices) == set(outputs) and all(manager.is_optimizable(k) == (k in outputs) for k in names), f'{list(transformer.slices)}')
n_net = sum(transformer.slices[k].stop - transformer.slices[k].start for k in groups['predicted_keys'] + groups['factor_keys'])
check('the network outputs only the predicted and factor entries', calibrator.network[-1].out_features == P.N_net == n_net == trainer.N_net_outputs
      and set(P.direct_keys) == set(groups['predicted_keys']) and set(P.factor_keys) == set(groups['factor_keys']), f'{P.N_net} of {trainer.N_outputs} stacked')
check('fitted and fitted shared keys passed to the trainer', set(trainer.fitted_keys) == set(groups['fitted_keys'])
      and set(trainer.fitted_shared_keys) == set(groups['fitted_shared_keys']), f'fitted {trainer.fitted_keys}, fitted shared {trainer.fitted_shared_keys}')
left = [k for k in names if not any(k in keys for keys in groups.values())]
check('the model passes the fitted shared keys and, of the inputs left to it, only the required ones',
      set(trainer.model_keys) == set(groups['fitted_shared_keys']) | {k for k in left if k in MODEL_REQUIRED_INPUTS}, f'left to the model {left}')
for bad, override in (('a key in two groups',                       dict(fitted_keys=['dx_ctrl', 'dy_ctrl', 'J_aniso'])),
                      ('a key listed twice in a group',             dict(fitted_keys=['dx_ctrl', 'dx_ctrl'])),
                      ('a factor key without a telemetry baseline', dict(factor_keys=groups['factor_keys'] + ['dn'], predicted_keys=['J_ctrl'])),
                      ('a direct calibrator with optimized shared keys', dict(type='direct'))):
    try:
        calibrator_key_groups({**calib_cfg, **override}, names); check(f'{bad} raises', False)
    except ValueError:
        check(f'{bad} raises', True)

configs_ = {f.name: json.load(open(f, encoding='utf-8')) for f in sorted((ROOT / 'machine_learning/calibrators/configs').glob('*.json'))}
listed   = lambda c: [k for g in KEY_GROUPS for k in c['calibrator'][g]]
check('every training config has the five key groups, without overlap, and no fixed_params / fixed_from_fits / prediction switches',
      all(set(KEY_GROUPS) <= set(c['calibrator']) and len(listed(c)) == len(set(listed(c)))
          and not {'fixed_params', 'fixed_from_fits'} & (set(c) | set(c['calibrator']) | set(c['model']))
          and not {'predict_LO_NCPAs', 'predict_Cn2_profile', 'predict_wind_speed', 'pre_init_astrometry', 'optimize_astrometry'} & set(c['trainer'])
          for c in configs_.values()), f'{list(configs_)}')

# ── Values of the shared and fitted keys ──────────────────────────────────────
with torch.no_grad():
    sv = trainer.optimized_shared_values()
med = lambda k: torch.tensor(np.stack([np.asarray(fits[i][k], dtype=np.float32).reshape(-1) for i in dataset.train_idx])).median(dim=0).values # torch: lower middle
check('optimized shared J_aniso starts at the median of the training fits', abs(sv['J_aniso'].item() - med('J_aniso').item()) < 1e-5, f"J_aniso {sv['J_aniso'].item():.4f}")
check('optimized shared keys are stored normalized', abs(trainer.optimized_params['J_aniso'].item() - (2.0 * sv['J_aniso'].item() - 3.0)) < 1e-5) # Uniform(1, 2)
wing0, angle0 = manager['wing_frac_ctrl'].clone(), manager['J_angle'].clone()
check('fitted shared keys are the medians of the training fits in the PSF model, not optimized',
      torch.allclose(wing0.cpu(), med('wing_frac_ctrl').expand_as(wing0.cpu())) and torch.allclose(manager['wing_width'].cpu(), med('wing_width').expand_as(manager['wing_width'].cpu()))
      and torch.allclose(angle0.cpu(), med('J_angle').expand_as(angle0.cpu()))
      and not any(manager.is_optimizable(k) or k in trainer.optimized_params for k in ('wing_frac_ctrl', 'J_angle')),
      f"wing_frac_ctrl {wing0[0].cpu().numpy().round(5).tolist()}, J_angle {angle0.reshape(-1)[0].item():.2f}")
check('fitted keys are the per-sample fits', torch.allclose(trainer.fitted['dx_ctrl'].cpu(), torch.tensor(np.stack([f['dx_ctrl'] for f in fits]), dtype=torch.float32)))

# ── Render the synthetic data with the truth stretch, recording what the PSF model receives ──
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
captured, _call = [], PSFModelNFM.__call__
PSFModelNFM.__call__ = lambda self, x_dict=None, **kw: (captured.append({k: v.detach().clone() for k, v in x_dict.items()}), _call(self, x_dict, **kw))[1]
with torch.no_grad():
    for _, tel_b, fit_b, config_b, idxs_b in trainer._make_loader(np.arange(N), shuffle=False):
        x = trainer._calibrator_friendly_fit_dict(fit_b)
        x['J_aniso'] = torch.full_like(x['J_aniso'], TRUE_ANISO) # J_angle: the fitted shared value of the PSF model
        PSF = trainer.run_model(x, config_b, idxs_b, trainer.lambda_id_sets[0])
        dataset.PSF_cubes[idxs_b.cpu().numpy()] = PSF.permute(0, 2, 3, 1).cpu().numpy()
        first = first if 'first' in dir() else idxs_b.cpu().numpy()
PSFModelNFM.__call__ = _call
check('synthetic PSFs rendered', np.isfinite(dataset.PSF_cubes).all() and dataset.PSF_cubes.max() > 0, f'peak {dataset.PSF_cubes.max():.3g}')
x0 = captured[0]
check('run_model passes the per-sample fits of the fitted keys', torch.allclose(x0['dx_ctrl'].cpu(), torch.tensor(np.stack([fits[i]['dx_ctrl'] for i in first]), dtype=torch.float32)))
check('run_model passes the fitted shared values, broadcast to the batch', torch.allclose(x0['wing_frac_ctrl'].cpu(), wing0[:1].cpu().expand(len(first), -1))
      and torch.allclose(x0['J_angle'].cpu().reshape(-1), angle0.reshape(-1)[:1].cpu().expand(len(first))))
check('run_model passes the per-sample fitted phase bump', torch.allclose(x0['LO_coefs'][:, 0].cpu(), torch.tensor([fits[i]['LO_coefs'][0] for i in first], dtype=torch.float32)))
check('run_model does not pass the inputs left to the model (but the required ones)', not {k for k in left if k not in MODEL_REQUIRED_INPUTS} & set(x0)
      and set(outputs) <= set(x0), f'passed {sorted(x0)}')

# ── Pretraining in the network layout (predicted and factor keys) ─────────────
fit_all, tel_all = trainer._train_fit_dict()
targets = trainer._pretrain_target_vector(fit_all, tel=tel_all)
check('pretraining targets have the network layout', tuple(targets.shape) == (len(dataset.train_idx), P.N_net))
with torch.no_grad():
    x_fit = trainer._params_from_pred(targets, tel_all)
err = max((x_fit[k].reshape(len(tel_all), -1) - trainer._calibrator_friendly_fit_dict(fit_all)[k].reshape(len(tel_all), -1)).abs().max().item()
          for k in P.direct_keys + [k for k in P.factor_keys if k != 'Cn2_weights']) # the Cn2 targets are clamped at 1e-3
check('the targets reproduce the fits of the predicted and factor keys', err < 1e-3, f'max |diff| {err:.2e}')
trainer.pretrain(num_epochs=40, patience=100, save_path=scratch / 'pretrain.pth')

# ── End-to-end training: the optimized shared J_aniso must move towards the truth ──
initial = {k: v.detach().clone() for k, v in trainer.optimized_params.items()}
for g in trainer.optimizer.param_groups:
    if any(p is trainer.optimized_params['J_aniso'] for p in g['params']):
        g['lr'] = 3e-2 # faster than the default lr / 10 so that a short run shows the trend
with torch.no_grad():
    a0 = trainer.optimized_shared_values()['J_aniso'].item()
train_losses, val_losses = trainer.train(num_epochs=25, patience=100, nan_recovery=False, checkpoint_path=scratch / 'ckpt.pth')
with torch.no_grad():
    sv = trainer.optimized_shared_values()
a1 = sv['J_aniso'].item()
check('training loss decreased', np.mean(train_losses[-5:]) < np.mean(train_losses[:5]), f'{np.mean(train_losses[:5]):.3f} -> {np.mean(train_losses[-5:]):.3f}')
# The shared jitter stretch alone: the trained network and the other shared keys fixed, J_aniso learned vs at the fit median,
# on the training stars (the end-to-end network overfits 18 stars, the 6 validation stars are too noisy to resolve this)
learned = {'J_aniso': trainer.optimized_params['J_aniso'].detach().clone()}
train_eval = trainer._make_loader(dataset.train_idx, shuffle=False)
loss_learned = trainer.validate(loader=train_eval)
for k in learned: trainer.optimized_params[k].data = initial[k].clone()
loss_initial = trainer.validate(loader=train_eval)
for k in learned: trainer.optimized_params[k].data = learned[k]
check('the learned J_aniso lowers the training loss', loss_learned < loss_initial, f'{loss_initial:.4f} (fit median) -> {loss_learned:.4f} (learned)')
check('J_aniso moved towards the truth', abs(a1 - TRUE_ANISO) < abs(a0 - TRUE_ANISO), f'{a0:.3f} -> {a1:.3f} (truth {TRUE_ANISO})')
check('the fitted shared (J_angle included) and fitted keys did not move', torch.equal(manager['wing_frac_ctrl'], wing0) and torch.equal(manager['J_angle'], angle0)
      and torch.allclose(trainer.fitted['dx_ctrl'].cpu(), torch.tensor(np.stack([f['dx_ctrl'] for f in fits]), dtype=torch.float32)))
_, x_dict = trainer._predict(tel_all)
check('the optimized shared keys are the same for every sample', all((x_dict[k].reshape(len(tel_all), -1).std(0).max().item() == 0) for k in P.optimized_shared_keys))

# ── Checkpoint round trip ─────────────────────────────────────────────────────
saved = {k: v.detach().clone() for k, v in trainer.optimized_params.items()}
for p in trainer.optimized_params.values():
    p.data.zero_()
trainer.load_checkpoint(scratch / 'ckpt.pth', load_optimizer=False)
err = max((trainer.optimized_params[k] - saved[k]).abs().max().item() for k in saved)
check('checkpoint restores the optimized shared keys', err < 1e-5, f'max |diff| {err:.2e}')

# ── Validation cubes, tuning of non-predicted keys ───────────────────────────
out = trainer.validate(return_cubes=True)
check('validate(return_cubes) gives the network outputs (predicted and factor keys)', tuple(out[3].shape) == (len(dataset.val_idx), P.N_net))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    trainer.tune_calibrator(['F_ctrl', 'dn'], num_epochs=1, lr=1e-4)
check('tune_calibrator skips the optimized shared keys with a warning', any('F_ctrl' in str(x.message) for x in w))

# ── Bundle: inference reproduces the trainer, the shared values go into the PSF model ──
bundle_path = scratch / 'synthetic_bundle.pth'
trainer.save_calibrator(bundle_path)
calib = load_NFM_calibrator(bundle_path, device=device)
check('bundle loads as NFMFactorCalibrator', isinstance(calib, NFMFactorCalibrator) and calib.parameterization.N_net == P.N_net)
trainer.calibrator.eval()
with torch.no_grad():
    x_tr = trainer._params_from_pred(trainer.calibrator(tel_all), tel_all)
    x_in = calib.forward_prepared(tel_all)
err = max(((x_in[k][:, 1:] if k == 'LO_coefs' else x_in[k]) - x_tr[k]).abs().max().item() for k in x_tr)
check('NFMFactorCalibrator.forward_prepared == trainer parameters', err < 1e-4, f'max |diff| {err:.2e}')
check('the bundle carries the fitted shared values (J_angle, wing shape)', set(calib.fitted_shared) == set(trainer.fitted_shared_keys)
      and torch.allclose(calib.fitted_shared['wing_frac_ctrl'].cpu().reshape(-1), wing0[0].cpu())
      and torch.allclose(calib.fitted_shared['J_angle'].cpu().reshape(-1), angle0.reshape(-1)[:1].cpu()), f'fitted_shared {list(calib.fitted_shared)}')

batch = [dataset[i] for i in range(4)]
_, tel_b, _, config_b, _ = dataset.collate_batch(batch, device=device)
with torch.no_grad():
    model = build_PSF_model(config_b, cfg['model'], device)
calib.calibrate_prepared(tel_b, model)
got = model.inputs_manager['J_aniso'].reshape(-1)
check('calibrate_prepared sets the learned J_aniso in the PSF model', torch.allclose(got, torch.full_like(got, a1), atol=1e-4), f'{got.tolist()}')
got = model.inputs_manager['J_angle'].reshape(-1)
check('calibrate_prepared sets the fitted shared values (J_angle, wing) and leaves the fitted keys (unknown for new data) at the model values',
      torch.allclose(model.inputs_manager['wing_frac_ctrl'], wing0[:1].expand_as(model.inputs_manager['wing_frac_ctrl']))
      and torch.allclose(got.cpu(), angle0.reshape(-1)[:1].cpu().expand_as(got.cpu())) and bool((model.inputs_manager['dx_ctrl'] == 0).all()),
      f'J_angle {got.tolist()}')

print(f'\n{sum(results)} / {len(results)} checks passed')
del trainer, calib, model
release_gpu_memory()
_tmp.cleanup()
sys.exit(0 if all(results) else 1)
