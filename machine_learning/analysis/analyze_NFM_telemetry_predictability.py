#%%
"""
Which fitted PSF-model parameters can be predicted from the telemetry?

For every output of the NFM calibrator (fitted values of the STD-star dataset): the cross-validated R^2 of a ridge and of a
random-forest regressor on the 31 telemetry features (folds grouped by acquisition time, as the training split). For the
parameters with a telemetry-derived base value (r0 from the header seeing, the header wind speed, the binned Cn2 fractions,
see telemetry_base_params): the variance explained by the base value alone and the predictability of the residual log-ratio,
i.e. what is left for the factor calibrator to learn. Finally, the PSF-level cost of not predicting a parameter group at all:
the per-sample fits with that group replaced by the training-set median, on the validation set (GPU, ~1 min per group).

Outputs: a table (CSV + stdout) and a figure in data/analysis/NFM_calibrator/.
Run from machine_learning/ in the TipTorch conda env, e.g. python analyze_NFM_telemetry_predictability.py [--device cuda:1] [--skip-psf]
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# machine_learning/ on sys.path (the `calibrators` package), from a script or from IPython cells
_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd().resolve()
sys.path.insert(0, str(next((p for p in (_here, *_here.parents) if (p / 'calibrators').is_dir()), _here / 'machine_learning')))
from tiptorch._config import *
from calibrators.NFM_calibrator import NFMDataset, kfold_splits, telemetry_base_params, build_NFM_trainer, release_gpu_memory, load_STD_dataset, CONFIGS_FOLDER
from calibrators.NFM_validation_utils import predict_cubes, core_halo_metrics, metrics_table
from data_processing.telemetry_preprocessor import TelemetryPreprocessor
from data_processing.Cn2_profile import Cn2Grid

_parser = argparse.ArgumentParser(description="Telemetry predictability of the fitted NFM PSF parameters")
_parser.add_argument('--config',   type=str, default=None, help='training config (default: calibrators/configs/NFM_calibrator_config.json)')
_parser.add_argument('--device',   type=str, default=None)
_parser.add_argument('--skip-psf', action='store_true', help='skip the PSF-level cost of the median replacement (no GPU needed)')
try:
    _args = _parser.parse_args()
except SystemExit:
    _args = _parser.parse_args([])

with open(_args.config or CONFIGS_FOLDER / 'NFM_calibrator_config.json') as f:
    cfg = json.load(f)

device     = torch.device(_args.device) if _args.device else default_device
STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])
OUT_FOLDER = DATA_FOLDER / 'analysis' / 'NFM_calibrator'
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

#%%
dataset  = load_STD_dataset(cfg['derotated_PSFs'])
features = dataset.features
N        = len(dataset)

X_tel  = np.asarray([list(t.values()) for t in dataset.telemetry], dtype=np.float64) # standardized and imputed, [N, 31]
X_phys = TelemetryPreprocessor.from_state_dict(dataset.telemetry_preprocessor).inverse_transform(X_tel)

# Targets: one column per calibrator output, GL layer of the Cn2 profile reconstructed as in the pretraining
fit = {k: np.asarray([np.atleast_1d(d[k]) for d in dataset.fitted_vals], dtype=np.float64) for k in dataset.fitted_vals[0]}
Cn2 = np.clip(fit['Cn2_weights'], 1e-3, None)
fit['Cn2_weights'] = np.hstack([np.clip(1.0 - Cn2.sum(-1, keepdims=True), 1e-3, None), Cn2])

targets = {}
for key in ['r0', 'L0', 'dn', 'wind_speed_single', 'Cn2_weights', 'J_ctrl', 'F_ctrl', 'LO_coefs']:
    vals = fit[key]
    for j in range(vals.shape[1]):
        name = key if vals.shape[1] == 1 else f'{key}[{j}]'
        targets[name] = vals[:, j]

# Folds grouped by acquisition time over the whole dataset (the training split is grouped the same way)
folds = kfold_splits(dataset, np.arange(N), n_splits=5, random_state=cfg['trainer']['random_state'])
print(f'{N} samples, {len(features)} features, {len(targets)} targets, {len(folds)} time-grouped folds')

#%%
# Cross-validated R^2 of a linear and of a non-linear regressor per target. Targets are standardized, so R^2 compares across
# parameters; the random forest is the upper bound of what the small calibrator network could extract.
def CV_R2(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=folds)
    return 1.0 - np.mean((y - y_pred)**2) / np.var(y)

ridge  = lambda: Ridge(alpha=10.0)
forest = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)

rows = []
for name, y in targets.items():
    y_std = (y - y.mean()) / y.std()
    rows.append(
    {
        'target': name,
        'R2_ridge': CV_R2(ridge(), X_tel, y_std),
        'R2_forest': CV_R2(forest(), X_tel, y_std),
        'median': np.median(y), 'std': y.std()
    })
    
    print(f"{name:20s} R2 ridge {rows[-1]['R2_ridge']:+.3f}  forest {rows[-1]['R2_forest']:+.3f}", flush=True)

table = pd.DataFrame(rows).set_index('target')

#%%
# Factor parameters: how much of the variance the telemetry-derived base value explains by itself (R^2 of using it directly,
# in log space), and how predictable the remaining log-ratio is. A base that explains most of the variance with an
# unpredictable residual is best used with a shrinkage prior (factor = 1); a predictable residual is what the factor head learns.
_Cn2_grid = Cn2Grid.from_state_dict(dataset.Cn2_grid) if getattr(dataset, 'Cn2_grid', None) else None # grid features of the caches since 2026-10-13
base = {k: v.numpy().reshape(N, -1) for k, v in telemetry_base_params(X_phys, features, L0_const=np.median(fit['L0']), Cn2_grid=_Cn2_grid).items()}

for key in ['r0', 'L0', 'wind_speed_single', 'Cn2_weights']:
    for j in range(base[key].shape[1]):
        name  = key if base[key].shape[1] == 1 else f'{key}[{j}]'
        y     = np.log(np.clip(fit[key][:, j], 1e-4, None))
        b     = np.log(np.clip(base[key][:, j], 1e-4, None))
        ratio = y - b
        table.loc[name, 'corr_base']      = np.corrcoef(y, b)[0, 1]
        table.loc[name, 'R2_base_direct'] = 1.0 - np.var(ratio) / np.var(y)         # fitted ~ base, no learning at all
        table.loc[name, 'std_log_ratio']  = ratio.std()                              # width of the shrinkage prior
        table.loc[name, 'R2_ratio_forest'] = CV_R2(forest(), X_tel, (ratio - ratio.mean()) / ratio.std())
        print(f"{name:20s} corr(fit, base) {table.loc[name, 'corr_base']:+.3f}  R2 base alone {table.loc[name, 'R2_base_direct']:+.3f}  "
              f"log-ratio std {ratio.std():.3f}  R2 of the ratio (forest) {table.loc[name, 'R2_ratio_forest']:+.3f}", flush=True)

#%%
# PSF-level cost of a parameter group: the validation-set metrics of the per-sample fits with the group replaced by the
# training-set median (in the calibrator output space, so the Cn2 median is a median of the softmax logits)
psf_rows = {}
if not _args.skip_psf:
    PSF_model, calibrator, outputs_transformer, trainer = build_NFM_trainer(cfg, dataset, device, weights_dir=OUT_FOLDER / 'tmp')
    lambda_nm = trainer.lambda_full.cpu().numpy().ravel() * 1e9

    with torch.no_grad():
        Y_train = torch.cat([trainer._pretrain_target_vector(fit_b, device) for _, _, fit_b, _, _ in trainer.train_loader])
        Y_med   = Y_train.median(dim=0).values

    def fits_with_median(group):
        def predict(tel_b, fit_b, idxs_b):
            y = trainer._pretrain_target_vector(fit_b, device)
            for key in ([group] if group else list(outputs_transformer.slices)):
                y[:, outputs_transformer.slices[key]] = Y_med[outputs_transformer.slices[key]]
            return outputs_transformer.unstack(y)
        return predict

    runs = {'fits (oracle)': lambda tel_b, fit_b, idxs_b: outputs_transformer.unstack(trainer._pretrain_target_vector(fit_b, device))}
    runs.update({f'median {key}': fits_with_median(key) for key in outputs_transformer.slices})
    runs['median all'] = fits_with_median(None)

    for name, predict in runs.items():
        D, P, _ = predict_cubes(trainer, predict)
        psf_rows[name] = core_halo_metrics(D, P, trainer, lambda_nm)
        print(f"{name:24s} balanced loss {psf_rows[name].get('balanced_loss_mean', float('nan')):.4f}  "
              f"|peak err| {psf_rows[name]['peak_err_pct_median']:.1f} %  EE2 err {psf_rows[name]['EE2_err_pct_median']:.1f} %", flush=True)
        release_gpu_memory()

    print('\n' + metrics_table(psf_rows))
    for name, m in psf_rows.items():
        key = name.replace('median ', '')
        if key in outputs_transformer.slices:
            cols = [t for t in table.index if t == key or t.startswith(key + '[')]
            table.loc[cols, 'psf_cost_balanced_loss'] = m['balanced_loss_mean'] - psf_rows['fits (oracle)']['balanced_loss_mean']
            table.loc[cols, 'psf_cost_peak_err_pct']  = m['peak_err_pct_median'] - psf_rows['fits (oracle)']['peak_err_pct_median']

    with open(OUT_FOLDER / 'median_replacement_metrics.json', 'w') as f:
        json.dump(psf_rows, f, indent=1)

#%%
pd.set_option('display.width', 200); pd.set_option('display.float_format', '{:.3f}'.format)
print(table)
table.to_csv(OUT_FOLDER / 'telemetry_predictability.csv')

fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(len(table))
ax.bar(x - 0.2, table['R2_ridge'],  width=0.4, label='ridge (linear)')
ax.bar(x + 0.2, table['R2_forest'], width=0.4, label='random forest')

if 'R2_base_direct' in table:
    ax.scatter(x, table['R2_base_direct'], color='k', marker='_', s=300, label='telemetry base value alone', zorder=3)
    
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(table.index, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('cross-validated $R^2$'); ax.set_ylim(-0.3, 1); ax.grid(alpha=0.3, axis='y'); ax.legend()
ax.set_title('Predictability of the fitted PSF parameters from the telemetry (5 time-grouped folds)')
plt.tight_layout()
plt.savefig(OUT_FOLDER / 'telemetry_predictability.png', dpi=150)

print(f'Saved -> {OUT_FOLDER}')
