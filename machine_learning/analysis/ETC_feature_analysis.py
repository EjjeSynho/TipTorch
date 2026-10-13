"""
Which inputs of the PSF-R calibrator matter, and how well the ETC can replace them (ExecPlan `agents/execplans/ETC_calibrator.md`).

1. Predictability of every non-ETC calibrator input from the six ETC inputs (grouped 5-fold CV R2 of a random forest).
2. Gradient importance of the PSF-R calibrator (`NFMCalibrator`) per input feature and per feature group.
3. Ablation: feature groups replaced by the calibrator's trained imputer, effect on the FWHM / Strehl / EE50 of the validation
   set measured against the on-sky metrics of `validate_ETC_calibrator.py` (`params_onsky.csv` must exist).

Outputs: `data/Enrico_ETC/ETC_calibrator/features_report.md` (+ CSVs) and `data/Enrico_ETC/plots/ETC_calibrator/features_*.png`.
Requirements: the PSF-R calibrator bundle (`NFM_calibrator_config.json` metadata), the ETC bundle, the dataset cache, a CUDA device.

    python machine_learning/analysis/ETC_feature_analysis.py [--no-psfao] [--processes 40]
"""

import os
import sys
import json
import argparse
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / 'machine_learning', REPO_ROOT / 'development'):
    sys.path.insert(0, str(p))

from tiptorch._config import WEIGHTS_FOLDER, default_device
from calibrators.NFM_calibrator import NFMDataset, NFMCalibrator
from calibrators.ETC_calibrator import ETC_INPUTS, PUPIL_PATH
from calibrators import ETC_validation_utils as V

FEATURE_GROUPS = {
    'ETC inputs':     ETC_INPUTS,
    'Site / profile': ['Wind dir (header)', 'Wind speed (header)', 'Free Atmosphere Seeing ["]', 'MASS Tau0 [s]', 'MASS-DIMM Turb Velocity [m/s]', 'MASS_FRACGL',
                       'IA_FWHMLINOBS', 'LGS_TUR_ALT', 'LGS photons, [photons/m^2/s]', 'MASS_TURB total', 'Cn2 fraction below 2000m',
                       'Cn2_alt_binned_2', 'Cn2_frac_binned_2', 'Cn2_alt_binned_3', 'Cn2_frac_binned_3'],
    'AO diagnostics': ['LGS_STREHL', 'LGS_TURVAR_RES', 'LGS_FWHM_GAIN', 'LGS_SLOPERMS', 'theta0'],
    'Nuisance':       ['Exp. time', 'Tel. azimuth', 'Par. angle', 'Derot. angle', 'Temperature (header)'],
}
GROUP_COLORS = {'ETC inputs': '#1f8a4c', 'Site / profile': '#2a78d6', 'AO diagnostics': '#eb6834', 'Nuisance': '#6b6a66'}
ABLATIONS = {'real telemetry': [], 'impute AO diagnostics': ['AO diagnostics'], 'impute nuisance': ['Nuisance'], 'impute site / profile': ['Site / profile'],
             'impute AO + nuisance': ['AO diagnostics', 'Nuisance'], 'impute all non-ETC': ['AO diagnostics', 'Nuisance', 'Site / profile']}


def psfr_calibrator():
    with open(REPO_ROOT / 'machine_learning' / 'calibrators' / 'configs' / 'NFM_calibrator_config.json') as f:
        cfg = json.load(f)
    with open(WEIGHTS_FOLDER / cfg['weights_subdir'] / f'{cfg["name"]}_meta.json') as f:
        return NFMCalibrator(json.load(f)['bundle_path'], device=default_device)


def predictability(Z, features, groups_cv):
    X = Z[:, [features.index(f) for f in ETC_INPUTS]]
    rows = []
    for g, fs in FEATURE_GROUPS.items():
        for f in fs:
            if g != 'ETC inputs' and f in features:
                r2 = cross_val_score(RandomForestRegressor(100, min_samples_leaf=5, random_state=0, n_jobs=8), X, Z[:, features.index(f)], cv=GroupKFold(5), groups=groups_cv, scoring='r2').mean()
                rows.append({'group': g, 'feature': f, 'R2': r2})
    return pd.DataFrame(rows).sort_values('R2', ascending=False)


def gradient_importance(cal, tel_scaled):
    """ mean |d output_j / d input_i| over the validation telemetry -> DataFrame [outputs x features] """
    tel = torch.as_tensor(tel_scaled, device=cal.device).requires_grad_(True)
    out = cal.net(tel)
    J = np.stack([torch.autograd.grad(out[:, j].sum(), tel, retain_graph=True)[0].abs().mean(0).cpu().numpy() for j in range(out.shape[1])])
    names = [name if sl.stop - sl.start == 1 else f'{name}[{k}]' for name, sl in cal.outputs_transformer.slices.items() for k in range(sl.stop - sl.start)]
    return pd.DataFrame(J, index=names, columns=cal.features)


@torch.no_grad()
def render_psfr(cal, renderer, dataset, etc, tel_scaled, phase_bump, batch_size=16):
    """ PSF-R calibrator on standardized telemetry with the real configs of the validation observations """
    PSFs = []
    for start in range(0, len(etc), batch_size):
        ids = list(range(start, min(start + batch_size, len(etc))))
        x = cal.forward_prepared(torch.as_tensor(tel_scaled[ids], device=cal.device))
        x['LO_coefs'] = torch.cat([torch.full((len(ids), 1), phase_bump, device=cal.device), x['LO_coefs'][:, 1:]], 1)
        configs = []
        for i in etc['dataset_idx'].to_numpy()[ids]:
            c = deepcopy(dataset.configs[i])
            c['sources_science']['Wavelength'] = [w * 1e-9 for w in renderer.wavelengths_nm]
            c['telescope']['PathPupil'], c['telescope']['PupilAngle'] = str(PUPIL_PATH), 0.0
            configs.append(c)
        model = renderer.make_model(configs)
        model.update_manager_params(x)
        PSFs.append(model().detach().cpu())
        model.cleanup()
    return torch.cat(PSFs, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-psfao', action='store_true')
    parser.add_argument('--processes', type=int, default=min(40, os.cpu_count() or 1))
    args = parser.parse_args()

    cal, etc_cal = psfr_calibrator(), V.load_bundle()
    dataset = NFMDataset(V.DATASET_CACHE)
    features = list(cal.features)
    etc = V.load_validation(etc_cal, dataset)
    tel_val = np.array([list(dataset.telemetry[i].values()) for i in etc['dataset_idx']], dtype=np.float32)
    Z_all = np.array([list(dataset.telemetry[i].values()) for i in range(len(dataset))], dtype=np.float64)
    feat_group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    onsky = pd.read_csv(V.OUT_DIR / 'params_onsky.csv').set_index(['sample', 'wave'])
    lines = ['# PSF-R calibrator inputs in the ETC regime', '', '| group | features |', '|---|---|'] + [f'| {g} | {", ".join(fs)} |' for g, fs in FEATURE_GROUPS.items()]

    # 1. predictability from the ETC inputs
    pred = predictability(Z_all, features, np.asarray(dataset.time_groups))
    pred.to_csv(V.OUT_DIR / 'feature_predictability.csv', index=False)
    lines += ['', '## Predictability of the non-ETC inputs from the six ETC inputs (grouped 5-fold CV R2, random forest)', '', pred.round(2).to_markdown(index=False)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(pred)), pred['R2'].clip(lower=-0.1), color=[GROUP_COLORS[g] for g in pred['group']])
    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(pred['feature'], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('CV R$^2$')
    ax.set_title('Predictability of the non-ETC calibrator inputs from the ETC inputs')
    for g, c in GROUP_COLORS.items():
        ax.bar([0], [0], color=c, label=g)
    ax.legend(frameon=False)
    V._style(ax)
    plt.tight_layout()
    fig.savefig(V.PLOTS_DIR / 'features_predictability.png', dpi=120)
    plt.close(fig)

    # 2. gradient importance of the PSF-R calibrator
    imp = gradient_importance(cal, tel_val)
    imp.to_csv(V.OUT_DIR / 'feature_gradient_importance.csv')
    out_groups = [n.split('[')[0] for n in imp.index]
    share = imp.T.groupby(feat_group).sum().T.groupby(out_groups).mean()
    share = share.div(share.sum(1), axis=0)
    glob = imp.sum(0).sort_values(ascending=False)
    lines += ['', '## Share of the calibrator gradient importance per feature group (rows: predicted parameters)', '', share.round(2).to_markdown(),
              '', '## Global gradient importance per input feature', '', pd.DataFrame({'feature': glob.index, 'group': [feat_group[f] for f in glob.index], 'importance': glob.values}).round(3).to_markdown(index=False)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [2.2, 1]})
    axes[0].bar(range(len(glob)), glob.values, color=[GROUP_COLORS[feat_group[f]] for f in glob.index])
    axes[0].set_xticks(range(len(glob)))
    axes[0].set_xticklabels(glob.index, rotation=60, ha='right', fontsize=8)
    axes[0].set_title('PSF-R calibrator: gradient importance of the input features (validation set)')
    im = axes[1].imshow(share.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(share.shape[1]))
    axes[1].set_xticklabels(share.columns, rotation=30, ha='right')
    axes[1].set_yticks(range(share.shape[0]))
    axes[1].set_yticklabels(share.index)
    for r in range(share.shape[0]):
        for c in range(share.shape[1]):
            axes[1].text(c, r, f'{share.values[r, c]:.2f}', ha='center', va='center', fontsize=8, color='white' if share.values[r, c] > 0.6 else '#1a1a19')
    axes[1].set_title('Importance share per feature group')
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    plt.tight_layout()
    fig.savefig(V.PLOTS_DIR / 'features_importance.png', dpi=120)
    plt.close(fig)

    # 3. ablation of the feature groups, measured against the on-sky metrics
    bins, waves_nm = V.nearest_bins(np.array(dataset.configs[0]['sources_science']['Wavelength']) * 1e9)
    renderer, pupil = etc_cal.renderer(waves_nm), V.load_pupil()
    rows = []
    with mp.get_context('spawn').Pool(args.processes) as pool:
        for name, gs in ABLATIONS.items():
            Z = tel_val.astype(np.float64).copy()
            if gs:
                Z[:, [features.index(f) for g in gs for f in FEATURE_GROUPS[g]]] = np.nan
                Z = cal.telemetry_preprocessor.impute(Z)
            PSFs = render_psfr(cal, renderer, dataset, etc, Z.astype(np.float32), etc_cal.phase_bump)
            t = V.measure_PSFs(PSFs.numpy(), waves_nm, pupil, pool, not args.no_psfao).set_index(['sample', 'wave'])
            for wave in V.NOMINAL_WAVES:
                for p in ('fwhm', 'strehl_tt', 'ensquared_tt'):
                    a, b = t.xs(wave, level='wave')[p], onsky.xs(wave, level='wave')[p]
                    ok = np.isfinite(a) & np.isfinite(b)
                    rows.append({'ablation': name, 'wave': wave, 'param': p, 'MAE': (a - b)[ok].abs().mean(), 'bias': (a - b)[ok].mean(), 'rho': spearmanr(a[ok], b[ok]).statistic,
                                 'std_ratio': a[ok].std() / b[ok].std(), 'W1': wasserstein_distance(a[ok], b[ok])})
    abl = pd.DataFrame(rows)
    abl.to_csv(V.OUT_DIR / 'feature_ablation.csv', index=False)
    lines += ['', '## Ablation at 700 nm: feature groups replaced by the trained imputer (paired against the on-sky metrics)', '',
              'MAE / bias in the units of the parameter, rho = Spearman correlation with on-sky, std ratio = predicted / on-sky spread, W1 = distribution distance.', '',
              abl[abl.wave == 700].drop(columns='wave').round(3).to_markdown(index=False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, p in zip(axes, ('fwhm', 'strehl_tt', 'ensquared_tt')):
        s = abl[(abl.param == p) & (abl.wave == 700)].set_index('ablation')
        x = np.arange(len(s))
        ax.bar(x - 0.2, s['MAE'], width=0.4, color='#eb6834', label='paired MAE')
        ax.bar(x + 0.2, s['W1'], width=0.4, color='#2a78d6', label='W1 to on-sky')
        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=35, ha='right', fontsize=8)
        ax.set_title(V.PARAMS[p][0] + ' at 700 nm')
        V._style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle('PSF-R calibrator with feature groups replaced by the trained imputer')
    plt.tight_layout()
    fig.savefig(V.PLOTS_DIR / 'features_ablation.png', dpi=120)
    plt.close(fig)
    (V.OUT_DIR / 'features_report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-10:]))
