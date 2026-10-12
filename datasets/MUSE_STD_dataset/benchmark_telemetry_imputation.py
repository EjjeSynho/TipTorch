#%%
"""
Benchmark of the imputation methods for the reduced MUSE NFM STD stars telemetry.

Missing telemetry in this dataset comes in blocks (e.g., the whole MASS-DIMM or Cn2 group is absent at once),
so the benchmark masks the observed entries with real missingness patterns borrowed from other samples
instead of masking random cells. Imputers are fitted on the training folds and applied to the held-out fold,
which mimics how the saved imputer is used at inference time by the calibrator.
The error is computed on the masked entries in standardized units (z-scores).
"""
try:
    ipy = get_ipython() # NameError if not running under IPython
    if ipy:
        ipy.run_line_magic('reload_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')
except NameError:
    pass

import pickle
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning

from tiptorch._config import project_settings
from data_processing.MUSE_data_utils import filter_dataframe, reduce_dataframe

warnings.filterwarnings('ignore', category=ConvergenceWarning)

STD_FOLDER = Path(project_settings["MUSE_STD_data_folder"])

include_slow = False # ExtraTrees-based iterative imputer, ~50x slower than the rest


class LowRankImputer:
    """
    Imputes missing values assuming the data lie close to a low-rank linear subspace (probabilistic PCA).

    fit():       learns the mean and the top-rank principal components with iterative SVD (hard-impute),
                 the variance of the discarded components is treated as isotropic noise
    transform(): for every sample, computes the posterior mean of its latent coordinates from the observed
                 entries only, and fills the missing entries with the reconstruction. Observed entries are kept.
    """
    def __init__(self, rank=10, max_iter=500, tol=1e-6):
        self.rank, self.max_iter, self.tol = rank, max_iter, tol

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        miss = np.isnan(X)
        r = min(self.rank, min(X.shape) - 1)

        X_filled = np.where(miss, np.nanmean(X, axis=0), X)
        for _ in range(self.max_iter):
            mean = X_filled.mean(axis=0)
            U, S, Vt = np.linalg.svd(X_filled - mean, full_matrices=False)
            recon = mean + (U[:, :r] * S[:r]) @ Vt[:r]
            delta = np.sum((recon[miss] - X_filled[miss])**2) / max(np.sum(X_filled[miss]**2), 1e-12)
            X_filled[miss] = recon[miss]
            if delta < self.tol:
                break

        N, P = X.shape
        eigvals = S**2 / N
        self.mean_       = mean
        self.components_ = Vt[:r]                                            # [r x P]
        self.noise_var_  = max(eigvals[r:].sum() / max(P - r, 1), 1e-12)     # isotropic noise variance
        self.latent_var_ = np.maximum(eigvals[:r] - self.noise_var_, 1e-12)  # prior variance of latent coordinates
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        miss = np.isnan(X)
        prior_prec = self.noise_var_ / self.latent_var_

        for i in np.where(miss.any(axis=1))[0]:
            obs = ~miss[i]
            V_o = self.components_[:, obs].T  # [N_obs x r]
            z = np.linalg.solve(V_o.T @ V_o + np.diag(prior_prec), V_o.T @ (X[i, obs] - self.mean_[obs]))
            X[i, ~obs] = self.mean_[~obs] + z @ self.components_[:, ~obs]
        return X


class OutliersToMedian:
    """ The previous pipeline: imputed values beyond ±threshold (z-score) are replaced by the per-feature median """
    def __init__(self, imputer, threshold=3):
        self.imputer, self.threshold = imputer, threshold

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.median_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        miss = np.isnan(X)
        X_imp = self.imputer.transform(X)
        outliers = miss & (np.abs(X_imp) > self.threshold)
        X_imp[outliers] = np.broadcast_to(self.median_, X_imp.shape)[outliers]
        return X_imp

with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
    muse_df = pickle.load(handle)

telemetry_df = reduce_dataframe(filter_dataframe(muse_df.copy()))
telemetry_df = telemetry_df[telemetry_df.select_dtypes(include="number").columns]
X = telemetry_df.values.astype(float)
N, P = X.shape

print(f'{N} samples x {P} features, {100*np.isnan(X).mean():.1f}% missing values')

#%%
def make_imputers(X_range):
    lo, hi = X_range
    imputers = {
        'Median':                     lambda: SimpleImputer(strategy='median'),
        'Iterative':                  lambda: IterativeImputer(max_iter=200, random_state=3),
        'Iterative + outliers->median (old)': lambda: OutliersToMedian(IterativeImputer(max_iter=200, random_state=3)),
        'Iterative, median init':     lambda: IterativeImputer(max_iter=200, initial_strategy='median', random_state=3),
        'Iterative, clipped':         lambda: IterativeImputer(max_iter=200, min_value=lo, max_value=hi, random_state=3),
        'Iterative, median init, clipped': lambda: IterativeImputer(max_iter=200, initial_strategy='median', min_value=lo, max_value=hi, random_state=3),
        'Iterative, 10 nearest, clipped':  lambda: IterativeImputer(max_iter=200, n_nearest_features=10, min_value=lo, max_value=hi, random_state=3),
        'KNN, k=5':                   lambda: KNNImputer(n_neighbors=5,  weights='distance'),
        'KNN, k=10':                  lambda: KNNImputer(n_neighbors=10, weights='distance'),
        'KNN, k=20':                  lambda: KNNImputer(n_neighbors=20, weights='distance'),
        'Low-rank, r=5':              lambda: LowRankImputer(rank=5),
        'Low-rank, r=10':             lambda: LowRankImputer(rank=10),
        'Low-rank, r=15':             lambda: LowRankImputer(rank=15),
    }
    if include_slow:
        imputers['Iterative, ExtraTrees'] = lambda: IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=3),
            max_iter=10, initial_strategy='median', random_state=3)
    return imputers


def mask_with_real_patterns(X, rng):
    """ For every sample, hide its observed entries following the NaN pattern of a randomly picked incomplete sample """
    nan_mask = np.isnan(X)
    incomplete = np.where(nan_mask.any(axis=1))[0]
    X_masked, hidden = X.copy(), np.zeros_like(nan_mask)

    for i in range(len(X)):
        pattern = nan_mask[rng.choice(incomplete)] & ~nan_mask[i]
        if pattern.all() or (~nan_mask[i] & ~pattern).sum() == 0:
            continue  # keep at least one observed value per row
        hidden[i] = pattern
    X_masked[hidden] = np.nan
    return X_masked, hidden


def run_benchmark(X, n_repeats=3, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    records = []

    for rep in range(n_repeats):
        X_masked, hidden = mask_with_real_patterns(X, rng)
        folds = KFold(n_folds, shuffle=True, random_state=seed + rep).split(X)

        for fold, (train_rows, test_rows) in enumerate(folds):
            # Imputers are fitted on the training rows with their own native NaNs, test rows additionally masked
            scaler  = StandardScaler().fit(X[train_rows])
            X_train = scaler.transform(X[train_rows])
            X_test  = scaler.transform(X_masked[test_rows])
            X_true  = scaler.transform(X[test_rows])
            hidden_test = hidden[test_rows]

            X_range = (np.nanmin(X_train, axis=0), np.nanmax(X_train, axis=0))

            for name, make in make_imputers(X_range).items():
                t0 = time.time()
                X_imp = make().fit(X_train).transform(X_test)
                dt = time.time() - t0

                err = (X_imp - X_true)[hidden_test]
                cols = np.where(hidden_test)[1]
                records.append({
                    'method': name, 'repeat': rep, 'fold': fold, 'time': dt,
                    'RMSE': np.sqrt(np.mean(err**2)),
                    'MAE':  np.mean(np.abs(err)),
                    'max |z| imputed': np.abs(X_imp[hidden_test]).max(),
                    'per_feature_SE': pd.Series(err**2).groupby(cols).sum().reindex(range(X.shape[1]), fill_value=0).values,
                    'per_feature_N':  np.bincount(cols, minlength=X.shape[1]),
                })
            print(f'Repeat {rep+1}/{n_repeats}, fold {fold+1}/{n_folds} done')

    return pd.DataFrame(records)


results = run_benchmark(X)

#%%
summary = results.groupby('method').agg(
    RMSE = ('RMSE', 'mean'),
    RMSE_std = ('RMSE', 'std'),
    MAE  = ('MAE',  'mean'),
    max_abs_z = ('max |z| imputed', 'max'),
    time_s = ('time', 'mean'),
).sort_values('RMSE')

# Paired comparison with the reference imputer: every method sees the same folds and masks
reference = 'Iterative'
RMSE_paired = results.pivot_table(index=['repeat', 'fold'], columns='method', values='RMSE')
RMSE_diff = RMSE_paired.sub(RMSE_paired[reference], axis=0)
summary['dRMSE vs ref.'] = RMSE_diff.mean()
summary['dRMSE std. err.'] = RMSE_diff.std() / np.sqrt(len(RMSE_diff))
summary['folds better'] = (RMSE_diff < 0).sum().astype(str) + f'/{len(RMSE_diff)}'

print('\nImputation error on masked entries (standardized units, lower is better):')
print(summary.round(3).to_string())

# Per-feature RMSE for the best methods vs median baseline
per_feature = {
    name: np.sqrt(np.stack(g['per_feature_SE']).sum(0) / np.maximum(np.stack(g['per_feature_N']).sum(0), 1))
    for name, g in results.groupby('method')
}
per_feature = pd.DataFrame(per_feature, index=telemetry_df.columns)
per_feature = per_feature[['Median'] + summary.index[:4].tolist()]
per_feature = per_feature[np.stack(results['per_feature_N']).sum(0) > 0]

print('\nPer-feature RMSE (only features which are ever missing):')
print(per_feature.round(3).sort_values('Median', ascending=False).to_string())

# %%
