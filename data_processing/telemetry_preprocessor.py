import numpy as np
import pandas as pd


class TelemetryPreprocessor:
    """
    Standardizes and imputes telemetry features: a fitted StandardScaler and IterativeImputer merged into a
    single lightweight object that stores only plain arrays. Used to bring new (e.g. science) telemetry to the
    same representation as the one the calibrator was trained on.

    The imputer must use a linear estimator (default BayesianRidge, without posterior sampling). Every imputation
    step is then x_f = clip(X @ coef + intercept, min_f, max_f), which reproduces IterativeImputer.transform exactly,
    while the original sklearn object stores hundreds of fitted estimators with all their internals (~5 MB).

    The state is a dict of numpy arrays and strings, so it can be stored inside *.pt files and loaded
    regardless of the installed scikit-learn version.
    """
    
    def __init__(self, features, mean, scale, initial_fill, step_feature, step_coef, step_intercept, min_value, max_value, meta=None):
        self.features       = list(features)
        self.mean           = np.asarray(mean,           dtype=np.float64) # [N_features]
        self.scale          = np.asarray(scale,          dtype=np.float64) # [N_features]
        self.initial_fill   = np.asarray(initial_fill,   dtype=np.float64) # [N_features], in scaled units
        self.step_feature   = np.asarray(step_feature,   dtype=np.int64)   # [N_steps]
        self.step_coef      = np.asarray(step_coef,      dtype=np.float64) # [N_steps x N_features], zero for the imputed feature itself
        self.step_intercept = np.asarray(step_intercept, dtype=np.float64) # [N_steps]
        self.min_value      = np.asarray(min_value,      dtype=np.float64) # [N_features], in scaled units
        self.max_value      = np.asarray(max_value,      dtype=np.float64) # [N_features], in scaled units
        self.meta           = dict(meta or {})


    @classmethod
    def from_sklearn(cls, scaler, imputer, meta=None):
        """ Converts a fitted StandardScaler + IterativeImputer (fitted on the scaler's output) """
        import sklearn

        features = list(scaler.feature_names_in_)
        N_features = len(features)

        if imputer.n_features_in_ != N_features:
            raise ValueError(f'Imputer expects {imputer.n_features_in_} features, scaler has {N_features}')
        
        if imputer.sample_posterior:
            raise ValueError('Imputers with sample_posterior=True are stochastic and cannot be converted')

        initial_fill = np.asarray(imputer.initial_imputer_.statistics_, dtype=np.float64)
        if initial_fill.shape[0] != N_features or np.isnan(initial_fill).any():
            raise ValueError('Imputer was fitted with completely empty features, which is not supported')

        step_feature, step_coef, step_intercept = [], [], []
        
        for step in imputer.imputation_sequence_:
            estimator = step.estimator
            
            if not (hasattr(estimator, 'coef_') and hasattr(estimator, 'intercept_')):
                raise ValueError(f'Only linear estimators are supported, got {type(estimator).__name__}')
            
            coef = np.zeros(N_features)
            coef[step.neighbor_feat_idx] = estimator.coef_
            step_feature.append(step.feat_idx)
            step_coef.append(coef)
            step_intercept.append(float(estimator.intercept_))

        meta = {
            'sklearn_version': sklearn.__version__,
            'imputer_n_iter':  int(imputer.n_iter_),
            'n_fit_samples':   int(scaler.n_samples_seen_.max() if np.ndim(scaler.n_samples_seen_) else scaler.n_samples_seen_),
        } | dict(meta or {})

        return cls(
            features       = features,
            mean           = scaler.mean_,
            scale          = scaler.scale_,
            initial_fill   = initial_fill,
            step_feature   = np.array(step_feature, dtype=np.int64),
            step_coef      = np.array(step_coef).reshape(-1, N_features),
            step_intercept = np.array(step_intercept),
            min_value      = np.broadcast_to(imputer._min_value, (N_features,)),
            max_value      = np.broadcast_to(imputer._max_value, (N_features,)),
            meta           = meta,
        )

    # ------------------------------------------------- Serialization -------------------------------------------------
    def state_dict(self):
        return {
            'features':       list(self.features),
            'mean':           self.mean,
            'scale':          self.scale,
            'initial_fill':   self.initial_fill,
            'step_feature':   self.step_feature,
            'step_coef':      self.step_coef,
            'step_intercept': self.step_intercept,
            'min_value':      self.min_value,
            'max_value':      self.max_value,
            'meta':           dict(self.meta),
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)

    # --------------------------------------------------- Transforms ---------------------------------------------------
    def _as_array(self, X):
        """ Selects and orders the features if a DataFrame is given, otherwise expects them in the right order """
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.features if f not in X.columns]
            
            if missing:
                raise KeyError(f'Missing telemetry features: {missing}')
            X = X[self.features]
            
        X = np.array(X, dtype=np.float64, ndmin=2)
        if X.shape[1] != len(self.features):
            raise ValueError(f'Expected {len(self.features)} features, got {X.shape[1]}')
        
        return X


    def scale_features(self, X):
        """ Physical units -> standardized units, NaNs are preserved """
        return (self._as_array(X) - self.mean) / self.scale


    def impute(self, Z):
        """ Fills NaNs in standardized features, observed values are kept as they are """
        Z = np.array(Z, dtype=np.float64, ndmin=2)
        missing = np.isnan(Z)
        
        if not missing.any():
            return Z

        Z_filled = np.where(missing, self.initial_fill, Z)
        
        for f, coef, intercept in zip(self.step_feature, self.step_coef, self.step_intercept):
            rows = missing[:, f]
            if rows.any():
                Z_filled[rows, f] = np.clip(Z_filled[rows] @ coef + intercept, self.min_value[f], self.max_value[f])
        
        return Z_filled


    def transform(self, X):
        """ Physical units with NaNs -> standardized and imputed features [N_samples x N_features] """
        return self.impute(self.scale_features(X))


    def inverse_transform(self, Z):
        """ Standardized -> physical units (imputation itself is not reversible) """
        return np.asarray(Z, dtype=np.float64) * self.scale + self.mean


    def __repr__(self):
        return f'TelemetryPreprocessor({len(self.features)} features, {len(self.step_feature)} imputation steps)'
