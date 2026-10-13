# NFM factor calibrator: predictability analysis, implementation, regularization, training

Started 2026-10-13. Follows `agents/PLANS.md`. Continues `balanced-loss-calibrator-execplan.md`, whose conclusion was that the
calibrator error (19 % median peak error vs 4.5 % for the per-sample fits) is a telemetry -> parameters problem, not a loss problem.

## Goal

1. Analyse which fitted PSF parameters can be predicted from the 31 telemetry features at all, and for which ones a telemetry-
   derived base value (seeing -> r0, header wind speed, binned Cn2 fractions) already explains most of the variance.
2. Finish the factor calibrator (`SmallFactorNet` / `NFMFactorCalibrator` / `NFMFactorCalibratorTrainer`, sketched in
   `machine_learning/calibrators/NFM_calibrator.py` on 2026-06-29 and never trained): the network predicts corrections to
   telemetry-derived base values instead of absolute values.
3. Add the regularizations discussed: shrinkage of the corrections towards the base / dataset median (a Gaussian prior with
   data-driven widths) and a parameter-space term keeping the end-to-end solution close to the per-sample fits.
4. Train it with the same split / seed / loss as the direct calibrator and compare with the same core / halo metrics.

## Current state (as found)

- The factor sketch duplicates ~900 lines of `NFMCalibratorTrainer` (loaders, astrometry, `run_model`, validate, train, k-fold,
  bundle) with the legacy loss hard-coded, `softplus` multipliers, a factor key `wind_speed` that does not exist in the PSF model
  (`wind_speed_single`), base values that the caller has to supply, and no training script, config, test or validation.
- Per-sample configs only carry seeing and wind speed per sample; `L0` (27.44 m) and `Cn2Weights` (median AOF profile) are
  dataset constants there. The telemetry has per-sample Cn2 fractions (`Cn2_frac_binned_2/3`, `Cn2 fraction below 2000m`) but
  no outer scale.
- Fitted values per sample: `r0`, `L0`, `dn`, `wind_speed_single` (scalars), `Cn2_weights` (2 non-GL layers), `J_ctrl`, `F_ctrl`
  (5 spline nodes), `LO_coefs` (8, first = phase bump). Model transforms: `Uniform` ranges, `SoftmaxInv` for `Cn2_weights`.

## Design

- Base values from the telemetry only (physical units via the bundle's `TelemetryPreprocessor.inverse_transform`, so the
  inference API stays `calibrate(reduced_telemetry, PSF_model)` like `NFMCalibrator`):
  `r0 = 0.976 lambda / seeing` at 500 nm from `Seeing (header)`, `wind_speed_single = Wind speed (header)`,
  `Cn2_weights = [1 - f2 - f3, f2, f3]` from `Cn2_frac_binned_2/3` (clipped), `L0 = median fitted L0` (constant).
- Network outputs are corrections: `value = base * exp(delta)` (factor params), `value = median + delta` in the normalized
  space (`dn`, `J_ctrl`), dataset-level learnable constants (`F_ctrl`, `LO_coefs`). `delta = 0` reproduces the prior.
- Regularization (both configurable in the `loss` section):
  `shrink_weight * mean(delta^2 / sigma_k^2)` with `sigma_k` = training-set std of `log(fitted / base)` (factor params) or of the
  normalized fitted value (direct params); `fit_reg_weight * standardized distance to the per-sample fits` (the pretraining
  criterion) added to the end-to-end loss, available to the direct calibrator as well.
- `NFMFactorCalibratorTrainer` becomes a subclass of `NFMCalibratorTrainer` with two hooks (`_predict(tel, idxs) -> x_dict`,
  `_regularization(x_dict, idxs)`) plus the constants in optimizer / checkpoint / bundle; `NFMFactorCalibrator` a subclass of
  `NFMCalibrator`. Everything else (balanced loss, NaN guard, validate, k-fold, tune) is inherited.

## Plan

1. `machine_learning/analyze_NFM_telemetry_predictability.py`: per fitted parameter, CV R^2 (time-grouped folds) of ridge and
   random-forest regressors on the 31 features; for the factor params the variance explained by the base value alone and the
   R^2 of the residual log-ratio; PSF-level cost of replacing each group by the training-set median in the oracle (GPU).
   Outputs: table + figure in `data/analysis/NFM_calibrator/`.
2. Refactor the trainer hooks, implement the factor calibrator and the regularizations, bundle round trip in
   `tests/test_NFM_calibrator_bundle.py --factor`.
3. `machine_learning/train_NFM_factor_calibrator.py` + `NFM_factor_calibrator_config.json`; train on GPU (seed 43, shared split).
4. `machine_learning/compare_NFM_calibrators.py` (the scratch `eval_runs.py` made permanent): metrics table for the direct,
   factor and regularized runs.

## Progress

- [x] 1 analysis (`data/analysis/NFM_calibrator/telemetry_predictability.{csv,png}`, `median_replacement_metrics.json`)
- [x] 2 implementation (`NFM_calibrator.py`: hooks in `NFMCalibratorTrainer`, `FactorParameterization`, `NFMFactorCalibrator(Trainer)`,
      `load_NFM_calibrator`, `build_NFM_trainer`; `NFM_validation.py`; `train_NFM_calibrator.py` trains both types; bundle test extended)
- [x] 3 training (factor shrink 0.1 / 0.01 / 0, factor fit-reg 0.1, direct fit-reg 0.1; official factor calibrator retrained with shrink 0)
- [x] 4 comparison (`compare_NFM_calibrators.py`, table below)

## Analysis results (2026-10-13, 534 samples, 5 time-grouped folds)

Cross-validated R^2 (standardized targets) and PSF-level cost of replacing the group by the training-set median in the fits
(validation set, |peak error| median; the fits themselves: 4.5 %):

| parameter | R^2 ridge | R^2 forest | base value | R^2 base alone | R^2 of log(fit/base) | cost of the median |
|---|---|---|---|---|---|---|
| r0 | 0.80 | 0.84 | seeing (header) | -0.01 (offset: mean log-ratio -0.25) | 0.72 | +12.7 % |
| dn | 0.63 | 0.65 | - | | | +15.6 % |
| wind_speed_single | 0.99 | 0.98 | wind speed (header) | 0.86 | 0.19 | +0.2 % |
| L0 | 0.20 | 0.21 | none (constant) | | 0.14 | +0.2 % |
| Cn2_weights (GL, 2, 3) | 0.24 / 0.25 / 0.07 | 0.30 / 0.32 / 0.20 | binned fractions | 0.14 / 0.14 / 0.0 | 0.02 / 0.21 / 0.18 | +8.7 % |
| J_ctrl (5) | 0.13-0.17 | 0.17-0.21 | - | | | +6.0 % |
| F_ctrl (5) | 0.10-0.41 | 0.13-0.60 | - | | | +0.4 % |
| LO_coefs (bump + 7) | -0.09-0.19 | -0.02-0.24 | - | | | +12.6 % (17.1 % peak error) |

- Predictable and important: `r0` (the seeing-derived base + a predictable log-ratio) and `dn`. `wind_speed` is given by the
  header; `L0` and `F_ctrl` do not matter for the PSF (constants are enough). `Cn2_weights` matter (8.7 %) but are only weakly
  predictable, `J_ctrl` moderately.
- The LO coefficients (NCPA / static aberrations) are essentially unpredictable from the telemetry (R^2 < 0.1 except one mode)
  yet cost 12.6 % of peak error when fixed to their median: this is the floor of any telemetry-only calibrator, and it is of
  the order of the whole error of the current calibrator (19 %). The remaining lever is a better prediction of r0 / dn / Cn2.

## Follow-up: Cn2 profile input (2026-10-13)

The 8 AOF layers are K-means-clustered into 3 layers with free altitudes (`recompute_simplified_Cn2`), but the PSF-model configs
use the median profile at fixed heights [0, 8.5, 17.4 km] for every sample, so the fitted `Cn2_weights` are weights at the fixed
heights and the per-sample altitudes never reach the model. Proposed: project each profile onto the model heights with the
5/3-moment-preserving binning of `tiptorch.tools.utils.BinCn2` and fixed altitude boundaries (fixed-length, missing layers drop
out), use it as the Cn2 base value of the factor calibrator and optionally add the cumulative fraction below a few fixed
altitudes as features. First check, in the fitting pipeline, that fits with the per-sample projected profile track the telemetry
(log-ratio scatter now 1.1-1.7); only then consider per-sample `Cn2Heights` (weights predicted, heights fixed).

## Training results (2026-10-13, split 427 / 107, seed 43, balanced loss, validation-set medians)

| run | balanced loss | abs. peak err | peak ratio p16-p84 | peak ratio 485/722/928 nm | EE(r<=2) err | profile max err 485/722/928 nm |
|---|---|---|---|---|---|---|
| direct, balanced (reference) | 0.350 | 18.7 % | 0.71-1.26 | 0.96 / 0.96 / 0.98 | 12.8 % | 20.5 / 13.7 / 12.7 % |
| direct + fit-reg 0.1 | 0.350 | 21.8 % | 0.78-1.42 | 1.09 / 1.07 / 1.10 | 12.5 % | 16.8 / 16.1 / 16.2 % |
| **factor, no shrinkage** | **0.343** | **15.5 %** | **0.77-1.27** | **1.04 / 1.00 / 1.01** | **11.0 %** | **16.3 / 10.6 / 10.3 %** |
| factor + shrink 0.01 | 0.351 | 16.9 % | 0.76-1.30 | 1.03 / 1.00 / 1.03 | 12.0 % | 13.4 / 11.9 / 11.9 % |
| factor + shrink 0.1 | 0.356 | 19.4 % | 0.85-1.47 | 1.18 / 1.11 / 1.14 | 11.3 % | 19.5 / 12.1 / 13.2 % |
| factor + fit-reg 0.1 | 0.354 | 20.0 % | 0.84-1.45 | 1.16 / 1.14 / 1.14 | 12.1 % | 21.5 / 12.8 / 14.2 % |
| fits with constant F_ctrl / LO_coefs (medians) | 0.247 | 20.2 % | 1.07-1.39 | 1.19 / 1.24 / 1.19 | 8.5 % | 10.2 / 12.6 / 15.0 % |
| fits (oracle) | 0.126 | 4.5 % | 0.91-1.04 | 0.97 / 1.00 / 1.00 | 1.6 % | 3.5 / 2.0 / 3.9 % |

- The factor calibrator without regularization beats the direct one on every core metric (15.5 % vs 18.7 % peak error, no
  bias, tighter spread) and matches it in the halo. The direct runs of the same config scatter by ~1 % in peak error
  (18.7 / 18.7 / 19.1), so the 3-point gain is above the run-to-run scatter, but it is a single seed.
- Both regularizations at weight 0.1 are too strong: the prior pulls the core 10-20 % too bright (the medians of the fits
  give a sharper PSF than typical, see the "constant medians" row) and the networks end up worse than unregularized. A weak
  shrinkage (0.01) lands in between (16.9 %): the shrinkage is monotonically harmful here and both terms stay at 0 in the
  config (kept as options, they are cheap to re-test once the priors are better, e.g. with a projected Cn2 profile).
- Seeded runs are bit-reproducible: the official retrain of the factor calibrator (shrink 0) gives the same numbers as the
  earlier `factor_noshrink` run.
- The learned dataset-level constants matter: the trained factor calibrator (15.5 %) beats the fits with *median*
  constants (20.2 %), i.e. the end-to-end optimized LO_coefs / F_ctrl are a better "average NCPA" than the median.

## Surprises & discoveries

## Decision log

## Decision log

- Factor trainer as a subclass of the direct trainer (hooks `_predict` / `_params_from_pred` / `_regularization` /
  `_extra_state`) instead of finishing the 900-line copy: everything added to one trainer (balanced loss, NaN guard, k-fold,
  bundles) is shared, and `train_NFM_calibrator.py` / `validate_NFM_calibrator.py` work for both types.
- Base values derived from the telemetry inside the calibrator (not supplied by the caller): the inference API stays
  `calibrate(reduced_telemetry, PSF_model)`.
- Default config `NFM_factor_calibrator_config.json`: `shrink_weight 0`, `fit_reg_weight 0` (see results).

## Outcomes & retrospective

- Official factor calibrator: `data/weights/NFM_factor_calibrator/factor_calibrator_{checkpoint,bundle,meta,split}` (shrink 0,
  15.5 % median peak error on the validation set vs 18.7 % for `NFM_calibrator/best_calibrator_*`); the shrink-0.1 run is
  archived in `NFM_factor_calibrator_shrink0.1/`, the variants in `NFM_calibrator_lossexp/{factor_noshrink,factor_shrink001,
  factor_fitreg,direct_fitreg}/`. Bundle round trip: `tests/test_NFM_calibrator_bundle.py --weights-subdir NFM_factor_calibrator
  --name factor_calibrator` (13/13).
- Not done / next: the Cn2 input (follow-up above), more seeds for the factor-vs-direct comparison, and the ETC-regime
  validation of the factor bundle (`validate_NFM_calibrator.py` ETC section uses `NFMCalibrator`; switch it to
  `load_NFM_calibrator`).
