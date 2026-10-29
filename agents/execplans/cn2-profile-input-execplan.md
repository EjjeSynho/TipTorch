# Cn2 profile input of the NFM calibrator: repaired 8-layer input, fixed-grid binning, learned 3-layer reduction

Superseded on 2026-10-29 by `cn2-fixed-grid-execplan.md`: the 'grid' binning is now 3 layers at fixed altitudes with a (5/3)-moment
reprojection (no 8-node features, boundaries or per-sample heights), and the learned mixing / offset are removed.

Started 2026-10-13. Follows `agents/PLANS.md`. Continues the Cn2 follow-up of `factor-calibrator-execplan.md`: the factor calibrator
(15.5 % median peak error) is limited by the telemetry -> parameter mapping, and the Cn2 profile was the next lever.

## Goal

1. Improved binning: replace the weighted K-means reduction of the 8 AOF layers (free, sample-dependent cluster boundaries, so
   "layer 2" of one sample sits at 4 km and of another at 15 km) by fixed altitude boundaries with (5/3)-moment effective heights.
2. 8-layer inputs: the calibrator receives the whole profile as fixed-length features (fractions at 8 fixed altitude nodes), with
   the invalid layers repaired inside each profile instead of discarding them.
3. 3-layer output to the PSF model: the model keeps 3 layers (weights predicted, heights from the binning of the same profile).
4. Better handling via training: the factor calibrator's Cn2 prior is the physical binning of the 8-node input, and the 8 -> 3
   reduction matrix is learned end-to-end (soft assignment initialized at the physical one), with corrections on top.

## Current state (as found)

- Raw profiles: `ALT{i}` / `CN2_FRAC_ALT{i}` (i = 1..8) in `All data` of the cached cubes (`Raw Cn2 data` = SPARTA_CN2_DATA);
  588 samples, 61 without profile, 527 with. Invalid entries: altitude placeholders (1000 km, 0 km, 5e2 km) for the empty upper
  layers, 14 fractions of 1e21, 45 profiles with 1-2 % of mass at an invalid altitude, 11 profiles with unsorted altitudes.
  Median altitudes per layer index: 0.06 / 2.1 / 6.6 / 11.3 / 14.9 / 18.6 / 22.7 / 26.2 km (tight per index).
- `recompute_simplified_Cn2` (telemetry features `Cn2_alt/frac_binned_2/3`) and `InitNFMConfig` (configs) both K-means the
  profile into 3 layers with free altitudes (482 unique height triplets in the cache); the fitted `Cn2_weights` refer to those
  per-sample heights. The fits put a median of 98 % in the ground layer (upper layers 1.8 % / 0.4 %), far from the AOF
  profiler (60-70 % GL): the PSF constrains the profile weakly, the telemetry prior is the better source.
- `telemetry_base_params` builds the factor calibrator's Cn2 base from `Cn2_frac_binned_2/3`.

## Design

- `data_processing/Cn2_profile.py`: `Cn2Grid` (nodes = median AOF altitudes with the first at 0, boundaries between nodes,
  median grid profile) with `repair`, `to_grid` (linear mass split between the two neighbouring nodes: mass and mean altitude
  preserved), `to_layers` (bin masses, (5/3)-moment heights, GL at 0). Stored in `data/reduced_telemetry/MUSE/AOF_Cn2_grid.json`,
  copied into the dataset cache (`Cn2_grid`) and the calibrator bundles.
- Telemetry features (`reduce_dataframe(..., Cn2_binning='grid')`): `Cn2_grid_1..8` + `Cn2_alt_binned_2/3` (effective heights of
  the model layers) replace `Cn2_frac_binned_2/3`; `Cn2 fraction below 2000m` and `theta0` from the repaired profile.
  `Cn2_binning='kmeans'` keeps the old features for old bundles (`NFMCalibrator.prepare_telemetry` reads the mode from the bundle).
- Configs (`InitNFMConfig`): weights and heights of the 3 bins of the same grid projection; median grid profile when missing.
- Boundaries chosen with `machine_learning/analyze_Cn2_binning.py`: PSF of the model with the full repaired 8-layer profile vs.
  3-layer reductions (K-means as now, fixed boundaries candidates) for the fitted parameters of validation samples.
- Factor calibrator: Cn2 base = normalize(grid @ M) with M = softmax(logits) [8, 3], logits a dataset-level parameter
  initialized at the membership matrix (`calibrator.Cn2_mix: learned | fixed`); pretraining target of the Cn2 corrections is the
  prior itself (`calibrator.Cn2_targets: prior | fits`), since the fitted profiles are unreliable.
- Dataset cache regenerated (`create_STD_dataset.py` cells refactored into `fit_telemetry_preprocessor` / `build_dataset_cache`
  of `STD_dataset_utils.py`, run headless); the old cache kept as `muse_STD_stars_dataset_kmeansCn2.pt`.

## Plan

1. Cn2 profile module + unit test (`tests/test_Cn2_profile.py`).
2. Binning analysis script, choose the boundaries, store the grid JSON.
3. Telemetry features / configs / dataset utilities; regenerate the dataset cache (new features, preprocessor, configs).
4. Calibrator: base from the grid, learned mixing, Cn2 pretraining targets, bundle fields, `prepare_telemetry` mode.
5. Update the validation script (feature names) and the bundle test; train factor + direct calibrators; compare.
6. ExecPlan / memory.

## Progress

- [x] Cn2 profile module + `tests/test_Cn2_profile.py` (15/15) (2026-10-13).
- [x] Telemetry features (`compute_Cn2_grid_features`, `reduce_dataframe(Cn2_binning=...)`), configs (`InitNFMConfig`), dataset
      utilities (`store_Cn2_grid`, `fit_telemetry_preprocessor`, `build_dataset_cache`), `create_STD_dataset.py` cells.
- [x] Calibrator: `telemetry_base_params(Cn2_grid, Cn2_mix)`, `FactorParameterization(Cn2_grid, Cn2_targets)`, learned mixing
      logits in the factor trainer / bundle, `Cn2_grid` + `Cn2_binning` in the bundles, validation script, bundle test; smoke test
      on the old cache passes (K-means path, prior targets, bundle round trip).
- [x] Binning analysis (`analyze_Cn2_binning.py`): boundaries 4 / 13 km, ground layer pinned, per-sample heights; grid stored in
      `data/reduced_telemetry/MUSE/AOF_Cn2_grid.json`.
- [x] Dataset regeneration (headless run of the refactored cells): 37 features (31 - `Cn2_frac_binned_2/3` + `Cn2_grid_1..8`
      + `Cn2_layer_alt_2/3`), same 534 samples / split / cubes / fitted values as before, configs with the binned layers
      (median heights 0 / 8.55 / 18.1 km). Old cache kept as `muse_STD_stars_dataset_kmeansCn2.pt`, old telemetry pickle as
      `muse_STD_stars_telemetry_kmeansCn2.pickle`, old weights as `data/weights/NFM_{factor_,}calibrator.kmeansCn2/`.
      `muse_fitted_df.pickle` was missing from the STD folder: the fitted values were taken from the previous cache.
- [x] Training / ablations (seed 43, split 427 / 107, balanced loss; runs stop after 45-60 epochs now, ~20 min each):
      see "Training results" below. Official runs retrained with the final defaults (`Cn2_mix learned`, `Cn2_offset true`,
      `Cn2_targets prior`): `data/weights/NFM_factor_calibrator/`, `data/weights/NFM_calibrator/`; ablations in
      `data/weights/NFM_calibrator_cn2exp/{factor_fixedmix,factor_fits,factor_offset}/`. Bundle tests 14/14 (factor), 12/12 (direct).

## Training results (2026-10-13, validation-set medians, `compare_NFM_calibrators.py`)

| run (37 grid features unless stated) | balanced loss | abs. peak err | peak ratio 485/722/928 nm | EE2 err | profile max err 485/722/928 nm |
|---|---|---|---|---|---|
| direct, K-means features (old cache, reference) | 0.365 | 18.7 % | 0.96 / 0.96 / 0.98 | 12.8 % | 20.5 / 13.7 / 12.7 % |
| factor, K-means features (old cache, reference) | 0.343 | 15.5 % | 1.04 / 1.00 / 1.01 | 11.0 % | 16.3 / 10.6 / 10.3 % |
| **direct, grid features** | **0.350** | **17.9 %** | 1.01 / 1.02 / 1.01 | 11.9 % | 23.2 / - / - |
| factor: mixing learned, prior targets, no offset | 0.346 | 17.5 % | 0.94 / 0.93 / 0.98 | 11.7 % | 14.8 / 13.2 / 12.0 % |
| factor: mixing fixed, prior targets | 0.351 | 17.7 % | 0.99 / 0.97 / 0.97 | 10.9 % | 17.1 / 13.4 / 14.0 % |
| factor: mixing learned, fitted targets (analogue of the old prior) | 0.350 | 17.0 % | 1.03 / 1.01 / 1.04 | 11.7 % | 14.4 / 12.3 / 12.6 % |
| **factor: mixing + offset learned, prior targets (default)** | **0.341** | 18.4 % | 0.96 / 0.94 / 0.98 | 11.0 % | 17.4 / 12.3 / 11.0 % |
| fits (oracle, fitted profile at the new heights) | 0.135 | 5.6 % | 0.98 / 1.00 / 1.01 | 2.2 % | 5.1 / 3.1 / 5.1 % |

- The direct calibrator improves with the grid features (0.365 -> 0.350, 18.7 -> 17.9 %). The factor variants are all within
  0.341-0.351 / 17-18.4 %, i.e. at the level of the old factor run's loss but 2-3 points worse in median peak error; the
  differences between the four variants are of the order of the seed-to-seed scatter seen before (one config: median peak
  ratio 0.96 vs 1.06). The Cn2 input is therefore not the lever the predictability analysis suggested.
- Why: with the honest AOF profile as prior (zero corrections) the PSF model's core is far too faint (peak ratio ~0.4, the
  "fits via parameterization" row of `compare_factor_cn2.json`): the model wants far less upper-layer turbulence than the
  profiler reports, which is what the fits express with 98 % ground layer (mean log-ratio fit / telemetry -2.2 / -3.1 for the
  upper layers). The learned offset settles at a much milder ground / upper ratio (x1.7 / x0.6 / x0.6), the learned mixing
  stays close to the altitude bins (a few % of the 15 km node routed to the mid layer). The discrepancy between the model's
  response to the upper layers (LTAO tomography / cone terms) and the data is a PSF-model question, the same one behind the
  ETC white-profile anomaly noted earlier.

## Surprises & Discoveries

- The ETC calibrator files (`calibrators/ETC_calibrator.py`, `train/validate_ETC_calibrator.py`, `tests/test_ETC_calibrator.py`,
  `development/run_tiptorch_random_carlo.py`) are being modified in parallel (uncommitted diffs appearing during this work); they
  are left untouched here. The old `Cn2_frac_binned_*` features stay available through `Cn2_binning='kmeans'`.

- Binning analysis (64 validation samples, fitted parameters, telemetry profiles; `data/analysis/NFM_calibrator/Cn2_binning.csv`):
  reductions vs. the PSF of the full repaired 8-layer profile, median balanced loss / |peak error|:
  grid of 8 fixed nodes (linear split) 0.0105 / 5.7 %; K-means (current) 0.041 / 10.9 %; median profile 0.21 / 34.5 %;
  bins 4-13 km with per-sample (5/3)-moment heights 0.0195 / 7.2 % (best; 4-16.5 km: 0.036 / 8.3 %); the same bins with
  dataset-level heights 0.056 / 12.3 %; bins 1-13 km (the 2 km node out of the ground layer) 0.125 / 24.5 %.
  The PSF model is far more sensitive to the layer heights than expected, in particular to what sits in the first layer: it is
  the only moving layer (`wind_speed_single` is padded with zeros for the upper layers), so the mass of the first layer sets
  the servo-lag error. Hence per-sample heights are kept and the ground bin must contain the 2 km node.

## Decision Log

- Fixed altitude bins between grid nodes instead of K-means with free altitudes: the weights of the model layers then mean the
  same thing for every sample (the calibrator input / output has a stable definition), and the (5/3)-moment heights reproduce
  the full profile's PSF better than the K-means centroids (analysis above).
- Cn2 pretraining targets default to the prior (zero corrections): the fitted profiles (98 % ground layer) do not track the
  AOF profiler, so log(fit / base) targets would teach the network the fits' bias before the end-to-end stage.
- The learned 8 -> 3 mixing is a dataset-level softmax matrix (one per calibrator, not per sample) trained with the constants
  at lr / 10, initialized at the altitude bins with 1 % leakage: a zero network output still reproduces the physical binning.

## Reverted to K-means by default (2026-10-13, user decision)

- `MUSE_data_utils.CN2_BINNING = 'kmeans'` is the default of `InitNFMConfig` and `reduce_dataframe` (`'grid'` keeps the new path);
  `fit_telemetry_preprocessor` / `create_STD_dataset.py` follow it and store the grid only in grid mode. Training defaults back to
  the previous factor mechanism (`Cn2_mix fixed`, `Cn2_targets fits`, `Cn2_offset false`).
- Data restored from the backups: `dataset_cache/muse_STD_stars_dataset.pt`, `muse_STD_stars_telemetry.pickle`,
  `data/weights/NFM_{factor_,}calibrator/` are the K-means ones again; the grid versions are kept as `*_gridCn2.pt`,
  `*_gridCn2.pickle` and `data/weights/NFM_{factor_,}calibrator.gridCn2/` (+ `NFM_calibrator_cn2exp/` ablations). To go back to
  the grid: set `CN2_BINNING = 'grid'`, swap the files back (or regenerate the cache), set the factor config options.

## Outcomes & Retrospective

- Delivered: `data_processing/Cn2_profile.py` (+ `tests/test_Cn2_profile.py`, 16/16), grid-based Cn2 features / configs in
  `MUSE_data_utils.py` (`compute_Cn2_grid_features`, `reduce_dataframe(Cn2_binning, Cn2_grid)`, `InitNFMConfig`), dataset
  utilities (`store_Cn2_grid`, `fit_telemetry_preprocessor`, `build_dataset_cache`) with `create_STD_dataset.py` on top of them,
  `analyze_Cn2_binning.py` (+ `data/analysis/NFM_calibrator/Cn2_binning.{csv,png}`), the factor calibrator's grid prior with
  learned mixing / offset and prior targets (`NFM_calibrator.py`), bundles carrying `Cn2_grid` / `Cn2_binning`, the updated
  validation / comparison / predictability scripts and bundle test, the regenerated cache and retrained calibrators.
- Scientific outcome: the binning is now well defined and 1.5x closer to the full-profile PSF than K-means, the direct calibrator
  gains ~1 point of peak error, the factor calibrator does not gain. The PSF model's response to the upper layers (far too
  pessimistic vs. the data: fits at 98 % GL, learned offset x1.7 / x0.6) is the thing to investigate next (tomographic / cone
  terms of the LTAO model, also behind the ETC white-profile anomaly), before any further telemetry-side work on Cn2.
- Not done: the ETC calibrator module (edited in parallel) does not synthesize the grid features itself yet
  (`Cn2_grid_features_from_GLF` in `NFM_calibrator.py` is ready for it); refitting the per-sample PSF parameters at the new
  layer heights (the fitted `Cn2_weights` still refer to the K-means layers; the oracle moved from 4.5 % to 5.6 % peak error).
