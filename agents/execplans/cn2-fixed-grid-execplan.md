# Cn2 profile: two modes only, K-means or 3 fixed layers

This ExecPlan follows `agents/PLANS.md`. Started 2026-10-29. Simplifies the result of `cn2-profile-input-execplan.md`.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

The Cn2 handling of the NFM calibrator had two binnings ('kmeans', 'grid') and a 'both' feature mode, plus three factor-calibrator
switches on top of them (`Cn2_mix` learned / fixed, `Cn2_offset`, `Cn2_targets`). Most combinations were unused. After this change there
are two modes:

- `kmeans`: as before, weighted K-means of the 8 AOF layers into 3 layers with per-sample altitudes.
- `grid`: 3 PSF-model layers at fixed altitudes, the same for every sample (the dataset medians of the per-sample K-means altitudes,
  0 / 9.15 / 19.1 km on the full muse_df); each AOF profile is reprojected onto them.

## Current State

- `Cn2Grid` (data_processing/Cn2_profile.py): 8 nodes at the median AOF layer altitudes, linear mass split onto the nodes (mass and
  mean altitude preserved), then boundaries (4 / 13 km) bin the nodes into 3 layers with per-sample (5/3)-moment heights.
- Grid features: `Cn2_grid_1..8` + `Cn2_layer_alt_2/3`. Configs: bin masses and per-sample heights.
- Factor calibrator: Cn2 base = normalize(grid @ membership or softmax(Cn2_mix_logits)) * exp(Cn2_offset).
- All configs use `Cn2_mix: fixed`, `Cn2_offset: false`; the active dataset cache is K-means (`CN2_BINNING = 'kmeans'`).

## Target Behavior

- `Cn2Grid(nodes_km, layer_altitudes_km, median_fracs, pin_ground_layer)`: the nodes are the PSF-model layers. `to_grid` splits the
  mass of an AOF layer between the two nodes around it so that the total mass and the (5/3)-moment of the altitude are preserved
  (anisoplanatism scaling); layers above the top node go to it; with `pin_ground_layer` the first AOF layer goes to the ground node.
- Grid features: `Cn2_grid_1..3` (the reprojected weights). Configs: `Cn2Heights` = the nodes for every sample, `Cn2Weights` = the
  reprojected weights (median profile when the sample has none).
- `store_Cn2_grid` fits the nodes from the dataset (median per-sample K-means altitudes) unless given.
- Factor calibrator: Cn2 base = normalized grid weights (grid) or the K-means fractions (kmeans). No mixing, no offset.
- `reduce_dataframe(Cn2_binning)` accepts 'kmeans' or 'grid' only.

## Non-Goals

- Rebuilding the dataset cache or refitting the per-sample parameters in grid mode (needs the STD data and GPU time; the fits must be
  redone with the grid configs, since the fitter keeps the config heights).
- Changing the K-means path or the default `CN2_BINNING = 'kmeans'`.
- `Cn2_targets` ('fits' / 'prior') stays.

## Progress

- [x] Inspect relevant files and current behavior.
- [x] Rewrite `Cn2Grid` / `fit_Cn2_grid`, update `tests/test_Cn2_profile.py` (14/14).
- [x] Features / configs (`MUSE_data_utils.py`), `store_Cn2_grid`, `create_STD_dataset.py`.
- [x] Calibrator: remove the mixing / offset, simplify `telemetry_base_params` and `Cn2_grid_features_from_GLF`; configs JSON.
- [x] `analyze_Cn2_binning.py`: compare K-means, the fixed 3-layer grid (5/3 and linear split, GL pinned / split) and the median
      profile against the 8-layer reference. Not run (GPU, ~5 min).
- [x] Run validation (see Outcomes).
- [x] Update `Outcomes & Retrospective`.

## Surprises & Discoveries

- The stored working copy `AOF_Cn2_grid.json` is in the legacy format (8 nodes + boundaries). It is rewritten by the next
  `store_Cn2_grid` call (create_STD_dataset.py); `Cn2Grid.from_state_dict` rejects legacy states with an explicit message.
- `cn2-profile-input-execplan.md` measured the old bins with dataset-level heights at a balanced loss of 0.056 vs 0.041 for K-means
  (8-layer reference). The new grid differs (moment-preserving reprojection instead of bin masses), so the analysis must be rerun.

## Decision Log

- 3 layers at the median K-means altitudes, not 8 layers at the median AOF altitudes: the network outputs and the fits stay 3-layer,
  and the PSF model cost does not grow (user decision, 2026-10-29).
- (5/3)-moment-preserving split instead of the linear (mean-altitude) one: with only 3 nodes, the AOF layer at 2.1 km (34 % of the median
  profile) goes 91 / 9 % to 0 / 8.6 km instead of 76 / 24 %, which keeps its anisoplanatism contribution (user decision, 2026-10-29).
- The 'both' feature mode of `reduce_dataframe` is removed with the other unused options.

## Plan of Work

See Progress. The grid is still copied into the bundle / dataset cache via `Cn2Grid.state_dict()`.

## Concrete Steps

Run from repository root with the AO-torch environment:

    python tests/test_Cn2_profile.py
    python tests/test_factor_calibrator_key_groups.py

## Validation and Acceptance

- `tests/test_Cn2_profile.py` passes: repair, mass and (5/3)-moment preservation, pinned ground layer, top-node clipping, state round
  trip, legacy-state rejection, `fit_Cn2_grid`.
- The modified modules import; the factor key-group test passes (K-means features, no grid).

## Idempotence and Recovery

Code only. The legacy grid JSON is not touched until `store_Cn2_grid` runs.

## Interfaces and Dependencies

`Cn2Grid` loses `boundaries`, `membership`, `to_layers`, `default_layers`, `N_grid`; `process` returns `(weights, n_repaired, valid)`.
`telemetry_base_params` loses `Cn2_mix` / `Cn2_offset`; `NFMFactorCalibratorTrainer` loses `Cn2_mix` / `Cn2_offset`; bundles no longer
carry `Cn2_mix_logits` / `Cn2_offset` (old keys are ignored).

## Documentation Updates

Docstrings / comments of the touched modules, the "calibrator" notes of `train_NFM_calibrator.py`.

## Artifacts and Notes

None yet.

## Outcomes & Retrospective

- Delivered: two Cn2 modes ('kmeans', 'grid'); `Cn2Grid` reduced to nodes = model layers with the (5/3)-moment reprojection;
  `Cn2_mix` / `Cn2_offset` removed from the factor trainer, the inference class, the bundles and the configs; 'both' removed.
- Fitted from the full `muse_df` as in create_STD_dataset.py, the grid is 0 / 9.15 / 19.1 km (the 534 cached samples alone give
  medians of 8.6 / 18.3 km) with the median profile 0.731 / 0.172 / 0.097 (K-means configs: 0.716 / 0.169 / 0.103).
- Validation: `tests/test_Cn2_profile.py` 14/14. Smoke test on the STD data (grid fitted, not stored): 525 / 588 grid feature rows
  with a profile, all summing to 1; 'both' rejected; factor base values = the normalized features; a grid-mode config has the nodes as
  heights and the feature weights. `tests/test_factor_calibrator_key_groups.py` 33/34, identical to HEAD: the failing check ("the learned
  J_aniso / J_angle lower the training loss", 0.8401 -> 0.8520) fails the same way at HEAD and is unrelated. Its line 220 still listed
  `wind_speed_single`, which commit 8854768 removed from the factor keys (KeyError); it now takes the factor keys from the parameterization.
- Not done: running `analyze_Cn2_binning.py`, rebuilding the dataset cache / refitting / retraining in grid mode.
