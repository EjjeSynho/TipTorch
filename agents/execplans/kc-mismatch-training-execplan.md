# MUSE NFM STD stars with a DM correction-radius (k_c) mismatch: detection, labels and training integration

This ExecPlan follows `agents/PLANS.md`. Started 2026-10-16.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

Some MUSE NFM STD stars were recorded with fewer DM modes corrected, so their AO cut-off frequency k_c (correction radius) is smaller
than the nominal k_c = 1 / (2 DmPitch) = 1 / (2 x 0.22 m) that the PSF model uses for every sample. In the residuals (data - fit) these
samples show an azimuthally coherent ring just inside the nominal correction radius r_c(λ) = λ k_c / pixel scale, at the same r / r_c(λ)
for all wavelength bins. The user wants to keep these PSFs in the training set (they carry useful information), so after this change:

- the samples are detected automatically from the fitted residuals, labeled by MUSE filename (robust to re-indexing) with their k_c
  in percent of the nominal one, and the label is a column of `muse_df.pickle` (`'k_c mismatch'`, `'k_c fraction'`, `'k_c group'`);
- the labeled samples are grouped by the closest averaged k_c, their PSF-model configs carry the scaled DM pitch, they form their own
  training batches, and the PSF model is rebuilt for the k_c of every batch;
- the fitting (`STD_fitter.py`) and `development/MUSE_onsky.py` react to the label.

Observable outcomes:

- `python datasets/MUSE_STD_dataset/kc_mismatch.py [--force]` writes `STD_FOLDER/kc_ring_scores.csv`, appends new detections to
  `STD_FOLDER/kc_mismatch_labels.txt` and refreshes the three columns of `muse_df.pickle`; it is called from `create_STD_dataset.py`.
- `python tests/test_kc_mismatch_groups.py` (14 checks) and `python tests/test_kc_group_training_smoke.py` (8 checks on the real cache) pass.
- A model updated to a scaled-pitch config with `Update(grids=True)` reproduces a model built from that config exactly.

## Current State (before the change)

- `TipTorch.InitValues` takes `self.pitch = config['DM']['DmPitchs'].min()` and `self.kc = 1 / (2 pitch)`; `InitGrids` builds the
  corrected-frequency mask, `nOtf_AO`, the PSD padder, the piston filter and the aliasing combs from it. One k_c per model instance.
- `['DM', 'DmPitchs']` is in `SINGLETON_VALUES` of `config_manager.py`: `Merge` keeps the pitch of the first config, so a batch mixing k_c
  would silently use the first sample's k_c.
- `NFMCalibratorTrainer._make_loader` uses a plain shuffled `DataLoader`; `run_model` sets the batch config and calls
  `Update(grids=False, tomography=True)` (single wavelength set) or `SetWavelengths` (which calls `Update(grids=True)`).
- `InitNFMConfig` sets `DmPitchs = [0.22]` for every sample; `LoadSTDStarCache` is the single place where the per-sample configs are born
  (fitter, onsky script and `build_dataset_cache` all use it).
- `muse_df.pickle` has the hand labels (`Corrupted`, `Non-point`, ...), built from `labels.txt` in `create_STD_dataset.py`.

## Target Behavior

1. Detection (`datasets/MUSE_STD_dataset/kc_mismatch.py`): relative residual (data - fit) / <fit>(r) per wavelength, azimuthal
   median in bins of u = r / r_c(λ) (suppresses streaks / the tomographic cross), median over wavelength, bump above baseline in
   u ∈ [0.76, 1.08], angular coherence (16 sectors), wavelength consistency, sub-bin peak position = k_c fraction. Classes strong / weak /
   none (thresholds set by eye on the 2026-10 galleries). `update_kc_mismatch_labels(muse_df)` scans only fits absent from the scores CSV.
2. Labels file `STD_FOLDER/kc_mismatch_labels.txt`: `<Filename>: k_c <p> % (ring|fit), <class>, amp <v>, <date>`; classes `manual`
   (mismatch by hand) and `none` (vetoed by hand, excluded, never re-added); `(fit)` values are never overwritten by the detector.
3. `add_kc_mismatch_flag` (STD_dataset_utils.py): groups the labeled samples with `group_kc_fractions` (smallest 1-D k-means such that
   every sample is within `KC_GROUP_TOLERANCE` = 3 % of its group mean) and writes `'k_c mismatch'` (bool), `'k_c fraction'` (group mean,
   1.0 nominal), `'k_c group'` (0 nominal, 1..G by increasing k_c). All three are registered in `MUSE_data_utils.py` as non-telemetry.
4. `LoadSTDStarCache`: `apply_kc_fraction(config, fraction)` scales `DmPitchs` by 1 / fraction, so every consumer gets the right k_c.
5. `MultipleTargetsInDifferentObservations` raises when the configs have different pitches.
6. Dataset cache: `kc_fraction`, `kc_group` arrays (older caches: derived from the configs' pitches). `NFMDataset` exposes them;
   `KcGroupBatchSampler` makes single-group batches (smaller last batches per group, batch order shuffled with torch's RNG);
   `run_model` calls `Update(grids=True)` when the batch pitch differs from the model's; `build_NFM_trainer` picks its representative
   batch from one group.
7. `STD_fitter.py`: the module flag `APPLY_KC_CORRECTION` (next to `DEROTATE_PSF`) decides whether the flagged samples are fitted with
   their label k_c or the nominal one; the flag and the group go into the fit pickle. `load_and_fit_sample(id, device, kc_fraction=None)` is
   importable (no module-level fit loop), a `kc_fraction` value forces the k_c (used by the scan). The CLI takes device / ids / folder only. `MUSE_onsky.py` prints the label of the selected ids and refuses ids of different groups in one model.
8. `kc_mismatch.py --scan`: refines the k_c of the labeled samples by fitting with the k_c forced to ring estimate x {0.86, 0.90, 0.94, 0.98}
   (parallel `STD_fitter.py` subprocesses), takes the parabolic loss minimum, writes `(fit)` values to the labels and refreshes `muse_df`.

## Non-Goals

- Fitting k_c as a continuous parameter inside TipTorch (k_c changes the grid sizes; it is not differentiable).
- Mixed-k_c batches (would need per-observation masks / padding in the PSD; the group batching avoids it).
- Science observations (`MUSEObservation`, `NFMCalibrator` inference) taken in the affected periods: the calibrator predicts the
  parameters, but the model's k_c must be set from the date / header by the caller; not addressed here.
- Re-fitting the whole dataset and regenerating the dataset cache: the fits are being redone by the user anyway; the cache must be
  rebuilt with `create_STD_dataset.py` after the fits (`build_dataset_cache` stores the new fields).

## Progress

- [x] (2026-10-16) Detector, galleries, scores CSV; 25 strong + 15 weak + 2 manual (437, 443) detections; `'k_c mismatch'` column.
- [x] (2026-10-16) Labels file with k_c percent, incremental `update_kc_mismatch_labels` hooked into `create_STD_dataset.py`.
- [x] (2026-10-16) Grouping (`group_kc_fractions`, 3 % tolerance -> 3 groups: 90.4 % (14), 93.5 % (19), 96.9 % (9) from the ring
      estimates), `apply_kc_fraction` in `LoadSTDStarCache`, cache fields, merge guard, `KcGroupBatchSampler`, `run_model` grid rebuild,
      fitter / onsky reactions, `tests/test_kc_mismatch_groups.py` (14/14), `tests/test_kc_group_training_smoke.py` (8/8, GPU 1).
- [x] (2026-10-16) Validation fit of id 19 with k_c forced to 100 / 90.2 / 87 / 84 / 80 %: loss 0.214 / 0.189 / 0.179 / 0.172 / 0.174,
      ring amplitude 0.49 / 0.31 / 0.21 / 0.11 / 0.01 -> the ring estimate (89 %) is biased high, the real k_c is ~83 %.
- [x] (2026-10-16) `kc_mismatch.py --scan` over all 42 labeled samples (factors 0.86-0.98 of the ring estimate, then 0.98-1.06 for the
      25 samples at the upper edge; ~1.5 h on cuda:0 with 6 jobs). Fitted k_c -> labels `(fit)`; 6, 34, 51 fit best at >= 99 % of the
      nominal k_c and were set to the `none` class (not a mismatch by the fit criterion). Final labels: 39 mismatched samples in 4 groups
      (3 % tolerance): 82.5 % (6), 85.9 % (9), 90.6 % (15), 95.1 % (9). Period A spans 80-91 %, period B 89-94 %, the 2018-19 ones 91-97 %.
      Results: `STD_FOLDER/kc_scan_results.csv`, fits under `STD_FOLDER/fitted_kc_scan/<id>/kc_<f>/`. Tests re-run: 14/14.
- [ ] User's flow: refit (`launch_fitting.py`, the configs now carry the group k_c), `STD_fitted_df.py`, `create_STD_dataset.py` (new
      cache with `kc_fraction` / `kc_group`), retrain. Science observations of the affected periods: set the k_c by hand (see Non-Goals).

## Surprises & Discoveries

- Every sample of 2019-04-06..2019-05-13 and of 2024-10-29..2025-01-05 shows the ring: configuration events, not random bad fits. A
  weaker ring (k_c 95-98 % by the ring estimate) exists throughout 2018-09..2019-11; the user's notes in `development/MUSE_onsky.py`
  list period-B samples (440, 446, 449, 451, 453) and 475 as "maybe" (no ring in the residual).
- Non-point / Tomography-error samples (79, 192, 235, 338, 345, 367, 416, 420) produce cross / streak bumps at u <= 0.88 that pass an
  amplitude test; the azimuthal median + coherence + `ring_u >= 0.90` for the weak class reject them.
- The ring position is not the real k_c: for id 19 the fit loss minimum is at ~83 % while the ring sits at 89 % (the fitted r0 / dn
  absorb part of the mismatch: dn 22.6 -> 16.2 between 100 % and 80 %). A model-vs-model test (no re-fit) shows no bias (89.6 % for
  90.2 %), so the bias comes from the re-optimization. Hence the fit-based refinement (`kc_mismatch.py --scan`).
- `DmPitchs` is a singleton in `ConfigManager.Merge`: a mixed batch would silently take the first sample's k_c (now an error).
- `SetWavelengths` (multi-wavelength-set training) already rebuilds the grids on every batch, so only the single-set path needed the
  explicit `grids=True` on a k_c change.
- The data folder has no raw / cube FITS files, so the mode count could not be confirmed from headers.

## Decision Log

- Decision: label by MUSE filename, not by sample id (ids change when the cache is re-indexed).
- Decision: group by k_c value (1-D k-means with a 3 % tolerance), not by observing period; with 3 % the groups follow the three
  populations closely (A: 12/14 in group 1, B: 14/16 in group 2, 2018-19 weak: 8/11 in group 3 with the ring estimates).
  Alternatives: 1.5 % gives 6 groups of 4-10 samples (over-fits the ~1-2 % scatter of the estimate), 3.5 % merges A with part of B.
- Decision: one k_c per batch (group sampler + `Update(grids=True)` on change) instead of per-observation masks in TipTorch: no change
  to the PSD math, a few grid rebuilds per epoch (~0.1 s each), smaller batches for the groups (LayerNorm, no BatchNorm, so fine).
- Decision: the k_c of the labels is refined by fits (`(fit)` source) and the ring estimate is kept only as the first guess; the detector
  never overwrites a fitted value.
- Decision: `kc_mismatch.py --scan` reuses `STD_fitter.py` as subprocesses (the production fitting routine, incl. the balanced loss and
  the flag handling) rather than a separate fitting loop.

## Plan of Work / Concrete Steps

1. `datasets/MUSE_STD_dataset/kc_mismatch.py` (one `#%%`-cell module, merged from the former detect / scan scripts on
   2026-10-16): ring diagnostics, `update_kc_mismatch_labels`, gallery, `scan_kc_by_fits` (`--scan`).
2. `datasets/MUSE_STD_dataset/STD_dataset_utils.py`: labels parser / writer, `group_kc_fractions`, `apply_kc_fraction`,
   `kc_fraction_from_config`, `add_kc_mismatch_flag`, `LoadSTDStarCacheByID` / `LoadSTDStarCache` / `build_dataset_cache` changes.
3. `data_processing/MUSE_data_utils.py`: register the three columns. `src/tiptorch/managers/config_manager.py`: merge guard.
4. `machine_learning/calibrators/NFM_calibrator.py`: `NFMDataset.kc_group / kc_fraction`, `KcGroupBatchSampler`, `_make_loader`,
   `run_model`, `build_NFM_trainer`, `_kc_groups_summary`.
5. `datasets/MUSE_STD_dataset/STD_fitter.py`, `development/MUSE_onsky.py`, `create_STD_dataset.py` hooks.
6. `kc_mismatch.py --scan`; tests.

## Validation and Acceptance

- `source ~/miniforge3/etc/profile.d/conda.sh && conda activate TipTorch && python tests/test_kc_mismatch_groups.py cuda:0` -> 14/14.
- `python tests/test_kc_group_training_smoke.py cuda:1` -> 8/8 (loads the 3.6 GB cache, ~3 min).
- `cd datasets/MUSE_STD_dataset && python STD_fitter.py cuda:0 19 19 <out>` -> fit with the label k_c (`APPLY_KC_CORRECTION = True`);
  `python kc_mismatch.py --scan --ids 19 --factors 0.91 --no_apply` -> one forced-k_c fit through the scan path.
- Galleries: `STD_FOLDER/plots/kc_ring_gallery.png`, scan plots in the session scratchpad (`fit_kc/kc_scan_19.png`).

## Idempotence and Recovery

- `kc_mismatch.py` is incremental (scores CSV keyed by filename); `--force` rescans. Labels are never removed automatically;
  hand classes (`manual`, `none`) and `(fit)` values are preserved.
- `kc_mismatch.py --scan` reuses existing fits in the scan folder (resume after an interruption); `--no_apply` leaves the labels alone.
- Backup of the dataframe before the labels: `STD_FOLDER/muse_df_backup_2026-10-16.pickle`.

## Interfaces and Dependencies

- `muse_df` columns: `'k_c mismatch'` bool, `'k_c fraction'` float (group mean, 1.0 nominal), `'k_c group'` int (0 nominal).
- Fit pickles: `'k_c fraction'`, `'k_c group'` (-1 when the k_c was forced via `load_and_fit_sample(..., kc_fraction=v)`).
- Dataset cache: `'kc_fraction'` [N], `'kc_group'` [N]; configs with the scaled `DmPitchs`.
- `NFMDataset.kc_group / kc_fraction`; `KcGroupBatchSampler(groups, batch_size, shuffle)`.

## Artifacts and Notes

- `STD_FOLDER/kc_ring_scores.csv`, `kc_mismatch_labels.txt`, `kc_scan_results.csv`, `plots/kc_ring_gallery.png`, `plots/kc_ring_scatter.png`.
- Session scratchpad: `ring/` (exploration), `fit_kc/` (id 19 scans), `smoke.log`, `kc_scan.log`.

## Outcomes & Retrospective

- The ring in the residuals is a reliable detector (whole observing periods light up, impostors are rejected), but its position is a
  biased k_c estimate (3-8 % high after a fit); the fit-based scan is the reliable estimator. The detector remains the cheap first pass
  and the scan the refinement; both are idempotent and keyed by filename.
- For period A the fitted k_c (80-86 %) lowers the fit loss by ~8 % (median) relative to the nominal-k_c fits and removes the ring
  (amplitude 0.49 -> 0.11 on id 19); for period B and the 2018-19 samples the gain is small (the mismatch is 5-10 %).
- The training-side change is contained: a batch sampler, a grid rebuild on k_c change, a merge guard; the PSD math is untouched.
- Open: regenerate fits / cache / weights with the new labels; the scan folder (`fitted_kc_scan/`, ~170 fits) can be deleted afterwards.
