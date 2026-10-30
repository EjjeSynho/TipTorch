# NFM calibrator module clean-up

This ExecPlan follows `agents/PLANS.md`. Started 2026-10-30.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

`machine_learning/calibrators/NFM_calibrator.py` (~2000 lines) grew hooks, one-line helpers and duplicated logic. Make it easier to read without
changing the behavior of the training or of the inference, and remove three options the user decided to drop.

## Current State

- One-line helpers and hooks used once (`default_split`, `fits_median`, `_build_wavelength_sets`, `_train_fit_targets`, `_resample_spline_ctrls`,
  `_trainable_tensors` / `_extra_param_groups` / `_dataset_level_params`, `to_stacked`, `pretrain_targets`, `_difficulty_to_weights`).
- The normalized-space stacking loop exists twice (`_params_to_normalized`, `_pretrain_target_vector`), the inverse target variance twice
  (pretraining standardization, `fit_reg_weight` regularization).
- Dead state: `lambda_sets`, `loss_cfg`, the `collate_fn` path (always `dataset.collate_batch`), bundle keys `predict_phase_bump` / `loss_cfg`
  (no readers), `validate(verbose)`, the return value of `load_checkpoint` (unused).

## Target Behavior

- Removed (user decision): the parameter-space regularization (`fit_reg_weight`, `shrink_weight`, `delta_std`), `Cn2_targets` ('prior' /
  'fits': always the fits), `pretrain_standardize` (pretraining always compares standardized outputs).
- Kept (user decision): epoch checkpoints, k-fold difficulty weighting, `tune_calibrator`, wavelength subsets, `legacy_weight`.
- Every name other scripts / tests use stays, except `default_split`, `fits_median`, `_regularization` and the `device` argument of
  `_pretrain_target_vector`, whose callers are updated.
- Older bundles / checkpoints still load: `delta_std` / `Cn2_targets` of the factor parameterization and `fit_reg_weight` / `shrink_weight` of
  the stored training configs are dropped on load (nonzero weights give a warning).

## Non-Goals

- No change of the network, the parameterization, the losses or the training loop arithmetic (synthetic test numbers must stay identical).

## Progress

- [x] Inspect the module and its external uses; user decisions on the optional features.
- [x] Rewrite the module (1993 -> 1425 lines), staged in a scratch copy: the user was editing the file at the same time.
- [x] Update callers (run_ablation, train_ETC, train_NFM, compare, analyze_predictability, two tests), configs, training-script notes.
- [x] Validation (in a separate worktree with the new module).
- [x] Put the new module in place (the user's copy was unchanged since the snapshot the rewrite includes).
- [x] Outcomes.

## Surprises & Discoveries

- The user edited NFM_calibrator.py during the rewrite (reformatted the reduce_dataframe call of prepare_telemetry, blank lines): carried
  over into the new version.

## Decision Log

- The optimized shared parameters live in the base trainer (`optimized_params`, empty for the direct calibrator), so the optimizer groups and
  the gradient checks need no hooks.

## Plan of Work

Rewrite the module top to bottom, then fix the callers listed in Target Behavior.

## Concrete Steps

    python tests/test_factor_calibrator_key_groups.py
    python tests/test_Cn2_profile.py

## Validation and Acceptance

The synthetic factor test gives the same numbers as before the rewrite (training loss 1.328 -> 0.331, J_aniso 1.118 -> 1.177, 32/33 with
the known failing check); all touched files compile.

## Idempotence and Recovery

Code only; git restores the previous module.

## Interfaces and Dependencies

See Target Behavior.

## Documentation Updates

The config notes of `train_NFM_calibrator.py`.

## Artifacts and Notes

None yet.

## Outcomes & Retrospective

- `NFM_calibrator.py` 1993 -> 1425 lines; public API unchanged except `default_split` / `fits_median` (callers updated) and the signatures
  listed in Target Behavior. Removed internals: `_build_wavelength_sets`, `_train_fit_targets`, `_resample_spline_ctrls`, `_trainable_tensors`,
  `_extra_param_groups`, `_dataset_level_params`, `_difficulty_to_weights`, `to_stacked`, `pretrain_targets` / `_stacked_targets` (now
  `targets()`), `_tree_to_cpu` (module-level `_to_cpu`), `lambda_sets`, `loss_cfg`, the `collate_fn` argument, `validate(verbose)`, the bundle
  keys `predict_phase_bump` / `loss_cfg`.
- Factor path: tests/test_factor_calibrator_key_groups.py gives the same numbers as before (32/33, training loss 1.328 -> 0.331, J_aniso
  1.118 -> 1.177, final 1.1767594814300537; the failing check fails identically before and after).
- Direct path, real dataset cache, seeded build of NFM_calibrator_config.json with the old and the new module: validation loss, predicted
  cubes, pretraining targets, network after 2 pretraining epochs and bundle forward_prepared all equal (max |diff| 0); the new module loads a
  bundle saved by the old one with identical outputs.
- tests/test_Cn2_profile.py 14/14.
- Not run to the end: tests/test_kc_group_training_smoke.py stops at the trainer build because the current dataset cache's fits carry no
  J_aniso (an optimized shared key of the factor config), the same with the old module; the fits of this cache also lack J_ctrl (the direct
  config pretrains it to a neutral target). Refit the dataset in the regime of the "model" section and rebuild the cache.
