# NFM calibrator: balanced PSF loss vs. legacy loss

Started 2026-10-12. Follows `agents/PLANS.md`.

## Goal

With the dynamic-range-balanced PSF loss (`BalancedPSFLoss` in `src/tiptorch/tools/utils.py`, config section `loss` of
`machine_learning/NFM_calibrator_config.json`) the calibrator trained by `machine_learning/train_NFM_calibrator.py` looks much
worse in `machine_learning/validate_NFM_calibrator.py` than the calibrators trained before with the legacy linear MSE + MAE loss.
Find out why, improve the training and make sure the calibrator bundle (network + outputs transformer + telemetry scaler/imputer)
is saved and loaded correctly.

## Current state (as found)

- `data/weights/NFM_calibrator/best_calibrator_*` is the balanced-loss run of 2026-10-12 18:47 (534-sample dataset, 427/107
  time-grouped split, best val 0.365 at epoch 23). All 534 fits in `fitted/` were redone on 2026-10-12 with the balanced loss
  of `STD_fitter.py`, so the pretraining targets are balanced-loss fits as well.
- `deprecated/train_NFM_calibrator.py` is the old notebook-style script and has no balanced loss; the live code is
  `machine_learning/train_NFM_calibrator.py` + `machine_learning/calibrators/NFM_calibrator.py`.
- Validation-set metrics (same split, scratch script `diag_val_metrics.py`), median over samples:

  | predictor | balanced loss | legacy loss | abs. peak err | EE(r<=2) err | profile max err 485/722/928 nm |
  |---|---|---|---|---|---|
  | current balanced checkpoint (epoch 23) | 0.365 | 0.378 | 18.7 % | 11.5 % | 19.9 / 15.3 / 12.1 % |
  | pretrain only | 0.412 | 0.420 | 22.6 % | 14.3 % | 22.1 / 17.9 / 16.0 % |
  | per-sample fits ("oracle") | 0.126 | 0.167 | 4.5 % | 1.6 % | 3.5 / 2.0 / 3.9 % |

  The calibrator is unbiased in the core (median peak ratio 1.0-1.05) but scatters a lot (p16-p84 of the peak ratio 0.76-1.41).
  Beyond r = 20 px the model is 8-20 % fainter than the data for the fits too (a PSF-model limitation, hence `r_max = 40`).
- The Sep-30 legacy bundle (`NFM_calibrator_bundle.pth`) cannot be used as a reference: the PSF model (`dn` normalization) changed
  since, its predictions give a 60-80 % too bright core now.

## Surprises & discoveries

- `validate_NFM_calibrator.py` loaded the telemetry preprocessor from the *checkpoint* (`best_calibrator_checkpoint.pth`,
  weights only) instead of the bundle, which raises `ValueError('Calibrator bundle has no telemetry preprocessor')`. Fixed:
  the bundle is used, with the dataset-cache preprocessor as a fallback.
- Training runs were not seeded (network init, batch order, dropout), so runs differing only in the loss were not comparable;
  the pretraining was always redone. Added `seed` and `reuse_pretrain` to the training config.
- One training run takes 45-52 GB of GPU memory (full graph of 16 samples x 31 wavelengths x 199^2), so two runs do not fit on
  one 96 GB GPU: the A/B runs go sequentially (`queue_lossexp.sh`), ~15-20 min each.

- The three balanced A/B runs crashed in epoch 0 with `torch.linalg.lstsq: Argument 4 has illegal value` (tomographic
  reconstructor). Root cause, found with `torch.autograd.detect_anomaly`: `PSD2PSF` computed `self.OTF.abs()` on the full complex
  OTF before indexing its centre; where the OTF underflows to exactly zero (far wings, low r0 / blue end, sample 253 of the dataset)
  the gradient of `abs()` is 0/0 and the zero gradient of the indexing turns into NaN, which poisons the whole batch. The loss stays
  finite, so the NaN recovery never triggered: the optimizer step wrote NaN weights and the next forward pass crashed. Independent
  of the loss (legacy and balanced alike), it only depends on which samples share a batch. Fixed in `TipTorch.py` (index, then
  `abs()`), and `NFMCalibratorTrainer.train` now skips batches with non-finite gradients instead of stepping.

## Plan

1. Controlled A/B on GPU 1 (same seed 43, same pretrain weights, same split), weights under
   `data/weights/NFM_calibrator_lossexp/<name>/`, configs/logs in the session scratchpad:
   `legacy` (linear MSE + MAE, LO sign penalty), `balanced` (current config), `balanced_d1` (Huber delta 1.0, scale 20: quadratic
   in asinh space for the typical 10-40 % errors), `balanced_mix` (balanced + 1.0 x legacy).
2. Evaluate every run with the same core/halo metrics as above and decide on the loss / config.
3. Bundle round trip: a test that the bundle preprocessor equals the dataset-cache one, reproduces the stored telemetry vectors
   from the raw telemetry, and that `NFMCalibrator` reproduces the trainer's network outputs.
4. Update the training script / config / validation accordingly.

## Progress

- [x] Diagnostics of the current checkpoint, the pretrain-only net and the fits (table above).
- [x] `seed`, `reuse_pretrain` in `train_NFM_calibrator.py`; preprocessor loading fixed in `validate_NFM_calibrator.py`.
- [x] A/B runs (2026-10-12, `queue_lossexp.sh`, one run at a time on GPU 1).
- [x] Metrics of the A/B runs (table below), decision.
- [x] Bundle round-trip test `tests/test_NFM_calibrator_bundle.py` (10/10 checks pass on the current bundle).

## Decision log

- Kept the per-sample fits as they are (balanced-loss fits): their core accuracy (4.5 % peak error) is good, so they are a
  valid pretraining target for both losses.

## Results of the A/B (2026-10-12/13)

Same split (427 / 107), seed 43, same pretrain weights, early stopping on the run's own validation loss. Validation-set medians
(`eval_runs.py`); `current_best` is the user's 2026-10-12 18:47 run (balanced config, unseeded pretrain) for reference.

| run | balanced loss | legacy loss | abs. peak err | peak ratio p16-p84 | EE(r<=2) err | profile max err 485/722/928 nm | rel. bias 20-40 px |
|---|---|---|---|---|---|---|---|
| legacy | 0.380 | 0.362 | 19.1 % | 0.72-1.31 | 14.1 % | 17.9 / 13.7 / 13.1 % | -4.1 % |
| balanced (current config) | 0.350 | 0.360 | 18.7 % | 0.71-1.26 | 12.8 % | 20.5 / 13.7 / 12.7 % | -4.6 % |
| balanced_d1 (Huber delta 1, scale 20) | 0.362 | 0.359 | 18.8 % | 0.73-1.32 | 12.7 % | 21.6 / 15.8 / 13.8 % | -5.6 % |
| balanced_mix (+ 1.0 x legacy) | 0.364 | 0.360 | 19.5 % | 0.71-1.32 | 13.1 % | 22.2 / 14.0 / 13.4 % | -2.7 % |
| current_best (balanced, other RNG) | 0.365 | 0.378 | 18.7 % | 0.76-1.41 | 11.5 % | 19.9 / 15.3 / 12.1 % | -8.0 % |
| pretrain only | 0.444 | 0.504 | 27.4 % | 0.83-1.85 | 15.7 % | 27.7 / 23.4 / 21.6 % | -14.2 % |
| per-sample fits (oracle) | 0.126 | 0.167 | 4.5 % | 0.91-1.04 | 1.6 % | 3.5 / 2.0 / 3.9 % | -10.7 % |

- On the same data, the balanced and the legacy loss give calibrators of the same quality: the differences between the two
  (and between the balanced variants) are within the run-to-run scatter of the same config (`balanced` vs `current_best`:
  median peak ratio at 485 nm 0.96 vs 1.06). The balanced loss is even marginally better in the core (|peak err|, EE2) and
  equal in the halo; the legacy loss has the smallest profile error at 485 nm.
- The perceived degradation "balanced much worse than legacy" is therefore not caused by the loss. What changed between the
  Sep-30 legacy calibrators and the Oct-12 balanced one besides the loss: the dataset cache (534 samples, time-grouped split
  instead of a random one, so the validation set is harder: no near-duplicates of training samples), the fits (redone with
  the balanced loss, `dn` re-normalized) and the PSF model itself (PSD terms streamlined, 2026-10-10/11). The Sep-30 bundle
  cannot be re-validated on the current model (60-80 % too bright core), so a direct comparison with the old numbers is not
  possible; the old numbers were obtained on a random split that leaks acquisition-time neighbours into the validation set.
- The gap that matters is calibrator vs. fits (19 % vs 4.5 % peak error, loss 0.35 vs 0.13): the telemetry -> parameters
  mapping, not the pixel loss. Neither a quadratic asinh loss nor a legacy mix helps; they are not adopted.

## Outcomes & retrospective

- Kept the balanced loss as the default (`NFM_calibrator_config.json` unchanged apart from `seed` / `reuse_pretrain`).
- Fixed: NaN gradients from `abs()` of the zero-underflowed OTF (`TipTorch.py`), non-finite-gradient guard in
  `NFMCalibratorTrainer.train`, preprocessor loading in `validate_NFM_calibrator.py`, `--device` / `seed` / `reuse_pretrain` in
  `train_NFM_calibrator.py`, bundle export moves tensors to CPU and stores `loss_cfg`, warning when the dataset cache has no
  preprocessor. New test `tests/test_NFM_calibrator_bundle.py`.
- A/B artefacts: `data/weights/NFM_calibrator_lossexp/{legacy,balanced,balanced_d1,balanced_mix}/` (checkpoints, bundles,
  meta, shared `pretrain_weights.pth`), evaluation script `eval_runs.py` in the session scratchpad.
- Lesson: a training run needs ~50 GB of GPU memory and `resolve_device` silently falls back to the CPU (or to `cuda:0`
  under `CUDA_VISIBLE_DEVICES`), check the device table in the log after launching.

## Re-balancing for the peak (2026-10-19, STD-star fits)

**Observation (user):** the full refits of the dataset with the balanced loss (`fitted_balanced`) fit the peaks worse than the legacy
MSE + MAE loss (`fitted_legacy`), while the legacy fits are only marginally worse on the overall shape. Both sets are raw-frame fits with
identical regularizers; only the PSF loss differs.

**Diagnosis** (`datasets/MUSE_STD_dataset/compare_fit_losses.py`, outputs `data/analysis/NFM_STD_loss/`): over 534 stars the balanced
fits have a median |peak err| of 6.3 % (signed +4.4 %: model too faint) and a FWHM error of 7.5 % (model 6 % too wide), the legacy fits 3.8 %
and 4.2 %; the balanced fits are better in the halo (10-40 px flux error 1.0-1.8 % vs 3.4-4.2 %) and in the 0-1.5 px and 10-40 px shape errors.
The degradation grows with Strehl (SR quartiles: -0.1, +1.6, +2.8, +4.6 % signed peak error) and towards the red (846-900 nm: 10-11 % vs 5-6 %).
Cause: the balanced loss compares every pixel in log space, so the peak pixel weighs as much as any other core pixel (the central 9 px hold
16 % of the loss, the peak alone ~4 %) and a 5-10 % peak error costs ~1-5 % of the loss; in return the fit matches the faint pixels flanking the
core, where the legacy fit is 25-80 % off in log terms (46 % of the balanced-loss advantage of the balanced fit over the legacy fit is earned
inside r < 1.5 px). The balanced fit buys this with a blurred core (higher jitter / dn).

**Variants refitted on a 30-sample subset** (every 18th ID, `data/analysis/NFM_STD_loss/loss_variants_30_samples.csv`, medians of the
per-sample medians, |peak err| / red-bin |peak err| / FWHM err / reference balanced loss / 10-20 px shape err):

| variant | |peak| % | red bins % | FWHM % | balanced loss | shape 10-20 px % |
|---|---|---|---|---|---|
| balanced (as is) | 5.7 | 11.9 | 7.3 | 0.12 | 5.4 |
| legacy | 3.7 | 6.3 | 2.3 | 0.17 | 6.6 |
| huber_delta 0.3 / 1.0 | 6.6 / 6.1 | 11.9 / 12.0 | 7.0 / 7.6 | 0.12 / 0.13 | 5.4 / 5.6 |
| radial_power 3 | 5.2 | 10.2 | 4.2 | 0.14 | 6.3 |
| core weights x5 (r <= 1.5 px) | 5.6 | 11.1 | 4.8 | 0.14 | 5.7 |
| balanced + 0.3 x legacy MSE | 5.5 | 8.9 | 6.0 | 0.12 | 5.4 |
| balanced + 1 x legacy MSE | 4.3 | 6.8 | 4.6 | 0.12 | 5.4 |
| balanced + 3 x legacy MSE | 4.7 | 6.6 | 4.1 | 0.13 | 5.6 |
| balanced + 10 x legacy MSE | 4.4 | 6.5 | 2.9 | 0.15 | 5.7 |

Re-weighting inside the balanced loss (Huber delta, radial power, core boost) does not recover the peak: in log space the single peak pixel
can never dominate. Adding the linear legacy term (dominated by the brightest pixels) does: with weight 1 the peak error drops from 5.7 % to
4.3 % (red bins 11.9 % -> 6.8 %), the balanced loss value and the halo errors stay exactly at the balanced fit's, and only 1 of 30 samples gets
worse by > 1 %. Weights 3-10 move the fits towards the legacy ones (balanced loss 0.13-0.15, halo bias grows) for no further peak gain.

**Implemented:** `BalancedPSFLoss(mse_weight=...)` in `src/tiptorch/tools/utils.py` (default 0.0 = unchanged): a linear term
`mse_weight * mean((model - data)^2)` per wavelength added to the asinh loss; `PSF_LOSS_CFG["mse_weight"]` in `STD_fitter.py` and the same key in
the calibrator loss config (forwarded to the class with the other parameters). 1.8e7 = `w_MSE` x 2e4, the MSE term of the legacy loss.
A first implementation as `legacy_weight` (the full legacy MSE + MAE term, mirroring the calibrator's existing knob) was tested through the
real fitter path and rejected: with the MAE part the halo goes back to the legacy solution (10-20 px shape error 6.6 %, 20-40 px bias -4.0 %,
legacy loss value 0.14) for a smaller peak gain (5.3 %); the MAE is dominated by the ~40 000 wing pixels. Confirmation run of the final code
path (`mse_weight = 1.8e7`, same 30 samples): median |peak err| 4.3 % (balanced 5.7 %, legacy 3.7 %), red bins 6.9 % (11.9 / 6.3 %), FWHM
error 5.0 % (7.3 / 2.3 %), balanced loss 0.12 (= balanced fit), 10-20 px shape error 5.4 % (5.4 / 6.6 %), 20-40 px halo bias -2.0 %
(-1.8 / -4.2 %), 1 of 30 samples worse than the balanced fit by > 1 %. Identical to the prototype within 0.1 %.

**Decision (user, 2026-10-19):** adopted. `mse_weight = 1.8e7` is now the default of `PSF_LOSS_CFG` in `STD_fitter.py` and of the loss section of
`NFM_calibrator_config.json` / `NFM_factor_calibrator_config.json` (the ETC config was left at 0: untested in that regime). The existing calibrator
`legacy_weight` (MSE + MAE) is not the same thing and stays 0. The fit sets `fitted_balanced` / `fitted_legacy` predate the change; the cached cubes
are unaffected, only a refit (and a dataset-cache rebuild if the training should use the new fits) follows from it.
