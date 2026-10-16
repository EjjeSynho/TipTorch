# New smarter loss function

This ExecPlan follows `agents/PLANS.md`. Started 2026-10-14.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Context

As you can see in `./datasets/MUSE_STD_dataset/create_STD_dataset.py`, all NFM PSFs are manually labeled with different flags. Here is what they mean:

- "Bad," "medium," or "high quality" refers to how blurry a PSF appears or, in other words, how strongly it is affected by HO and TT errors. When two flags are checked simultaneously, the PSF's quality lies somewhere in between.

- "Corrupted" speaks for itself: it means that the PSF must be discarded from the training set.

- "Non-point source" indicates that the core of the PSF looks distorted or asymmetric, which can be caused by the "phase-bump" defect or by quasi-static NCPAs.

- "Low AO error" means that the region of the PSF inside the AO halo is quite deep, indicating good AO correction. Meanwhile, the region outside the corrected radius can exhibit a strong halo.

- "Has streaks" means that the TT error is low enough not to blur the diffractive streaks of the PSF.

- Finally, the very important one is the "Tomographic error" flag, which indicates the presence of a cross-like pattern that appears when higher $C_n^2$ layers have greater weight.

You can find these label flags inside `muse_df.pickle`.

# Goal

Your goal is to design a new smart loss function and use it to train the update-factor MUSE NFM calibrator. It can take the `BalancedLoss` as a starting point, but it **must** use the information from the labels. Of course, some flag assignments are subjective, so they should influence the loss function without fully defining its regimes.

# Instructions

Given this description, this has the following implications for the smart loss function:

- PSFs with the "Non-point source" flag enabled must try to explain asymmetry using the phase-bump error contributor (the 0th LO NCPAs term).

- Low-quality PSFs are less peak-dominated, so the loss might weight the PSF more uniformly in this case.

- The "Has streaks" flag implies that the TT contribution should be low for that sample.

- The "Tomographic error" flag should cause the loss to emphasize the cross-like feature outside the PSF core, but before the halo cut-off.


## Purpose / Big Picture

After this change, the factor calibrator (`NFMFactorCalibratorTrainer`, `machine_learning/calibrators/NFM_calibrator.py`) can be trained with
a label-aware PSF loss (`"type": "label_aware"` in the `loss` section of a training config) that reads the manual flags of every
training PSF and uses them as soft hints: they change pixel weights and parameter priors, never the target (the data). Observable
outcome: `calibrators/train_NFM_calibrator.py --config calibrators/configs/NFM_factor_calibrator_labels_config.json` trains, and
`analysis/compare_NFM_calibrators.py --labels` reports the usual core / halo metrics plus per-label metrics (core asymmetry
residual of the non-point PSFs, cross residual of the tomographic PSFs, predicted jitter of the streak PSFs) next to the
baseline factor calibrator trained with the balanced loss.

## Current State

- PSF loss: `BalancedPSFLoss` (`src/tiptorch/tools/utils.py`): per pixel `e = asinh(pred / s) - asinh(data / s)`,
  `s = asinh_k * sigma_noise[sample, λ]`, Huber penalty, radial weights `max(r, r_min)^(-radial_power)` (sum 1) cut at `r_max`
  (40 px in the configs); returns `[B]` (sum over pixels, mean over λ). Tensors: torch float32 on the training device,
  PSF cubes `[B, N_λ, 199, 199]`, sum-normalized, background-subtracted.
- Trainer: `NFMCalibratorTrainer._loss_per_sample(PSF_data, PSF_pred, x_dict_pred)` -> `[B]`; `loss_fn` adds `reg`
  (`_regularization(x_pred, x_dict, fit_b, tel_b)`, parameter-space terms). No sample ids reach the loss.
- Phase bump: `trainer.phase_bump` [N_dataset] = fitted `LO_coefs[:, 0]`, fixed (no gradient) and re-attached in `run_model`;
  the bundles predict bump = 0 (the defect has been fixed at the instrument level since).
- Factor calibrator config: `calibrators/configs/NFM_factor_calibrator_config.json` (balanced loss, `r_max` 40, scale 3,
  seed 43, time-grouped split 427 / 107). Baseline run: `data/weights/NFM_factor_calibrator/factor_calibrator_checkpoint.pth`
  (best val loss 0.3430, median |peak err| 15.5 %), bit-reproduced by the 2026-10-14 10:26 run.
- Labels: boolean columns of `muse_df.pickle` (`STD_FOLDER`): `Good quality`, `Medium quality`, `Bad quality`, `Corrupted`,
  `Non-point`, `Has streaks`, `Low AO errs`, `Tomography error`. The dataset cache (`dataset_cache/muse_STD_stars_dataset.pt`)
  drops `Corrupted` and `Bad quality` samples and does not store the labels.

## Target Behavior

New module `machine_learning/calibrators/NFM_label_loss.py`:

    load_PSF_labels(dataset) -> pd.DataFrame [N_dataset, flags], aligned with dataset.sample_ids
    label_scores(labels, PSF_cubes, confidence, quality_data_weight, device) -> dict of float tensors [N_dataset] in [0, 1]:
        blur, non_point, streaks, tomography
    class LabelAwarePSFLoss(BalancedPSFLoss):
        __call__(PSF_data, PSF_pred, scores=None, lambda_ids=None) -> [B]
        (scores = None or all scores 0: identical to BalancedPSFLoss)

Per sample `i` (soft scores `s ∈ [0, 1]`), with `e` the asinh residual of `BalancedPSFLoss` and `ρ` its Huber penalty:

    L_i = mean_λ [ Σ_x W_i(r) ρ(e)                                         quality-adaptive radial weights
                 + odd_weight   * s_np_i   * Σ_x W_odd(r)    ρ(odd(e))       core asymmetry (r <= odd_r_max)
                 + cross_weight * s_tomo_i * Σ_x W_cross(x,λ) ρ(even(e) - <e>_θ) ]   cross between the core and the AO radius
          + J prior (streaks) + bump prior (non-point), in NFMCalibratorTrainer._regularization

- Soft scores: a flag is trusted with probability `confidence` (flagged: c, unflagged: 1 - c). Blurriness mixes the quality label
  (good 0, good+medium 0.5, medium 1) with the data (rank of the core concentration EE(r<=2) / EE(r<=40) over the dataset).
- Quality: `W_i ∝ max(r, r_min_i)^(-p_i)`, `p_i`, `r_min_i` interpolated between (`radial_power`, `r_min`) and
  (`radial_power_blurry`, `r_min_blurry`) by the blurriness; normalized per sample.
- Non-point: the phase bump becomes a per-sample latent trained with the network (`trainer.optimize_phase_bump`), initialized at
  the fitted value, clamped at >= 0, with the prior `bump_prior_weight * (1 - s_np) * (bump / bump_prior_sigma)^2` (pulled to 0
  unless the PSF is flagged); the odd (point-antisymmetric) part of the residual in the core is emphasized for flagged PSFs, so the
  bump is fitted to the asymmetry and not to the symmetric blur. Validation samples keep their fitted bump; bundles still predict 0.
- Streaks: `streak_J_weight * relu(slope + streak_J_margin)^2`, `slope` the regression slope of the predicted jitter (mean of the
  spline nodes, in units of the training spread of the fits) on the streak score within the batch: the streak PSFs must have a lower
  jitter than the others (by `margin * (2c - 1)` spreads), without any force on the jitter level (invariant to a global shift).
- Tomography: annulus `cross_r_min <= r <= r_AO(λ) = λ / (2 * DM_pitch)` (in px), angular window
  `((1 + cos 4(θ - cross_angle)) / 2)^cross_sharpness` (arms on the LGS azimuths 0/90/180/270° of the derotated frame), applied to the
  point-symmetric, azimuthal-mean-subtracted residual (the PSD model can only produce centro-symmetric halo structure).

## Non-Goals

- No change of the PSF model, of the fits or of the dataset cache content (labels are read from `muse_df.pickle`; the cache builder
  only gains an optional `labels` entry for future caches).
- No inference-time use of the labels or of the trained bumps: bundles keep `predict_phase_bump: False`.
- The `Low AO errs` flag is loaded but not used (no instruction for it, see Decision Log).

## Progress

- [x] (2026-10-14 11:45 UTC) Read the code (trainer, balanced loss, dataset builder, configs) and the label statistics.
- [x] (2026-10-14 11:50 UTC) Data / model diagnostics of the flags (see Surprises & Discoveries).
- [x] (2026-10-14 12:00 UTC) Implemented `NFM_label_loss.py`, trainer hooks (ids / λ ids into the loss, learnable bump, label priors),
      `NFM_factor_calibrator_labels_config.json`, labels in `build_dataset_cache`, `--labels` / `--zero-bump` in `compare_NFM_calibrators.py`.
- [x] (2026-10-14 12:00 UTC) `tests/test_NFM_label_loss.py`: 17/17 pass (incl. the dataset labels); one-batch training step on the GPU: gradients
      reach only the batch bumps, the checkpoint stores them, the validation entries stay at their fits, peak memory 11.8 GB at batch 4.
- [x] (2026-10-14 12:02 UTC) Term weights calibrated on the baseline calibrator (training set), see Decision Log.
- [x] (2026-10-14 12:25 UTC) Full label-aware run (`NFM_factor_calibrator_labels/`, seed 43, same split, pretraining bit-identical to
      the baseline's): worse than the baseline, 20.5 % vs 15.5 % median |peak err| (table in Artifacts).
- [x] (2026-10-14 12:28 UTC) Comparison with the baseline (`compare_NFM_calibrators.py --labels [--zero-bump]`). Queued ablations
      (bump fixed, labels shuffled) cancelled: pointless while the full loss is worse.
- [x] One-component runs on top of the balanced loss (`NFM_calibrator_labelexp/comp_{streak,nonpoint,tomo,quality}/`, configs in the
      session scratchpad): streak prior redesigned (jitter slope), non-point with a 5x stronger bump prior. Partially: stopped on
      request at 13:15 UTC (a second agent session had been running in parallel on the same files), to be redone. Done, not yet
      evaluated: `comp_streak`, `comp_nonpoint`. Killed mid-run (incomplete, delete before redoing): `comp_tomo`, `labels_nobump`.
      Not started: `comp_quality`. The scratchpad configs are session-local: regenerate them from
      `NFM_factor_calibrator_labels_config.json` (one component on, the others at their neutral values).
      Addendum (2026-10-14 13:30 UTC, the session that ran these jobs): not knowing about the stop, it deleted the killed `comp_tomo/`,
      restarted `comp_tomo` at 13:17 (queue: `comp_quality`, then the baseline config with seeds 44 / 45 for the run-to-run scatter)
      and stopped everything again at ~13:28 once it found the note above. `comp_tomo/` now holds that second incomplete attempt
      (~epoch 12, delete before redoing). `comp_streak/comp_streak_checkpoint_backup.pth` is the diverged linear-tilt attempt (discard).
      `comp_streak` and `comp_nonpoint` have been evaluated (Artifacts): both worse than the baseline. GPUs left idle.
      Redone (2026-10-14 13:25-14:32 UTC, one session, sequential on GPU 0, each run capped at 1 h, `queue_redo.sh`): the incomplete
      `comp_tomo/` and `labels_nobump/` deleted, then `comp_tomo`, `comp_quality` and the baseline config with seeds 44 / 45
      (`NFM_calibrator_labelexp/baseline_s4{4,5}/`); all exited cleanly.
- [x] (2026-10-14 14:55 UTC) Final evaluation of the 8 calibrators, both bump protocols (`eval_all.sh` -> `cmp_final_{fitted,zero}.json`,
      `summarize.py`), table in Artifacts.
- [x] (2026-10-14 15:00 UTC) `Outcomes & Retrospective`.

## Surprises & Discoveries

- Observation: the flags agree with the per-sample fits, i.e. they carry real information.
  Evidence (534 cached samples, medians): fitted bump 86 (Non-point) vs 20; upper-layer Cn2 weight 0.057 (Tomography error) vs
  0.028 and dn 42 vs 28; dn 24 / 35 / 52 for good / good+medium / medium quality; core concentration EE(r<=2) / EE(r<=40) vs
  quality label: Spearman 0.60. Has streaks: fitted jitter only slightly lower (14.9 vs 16.0 mas at the middle node), i.e. the fits do
  not use the information that the label carries.
  Consequence: the jitter prior of the streak PSFs adds information the fits do not have; the quality label and the data agree well
  enough to blend them.
- Observation: "Bad quality" and "Corrupted" PSFs are already excluded from the cache: the quality levels present are good (274),
  good + medium (182), medium (77) and one sample with no quality flag.
  Consequence: the blurriness score spans good -> medium only.
- Observation: the raw point-asymmetry of the data core is dominated by the sub-pixel decentering (~0.55 for every class), not by the
  flag. The odd part of the residual (model shifted by the fitted dx / dy vs data) is what carries the asymmetry.
  Evidence: `|P - rot180(P)| / |P + rot180(P)|` within 4 px, medians 0.57 (Non-point) vs 0.57; odd residual of the fitted model
  0.03-0.05 at r = 1-5 px with the fitted bump vs 0.05-0.08 with bump = 0; the bump's own odd signature fades beyond ~7 px.
  Consequence: the asymmetry term compares odd parts of model and data, within `odd_r_max` = 8 px.
- Observation: the data have a strong top-bottom dipole (odd) halo within ~30 px, strongest for the tomographic PSFs, which no
  PSD term can produce (a PSD is centro-symmetric). The even anisotropy shows the "+" cross along the image axes (LGS azimuths
  0/90/180/270° in the derotated frame, `sources_HO` of the configs) and the spider spikes along the diagonals (streak PSFs).
  Evidence: mean azimuthal-mean-subtracted asinh maps per class (scratch `label_diag.py`, `tomo_indiv.py`).
  Consequence: the tomography term uses the even part of the residual and an angular window on the axes, which also keeps the
  diagonal spider spikes out of it.

## Decision Log

- Decision: streak prior = hinge on the batch slope of the jitter vs the streak score (`streak_J_weight` 0.2, `streak_J_margin` 0.5).
  Rationale: the first form (hinge on the jitter of the streak PSFs at the median fitted jitter of the flagged training PSFs) lowered
  the predicted jitter of all PSFs (17.8 / 18.0 -> 15.6 / 15.6 mas for streak / other validation PSFs): the telemetry barely tells the
  streak PSFs apart, so the network shifted the level, and the cores came out 11-17 % too bright. A linear zero-mean tilt
  (centred scores x jitter) diverged within two epochs (val loss -1.29): unbounded, the PSF loss does not hold the jitter. The slope
  hinge is bounded and invariant to a global jitter shift (gradient sums to zero over the batch).
  Date/Author: 2026-10-14, Claude.

- Decision: labels act as soft weights / priors with a trust level (`confidence`), blurriness additionally mixed with a data-driven
  score.
  Rationale: the flags are subjective; with this form the expected loss is the mixture of the "flag right" and "flag wrong" regimes,
  and the data always remain the target.
  Date/Author: 2026-10-14, Claude.
- Decision: the phase bump becomes a per-sample training latent pulled to 0 unless flagged non-point, instead of the fixed fit.
  Rationale: the network never sees the bump (bundles predict 0), so a bump that absorbs symmetric blur of a point-like PSF during
  training biases the network towards too sharp PSFs at inference; for the non-point PSFs the bump must explain the asymmetry.
  Date/Author: 2026-10-14, Claude.
- Decision: azimuthal mean of the cross term = least-squares piecewise-linear profile in r, not integer radius bins.
  Rationale: with bins, the axis pixels sit at the bin centres and the diagonal ones do not, so a steep radial residual looked like a
  cross (test: 0.081 for a purely radial residual vs 0.27 for a real '+' error; with the piecewise-linear fit 0.010).
  Date/Author: 2026-10-14, Claude.
- Decision: term weights odd 0.5 (r <= 6 px), cross 2.0, streak J 0.1 (hinge at the median fitted jitter of the flagged training
  PSFs, 14.3-16.7 mas over the 5 nodes, in units of the training spread 3.6-4.7 mas), bump prior 0.02 with sigma 50, confidence 0.8,
  quality data weight 0.5, blurry radial power 1 / r_min 2 px, bump learning rate 0.5.
  Rationale (baseline factor calibrator on the training set, medians): base term 0.25; odd term 0.16 (0.18 non-point), cross 0.057
  (0.047 tomographic): with these weights a flagged PSF gets +25-35 % of its base loss from its label term, an unflagged one +5-10 %.
  The baseline predicts J ~ 17.9 mas for streak and other PSFs alike (fits: 15.5), the streak prior is ~ +15 % at that level.
  Bump: setting it to 0 raises the base loss of the non-point PSFs by 0.072 (median, p84 0.17) and of the others by 0.000 (p84 0.03),
  so the prior (0.016 (b / 50)^2 unflagged, 0.004 (b / 50)^2 flagged) removes the bumps that do not pay and keeps those of the
  flagged PSFs.
  Date/Author: 2026-10-14, Claude.
- Decision: phase bump lower bound min(fit, 0) instead of 0.
  Rationale: 104 fits are slightly negative (min -9.9); a plain clamp would also move the validation entries.
  Date/Author: 2026-10-14, Claude.
- Decision: `Low AO errs` unused.
  Rationale: no instruction; its information (deep AO-corrected region) overlaps with the quality flags (dn 24 vs 36).
  Date/Author: 2026-10-14, Claude.

## Plan of Work

1. `NFM_label_loss.py` (labels, scores, loss). The loss subclasses `BalancedPSFLoss` and reduces to it for zero scores.
2. `NFM_calibrator.py`: `loss_fn` / `_loss_per_sample` / `_regularization` take the batch ids and λ ids; the trainer builds the label
   scores and the loss for `type = label_aware`, makes the bump trainable (`optimize_phase_bump`), adds the streak / bump priors,
   stores the bumps in the checkpoints. `NFMDataset` reads an optional `labels` entry; `build_dataset_cache` writes it.
3. `calibrators/configs/NFM_factor_calibrator_labels_config.json`; `train_NFM_calibrator.py` passes `optimize_phase_bump`.
4. Checks (`tests/test_NFM_label_loss.py`), weight calibration on the baseline, training runs, comparison.

## Concrete Steps

Run from the repository root with the `TipTorch` environment (`LD_LIBRARY_PATH=$CONDA_PREFIX/envs/TipTorch/lib` for matplotlib):

    python tests/test_NFM_label_loss.py --device cuda:0
    python machine_learning/calibrators/train_NFM_calibrator.py --config machine_learning/calibrators/configs/NFM_factor_calibrator_labels_config.json --device cuda:0
    python machine_learning/analysis/compare_NFM_calibrators.py --config machine_learning/calibrators/configs/NFM_factor_calibrator_config.json --labels --device cuda:0 \
        baseline=<.../NFM_factor_calibrator/factor_calibrator_checkpoint.pth> labels=<.../factor_calibrator_labels_checkpoint.pth>

## Validation and Acceptance

- Zero scores: `LabelAwarePSFLoss` equals `BalancedPSFLoss` to float precision; all terms finite; gradients reach the bump.
- The label-aware run trains without NaN, and on the validation set it is not worse than the baseline on the core / halo metrics
  (median |peak err|, EE, profile errors) while improving the label-specific metrics on the flagged subsets.

## Idempotence and Recovery

The default configs are unchanged (`balanced` loss): the label-aware loss is opt-in via `"type": "label_aware"`. Runs go to their own
weights folder (`weights_subdir`), the baseline weights are not touched.

## Interfaces and Dependencies

No new dependencies. Config keys of the `loss` section (type `label_aware`): `confidence`, `quality_data_weight`,
`radial_power_blurry`, `r_min_blurry`, `odd_weight`, `odd_r_max`, `cross_weight`, `cross_r_min`, `cross_angle`, `cross_sharpness`,
`streak_J_weight`, `streak_J_margin`, `bump_prior_weight`, `bump_prior_sigma`; trainer section: `optimize_phase_bump`, `phase_bump_lr`.

## Documentation Updates

Docstrings of the new module / loss, comments in the config.

## Artifacts and Notes

Validation set (107), `compare_NFM_calibrators.py --config NFM_factor_calibrator_config.json --labels`, validation bumps = fits;
`balanced loss` is the plain BalancedPSFLoss for every run (comparable), not the training loss:

| metric | baseline (balanced, factor) | label-aware v1 | fits (oracle) |
|---|---|---|---|
| balanced loss | 0.343 | 0.355 | 0.126 |
| median abs. peak err | 15.5 % | 20.5 % | 4.5 % |
| peak ratio p16-p84 | 0.77-1.27 | 0.83-1.50 | 0.91-1.04 |
| peak ratio 485 / 722 / 928 nm | 1.04 / 1.00 / 1.01 | 1.11 / 1.14 / 1.17 | 0.97 / 1.00 / 1.00 |
| EE2 abs. err | 11.0 % | 12.2 % | 1.6 % |
| profile max err 485 / 722 / 928 nm | 16.3 / 10.6 / 10.3 % | 19.0 / 16.4 / 16.8 % | 3.5 / 2.0 / 3.9 % |
| rel. bias 20-40 px | -6.2 % | -3.9 % | -10.7 % |
| abs. peak err: non-point (22) / point (85) | 18.8 / 15.1 % | 19.4 / 20.5 % | 4.2 / 4.5 % |
| abs. peak err: tomography (11) / streaks (46) | 13.0 / 15.2 % | 29.2 / 15.5 % | 4.1 / 4.6 % |
| abs. peak err: good (50) / good+med (39) / medium (18) | 15.6 / 15.8 / 14.9 % | 16.7 / 24.7 / 24.6 % | 4.8 / 4.6 / 2.9 % |
| odd term non-point / point | 0.167 / 0.200 | 0.177 / 0.209 | 0.053 / 0.051 |
| cross term tomography / other | 0.046 / 0.051 | 0.039 / 0.052 | 0.032 / 0.039 |
| predicted jitter streaks / other [mas] | 17.8 / 18.0 | 15.6 / 15.6 | 14.5 / 15.7 (fits) |

With the validation bumps set to 0 (`--zero-bump`, what the bundles do): baseline 16.0 %, label-aware v1 24.1 %, fits 7.1 %.
Learned training bumps (v1, prior 0.02): unflagged median 22.3 -> 22.0, flagged 85.7 -> 82.0, i.e. the prior hardly acted.

One-component runs (balanced loss + one label term, seed 43, same split; `cmp_comps1_{fitted,zero}.json` in that session's scratchpad):

| run | median abs. peak err | peak ratio 485 / 722 / 928 nm | abs. peak err non-point / tomography / streaks | jitter streaks / other [mas] | zero bump |
|---|---|---|---|---|---|
| baseline | 15.5 % | 1.04 / 1.00 / 1.01 | 18.8 / 13.0 / 15.2 % | 17.8 / 18.0 | 16.0 % |
| + streak slope prior (0.2, margin 0.5) | 17.4 % | 0.90 / 0.95 / 0.96 | 21.7 / 11.4 / 15.8 % | 18.7 / 20.3 | 16.8 % |
| + non-point (odd 0.5, learnable bump, prior 0.1) | 17.5 % | 1.03 / 0.99 / 0.98 | 23.6 / 17.3 / 17.5 % | 18.4 / 18.4 | 17.6 % |

The streak prior separates the jitter (1.6 mas) by raising the jitter of the other PSFs more than it lowers the streak ones: cores
4-10 % too faint. The 2-point gaps are about twice the only known run-to-run scatter (~1 point, direct calibrator); the scatter of
the factor baseline itself is unmeasured (seeds 44 / 45 were queued but not run).

Final comparison (2026-10-14 14:55 UTC), validation bumps = fits. The baseline is now the balanced factor config with three seeds
(s43 = the official `NFM_factor_calibrator/factor_calibrator_checkpoint.pth`, s44, s45); the label runs all use seed 43. `+` / `-`:
outside the seed range by more than its half-width.

| metric | baseline seeds min / mean / max | + streak | + non-point | + tomo | + quality | all v1 | fits |
|---|---|---|---|---|---|---|---|
| balanced loss | 0.343 / 0.351 / 0.358 | 0.354 | 0.348 | 0.340 | 0.346 | 0.355 | 0.126 |
| median abs. peak err | 15.5 / 17.4 / 18.8 % | 17.4 | 17.5 | 17.0 | 17.2 | 20.5 + | 4.5 |
| EE2 abs. err | 11.0 / 11.9 / 12.7 % | 11.2 | 11.3 | 11.0 | 12.2 | 12.2 | 1.6 |
| EE8 abs. err | 5.2 / 6.2 / 7.0 % | 5.4 | 5.7 | 5.6 | 5.5 | 5.2 | 0.8 |
| rel. bias 20-40 px | -6.2 / -5.3 / -4.4 % | -3.5 + | -6.3 | -6.6 | -3.5 + | -3.9 | -10.7 |
| peak ratio 485 / 928 nm | 0.97-1.04 / 0.99-1.02 | 0.90 - / 0.96 - | 1.03 / 0.98 - | 1.02 / 1.04 + | 1.02 / 1.00 | 1.11 + / 1.17 + | 0.97 / 1.00 |
| profile max err 485 nm | 14.2 / 15.4 / 16.3 % | 18.4 + | 15.2 | 17.9 + | 15.5 | 19.0 + | 3.5 |
| abs. peak err non-point (22) | 18.8 / 20.3 / 21.5 % | 21.7 | 23.6 + | 23.9 + | 20.4 | 19.4 | 4.2 |
| abs. peak err tomography (11) | 13.0 / 16.6 / 20.3 % | 11.4 | 17.3 | 15.9 | 16.7 | 29.2 + | 4.1 |
| abs. peak err medium (18) | 14.9 / 16.6 / 17.8 % | 14.8 | 10.4 - | 20.6 + | 11.9 - | 24.6 + | 2.9 |
| abs. peak err good+medium (39) | 15.2 / 16.1 / 17.1 % | 17.8 | 18.6 + | 16.0 | 18.6 + | 24.7 + | 4.6 |
| cross term tomography | 0.045 / 0.046 / 0.046 | 0.048 + | 0.046 | 0.044 - | 0.045 | 0.039 - | 0.032 |
| jitter streaks / other [mas] | 17.6-17.8 / 17.5-18.0 | 18.7 / 20.3 | 18.4 / 18.4 | 18.1 / 17.8 | 18.1 / 18.3 | 15.6 / 15.6 | 14.5 / 15.7 |

With the validation bumps set to 0 the picture is the same: baseline seeds 16.0 / 17.3 / 18.6 %, streak 16.8, non-point 17.6,
tomo 18.1, quality 17.2, all v1 24.1 %, fits 7.1 %. Per-subset numbers are medians over 11-50 PSFs and scatter by several points
between seeds (tomography: 13.0-20.3 %, medium 14.9-17.8 %).

## Outcomes & Retrospective

- Implemented: the label-aware loss (`NFM_label_loss.py`, opt-in `"type": "label_aware"`), the learnable per-sample phase bump, the
  streak / bump priors, the per-label metrics and the zero-bump protocol of `compare_NFM_calibrators.py`, `tests/test_NFM_label_loss.py`
  (17 / 17). The default configs and the official weights are unchanged.
- Result: the smart loss does not make the factor calibrator more accurate. Every single label term lands at 17.0-17.5 % median
  |peak err|, inside the seed range of the balanced baseline (15.5-18.8 %, mean 17.4 %), and the full v1 loss (20.5 %) is clearly worse
  (quality weighting + the first, absolute jitter prior: cores 11-17 % too bright). The terms do act on what they target: the cross term
  lowers the cross error of the tomographic PSFs (0.044 vs 0.045-0.046), the quality weighting and the non-point term improve the
  medium-quality PSFs (11.9 / 10.4 % vs 14.9-17.8 %) and the quality / streak terms the halo at 20-40 px (-3.5 % vs -4.4 to -6.2 %),
  but each time at the expense of other subsets (good + medium PSFs, non-point PSFs, blue core), so the total does not move.
- The reference 15.5 % of the official factor calibrator is the best of three seeds; a fair baseline is 17.4 % (range 15.5-18.8 %).
  Comparisons of losses on this split need several seeds per variant: a 2-point difference from one seed is noise.
- Why the labels do not help: they change which errors the loss emphasizes, but they cannot add information to the telemetry -> parameters
  map (the labels are not available at inference, and the flagged properties are barely predictable from the telemetry: the streak prior
  could only separate the predicted jitter by shifting the other PSFs). The calibrator error is dominated by the unpredictable parameters
  (NCPAs, Cn2), see `factor-calibrator-execplan.md`.
- The bump mechanism did not remove the train / inference mismatch: with the bumps of the validation PSFs at 0, the non-point run
  is not better than the baseline (17.6 % vs 16.0-18.6 %); at prior weight 0.02 the learned bumps stayed at the fits.
- Follow-ups if pursued: (1) several seeds per variant before any decision (or a seed ensemble of the baseline, which may itself beat the
  2-3-point seed scatter); (2) a milder quality weighting (the only term with a consistent gain on the medium PSFs and the halo) tested
  over seeds; (3) the labels as targets of an auxiliary head or as sample weights rather than as loss shape.
- Process lesson: a second agent session worked on the same files in parallel for ~1 h (killed jobs, rewrote the streak prior); check
  `ps` for foreign training / queue processes before launching GPU jobs, and wait on PIDs, not on `pgrep -f` patterns (a waiter of an
  older session matched its own command line and looped for 23 h).

## Revert and follow-up: phase bump suppressed in the fits (2026-10-14 15:45 UTC)

On request, all the code of the label-aware loss was reverted (`git checkout` of `NFM_calibrator.py`, `utils.py`, `NFM_validation_utils.py`,
`compare_NFM_calibrators.py`, `train_NFM_calibrator.py`, `STD_dataset_utils.py`; `NFM_label_loss.py`, its config and test deleted). A patch
of that state is in the session scratchpad (`smart_loss_tracked.patch`, `smart_loss_untracked.tgz`); this plan keeps the design and results.
The weights of the runs stay in `weights/NFM_factor_calibrator_labels/` and `weights/NFM_calibrator_labelexp/`.

Instead, the phase bump is kept out of the training data at the source, the per-sample fits (`datasets/MUSE_STD_dataset/STD_fitter.py`):

- Observation: the fitter already meant to suppress the bump of the PSFs without the "Non-point" flag (`suppress_bump = 1e3`), but the
  factor only multiplied `force_positive(bump)`, i.e. it penalized negative bumps only. 131 of the 402 unflagged PSFs have fitted bumps
  above 40 (median 22, flagged: 86).
- Change: for those PSFs, an additional `bump^2 * w_bump * suppress_bump` term (5e-2 per unit^2: bump 1 costs ~40 % of a typical fit
  loss) pulls the bump to 0; the "Non-point" PSFs and the multi-source fits are unchanged. The pickles store `bump suppressed`.
- Check (refit of samples 357-362 into the scratchpad, 6 s per sample on GPU 0): the bumps of the five unflagged PSFs go from 41-77 to
  |b| <= 0.002, the Non-point control (359) is bit-identical; the plain balanced PSF loss of the unflagged fits rises by 7-25 %
  (0.117 -> 0.145, 0.044 -> 0.054, 0.149 -> 0.178, 0.062 -> 0.070, 0.087 -> 0.093), peak errors stay 2.6-6.1 %. The blur the bump absorbed
  moves into the jitter (357: 12.9 -> 21.9 mas, 358: 12.1 -> 17.4) or dn (360: 27.9 -> 37.8, 361: 51.5 -> 59.3), parameters the
  calibrator can predict.
- Not done yet: the full refit (`launch_fitting.py`, overwrites `fitted/`: back it up first), `STD_fitted_df.py`, the dataset cache
  rebuild (`build_dataset_cache`) and a retraining of the factor calibrator over several seeds (the seed scatter is 15.5-18.8 %).
