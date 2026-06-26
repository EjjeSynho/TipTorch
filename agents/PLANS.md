# Codex Execution Plans

> Draft status: this is the finalized format for future ExecPlans. Do not treat it as active project workflow until it is explicitly adopted.

This file defines how to write and maintain an execution plan, called an ExecPlan, for this repository.

An ExecPlan is a self-contained, living implementation document. A competent developer or coding agent should be able to read a single ExecPlan, understand the goal, modify the code, validate the result, and resume work later without relying on hidden context.

Use this file as the ruleset for all concrete plans stored under:

```text
.agents/execplans/
```

Concrete plan filenames should be descriptive Markdown files, for example:

```text
.agents/execplans/chromatic-psf-model-execplan.md
.agents/execplans/pixel-mtf-otf-resampling-execplan.md
.agents/execplans/telemetry-preprocessing-execplan.md
```

## When to Use an ExecPlan

Use an ExecPlan for:

- Multi-file changes.
- New scientific or ML features.
- Refactors that affect public APIs, tensor shapes, training behavior, configuration files, or saved artifacts.
- Changes that require validation beyond a trivial unit test.
- Any task likely to take more than about one hour.
- Any task where resuming from an intermediate state should be possible.

An ExecPlan is unprefferable for:

- Typo fixes.
- Small comments or docstring edits.
- Insignificant changes in the current existing files.
- One-line bug fixes with obvious validation.

When in doubt, create a new ExecPlan.

## Core Requirements

Every ExecPlan must be:

1. **Self-contained**

   The plan must include all context needed to execute it. Do not assume the reader remembers previous chats, hidden prompts, or external design discussions.

2. **Outcome-focused**

   The plan must describe what new behavior exists after the change and how to observe it.

3. **Repository-specific**

   Mention exact files, functions, classes, commands, expected tensor shapes, config keys, and output artifacts.

4. **Scientific-code aware**

   For this repository, correctness is not only "the tests pass." The plan must preserve physical meaning, numerical stability, autograd compatibility where required, and reproducibility.

5. **Living**

   The following sections must be updated while work proceeds:

   - `Progress`
   - `Surprises & Discoveries`
   - `Decision Log`
   - `Outcomes & Retrospective`

6. **Validated**

   Every ExecPlan must include explicit validation commands and expected outcomes.

## Project-Specific Priorities

This repository contains scientific Python/PyTorch code for adaptive-optics PSF prediction and related modeling.

When writing or executing plans, prefer:

- Concise implementation without excessive code generation.
- Clear tensor-shape contracts.
- Keeping abbreviation in capital leters everywhere (e.g. PSF, NFM, OTF, etc.)
- Explicit wavelength, object number, and batch-axis conventions.
- Differentiable implementations when the component participates in training.
- Vectorized PyTorch operations over Python loops where reasonable.
- Numerically stable parameterizations for positive quantities, spectra, variances, FWHM values, fluxes, and covariance matrices.
- Config-driven behavior over hardcoded constants.
- Small composable functions with tests.
- Reproducible validation scripts.
- Backward-compatible APIs unless the plan explicitly states a migration path.

Avoid:
- Commenting obvious code segments.
- Excessive code generation.
- Silent unit changes.
- Hidden normalization changes.
- Mixing NumPy/CuPy and PyTorch in differentiable paths.
- Additing computation that could introduce excessive back-and-forth transfers between CPU and GPU.
- Performing parallelizable tasks on CPU.
- Detached tensors inside training-critical code unless explicitly justified.
- Large rewrites without intermediate validation.
- Changing scientific assumptions without recording the rationale.

## Required ExecPlan Structure

Ideally, ExecPlan uses the following structure.

```md
# <Short action-oriented title>

This ExecPlan follows `.agents/PLANS.md`.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

Describe the user-visible or scientist-visible outcome.

Good examples:

- After this change, the PSF predictor can represent wavelength-smooth model parameters by predicting spline coefficients instead of independent per-wavelength values.
- After this change, detector sampling is handled through a pixel-MTF-aware OTF pipeline rather than image-space interpolation, reducing aliasing artifacts in radial-profile comparisons.
- After this change, the flux-disentangling routine can process large MUSE cubes in wavelength batches without changing the final fitted spectra.

State how the result can be observed:

- A new command runs.
- A test passes.
- A notebook cell produces the same shape and normalization.
- A validation plot or metric changes as expected.
- A config option enables the behavior.

## Current State

Describe the relevant current implementation.

Include:

- Important files and paths.
- Important classes and functions.
- Current input and output tensor shapes.
- Existing configs.
- Existing tests or absence of tests.
- Current limitations or bugs motivating this plan.

For PSF/ML work, explicitly mention:

- Whether tensors are NumPy arrays or PyTorch tensors.
- Whether tensors live on CPU or GPU.
- Whether gradients are required.
- Expected dtype.
- Expected normalization, for example unit-flux PSFs or PSD units.
- Relevant wavelength, spatial, or baseline dimensions.

## Target Behavior

Describe the desired final behavior precisely.

Include:

- New or changed public functions/classes.
- New config keys.
- New command-line arguments, if any.
- Shape contracts.
- Units.
- Error handling.
- Backward compatibility behavior.

Example shape contract:

    Input telemetry shape:
        batch x num_features

    Input wavelength shape:
        batch x num_wavelengths
        or
        num_wavelengths

    Predicted parameter shape:
        batch x num_wavelengths x num_model_parameters

    Output PSF shape:
        batch x num_wavelengths x height x width

## Non-Goals

List what this plan deliberately does not do.

Examples:

- Does not retrain production weights.
- Does not change the FITS I/O format.
- Does not replace the analytical PSF model.
- Does not modify unrelated plotting utilities.
- Does not optimize GPU memory beyond avoiding obvious waste.
- Does not change science assumptions outside the specified component.

Non-goals prevent scope creep.

## Progress

Use checkboxes. Update this section whenever meaningful progress is made.

Format:

- [x] (YYYY-MM-DD HH TZ) Completed item.
- [ ] Pending item.
- [ ] Partially completed item. Completed: ... Remaining: ...

Initial example:

- [ ] Inspect relevant files and confirm current APIs.
- [ ] Add or update tests that define the desired behavior.
- [ ] Implement the minimal code changes.
- [ ] Run unit tests.
- [ ] Run a small scientific sanity check.
- [ ] Update documentation or examples.
- [ ] Record outcomes and remaining gaps.

## Surprises & Discoveries

Record unexpected findings here. Each entry should include evidence.

Format:

- Observation: ...
  Evidence: ...
  Consequence: ...

Examples:

- Observation: Existing PSFs are normalized after cropping, not before detector resampling.
  Evidence: `normalize_psf()` is called inside `crop_to_detector_grid()` in `src/...`.
  Consequence: Validation must check flux conservation before and after the new OTF path.
- Observation: The wavelength axis is sometimes represented as `Nλ` and sometimes folded into the batch axis.
  Evidence: The training loader returns `x.shape == (B, F)` but model output is reshaped to `(B, Nλ, H, W)`.
  Consequence: The new API must explicitly document the wavelength-axis convention.

## Decision Log

Record implementation decisions and rationale.

Format:

- Decision: ...
  Rationale: ...
  Date/Author: ...

Examples:

- Decision: Predict spline coefficients for chromatic parameters rather than predicting every wavelength independently.
  Rationale: This enforces smooth wavelength behavior and reduces output dimensionality for small training sets.
  Date/Author: YYYY-MM-DD, Codex.
- Decision: Keep achromatic parameters outside the spline head.
  Rationale: Some telemetry-driven terms should not vary with wavelength except through the analytical PSF model.
  Date/Author: YYYY-MM-DD, Codex.

## Plan of Work

Describe the implementation sequence in prose.

This section should be detailed enough that the implementer knows where to work and why. Prefer narrative milestones over vague bullet lists.

Each milestone should:

- Name the files to edit.
- Explain what changes.
- Explain why it is safe.
- Explain how it will be validated before moving on.

Example structure:

### Milestone 1: Inspect and lock current behavior

Inspect the current predictor, training loop, config reader, and PSF synthesis path. Identify where telemetry features, wavelength values, and analytical model parameters are assembled. Add characterization tests for current tensor shapes and normalization before changing behavior.

### Milestone 2: Add the new interface

Introduce the new configuration option and model-head class without changing the default behavior. The default path must reproduce the current output shapes and pass existing tests.

### Milestone 3: Implement the new behavior

Implement the new differentiable computation. Keep all training-critical operations in PyTorch. Avoid CPU transfers inside the forward pass. Add shape checks and clear error messages.

### Milestone 4: Validate scientifically

Run small synthetic tests and one representative validation example. Check shapes, finite values, flux normalization, gradient flow, and at least one physically meaningful sanity condition.

## Concrete Steps

List exact commands to run from the repository root. Use commands that are safe to rerun.

Before running python, select the right conda environment, which is likely `AO-torch` or `TipTorch`.

Example:

    python -m pytest tests/test_psf_shapes.py -q

    python -m pytest tests/test_chromatic_model.py -q

    python scripts/validate_small_psf_case.py \
        --config configs/examples/muse_nfm_small.ini \
        --device cuda \
        --max-samples 4

When expected output matters, include a short description:

Expected:

- Tests pass.
- No NaN or Inf values are present.
- Output PSF shape is `B x Nλ x H x W`.
- Total PSF flux is close to 1 within tolerance.
- Gradients exist for trainable model parameters.

## Validation and Acceptance

Define what "done" means. Include both software validation and scientific sanity checks.

At minimum, specify:

- Unit tests.
- Shape checks.
- Dtype/device checks when relevant.
- Gradient-flow checks when relevant.
- Numerical sanity checks.
- Backward-compatibility checks.
- A small end-to-end example.

Example acceptance criteria:

- Existing tests pass.
- New tests cover the new behavior.
- The default config produces the same output shapes as before.
- The new config path produces finite PSFs with unit flux.
- The forward pass remains differentiable with respect to trainable parameters.
- A small validation run completes without NaNs.
- The plan's `Outcomes & Retrospective` section is updated.

For PSF-specific tasks, consider checking:

- PSF flux normalization.
- OTF/PSF centering convention.
- No unexpected CPU/GPU device transfers.
- No accidental wavelength-order permutation.
- Radial profiles are computed on the intended grid.
- Pixel scale and wavelength units are explicit.
- Autograd works through the new path if training uses it.

## Idempotence and Recovery

Explain how to safely retry or roll back.

Include:

- Which generated files can be deleted.
- Which commands are safe to rerun.
- How to return to the old behavior.
- How to disable the new feature through config.
- Whether migrations are required.

Example:

The implementation must preserve the old behavior when `chromatic_head = "independent"` or when the new config key is absent. If the new path fails, set `chromatic_head = "legacy"` and rerun the existing validation command. Generated validation plots under `outputs/validation/` may be deleted and regenerated.

## Interfaces and Dependencies

Specify final expected interfaces. Include function signatures where useful.

Example:

    class WavelengthSplineHead(torch.nn.Module):
        def forward(
            self,
            telemetry: torch.Tensor,
            wavelengths: torch.Tensor,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            telemetry:
                Tensor with shape (B, F).
            wavelengths:
                Tensor with shape (Nλ,) or (B, Nλ).

            Returns
            -------
            params:
                Tensor with shape (B, Nλ, P).
            """

Mention new dependencies only if necessary. Prefer existing dependencies unless the plan explains why a new one is needed.

## Documentation Updates

State what documentation, comments, examples, or config templates must be updated.

Examples:

- Update example config file.
- Add docstring with tensor shapes.
- Add a short README section.
- Add comments explaining units or normalization.
- Update notebooks only if notebooks are part of the accepted workflow.

## Artifacts and Notes

Use this section for compact evidence gathered during execution.

Examples:

- Short test output.
- Important diff summary.
- Validation metric.
- Plot path.
- Before/after shape transcript.
- Small numerical comparison.

Do not paste huge logs. Include only useful snippets.

Example:

    Test command:
        python -m pytest tests/test_chromatic_model.py -q

    Result:
        6 passed in 3.2s

    Small validation:
        max_abs_flux_error = 2.1e-6
        all finite = True
        output shape = torch.Size([2, 8, 128, 128])

## Outcomes & Retrospective

Update this section at the end of the task, and also at major stopping points.

Include:

- What was implemented.
- What was validated.
- What remains incomplete.
- Any known limitations.
- Any follow-up plans needed.
- Whether the original purpose was achieved.

Example:

The spline-based chromatic head was implemented behind a config flag and validated on synthetic wavelength grids. Existing legacy behavior is unchanged when the flag is absent. Unit tests confirm output shapes, finite values, and gradient flow. Full retraining was not performed in this plan and remains a follow-up task.
```

## Style Rules for ExecPlans

Write plans in clear prose.

Avoid vague instructions such as:

- "Improve the model."
- "Fix the pipeline."
- "Make it robust."
- "Clean up the code."

Replace them with concrete statements:

- Add a config option named `chromatic_head` with values `legacy` and `spline`.
- Preserve output shape `(B, Nλ, H, W)`.
- Add a test that verifies the PSF flux sums to one within `1e-5`.
- Run `pytest tests/test_detector_mtf.py -q`.

## Rules for Codex While Executing an ExecPlan

When executing a plan:

- Read the whole ExecPlan before editing code.
- Read `.agents/PLANS.md`.
- Inspect the repository before assuming file locations.
- Update `Progress` before and after major changes.
- Record unexpected findings in `Surprises & Discoveries`.
- Record non-trivial implementation choices in `Decision Log`.
- Prefer small, verifiable edits.
- Run the validation commands listed in the plan.
- Add missing validation if the plan lacks enough tests.
- Do not silently change scope.
- Do not leave the plan stale.
- Finish by updating `Outcomes & Retrospective`.

## Minimal ExecPlan Skeleton

Use this skeleton when creating a new concrete ExecPlan.

```md
# <Title>

This ExecPlan follows `.agents/PLANS.md`.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

...

## Current State

...

## Target Behavior

...

## Non-Goals

...

## Progress

- [ ] Inspect relevant files and current behavior.
- [ ] Add or update tests.
- [ ] Implement changes.
- [ ] Run validation.
- [ ] Update documentation.
- [ ] Update `Outcomes & Retrospective`.

## Surprises & Discoveries

None yet.

## Decision Log

None yet.

## Plan of Work

...

## Concrete Steps

Run from repository root:

    ...

## Validation and Acceptance

...

## Idempotence and Recovery

...

## Interfaces and Dependencies

...

## Documentation Updates

...

## Artifacts and Notes

None yet.

## Outcomes & Retrospective

Not started.
```