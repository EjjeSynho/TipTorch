# Per-observation pupil rotation in TipTorch

This ExecPlan follows `agents/PLANS.md`. It implements the request of `agents/execplans/pupil_rotation.md`.

This is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` updated as implementation proceeds.

## Purpose / Big Picture

After this change, one `TipTorch` instance can simulate observations (OBs) with different pupil rotation angles: the config entry
`[telescope] PupilAngle` may hold one value per observation (`[N_obs]`) instead of a single scalar. The pupil, the apodizer, the static
phase (NCPAs, MUSE phase bump) and everything attached to the DM / WFS frame (WFS sampling, aliasing combs, DM optimization directions,
Moffat PSD) rotate per observation, while the image-plane quantities (`dx`, `dy`, the jitter kernel, the science source directions, the
wind) stay in the image frame. With a single angle (the only case supported before), the model behaves exactly as before and is not slower.

For the MUSE NFM calibrator this removes the need to derotate the on-sky STD-star PSFs (which damages the peak and adds unexplained
variance): the dataset cache can be built from the raw frames and the model rotates instead of the data.

Observable outcomes:

- `python tests/test_pupil_rotation.py` passes (rotation helper vs torchvision, rotated model vs rotated PSF, batched vs single-angle
  models, backward compatibility, gradient flow, config merging, axis conventions).
- `MultipleTargetsInDifferentObservations(configs)` keeps one `PupilAngle` per config.
- `build_dataset_cache(..., derotate_PSF=False)` stores raw PSFs with their pupil angles; `NFMDataset.collate_batch` no longer zeroes
  the angles, so the calibrator trains on raw frames with per-OB rotated models.

## Current State

- `src/tiptorch/PSF_models/TipTorch.py`: `InitValues` reads `PupilAngle` as a scalar (crashes on a plain float: `torch.deg2rad(float)`).
  `InitGrids` rotates the whole frequency grid once (`kx' = kx cos - ky sin`, i.e. the frame of the pupil functions is `R(+angle) k`,
  which rotates every pupil-attached feature by `-angle` in the image, clockwise). `_load_pupil_and_apodizer` rotates a FITS pupil with
  `TF.rotate(pupil, -angle)` (also clockwise); an externally given pupil tensor is used as is.
- `src/tiptorch/PSF_models/NFM_wrapper.py`: `_init_model` collapses per-OB angles to the first one (with a warning) and passes
  `PupilVLT(320, rotation_angle=angle)` as the pupil, which rotates **counter-clockwise** for positive angles: the pupil (spiders) and the
  PSD grids of the same model rotate in opposite senses whenever the angle is non-zero.
- `src/tiptorch/tools/static_phase.py`: `ZernikeBasis` builds the modes rotated by `computeZernike(angle)` (counter-clockwise, like
  `PupilVLT`), `MUSEPhaseBump` rotates its map with `TF.rotate(OPD, -angle - 45)` (clockwise). `PhaseMap.OPD2Phase` masks the phase with
  `pdims(model.pupil * model.apodizer, -2)`, i.e. a single `[1, 1, H, W]` mask.
- `src/tiptorch/managers/config_manager.py`: `['telescope', 'PupilAngle']` is in `SINGLETON_VALUES`, so `Merge` keeps the angle of the
  first config only.
- Data pipeline: `LoadSTDStarCache(derotate_PSF=True)` rotates the cubes by `scipy.ndimage.rotate(PSF, -PupilAngle)` and sets the config
  angle to 0; `build_dataset_cache` always derotates; `NFMDataset.collate_batch` forces `PupilAngle = 0.0`.
- Tensors: everything is PyTorch on `model.device` (`cuda:1` by default here), `float32` by default; the forward pass must stay
  differentiable w.r.t. the model inputs (incl. the LO NCPA coefficients) for the calibrator training.

## Target Behavior

Frames (documented in the `TipTorch` class docstring):

- Image frame: the data / output pixel grid. TipTorch's `x` axis is the image **row** axis (dim -2) and `y` the **column** axis (dim -1)
  for `dx, dy`, `src_dirs_x/y`, `Jx/Jy`, the wind `vx/vy`, `kx/ky` and `U/V`. (`kx` = first output of `meshgrid(..., indexing='ij')`.)
- `PupilAngle` [deg]: rotation of the pupil / DM / WFS frame relative to the image frame, **counter-clockwise on the image for positive
  values**, i.e. the model PSF of angle `a` is `scipy.ndimage.rotate(PSF(0), a)` (= `torchvision rotate(+a)` = `PupilVLT(rotation_angle=a)`).
  This is the sense of the MUSE pipeline (`PSF_derotated = rotate(PSF_raw, -PupilAngle)`), verified on raw STD stars (see Artifacts).
- Pupil frame (rotated with the pupil): pupil, apodizer, static OPD maps, WFS sampling `sinc(d k)`, aliasing combs, DM optimization
  directions, LGS asterism of tomographic systems, Moffat PSD orientation.
- Image frame (not rotated): `dx, dy`, jitter, science source directions, wind, single NGS/LGS anisoplanatism, differential refraction.

Shape contracts:

    config['telescope']['PupilAngle']: scalar, [1] or [N_obs]   (deg)
    model.pupil_angle:                 [1] (shared, also when all N_obs values are equal) or [N_obs]
    model.rotate_pupil:                bool, False when all angles are 0 -> no rotation overhead at all
    model.pupil, model.apodizer:       [H, W] reference orientation (angle 0), what the phase bases are built on
    model.pupil_rot, apodizer_rot:     [H, W] (no rotation / one angle) or [N_obs, H, W]
    model.kxy_AO_():                   (kx, ky) of the AO half grid in the pupil frame, [1 or N_obs, nOtf_AO_y, nOtf_AO_x];
                                       returns the stored grids themselves without rotation
    model.OTF_static_default:          [1 or N_obs, 1, nOtf, nOtf]
    PhaseMap(x):                       [N_src, N_wvl, H, W] complex field, OPD rotated per observation, masked per observation
    TipTorch(pupil=...):               None (FITS of the config), a [H, W] tensor (reference orientation, rotated by TipTorch when needed)
                                       or a callable angle_deg -> [H, W] mask, called per observation angle (analytic rotation)

## Non-Goals

- No separate DM angle (post-focal DM instruments): the DM / WFS frame is the pupil frame.
- No change of the x/y (row/column) axis convention of TipTorch (fitted `dx`, `dy`, `Jx`, `Jy` and the stored datasets depend on it).
- No rotation of the wind direction or of the differential refraction direction.
- No retraining of the calibrator, no rebuild of the dataset caches (the scripts gain the option only).
- No change of the FITS pupil orientation (`ut4pupil320.fits` is 45 deg off `PupilVLT(0)`, see Surprises).

## Progress

- [x] (2026-10-16) Inspect the model, wrappers, config manager, data pipeline and calibrator; check axis / rotation conventions numerically.
- [x] (2026-10-16) Determine the rotation sense from raw STD stars (`PupilVLT(+theta)` matches the raw frames).
- [x] (2026-10-16) `rotate_images` helper (batched affine-grid rotation, reproduces torchvision / scipy).
- [x] (2026-10-16) TipTorch: per-OB angles, `kxy_AO_()`, pupil / apodizer orientation, stale-pupil refresh in `Update`, consumers.
- [x] (2026-10-16) Static phase: per-OB OPD rotation and masks; Zernike / phase bump built in the reference frame.
- [x] (2026-10-16) Config manager: per-config `PupilAngle`.
- [x] (2026-10-16) NFM wrapper: analytic pupil generator, no angle collapsing.
- [x] (2026-10-16) Data pipeline / calibrator: `derotate_PSF` option of the cache, no angle override in `collate_batch`, STD fitter flag.
- [x] (2026-10-16) Fix the source azimuth convention of `tools/observations.py`.
- [x] (2026-10-16) Tests `tests/test_pupil_rotation.py` (8 / 8), timing of the single-angle path, `test_tiptop_integration.py` rerun,
      bit-exact comparison with the committed code on derotated STD stars, raw STD stars end to end through the NFM wrapper and the
      calibrator's collate path.
- [x] (2026-10-16) Record outcomes.
- [x] (2026-10-16) `development/MUSE_onsky.py`: `DEROTATE_PSF` switch, raw-frame stack of 6 phase-bump stars with streaks (IDs 462, 42, 14,
      144, 344, 436; angles 7-263 deg), list-safe flags, per-OB rotation-check figure (data / model at the red end, oriented pupil,
      rotated static OPD with the bump coefficient); the new cells were run as a script (one forward, no fit) and render correctly.
      The calibrator configs got `derotated_PSFs` (default true) so the derotated regime stays the explicit default for stacked OBs.
- [ ] Follow-ups (not in this plan): rebuild the raw-frame dataset cache (`DEROTATE_PSF = False` in `create_STD_dataset.py`), refit the
      STD stars on raw frames (`DEROTATE_PSF = False` in `STD_fitter.py`, separate output folder), retrain the calibrator on it.

## Surprises & Discoveries

- Observation: TipTorch's `x` axis is the image row axis for every quantity (`dx` moves the PSF along the rows, `src_dirs_x` elongates the
  PSF along the rows, `kx` varies along dim -2).
  Evidence: scratch check: `dx = +3` moved the centroid from row 50.0 to 52.8 (column unchanged); `src_dirs_x = 15"` gave a row variance
  of 213.6 vs 205.1 along the columns.
  Consequence: the "perpendicular elongation" seen in the PSF explorer is a convention mismatch, not a model bug: the explorer pad was
  drawn with `x` up to match, but `tools/observations.py` computed the source azimuth as `arctan2(y_peak, x_peak)` with `x_peak` the
  column, so science-field sources were elongated perpendicular to their offset. Fixed in `AddSourcesToModelConfig`; TipTorch's
  convention is now documented and tested.
- Observation: the three rotation paths disagreed in sense. `PupilVLT(+30)` = `TF.rotate(PupilVLT(0), +30)` (counter-clockwise, 580
  mismatching pixels vs 1712 for -30); the FITS path and the frequency grids rotated clockwise (DL PSF spikes at -30 deg, Moffat ellipse at
  -30.1 deg vs +29.7 deg for `TF.rotate(+30)` of the unrotated PSF); the MUSE phase bump also clockwise.
  Consequence: the NFM model with a non-zero angle had its spiders and its AO frame rotated in opposite directions. All paths now follow
  the counter-clockwise convention.
- Observation: the raw STD-star frames match `PupilVLT(+PupilAngle)`.
  Evidence: zero-lag correlation of the azimuthal profiles (asinh stretch, 12-30 px) of the raw red-end PSF with the DL PSF of
  `PupilVLT(+theta)` / `PupilVLT(-theta)`: ID 1: 0.143 / 0.048, ID 4: 0.186 / -0.049, ID 14: 0.085 / -0.035, ID 12: 0.068 / -0.059
  (IDs 11, 13 inconclusive, low SNR); the derotated frames vs `PupilVLT(0)` give the same values as raw vs `PupilVLT(+theta)`.
  Consequence: `PupilAngle` = the header-derived angle `(45 - parallactic) % 360` of the pipeline, counter-clockwise, with no sign change.
- Observation: `ut4pupil320.fits` is rotated by -45 deg with respect to `PupilVLT(0)` (azimuthal correlation 0.967 at -45 deg).
  Consequence: the FITS pupil and the analytic one are not interchangeable; the NFM wrapper keeps the analytic one.
- Observation: `Phase2OTF` cut the even-size correction as `phase_padded[:-1, :-1]`, which would slice the batch and wavelength dims of a
  4D phase. Never hit (odd grids). Fixed to `[..., :-1, :-1]`.
- Observation: `TipTorch_profiler.py` passes `'sum'` positionally as `retain_PSDs` and the device as `norm_regime` (the signature is
  `AO_config, AO_type, pupil, PSD_include, retain_PSDs, norm_regime, device`). Not touched here.

## Decision Log

- Decision: keep TipTorch's `x` = rows convention and fix the callers instead of transposing the model.
  Rationale: `dx`, `dy`, `Jx`, `Jy` and the Zernike orientation are baked into the fitted STD datasets and calibrator weights; a transposition
  would silently relabel them.
  Date/Author: 2026-10-16, Claude.
- Decision: positive `PupilAngle` = counter-clockwise on the image (scipy / torchvision sense), fixed by the test
  `PSF(a) == rotate(PSF(0), a)`.
  Rationale: this is the sense of the data pipeline's derotation and of the raw frames (Surprises), and of `PupilVLT`.
  Date/Author: 2026-10-16, Claude.
- Decision: rotated grids are computed by `kxy_AO_()` (and `_Moffat_grids`, `_aliasing_combs`) from the stored image-frame grids and kept
  in the grid cache keyed by the angle values (`_pupil_angle_key`), so they are built once per grid and set of angles, not at every call;
  only the AO half grids exist per observation. `k`, `k2`, the masks and the piston filters are rotation invariant and untouched.
  Rationale: zero overhead without rotation (the stored tensors are returned as they are) and no recomputation per forward with it:
  50-forward timing 3.04 ms (angle 0) vs 2.99 ms (angle 30), 3.43 ms (4 OBs at 0) vs 3.44 ms (4 OBs at 30 / -45 / 100 / 220).
  Date/Author: 2026-10-16, Claude.
- Decision: pupil-frame vs image-frame split (see Target Behavior); the science source directions use the image-frame grids, the LGS
  asterism of tomographic systems the pupil-frame grids, the single-GS anisoplanatism of SCAO / SLAO the image frame.
  Rationale: the 4LGSF is mounted on the telescope and rotates with the pupil; NGS are sky objects; source positions are measured in the image.
  Date/Author: 2026-10-16, Claude.
- Decision: the static OPD maps are generated in the reference frame and rotated per observation with `rotate_images` (bilinear,
  differentiable) inside `PhaseMap.forward`; the bases are not duplicated per observation.
  Rationale: `N_obs x N_modes x H x W` basis cubes would cost ~370 MB per 100 OBs for nothing; the fitted coefficients are then in the pupil
  frame for every OB, which is what a telemetry-driven calibrator needs.
  Date/Author: 2026-10-16, Claude.
- Decision: the NFM wrapper passes an analytic pupil generator (`PupilVLT` per angle, cached) rather than a rotated mask.
  Rationale: nearest-neighbour rotation of 1.5 px wide spiders breaks them into staircases; `PupilVLT` is 3 ms per angle.
  Date/Author: 2026-10-16, Claude.
- Decision: `U`, `V` stay in the image frame.
  Rationale: `dx`, `dy` are astrometric corrections in the data frame; the jitter anisotropy angle `Jxy` is a free parameter (isotropic in NFM).
  Date/Author: 2026-10-16, Claude.
- Decision: the wind direction and the differential refraction direction keep their image-frame interpretation.
  Rationale: out of scope; the wind direction is a telemetry value of uncertain frame in both regimes, the refraction direction is a separate
  science question (previous `TODO` kept).
  Date/Author: 2026-10-16, Claude.
- Decision: `Update()` re-orients the pupils automatically when the config angles changed since the last orientation (also with
  `pupils=False`).
  Rationale: the calibrator swaps the batch config and calls `Update(grids=False, pupils=False)` / `SetWavelengths` per batch; the pupils of
  the new batch must follow without touching the training loop.
  Date/Author: 2026-10-16, Claude.

## Plan of Work

### Milestone 1: conventions (done before coding)

Scratch checks of the axis mapping and of the three rotation paths (see Surprises); raw STD stars vs `PupilVLT(+/-theta)`.

### Milestone 2: TipTorch core

`tiptorch/tools/utils.py`: `rotate_images(images, angles_deg, mode)` (batched `affine_grid` + `grid_sample`, counter-clockwise, reproduces
`TF.rotate` to 1e-6 bilinear / exactly nearest).

`TipTorch.py`: `InitValues` reads `[1]` or `[N_obs]` angles, sets `rotate_pupil`; `kxy_AO_()`, `_Moffat_grids()`, `_aliasing_combs()`;
`ReconstructionFilter`, `TomographicReconstructors`, `OptimalDMProjector`, `AliasingPSD`, `MoffatPSD` use them; `_load_pupil_and_apodizer`
loads / generates the reference masks and `_orient_pupils` makes `pupil_rot` / `apodizer_rot`; `UpdateStaticOTF` and `Phase2OTF` accept
`[N_obs, H, W]`; `Update` refreshes stale pupils; class docstring with the frames.

### Milestone 3: static phase, config, wrappers

`static_phase.py`: `PhaseMap.rotate_OPD`, per-OB masks in `OPD2Phase`, reference-frame `ZernikeBasis` / `MUSEPhaseBump`.
`config_manager.py`: `PupilAngle` per config in `Merge`, shared in `MultipleTargetsInOneObservation`.
`NFM_wrapper.py`: generator pupil, no collapsing.

### Milestone 4: data pipeline, calibrator, observations

`STD_dataset_utils.build_dataset_cache(derotate_PSF)`, `create_STD_dataset.py` and `STD_fitter.py` flags, `NFMDataset` (no override,
`derotated` attribute), `tools/observations.py` azimuth.

### Milestone 5: validation

`tests/test_pupil_rotation.py`; timing of the single-angle forward before / after; `tests/test_tiptop_integration.py` rerun.

## Concrete Steps

Run from the repository root in the `TipTorch` conda env (`source ~/miniforge3/etc/profile.d/conda.sh && conda activate TipTorch`):

    python tests/test_pupil_rotation.py          # GPU (cuda:1 by default), ~1 min
    python tests/test_tiptop_integration.py      # CPU, unchanged helpers

Expected: every line `[PASS]`, exit code 0.

## Validation and Acceptance

- `rotate_images` reproduces `torchvision rotate` (bilinear max abs diff < 1e-5, nearest identical) for odd and even sizes.
- Rotated model (angle `a`, elliptic Moffat, aliasing, astigmatic static phase, no wind) vs `rotate(PSF(0), a)`: the Moffat ellipse angle
  and the spike angle follow `+a`; the asinh-image L1 distance to `rotate(PSF(0), +a)` is far below the one to `rotate(PSF(0), -a)`.
- A batch of 3 OBs with angles (0, 35, -70) reproduces the 3 single-angle models (max relative difference ~1e-5).
- A batch with all angles 0 reproduces the single model exactly and `kxy_AO_()` returns the stored grids (no rotation path).
- Gradients w.r.t. the LO coefficients are finite through the OPD rotation.
- `Merge` keeps `[N]` angles; `MultipleTargetsInOneObservation` keeps one.
- `dx` moves the PSF along the rows; `src_dirs_x` elongates it along the rows (documented convention).
- Single-angle forward timing unchanged.

## Idempotence and Recovery

All code changes are additive; a config with a scalar / zero `PupilAngle` follows the old path (no rotation). Set `derotate_PSF=True`
(default) in `build_dataset_cache` / the `DEROTATE_PSF` flags to keep the derotated regime. The tests write nothing. The scratch
checks live outside the repository.

## Interfaces and Dependencies

    TipTorch(AO_config, ..., pupil: None | torch.Tensor [H, W] | Callable[[float], array [H, W]], ...)
    TipTorch.kxy_AO_() -> (kx, ky)                       # pupil-frame AO half grids, [1 or N_obs, nOtf_AO_y, nOtf_AO_x]
    TipTorch.pupil_angle: torch.Tensor [1 or N_obs]      # deg, counter-clockwise
    TipTorch.rotate_pupil: bool
    TipTorch.pupil_rot, TipTorch.apodizer_rot            # [H, W] or [N_obs, H, W]
    tiptorch.tools.utils.rotate_images(images, angles_deg, mode='bilinear') -> torch.Tensor
    PhaseMap.rotate_OPD(OPD [N_src, N_wvl, H, W]) -> same shape
    build_dataset_cache(..., derotate_PSF=True)          # cache gains the 'derotated' entry
    NFMDataset.derotated: bool
    train_NFM_calibrator.py cfg["derotated_PSFs"]        # True: muse_STD_stars_dataset.pt (derotated, the default), False: *_raw.pt;
                                                         # the trainer refuses a cache of the other regime, the validation follows the training meta

No new dependencies.

## Documentation Updates

- `TipTorch` class docstring: frames and the angle sign.
- Docstrings of `rotate_images`, `kxy_AO_`, `_orient_pupils`, `PhaseMap.rotate_OPD`, `build_dataset_cache`.
- This plan.

## Artifacts and Notes

Scratch checks (`conventions*.py` in the session scratchpad, not in the repository):

    TF.rotate(+30) vs scipy rotate(+30): max|diff| = 5.5e-06 (same sense); scipy rotate(+90) moves the right bar up (counter-clockwise)
    PupilVLT(+30) vs TF.rotate(PupilVLT(0), +30): 580 mismatching px, vs -30: 1712
    FITS path PupilAngle=30: DL spikes at -30 deg; grids rotated by 30: Moffat ellipse at -30.1 deg (TF.rotate(+30) of PSF(0): +29.7 deg)
    FITS pupil vs PupilVLT(0): azimuthal correlation 0.967 at -45 deg
    dx=+3: centroid (row, col) 50.0, 50.0 -> 52.8, 50.0;  src_dirs_x=15": variance rows 213.6, cols 205.1 (on-axis 125.5 / 125.5)
    Raw STD stars, zero-lag correlation with PupilVLT(+theta) / PupilVLT(-theta): ID 1 0.143/0.048, ID 4 0.186/-0.049, ID 14 0.085/-0.035, ID 12 0.068/-0.059

Test run (`python tests/test_pupil_rotation.py`, cuda:1):

    rotate_images bilinear n=101/100, angles 30 / -72.5: max |diff| vs torchvision 1-2e-15; nearest pupil: identical
    PSF(35) vs rotate(PSF(0), +35): asinh distance 0.0429, vs -35: 0.1280
    Moffat ellipse angle: -0.9 deg at 0, 34.4 deg at 35 (counter-clockwise)
    batch (0, 35, -70) vs single models: max relative difference 6.9e-08, 6.6e-08, 0.0
    3 observations at angle 0 vs the single model: 3.2e-08; kxy_AO_ / pupil_rot / _aliasing_combs return the stored tensors
    gradients w.r.t. the LO coefficients finite and non-zero for every observation
    Update(pupils=False) with new angles: pupils and static OTF re-oriented, PSF equal to a fresh model (0.0)
    merged configs keep every PupilAngle; one shared angle for 4 sources in one observation
    dx = +3 px: centroid (50.00, 50.00) -> (52.74, 50.00); src_dirs_x = 15": variance rows 230.7, columns 221.8
    forward time: angle 0 3.0 ms, angle 30 3.2 ms (101 px, 1 wavelength, aliasing + Moffat on)
    8 / 8 tests passed;  tests/test_tiptop_integration.py: passed

Backward compatibility (scratch, committed code in a git worktree vs the working tree, `PSFModelNFM` on derotated STD stars 1 and 4 with
LO coefficients and dx set): PSF, pupil and static OTF max relative difference 0.0.

Raw STD stars 1, 4, 14, 12 (angles 219.3, 243.7, 94.9, 247.7 deg) in one `PSFModelNFM(multiple_obs=True)`: init 0.2 s, `pupil_rot`
[4, 320, 320], phase field [4, 8, 320, 320], finite PSFs; zero-lag spike correlation of the raw data with the rotated model / the
unrotated model: ID 4 0.073 / -0.256, ID 14 0.076 / -0.013, ID 12 0.041 / -0.184 (ID 1: -0.283 / -0.313, the unfitted halo dominates).
`NFMDataset.collate_batch` keeps the four angles; swapping the batch config through `Update(grids=False, pupils=False)` re-orients the
pupils (`pupil_rot` equal to the fresh model's).

## Outcomes & Retrospective

Implemented: per-observation pupil rotation in `TipTorch` (angles `[1]` or `[N_obs]` from `[telescope] PupilAngle`), with on-demand
pupil-frame grids (`kxy_AO_`, `_Moffat_grids`, `_aliasing_combs`), per-observation pupils / apodizers / static OTF (tensor or generator
input, `rotate_images` helper), per-observation static OPD rotation and masking in the phase bases, automatic re-orientation on config
changes, per-config angles in the config manager, the analytic pupil generator in the NFM wrapper, the raw-frame option of the dataset
cache / STD fitter, no angle override in the calibrator's batches, the source azimuth fix of `tools/observations.py`, the frames
documented in the `TipTorch` docstring, and `tests/test_pupil_rotation.py`.

Validated: see Artifacts. The single-angle / zero-angle path is bit-exact with the committed code and as fast as before; the rotation
sense matches the MUSE pipeline and the raw frames; batched observations reproduce the single-angle models; gradients flow.

Conventions settled: TipTorch's `x` is the image row axis (the explorer's "perpendicular" elongation was the azimuth convention of
`AddSourcesToModelConfig`, now fixed); a positive `PupilAngle` is counter-clockwise on the image; the FITS pupil of the MUSE config is
45 deg off the analytic one (not changed). On the DM question of the request: the conclusion holds for MUSE NFM (the DSM is the telescope
secondary, pre-focal, so the DM / WFS frame rotates on the detector together with the spiders), while for an instrument whose DM and WFS
sit behind the derotator (e.g. SPHERE / SAXO) the DM frame is fixed on the detector and only the telescope pupil rotates, which would need
the separate DM angle left for later.

Not done / limitations: the wind direction and the differential refraction direction are still interpreted in the image frame; `U`, `V`
are not rotated (jitter anisotropy stays an image-frame parameter); the raw-frame dataset cache, STD refits and calibrator retraining are
follow-ups. The LGS asterism of tomographic systems now rotates with the pupil (it did already with a single non-zero angle), so a
non-zero angle with an NGS asterism in LTAO / MCAO would need the asterism given in the pupil frame.
