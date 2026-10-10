# Execution Plan: Finalize TipTorch Integration in TipTop

## Goal

Resolve the remaining inconsistencies and missing features in the TipTorch–TipTop integration while preserving the existing design and matching the original P3 implementation as closely as possible.

## Scope and constraints

- Keep code changes minimal; do not redesign existing functionality. - Limit changes to `TipTorchSimulation.py` and the TipTorch repository. Avoid
  changes to other TipTop source files unless necessary.
- Preserve current behavior except where a task below explicitly requests a change.

## Work plan

1. **Connect supersampling and oversampling**
   - Check whether TipTop's `supersampling` parameter corresponds to TipTorch's `oversampling` parameter.
   - If they are equivalent, connect the parameters.

2. **Trace the image-transposition discrepancy**
   - Investigate the discrepancy visible in the second row of `TipTorch/tests/runs/visual_comparison/MAVIStest_P3_vs_TipTorch.png`.
   - Identify and document its root cause; do not fix it.

3. **Add SLAO support**
   - In TipTorch's `PSF_models/TipTorch.py`, add support for SLAO simulations, including ERIS-like simulations.
   - Implement anisoplanatism for non-tomographic AO cases and the cone-effect error, using P3 as a reference.
   - Follow TipTorch's batched, differentiable implementation approach.

4. **Move PSD components into `ComputePSD(...)`**
   - When the required configuration entries are present, initialize and compute `ExtraErrorPSD`, `WindShakePSD`, `PistonFilter`, and `FocusFilter` inside TipTorch's `ComputePSD(...)`.
   - Include them in the `PSDs` and `PSD_include` dictionaries, excluded by default.
   - Remove their computation and PSD insertion from `TipTorchSimulation.py`.

6. **Check for remaining integration gaps**
   - Identify other features or small details present in TipTop but missing from TipTorch.
   - Confirm any candidate gaps before implementing them, and keep changes within the scope and constraints above.

## Completion criteria

- The parameter mapping, transposition root cause, SLAO support, and PSD handling are addressed as specified above.
- Remaining TipTop/TipTorch integration gaps have been checked.
- Relevant tests or visual comparisons confirm that integration outputs remain as close as possible to the original P3 outputs.

---

# Execution record

This part follows `agents/PLANS.md` and is kept up to date while the work proceeds. It continues the record of
`agents/execplans/TipTop_integration_new.md` (2026-10-07), which describes the bridge design, the two TipTorch checkouts
and the validation setup; both are summarized here where needed.

## Context (as found on 2026-10-10)

- `TIPTOP/tiptop/TipTorchSimulation.py` (766 lines) is the TipTorch-backed `baseSimulation`: one TipTorch model holds the
  science pointings followed by the NGS directions, PSDs are `[N_src, N_wvl, nOtf, nOtf]` in nm² on an odd grid. The bridge
  itself applied the P3-style add-ons after `ComputePSD()`: `TiltFilter()` when the LO loop is on, `WindShakePSD()` from
  `[telescope] windPsdFile`, `_extra_error_PSD()` (HO for the pointings, LO for the NGS directions) and `FocusErrorPSD(GF_res)`.
- `TipTorch.ComputePSD()` had `#TODO: anisoplanatism for non-tomography!` and `#TODO: SLAO support!`; `SpatioTemporalPSD()`
  used `A = ones` in the non-tomographic branch. `_select_AO_correction` already labels a single LGS as `'SLAO'` (non-tomographic).
- Validation runs use the `TipTop` conda env with `PYTHONPATH=astro-tiptop/TipTorch/src` (the editable install points to
  `C:\Users\akuznets\Projects\TipTorch`). `tests/test_tiptop_integration.py`, `tests/PSD_comparison_with_P3.py` and
  `tests/test_tiptop_visual_comparison.py` are run as scripts (no pytest in that env).

## Progress

- [x] (2026-10-10) Task 1: `Super_Sampling` vs `oversampling` analysed; not equivalent, not connected (see Decision Log).
- [x] (2026-10-10) Task 2: transposition root cause identified on all 9 MAVIStest pointings and documented (Surprises, and a
      note in `tests/test_tiptop_visual_comparison.py`). No fix, as requested.
- [x] (2026-10-10) Task 3: SLAO support in `TipTorch.py`: `_anisoplanatism_phasor()` / `AnisoplanatismPSD()` (non-tomographic
      anisoplanatism, now inside `SpatioTemporalPSD`), `ConeEffectPSD()` (focal anisoplanatism, analytical pupil moments), stretched
      layer heights `h_()` fixed for NGS. PSD-level match with P3 on `perfTest/ERIS_LGS.ini` (+2 off-axis sources): cone effect
      within 0.5 % of peak, spatio-temporal PSD with anisoplanatism identical on-axis and for a diagonal source.
- [x] (2026-10-10) Task 4: `ComputePSD()` owns the add-ons as `PSD_include` entries `'extra error'`, `'wind shake'`, `'tilt filter'`,
      `'focus error'` (all off by default) plus `'cone effect'` and `'MCAO cone effect'` (on by default, gated by the AO geometry /
      config flag). Settings come from `[telescope] extraErrorNm/Exp/Min/Max, extraErrorLoNm/LoExp/LoMin/LoMax, TechnicalFoV,
      windPsdFile` and `[sensor_HO] addMcaoWFsensConeError` (`ParameterParser` defaults, `required_fields.yaml`,
      `SINGLETON_VALUES`). `TipTorchSimulation.py` lost `_extra_error_PSD()` and the PSD insertions; it only sets the flags.
- [x] (2026-10-10) Task 6: gaps confirmed with the user and implemented: `h_()` NGS fix, `Super_Sampling` 1D-profile resampling
      (`PSF_radial_profile_polar`, `resample_profile_cubic` in `tools/tiptop_integration.py`, used by `computePSF1D`), P3's
      `mcaoWFsensConePSD` as `MCAOConePSD()`. Declined by the user: `NumberReconstructedLayers` layer compression.
- [x] (2026-10-10) Tests: `test_tiptop_integration.py` (+ supersampled profiles), `PSD_comparison_with_P3.py` (+ SLAO cone effect,
      SLAO anisoplanatism, MCAO cone effect on TIPTOP's perfTest configs), `test_tiptop_visual_comparison.py` (+ `ERIS_LGS`).
- [x] (2026-10-10) End-to-end bridge runs on `ERIS_LGS`, `ERIS`, `MAVIStest`, `METIS` and a `dummy.ini` variant with
      `glFocusOnNGS`, `extraErrorNm`, `extraErrorLoNm = [20, 60]` (see Artifacts).
- [x] (2026-10-10, second round, user requirements) Performance / laziness pass on `TipTorch.py`: the LGS cone uses the center of mass of
      the LGS heights (`LGS_height`); add-on values are initialized only when the config has the entries, `PSD_include` add-on entries
      follow the config (`_enable_configured_addons`) and MUSE NFM sees none of them; grid-dependent shapes of the add-ons are cached
      (`_addon_cache`); `ComputePSD(update_addons_only=True)` re-applies the add-ons to the cached half-grid core (`PSD_core`) so the
      bridge no longer recomputes the PSD for the focus error; the wind-shake FITS is loaded once with the pupils (`_load_vibration_PSD`);
      the MCAO cone low-pass is real arithmetic; the cone-effect coefficients have finite gradients. All checks re-run.
- [x] (2026-10-10, third round, user requirements) Streamlined `TipTorch.py`: 15 functions removed (the full-grid wrappers
      `TiltFilter` / `ExtraErrorPSD` / `FocusErrorPSD` / `WindShakePSD`, `FocusFilter`, `AnisoplanatismPSD`, `MCAOConePSD`, the
      `_normalized_PSD*` helpers, the per-term shape helpers); one `_unit_spectrum(name)` gives the cached unit-power shapes; one
      `ConeEffectPSD(PSD_AO)` covers the single-LGS and the tomographic LGS cases on the AO grid and is a core term; the HO extra error
      is a core error absorber with the optimizable `extra_error_nm` (like the Moffat term) while 'LO extra error', 'wind shake',
      'tilt filter', 'focus error' are the LO terms applied by `_apply_LO_terms` (`ComputePSD(update_LO_terms_only=True)`);
      `ComputePSD` builds the AO terms from a name-to-callable table. Tests adapted; all checks re-run.

## Surprises & Discoveries
- Observation (third round): P3's `focalAnisoplanatismPSD` is added on the full grid, i.e. also in the ring kc < k < kc/g where the
  DM cannot act and the fitting error already holds the full atmospheric power. On ERIS_LGS that ring carries 17.6 nm of the
  171.6 nm cone-effect total (29 % more power than the fitting term there): a double count. TipTorch restricts the term to the
  AO-corrected area; the ERIS_LGS HO residual becomes 183.7 nm (P3 184.8 nm), and the PSD comparison crops and masks P3's map.
- Observation (second round): MUSE NFM LTAO now initializes no add-on values (`extra_error_nm`, `vibration_PSD`, `focus_error_nm`
  are `None`, `PSD_include` has only the six core terms); `ComputePSD` takes 7 ms warm on the GPU, the add-on-only update 0.7 ms
  (after a 0.4 s first call that builds the cached focus shape); ERIS_LGS `ComputePSD` takes 6 ms including the cone effect.
- Observation (second round): the real-arithmetic MCAO low-pass `1 - (1-z_p)² / (1 + z_p² - 2 z_p cos)` is 0/0 at the DC pixel when
  masked entries get `z_p = 1`; masked entries now use `f_cut = kc` and are zeroed afterwards. The cone-effect coefficients had
  `sqrt(0)` at the DC frequency (zero pupil variance), which made the gradient w.r.t. the LGS height NaN; the DC frequency is
  replaced by `dk` inside the moments and masked at the end, and the square roots are clamped: `d(cone)/dH` and `d(cone)/dh_l` are finite.
- Observation (second round): `ErrorBudget` sums over the source dimension, so a per-source spatio-temporal term (off-axis
  anisoplanatism) reports sqrt(N_src) times the per-source value; with the on-axis early exit of `_anisoplanatism_phasor`
  (`A = 1`, no per-source dimension) ERIS_LGS reports 42.0 nm instead of 59.4 nm for the same PSD. Pre-existing behaviour.

- Observation (task 1): TIPTOP's `[sensor_science] Super_Sampling = [step_mas, option]` is only used by `computePSF1D` to
  resample the 1D radial profile (option 1: cubic spline of the discrete profile, option 2: polar grid, P3 `FourierUtils`
  `interpolate_1d` / `precompute_polar_grid`). TipTorch's `oversampling` multiplies the PSD / OTF grid sampling; its P3
  counterpart is the automatic integer factor `frequencyDomain.kRef_ = ceil(2 / samp)`, which TipTorch reproduces through
  `sampling_factor /= clamp(sampling_factor, 0, 1).min()` (and `baseSimulation.overSamp` is derived from it, not from a config
  key). Consequence: the two parameters are unrelated; `Super_Sampling` is now implemented as the profile resampling it is.
- Observation (task 2): off-axis TipTorch PSFs are the transpose of P3's. Evidence (`scratchpad/ref/dump_runs.py` on
  `MAVIStest`): for the pointings on the x / y axes the PSD cores differ by 41-49 % of the peak as they are and by 7-11 % after
  transposing TipTorch's; the PSF second moments swap (`pointing 1 (-15", 0")`: P3 <xx>, <yy> = 50.6, 45.0 pixels², TipTorch
  43.6, 49.6); diagonal pointings are symmetric under the transpose; the on-axis PSD core also improves (21 % to 14 %, the wind
  direction). Root cause: `TipTorch._gen_grid` uses `torch.meshgrid(..., indexing='ij')`, so `kx` runs along the first (row)
  axis and a source offset in x (`beta_x*kx`), the wind x component (`vx*kx`) and the jitter `Jx` (along `U`) all act along
  the image rows. P3's tomographic branch has the same raw convention (`fx = Beta[0]*kxAO_` with `kx_` along axis 0 of
  `freq_array`), but TIPTOP's `baseSimulation.doOverallSimulation` does `self.PSD = self.PSD.transpose()` on the `[nOtf, nOtf,
  nSrc]` cube, which swaps kx and ky before MASTSEL and puts x along the columns. P3's own non-tomographic branch pairs the x
  offset with `kyAO_` (`kxAO_*th[1] + kyAO_*th[0]`) while its wind uses `kxAO_`, so P3 + TIPTOP is not self-consistent between
  modes either. Resolution: documented only (plan task 2), TipTorch's convention is consistent across its terms and its jitter.
- Observation (task 3): with the anisoplanatism phasor written as P3 does (`exp(+2i pi h theta.k)`) the off-axis spatio-temporal
  PSD differed from P3 by 6 % in total although the anisoplanatism PSD `2(1 - Re A) W_atm` matched exactly; with the conjugate
  phasor it is identical. Cause: `TipTorch.TransferFunctions` uses `z = exp(+2j pi f Ts)`, the conjugate of P3's
  `exp(-2j pi f / rate)`, so the imaginary part of `F*h1` has the opposite sign and `A` must be conjugated to keep `Re(F h1 A)`.
  Resolution: `_anisoplanatism_phasor` uses `exp(-2j pi h theta.k)` (documented in its docstring).
- Observation (task 6): `h_()` (layer heights stretched by the LGS cone) evaluated to `-0.0` for NGS configurations because
  `Height = 0` gives `h / (1 - h/0)`. Evidence: `ERIS.ini` (zenith 30 deg) had a zero differential-refraction term (P3: 8.6 nm);
  after the fix TipTorch's HO residual is 80.9 nm vs P3 81.1 nm (80.5 before). Resolution: `is_LGS = (GS_height > 0) &
  isfinite`, NGS heights are treated as infinite, as `_select_AO_correction` already did.
- Observation (task 4): the P3-style add-ons are normalized to a total power, so their internal half-grid versions need the
  full-grid sum: `_half_sum` counts the DC column once (`2*sum(half) - sum(DC column)`); the internal terms are stored in the
  `PSDs` dict in TipTorch's rad²/m² convention (`/ _PSD_norm()`) so that `ErrorBudget` reports them like the other terms.
- Observation (task 6): the tomographic WFS-noise PSD of TipTorch is complex-valued (`(PW @ C_b @ PW_t)` without `.real`), so
  `MCAOConePSD` takes the real part of the residual PSD it receives. The first `MAVIStest` run of `ComputePSD` with the MCAO
  cone term failed on this (`clamp is not supported for complex types`).
- Observation: importing P3 / CuPy before `torch` in a script makes `torch` fail to load `shm.dll` (WinError 127) in the
  `TipTop` env; importing `torch` first avoids it. Plain `python -c "import torch"` needs `KMP_DUPLICATE_LIB_OK=TRUE` there.
- Observation (task 1 / grids): for Nyquist-oversampled detectors (`ERIS`, 14 mas at 1.65 um) TipTorch's odd grid is one pixel
  smaller than P3's even one (`_to_odd(256) = 255`) and `oversampling` is normalized away, so the PSD comparison enlarges the
  field by two pixels (`SetImageSize(N_pix + 2)`) to get the same frequency step with the DC on the same pixel.
- Observation: `Super_Sampling` option 2 (polar grid) agrees with P3 to 2 % of the peak (median 1.5 % relative) because P3
  samples a bicubic *spline* (`RectBivariateSpline`) and TipTorch's batched version samples with `grid_sample(mode='bicubic')`
  (cubic convolution); option 1 (not-a-knot cubic spline, `torch.linalg.solve`) reproduces P3's `CubicSpline` to 1e-6.

## Decision Log

- Decision: `Super_Sampling` and `oversampling` stay separate. Rationale: see Surprises; connecting them would change the PSD
  grid for a parameter documented as a 1D-profile step. Date/Author: 2026-10-10, Claude.
- Decision (user, 2026-10-10): task 4's "PistonFilter" is read as the tip/tilt rejection `TiltFilter` and "FocusFilter" as
  `FocusErrorPSD` with the runtime `GF_res`; the piston filter and the tilt filter remain separate entities (`_spatial_filters`
  returns both; `PistonFilter` is unchanged).
- Decision (user, 2026-10-10): the LO extra error inside the model follows P3's `getPSDatNGSpositions` layout: when the config has
  `[sources_LO]`, its `N_LO` directions are the trailing `N_LO` sources of `sources_science` and get `extraErrorLoNm`
  (interpolated over `TechnicalFoV/2` when given as `[center, edge]`); the bridge writes the LO field into `sources_LO` in
  `_build_model`. `extra_error_nm` / `extra_error_LO_nm` are `[N_src]` tensors that can also be set directly.
- Decision: the add-on terms are `PSD_include` entries, off by default (`'extra error'`, `'wind shake'`, `'tilt filter'`,
  `'focus error'`); `'cone effect'` and `'MCAO cone effect'` are on by default but computed only for a single LGS, respectively
  for a tomographic LGS system with `addMcaoWFsensConeError`, as P3 adds them automatically. The order in `ComputePSD` is P3's
  `powerSpectrumDensity`: AO terms (+ MCAO cone, + wind shake) padded, + fitting, + cone effect, x tilt filter, + extra error,
  + focus error. `PSDs['tilt filter']` holds the filter itself and is skipped by `ErrorBudget`. Date/Author: 2026-10-10, Claude.
- Decision: the bridge enables the entries as `baseSimulation` configures P3 (`TiltFilter=self.LOisOn`, wind shake only without
  an LO loop, extra error when any RMS is positive) and recomputes `ComputePSD()` with `'focus error'` enabled after MavisLO gives
  `GF_res` (astIndex None path), instead of adding `FocusErrorPSD` by hand. Date/Author: 2026-10-10, Claude.
- Decision: `ConeEffectPSD` keeps P3's definition (RMS fraction of a pupil sinusoid left after the best-scaled cone-compressed
  sinusoid, 5 phases, zero beyond `kc` after compression and at DC, linear radial interpolation clamped at the edge) but the
  pupil means of the sinusoids are analytical (`mean cos(2 pi f x) = sinc(f D)`) instead of 1001 pupil samples: loop-free,
  differentiable in the heights, and 0.14 % lower in total power than P3's discrete sum. Date/Author: 2026-10-10, Claude.
- Decision: the anisoplanatism phasor uses TipTorch's axis pairing (x with kx) and the conjugate sign (see Surprises); the
  P3 SCAO pairing (x with ky) is not reproduced because it contradicts P3's own tomographic branch and wind. Tests use a diagonal
  source where both agree and check the x source against the transposed map. Date/Author: 2026-10-10, Claude.
- Decision (user, 2026-10-10): implement `h_()` fix, `Super_Sampling` profiles and `addMcaoWFsensConeError`; do not implement
  the `NumberReconstructedLayers` compression of the atmosphere.
- Decision: `WindShakePSD` normalizes after masking the AO square to the circle `k <= kc` (total power = the integrated temporal
  PSD exactly, as in P3, which does not mask at all); the previous version masked after normalizing (0.02 % less power).
  Date/Author: 2026-10-10, Claude.

- Decision (user requirement, second round): layer heights are stretched by the center of mass of the LGS heights
  (`LGS_height = mean(GS_height)` over the constellation, infinite for NGS), also used by both cone-effect terms (P3 uses the first LGS).
- Decision (user requirement, second round): nothing add-on related is initialized or computed unless the config asks for it; the
  `PSD_include` add-on entries are all off by default and `_enable_configured_addons` turns on, at construction and only when no
  explicit `PSD_include` was given, `extra error` (RMS > 0), `wind shake` (`windPsdFile`), `cone effect` (single LGS) and
  `MCAO cone effect` (`addMcaoWFsensConeError` on a tomographic LGS system); `tilt filter` and `focus error` are left to the caller.
- Decision (user requirement, second round): all add-on shapes are computed on the half grids and cached per grid as unit-power
  spectra (`_cached`, reset in `InitGrids` and on dtype conversion); only the RMS factors vary between calls, keeping the terms
  differentiable w.r.t. the RMS values, the loop parameters (wind shake) and the heights (cone effects).
- Decision (user requirement, second round): `ComputePSD(update_addons_only=True)` reuses `PSD_core` (half grid, before the add-ons)
  and `_apply_addons` expands to the full grid only at the end; the bridge uses it for the focus error instead of recomputing.
- Decision (user requirement, second round): the wind-shake temporal PSD is loaded in `InitPupils` through `_load_vibration_PSD`,
  only if `vibration_PSD` is still `None` (it can be provided externally like the pupil), never on `ComputePSD`.

- Decision (user requirement, third round): the cone effect is a core term on the AO grid (not an add-on); the HO extra error is a
  core "error absorber" like the Moffat term (enabled by the config RMS or `PSD_include['extra error']`, then `extra_error_nm` [N_src]
  exists and can be optimized); the LO-related terms (LO extra error of the NGS directions, wind shake, tilt filter, focus error)
  form the "LO terms" group. The anisoplanatism of SCAO / SLAO is always on for off-axis sources (inside the spatio-temporal term).
- Decision (third round): the two cone effects are one `PSD_include` entry, `'cone effect'`, computed by `ConeEffectPSD(PSD_AO)`
  according to the AO regime (single LGS: focal anisoplanatism; tomographic LGS system with `addMcaoWFsensConeError`: unsensed volume).

## Concrete Steps

Run from `astro-tiptop/TipTorch` in the `TipTop` conda env (`KMP_DUPLICATE_LIB_OK=TRUE` when torch is imported first):

    $env:PYTHONPATH = 'C:\Users\akuznets\Projects\astro-tiptop\TipTorch\src'
    python tests\test_tiptop_integration.py            # "TIPTOP integration checks passed"
    python tests\PSD_comparison_with_P3.py --no-plots  # "PSD comparison with P3 passed" (about 1 min, needs the TIPTOP perfTest configs)
    python tests\test_tiptop_visual_comparison.py      # four figures in tests/runs/visual_comparison/

The throw-away scripts used here live in the session scratchpad (`ref/dump_runs.py`, `slao_check.py`, `conj_check.py`,
`bridge_smoke.py`, `gf_smoke.py`, `supsamp_diag.py`); they are not part of the repositories.

## Validation and Acceptance

- `tests/test_tiptop_integration.py`: previous checks plus `test_supersampled_radial_profiles` (polar profile of a Gaussian within
  2 %, spline resampling interpolates the samples to 1e-6).
- `tests/PSD_comparison_with_P3.py`: previous MUSE LTAO checks unchanged (contributors within 0.4 %, add-ons as before) plus
  `test_SLAO_cone_effect_matches_P3`, `test_SLAO_anisoplanatism_matches_P3` (ERIS_LGS + 10" diagonal and 10" x sources) and
  `test_MCAO_cone_effect_matches_P3` (MAVIStest with `addMcaoWFsensConeError = True`).
- `tests/test_tiptop_visual_comparison.py`: SR within 15 % of P3 on ERIS, MAVIStest, METIS and the new ERIS_LGS case.

## Artifacts and Notes

TipTorch bridge vs P3 `baseSimulation` after this work (HO / LO residuals in nm RMS, Strehl at the first wavelength):

| config | quantity | P3 | TipTorch bridge |
|---|---|---|---|
| ERIS_LGS (SLAO, 1 LGS 90 km, 1 NGS LO) | HO_res / LO_res | 184.8 / 53 | 184.5 / 53 |
| | SR / FWHM | 0.578 / 44.9 | 0.589 / 44.1 (max PSF diff 2.9 % of peak) |
| | before this work | | HO_res 80.4, SR 0.877 (no cone effect) |
| ERIS (SCAO) | HO_res / SR | 81.1 / 0.892 | 80.9 / 0.913 (80.5 before the `h_()` fix) |
| MAVIStest (MCAO) | HO_res (9) / SR (9) | unchanged | unchanged (139.8 ... 131.8 / 0.067 ... 0.275) |
| METIS (ELT SCAO, wind shake, extra error) | HO_res / SR | 148.4 / 0.862 | 145.8 / 0.865 (unchanged; budget: wind shake 61.2, extra 64.0) |
| dummy + glFocusOnNGS + extraErrorNm 30 + extraErrorLoNm [20, 60] | HO_res / LO_res / GF_res | 124.3 / 54.8 / 0.0 | 124.5 / 63.4 / 0.0 (LO: P3 FWHM artifact, see previous plan) |

PSD-level (`PSD_comparison_with_P3.py`): cone effect median -0.05 %, 90th pct 0.19 %, max 0.51 % of peak, total power 0.9986 of
P3's; SLAO spatio-temporal PSD on-axis max 0.17 % of peak, diagonal source 0.00 %; anisoplanatism PSD 0.00 % (x source after
transposition); HO residual per source 190.4 / 327.6 nm vs P3 190.7 / 327.7 nm (x source 324.6 vs 332.7, the axis convention).
MCAO cone effect (MAVIStest, `addMcaoWFsensConeError = True`, computed from each model's own residual PSD): zero on-axis in both,
off-axis pointings within 0.28-0.38 % of the peak, per-pointing maxima 0.6117 / 0.1894 (P3 0.6117 / 0.1895).

ERIS supersampled profiles vs P3 (`supsamp_diag.py`, step 3.5 mas): option 1 identical, option 2 within 2 % of the peak.

Final bridge error budgets (`bridge_smoke.py`): METIS fitting 107.0, spatio-temporal 20.9, chromatism 1.3, wind shake 61.2, extra error 64.0
(P3: 107.0 / 20.9 / 1.3 / 61.2 / 64.0), aliasing 34.0 and noise 18.8 (P3 45.3 / 15.6, TipTorch physics, unchanged); ERIS_LGS cone effect 171.6 nm
before the tilt filter.

## Outcomes & Retrospective

Implemented: SLAO support (anisoplanatism phasor in the non-tomographic spatio-temporal PSD, analytical cone-effect PSD, NGS
height fix), P3's MCAO LGS-WFS cone-effect PSD, the P3-style add-ons as switchable `ComputePSD` terms read from the TIPTOP config
entries (with the trailing-`sources_LO` convention for the LO extra error), `Super_Sampling` radial profiles, and the bridge
simplified accordingly. The transposition is explained (TIPTOP transposes P3's PSD cube; TipTorch keeps x along the first axis
everywhere) and left as is; `Super_Sampling` and `oversampling` are unrelated.

Validated: unit, PSD-level and visual checks pass; the SLAO HO residual went from 80 nm to 184.5 nm (P3 184.8 nm); ERIS, MAVIStest
and METIS results are unchanged or closer to P3. After the second round all numbers are identical (ERIS_LGS 184.5 nm, ERIS 80.9 nm,
MAVIStest and METIS unchanged, dummy focus variant 124.5 / 63.4 nm) and configs without the entries carry no add-on state. After the
third round ERIS_LGS gives 183.7 nm (cone effect inside the AO area only) and the dummy focus variant 124.2 nm (the HO extra error is now a
core term and is therefore tilt-filtered when an LO loop exists, whereas P3 adds it after the tilt filter: 0.3 nm here); everything else is unchanged.

Open: (1) nothing is committed in either repository, and the editable-installed checkout `C:\Users\akuznets\Projects\TipTorch` does
not carry these changes (the bridge needs `PYTHONPATH` or a mirror of `TipTorch.py`, `tools/tiptop_integration.py`,
`managers/parameter_parser.py`, `managers/config_manager.py`, `_resources/required_fields.yaml` and the tests). (2) `ErrorBudget`
sums the PSD over all sources and wavelengths, so its totals are only meaningful for a single source (pre-existing). (3) Option 2 of
`Super_Sampling` uses cubic convolution instead of a bicubic spline (2 % of the peak). (4) The MASTSEL LO jitter ellipse is applied
in TipTorch's axis convention (Jx along the first axis), consistent with its PSD but transposed with respect to P3 + TIPTOP.
