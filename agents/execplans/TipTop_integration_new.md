### TipTorch integration inside the TipTop infrastructure

## Goal
Improve the integration of TiTorch inside TipTop. It was already implemented but needs to be checked and revisited. In the end, the integration must give results as similar as possible to `baseSimulation.py`, but using TipTorch instead of P3.

## Directions 
- In general, **focus on performance, parallelizability, and differentiability where possible. Favor batched GPU-friendly solutions. Prefer PyTorch over NumPy or CuPy.** Try to do as much as possible calculation on GPU if it improves the performance.
- P3 must be replaced with TipTorch wherever possible in this integration. This is already achieved somehow.
- The resulting integration must utilize the TipTorch functionality at most, given that TipTorch excells at parallel simulation of multiple wavelengths or multiple targets
- On the TipTop side, implement all changes inside `TIPTOP/tiptop/TipTop_integration.py`. On the TipTorch's side, you're free to change nything, but mainly focus on putting necessary utils into `TipTorch/src/tools/tiptop_integration.py`and more fundamental changes into `TipTorch/src/tiptorch/PSF_models/TipTorch.py`.
- Move `extra_error_PSD`, `tiptilt_rejection_filter`, and `wind_shake_PSD` into `TipTorch.py` and make them members of the `TipTorch` class. When moving, re-implement them with PyTorch and make sure that the computation is parallelizable, has no for-loops, and prefferebly differentiable.
- Re-implement `circular_pupil(...)` from `tools/tiptop_integration.py` using `mask_circle` from `tools/utils.py`.
- Try to keep minimalistic code style similar to one present in `TipTorch.py`.

## Important notes
- Assume that the tip/tilt kernel normalization from TipTorch is the correct one. I tested it independently.
- Note that TipTorch produces PSDs with odd number of pixels to make sure that the middle pixel is the DC frequency. Thus, it must be padded when given to MASTSEL as an input
- Try to avoid iterating over the wavelengths when it comes to running TipTorch, execute the wavelengths in batches

---

# Execution record

This part follows `agents/PLANS.md` and is kept up to date while the work proceeds.

## Current State (as found on 2026-10-07)

- `TIPTOP/tiptop/TipTop_integration.py` (826 lines, uncommitted edits) is a P3-free re-implementation of
  `baseSimulation` on top of TipTorch + MASTSEL. It instantiates **one TipTorch model for the science
  directions** and **a second model per LO/Focus sensor call** (`_source_model`), computes the three
  "P3-style" PSD add-ons (`extra_error_PSD`, `tiptilt_rejection_filter`, `wind_shake_PSD`) from a
  helper module, and uses NumPy loops for every PSF metric (FWHM, EE, radial profiles).
- `TipTorch/src/tiptorch/tools/tiptop_integration.py` (206 lines) holds the helpers. The committed copy
  uses lower-case names (`extra_error_psd`, `psf_fwhm`, ...), while the bridge imports the capitalised
  names (`extra_error_PSD`, `PSF_FWHM`, ...): **the bridge did not import** at the start of this work.
- Two TipTorch checkouts exist: `C:\Users\akuznets\Projects\TipTorch` (editable-installed in the
  `TipTop` conda env, remote `TipToy.git`, HEAD `7aa3217`, carries the capitalised renames uncommitted)
  and `C:\Users\akuznets\Projects\astro-tiptop\TipTorch` (remote `TipTorch.git`, HEAD `03114d4`,
  two commits ahead, byte-identical `TipTorch.py`). Per this plan, edits go to the `astro-tiptop`
  checkout; validation runs use `PYTHONPATH=astro-tiptop/TipTorch/src`, which takes precedence over
  the editable install.
- `TipTorch.py` has no tilt filter, no extra-error PSD, no wind-shake PSD. `PistonFilter`/`FocusFilter`
  are AO-grid-only (they zero `[nOtf_AO//2, nOtf_AO//2]`), so they cannot be used on the full grid.
- TipTorch PSDs are `[N_src, N_wvl, nOtf, nOtf]` in nm², with **odd** `nOtf` (DC at `nOtf//2`).
  MASTSEL `psdSetToPsfSet(..., internal_grid_mode='even_legacy')` accepts odd PSDs but then forces
  `nPixPsf` parity to be odd when oversampling rebin is off, so the previous bridge bumped the
  requested PSF size from 200 to 201.

## Target Behavior

- `TipTorch` gains members `TiltFilter()`, `ExtraErrorPSD(rms_nm, exponent, k_min, k_max)`,
  `WindShakePSD(vibration_PSD, ...)`, `FocusErrorPSD(rms_nm)`, all PyTorch, loop-free, differentiable
  w.r.t. the RMS/PSD inputs, returning full-grid tensors broadcastable to `[N_src, N_wvl, nOtf, nOtf]`.
- `tiptorch.tools.tiptop_integration` keeps only bridge-specific helpers: jitter covariance algebra,
  `circular_pupil` built from `mask_circle`, odd->even PSD padding for MASTSEL, and **batched** PyTorch
  PSF metrics (`PSF_FWHM`, `PSF_encircled_energy`, `PSF_ensquared_energy`, `PSF_radial_profile`).
- `TipTop_integration.baseSimulation` runs **one** TipTorch model whose sources are the science
  pointings followed by the NGS directions (as P3 does with `getPSDatNGSpositions`), evaluates all
  wavelengths in one batch, feeds even-padded PSDs to MASTSEL so the requested `FieldOfView` is kept,
  and writes FITS headers with the same keyword names as `baseSimulation.py`.

## Non-Goals

- No change to `tiptop.py` backend selection, `TipTorchSimulation.py`, P3, MASTSEL or SEEING.
- No GPU support for MASTSEL beyond what it already does; no change to MavisLO.
- No retuning of the TipTorch physics (noise, aliasing, tomography) to match P3 numerically.

## Progress

- [x] (2026-10-07) Read `baseSimulation.py`, `TipTop_integration.py`, `TipTorch.py`, helper module, P3 and MASTSEL references.
- [x] (2026-10-07) Added `_spatial_filters`, `_normalized_PSD`, `TiltFilter`, `ExtraErrorPSD`, `FocusErrorPSD`, `WindShakePSD`, `_interp1d` to `TipTorch`.
- [x] (2026-10-07) Rewrote `tools/tiptop_integration.py`: moved functions removed, `circular_pupil` from `mask_circle`, `pad_PSD_to_even`, batched PyTorch metrics (`PSF_FWHM`, `PSF_encircled_energy`, `PSF_ensquared_energy`, `PSF_radial_profile`, `interpolate_curves`).
- [x] (2026-10-07) Rewrote `TipTop_integration.py` around a single science+NGS TipTorch model; MASTSEL `dk` convention fixed; FITS headers as `baseSimulation`.
- [x] (2026-10-07) Rewrote `tests/test_tiptop_integration.py` (13 checks; run as a script because `pytest` is not installed in the `TipTop` env).
- [x] (2026-10-07) Smoke + P3 comparison: `ERIS.ini` (SCAO), `dummy.ini` (LTAO + LO, 2 wavelengths), `MAVIStest.ini` (MCAO + LO, 9 pointings), including the asterism path (`doConvolveAsterism=False`, `astIndex=0`).
- [x] (2026-10-07) Smoke + P3 comparison: `METIS.ini` (ELT pupil FITS, static map, wind shake, extra error, telescope jitter) and a `dummy.ini` variant with `glFocusOnNGS` and `addAliasError`; per-term error budget vs P3 on `METIS`.
- [x] (2026-10-07) CPU device run, YAML config run (identical to INI), `savePSDs` FITS and 1D-profile JSON, FITS HDU/header layout identical to P3's output.
- [x] (2026-10-07) User decision: fix the TipTorch jitter kernel (`u_max` in `InitGrids` divided by 4); re-validated: Strehl/FWHM/EE now within a few percent of P3 on `MAVIStest` and `METIS`.
- [x] (2026-10-07) User decision: the three TipTorch files were mirrored into the editable-installed checkout `C:\Users\akuznets\Projects\TipTorch`; the bridge imports and runs there without `PYTHONPATH`.
- [x] (2026-10-07) Record outcomes.
- [x] (2026-10-07) Rewrote `tests/PSD_comparison_with_P3.py` as a runnable script with `test_*` functions (MUSE NFM LTAO vs P3 `fourierModel`): PSD contributors agree within 0.4 % (chromatism / differential refraction 2.9 %, refractive-index model), tomographic phasors within 0.1 % and `W_tomo` within 5 % of its peak; the `TiltFilter`, `FocusErrorPSD`, `ExtraErrorPSD`, `WindShakePSD` checks moved there from `test_tiptop_integration.py` and now compare against P3's own `TiltFilter`, `FocusFilter`, `extraErrorPSD`, `windShakePSD` (wind-shake power agrees to 0.02 %). Figures in `tests/runs/PSD_comparison/`.
- [x] (2026-10-07) Added `tests/test_tiptop_visual_comparison.py`: three visual tests (SCAO `ERIS`, MCAO `MAVIStest`, ELT `METIS`) rendering the P3 and TipTorch PSFs side by side with the normalized difference and radial profiles; they also assert Strehl agreement within 15 %. Found and fixed the one-pixel centering offset of the bridge on even grids.

## Surprises & Discoveries

- Observation: MASTSEL's `psdSetToPsfSet` expects nm² PSDs together with **half** the spatial-frequency step scaled by 1e9
  (`dk = 1e9*kcMax/resAO` in `baseSimulation`, i.e. `0.5e9 * PSDstep`).
  Evidence: `scratchpad/ref/mastsel_scaling_check.py`: a 100 nm RMS PSD gives SR 0.865 (= Maréchal) with `0.5e9*dk`
  and 0.964 with `1e9*dk`; the P3 reference JSONs show `dk/1e9/PSDstep = 0.50`.
  Consequence: the previous bridge passed `1e9*dk`, so its NGS spot PSFs had 4x too little turbulent broadening.
  Fixed in `_prepare_state` (`self.dk = 0.5e9 * model.dk`); after the fix the `MAVIStest` NGS FWHM agrees with P3 within
  2 % and `LO_res` within 0.3 %.
- Observation: `TipTorch.JitterKernel(Jx, Jy, Jxy)` broadens the PSF like a Gaussian of **2 x** the requested sigma.
  Evidence: `scratchpad/ref/diag_jitter_fwhm.py` on `MAVIStest`: for Jx = Jy = 3 mas (6 mas) the PSF peak drops to 0.501
  (0.199) of the jitter-free peak, while `scipy.ndimage.gaussian_filter` with sigma = 3 mas (6 mas) gives 0.891 (0.503);
  the best-fitting scipy sigma is 2.00 x the requested value in both cases. Analytically, `InitGrids` sets
  `u_max = (sampling*D/wvl/rad2mas)**2`, but the OTF grid `U, V in [-1, 1]` only reaches the Nyquist frequency
  `sampling*D/(2*wvl)`, so the exponent is 4 x too large (next to the `TODO: check if 1/2 factor is required` comment).
  Consequence: with the LO covariance converted exactly as MASTSEL's `ellipsesFromCovMats`, the TipTorch science Strehl
  with LO jitter was about half of P3's (`MAVIStest` on-axis: 0.103 vs 0.197) although `HO_res` and `LO_res` agree to 1 %.
  Resolution (user decision, 2026-10-07): `InitGrids` now sets `u_max = (sampling*D/wvl/rad2mas/2)**2`. After the fix the
  kernel for Jx = Jy = 6 mas matches a 6 mas `gaussian_filter` exactly (peak ratio 0.501 vs 0.503) and the on-axis
  `MAVIStest` Strehl is 0.193 vs P3's 0.197. **Previously fitted `Jx, Jy` values of TipTorch correspond to twice their
  nominal sigma under the new normalization.**
- Observation: ray-sampled FWHM contours with bilinear interpolation underestimate the minor axis of PSFs that are only a
  few pixels wide (the diagonal rays), while P3's contour method does not.
  Evidence: `MAVIStest` NGS spot: ours (43.5, 35.4) mas vs P3 (43.5, 43.1); with bicubic sampling (43.2, 42.7).
  Consequence: `PSF_FWHM` samples with `mode='bicubic'`; the remaining ~1 % is P3's own interpolation scheme.
- Observation: for the undersampled `dummy.ini` (20 mas pixels, 0.65 pixels per lambda/D) P3's `getFWHM` falls back to
  its "cutting" method and reports exactly one pixel (science 20.0 mas, NGS 32.2 mas = one LO pixel), see "FWHM is too
  small" in `scratchpad/ref/p3_run.log`. P3 also oversamples that PSD grid 16 x (`N = 3200`, `overSamp = 4`, 105 s), while
  TipTorch uses a 627-pixel Nyquist grid (2 s) and resamples the PSF to the detector with `interpolate(mode='bilinear')`.
  Consequence: FWHM-driven quantities (`NGS_FWHM`, hence `LO_res`) cannot be compared on this configuration. TipTorch's
  bilinear resampling also has no pixel-MTF / box-average step, unlike the P3 + MASTSEL rebin; this is a TipTorch-core
  (`OTF2PSF`) matter outside this plan.
- Observation: the jitter-free science PSFs agree with P3 in Strehl definition: `MAVIStest` on-axis SR 0.2645 (ours, DL
  reference from `TipTorch.DLPSF`) vs 0.2634 (P3 `getStrehl(method='max', psfInOnePix=True)` on the same image) vs 0.2636
  (Maréchal from `HO_res`).
- Observation: on even `FieldOfView` grids TipTorch's `OTF2PSF` pads its odd grid (nOtf = N_pix - 1) on the right/bottom
  (`torchvision CenterCrop`), so the PSF peak sits on pixel N//2 - 1, whereas P3 and MASTSEL center even PSFs on N//2
  (`centeredPixelCoords`). Evidence: peak pixels P3 (128, 128) vs TipTorch (127, 127) on `ERIS`, (256, 256) vs (255, 255)
  on `MAVIStest`, (1024, 1024) vs (1023, 1023) on `METIS`; the difference panels of the visual tests were one-pixel dipoles
  of 40-70 % of the peak. Consequence: the bridge rolls the science, DL and OL PSFs by one pixel on that path
  (`_pixel_centered`), which also makes the encircled-energy center (N//2) coincide with the peak. On the interpolation
  path (nOtf > N_pix, e.g. `dummy.ini`) TipTorch's bilinear resampling with `align_corners=False` centers the PSF between
  pixels; this is a TipTorch-core convention left unchanged.
- Observation: `ConfigManager.Convert` drops the `dtype` argument when it recurses into sub-dictionaries, so nested entries
  always get `default_torch_type`. Harmless here because the bridge uses that dtype by default.
- Observation: `pytest` is not installed in the `TipTop` conda env; the test module keeps a `__main__` runner.

## Decision Log

- Decision: one TipTorch model with N_src = nPointings + nNaturalGS directions (science first, NGS last), built in
  `_prepare_state` (first call or `astIndex is None`), not in `__init__`.
  Rationale: this is P3's `getPSDatNGSpositions` layout; it shares the tomographic reconstructor and all PSD terms between
  science and NGS directions in one batched `ComputePSD`, and `asterismSimulation` rewrites the LO field right before the
  calls with `firstSimCall = True`, which is exactly when the model is rebuilt. Date/Author: 2026-10-07, Claude.
- Decision: NGS PSDs, `HO_res` and the OL/DL PSFs are taken at the science wavelength index closest to the shortest
  wavelength (`iRef`). Rationale: P3 evaluates its per-source PSD at `wvlRef = min(wvl)`; TipTorch's PSD differs per
  wavelength only through chromatism and differential refraction. Date/Author: 2026-10-07, Claude.
- Decision: PSDs handed to MASTSEL are zero-padded to even size (`pad_PSD_to_even`, Nyquist row/column in front) only
  when the requested PSF size is even. Rationale: MASTSEL's `even_legacy` mode requires `nPixPsf` parity to match the PSD
  parity when no oversampling rebin happens; padding ourselves keeps `FieldOfView` as requested (the old bridge produced
  201-pixel NGS PSFs for a 200-pixel request). Date/Author: 2026-10-07, Claude.
- Decision: the extra-error, wind-shake, tilt and focus PSDs are members of `TipTorch` returning full-grid tensors that
  broadcast against `[N_src, N_wvl, nOtf, nOtf]`; the RMS inputs may be per-source tensors so P3's science/NGS split of
  `extraErrorNm` / `extraErrorLoNm` is one batched call per spectral shape. `WindShakePSD` reuses `TransferFunctions`
  (P3's `z = exp(-2j*pi*f/rate)` is the conjugate convention, which leaves `abs(sum(rtf**2 * PSD) * df)` unchanged).
  Date/Author: 2026-10-07, Claude.
- Decision: `FocusErrorPSD` added although not in the plan. Rationale: the previous bridge called `FocusFilter(k2)` on the
  full grid, whose `[nOtf_AO//2, nOtf_AO//2]` DC-removal index is only valid on the AO grid. Date/Author: 2026-10-07, Claude.
- Decision: PSF metrics are batched PyTorch re-implementations of P3's definitions (contour FWHM, `round(r)` radial bins,
  EE normalized to its maximum, ensquared squares about the peak) instead of calls into P3 `FourierUtils`.
  Rationale: the plan asks to replace P3 and prefers batched PyTorch; `interpolate_curves` is linear where TIPTOP uses
  cubic `interp1d` (differences stay below the one-pixel binning granularity). Date/Author: 2026-10-07, Claude.
- Decision: `computeMetrics` without PSFs reproduces `baseSimulation`'s formulas (mean penalty, Maréchal SR at `wvlRef`,
  `2.355*LO_res/scale/sqrt(2)` quadrature FWHM) instead of the covariance-eigenvalue variant of the old bridge.
  Rationale: `asterismSimulation` consumes these numbers. Date/Author: 2026-10-07, Claude.
- Decision: no `Super_Sampling` radial-profile interpolation in `computePSF1D` (the option is parsed and written to the
  header only). Rationale: scope; the discrete profile is P3's default path. Date/Author: 2026-10-07, Claude.
- Decision: fix `TipTorch.InitGrids` (`u_max` divided by 4) rather than compensating in the bridge or leaving the kernel.
  Rationale: the measured factor of 2 in sigma and the Nyquist derivation agree; compensation in the bridge would hide a
  core bug. Date/Author: 2026-10-07, user (asked), Claude.
- Decision: mirror the three changed TipTorch files into `C:\Users\akuznets\Projects\TipTorch`, the checkout that the
  `TipTop` env imports. Rationale: otherwise the bridge is unusable from that env. Date/Author: 2026-10-07, user (asked), Claude.

## Concrete Steps

Run from `astro-tiptop/TIPTOP` in the `TipTop` conda env; `PYTHONPATH` makes the `astro-tiptop` TipTorch checkout win over
the editable install of `C:\Users\akuznets\Projects\TipTorch`:

    $env:PYTHONPATH = 'C:\Users\akuznets\Projects\astro-tiptop\TipTorch\src'
    python ..\TipTorch\tests\test_tiptop_integration.py          # expected: "TIPTOP integration checks passed"
    python -c "from tiptop.TipTop_integration import baseSimulation as S; s = S('tiptop/perfTest', 'MAVIStest', 'out', 'mavis', tiptorch_device='cuda'); s.doOverallSimulation(); s.computeMetrics(); s.saveResults(); print(s.HO_res, s.LO_res, s.sr)"

The comparison scripts used during this work live in the session scratchpad
(`scratchpad/ref/run_p3_reference.py`, `run_tiptorch_bridge.py`, `mastsel_scaling_check.py`, `diag_jitter_fwhm.py`,
`error_budget_compare.py`, `fits_compare.py`, `misc_checks.py`); they are throw-away and not part of the repositories.

## Validation and Acceptance

- Unit checks (`tests/test_tiptop_integration.py`): jitter covariance algebra and TipTorch kernel rotation sign, nm² to mas
  conversion, batched FWHM / EE / ensquared / radial profile on Gaussian stacks, `mask_circle` pupil area, even padding keeps
  the DC at n//2, `TiltFilter` = Sasiela formula and zero at the origin, `ExtraErrorPSD` per-source RMS and band limits,
  `FocusErrorPSD` RMS, `WindShakePSD` reproduces P3's temporal integration.
- End-to-end vs the P3-backed `baseSimulation` on `perfTest` configs, see Artifacts and Notes.

## Idempotence and Recovery

All changes are plain source edits; `git checkout -- <file>` in `TIPTOP` and `TipTorch` restores the previous state.
`TipTop_integration.py` is only used when imported explicitly; `tiptop.py` still instantiates the P3 `baseSimulation`.

## Artifacts and Notes

P3 vs TipTorch bridge (`HO_res`, `LO_res` in nm RMS; Strehl, FWHM in mas, EE at 50 mas; `sim` wall time of
`doOverallSimulation`, RTX GPU for TipTorch, CuPy for P3/MASTSEL):

| config | quantity | P3 | TipTorch bridge |
|---|---|---|---|
| ERIS (SCAO, 1.65 µm, 14 mas) | HO_res | 81.1 | 80.5 |
| | SR / FWHM / EE | 0.892 / 44.1 / 0.734 | 0.914 / 43.1 / 0.734 |
| | time | 0.6 s | 2.1 s (0.1 s on CPU) |
| MAVIStest (MCAO + 3 NGS, 9 pointings, 550 nm, 7 mas) | HO_res (9) | 139.8 ... 97.9 | 139.8 ... 95.8 (max 2 %) |
| | LO_res (9) | 37.6 ... 64.0 | 37.5 ... 63.9 (max 0.5 %) |
| | NGS SR / FWHM / EE | 0.803 / 43.8 / 0.636 | 0.806 / 42.9 / 0.638 |
| | SR (9 pointings, with LO jitter) | 0.0675 ... 0.2598 | 0.0674 ... 0.2754 (max +6 %, on-axis 0.193 vs 0.197) |
| | FWHM (9) | 16.3 ... 15.3 | 15.8 ... 15.0 (-3 %) |
| | EE (9) | 0.141 ... 0.294 | 0.140 ... 0.307 (max +4 %) |
| | SR on-axis (no jitter) | 0.263 (P3 `getStrehl` on our PSF) | 0.265 |
| | time | 15.1 s | 2.6 s |
| METIS (ELT SCAO, 3.7 µm, 2048 px, static map, wind, extra, jitter 6 mas) | HO_res | 148.4 | 145.8 |
| | fit / spatio-temporal / chromatism / wind shake / extra | 107.0 / 20.9 / 1.3 / 61.2 / 64.0 | 107.0 / 20.9 / 1.3 / 61.2 / 64.0 |
| | aliasing / noise | 45.3 / 15.6 | 34.0 / 18.8 (TipTorch vs P3 physics, out of scope) |
| | SR / FWHM / EE | 0.862 / 20.51 / 0.838 | 0.865 / 20.49 / 0.837 |
| | time | 8.9 s | 0.9 s |
| dummy (LTAO + 1 NGS 2x2, 2 wavelengths, 20 mas undersampled) | HO_res | 120.6 | 120.9 |
| | LO_res | 54.7 | 63.4 (P3 NGS FWHM is a 1-pixel artifact, see Surprises) |
| | SR (2 wavelengths) | 0.161 / 0.182 | 0.153 / 0.187 |
| | time | 105 s (N = 3200) | 0.9 s (N = 627) |

(Strehl, FWHM and EE values are after the jitter-kernel fix; before it, jittered Strehls were about half of P3's.)

Visual comparison (`python tests/test_tiptop_visual_comparison.py`, figures in `tests/runs/visual_comparison/`, gitignored):
after the centering fix the maximum |TipTorch - P3| relative to the P3 peak is 0.5 % on `METIS`, 1.4 % on the `MAVIStest`
on-axis PSF (7-9 % on the two off-axis pointings, where the elongation directions differ slightly) and 3.4 % on `ERIS`;
the radial profiles overlap over five decades.

FITS output: same HDU list and shapes, identical header key sets on all HDUs (`fits_compare.py` on `MAVIStest`, `METIS`).
The asterism path (`doConvolveAsterism = False`, `doOverallSimulation(0)`, `computeMetrics`) runs in 0.05 s per asterism
after the first call. YAML and INI inputs give identical numbers. Focus sensing (`glFocusOnNGS`) and LO aliasing
(`addAliasError`) paths run on the `dummy` variant.

## Outcomes & Retrospective

Implemented: `TipTorch` now owns the P3-style PSD add-ons (`TiltFilter`, `ExtraErrorPSD`, `FocusErrorPSD`, `WindShakePSD`,
all batched PyTorch, no loops, differentiable in the RMS / temporal-PSD inputs); the helper module shrank to the bridge
algebra plus batched PyTorch PSF metrics and the `mask_circle` pupil; the TIPTOP bridge runs one TipTorch model for all
science + NGS directions and all wavelengths, pads odd PSDs for MASTSEL, and reproduces `baseSimulation`'s attributes,
metrics formulas and FITS layout. Two latent bugs of the previous bridge were found and fixed (MASTSEL `dk` convention, 4x in
NGS spot variance; `FocusFilter` on the wrong grid) and one in the metric design (bilinear ray sampling).

Validated: unit checks pass; HO/LO residuals, NGS spot metrics, wind-shake and extra-error terms, jitter-free Strehl and the
output format agree with the P3 pipeline at the 1-2 % level on SCAO, MCAO and ELT configurations; 3-40x faster.

Resolved with the user: (1) the `JitterKernel` normalization (`u_max`) was corrected in `InitGrids`; jittered Strehls now
agree with P3 within a few percent, but fitted `Jx, Jy` from earlier TipTorch runs correspond to twice their nominal sigma.
(2) `TipTorch.py`, `tools/tiptop_integration.py` and `tests/test_tiptop_integration.py` were copied into the
editable-installed checkout `C:\Users\akuznets\Projects\TipTorch` (uncommitted there, as in `astro-tiptop/TipTorch`), so the
bridge imports in the `TipTop` env without `PYTHONPATH`.

Open: (3) aliasing and noise PSDs differ from P3 by 10-25 % (TipTorch physics, unchanged here). (4) TipTorch resamples
undersampled PSFs with bilinear interpolation rather than a pixel-integrating rebin (`OTF2PSF`), and `Super_Sampling`
radial profiles are not implemented. (5) `tiptop.py` still selects the P3 backend; wiring `backendHO='tiptorch'` to this
class is the natural next step. (6) Nothing is committed in either repository.
