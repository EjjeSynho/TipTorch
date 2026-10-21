# Anisotropic jitter kernel and chromatic defocus for the NFM STD-star fits

## Context

The 2D median residual of the hybrid-loss fits (`datasets/MUSE_STD_dataset/analyze_fit_residuals.py`, `data/analysis/NFM_STD_residuals/fitted/`)
is dominated by a structure fixed in the DETECTOR frame, not in the pupil frame: at 3-8 px from the peak the data are brighter than the model
along the image columns and fainter along the rows (a quadrupole of +/-0.5 % of the peak at 928 nm), at every wavelength, surviving the median
over pupil angles spanning the full circle. The data cores are elongated along the columns in 80 % (485 nm) to 96 % (928 nm) of the stars
(second-moment ellipticity within 8 px: detector-fixed e1 +0.02 blue to +0.06 red, the model +0.01). The model had no instrument-frame
anisotropy at all: the wrapper cloned one isotropic `J` into `Jx` / `Jy` and froze `Jxy` at 0, and the LO NCPAs rotate with the pupil.

User's proposal (2026-10-20): go back to the chromatic-defocus scheme (the chromatic part of the blur is a chromatic defocus term, the jitter
kernel is achromatic) and make the jitter kernel anisotropic: `Jx`, `Jy` and a kernel angle `J_angle` kept aligned with the image axes.

## Changes

- `src/tiptorch/PSF_models/NFM_wrapper.py`: constructor flag `anisotropic_jitter` (saved / loaded / copied). Inputs: `Jx`, `Jy` (achromatic
  `[N_obs, 1]` with `chrom_defocus`, splines `Jx_ctrl` / `Jy_ctrl` otherwise) and `J_angle` [deg, `[N_obs]`]; without the flag the isotropic
  `J` / `J_ctrl` and the frozen `Jxy` stay as before. `_resolve_jitter` maps them onto TipTorch's `Jx` / `Jy` / `Jxy`; `jitter_params` lists
  the active ones and `polychromatic_params` follows. TipTorch builds `JitterKernel` on the image-frame frequency grid, so the kernel does not
  turn with the pupil angle: `J_angle = 0` keeps the axes aligned with the image axes whatever the derotator state. TipTorch's x is the first
  image axis (rows), like `dx`: `Jx` blurs along the rows, `Jy` along the columns.
  Fixed on the way (pre-existing, chromatic-defocus regime): `phase_func` hard-coded the sparse wavelength count (no full-spectrum batches);
  `SimulateSpectralRange` read the chromatic defocus from the wrong dict and restored the static OTF with a full-grid phase; `load()` failed on
  the raw list configs of the multiple-observation regime.
- `datasets/MUSE_STD_dataset/STD_fitter.py`: flags `CHROM_DEFOCUS`, `ANISOTROPIC_JITTER` (both True as requested), passed to the model; the
  achromatic jitter entries join `include_general`; the pickles record both flags.
- `datasets/MUSE_STD_dataset/STD_fitted_df.py`: handles `Jx` / `Jy` / `J_angle` (singulars when achromatic), `Jx_ctrl` / `Jy_ctrl` /
  `chrom_defocus_ctrl` (splines), no longer assumes `J_ctrl` exists.
- `development/MUSE_onsky.py`: same flags, prints the fitted kernel; `mse_weight` added to its loss config.
- Checks (in-session): forward, anisotropy sign flip at `J_angle` 90 deg, full-spectrum render with the sparse state restored, save / load
  round trip, for all four flag combinations and both observation regimes; `tests/test_kc_mismatch_groups.py` passes.

## Results (30-sample subset, every 18th ID; `data/analysis/NFM_STD_residuals/jitter_regimes/`)

Medians over the samples of the per-sample medians. loss = balanced + MSE loss of the fitter config; e1 resid = data - model ellipticity
(image frame, r <= 8 px); quadrupole = median residual in the vertical minus the horizontal lobes at 3-8 px, λ average [% of peak].

| regime | loss | |peak err| | FWHM err | shape err 1.5-3.5 / 3.5-10 px | e1 resid | quadrupole |
|---|---|---|---|---|---|---|
| chromatic isotropic J (fitted/, reference) | 0.150 | 4.37 % | 5.1 % | 7.8 / 8.2 % | +0.015 | +0.21 |
| chromatic defocus + isotropic J | 0.151 | 5.82 % | 3.2 % | 7.0 / 8.3 % | +0.017 | +0.29 |
| chromatic anisotropic Jx, Jy, J_angle | 0.119 | 4.89 % | 4.8 % | 6.5 / 7.6 % | +0.015 | +0.25 |
| chromatic defocus + achromatic anisotropic | 0.121 | 5.52 % | 3.6 % | 6.7 / 7.7 % | +0.015 | +0.25 |
| the two chromatic regimes with 400 instead of 200 iterations | same | same | same | same | same | |

- The anisotropic kernel lowers the loss by 9-11 % in 26-28 of 30 stars and the 1.5-10 px shape errors by ~0.6 %. It fits `Jy` (columns)
  18-19 mas against `Jx` (rows) 13-14 mas with `J_angle` 3-5 deg: the elongation direction and orientation the residual maps predicted.
- It does NOT remove the quadrupole: the lobe difference stays at +0.25 % of peak, the e1 residual at +0.015. The residual lives at 3-12 px
  (data / model along the columns rises from 1.0 at 2 px to 1.1-1.2 at 6-12 px at 791-928 nm; along the rows the model is 4-5 % too bright),
  where a 15-20 mas Gaussian only moves flux between the core and the first ring. The missing component is a wide, anisotropic wing in the
  detector frame, not a jitter.
- The chromatic defocus is not used by the fits (median node values 0-2 nm) and costs peak accuracy (5.8 % vs 4.4 % isotropic, 5.5 % vs
  4.9 % anisotropic) while improving the FWHM error; the chromatic jitter spline absorbs the chromatic blur better.
- The peak error of the anisotropic fits is 0.5 % worse than the reference and does not recover with twice the iterations: a genuine trade-off
  of the Gaussian kernel (the fit widens the kernel along the columns to feed the 3-8 px lobes and softens the peak).
- The wind direction of the servo-lag term (not fitted, image frame) is not the origin of the detector-fixed asymmetry (its share of the
  ellipticity is < 0.01 for both data and model).

## Decision log

- Implemented exactly the proposed scheme plus the chromatic anisotropic variant, so the two ingredients could be separated; defaults in
  `STD_fitter.py` left at the proposed regime (`CHROM_DEFOCUS = ANISOTROPIC_JITTER = True`) pending the user's decision.
- Not propagated to the calibrator (the user's plan: fits first); the calibrator keeps `chrom_defocus=False`, isotropic `J_ctrl`.
- Pre-existing index mismatch noticed, not changed: the fitter's "first defocus positive" penalty acts on `LO_coefs[:, 2]`, but the composite
  basis is [bump, defocus, astig, ...] (Zernike modes start at Noll 2, `basis[2:]` starts at defocus), so index 2 is astigmatism; the
  wrapper's chromatic defocus correctly uses index 1.

## Outcomes & next steps

- Evidence favours the chromatic ANISOTROPIC jitter (`CHROM_DEFOCUS = False`, `ANISOTROPIC_JITTER = True`) over the proposed regime if the
  anisotropic kernel is adopted: same loss gain, better peak. Whether the 10 % loss gain is worth 0.5 % of peak error is the user's call.
- The detector-frame quadrupole needs a different ingredient: a wide anisotropic kernel fixed in the image frame (e.g. a 1D Lorentzian /
  Moffat along the columns, the slicer / spectrograph spatial spread), or an anisotropic PSD term. Its radial run (ratio 1.0 at 2 px rising to
  1.2 at 6-12 px in the red) is the constraint to fit it against.

## Image-frame wing kernel (2026-10-20, second step)

**Implemented** (`TipTorch.WingKernel`, `PSFModelNFM(wing_kernel=True)`, `STD_fitter.py` `WING_KERNEL`): a multiplicative OTF factor on the
image-frame grid next to the jitter kernel, `(1 - f) + f * exp(-(2 pi w |nu|)^p)` with `nu` the spatial frequency along the image columns
rotated by `wing_angle`, i.e. a one-dimensional Lorentzian wing (p = 1) of half-width `w` [px] holding the flux fraction `f`. Inputs
`wing_frac_ctrl` (spline over λ), `wing_width`, `wing_angle` (fixed at 0 unless `FIT_WING_ANGLE`); unity at zero frequency, muted with the
jitter in `OLPSF`, not applied to the DL PSF. Default off everywhere: the pre-change regime (chromatic isotropic `J_ctrl`, no wing) is the
wrapper's default and reproduces the stored fits bit for bit (checked on ID 100: max |diff| / peak = 0). `STD_fitter.py` defaults:
`CHROM_DEFOCUS = ANISOTROPIC_JITTER = False`, `WING_KERNEL = True`.
Checks: `wing_frac = 0` gives the plain model exactly; f = 0.02, w = 10 px raises the 20-30 px column / row ratio of the model from 0.99 to
1.06 (928 nm) and `wing_angle = 90` turns it below 1; full-spectrum render restores the sparse state; save / load; `OLPSF`; the k_c
regression test passes. Fixed on the way: TipTorch re-runs its parameter block on grid updates, which reset the wing switch.

**Results, 30-sample subset** (same table conventions; quadrupole = vertical minus horizontal lobes of the median residual, % of peak;
col/row = data / model ratio of the column to row sector means):

| regime | loss | |peak err| | FWHM err | quadrupole 3-8 px | col/row 10-30 px (red) | e1 resid | shape err 1.5-3.5 / 3.5-10 px |
|---|---|---|---|---|---|---|---|
| chromatic isotropic J (reference) | 0.150 | 4.37 % | 5.1 % | +0.18 | 1.12 | +0.015 | 7.8 / 8.2 % |
| anisotropic jitter | 0.119 | 4.89 % | 4.8 % | +0.20 | 1.11 | +0.015 | 6.5 / 7.6 % |
| wing + isotropic jitter | 0.134 | 4.38 % | 2.6 % | +0.006 | 1.00 | +0.003 | 6.6 / 7.2 % |
| wing + anisotropic jitter | 0.109 | 3.87 % | 3.3 % | +0.009 | 1.00 | +0.003 | 6.0 / 6.9 % |

The wing removes the detector-frame quadrupole (3-8 px and 10-30 px) and the column / row excess at every wavelength, improves the loss in
29-30 of 30 stars (-11 % alone, -18 % with the anisotropic jitter), keeps or improves the peak error and halves the FWHM error. Fitted
values (free width, unbounded): w ~ 1.0 px, f from 0.07-0.09 (485 nm) to 0.26-0.36 (928 nm).

**Degeneracy.** The (f, w) pair is poorly constrained along a valley: from starting widths of 0.5 / 3 / 8 px the same star converges to
w = 48-57 px with f = 0.03-0.05 (ID 100), to w = 0.1 px with f > 1 (unphysical; ID 236) or w = 0.9 px with f = 0.1, or to a stable
w = 1.4-1.5 px (ID 350), at equal loss. Hence (a) penalties keep f in [0, 1] and w in `WING_WIDTH_BOUNDS` = (0.5, 20) px, (b) `FIT_WING_WIDTH =
False` fixes the width at `WING_WIDTH_PX` (1 px, the value the free fits prefer when they stay narrow). Comparison of the bounded free width
and the fixed 1 px width on the subset: see below.

**Bounded free width** (`wing_bnd`: f in [0, 1], w in [0.5, 20] px, start 1 px): loss -11.1 % vs the reference in 30 / 30 stars, |peak err|
4.31 %, FWHM err 3.4 %, quadrupole -0.04 / +0.003 (3-8 / 10-30 px), col/row 1.00, e1 resid +0.001; width p16 / p50 / p84 = 0.5 / 1.15 /
4.3 px, red fraction 0.05 / 0.27 / 0.47, no star above f = 1 (7 of 30 without the bounds). The fixed 1 px width (`wing_w1`) is reported
below (a first run of it still fitted the width: `set_optimizable` only touches the listed names, so an input created optimizable stays
fitted unless excluded explicitly; the fitter now excludes `wing_width` when `FIT_WING_WIDTH = False`. The same mechanism leaves `L0` and
`wind_speed_single` fitted when their flags are off: pre-existing, not changed).

**Fixed 1 px width** (`wing_w1`, width excluded from the fit): loss -9.7 % in 29 / 30 stars, |peak err| 4.44 %, FWHM err 3.3 %, quadrupole
-0.04 / +0.006, col/row 1.01, e1 resid +0.001, red fraction p16 / p50 / p84 = 0.13 / 0.19 / 0.27: the quadrupole is gone as well, with a
tighter fraction distribution, at a slightly worse loss and peak than the bounded free width (0.128 / 4.31 %).

**Recommendation:** wing on with the bounded free width (`FIT_WING_WIDTH = True`, the fitter default) for the fits; for the calibrator, the
fixed 1 px width is the simpler target (one chromatic parameter `wing_frac` to predict, no degenerate width). The anisotropic jitter on top
is a further -7 % of loss and -0.4 % of peak error (`wing_aniso`) and is the user's choice, as before.

## Adopted regime and the wing width (2026-10-20, user's decision)

Adopted: wing + anisotropic chromatic jitter (`Jx_ctrl`, `Jy_ctrl`, `J_angle`), no chromatic defocus. To remove the (f, w) degeneracy the width
is fixed at a dataset-wide value and only the chromatic fraction is fitted per star (as the wing will be a shared, LO-NCPA-like quantity in the
calibrator). Fixed-width scan on the subset in that regime (`data/analysis/NFM_STD_residuals/jitter_regimes/eval_scan.csv`):

| width [px] | loss | |peak err| | red-bin |peak err| | quadrupole 3-8 px | FWHM err | f at 485 / 928 nm |
|---|---|---|---|---|---|---|
| free (bounded) | 0.109 | 3.87 % | 6.7 % | +0.009 | 3.3 % | 0.07 / 0.26 |
| 0.5 | 0.108 | 3.96 % | 6.9 % | +0.048 | 3.3 % | 0.03 / 0.27 |
| 0.75 | 0.108 | 3.85 % | 6.4 % | +0.029 | 3.6 % | 0.03 / 0.21 |
| 1.0 | 0.110 | 4.02 % | 7.0 % | +0.024 | 3.1 % | 0.03 / 0.18 |
| 1.5 | 0.110 | 3.85 % | 7.0 % | +0.013 | 3.9 % | 0.03 / 0.15 |
| 2.0 | 0.110 | 4.17 % | 6.9 % | +0.022 | 4.1 % | 0.03 / 0.12 |
| 3.0 | 0.110 | 4.03 % | 7.0 % | +0.043 | 4.1 % | 0.04 / 0.09 |

The loss is flat from 0.5 to 3 px (within 1 % of the free width, 9-15 of 30 stars worse by > 1 %): the width is exchangeable against the
fraction, which is the degeneracy itself, so fixing it costs nothing. 1 px is kept (`WING_WIDTH_PX = 1.0`, `FIT_WING_WIDTH = False`): the
middle of the plateau and where the bounded free fits converge most often (median 1.15 px). Fitter defaults now: `CHROM_DEFOCUS = False`,
`ANISOTROPIC_JITTER = True`, `WING_KERNEL = True`, `FIT_WING_WIDTH = False`, `WING_WIDTH_PX = 1.0`; the same in `development/MUSE_onsky.py`.
The wrapper's own default (all flags off) stays the pre-change isotropic `J_ctrl` model.

Next (user): full refit of the dataset with these defaults; then the calibrator with `wing_frac` as a shared (dataset-level) chromatic
parameter like the LO NCPAs and the width fixed at 1 px.

## Blurry PSFs: dataset priors instead of free parameters (2026-10-20)

The column excess is an instrument property, equally present in the 77 blurry stars (data column / row sector ratio at 10-30 px, red: 1.093 vs
1.087 for the sharp ones) but a blurry PSF cannot constrain the wing fraction (subset: sharp stars give 0.04 -> 0.18 from 485 to 935 nm,
the blurry ones scatter 0.03-0.30 with no trend), nor the LO NCPAs, L0, the wind speed or the Cn2 profile. Implemented (user's decision):
for `blurry_PSF_flag` stars the phase bump stays suppressed (pulled to 0), the LO coefficients (bump excluded) and the wing fraction are
pulled to the dataset medians of the sharp stars with the x1000 weight already used for the LO suppression (they end within 1 nm / 0.003 of
the priors), and L0 / wind speed are excluded from the fit (they are created optimizable, so leaving them out of the include list was not
freezing them) and stay with the Cn2 weights at the reduced-telemetry values of the config. Sharp stars: unchanged (a weak wing pull,
`w_wing_prior` = 1e-2, and `w_LO_prior` = 0).
Code: `STD_dataset_utils.compute_fit_priors` / `load_fit_priors` / `blurry_PSF_mask` and the file `fitted_priors.pickle` in the STD folder
(refreshed by `STD_fitted_df.py` after each fit of the dataset; the fitter warns and falls back to 0 / the built-in wing curve without it);
`STD_fitter.py` (`FIT_PRIORS`, loss terms, exclusions, pickle entries `blurry` and `priors`); `development/MUSE_onsky.py` same logic.
Bootstrap priors (2026-10-20): LO coefficients from the 457 sharp stars of `fitted/` (isotropic-jitter fits: [0, 0, 0, 0, 36, 35, -13, 10] nm),
wing fraction / Jx / Jy / J_angle from the 25 sharp stars of the subset in the adopted regime.
Check on two blurry subset stars (350, 491): priors reproduced, loss 0.148 / 0.111 vs 0.132 / 0.101 with everything free, peak error 11.4 /
10.4 % vs 10.1 / 4.1 %: the price of not letting unconstrained parameters absorb the residual of a blurry PSF.

## Defocus sign penalty and dev-script alignment (2026-10-20)

- The "first defocus positive" penalty acted on `LO_coefs[:, 2]` (astigmatism): the composite basis is [bump, defocus, astig, astig, coma,
  coma, trefoil, trefoil], so defocus is index 1. Fixed in `STD_fitter.py`, `development/MUSE_onsky.py` and `tools/observations.py` (the
  calibrator's `LO_sign_penalty`, off in the configs, intentionally penalizes c[:, 2] and is unchanged). With the `w_LO` weight (1e-7) the
  penalty was ineffective anyway (a -70 nm defocus cost 5e-4 against a loss of 0.13): it now uses `w_bump` (5e-5) when the phase bump is
  suppressed, where the sign of defocus is a pure convention, and stays weak for phase-bump stars, where the sign is identifiable (the
  calibrator notes that forcing it there biases the fit). Effect on 477: defocus -69 nm -> ~0 (astigmatism takes -52 nm), loss +1.5 %,
  peak error at 485 nm 13 % -> 4-7 %, median |peak err| 3.6 % -> 2.7 %; on 100: defocus +2 nm, loss 0.1132.
- `development/MUSE_onsky.py` now follows `STD_fitter.py`: MAP prior on with the same prior widths (dataset spread of the telemetry values),
  wind direction not fitted, `F_norm_λ_ctrl` excluded from the fit, loss config keys `w_LO_prior` / `w_wing_prior`. Without the MAP prior
  the outer scale of 477 ran to 3.5 m (trimming the model's 5 % too bright red halo at 20-40 px at the expense of the seeing-dominated blue
  core: 485 nm peak error 13.6 % vs 1.1 % with L0 held at the telemetry 15 m). Remaining deliberate difference: 8 spline nodes in the dev
  script vs 5 in the fitter (8 nodes halve the blue centroid lag of the chromatic position track; 477 is bistable in the blue: 5 / 8 / 10
  nodes gave 9-13 / 0.6-1 / 15 % peak error at 485 nm in different runs, the two defocus-sign branches).
- Blue-mismatch findings for the six "poor blue" stars (338, 457, 477, 467, 458, 494): in the dev script the first cause was the legacy loss
  (338: 485 nm profile error 15.6 % -> 4.6 % with the balanced loss), the second the free outer scale (477, above); in the dataset fits they
  are not outliers (485 nm peak-error ranks 52-381 of 534). Dataset-wide the model centroid lags the data by 0.07 px at 485 nm (0.03 px
  elsewhere), partly the 5-node natural spline, partly a fit compromise; background matching and a larger loss radius do not help the blue.

## Defocus sign convention (2026-10-20, user's request)

- Data: among the 103 phase-bump stars with a fitted bump > 20 (sign of defocus observable through the bump) the fitted defocus is positive
  in only 56 % (median +9 nm, peaks at +/-30-60 nm, sign varying by year), so the physical defocus has no fixed sign and stays free for
  bump stars; the bump itself stays positive always (reference-slope bias). For the bump-suppressed stars the sign is unobservable: the PSF
  is invariant under the joint sign flip of the EVEN Zernike modes (defocus + astigmatisms; comas and trefoils keep their sign), which is why
  a penalty on defocus alone drove defocus to 0 and moved the content into the astigmatisms (477: astigmatism -52 nm).
- Implemented (`STD_fitter.py` `DEFOCUS_SIGN = 'flip'`, `DEFOCUS_INIT_NM = 20`, helper `even_zernike_composite_ids` in STD_dataset_utils;
  same in `development/MUSE_onsky.py`): bump-suppressed stars start at +20 nm defocus and, if the fit ends with a negative defocus, the even
  modes are reflected and the fit re-run from the mirror branch (accepted when within 0.5 % of the loss; pickles record `even modes
  flipped`). 'penalty' (strong positivity penalty) and 'free' remain as options.
- 30-sample subset (`data/analysis/NFM_STD_residuals/jitter_regimes/eval_sign.csv`): all conventions have the same loss (0.108-0.111);
  'free' gives a random sign (36 % positive), 'penalty' |defocus| 8 nm (zeroed), 'flip' keeps the magnitude (|defocus| 34-50 nm) with all
  free stars positive. Flip from 0 vs from -20 nm: same sign in 26 / 30 (the 4 others are blurry stars pinned at the prior ~0), loss
  difference 0.1 % median; the +20 nm start gives the fewest reflections (30 %) and the best peak error (3.7 % vs 4.0-4.1 %), hence the default.
- Remaining non-uniqueness is not the sign: a few stars (21, 120, 452) land in different local minima from different starts (defocus
  magnitude differing by 30-50 nm, loss up to 13 % apart): a two-start fit (keep the lower loss) would be the next step if it matters.
- `development/MUSE_onsky.py` (2026-10-20): multi-start fit (`DEFOCUS_STARTS`, default +20 / -20 nm; `fit_from` runs one full fit per start
  with the retry and the even-mode reflection, the lowest loss is kept). On 477 both starts reach loss 0.1291 with a positive defocus (the
  optimizer crosses from -20 nm to +67 nm by itself), yet the 485 nm peak error still differs between runs of equal loss (0.8-17 %): the
  blue peak of this star lies along a flat direction of the loss, which no start strategy fixes; it would need a loss term that sees it.

## Per-wavelength peak term of the loss (2026-10-20)

- The blue PEAK error of 477 swung between runs of equal loss (0.3 / 4 / 11 / 13 / 18 % at 485 nm): the blue core of this star is one-sided
  (2-6 px: data 4-10 % of peak brighter than the model in one 90-deg sector), the model cannot represent it, and nothing in the loss looks at
  the per-wavelength peak (the MSE term is dominated by the red peaks). Added `BalancedPSFLoss(peak_weight, peak_box=7)`:
  `peak_weight * (max(model) / max(data) - 1)^2` per wavelength within the central 7 px. Default 3.0 in `STD_fitter.py`, the dev script and
  the two NFM calibrator configs (kept in sync like `mse_weight`).
- 477, both starts: weight 3 keeps the 485-570 nm peaks within +/-2.5 % (median |peak err| 2.2-2.5 % vs 2.7-2.9 % without), reference loss
  +1.2 %. 30-sample subset (`eval_peakw.csv`): median |peak err| 3.68 -> 2.78 (weight 1) -> 2.50 % (weight 3), bluest four bins 4.2 -> 1.7 %,
  worst bin 12.4 -> 9.4 %, FWHM error 3.3 -> 1.6 %, halo shape errors unchanged (6.0 / 7.3 / 5.2 / 6.9 % at 1.5-40 px), reference asinh + MSE
  loss +1.4 % (19 / 30 stars worse by > 1 %).

## Elliptical wing: transverse width (2026-10-20, user's request)

**Problem.** With the 1 px Lorentzian line the model showed a razor-thin bright streak through the core along the columns in the reddest bins
(f up to 0.2-0.35 at 935 nm): visible in the model images and as a thin model-minus-data line in the residuals of the sharp red PSFs. The data's
excess over the no-wing model is a *band*: transverse HWHM 3-4 px (17 % of the local halo on the axis, 7 % at 4-6 px off it, |x| 6-30 px),
flat along the axis (12-15 % of the local halo from 5 to 30 px, dropping beyond the control radius). Widening the 1D line along its axis makes the
streak brighter, not fainter (the Lorentzian tail scales with f·w; the width scan 0.5-3 px overshot the data's band more at 8-15 px).

**Change.** `TipTorch.WingKernel(frac, width, angle, width_t)`: MTF `(1 - f) + f exp(-(2π sqrt((w ν_a)² + (w_t ν_t)²))^p)`; `width_t = 0`
reproduces the 1D line, `width_t > 0` is an elliptical Moffat (β = 1.5 for p = 1) halo. New model attribute / wrapper input `wing_width_t`
(achromatic, shared, non-optimizable by default, 0 for old pickles). Flags `WING_WIDTH_T_PX`, `FIT_WING_WIDTH_T`, `WING_FRAC_MAX` (penalty cap,
1.0 = off) in `STD_fitter.py` and `development/MUSE_onsky.py`; `STD_fitted_df.py` carries `wing_width_t`. Note that in 2D the along-axis tail
falls as r⁻³ instead of the line's r⁻², so the along-axis width must exceed the transverse one for the wing to stay elongated along the columns.

**Scan (30-star subset, current fitter: peak term, flip convention; `scratchpad/lossvar/wreg_*`, `eval_wreg.py`).** Medians; "line" = on-axis
row minus the rows 4-6 px off the axis at |x| 8-25 px in % of the local ring, reddest 4 bins (data 14.9 %); overshoot = model - data per star.

| wing (w × w_t px) | loss | peak err | quad 3-8 [% peak] | line model | overshoot p50 / p90 / max | f (935 nm) | halo bias 40-60 px |
|---|---|---|---|---|---|---|---|
| 1 × 0 (line) | 0.121 | 2.46 % | 0.057 | 16.3 | +3.3 / +17.6 / +28 % | 0.185 | +25 % |
| 1 × 3 | 0.136 | 2.41 | 0.21 | 7.0 | −9.3 / −1.3 / +1.4 | 0.027 | +28 |
| 3 × 3 | 0.132 | 2.19 | 0.19 | 7.4 | −9.0 / −0.9 / +1.7 | 0.091 | +23 |
| **5 × 1.5** | **0.122** | **2.48** | **0.045** | **13.3** | **−2.3 / +4.4 / +6.9** | **0.100** | **+22** |
| 5 × 3 | 0.124 | 2.41 | 0.09 | 9.3 | −6.7 / +0.6 / +3.4 | 0.129 | +17 |
| 5 × 5 | 0.134 | 2.22 | 0.19 | 7.5 | −8.8 / −0.9 / +1.7 | 0.083 | +22 |
| 10 × 3 | 0.125 | 2.40 | 0.10 | 13.8 | −1.5 / +5.0 / +10 | 0.080 | +22 |
| free (→ 1.6 × 0.5) | 0.117 | 2.47 | 0.036 | 12.5 | −0.7 / +4.1 / +6.8 | 0.184 | +22 |

No wing: quad 3-8 = 0.20 % of the peak. Round (w_t ≥ w) wings lose the quadrupole suppression (0.2 % again) and 4-7 % of loss; 5 × 1.5 px keeps
the loss and peak error of the line within 1 %, keeps the quadrupole suppressed, never overshoots the data's band by more than 7 % of the local
halo (28 % for the line), halves the wing flux and reduces the far-halo deficit. Free widths return to a thin line (the loss weighting barely sees
the streak), so the widths stay fixed. No amplitude cap is needed (`WING_FRAC_MAX = 1.0`): the fraction drops by itself with the 2D shape.

**Adopted defaults:** `WING_WIDTH_PX = 5.0`, `WING_WIDTH_T_PX = 1.5`, widths fixed, fraction free per star. Wing-fraction prior
(`WING_FRAC_PRIOR_DEFAULT` and `fitted_priors.pickle`) refreshed to the 5 × 1.5 px medians [0.046, 0.055, 0.066, 0.065, 0.098]. Calibrator: the
shared wing is now two fixed widths + the chromatic fraction. Figures: `scratchpad/lossvar/streak_look.png` (line), `wreg_look.png` (variants),
`streak_shape.png` (data's band profile).

## Fitter building blocks, two-start fit in the fitter, dev script on top of them (2026-10-20, user's request)

**STD_fitter.py** is now a set of module-level functions that `load_and_fit_sample` composes, so the dev script can reuse them instead of
mirroring them: `sample_fit_flags(ids)` (blurry / bump / fit-L0-wind-Cn2 regime from muse_df, several OBs = nothing suppressed),
`build_PSF_model(configs, device)` (model flags, fixed wing widths and power, backup state), `build_loss(PSF_model, PSF_0, flags, device)`
→ `(func, loss_fn)` (PSF loss, LO regularization with the dataset priors, MAP terms, wing bounds), `make_minimizer(PSF_model, func, loss_fn)`
→ `minimize_params(include, exclude, max_iter, ...)`, `parameter_sets(PSF_model, flags)` → `(include, exclude)`, `peak_error(...)` and
`multistart_fit(...)`. The dev script's two-start fit is the fitter's default: `DEFOCUS_STARTS = (20.0, -20.0)` [nm], `SELECT_BY = 'peak'`
(least median |peak error|; 'loss' = lowest loss), `FLIP_TOL = 0.005`; `DEFOCUS_INIT_NM` is gone (the starts replace it, and they apply to
the phase-bump stars too, which started at 0 before). The pickles record `defocus starts`, `selected by`, `start`, `start results`
({start: (loss, peak error, defocus, flipped, converged)}) and `peak error`.

**Check (30-star subset):** the refactored fitter with a single +20 nm start reproduces the previous fits (loss within 0.05 % for 25/30 stars;
the 5 others differ by 1-4 % and are the phase-bump stars, which now start at +20 nm instead of 0, not a regression of the code). The two-start
default keeps the same loss (median ratio 1.000) and peak error (2.23 % vs 2.24 %) on this subset; it matters for the bistable stars (477:
blue peak error 0.6-13 % between branches). Cost: the fit time doubles (~30 s per star on one GPU).

**development/MUSE_onsky.py** keeps only what the dev session needs: the sample / wavelength / derotation / k_c choices, the model built with
`fitter.build_PSF_model`, the objective and parameter sets from `fitter.sample_fit_flags / build_loss / make_minimizer / parameter_sets`,
`fitter.multistart_fit(verbose=True)` and the plots. A regime change is an override of a fitter flag before building the model
(`fitter.WING_WIDTH_T_PX = 3.0`, `fitter.DEFOCUS_STARTS = (20.0,)`, ...). The headless run on 477 matches the previous standalone flow.

**STD_fitted_df.py** stores the fit priors explicitly (`compute_fit_priors(..., out_path=FIT_PRIORS_FILE, store=True)`); the priors now include
`wing_width` / `wing_width_t`, and the fitter warns at import when the priors come from fits with another wing shape.
