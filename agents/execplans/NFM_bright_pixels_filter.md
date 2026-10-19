# Central pixels mismatch for NFM STD stars

## Context

While studying the residuals of the fitting to the STD stars residuals, I noticed large (~10-30%) errors on peaks, centered around just a few bright central pixels. My hypothesis that it may happen due the overexposure of the brigtest pixels and charge overfill, which cause these pixel to output incorrect values. Hence, such pixels values can vary significantly between spectral slices, hich doesn't look like a real astrophysical spectral signal with narrom lines. When processing other on-sky NFM observations with the routines from `tools/observations.py`, I actually cover such cases with `_detect_bad_exposures()`.


**What speaks for this hypothesis:** the rest of the PSF fits super well to the data, apart from just a few pixels in the center
Meanwhile, trying to fit the few central sharp pixels introduces bias the entire core since model produces "smooth" PSFs.

**What speaks against:**  the peak error seem to be present at all averaged spectral slices. Also it's a bit strange that this error survives averaging over the spectral bin.


## Instructions

- Find samples with the worst peak accuracy that can be attributed to a few worst pixels. After this, give me the suspicious IDs, so I can upload the relevant full-spectral cubes to this machine.

- Check the functions inside `tools/observations.py` like `_detect_bad_exposures()` and connected, see if you can re-use the same logic when processing the STD stars data.

- Then, investigate the full spectral cubes with a procedure like `_detect_bad_exposures()`.

- You can also investigate pixel std maps. But I warn you that hey might no show anything interesting.

- Would be nice to move code/functions that can be re-used both for observations and STD dataset compiler into the `MUSE_data_utils.py`.


## Goal

I want you to prove of disprove this hypothesis. If it's false, then what could be the reason behind such bad peak behaviour?

## Possible solution

If the problem is, indeed, in overexposed pixels, produce a mask for them.


---

## Progress (2026-10-16)

- [x] Peak-error survey of all 534 fitted STD stars (`datasets/MUSE_STD_dataset/analyze_peak_pixels.py`, outputs in
      `data/analysis/NFM_STD_peaks/`: `peak_pixels.csv` per sample, `peak_pixels_per_wvl.csv` per (sample, bin), overview and worst-12 gallery PNGs).
      Metrics per bin: peak error, worst-pixel share of the central 7x7 residual, 8-neighbour ring error, data/model sharpness ratio, and from the
      raw cached cubes the spectral-variability excess of the peak pixel (CV from the per-bin `std` map, relative to its 8 neighbours), an exposure
      proxy (peak flux per slice x exposure time) and single-slice spikes of the 11x11 box spectrum.
- [x] Alternative explanations tested: derotation resampling (scipy cubic-spline `rotate` of the data before fitting), sub-pixel centroid position,
      model rendering (bilinear point sampling vs pixel-integrating 'area' downsampling), fitted jitter.
- [x] Refit experiments on 17 samples (12 worst of both signs + 5 controls), four variants: derotated/raw data x bilinear/area rendering
      (temporary copies of `STD_fitter.py`, results kept in the session scratchpad `fitted_raw/`, `fitted_area_derot/`, `fitted_area_raw/`).
- [x] Shared QC code moved to `data_processing/MUSE_data_utils.py`: `heal_and_smooth_spectrum`, `approx_peak_SNR`, `detect_flickering_peaks`
      (the body of `MUSEObservation._detect_bad_exposures`), `spectral_outlier_scores` (the spectral part of `MUSEObservation.FilterHotPixels`),
      `peak_variability_excess` (binned-cache metric), constant `NA_GAP_SLICES`. `tools/observations.py` now delegates to them; the old and new
      implementations give identical outputs on synthetic data (checked in-session), regression test `tests/test_bright_pixel_qc.py` passes.
- [x] `datasets/MUSE_STD_dataset/inspect_full_cube_peaks.py`: the full-cube investigation (flicker detection, STAT-whitened hot-pixel scores at the
      peak and its ring, ratio scatter vs noise, bias of the binned peak from flagged slices, optional per-voxel mask `.npz`). Smoke-tested without
      cubes only: the full cubes are not on this machine.
- [ ] Run `inspect_full_cube_peaks.py` on the uploaded cubes of the suspicious IDs (below) and decide on the mask.
- [ ] Decide whether to refit the dataset in the raw frame (`DEROTATE_PSF = False` in `STD_fitter.py`, separate output folder) and rebuild the
      dataset cache / calibrators from it.

## Suspicious IDs (upload the full cubes of these)

| group | IDs | why |
|---|---|---|
| persistent in the raw-frame refit, data core sharper than any model | 5, 252, 366, 525, 518, 300, 350, 412 | +15-35 % peak error in both frames; the full cube tells whether the peak pixel is a stable sharp core or an unstable pixel |
| persistent, data core flatter than the model | 489, 404, 406 | -5 to -25 % in both frames; 489 has a flat-topped 2x2 core in the top exposure decile (saturation-like) |
| largest spectral-variability excess of the peak pixel in the binned cache | 4, 577, 27, 264, 383 | CV excess 16-31x at single bins; the candidates for genuine flicker / cosmic rays |
| derotation-artefact controls | 469, 409 | 64 % / 20 % peak error in the official fits, 8 % / 5 % in the raw frame; should look clean in the full cube |

Command once the cubes are on this machine (filenames are matched to the IDs through `muse_df['Filename']`):

    python datasets/MUSE_STD_dataset/inspect_full_cube_peaks.py --cubes <folder> --write-masks

## Surprises & Discoveries

- The official fits are made on DEROTATED frames (`RotatePSF`, scipy `rotate`, cubic spline) of an undersampled core (λ/D = 0.6-0.96 px at
  25 mas/px). The resampling reshapes the central pixel: one rotation of the fitted model's own peak loses up to 14 % (5th percentile per bin), and
  the "model too sharp by 20-120 % at one pixel" cases are those with the largest rotation blur (red bins with model blur < 0.85: median peak
  error -14 %, 54 % of them below -10 %). Refitting the same stars on the raw frames (model rotated instead) removes them: 469 -64 % -> +8 %,
  412 -24 % -> (+35 %, see below), 409 -20 % -> +5 %, 404 -19 % -> -4 %, 345 +24 % -> +3 %.
- Over the whole dataset the peak error is small: median |peak err| 4.0 % per sample (per-bin median 3-6 %, 16-84 % range +/-8 % in the blue,
  +/-12-19 % at 846-928 nm), 58 % of the bins with the data peak above the model.
- The spectral std maps of the cache do show something, but not what the hypothesis needs: 359 of 8544 (sample, bin) pairs have a peak-pixel
  CV excess > 2 over the ring. The implied number of deviant slices is 1-2 per bin (median; p90 9, of ~110): single-slice events (cosmic rays /
  isolated bad voxels) that bias a bin's sum by <~2 %. Their peak error is unbiased (median -0.1 %, |.| 6.8 % vs 4.1 % overall). The excess does
  not follow the exposure level at all (Spearman -0.02 with peak flux x exposure time), which rules out a saturation / charge-overflow origin.
- The remaining large errors in the raw frame are positive (data core sharper than the model): the fitted jitter is 10-21 mas for every star
  (median 15.5 mas, ~1.5 px FWHM of blur) and the model cannot put a single pixel 1.4-1.8x above its neighbours while matching the ring and
  the AO halo. Pixel-integrating rendering ('area') makes this worse (raw frame: median |peak err| 16.6 % vs 9.0 % bilinear), so the model's
  point sampling of the oversampled PSF is not the cause.
- Latent issue noticed, behaviour preserved: `MUSEObservation._detect_bad_exposures` passes the ROI-local coordinates to the SNR estimate, which
  compares them with the global `x_peak` / `y_peak` of the other sources (the exclusion of neighbouring sources from the noise border is therefore
  ineffective unless the ROI starts at the origin). Now visible as the `roi_coords` argument of `approx_peak_SNR`.
- Telluric / stellar lines (O2 A-band at 764 nm, B-band at 687 nm, H-alpha 656 nm, H-beta 486 nm, water at 928 nm) dominate the per-bin
  spectral std of every pixel; only the ratio to the neighbouring pixels isolates a pixel's own variability.

## Decision Log

- Peak error defined as in `machine_learning/calibrators/NFM_validation_utils.py` (max of the central 7x7, (data - model) / data) so the numbers
  are comparable with the calibrator validation.
- No mask produced yet: the evidence points away from over-exposed pixels for the bulk of the error, and a mask built from the binned cache
  alone (CV excess) would act on bins biased by <~2 %. The mask machinery is in `inspect_full_cube_peaks.py --write-masks` for the case the
  full cubes show otherwise.
- The flicker detector was moved, not changed: `detect_flickering_peaks` reproduces `_detect_bad_exposures` exactly (same thresholds, the
  adaptive quantile threshold still flags ~1 % of the slices of a perfectly clean source, so its output is a candidate list, not a mask by itself).
- Model rendering left as is (bilinear): the 'area' variant was worse in both frames.
- The refit experiments used temporary copies of `STD_fitter.py` (deleted afterwards); the official `fitted/` folder was not touched.

## Outcomes & Retrospective

- Hypothesis (over-exposed, flickering central pixels bias the binned peak): **not supported** by the data available on this machine. The
  peak-pixel variability is uncorrelated with the exposure level, small in amplitude, and the two populations of large peak errors have other
  explanations: (1) the derotation resampling of the data before fitting (the extreme, single-pixel, "model too sharp" cases: fixed by fitting
  the raw frames), (2) a core that is sharper than the model can make with the jitter the rest of the PSF demands (the +15-35 % cases, present in
  both frames). The full cubes of the listed IDs remain the final check for (2): a smooth peak / ring ratio there means the sharp pixel is a real
  (instrumental or pipeline-resampling) property of the data and the model needs a sharper core, not a mask.
- Numbers on the 17 refit samples (median over samples of the median |peak err| per sample): official derotated fits 19.0 %, raw-frame fits
  9.0 %, derotated + area 17.2 %, raw + area 16.6 %. The ring error stays at 3-5 % in all variants.
- Recommendation: switch `STD_fitter.py` to the raw-frame regime (`DEROTATE_PSF = False`, the model rotates its pupil) for the next full refit,
  then rebuild the dataset cache with `derotate_PSF=False` and retrain; the derotation can also be removed from the calibrator data path the
  same way. Keep the derotated `fitted/` until then.
