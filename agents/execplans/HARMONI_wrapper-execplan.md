## TipTorch wrapper for the HARMONI instrument

The goal of this plan is to create a TipTorch model wrapper for HARMONI instrument. This wrapper must be almost exactly the same as one for MUSE-NFM which lives in `src\tiptorch\PSF_models\NFM_wrapper.py`.

# Requirements
The wrapper must retain all the key features with a special emphasis on:
- Full-spectrum simulation;
- Splines used for parameters interpolation;
- Same model parameters managment scheme;
- Fitting-ready implementation;
- Zernike NCPAs;
- Ensure that user can disable Zernike aberrations;
- Support of the static WFE map loaded from a file (new compared to NFM).

What it __MUST NOT__ include is:
- Chromatic defocus;

Take the relevant inspirations from the development file `development\HARMONI_example.py`.

# Basic plan
1. Read and understand the NFM, comprehend its functionality;
2. Read carefully `development\HARMONI_example.py` and see how the draft initialization of the HARMONI simulation currently looks like.
3. Create an outline of what it does;
4. Implement the same logic for the HARMONI wrapper;
5. Precisely fullfill the requirements outlined above;
6. Preserve the same code architecure and naming conventions as much as possible;
7. Stay concise, better less comments that large code in this case
8. Good luck!

# Progress

- [x] Reviewed `NFM_wrapper.py`, `TipTorch.py`, the phase-basis helpers, config management, and `development/HARMONI_example.py`.
- [x] Added `PSFModelHARMONI` with MCAO initialization and the NFM wrapper's parameter, spline, fitting, spectral-range, save/load, and copy interfaces.
- [x] Added a file-backed static WFE mode and optional tip/tilt-free Zernike modes.
- [x] Ensured no chromatic-defocus input is exposed.
- [x] Retained `telescope.PathStaticOn` when loading instrument configs.
- [x] Added and ran focused wrapper tests.
- [x] Ran a CPU smoke test with the local HARMONI config, ELT pupil, and static WFE FITS map.
- [x] Updated `development/HARMONI_example.py` to use the wrapper and simulate the reference cube's full H-band spectrum.
- [x] Added configurable, flux-preserving spectral binning to the HARMONI example using either `N_bins` or `Δλ_bin`.

# Surprises & Discoveries

- `ConfigManager.Load()` previously removed `PathStaticOn` because it was absent from the bundled required-fields schema.
- HARMONI's draft config contains a single sparse science wavelength, so the wrapper normalizes that scalar to a one-element wavelength vector before creating managed inputs.
- The supplied static WFE FITS file contains nanometre-scale values and has the same 480×480 sampling as the ELT pupil.

# Decision Log

- Reused `PSFModelNFM` as the behavioral base to keep its fitting and full-spectrum interfaces aligned.
- Replaced the NFM-specific VLT pupil and LTAO setup with config-loaded ELT pupil data and MCAO.
- Represented static WFE and Zernikes as one managed arbitrary basis. The static mode starts at coefficient 1, while Zernike coefficients start at zero.
- Kept `LO_NCPAs` as the overall aberration switch and added `use_Zernike` and `use_static_WFE` as independent component switches.
- Used a configurable HARMONI-wide default spectral grid from 470 nm to 2450 nm; callers can provide band-specific bounds and slice counts.

# Outcomes & Retrospective

The HARMONI wrapper now supports sparse and full-spectrum simulation, natural-cubic-spline chromatic parameters, managed fitting inputs, optional Zernike NCPAs, static WFE loaded from FITS, and serialization. A real MCAO forward pass returned a finite 15×15 PSF normalized to unit flux. Focused tests cover phase-mode composition and toggles, absence of chromatic defocus, batched full-spectrum simulation, and save/load restoration. The development example exercises the wrapper on the reference H-band cube, bins the data and model wavelength grid consistently, and compares data and model slices across the resulting spectrum.
