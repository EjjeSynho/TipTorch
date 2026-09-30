### TipTorch integration inside the TipTop infrastructure
## General scope
The goal of this plan is to integrate a TipTorch model inside the TIPTOP infrustructure. As you may see, TipTorch shares a lot incommon in functionality with P3, so the goal of this integration is to completely replace P3. In the end, the user must be able to choose which backend to use between P3 and TipTorch, butlet's leave it for later. Please do all the changes only inside `TIPTOP/tiptop/TipTop_integration.py` and the TipTorch repo.

## Code resources
As a reference of how the end structure should look like, please refer to `TIPTOP/tiptop/TipTop_integration.py`. The `TipTop_integration.py` already have a draft of the structure of how the integration must look like. This file is the main one in the integration. try not to touch anything else inside the TIPTOP repo. All secondary auxilliary functionality must be put in `TipTorch/src/tools/tiptop_integration.py`.

# Please note when implementing:
- When computing total tip-tilt jitter, the original, in the original `TIPTOP/tiptop/baseSimulation.py`, two consequential convolutions with the gaussian kernels were done. replace this part by computing the combined Gaussian kernel directly. The draft of this procedure is outlined in `TipTorch/development/Gauss_rotate.py`;
- When computing the LO WFS PSFs, let it use MASTSEL for now;
- Perhaps, consider implementing a dataclass for NGS parameters, because they seem to be carried aound in a very cumbersome format;
- In this execution plan report your progress. If you find a cricial part of functionality missing or some critical issue with the implementation, please report it here, too.

## Execution plan
Please follow these steps:
1. Create `tiptop_integration.py`
2. Carefully read `baseSimulation.py`, get the structure and logic;
3. Then, read `TipTop_integration.py` and find parallels with the `baseSimulation.py`;
4. Finish developing the `Gauss_rotate.py` and put the resulting code into a newly created `tiptop_integration.py` for later use;
5. Finally, implement the requested integration.

## Progress

- [x] Read `TIPTOP/tiptop/baseSimulation.py` and identified its HO, LO, focus, output, and asterism paths.
- [x] Found and reviewed the available TipTorch draft at `TIPTOP/tiptop/TipTorchSimulation.py`.
- [x] Added covariance-based Gaussian jitter helpers to `src/tiptorch/tools/tiptop_integration.py`.
- [x] Built `TIPTOP/tiptop/TipTop_integration.py` with TipTorch HO science PSFs, MASTSEL LO/focus WFS PSFs, MavisLO covariance, combined jitter, asterism caching, metrics, and FITS/JSON output.
- [x] Carried over TIPTOP's extra-error PSD, NGS field-dependent extra error, and no-LO wind-shake PSD from the P3 path.
- [x] Added TipTorch tilt rejection when LO correction is active, fixed signed Gaussian jitter rotation and requested even PSF size, and applied static WFE maps when supplied.
- [x] Validated Gaussian math and ran INI, YAML, LO, no-LO, focus, aliasing, static WFE, extra-error, wind-shake, multi-pointing, and indexed asterism smoke paths.

## Surprises & Discoveries

- The named draft `TIPTOP/tiptop/TipTop_integration.py` is missing. `TipTorchSimulation.py` appears to be the intended draft and has local, uncommitted edits. It is being left untouched.
- `Gauss_rotate.py` includes a Gaussian *product* demonstration, but the two TIPTOP kernels are *convolved*. Their covariance matrices therefore add.
- The available draft's save and legacy LO paths still refer to P3 objects, and its active NGS path uses TipTorch; those do not satisfy the requested integration.
- TipTorch returned a complex PSD from `ComputePSD()`; the bridge uses its nonnegative real part for PSD variance and MASTSEL conversion.
- TipTorch's jitter kernel previously discarded negative ellipse angles and used an incorrect Gaussian scale factor. Its small odd OTF grid also returned 255 pixels for a requested 256-pixel image.
- MASTSEL in the local environment selected CuPy even with `MASTSEL_DISABLE_GPU=TRUE`; smoke runs used a writable CuPy cache. The bridge accepts both CuPy and NumPy outputs.

## Decision Log

- Put reusable tensor and covariance conversions in the installed `tiptorch.tools` package, at `src/tiptorch/tools/tiptop_integration.py`.
- Preserve the existing modified draft while creating the requested bridge as a new TIPTOP file.
- Keep the public class name `baseSimulation` and the original constructor options in the new module so existing call patterns can use it directly. Backend selection in TIPTOP's existing entry point remains a later task, as requested in General scope.
- Use MASTSEL's `psdSetToPsfSet` with TipTorch NGS-direction PSDs, subaperture masks, and optional static WFE; use MavisLO for residual matrices.
- Apply the P3-equivalent tilt rejection to TipTorch PSDs when LO is active to avoid including the corrected tilt twice.
- Cache the HO PSD and baseline FWHM for indexed asterism metrics; add focus PSD only to full-field results.

## Validation

- `conda run -n TipTop python TipTorch/tests/test_tiptop_integration.py`: passed Gaussian covariance, signed OTF rotation, jitter units, PSF width, and tilt-filter checks.
- `TIPTOP/tiptop/perfTest/dummy.ini` CPU bridge run: finite 2×1×200×200 cube, source flux near one, finite LO residual, FITS and JSON saved.
- `TIPTOP/tiptop/perfTest/ERIS.ini` CPU bridge run: finite 1×256×256 cube with unit flux and no LO sensor.
- Temporary variants of `dummy.ini`: focus, LO aliasing, two science pointings, static WFE map, YAML input, and indexed asterism all completed. Temporary test artifacts were removed.
- A temporary extra-error variant produced a 3600 nm² science extra PSD and a 400 nm² on-axis LO extra PSD as configured. A temporary no-LO wind-shake variant using the bundled FITS spectrum produced a finite 1×256×256 PSF.

## Outcomes & Retrospective

The new module can be instantiated directly as `tiptop.TipTop_integration.baseSimulation` and run on the tested TIPTOP configurations. It does not change the existing `tiptop.py` backend selector, which the requested plan leaves for later. The existing modified `TipTorchSimulation.py` remains untouched. A larger ELT/MORFEO configuration has not yet been compared numerically with P3, so scientific agreement at that scale is still unverified.

Of course, you are free to add necessary substeps as you wish. Finally, Godspeed and good luck! What else people say? Ah, sure, make no mistakes! Let's cook.

