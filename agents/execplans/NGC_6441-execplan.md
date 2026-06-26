The goal of this plan is to create a data processing routine, which should be almost the same as one in `datasets/HST_NFM_omega_cen.py` as folows:

1. Read and understand the Omega Cen routine, extract the basic steps.
2. Create an outline of what it does.
3. Implement the same logic for the NGC 6441 data, taken from the ([HUGS dataset](https://archive.stsci.edu/prepds/hugs/)), retaining the same logic.
4. The downloaded dataset lives under `F:\ESO\Data\MUSE\omega_cluster\NGC6441_data\`, look through this folder. NOTE: the astrometry and photometry are provided in large text files. Don't read them fully, just scan the first dozen lines.
5. Despite my request of following strictly the logic of the omega Cen routine, depart from this instruction if the structure of data requires it
6. Neverthess, enforce the same structure of output as in the omega cen routines.
7. Stay concise, better less comments that large code in this case
8. Good luck!


## Execution Notes

### Omega Cen Routine Outline

The Omega Cen routine matches raw and reduced MUSE cubes, loads HST astrometry and per-filter photometry, selects HST sources around the MUSE pointing center, converts AB magnitudes to MUSE flux units with `synphot`/`stsynphot`, caches the selected HST data, loads or generates a MUSE cube cache, detects sources in the binned MUSE cube, fits an affine HST-to-MUSE alignment, filters sources to the MUSE NFM field of view, and exports `metadata/HST_srcs_<cube_stem>.csv`.

### Progress

- [x] (2026-06-26 12:05 Europe/Berlin) Read `datasets/HST_NFM_omega_cen.py` and extracted the processing flow.
- [x] (2026-06-26 12:10 Europe/Berlin) Scanned `F:\ESO\Data\MUSE\omega_cluster\NGC6441_data\` and the first catalog header/data lines without loading full text files manually.
- [x] (2026-06-26 12:25 Europe/Berlin) Implemented `datasets/HST_NFM_NGC_6441.py` with HUGS text-catalog loading and Omega-style output columns.
- [x] (2026-06-26 12:45 Europe/Berlin) Validated syntax with `conda run -n AO-torch python -m py_compile datasets/HST_NFM_NGC_6441.py`.
- [x] (2026-06-26 12:58 Europe/Berlin) Ran the full routine and generated the final CSV.

### Surprises & Discoveries

- Observation: The reduced cube `DATACUBEFINALscipost_20220531T034028_d33feab3.fits` is tagged `OBJECT = NGC 6441` and lies near the HUGS catalog center.
  Evidence: FITS header scan found it within about 0.001 deg of the NGC 6441 catalog center.
  Consequence: The script auto-selects this cube instead of requiring a hardcoded filename.

- Observation: HUGS catalog IDs are not safe for pandas label-based subsetting in the selected region.
  Evidence: `.loc[df_sel.index]` expanded photometry from 4697 astrometric rows to 4701 rows.
  Consequence: The routine uses positional `iloc` selection and row pointers for final per-filter flux assignment.

- Observation: GPU/CuPy cache generation failed on the full NGC 6441 MUSE cube.
  Evidence: `cudaErrorMemoryAllocation` / `cudaErrorAlreadyMapped` occurred during `LoadCachedDataMUSE`.
  Consequence: The script sets `CUDA_VISIBLE_DEVICES=""` before importing TipTorch so cube cache generation runs through the CPU path.

### Decision Log

- Decision: Vectorize the AB-magnitude to MUSE-flux conversion per filter instead of looping over stars with `synphot`.
  Rationale: The HUGS selected field is dense; the Omega-style per-star loop exceeded a ten-minute run. A 0 ABmag flux conversion per filter with `10**(-0.4 * mag)` preserves the same constant-flux AB definition.
  Date/Author: 2026-06-26, Codex.

- Decision: Repair stale photometry caches when astrometry, magnitude, and flux row counts differ.
  Rationale: A failed intermediate run created a malformed cache; reruns should be idempotent.
  Date/Author: 2026-06-26, Codex.

### Artifacts and Notes

Validation commands:

    conda run -n AO-torch python -m py_compile datasets/HST_NFM_NGC_6441.py
    conda run -n AO-torch python -u datasets/HST_NFM_NGC_6441.py

Generated files:

    F:\ESO\Data\MUSE\omega_cluster\metadata\selected_srcs_DATACUBEFINALscipost_20220531T034028_d33feab3.pkl
    F:\ESO\Data\MUSE\omega_cluster\cached_cubes\DATACUBEFINALscipost_20220531T034028_d33feab3.pickle
    F:\ESO\Data\MUSE\omega_cluster\metadata\HST_srcs_DATACUBEFINALscipost_20220531T034028_d33feab3.csv

Output validation:

    CSV shape: (600, 9)
    CSV columns: ID, x, [asec], y, [asec], flux (total, normalized), F275W, F336W, F438W, F606W, F814W
    Cache lengths: Astrometry=4697, AB magnitudes=4697, Fluxes=4697

### Outcomes & Retrospective

Implemented `datasets/HST_NFM_NGC_6441.py`, adapting the Omega Cen workflow to HUGS NGC 6441 text catalogs while preserving the same final CSV structure. The routine auto-selects the NGC 6441 MUSE cube, repairs mismatched caches, uses CPU cube loading to avoid GPU memory errors, and successfully generated the HST source CSV for the NGC 6441 cube. Remaining warnings are non-fatal: missing Vega reference data for `stsynphot`, an Astropy WCS date fix, and a Windows/MKL KMeans warning.
