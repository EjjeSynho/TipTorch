# TipTop and TipTorch: performance comparison

The goal of this execution plan is to implement additional performance tests to compare the runtime of "vanilla" TipTop and TipTorch.

## Instructions
1. Write a test similar to `tests/test_tiptop_visual_comparison.py`, but without any visuals. Run each of these test cases several times with vanilla TipTop and with TipTop + integrated TipTorch, and compute the median performance gain (or loss) from using TipTorch for every case (e.g., MAVIS, ERIS, ERIS LGS).

2. After this, write two tests that compare TipTop against *pure* TipTorch. Matching the PSFs exactly is not the point of these tests; the goal is to match the number of simulated PSFs or spectral slices.

    - First, write a polychromatic test for MUSE NFM PSFs. Simulate one synthetic PSF for all 3681 spectral slices of MUSE NFM, spanning the entire spectrum. Do it with both TipTop and TipTorch. In the latter case, use the `PSFModelNFM` wrapper, which already includes the necessary optimizations for full-spectrum rendering. Compare the performance of both.

    - Second, write a batched performance test. Generate MUSE NFM configs for 16 objects at 31 wavelengths. Render them using TipTop and then using TipTorch. In the latter case, use the wrapper's and TipTorch's intrinsic capabilities to parallelize the rendering.

## Final goal
The final goal is to make a fair performance comparison of both codes.

## Implementation (2026-10-22)

Scripts (plain scripts printing `[PASS]` / `[FAIL]`, no pytest in the env; run from the repository root in the `TipTorch` conda env):

| Script | What it times | Results |
|---|---|---|
| `tests/test_tiptop_backend_performance.py` | TIPTOP end to end (constructor + `doOverallSimulation`) with the P3 backend vs the TipTorch backend on perfTest ERIS, MAVIStest, METIS, ERIS_LGS; one warm-up + `--repeats` (5) timed runs, median gain per case | `tests/runs/performance/backend_performance.json` |
| `tests/test_tiptop_polychromatic_performance.py` | one MUSE NFM PSF at the 3681 slices (475-935 nm): TIPTOP with one 3681-wavelength `.ini` vs `PSFModelNFM.SimulateSpectralRange` (batched slices) | `tests/runs/performance/polychromatic_performance.json` |
| `tests/test_tiptop_batched_performance.py` | 16 objects x 31 wavelengths: TIPTOP one `.ini` per object (sequential and 4-thread pool) vs one `PSFModelNFM(multiple_obs=True)` forward pass on the merged configs (+ a per-object TipTorch loop for context) | `tests/runs/performance/batched_performance.json` |

Shared helpers in `tests/tiptop_performance_utils.py`: astro-tiptop checkouts on `sys.path` (from `TipTop_path`), cupy on the TipTorch GPU,
`timed_runs` (warm-up + median), the MUSE NFM configurations of both codes from the same system description (`perfTest/MUSE_LTAO.ini`
for TIPTOP, `parameter_files/muse_ltao.ini` for TipTorch: same telescope, atmosphere, WFSs, DM and RTC), `random_observations`.

### Progress
- [x] Backend comparison script (no figures) with repeats and median gain per perfTest case.
- [x] Polychromatic MUSE NFM script (3681 slices) for TIPTOP and the NFM wrapper.
- [x] Batched MUSE NFM script (16 objects x 31 wavelengths) for TIPTOP (sequential and thread pool) and the batched wrapper.
- [x] Runs on GPU 1 (`cuda:1` of project_config.json), results recorded below.

### Surprises & Discoveries
- `perfTest/MUSE_LTAO.ini` cannot run through TIPTOP as is: MASTSEL's `MavisLO` needs `[telescope] TechnicalFoV`, which the file lacks
  (the ETC configuration `muse_ltao_Guido.ini` has it). The helper adds `TechnicalFoV = 120` and the local VLT pupil FITS path.
- A multi-wavelength TIPTOP configuration does not render on the requested grid: P3's `psdExpansion` multiplies `FieldOfView` by
  `ceil(lambda_max / lambda_min)` (2 for MUSE NFM) and the LO-loop path (jitter convolution) doubles it again (200 px without the LO sections, measured), so a 100 px request comes out as a
  [nWvl, 400, 400] cube (TipTorch renders the requested 100 px). The TIPTOP cost per slice (~0.15 s) is therefore partly spent on pixels
  that were not asked for; reported as is, since it is how TIPTOP handles polychromatic requests.
- cupy's current device is thread-local: in the TIPTOP thread-pool mode every worker has to select the GPU again, otherwise P3/MASTSEL
  silently run on GPU 0 while TipTorch runs on GPU 1.
- `torch.cuda.max_memory_allocated()` without a device argument reports GPU 0; the project device is `cuda:1`.

### Decision Log
- Timed scope = configuration parsing + model setup + PSF rendering (`Simulation(...)` + `doOverallSimulation()`), as the visual test;
  the metrics (`computeMetrics`: SR, FWHM contour, EE) are excluded because P3 computes them one PSF at a time in numpy and would
  inflate the polychromatic gain for reasons unrelated to PSF rendering.
- One untimed warm-up run before the timed ones for both codes (CUDA context, cupy / torch kernel compilation, file caches); TIPTOP's
  polychromatic run (~10 min) is run once by default (`--tiptop-repeats 1`, no warm-up), TipTorch 5 times.
- Both codes on the same GPU (`cuda:1`): P3 and MASTSEL use cupy, so the comparison is GPU vs GPU, not CPU vs GPU.
- MUSE NFM system description taken from TIPTOP's own `perfTest/MUSE_LTAO.ini` (identical to TipTorch's `muse_ltao.ini`) rather than
  the tuned ETC file: no `'optimize'` LO settings, no extra static terms, so both codes model the same system. TipTorch uses the
  analytic `PupilVLT`, TIPTOP the pupil FITS; the wrapper keeps its defaults (LO NCPAs on, 5 spline nodes, physics-based regime).
- The polychromatic TipTorch run uses `sequential=False` and 256 slices per forward pass (one source, so `sequential` is immaterial);
  the batched run is a single `model()` call on the merged 16-observation config (the wrapper's `multiple_obs=True` regime).

### Outcomes & Retrospective (runs of 2026-10-22, RTX PRO 6000 Blackwell, `cuda:1`, P3 / MASTSEL on cupy on the same GPU)

**1. TIPTOP with the P3 backend vs TIPTOP with the TipTorch backend** (end to end, median of 5 runs after a warm-up; all cases pass):

| case | P3 [s] | TipTorch [s] | gain | PSF cube |
|---|---|---|---|---|
| ERIS (SCAO) | 0.026 | 0.014 | 1.9x | [1, 256, 256] |
| MAVIStest (MCAO, 9 pointings) | 0.366 | 0.135 | 2.7x | [9, 512, 512] |
| METIS (ELT SCAO, 2048 px) | 0.496 | 0.026 | 19x | [1, 2048, 2048] |
| ERIS_LGS (SLAO, LO loop) | 0.109 | 0.089 | 1.2x | [1, 256, 256] |

The warm (cupy / CUDA context already up) P3 runs are much faster than the ~10-15 s per case quoted in the docstring of
`tests/test_tiptop_visual_comparison.py` (cold first calls in a fresh process: ERIS 0.06 s, MAVIStest 0.77 s, METIS 0.57 s for P3;
0.23 s, 0.75 s, 0.24 s for TipTorch). The gain of the TipTorch backend grows with the grid size (METIS) and shrinks when MASTSEL's LO
loop (`MavisLO`, the same code in both backends) dominates (ERIS_LGS).

**2. One MUSE NFM PSF at the 3681 slices** (100 px requested; TIPTOP once, TipTorch median of 5):

| code | total [s] | per slice [ms] | cube |
|---|---|---|---|
| TIPTOP (P3 + MASTSEL, one 3681-wavelength `.ini`) | 727.7 | 197.7 | [3681, 400, 400] |
| TipTorch (`PSFModelNFM.SimulateSpectralRange`, 256 slices per pass) | 0.72 (0.03 setup + 0.51 render) | 0.20 | [3681, 100, 100] |

TipTorch is ~1000x faster; peak VRAM 4.8 GB. TIPTOP renders 16x more pixels per slice than requested (see Surprises), which accounts
for part of its cost per slice, but not for three orders of magnitude.

**3. 16 objects x 31 wavelengths** (100 px requested; TIPTOP median of 2, TipTorch median of 5):

| code | total [s] | per PSF [ms] | cube |
|---|---|---|---|
| TIPTOP, one `.ini` per object, sequential | 26.8 | 54 | [16, 31, 400, 400] |
| TIPTOP, 4-thread pool (as the ETC scripts) | 26.0 | 52 | [16, 31, 400, 400] |
| TipTorch, one merged `PSFModelNFM(multiple_obs=True)`, one forward pass | 0.36 (0.09 setup + 0.08 forward) | 0.7 | [16, 31, 100, 100] |
| TipTorch, one single-source model per object (context) | 3.45 | 7.0 | [16, 31, 100, 100] |

The batched TipTorch model is ~75x faster than TIPTOP (72x against the thread pool, which brings TIPTOP nothing: the GIL and a single
cupy stream serialize the workers) and ~10x faster than its own per-object loop; peak VRAM 7.9 GB.

Retrospective: the comparison is GPU vs GPU and like for like in system description and in the number of rendered PSFs; the remaining
asymmetries (TIPTOP's enlarged grid in multi-wavelength mode, TIPTOP's LO loop vs the wrapper's fitted jitter, analytic vs FITS pupil) are
documented in the scripts. Re-run with `--fov`, `--repeats`, `--tiptop-repeats`, `--batch`, `--workers` to probe other regimes.

## Follow-up (2026-10-22): accelerating TIPTOP, presentation outputs

Request: try to speed TIPTOP up (more threads, TIPTOP's internal batching) and make the tests emit copy-paste-ready results (tables, bar plots).

### What TIPTOP offers
- No batching across configurations: `overallSimulation` / `baseSimulation` build one `fourierModel` per `.ini`; the only internal batching
  is over the science directions and the wavelengths of one configuration (both already used: the MUSE `.ini` files carry all wavelengths).
- `gpuIndex` selects the GPU; P3, MASTSEL and SEEING each have a `<PKG>_DISABLE_GPU` flag for a numpy (CPU) run.
- The ETC scripts parallelize with a `ThreadPoolExecutor` (4 workers).

### Modes added to the tests (`tiptop_worker_init`, `simulate_tiptop_ini` in `tests/tiptop_performance_utils.py`)
- Batched test (`--tiptop-modes`): `sequential`, `threads` (--threads 4), `processes` (--processes 4, a spawn process pool on the same GPU,
  one cupy context per worker), `cpu` (--cpu-workers 32, a process pool with the three DISABLE_GPU flags set before the imports). Pools are
  started once, warmed up and timed like the single-process modes; the cupy pool of the parent is released between modes (`free_cupy_pool`).
- Polychromatic test: `single` (one 3681-wavelength `.ini`), `chunks` (VRAM-budgeted contiguous spectral chunks, --chunk-vram 20 GB each, in a
  pool of --processes 4 GPU workers), `cpu-chunks` (opt-in, --cpu-chunks 120 equal chunks in a pool of --cpu-workers 32 CPU processes); the
  pooled modes are timed cold (pool start-up included).

### Surprises & Discoveries (follow-up)
- A TIPTOP multi-wavelength MUSE simulation holds from ~3.5 GB (475 nm) to ~0.3 GB (930 nm) of VRAM per wavelength (the per-wavelength
  oversampled MASTSEL grids): ~12 GB for 31 wavelengths spread over the spectrum, ~70 GB for 31 blue slices. 8 threads or 6 GPU processes
  run out of the 97 GB with the 31-wavelength jobs; the 3681-slice job is cut into VRAM-budgeted chunks (<= 20 GB each) for a 4-worker pool.
- cupy's memory pool keeps the cached VRAM of the parent process (99 GB after a 6-thread run) and starves any process pool started
  afterwards: `free_all_blocks()` between modes is mandatory.
- Threads: 6 threads = 4 threads = sequential (the GIL and one cupy stream serialize the work).
- CPU mode needs `SEEING_DISABLE_GPU` as well as the P3 and MASTSEL flags: P3's telescope pupil comes from SEEING, and with it on cupy
  MASTSEL's `congrid` fails on the implicit cupy -> numpy conversion. One CPU core renders a MUSE slice in ~0.6 s (vs ~0.2 s per slice on the
  GPU stream), so a 32-process CPU pool can beat the single GPU stream on the polychromatic job.
- `node` is not installed: the chart colors are the documented, validated reference palette of the dataviz skill (slots 1-3), not re-validated here.

### Presentation outputs
Every test writes `tests/runs/performance/<test>.{json,md,png,svg}`: the markdown table (median / min / max / runs / cube, speed-ups) and a
horizontal bar chart of the median run times (log scale, values at the bar tips, the speed-up of TipTorch per group, TIPTOP blue, TipTorch
orange). `tests/plot_tiptop_performance.py` assembles the three records into `summary.{png,svg,md}` (one panel per test).

### Outcomes of the follow-up (runs of 2026-10-22, same GPU; `tests/runs/performance/summary.{png,svg,md}`)

**Batched, 16 objects x 31 wavelengths** (TIPTOP median of 2, TipTorch of 5):

| mode | median | note |
|---|---|---|
| TIPTOP sequential | 27.4 s | |
| TIPTOP 4 threads (GPU) | 24.7 s | 6 threads: 25.1 s; 8 threads: out of VRAM |
| TIPTOP 4 processes (GPU) | 15.7 s | the best TIPTOP mode (1.7x); 6 processes: out of VRAM |
| TIPTOP 32 processes (CPU) | 43.1 s | 16 processes: slower |
| TipTorch, one batched forward pass | 0.36 s | 44x faster than the best TIPTOP mode (75x than sequential) |
| TipTorch, one model per object | 3.3 s | context |

**Polychromatic, 3681 slices, one PSF** (TIPTOP 1 run per mode, TipTorch median of 5; GPU-only comparison requested afterwards, the CPU row
is kept here for the record and dropped from the stored results / charts):

| mode | time | note |
|---|---|---|
| TIPTOP, one 3681-wavelength configuration | 11.8 min | |
| TIPTOP, 236 VRAM-budgeted chunks in a pool of 4 GPU processes | 4.1 min | the best TIPTOP mode (2.9x); chunks of <= 20 GB: 5 blue to 40 red slices |
| TIPTOP, 120 chunks in a pool of 32 CPU processes | 7.5 min | 1.6x; 32 chunks: 10.6 min (load imbalance) |
| TipTorch, `SimulateSpectralRange` | 0.73 s | 340x faster than the best TIPTOP mode (970x than the single configuration) |

The GPU pool only works with wavelength-dependent chunk sizes: a TIPTOP multi-wavelength MUSE run holds ~3.5 GB of VRAM per slice at 475 nm,
~1 GB at 600 nm and ~0.3-0.5 GB beyond 700 nm (cupy pool high-water marks of chunk runs; MASTSEL oversamples the bluest slices the most),
so equal chunks of 31 slices (~70 GB at the blue end) ran out of memory with 4 and 3 workers. `tiptop_VRAM_per_slice_GB` /
`VRAM_budgeted_chunks` in the utils encode the measured steps; `--chunk-vram` sets the budget. Every chunk recomputes the LO loop (~1 s).

Retrospective: TIPTOP's practical ceiling on this hardware is a 4-process GPU pool: 1.7x for many small jobs and 2.9x for one long
polychromatic job once the chunks respect the wavelength-dependent VRAM footprint; threads bring nothing, VRAM (not compute) limits the GPU
pools, and TIPTOP's own batching (multi-wavelength configurations) is already used. The speed-ups of TipTorch quoted above are against these accelerated TIPTOP modes. The GPU process pool
needs cupy's memory pool and cuFFT plan cache released per job (`free_vram`) when the jobs have different grids, otherwise the workers grow
by several GB per job. `--merge` re-runs one expensive mode and keeps the stored rows of the others.
