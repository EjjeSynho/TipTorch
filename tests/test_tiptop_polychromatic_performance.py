"""
Polychromatic MUSE NFM performance: one synthetic PSF at all 3681 spectral slices of the MUSE NFM cube (475-935 nm, 0.125 nm step),
rendered by vanilla TIPTOP (P3 + MASTSEL) and by pure TipTorch through the `PSFModelNFM` wrapper (`SimulateSpectralRange`, all slices
in wavelength batches on the GPU). The PSFs are not meant to match; the point is the number of rendered slices.

TIPTOP modes: `single` (one 3681-wavelength `sources_science.Wavelength` configuration, TIPTOP's own batching over wavelengths,
~0.2 s per slice), `chunks` (the spectrum split into contiguous chunks sized to a VRAM budget of --chunk-vram GB each, one
multi-wavelength `.ini` per chunk, run by a pool of --processes workers on the same GPU, every worker with its own cupy context; a
TIPTOP multi-wavelength MUSE run holds ~3.5 GB of VRAM per slice at 475 nm and ~0.3 GB at 930 nm, so the blue chunks are 5 slices and the
red ones 40) and, off by default, `cpu-chunks` (--cpu-chunks equal chunks in a pool of --cpu-workers processes with P3 / MASTSEL on
numpy, one core each). Every chunk recomputes TIPTOP's LO loop (~1 s). TipTorch: one `PSFModelNFM` on a 5-wavelength sparse grid, the full spectrum rendered by `SimulateSpectralRange`
in batches of --batch slices.

Both codes describe the same AO system (TIPTOP's perfTest/MUSE_LTAO.ini and TipTorch's parameter_files/muse_ltao.ini, 25 mas pixels,
--fov pixels requested, 100 by default). Timed per code: the complete job (configuration + model setup + the rendering of the 3681
slices); TipTorch also reports the rendering alone. TIPTOP's psdExpansion enlarges the field of view of a multi-wavelength
configuration by ceil(lambda_max / lambda_min) and the LO-loop path doubles it again (200 px without the LO sections), so its cubes are [3681, 4 * fov, 4 * fov];
the TipTorch cube is [3681, fov, fov]. TIPTOP's single mode takes ~12 min, hence --tiptop-repeats 1 by default (no warm-up); the pooled
modes include the pool start-up (cold workers), as a user would run them; TipTorch is run --repeats times (default 5).

Requirements: the astro-tiptop checkouts at `TipTop_path` of project_config.json, the local VLT pupil FITS, a CUDA device, ~15 GB of
host memory for TIPTOP's cube. Runtime ~20 min. Results: tests/runs/performance/polychromatic_performance.{json,md,png,svg}.
Run: python tests/test_tiptop_polychromatic_performance.py [--repeats 5] [--tiptop-repeats 1] [--tiptop-modes chunks single] [--merge]
                                                            [--chunk-vram 20] [--processes 4] [--cpu-chunks 120] [--cpu-workers 32] [--batch 256] [--fov 100]
"""

import argparse
import functools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiptop_performance_utils import (DEVICE, MUSE_NFM_N_SLICES, OUTPUT, MUSE_NFM_wavelengths, MUSE_tiptorch_config, VRAM_budgeted_chunks,
                                      free_cupy_pool, report, select_GPU_for_cupy, simulate_tiptop_ini, synchronize, timed_runs,
                                      tiptop_worker_init, write_MUSE_tiptop_ini)

STEM = 'polychromatic_performance' # tests/runs/performance/<STEM>.{json,md,png,svg}
N_SPARSE = 5 # control wavelengths of the wrapper's spline representation (the sparse grid the model is built on)
TIPTOP_MODES = ('single', 'chunks', 'cpu-chunks')


def tiptop_single(fov):
    ''' TIPTOP: one configuration with the 3681 science wavelengths; returns the cube [3681, N, N] '''
    config_dir = OUTPUT / 'tiptop_outputs'
    name = f'MUSE_NFM_{MUSE_NFM_N_SLICES}_slices'
    write_MUSE_tiptop_ini(config_dir / f'{name}.ini', MUSE_NFM_wavelengths(MUSE_NFM_N_SLICES), fov)

    def run():
        cube = simulate_tiptop_ini(config_dir, name)
        synchronize()
        return dict(cube=cube)
    return run


def tiptop_chunks(fov, chunks, workers, use_gpu):
    '''
    TIPTOP: the spectrum in contiguous chunks, one multi-wavelength `.ini` each, run by a (cold) process pool of `workers`. `chunks` is a
    list of wavelength arrays: VRAM_budgeted_chunks for the GPU pool (a TIPTOP multi-wavelength MUSE run holds ~3.5 GB of VRAM per slice
    at 475 nm down to ~0.3 GB at 930 nm, see tiptop_VRAM_per_slice_GB) or equal splits for the CPU pool.
    '''
    config_dir = OUTPUT / 'tiptop_outputs'
    names = []
    for k, chunk in enumerate(chunks):
        names.append(f'MUSE_NFM_chunk{len(chunks)}{"" if use_gpu else "cpu"}_{k:03d}')
        write_MUSE_tiptop_ini(config_dir / f'{names[-1]}.ini', chunk, fov)

    def run():
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=tiptop_worker_init,
                                 initargs=(DEVICE.index if use_gpu else None, use_gpu)) as pool:
            simulate = functools.partial(simulate_tiptop_ini, free_vram=use_gpu) # one cuFFT plan cache per chunk grid otherwise
            paths = list(pool.map(simulate, [config_dir] * len(chunks), names, [config_dir / f'{n}.npy' for n in names]))
        return dict(cube=np.concatenate([np.load(p) for p in paths]))
    return run


def tiptorch_full_spectrum(fov, batch):
    ''' TipTorch: PSFModelNFM on a 5-wavelength sparse grid, the full spectrum rendered by SimulateSpectralRange in batches of `batch` slices '''
    from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

    def run():
        t0 = time.perf_counter()
        config = MUSE_tiptorch_config(MUSE_NFM_wavelengths(N_SPARSE), fov)
        model = PSFModelNFM([config], multiple_obs=False, LO_NCPAs=True, N_spline_nodes=N_SPARSE, device=DEVICE)
        synchronize()
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        PSFs, wavelengths = model.SimulateSpectralRange(λ_batch_size=batch, sequential=False, force_cpu=True)
        synchronize()
        t_render = time.perf_counter() - t0
        assert PSFs.shape[1] == wavelengths.numel() == MUSE_NFM_N_SLICES, PSFs.shape
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9 if DEVICE.type == 'cuda' else 0.0
        model.cleanup()
        return dict(cube=PSFs[0].numpy(), setup_s=t_setup, render_s=t_render, peak_GB=peak)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--repeats', type=int, default=5, help='timed TipTorch runs (after one warm-up run)')
    parser.add_argument('--tiptop-repeats', type=int, default=1, help='timed TIPTOP runs per mode (no warm-up when 1: ~12 min each in single mode)')
    parser.add_argument('--tiptop-modes', nargs='+', default=['chunks', 'single'], choices=TIPTOP_MODES,
                        help='TIPTOP modes to time (GPU modes by default, the pooled one first: the single mode leaves VRAM cached in this process)')
    parser.add_argument('--chunk-vram', type=float, default=20.0, help='VRAM budget [GB] per chunk of the GPU pool (4 workers x 20 GB on a 97 GB card)')
    parser.add_argument('--processes', type=int, default=4, help='GPU process pool of the chunks mode')
    parser.add_argument('--cpu-chunks', type=int, default=120, help='equal spectral chunks of the cpu-chunks mode')
    parser.add_argument('--cpu-workers', type=int, default=32, help='CPU process pool of the cpu-chunks mode (one core per worker)')
    parser.add_argument('--batch', type=int, default=256, help='wavelength slices per TipTorch forward pass')
    parser.add_argument('--fov', type=int, default=100, help='requested science field of view [px] (25 mas pixels)')
    parser.add_argument('--merge', action='store_true', help='keep the stored rows of the TIPTOP modes not run this time (re-run one expensive mode)')
    args = parser.parse_args()

    select_GPU_for_cupy()
    group = f'{MUSE_NFM_N_SLICES} slices, one PSF'
    print(f'device: {DEVICE}; {group}, {args.fov} px requested')

    print('\nTipTorch (PSFModelNFM.SimulateSpectralRange)')
    tt_median, tt_times, tt = timed_runs(tiptorch_full_spectrum(args.fov, args.batch), args.repeats, label='TipTorch')
    assert tt['cube'].shape[0] == MUSE_NFM_N_SLICES and np.isfinite(tt['cube']).all(), tt['cube'].shape
    print(f"    setup {tt['setup_s']:.2f} s + render {tt['render_s']:.2f} s (last run), peak VRAM {tt['peak_GB']:.1f} GB, cube {tt['cube'].shape}")

    rows, gains = [], {}
    gpu_chunks = VRAM_budgeted_chunks(MUSE_NFM_wavelengths(MUSE_NFM_N_SLICES), args.chunk_vram)
    cpu_chunks = np.array_split(MUSE_NFM_wavelengths(MUSE_NFM_N_SLICES), args.cpu_chunks)
    runners = {'single': tiptop_single(args.fov), 'chunks': tiptop_chunks(args.fov, gpu_chunks, args.processes, True),
               'cpu-chunks': tiptop_chunks(args.fov, cpu_chunks, args.cpu_workers, False)}
    labels = {'single': 'TIPTOP, one 3681-wavelength configuration',
              'chunks': f'TIPTOP, {len(gpu_chunks)} chunks (<= {args.chunk_vram:g} GB each) in a pool of {args.processes} GPU processes',
              'cpu-chunks': f'TIPTOP, {len(cpu_chunks)} chunks in a pool of {args.cpu_workers} CPU processes'}
    for mode in args.tiptop_modes:
        print(f'\n{labels[mode]}')
        free_cupy_pool() # the VRAM cached by this process must not starve the workers of the pooled modes
        try:
            median, times, p3 = timed_runs(runners[mode], args.tiptop_repeats, label=labels[mode], warmup=args.tiptop_repeats > 1)
        except Exception as e: # e.g. cupy out of memory with too many GPU workers: report and go on with the other modes
            print(f'    FAILED ({type(e).__name__}: {str(e)[:100]}), mode skipped')
            continue
        assert p3['cube'].shape[0] == MUSE_NFM_N_SLICES and np.isfinite(p3['cube']).all(), p3['cube'].shape
        print(f"    cube {p3['cube'].shape}")
        rows.append(dict(name=labels[mode], group=group, code='TIPTOP', mode=mode, median_s=median, times_s=times, cube_shape=list(p3['cube'].shape)))
        gains[labels[mode]] = median / tt_median
    if args.merge and (OUTPUT / (STEM + '.json')).exists():
        stored = json.load(open(OUTPUT / (STEM + '.json')))['rows']
        rows += [r for r in stored if r['code'] == 'TIPTOP' and r.get('mode') not in args.tiptop_modes]
        gains.update({r['name']: r['median_s'] / tt_median for r in rows if r['name'] not in gains})
    assert rows, 'no TIPTOP mode completed'

    rows.append(dict(name=f'TipTorch, SimulateSpectralRange ({args.batch} slices per pass)', group=group, code='TipTorch', median_s=tt_median, times_s=tt_times,
                     setup_s=tt['setup_s'], render_s=tt['render_s'], cube_shape=list(tt['cube'].shape), batch=args.batch, peak_VRAM_GB=tt['peak_GB']))
    fastest = min((r for r in rows if r['code'] == 'TIPTOP'), key=lambda r: r['median_s'])
    report(f'MUSE NFM full spectrum, {group}: TIPTOP vs pure TipTorch', rows, STEM, gains={group: fastest['median_s'] / tt_median},
           subtitle=f'{args.fov} px requested, TIPTOP {args.tiptop_repeats} run(s) per mode, TipTorch median of {args.repeats}; speed-up against the fastest TIPTOP mode',
           extra=dict(gain_per_mode=gains, fov_requested=args.fov, per_slice_ms={r['name']: 1e3 * r['median_s'] / MUSE_NFM_N_SLICES for r in rows}))
    ok = all(g > 1 for g in gains.values())
    print(f"[{'PASS' if ok else 'FAIL'}] full MUSE NFM spectrum: " + ', '.join(f"{r['name']} {r['median_s']:.1f} s" for r in rows if r['code'] == 'TIPTOP')
          + f"; TipTorch {tt_median:.2f} s ({1e3 * tt_median / MUSE_NFM_N_SLICES:.2f} ms/slice) -> TipTorch {fastest['median_s'] / tt_median:.0f}x faster than the fastest TIPTOP mode")
    return gains


if __name__ == '__main__':
    main()
