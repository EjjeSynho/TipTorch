"""
Batched MUSE NFM performance: 16 objects observed in different conditions (seeing, L0, zenith angle, wind speed drawn by
`random_observations`), each at 31 wavelengths spanning the MUSE NFM spectrum (475-935 nm), rendered by vanilla TIPTOP and by pure
TipTorch. The PSFs are not meant to match; the point is the number of simulated objects x wavelengths.

TIPTOP has no batching across observations (one `fourierModel` per configuration; its only internal batching is over the science
directions and wavelengths of one configuration, which the 31-wavelength `.ini` files use), so TIPTOP is accelerated with pools over the
16 configurations: `sequential` (one after the other), `threads` (a thread pool as the ETC scripts, GIL-bound), `processes` (a process
pool on the same GPU: every worker holds its own cupy context, ~12 GB of VRAM per multi-wavelength MUSE simulation, so at most ~6 on a
97 GB card) and `cpu` (a process pool with P3 / MASTSEL on numpy, one core per worker). The pools are started once (outside the timing)
and warmed up like the single-process modes. TipTorch: the 16 configurations are merged by `MultipleTargetsInDifferentObservations`
into one `PSFModelNFM(multiple_obs=True)` and a single forward pass renders the [16, 31, N, N] cube; a per-object TipTorch loop (16
single-source models) is timed for context.

Both codes describe the same AO system (TIPTOP's perfTest/MUSE_LTAO.ini and TipTorch's parameter_files/muse_ltao.ini, 25 mas pixels,
--fov pixels requested, 100 by default). Timed per code: the complete job (configuration + model setup + rendering of the 16 x 31 PSFs);
TipTorch also reports the forward pass alone. TIPTOP's multi-wavelength cubes come out on a 4 * fov grid (psdExpansion + MASTSEL
oversampling), TipTorch's on fov pixels.

Requirements: the astro-tiptop checkouts at `TipTop_path` of project_config.json, the local VLT pupil FITS, a CUDA device, many cores
for the cpu mode. Runtime ~10 min. Results: tests/runs/performance/batched_performance.{json,md,png,svg}.
Run: python tests/test_tiptop_batched_performance.py [--repeats 5] [--tiptop-repeats 2] [--tiptop-modes sequential threads processes cpu] [--merge]
                                                      [--threads 4] [--processes 4] [--cpu-workers 32] [--fov 100]
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiptop_performance_utils import (DEVICE, OUTPUT, MUSE_NFM_wavelengths, MUSE_tiptorch_config, random_observations, report,
                                      free_cupy_pool, select_GPU_for_cupy, simulate_tiptop_ini, synchronize, timed_runs, tiptop_worker_init,
                                      write_MUSE_tiptop_ini)

STEM = 'batched_performance' # tests/runs/performance/<STEM>.{json,md,png,svg}
N_OBJECTS, N_WAVELENGTHS = 16, 31
TIPTOP_MODES = ('sequential', 'threads', 'processes', 'cpu')


def write_tiptop_configs(observations, wavelengths, fov):
    config_dir = OUTPUT / 'tiptop_outputs'
    names = []
    for i, obs in enumerate(observations):
        names.append(f'MUSE_NFM_batch_{i:02d}')
        write_MUSE_tiptop_ini(config_dir / f'{names[-1]}.ini', wavelengths, fov, obs)
    return config_dir, names


def tiptop_pool(mode, workers):
    ''' The executor of a TIPTOP mode (None for the sequential loop); started once and reused by the warm-up and the timed runs '''
    if mode == 'sequential':
        return None
    if mode == 'threads':
        return ThreadPoolExecutor(workers, initializer=tiptop_worker_init, initargs=(DEVICE.index, True))
    use_gpu = mode == 'processes'
    return ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=tiptop_worker_init,
                               initargs=(DEVICE.index if use_gpu else None, use_gpu))


def tiptop_batch(config_dir, names, pool):
    ''' TIPTOP: one `.ini` per object, run in the pool (or one after the other); returns the cube [N_obj, nWvl, N, N] '''
    def run():
        if pool is None:
            cubes = [simulate_tiptop_ini(config_dir, name) for name in names]
        else:
            cubes = list(pool.map(simulate_tiptop_ini, [config_dir] * len(names), names))
        synchronize()
        return dict(cube=np.stack(cubes))
    return run


def tiptorch_batch(observations, wavelengths, fov):
    ''' TipTorch: the 16 configurations merged into one PSFModelNFM(multiple_obs=True), rendered by a single forward pass '''
    from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

    def run():
        t0 = time.perf_counter()
        configs = [MUSE_tiptorch_config(wavelengths, fov, obs) for obs in observations]
        model = PSFModelNFM(configs, multiple_obs=True, LO_NCPAs=True, N_spline_nodes=5, device=DEVICE)
        synchronize()
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            PSFs = model()
        synchronize()
        t_render = time.perf_counter() - t0
        assert PSFs.shape[:2] == (len(observations), len(wavelengths)), PSFs.shape
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1e9 if DEVICE.type == 'cuda' else 0.0
        cube = PSFs.cpu().numpy()
        model.cleanup()
        return dict(cube=cube, setup_s=t_setup, render_s=t_render, peak_GB=peak)
    return run


def tiptorch_sequential(observations, wavelengths, fov):
    ''' TipTorch without batching: one single-source PSFModelNFM per object, for context '''
    from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

    def run():
        cubes = []
        for obs in observations:
            model = PSFModelNFM([MUSE_tiptorch_config(wavelengths, fov, obs)], multiple_obs=False, LO_NCPAs=True, N_spline_nodes=5, device=DEVICE)
            with torch.no_grad():
                cubes.append(model()[0].cpu().numpy())
            model.cleanup()
        synchronize()
        return dict(cube=np.stack(cubes))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--repeats', type=int, default=5, help='timed TipTorch runs (after one warm-up run)')
    parser.add_argument('--tiptop-repeats', type=int, default=2, help='timed TIPTOP runs per mode (after one warm-up run)')
    parser.add_argument('--tiptop-modes', nargs='+', default=['sequential', 'threads', 'processes'], choices=TIPTOP_MODES,
                        help='TIPTOP acceleration modes to time (GPU modes by default; cpu is opt-in)')
    parser.add_argument('--threads', type=int, default=4, help='workers of the thread-pool mode')
    parser.add_argument('--processes', type=int, default=4, help='workers of the GPU process-pool mode (~12 GB of VRAM each)')
    parser.add_argument('--cpu-workers', type=int, default=32, help='workers of the CPU process-pool mode (one core each)')
    parser.add_argument('--fov', type=int, default=100, help='requested science field of view [px] (25 mas pixels)')
    parser.add_argument('--merge', action='store_true', help='keep the stored rows of the TIPTOP modes not run this time (re-run one expensive mode)')
    args = parser.parse_args()

    select_GPU_for_cupy()
    observations = random_observations(N_OBJECTS, seed=0)
    wavelengths = MUSE_NFM_wavelengths(N_WAVELENGTHS)
    group = f'{N_OBJECTS} objects x {N_WAVELENGTHS} wavelengths'
    print(f'device: {DEVICE}; {group}, {args.fov} px requested')

    print('\nTipTorch, one batched forward pass (PSFModelNFM, multiple_obs=True)')
    tt_median, tt_times, tt = timed_runs(tiptorch_batch(observations, wavelengths, args.fov), args.repeats, label='TipTorch batched')
    assert tt['cube'].shape[:2] == (N_OBJECTS, N_WAVELENGTHS) and np.isfinite(tt['cube']).all(), tt['cube'].shape
    print(f"    setup {tt['setup_s']:.2f} s + forward {tt['render_s']:.3f} s (last run), peak VRAM {tt['peak_GB']:.1f} GB, cube {tt['cube'].shape}")

    print('\nTipTorch, one model per object (for context)')
    seq_median, seq_times, seq = timed_runs(tiptorch_sequential(observations, wavelengths, args.fov), args.repeats, label='TipTorch per object')
    assert seq['cube'].shape[:2] == (N_OBJECTS, N_WAVELENGTHS)

    rows, gains = [], {}
    config_dir, names = write_tiptop_configs(observations, wavelengths, args.fov)
    workers = dict(sequential=1, threads=args.threads, processes=args.processes, cpu=args.cpu_workers)
    labels = dict(sequential='TIPTOP sequential', threads=f'TIPTOP {args.threads} threads (GPU)', processes=f'TIPTOP {args.processes} processes (GPU)',
                  cpu=f'TIPTOP {args.cpu_workers} processes (CPU)')
    for mode in args.tiptop_modes:
        print(f'\n{labels[mode]}')
        free_cupy_pool()
        pool = tiptop_pool(mode, workers[mode])
        try:
            median, times, p3 = timed_runs(tiptop_batch(config_dir, names, pool), args.tiptop_repeats, label=labels[mode])
        except Exception as e: # e.g. cupy out of memory with too many GPU workers: report and go on with the other modes
            print(f'    FAILED ({type(e).__name__}: {str(e)[:100]}), mode skipped')
            continue
        finally:
            if pool is not None:
                pool.shutdown()
            free_cupy_pool() # the VRAM cached by this process must not starve the workers of the next mode
        assert p3['cube'].shape[:2] == (N_OBJECTS, N_WAVELENGTHS) and np.isfinite(p3['cube']).all(), p3['cube'].shape
        print(f"    cube {p3['cube'].shape}")
        rows.append(dict(name=labels[mode], group=group, code='TIPTOP', mode=mode, workers=workers[mode], median_s=median, times_s=times, cube_shape=list(p3['cube'].shape)))
        gains[labels[mode]] = median / tt_median
    if args.merge and (OUTPUT / (STEM + '.json')).exists():
        stored = json.load(open(OUTPUT / (STEM + '.json')))['rows']
        rows += [r for r in stored if r['code'] == 'TIPTOP' and r.get('mode') not in args.tiptop_modes]
        gains.update({r['name']: r['median_s'] / tt_median for r in rows if r['name'] not in gains})
    assert rows, 'no TIPTOP mode completed'

    rows += [
        dict(name='TipTorch, one batched forward pass', group=group, code='TipTorch', median_s=tt_median, times_s=tt_times, setup_s=tt['setup_s'],
             render_s=tt['render_s'], cube_shape=list(tt['cube'].shape), peak_VRAM_GB=tt['peak_GB']),
        dict(name='TipTorch, one model per object', group=group, code='context', median_s=seq_median, times_s=seq_times, cube_shape=list(seq['cube'].shape)),
    ]
    fastest = min((r for r in rows if r['code'] == 'TIPTOP'), key=lambda r: r['median_s'])
    n_PSFs = N_OBJECTS * N_WAVELENGTHS
    report(f'MUSE NFM batch, {group}: TIPTOP vs pure TipTorch', rows, STEM, gains={group: fastest['median_s'] / tt_median},
           subtitle=f'{args.fov} px requested, TIPTOP median of {args.tiptop_repeats} runs, TipTorch of {args.repeats}; speed-up against the fastest TIPTOP mode',
           extra=dict(gain_per_mode=gains, observations=observations, fov_requested=args.fov, per_PSF_ms={r['name']: 1e3 * r['median_s'] / n_PSFs for r in rows}))
    ok = all(g > 1 for g in gains.values())
    print(f"[{'PASS' if ok else 'FAIL'}] {group}: " + ', '.join(f"{r['name']} {r['median_s']:.1f} s" for r in rows if r['code'] == 'TIPTOP')
          + f"; TipTorch {tt_median:.2f} s batched ({seq_median:.2f} s per object) -> TipTorch {fastest['median_s'] / tt_median:.0f}x faster than the fastest TIPTOP mode ({fastest['name']})")
    return gains


if __name__ == '__main__':
    main()
