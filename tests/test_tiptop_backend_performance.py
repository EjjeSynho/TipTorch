"""
Runtime of TIPTOP with its P3 backend (`tiptop.baseSimulation`) against TIPTOP with the integrated TipTorch backend
(`tiptop.TipTorchSimulation`) on the perfTest configurations of `tests/test_tiptop_visual_comparison.py`, without the figures:

    SCAO  - ERIS.ini      VLT 8 m, 1.65 um, 14 mas pixels, no LO loop
    MCAO  - MAVIStest.ini VLT 8 m, 8 LGS, 3 NGS, 9 science pointings at 550 nm, 7 mas pixels
    ELT   - METIS.ini     ELT 38.5 m SCAO, 3.7 um, 4 mas pixels, ELT pupil FITS, M1 static map, wind shake, extra error, 6 mas jitter
    SLAO  - ERIS_LGS.ini  VLT 8 m, one LGS at 90 km (cone effect), one NGS for the LO loop, 1.65 um, 14 mas pixels

Every case is run end to end (configuration parsing, model build, PSDs, LO loop, science PSFs: constructor + doOverallSimulation) once
untimed and then --repeats times (default 5) with each backend; the gain of TipTorch is the ratio of the median run times. Both
backends run on the GPU of the project config (P3 / MASTSEL through cupy on the same device); the metrics (SR, FWHM, EE) are not timed.

Requirements: the astro-tiptop checkouts at `TipTop_path` of project_config.json (branch tipTorch2 of TIPTOP for TipTorchSimulation.py,
its `tiptop/data` FITS files for METIS) and a CUDA device. Runtime about 5 minutes (dominated by P3: MAVIStest ~15 s, METIS ~10 s per run).
Results: tests/runs/performance/backend_performance.{json,md,png,svg}. Run: python tests/test_tiptop_backend_performance.py [--repeats N] [--cases ERIS ...]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiptop_performance_utils import DEVICE, PERF_TEST, TIPTOP_ROOT, OUTPUT, report, run_tiptop_simulation, select_GPU_for_cupy, timed_runs

CASES = {
    'ERIS':      'SCAO on the VLT (no LO loop)',
    'MAVIStest': 'MCAO on the VLT, 8 LGS + 3 NGS, 9 science pointings',
    'METIS':     'SCAO on the ELT with pupil FITS, M1 static map, wind shake, extra error and telescope jitter',
    'ERIS_LGS':  'SLAO on the VLT (one LGS, cone effect, LO loop on one NGS)',
}


def _host(x):
    return np.asarray(x.get() if hasattr(x, 'get') else x, dtype=float)


def _backend_runs(config, repeats):
    ''' Median run times of both backends on perfTest/<config>.ini and the PSF cube shapes '''
    from tiptop.baseSimulation import baseSimulation as P3Simulation
    from tiptop.TipTorchSimulation import baseSimulation as TipTorchSimulation

    scratch = OUTPUT / 'tiptop_outputs'
    scratch.mkdir(parents=True, exist_ok=True)
    backends = (('P3', P3Simulation, {}), ('TipTorch', TipTorchSimulation, {'tiptorch_device': DEVICE}))
    rows, shapes = [], {}

    cwd = Path.cwd()
    os.chdir(TIPTOP_ROOT) # the configs reference tiptop/data relative to the TIPTOP root
    try:
        for name, Simulation, kwargs in backends:
            run = lambda: run_tiptop_simulation(Simulation, PERF_TEST, config, scratch, **kwargs)
            median, times, sim = timed_runs(run, repeats, label=f'{config} {name}')
            cube = _host(sim.cubeResultsArray)
            shapes[name] = cube.shape
            assert np.isfinite(cube).all(), f'{config} {name}: non-finite PSF'
            rows.append(dict(name=f'{name} backend', group=config, code='TIPTOP' if name == 'P3' else 'TipTorch', backend=name, config=config,
                             median_s=median, times_s=times, cube_shape=list(cube.shape)))
            del sim
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        os.chdir(cwd)
    assert shapes['P3'] == shapes['TipTorch'], shapes
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--repeats', type=int, default=5, help='timed runs per backend and case (after one warm-up run)')
    parser.add_argument('--cases', nargs='+', default=list(CASES), choices=list(CASES))
    args = parser.parse_args()

    select_GPU_for_cupy()
    print(f'device: {DEVICE}, {args.repeats} timed runs per case after a warm-up run')

    rows, gains = [], {}
    for config in args.cases:
        print(f'\n{config}: {CASES[config]}')
        case_rows = _backend_runs(config, args.repeats)
        rows += case_rows
        P3, TT = (next(r for r in case_rows if r['backend'] == b) for b in ('P3', 'TipTorch'))
        gains[config] = P3['median_s'] / TT['median_s']
        ok = gains[config] > 1
        print(f"[{'PASS' if ok else 'FAIL'}] {config}: P3 {P3['median_s']:.2f} s, TipTorch {TT['median_s']:.2f} s -> TipTorch is {gains[config]:.1f}x faster"
              f" (cube {tuple(P3['cube_shape'])})")

    report('TIPTOP end-to-end run time: P3 backend vs TipTorch backend', rows, 'backend_performance', gains=gains,
           subtitle=f'perfTest configurations, median of {args.repeats} runs on {torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"}',
           extra=dict(repeats=args.repeats))
    print('\nmedian gain of the TipTorch backend over P3:')
    for config, gain in gains.items():
        print(f'    {config:<10} {gain:5.1f}x')
    return gains


if __name__ == '__main__':
    main()
