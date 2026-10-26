"""
Shared helpers of the TIPTOP vs TipTorch performance tests (`tests/test_tiptop_*_performance.py`):

- the astro-tiptop checkouts (`TipTop_path` of project_config.json) on `sys.path`, P3 / MASTSEL (cupy) on the same GPU as TipTorch;
- `timed_runs`: one warm-up call followed by `repeats` timed calls (wall clock, CUDA-synchronized), returning the median and all times;
- MUSE NFM configurations for both codes from the same system description: TIPTOP's `perfTest/MUSE_LTAO.ini` (written as a temporary
  `.ini` with the local VLT pupil FITS) and TipTorch's `parameter_files/muse_ltao.ini` (loaded through `ConfigManager`), with the science
  wavelengths, the field of view and the observing conditions (seeing, L0, zenith angle, wind) set per object;
- `MUSE_NFM_wavelengths`: the N evenly spaced slices of the 475-935 nm MUSE NFM spectrum (3681 slices of 0.125 nm in the data cubes);
- `report`: a text table of the timings and a JSON record in `tests/runs/performance/`.
"""

import json
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch

from tiptorch._config import DATA_FOLDER, default_device, project_settings

ASTRO_TIPTOP = Path(project_settings['TipTop_path'])
for repo in ('TIPTOP', 'P3', 'MASTSEL', 'SYMAO', 'SEEING'): # the astro-tiptop checkouts, importable without installing them
    if str(ASTRO_TIPTOP / repo) not in sys.path:
        sys.path.insert(0, str(ASTRO_TIPTOP / repo))

TIPTOP_ROOT = ASTRO_TIPTOP / 'TIPTOP'
PERF_TEST   = TIPTOP_ROOT / 'tiptop' / 'perfTest'
OUTPUT      = Path(__file__).resolve().parent / 'runs' / 'performance'
DEVICE      = torch.device(default_device) if torch.cuda.is_available() else torch.device('cpu')

MUSE_TIPTOP_INI   = PERF_TEST / 'MUSE_LTAO.ini'
MUSE_TIPTORCH_INI = DATA_FOLDER / 'parameter_files' / 'muse_ltao.ini'
VLT_PUPIL_FITS    = DATA_FOLDER / 'calibrations' / 'VLT_CALIBRATION' / 'VLT_PUPIL' / 'ut4pupil320.fits'

MUSE_NFM_LAMBDA_MIN, MUSE_NFM_LAMBDA_MAX, MUSE_NFM_N_SLICES = 475e-9, 935e-9, 3681 # [m], the MUSE NFM spectral range and cube depth


def select_GPU_for_cupy(device=DEVICE):
    ''' P3 and MASTSEL compute on cupy's current device (GPU 0 by default): put them on the GPU TipTorch uses. Thread-local in cupy. '''
    if device.type != 'cuda':
        return
    try:
        import cupy as cp
        cp.cuda.Device(device.index or 0).use()
    except ImportError:
        pass


def free_cupy_pool():
    ''' Release the VRAM cached by cupy's memory pool in this process (TIPTOP's multi-wavelength runs leave tens of GB cached, which
    would starve the worker processes of a pool started afterwards) '''
    try:
        import cupy as cp
        cp.fft.config.get_plan_cache().clear() # the cuFFT plans of MASTSEL's grids keep their work areas (several GB) alive
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass


def synchronize(device=DEVICE):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def timed_runs(fn, repeats, label='', warmup=True, verbose=True):
    '''
    Run fn() once untimed (CUDA context, kernel compilation, file caches) and then `repeats` times timed, with the GPU synchronized
    before and after each call. Returns (median_s, times_s, last_result).
    '''
    result = None
    if warmup:
        synchronize()
        t0 = time.perf_counter()
        result = fn()
        synchronize()
        if verbose:
            print(f'    {label} warm-up (cold, untimed): {time.perf_counter() - t0:.3f} s')
    times = []
    for i in range(repeats):
        synchronize()
        t0 = time.perf_counter()
        result = fn()
        synchronize()
        times.append(time.perf_counter() - t0)
        if verbose:
            print(f'    {label} run {i+1}/{repeats}: {times[-1]:.3f} s')
    return float(np.median(times)), times, result


def MUSE_NFM_wavelengths(n):
    ''' n evenly spaced wavelengths [m] spanning the MUSE NFM spectrum (n = 3681 gives the cube slices) '''
    return np.linspace(MUSE_NFM_LAMBDA_MIN, MUSE_NFM_LAMBDA_MAX, n)


def random_observations(n, seed=0):
    ''' n sets of observing conditions {seeing ["], L0 [m], zenith angle [deg], wind speed factor} drawn from typical MUSE NFM ranges '''
    rng = np.random.default_rng(seed)
    return [dict(seeing=float(s), L0=float(l), zenith_angle=float(z), wind_factor=float(w))
            for s, l, z, w in zip(rng.uniform(0.5, 1.2, n), rng.uniform(15.0, 40.0, n), rng.uniform(5.0, 50.0, n), rng.uniform(0.5, 2.0, n))]


def tiptop_VRAM_per_slice_GB(wavelength_m):
    '''
    Peak VRAM [GB] that a TIPTOP multi-wavelength MUSE NFM simulation (100 px requested, LO loop on) holds per science wavelength, from
    cupy pool high-water marks of chunk runs on 2026-10-22: ~3.5 GB per slice at 475 nm (4 slices 14 GB, 8 slices 24 GB), ~1 GB at 600 nm,
    ~0.45 GB from 700 nm on (16 slices 6.4 GB) and ~0.3 GB at 930 nm. MASTSEL oversamples the bluest slices the most, hence the steps.
    '''
    nm = np.asarray(wavelength_m) * 1e9
    return np.where(nm < 560, 3.5, np.where(nm < 660, 1.1, 0.5))


def VRAM_budgeted_chunks(wavelengths, budget_GB):
    ''' Contiguous chunks of `wavelengths` whose estimated TIPTOP VRAM (tiptop_VRAM_per_slice_GB) stays within budget_GB each '''
    chunks, current, used = [], [], 0.0
    for wavelength, cost in zip(wavelengths, tiptop_VRAM_per_slice_GB(wavelengths)):
        if current and used + cost > budget_GB:
            chunks.append(np.array(current))
            current, used = [], 0.0
        current.append(wavelength)
        used += cost
    if current:
        chunks.append(np.array(current))
    return chunks


def _fmt_list(values):
    return '[' + ', '.join(f'{v:.8g}' for v in values) + ']'


def write_MUSE_tiptop_ini(path, wavelengths, fov, observation=None):
    '''
    TIPTOP MUSE NFM configuration `path` from perfTest/MUSE_LTAO.ini: science wavelengths [m], FieldOfView [px], the local VLT pupil FITS,
    TechnicalFoV (MASTSEL's LO module needs it) and, if given, the observing conditions of `random_observations`
    '''
    config = ConfigParser(inline_comment_prefixes=(';', '#'))
    config.optionxform = str
    config.read(MUSE_TIPTOP_INI)
    config['telescope']['PathPupil']    = f"'{VLT_PUPIL_FITS}'"
    config['telescope']['TechnicalFoV'] = '120'
    config['sources_science']['Wavelength'] = _fmt_list(wavelengths)
    config['sensor_science']['FieldOfView'] = str(int(fov))
    if observation is not None:
        config['telescope']['ZenithAngle'] = f"{observation['zenith_angle']:.6g}"
        config['atmosphere']['Seeing']     = f"{observation['seeing']:.6g}"
        config['atmosphere']['L0']         = f"{observation['L0']:.6g}"
        wind = np.asarray(eval(config['atmosphere']['WindSpeed']), dtype=float) * observation['wind_factor']
        config['atmosphere']['WindSpeed']  = _fmt_list(wind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        config.write(f)
    return path


def MUSE_tiptorch_config(wavelengths, fov, observation=None):
    ''' TipTorch MUSE NFM configuration (list form, one observation) matching write_MUSE_tiptop_ini '''
    from tiptorch.managers.config_manager import ConfigManager

    config = ConfigManager().Load(MUSE_TIPTORCH_INI)
    config['sources_science']['Wavelength'] = [float(w) for w in wavelengths]
    config['sensor_science']['FieldOfView'] = int(fov)
    if observation is not None:
        config['telescope']['ZenithAngle'] = observation['zenith_angle']
        config['atmosphere']['Seeing']     = observation['seeing']
        config['atmosphere']['L0']         = observation['L0']
        config['atmosphere']['WindSpeed']  = (np.asarray(config['atmosphere']['WindSpeed'], dtype=float) * observation['wind_factor']).tolist()
    return config


def tiptop_worker_init(device_index=None, use_gpu=True):
    '''
    Initializer of the TIPTOP worker threads / processes: selects the GPU (cupy's current device is thread-local) or, in a fresh
    process, disables the GPU of P3 and MASTSEL before they are imported (CPU mode, one numpy process per worker)
    '''
    import os
    if not use_gpu:
        for package in ('P3', 'MASTSEL', 'SEEING'): # every astro-tiptop package has its own flag (P3's telescope pupil comes from SEEING)
            os.environ[f'{package}_DISABLE_GPU'] = 'TRUE'
        return
    try:
        import cupy as cp
        cp.cuda.Device(device_index if device_index is not None else 0).use()
    except ImportError:
        pass


def simulate_tiptop_ini(config_dir, name, out_npy=None, free_vram=False):
    '''
    Run TIPTOP (P3 backend) on <config_dir>/<name>.ini and return the on-axis PSF cube [nWvl, N, N] (float32), or, with `out_npy`, save it
    there and return the path (process pools: a 3681-slice cube is several GB, better on disk than through a pipe). Importable by workers.
    `free_vram` releases cupy's pool and cuFFT plan cache afterwards: a worker that runs configurations with different wavelength grids
    caches a plan (with its work area) per grid and grows by several GB per configuration otherwise.
    '''
    from tiptop.baseSimulation import baseSimulation
    sim = baseSimulation(str(config_dir), name, str(config_dir), f'{name}_out', verbose=False)
    sim.doOverallSimulation()
    cube = np.asarray(sim.cubeResultsArray)
    cube = (cube[:, 0] if cube.ndim == 4 else cube).astype(np.float32)
    if free_vram:
        del sim
        free_cupy_pool()
    if out_npy is not None:
        np.save(out_npy, cube)
        return str(out_npy)
    return cube


def run_tiptop_simulation(Simulation, config_dir, name, output_dir, **kwargs):
    ''' Construct a TIPTOP simulation on <config_dir>/<name>.ini and run it end to end (configuration, PSDs, LO loop, PSFs) '''
    sim = Simulation(str(config_dir), name, str(output_dir), f'{name}_out', verbose=False, **kwargs)
    sim.doOverallSimulation()
    return sim


# ----------------------------------------------- Presentation outputs -----------------------------------------------
# Colors of the two codes (the validated reference palette of the dataviz skill, slots 1 / 2 / 3); text keeps neutral inks
COLORS = {'TIPTOP': '#2a78d6', 'TipTorch': '#eb6834', 'context': '#1baf7a'}
INK, INK_SOFT, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _seconds(t):
    return f'{t*1e3:.0f} ms' if t < 1 else (f'{t:.2f} s' if t < 10 else (f'{t:.1f} s' if t < 100 else f'{t/60:.1f} min'))


def markdown_table(title, rows, gains=None):
    ''' Markdown table of the timing rows (median, min, max, runs, cube) and of the speed-ups, ready to paste '''
    lines = [f'**{title}**', '', '| case | code | median | min | max | runs | PSF cube |', '|---|---|---|---|---|---|---|']
    for r in rows:
        t = r['times_s']
        cube = ' x '.join(str(n) for n in r['cube_shape']) if r.get('cube_shape') else ''
        lines.append(f"| {r.get('group', '')} | {r['name']} | {_seconds(r['median_s'])} | {_seconds(min(t))} | {_seconds(max(t))} | {len(t)} | {cube} |")
    if gains:
        lines += ['', '| speed-up of TipTorch | |', '|---|---|'] + [f'| {k} | {v:.1f}x |' for k, v in gains.items()]
    return '\n'.join(lines) + '\n'


def draw_bars(ax, rows, gains=None, label_width=22):
    '''
    Draw on `ax` the horizontal bars of the median run times (log scale: the codes differ by orders of magnitude), one bar per row grouped
    by row['group'], colored by row['code'] ('TIPTOP' / 'TipTorch' / 'context'), the value at the tip of every bar and the speed-up of
    TipTorch under the group label. Shared by the per-test charts and the summary figure.
    '''
    import textwrap
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(r['group'] for r in rows))
    y_ticks, y_labels, y = [], [], 0.0
    for g in groups:
        members = [r for r in rows if r['group'] == g]
        y0 = y
        for r in members:
            ax.barh(y, r['median_s'], height=0.55, color=COLORS[r['code']], edgecolor='none')
            ax.text(r['median_s'] * 1.12, y, f"{r['name']}: {_seconds(r['median_s'])}", va='center', ha='left', fontsize=9, color=INK)
            y -= 1.0
        y_ticks.append((y0 + y + 1.0) / 2)
        label = '\n'.join(textwrap.wrap(g, label_width))
        if gains and g in gains:
            label += f"\nTipTorch {gains[g]:.0f}x faster" if gains[g] >= 10 else f"\nTipTorch {gains[g]:.1f}x faster"
        y_labels.append(label)
        y -= 0.7
    ax.set_xscale('log')
    x_min, x_max = min(r['median_s'] for r in rows), max(r['median_s'] for r in rows)
    ax.set_xlim(x_min / 2, x_max * 10 ** (2.0 * max(len(r['name']) for r in rows) / 45)) # room for the longest label right of the longest bar
    ax.set_yticks(y_ticks, y_labels, fontsize=9.5, color=INK)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    codes = [c for c in ('TIPTOP', 'TipTorch', 'context') if any(r['code'] == c for r in rows)]
    names = {'TIPTOP': 'TIPTOP (P3 + MASTSEL)', 'TipTorch': 'TipTorch', 'context': 'TipTorch, unbatched (context)'}
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in codes], [names[c] for c in codes], loc='lower right', frameon=False, fontsize=9)


def figure_height(rows):
    return 1.4 + 0.5 * len(rows) + 0.5 * len(set(r['group'] for r in rows))


def plot_bars(title, subtitle, rows, path, gains=None):
    ''' Bar chart of the timing rows (see draw_bars) with a title and a wrapped subtitle, saved as PNG and SVG '''
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, figure_height(rows)))
    draw_bars(ax, rows, gains)
    ax.set_xlabel('run time [s], median over repeats (log scale)', color=INK_SOFT, fontsize=9.5)
    ax.set_title(title + ('\n' + '\n'.join(textwrap.wrap(subtitle, 110)) if subtitle else ''), loc='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=160)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
    return path.with_suffix('.png')


def report(title, rows, stem, gains=None, subtitle='', extra=None):
    '''
    Print the table of timing rows {name, group, code, median_s, times_s, cube_shape, ...} and write to tests/runs/performance/:
    <stem>.json (rows, gains, extra), <stem>.md (markdown table) and <stem>.png / .svg (bar chart). Returns the JSON record.
    '''
    OUTPUT.mkdir(parents=True, exist_ok=True)
    width = max(len(r['name']) for r in rows) + 2
    print(f'\n{title}')
    print(f"{'case':<{width}}{'median [s]':>12}{'min [s]':>10}{'max [s]':>10}  runs")
    for r in rows:
        t = r['times_s']
        print(f"{r['name']:<{width}}{r['median_s']:>12.3f}{min(t):>10.3f}{max(t):>10.3f}  {len(t)}")
    if gains:
        print('speed-up of TipTorch: ' + ', '.join(f'{k} {v:.1f}x' for k, v in gains.items()))
    record = dict(title=title, subtitle=subtitle, device=str(DEVICE), rows=rows, gains=gains or {}, **(extra or {}))
    with open(OUTPUT / f'{stem}.json', 'w') as f:
        json.dump(record, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    (OUTPUT / f'{stem}.md').write_text(markdown_table(title + (f' ({subtitle})' if subtitle else ''), rows, gains))
    figure = plot_bars(title, subtitle, rows, OUTPUT / stem, gains)
    print(f'results written to {OUTPUT / stem}.json / .md / .png / .svg')
    return record
