"""
Visual comparison of TIPTOP PSFs computed with the P3 backend (`tiptop.baseSimulation`) and with the TipTorch backend
(`tiptop.TipTop_integration`): both are run on the same `perfTest` configuration and rendered side by side, with the
normalized difference and the radial profiles. One figure per test:

    SCAO  - ERIS.ini      VLT 8 m, 1.65 um, 14 mas pixels, no LO loop
    MCAO  - MAVIStest.ini VLT 8 m, 8 LGS, 3 NGS, 9 science pointings at 550 nm, 7 mas pixels
    ELT   - METIS.ini     ELT 38.5 m SCAO, 3.7 um, 4 mas pixels, ELT pupil FITS, M1 static map, wind shake, extra error, 6 mas jitter

Requirements: the `TipTop` conda env (P3, MASTSEL, TIPTOP, TipTorch), the TIPTOP repository next to the TipTorch one or
`TIPTOP_PATH` pointing to it (its `tiptop/data` FITS files are needed for METIS), and optionally a CUDA device.
Runtime about one minute in total, dominated by P3 (MAVIStest ~15 s, METIS ~10 s). Figures go to tests/runs/visual_comparison/.
Run: python tests/test_tiptop_visual_comparison.py  (the TipTop env has no pytest).
"""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from tiptorch.tools.tiptop_integration import PSF_radial_profile

TIPTOP_ROOT = Path(os.environ.get('TIPTOP_PATH', Path(__file__).resolve().parents[2] / 'TIPTOP'))
PERF_TEST = TIPTOP_ROOT / 'tiptop' / 'perfTest'
OUTPUT = Path(__file__).resolve().parent / 'runs' / 'visual_comparison'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SR_TOLERANCE = 0.15 # relative, on the rendered pointings


def _host(x):
    return np.asarray(x.get() if hasattr(x, 'get') else x, dtype=float)


def _per_pointing(values, nWvl):
    ''' TIPTOP metric lists ([nPointings] or [nWvl][nPointings]) as a float array at the first wavelength '''
    return np.array([float(_host(v)) for v in (values[0] if nWvl > 1 else values)])


def _run_both(config):
    ''' Run the P3 and the TipTorch baseSimulation on perfTest/<config>.ini; PSF cubes are [nPointings, N, N] at the first wavelength '''
    from tiptop.baseSimulation import baseSimulation as P3Simulation
    from tiptop.TipTop_integration import baseSimulation as TipTorchSimulation

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    cwd = Path.cwd()
    os.chdir(TIPTOP_ROOT) # the configs reference tiptop/data relative to the TIPTOP root
    try:
        for name, Simulation, kwargs in (('P3', P3Simulation, {}), ('TipTorch', TipTorchSimulation, {'tiptorch_device': DEVICE})):
            t0 = time.time()
            sim = Simulation(str(PERF_TEST), config, str(OUTPUT), f'{config}_{name}', verbose=False, **kwargs)
            sim.doOverallSimulation()
            elapsed = time.time() - t0
            sim.computeMetrics()
            cube = _host(sim.cubeResultsArray)
            cube = cube[0] if sim.nWvl > 1 else cube
            results[name] = dict(
                cube = cube / cube.sum(axis=(-2, -1), keepdims=True), # P3 cubes are not unit-flux in every configuration
                psInMas = float(_host(sim.psInMas)), wvl = float(sim.wvl[0]), time = elapsed,
                sr = _per_pointing(sim.sr, sim.nWvl), fwhm = _per_pointing(sim.fwhm, sim.nWvl),
                HO_res = _host(sim.HO_res), LO_res = _host(sim.LO_res) if sim.LOisOn else None,
                pointings = _host(sim.pointings),
            )
    finally:
        os.chdir(cwd)
    return results


def _crop(image, size):
    c = image.shape[0] // 2
    return image[c - size//2 : c + size//2, c - size//2 : c + size//2]


def _render(config, title, results, pointings, crop):
    P3, TT = results['P3'], results['TipTorch']
    ps = TT['psInMas']
    extent = np.array([-1, 1, -1, 1]) * crop/2 * ps
    fig, axes = plt.subplots(len(pointings), 4, figsize=(18, 4.3*len(pointings)), squeeze=False)

    for row, i in enumerate(pointings):
        a, b = _crop(P3['cube'][i], crop), _crop(TT['cube'][i], crop)
        peak = max(a.max(), b.max())
        norm = LogNorm(vmin=peak*1e-5, vmax=peak)
        x, y = P3['pointings'][:, i]
        for ax, image, res, label in ((axes[row, 0], a, P3, 'P3 + MASTSEL'), (axes[row, 1], b, TT, 'TipTorch + MASTSEL')):
            ax.imshow(image, norm=norm, extent=extent, cmap='inferno', origin='lower')
            LO = f", LO {res['LO_res'][i]:.0f} nm" if res['LO_res'] is not None else ''
            ax.set_title(f"{label} - pointing {i} ({x:.1f}\", {y:.1f}\")\nSR {res['sr'][i]:.3f}, FWHM {res['fwhm'][i]:.1f} mas, HO {res['HO_res'][i]:.0f} nm{LO}", fontsize=9)
            ax.set_xlabel('[mas]')

        difference = (b - a) / a.max()
        limit = np.abs(difference).max()
        im = axes[row, 2].imshow(difference, extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit, origin='lower')
        axes[row, 2].set_title(f'(TipTorch - P3) / P3 peak, max |diff| = {limit:.3f}', fontsize=9)
        axes[row, 2].set_xlabel('[mas]')
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046)

        for res, label, style in ((P3, 'P3', '-'), (TT, 'TipTorch', '--')):
            radii, profile = PSF_radial_profile(torch.as_tensor(res['cube'][i], dtype=torch.float64)[None], ps)
            keep = radii <= extent[1]
            axes[row, 3].semilogy(radii[keep].numpy(), profile[0, keep].numpy(), style, label=label)
        axes[row, 3].set_title('radial profile (unit flux)', fontsize=9)
        axes[row, 3].set_xlabel('radius [mas]')
        axes[row, 3].grid(alpha=0.3)
        axes[row, 3].legend()

    fig.suptitle(f"{config}: {title}\n{TT['wvl']*1e9:.0f} nm, {ps:g} mas/pixel, log10 stretch over 5 decades  |  run time P3 {P3['time']:.1f} s, TipTorch {TT['time']:.1f} s ({DEVICE})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT / f'{config}_P3_vs_TipTorch.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _visual_case(config, title, pointings, crop):
    results = _run_both(config)
    P3, TT = results['P3'], results['TipTorch']
    assert P3['cube'].shape == TT['cube'].shape, (P3['cube'].shape, TT['cube'].shape)
    assert np.isfinite(TT['cube']).all() and np.isfinite(P3['cube']).all()
    path = _render(config, title, results, pointings, crop)
    assert path.exists()
    for i in pointings:
        assert abs(TT['sr'][i] - P3['sr'][i]) <= SR_TOLERANCE * P3['sr'][i], (i, TT['sr'][i], P3['sr'][i])
    print(f'{config}: figure saved to {path}')
    return results


def test_visual_SCAO_ERIS():
    _visual_case('ERIS', 'SCAO on the VLT (no LO loop)', pointings=[0], crop=64)


def test_visual_MCAO_MAVIS():
    # on-axis, a 15" pointing and a 21.2" corner pointing of the 3x3 science grid
    _visual_case('MAVIStest', 'MCAO on the VLT, 8 LGS + 3 NGS, 9 science pointings', pointings=[4, 1, 0], crop=96)


def test_visual_ELT_METIS():
    _visual_case('METIS', 'SCAO on the ELT with pupil FITS, M1 static map, wind shake, extra error and telescope jitter', pointings=[0], crop=160)


if __name__ == '__main__':
    for test in (test_visual_SCAO_ERIS, test_visual_MCAO_MAVIS, test_visual_ELT_METIS):
        test()
    print('visual comparison figures written to', OUTPUT)
