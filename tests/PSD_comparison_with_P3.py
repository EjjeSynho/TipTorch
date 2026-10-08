#%%
"""
PSD-level comparison of TipTorch against P3's `fourierModel` on the MUSE NFM LTAO configuration
(`DATA_FOLDER/parameter_files/muse_ltao.ini`).

Compared term by term, on P3's spatial-frequency grid:
  - the six PSD contributors: fitting, aliasing, WFS noise, spatio-temporal, chromatism, differential refraction;
  - the tomographic reconstructor and the layer / DM projectors (W_tomo, W_alpha, P_beta_DM, P_beta_L, freq_t);
  - the P3-style PSD add-ons that are `TipTorch` members: `TiltFilter`, `FocusErrorPSD`, `ExtraErrorPSD`, `WindShakePSD`
    against P3's `TiltFilter`, `FocusFilter`, `extraErrorPSD` and `windShakePSD`.

TipTorch's oversampling is tuned so that its odd PSD grid has the same frequency step as P3's even grid with the DC on
the same pixel (nOtf = 1153 vs 1152 here); TipTorch maps are compared after dropping their last row and column.

Requirements: the `TipTop` conda env (P3 installed with CuPy), the MUSE LTAO parameter file and the VLT pupil calibration
in the TipTorch data folder, and the TIPTOP repository (`TIPTOP_folder` in project_config.json) for the wind-shake
temporal PSD FITS. Runtime about 20 s on a GPU.

Run `python tests/PSD_comparison_with_P3.py` for the checks and the figures (saved to tests/runs/PSD_comparison/, shown
unless `--no-plots` is given), or execute the `#%%` cells interactively. The `test_*` functions also work under pytest.
"""

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from astropy.io import fits

from tiptorch._config import DATA_FOLDER, default_device, default_torch_type, project_settings
from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager

PATH_INI      = DATA_FOLDER / 'parameter_files' / 'muse_ltao.ini'
WIND_PSD_FILE = Path(project_settings['TIPTOP_folder']) / 'TIPTOP' / 'tiptop' / 'data' / 'morfeo_windshake8ms_psd_2022_1k.fits'
OUTPUT        = (Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()) / 'runs' / 'PSD_comparison'
SHOW_PLOTS    = '--no-plots' not in sys.argv

PSD_TERMS      = ['fitting', 'aliasing', 'WFS noise', 'spatio-temporal', 'chromatism', 'diff. refract']
EXTRA_ERROR    = dict(rms_nm=60.0, exponent=-2.0, k_min=0.0, k_max=0.0) # [telescope] extraErrorNm / Exp / Min / Max
FOCUS_ERROR_NM = 30.0


# ------------------------------------------------ Models ------------------------------------------------
def host(x):
    ''' CuPy / NumPy / torch input as a real NumPy array '''
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().real
    return np.asarray(x.get() if hasattr(x, 'get') else x).real


@lru_cache(maxsize=1)
def build_models():
    ''' P3 fourierModel and a TipTorch model of the same configuration, with TipTorch's frequency step matched to P3's '''
    from p3.aoSystem.fourierModel import fourierModel

    content = PATH_INI.read_text().replace('$CALIBRATIONS_PATH$', '/aoSystem/data/') # P3 resolves calibrations inside its own package
    fd, temp_ini = tempfile.mkstemp(suffix='.ini', dir=PATH_INI.parent)
    with os.fdopen(fd, 'w') as stream:
        stream.write(content)
    try:
        P3 = fourierModel(temp_ini, path_root=None, calcPSF=False, verbose=False, display=False, getErrorBreakDown=False,
                          getFWHM=False, getEncircledEnergy=False, getEnsquaredEnergy=False, displayContour=False)
    finally:
        os.remove(temp_ini)
    P3.ao.tel.extraErrorNm, P3.ao.tel.extraErrorExp = EXTRA_ERROR['rms_nm'], EXTRA_ERROR['exponent']
    P3.ao.tel.extraErrorMin, P3.ao.tel.extraErrorMax = EXTRA_ERROR['k_min'], EXTRA_ERROR['k_max']
    P3.ao.windPsdFile = str(WIND_PSD_FILE)

    manager = ConfigManager()
    config = manager.Convert(manager.Load(str(PATH_INI)), framework='pytorch', device=default_device, dtype=default_torch_type)
    model = TipTorch(AO_config=config, AO_type='LTAO', norm_regime=None, device=default_device, oversampling=1, retain_PSDs=True)
    # P3 samples the PSD at kRef times Nyquist: give TipTorch the same step, which makes its odd grid one pixel larger than P3's
    model.oversampling = float(host(P3.freq.k_).min() / model.sampling_factor.max().item()) * 1.001
    model.Update(grids=True, pupils=True, tomography=True)
    model.ComputePSD() # fills model.PSDs (half grids, rad² at the atmosphere wavelength) and the tomographic operators
    assert model.nOtf == P3.freq.nOtf + 1, (model.nOtf, P3.freq.nOtf)
    return P3, model


# ------------------------------------------------ Grids and units ------------------------------------------------
def on_P3_grid(x, model):
    ''' TipTorch half- or full-grid tensor as a NumPy map on P3's grid; TipTorch's grid is one pixel larger with the DC on the same pixel '''
    x = x.detach()
    if x.shape[-1] != x.shape[-2]:
        x = model.half_PSD_to_full(x)
    return host(x.squeeze())[..., :-1, :-1]


def operator_on_P3_grid(W, model):
    ''' TipTorch operator [1, nOtf_AO_y, nOtf_AO_x, a, b] on the half grid as a complex NumPy array [resAO, resAO, a, b] '''
    W = W[0].detach()
    full = torch.cat([W, torch.flip(W[:, :-1], dims=(0, 1))], dim=1)
    return full.cpu().numpy()[:-1, :-1]


def P3_PSD_to_nm2(P3):
    ''' P3 PSDs are rad² at wvlRef per grid pixel: nm² = PSD * (dk * rad2nm)² '''
    return ((2*host(P3.freq.kcMax_) / P3.freq.resAO) * P3.freq.wvlRef*1e9 / (2*np.pi))**2


def embed_AO(P3, x):
    ''' Place an AO-area map [resAO, resAO] on P3's full grid '''
    i1 = int(np.ceil(P3.freq.nOtf/2 - P3.freq.resAO/2))
    full = np.zeros((P3.freq.nOtf, P3.freq.nOtf), dtype=x.dtype)
    full[i1 : i1 + P3.freq.resAO, i1 : i1 + P3.freq.resAO] = x
    return full


def zero_DC(x):
    x = x.copy()
    x[..., x.shape[-2]//2, x.shape[-1]//2] = 0.0
    return x


# ------------------------------------------------ Terms ------------------------------------------------
def PSD_terms(P3, model):
    ''' The six PSD contributors of both models in rad² at the science wavelength (AO-area terms on the AO grid, fitting on the full grid) '''
    P3_terms = {
        'fitting':         P3.fittingPSD(),
        'aliasing':        P3.aliasingPSD(),
        'WFS noise':       P3.noisePSD(),
        'spatio-temporal': P3.spatioTemporalPSD(),
        'chromatism':      P3.chromatismPSD(),
        'diff. refract':   P3.differentialRefractionPSD(),
    }
    scale = (model.wvl_atm / model.wvl).item()**2 # TipTorch PSDs are rad² at the atmosphere wavelength
    P3_terms = {key: host(value).squeeze() for key, value in P3_terms.items()}
    TT_terms = {key: zero_DC(on_P3_grid(model.PSDs[key], model)) * scale for key in P3_terms}
    return P3_terms, TT_terms


def addon_terms(P3, model):
    ''' TipTorch's P3-style PSD members against their P3 originals, on P3's full grid (nm², the tilt filter is dimensionless) '''
    nm2 = P3_PSD_to_nm2(P3)
    focus = host(P3.FocusFilter())
    P3_terms = {
        'tilt filter': host(P3.TiltFilter()),
        'focus error': focus / focus.sum() * FOCUS_ERROR_NM**2,
        'extra error': host(P3.extraErrorPSD()) * nm2,
        'wind shake':  embed_AO(P3, host(P3.windShakePSD())) * nm2,
    }
    TT_terms = {
        'tilt filter': on_P3_grid(model.TiltFilter(), model),
        'focus error': on_P3_grid(model.FocusErrorPSD(FOCUS_ERROR_NM), model),
        'extra error': on_P3_grid(model.ExtraErrorPSD(**EXTRA_ERROR), model),
        'wind shake':  on_P3_grid(model.WindShakePSD(fits.getdata(WIND_PSD_FILE)), model),
    }
    return P3_terms, TT_terms


def P3_layer_projector(P3, source=0):
    ''' P3's layer-to-direction projector P_beta_L and the temporal frequencies, as P3 builds them inside spatioTemporalPSD '''
    import cupy as cp
    nK, nL = P3.freq.resAO, P3.ao.atm.nL
    heights = P3.ao.atm.heights * P3.strechFactor
    delta_T = P3.ao.rtc.holoop['delay'] / P3.ao.rtc.holoop['rate']
    wind_x, wind_y = np.cos(np.deg2rad(P3.ao.atm.wDir)), np.sin(np.deg2rad(P3.ao.atm.wDir))
    beta_x, beta_y = P3.ao.src.direction[:, source]

    P_beta_L = cp.zeros([nK, nK, 1, nL], dtype=complex)
    freq_t = []
    for j in range(nL):
        freq_t.append(wind_x[j]*P3.freq.kxAO_ + wind_y[j]*P3.freq.kyAO_)
        phase = heights[j] * (beta_x*P3.freq.kxAO_ + beta_y*P3.freq.kyAO_) - delta_T * P3.ao.atm.wSpeed[j] * freq_t[j]
        P_beta_L[:, :, 0, j] = cp.exp(2j*cp.pi*phase)
    return host(P_beta_L), host(cp.stack(freq_t, axis=2))


def reconstructor_terms(P3, model):
    ''' Tomographic operators of both models on the AO grid for layer 0 / guide star 0: magnitudes of the reconstructors, real parts of the phasors '''
    AO_mask = model.mask_corrected_AO.unsqueeze(-1).unsqueeze(-1)
    mask_P3 = host(P3.freq.mskInAO_)
    P_beta_L_P3, freq_t_P3 = P3_layer_projector(P3)
    P3_terms = {
        'W_tomo':    np.abs(host(P3.tomographicReconstructor()))[..., 0, 0],
        'W_alpha':   np.abs(host(P3.Walpha))[..., 0, 0],
        'P_beta_DM': np.real(host(P3.PbetaDM[0]).astype(complex))[..., 0, 0] * mask_P3,
        'P_beta_L':  np.real(P_beta_L_P3)[..., 0, 0] * mask_P3,
        'freq_t':    np.abs(freq_t_P3)[..., 0],
    }
    TT_terms = {
        'W_tomo':    np.abs(operator_on_P3_grid(model.W_tomo, model))[..., 0, 0],
        'W_alpha':   np.abs(operator_on_P3_grid(model.W_alpha * AO_mask, model))[..., 0, 0],
        'P_beta_DM': np.real(operator_on_P3_grid(model.P_beta_DM * AO_mask, model))[..., 0, 0],
        'P_beta_L':  np.real(operator_on_P3_grid(model.P_beta_L * AO_mask, model))[..., 0, 0],
        'freq_t':    np.abs(operator_on_P3_grid((model.freq_t / model.wind_speed.view(1, 1, 1, -1)).unsqueeze(-1), model))[..., 0, 0],
    }
    return P3_terms, TT_terms


# ------------------------------------------------ Statistics and plots ------------------------------------------------
def difference_stats(TT, P3, title=None):
    ''' Relative difference (TipTorch - P3) / P3 in percent where P3 is significant, and the maximum absolute difference relative to P3's peak '''
    significant = np.abs(P3) > 1e-3 * np.abs(P3).max()
    relative = 100 * (TT[significant] - P3[significant]) / P3[significant]
    stats = dict(median_rel=np.median(relative), p90_rel=np.percentile(np.abs(relative), 90), max_abs_rel_peak=100*np.abs(TT - P3).max()/np.abs(P3).max())
    if title:
        print(f'{title:16s}: median {stats["median_rel"]:+6.2f} %,  |rel| 90th pct {stats["p90_rel"]:6.2f} %,  max |diff| {stats["max_abs_rel_peak"]:6.2f} % of peak')
    return stats


def radial_profile(image, dk):
    ''' Azimuthal mean about the central pixel; radii in [1/m] '''
    y, x = np.indices(image.shape)
    r = np.round(np.hypot(x - image.shape[1]//2, y - image.shape[0]//2)).astype(int).ravel()
    profile = np.bincount(r, image.ravel()) / np.bincount(r)
    return np.arange(profile.size) * dk, profile


def _finish(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f'{name}.png', dpi=110)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_maps(TT, P3, title, name, log=True, crop=None):
    ''' TipTorch and P3 maps on a shared color scale plus their relative difference, optionally cropped to the central crop x crop pixels '''
    if crop is not None and crop < TT.shape[0]:
        c = TT.shape[0] // 2
        TT, P3 = TT[c - crop//2 : c + crop//2, c - crop//2 : c + crop//2], P3[c - crop//2 : c + crop//2, c - crop//2 : c + crop//2]
    positive = np.concatenate([P3[P3 > 0].ravel(), TT[TT > 0].ravel()])
    norm = LogNorm(np.percentile(positive, 10), np.percentile(positive, 99.975)) if log and positive.size else None
    difference = 100 * (TT - P3) / np.where(np.abs(P3) > 1e-3*np.abs(P3).max(), P3, np.nan)
    limit = np.nanpercentile(np.abs(difference), 97.5)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, image, label in ((axes[0], TT, 'TipTorch'), (axes[1], P3, 'P3')):
        im = ax.imshow(image, norm=norm, cmap='viridis')
        ax.set_title(f'{label} {title}')
        ax.axis('off')
    fig.colorbar(im, ax=axes[:2], fraction=0.025)
    im = axes[2].imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[2].set_title('(TipTorch - P3) / P3 [%]')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    _finish(fig, name)


def plot_radial_profiles(P3_terms, TT_terms, dk, title, name, ylabel):
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, key in enumerate(P3_terms):
        for terms, label, style in ((P3_terms, 'P3', '--'), (TT_terms, 'TipTorch', '-')):
            radii, profile = radial_profile(terms[key], dk)
            ax.plot(radii, profile, style, color=f'C{i}', label=f'{label} {key}', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(dk, None)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel('spatial frequency [1/m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _finish(fig, name)


# ------------------------------------------------ Checks ------------------------------------------------
def test_PSD_contributors_match_P3():
    P3, model = build_models()
    P3_terms, TT_terms = PSD_terms(P3, model)
    print('PSD contributors (rad^2 at the science wavelength):')
    stats = {key: difference_stats(TT_terms[key], P3_terms[key], key) for key in PSD_TERMS}
    assert abs(stats['fitting']['median_rel']) < 1.0
    assert abs(stats['chromatism']['median_rel']) < 5.0 # air refractive index models differ slightly
    assert abs(stats['spatio-temporal']['median_rel']) < 5.0
    assert all(np.isfinite(TT_terms[key]).all() for key in PSD_TERMS)


def test_tomographic_operators_match_P3():
    P3, model = build_models()
    P3_terms, TT_terms = reconstructor_terms(P3, model)
    print('Tomographic operators (layer 0 / guide star 0):')
    stats = {key: difference_stats(TT_terms[key], P3_terms[key], key) for key in P3_terms}
    for key in ('W_alpha', 'P_beta_DM', 'P_beta_L', 'freq_t'):
        assert stats[key]['max_abs_rel_peak'] < 1.0, key
    assert stats['W_tomo']['max_abs_rel_peak'] < 10.0 # relative errors are dominated by near-zero entries, compare to the peak


def test_tilt_filter_matches_P3():
    P3, model = build_models()
    tilt_P3, tilt_TT = host(P3.TiltFilter()), on_P3_grid(model.TiltFilter(), model)
    assert tilt_TT.shape == tilt_P3.shape
    assert tilt_TT[tilt_TT.shape[0]//2, tilt_TT.shape[1]//2] < 1e-9 # tip/tilt is fully rejected at the origin
    assert np.all((tilt_TT >= 0) & (tilt_TT <= 1))
    assert np.abs(tilt_TT - tilt_P3).max() < 1e-2 # the two grids differ by 0.2 % in frequency step


def test_focus_error_PSD_matches_P3():
    P3, model = build_models()
    P3_terms, TT_terms = addon_terms(P3, model)
    focus_P3, focus_TT = P3_terms['focus error'], TT_terms['focus error']
    np.testing.assert_allclose(model.FocusErrorPSD(FOCUS_ERROR_NM).sum().item(), FOCUS_ERROR_NM**2, rtol=1e-5) # on TipTorch's own grid
    assert np.abs(focus_TT - focus_P3).max() < 1e-2 * focus_P3.max()


def test_extra_error_PSD_matches_P3():
    P3, model = build_models()
    P3_terms, TT_terms = addon_terms(P3, model)
    extra_P3, extra_TT = P3_terms['extra error'], TT_terms['extra error']
    np.testing.assert_allclose(model.ExtraErrorPSD(**EXTRA_ERROR).sum().item(), EXTRA_ERROR['rms_nm']**2, rtol=1e-5) # on TipTorch's own grid
    np.testing.assert_allclose(extra_P3.sum(), EXTRA_ERROR['rms_nm']**2, rtol=1e-6)
    assert extra_TT[extra_TT.shape[0]//2, extra_TT.shape[1]//2] == 0 # TipTorch removes the DC that P3 keeps
    assert difference_stats(zero_DC(extra_TT), zero_DC(extra_P3))['p90_rel'] < 1.0


def test_wind_shake_PSD_matches_P3():
    P3, model = build_models()
    P3_terms, TT_terms = addon_terms(P3, model)
    wind_P3, wind_TT = P3_terms['wind shake'], TT_terms['wind shake']
    assert np.isfinite(wind_TT).all() and wind_TT.min() >= 0
    np.testing.assert_allclose(wind_TT.sum(), wind_P3.sum(), rtol=1e-2) # same temporal integration of the rejection transfer function
    assert np.abs(wind_TT - wind_P3).max() < 2e-2 * wind_P3.max()


def run_all():
    for test in (test_PSD_contributors_match_P3, test_tomographic_operators_match_P3, test_tilt_filter_matches_P3,
                 test_focus_error_PSD_matches_P3, test_extra_error_PSD_matches_P3, test_wind_shake_PSD_matches_P3):
        test()
    print('PSD comparison with P3 passed')


#%% Checks and figures
if __name__ == '__main__':
    run_all()
    P3, model = build_models()
    dk = float(model.dk)

    #%% PSD contributors
    P3_PSDs, TT_PSDs = PSD_terms(P3, model)
    for key in PSD_TERMS:
        plot_maps(TT_PSDs[key], P3_PSDs[key], f'{key} PSD', f'PSD_{key.replace(" ", "_").replace(".", "")}')
    plot_radial_profiles(P3_PSDs, TT_PSDs, dk, 'PSD contributors', 'PSD_radial_profiles', 'PSD [rad² / (1/m)²]')

    #%% P3-style add-ons implemented as TipTorch members
    P3_addons, TT_addons = addon_terms(P3, model)
    for key in P3_addons:
        plot_maps(TT_addons[key], P3_addons[key], key, f'addon_{key.replace(" ", "_")}', log=key != 'tilt filter', crop=None if key == 'tilt filter' else 3*P3.freq.resAO)
    plot_radial_profiles({k: v for k, v in P3_addons.items() if k != 'tilt filter'}, {k: v for k, v in TT_addons.items() if k != 'tilt filter'},
                         dk, 'PSD add-ons', 'addon_radial_profiles', 'PSD [nm² / pixel]')

    #%% Tomographic operators
    P3_ops, TT_ops = reconstructor_terms(P3, model)
    for key in P3_ops:
        plot_maps(TT_ops[key], P3_ops[key], key, f'operator_{key}', log=False)
    print('figures written to', OUTPUT)
