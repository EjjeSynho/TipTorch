"""
Checks of the per-observation pupil rotation of TipTorch (see agents/execplans/pupil-rotation-execplan.md):

1. `rotate_images` reproduces torchvision's rotate (bilinear and nearest, odd and even sizes).
2. A model with PupilAngle = a (rotated pupil, static OPD, WFS / aliasing frame, elliptic Moffat) gives rotate(PSF(0), +a): the Moffat
   ellipse follows +a and the asinh-image distance to rotate(PSF(0), +a) is far below the one to rotate(PSF(0), -a).
3. A batch of observations with different angles reproduces the single-angle models; with all angles 0 the stored grids and masks are
   used as they are and the single model is reproduced exactly.
4. Gradients flow through the rotated static OPD; `Update` re-orients the pupils when the config angles change.
5. The config manager keeps one angle per merged config (and one shared angle for several sources in one observation).
6. Axes convention: dx moves the PSF along the rows, src_dirs_x elongates it along the rows.

Needs `parameter_files/muse_ltao.ini` of the TipTorch data folder; the pupil is the analytic `PupilVLT`. GPU (the default device of the
project config) ~1 min, CPU a few minutes. Run from the repository root in the `TipTorch` conda env (no pytest there):

    python tests/test_pupil_rotation.py
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiptorch._config import DATA_FOLDER, default_device
from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager, MultipleTargetsInDifferentObservations, MultipleTargetsInOneObservation
from tiptorch.tools.static_phase import ZernikeBasis
from tiptorch.tools.utils import PupilVLT, rotate_images

DEVICE = torch.device(default_device) if torch.cuda.is_available() else torch.device('cpu')
INI = DATA_FOLDER / 'parameter_files/muse_ltao.ini'


def check(condition, message):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {message}')
    return bool(condition)


# ---------------------------------------------------------------- helpers ----------------------------------------------------------------
def base_config(wind=False):
    ''' Single-observation MUSE LTAO config (list form), one wavelength, 101 px '''
    config = ConfigManager().Load(INI)
    config['sources_science']['Wavelength'] = [700e-9]
    config['sensor_science']['FieldOfView'] = 101
    if not wind:
        config['atmosphere']['WindSpeed'] = (np.asarray(config['atmosphere']['WindSpeed']) * 0.0).tolist()
    return config


def merged_config(angles, wind=False):
    ''' One observation per angle, merged for a batch (as the calibrator does) '''
    configs = []
    for angle in angles:
        config = base_config(wind)
        config['telescope']['PupilAngle'] = float(angle)
        configs.append(config)
    return MultipleTargetsInDifferentObservations(configs, device=DEVICE)


PSD_INCLUDE = {'fitting': True, 'WFS noise': True, 'spatio-temporal': True, 'aliasing': True, 'chromatism': False, 'diff. refract': False, 'Moffat': True}


def make_model(config, Moffat_ratio=3.0):
    model = TipTorch(config, AO_type='LTAO', pupil=lambda angle: PupilVLT(320, rotation_angle=angle), PSD_include=PSD_INCLUDE,
                     norm_regime='sum', device=DEVICE, oversampling=1)
    model.ratio = model.ratio * 0 + Moffat_ratio # elliptic Moffat along the pupil-frame x axis, visible in the PSF
    model.amp   = model.amp * 0 + 2.0
    model.alpha = model.alpha * 0 + 0.5
    return model


def static_phase(model, N_src=None):
    ''' Astigmatism + coma OPD of every source (the same in the pupil frame), [N_src, 1, H, W] complex field '''
    basis = ZernikeBasis(model, N_modes=8, ignore_pupil=False)
    coefs = torch.zeros(N_src or model.N_src, 8, device=DEVICE)
    coefs[:, 3], coefs[:, 5] = 120.0, 80.0 # [nm]
    return basis(coefs)


def ellipse_angle(P):
    ''' Angle of the major axis of the second moments, from the +row axis towards the +column axis [deg] '''
    P = P / P.sum()
    yy, xx = np.meshgrid(np.arange(P.shape[0]), np.arange(P.shape[1]), indexing='ij')
    cy, cx = (P*yy).sum(), (P*xx).sum()
    myy, mxx, mxy = (P*(yy-cy)**2).sum(), (P*(xx-cx)**2).sum(), (P*(yy-cy)*(xx-cx)).sum()
    return np.degrees(0.5*np.arctan2(2*mxy, myy - mxx))


def asinh_distance(P, Q, r_core=3):
    ''' Mean absolute difference of the asinh-stretched images outside of the core (where interpolation errors dominate) '''
    yy, xx = np.meshgrid(*[np.arange(P.shape[0]) - P.shape[0]//2]*2, indexing='ij')
    outside = np.hypot(yy, xx) > r_core
    stretch = lambda x: np.arcsinh(x / P.max() * 1e3)
    return np.abs(stretch(P) - stretch(Q))[outside].mean()


def rotate_np(P, angle):
    return TF.rotate(torch.as_tensor(P)[None], float(angle), interpolation=TF.InterpolationMode.BILINEAR)[0].numpy()


# ----------------------------------------------------------------- tests -----------------------------------------------------------------
def test_rotate_images_matches_torchvision():
    ok = True
    for n in (101, 100):
        yy, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) - (n-1)/2]*2, indexing='ij')
        img = torch.exp(-((xx - n/5)**2 + (yy/3)**2) / (2*(n/15)**2))
        for angle in (30.0, -72.5):
            ref = TF.rotate(img[None], angle, interpolation=TF.InterpolationMode.BILINEAR)[0]
            diff = (rotate_images(img, angle)[0] - ref).abs().max().item()
            ok &= check(diff < 1e-5, f'rotate_images bilinear n={n} angle={angle}: max |diff| vs torchvision = {diff:.1e}')

    pupil = torch.tensor(PupilVLT(320, 0)).double()
    angles = torch.tensor([30.0, -72.5], dtype=torch.float64)
    out = rotate_images(pupil, angles, mode='nearest')
    for i, angle in enumerate(angles.tolist()):
        ref = TF.rotate(pupil[None], angle, interpolation=TF.InterpolationMode.NEAREST)[0]
        ok &= check(torch.equal(out[i], ref), f'rotate_images nearest pupil angle={angle}: identical to torchvision')
    return ok


def test_rotated_model_matches_rotated_PSF():
    angle = 35.0
    models = {a: make_model(merged_config([a])) for a in (0.0, angle)}
    PSF = {a: m(phase=static_phase(m)).detach()[0, 0].cpu().numpy() for a, m in models.items()}

    ok = check(models[angle].rotate_pupil and not models[0.0].rotate_pupil, 'rotate_pupil flag follows the config angle')
    d_plus, d_minus = asinh_distance(PSF[angle], rotate_np(PSF[0.0], +angle)), asinh_distance(PSF[angle], rotate_np(PSF[0.0], -angle))
    # The residual of the right sign is the bilinear interpolation error of the spikes and halo (~0.04); the wrong sign is 3x worse
    ok &= check(d_plus < 0.5*d_minus, f'PSF({angle}) vs rotate(PSF(0), +{angle}): asinh distance {d_plus:.4f}, vs -{angle}: {d_minus:.4f}')
    ang_0, ang_a = ellipse_angle(PSF[0.0]), ellipse_angle(PSF[angle])
    ok &= check(abs(ang_0) < 3 and abs(ang_a - angle) < 3, f'Moffat ellipse angle: {ang_0:.1f} deg at 0, {ang_a:.1f} deg at {angle} (counter-clockwise)')

    # The pupil itself: the generator is called with the angle (analytic rotation)
    pupil_expected = torch.tensor(PupilVLT(320, angle), device=DEVICE, dtype=torch.float32)
    ok &= check(torch.equal(models[angle].pupil_rot, pupil_expected), 'pupil_rot is the analytic pupil at the angle')
    ok &= check(torch.equal(models[angle].pupil, models[0.0].pupil), 'model.pupil stays the reference (unrotated) mask')
    return ok


def test_batched_angles_match_single_models():
    angles = [0.0, 35.0, -70.0]
    batch = make_model(merged_config(angles, wind=True))
    phase = static_phase(batch)
    PSF_batch = batch(phase=phase).detach()

    ok = check(batch.pupil_angle.tolist() == angles and batch.pupil_rot.shape == (3, 320, 320), f'batch: pupil_angle {batch.pupil_angle.tolist()}, pupil_rot {tuple(batch.pupil_rot.shape)}')
    kx, ky = batch.kxy_AO_()
    ok &= check(kx.shape[0] == 3 and ky.shape[0] == 3 and kx.shape[1:] == batch.kx_AO.shape[1:], f'kxy_AO_ returns per-observation grids {tuple(kx.shape)}')
    ok &= check(batch.OTF_static_default.shape[0] == 3, f'OTF_static_default per observation {tuple(batch.OTF_static_default.shape)}')

    for i, angle in enumerate(angles):
        single = make_model(merged_config([angle], wind=True))
        PSF_single = single(phase=static_phase(single)).detach()
        rel = ((PSF_batch[i] - PSF_single[0]).abs().max() / PSF_single[0].abs().max()).item()
        ok &= check(rel < 1e-4, f'observation {i} (angle {angle}): batch vs single model max relative difference = {rel:.1e}')
    return ok


def test_zero_angles_use_stored_grids():
    batch = make_model(merged_config([0.0, 0.0, 0.0], wind=True))
    ok = check(not batch.rotate_pupil and batch.pupil_angle.numel() == 1, 'identical zero angles collapse to one, no rotation')
    kx, ky = batch.kxy_AO_()
    ok &= check(kx is batch.kx_AO and ky is batch.ky_AO, 'kxy_AO_ returns the stored grids themselves')
    ok &= check(batch.pupil_rot is batch.pupil and batch.OTF_static_default.shape[0] == 1, 'pupil_rot is the reference pupil, one static OTF')
    km, kn, PR = batch._aliasing_combs()
    ok &= check(km is batch.km and PR is batch.PR, '_aliasing_combs returns the stored combs')

    single = make_model(merged_config([0.0], wind=True))
    PSF_batch, PSF_single = batch(phase=static_phase(batch)).detach(), single(phase=static_phase(single)).detach()
    rel = ((PSF_batch - PSF_single).abs().max() / PSF_single.abs().max()).item()
    ok &= check(rel < 1e-5, f'3 observations at angle 0 vs the single model: max relative difference = {rel:.1e}')

    # Identical non-zero angles also share one rotated pupil and grid
    same = make_model(merged_config([20.0, 20.0], wind=True))
    ok &= check(same.rotate_pupil and same.pupil_angle.numel() == 1 and same.pupil_rot.ndim == 2, 'identical non-zero angles share one rotated pupil')
    return ok


def test_gradients_and_pupil_refresh():
    model = make_model(merged_config([0.0, 35.0, -70.0]))
    basis = ZernikeBasis(model, N_modes=8, ignore_pupil=False)
    coefs = torch.zeros(3, 8, device=DEVICE, requires_grad=True)
    PSF = model(phase=basis(coefs + torch.tensor([0, 0, 0, 100., 0, 50., 0, 0], device=DEVICE)))
    PSF.sum(dim=(-2, -1)).mul(torch.arange(1, 4, device=DEVICE).view(3, 1)).sum().backward()
    ok = check(coefs.grad is not None and torch.isfinite(coefs.grad).all(), 'finite gradients w.r.t. the LO coefficients through the rotated OPD')

    coefs2 = torch.zeros(3, 8, device=DEVICE, requires_grad=True)
    model(phase=basis(coefs2 + torch.tensor([0, 0, 0, 100., 0, 50., 0, 0], device=DEVICE)))[..., 50, 50].sum().backward()
    ok &= check((coefs2.grad[:, 3:6].abs() > 0).all(), 'non-zero peak gradients for every observation')

    # Update with new angles re-orients the pupils although pupils=False (the calibrator swaps the batch config)
    model = make_model(merged_config([0.0, 10.0]))
    config = merged_config([40.0, -25.0])
    model.Update(config=config, grids=False, pupils=False, tomography=True)
    model.ratio, model.amp, model.alpha = model.ratio * 0 + 3.0, model.amp * 0 + 2.0, model.alpha * 0 + 0.5 # Update resets the values from the config
    fresh = make_model(merged_config([40.0, -25.0]))
    ok &= check(torch.equal(model.pupil_rot, fresh.pupil_rot) and torch.allclose(model.OTF_static_default, fresh.OTF_static_default),
                'Update(pupils=False) with new angles re-orients the pupils and the static OTF')
    PSF_updated, PSF_fresh = model(phase=static_phase(model)).detach(), fresh(phase=static_phase(fresh)).detach()
    rel = ((PSF_updated - PSF_fresh).abs().max() / PSF_fresh.abs().max()).item()
    ok &= check(rel < 1e-5, f'updated model vs fresh model: max relative difference = {rel:.1e}')
    return ok


def test_config_manager_angles():
    merged = merged_config([5.0, -15.0, 25.0])
    angles = merged['telescope']['PupilAngle']
    ok = check(torch.is_tensor(angles) and angles.flatten().tolist() == [5.0, -15.0, 25.0], f"merged configs keep every PupilAngle: {angles.flatten().tolist()}")

    config = base_config()
    config['telescope']['PupilAngle'] = 12.0
    one_obs = MultipleTargetsInOneObservation(config, 4, device=DEVICE)
    angle = one_obs['telescope']['PupilAngle']
    ok &= check(torch.as_tensor(angle).numel() == 1 and float(torch.as_tensor(angle).flatten()[0]) == 12.0, f'several sources in one observation share one angle: {angle}')
    model = make_model(one_obs)
    ok &= check(model.N_src == 4 and model.pupil_angle.numel() == 1 and model.pupil_rot.ndim == 2, 'TipTorch: 4 sources, one rotated pupil')
    return ok


def test_axes_convention():
    model = make_model(merged_config([0.0], wind=True), Moffat_ratio=1.0)
    moments = lambda P: (lambda P, yy, xx: ((P*yy).sum().item(), (P*xx).sum().item(), (P*(yy - (P*yy).sum())**2).sum().item(), (P*(xx - (P*xx).sum())**2).sum().item()))(
        P / P.sum(), *torch.meshgrid(torch.arange(P.shape[0], dtype=P.dtype), torch.arange(P.shape[1], dtype=P.dtype), indexing='ij'))
    PSF_0 = model().detach()[0, 0].cpu()
    model.dx = model.dx * 0 + 3.0
    PSF_dx = model().detach()[0, 0].cpu()
    model.dx = model.dx * 0
    row0, col0, _, _ = moments(PSF_0)
    row1, col1, _, _ = moments(PSF_dx)
    ok = check(abs(row1 - row0 - 3) < 0.3 and abs(col1 - col0) < 0.1, f'dx = +3 px moves the PSF along the rows: ({row0:.2f}, {col0:.2f}) -> ({row1:.2f}, {col1:.2f})')

    model.src_dirs_x = model.src_dirs_x * 0 + np.tan(15 / 206265)
    model.src_dirs_y = model.src_dirs_y * 0
    _, _, var_rows, var_cols = moments(model().detach()[0, 0].cpu())
    ok &= check(var_rows > 1.02 * var_cols, f'src_dirs_x = 15" elongates the PSF along the rows: variance rows {var_rows:.1f}, columns {var_cols:.1f}')
    return ok


def test_single_angle_timing():
    ''' Forward time of the unrotated model (the default path) and of a single rotated angle, for the record '''
    times = {}
    for angle in (0.0, 30.0):
        model = make_model(merged_config([angle], wind=True))
        phase = static_phase(model)
        for _ in range(3):
            model(phase=phase)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            model(phase=phase)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        times[angle] = (time.time() - t0) / 10
    return check(times[30.0] < 1.5 * times[0.0] + 5e-3, f'forward time: angle 0 {times[0.0]*1e3:.1f} ms, angle 30 {times[30.0]*1e3:.1f} ms')


if __name__ == '__main__':
    print(f'device: {DEVICE}')
    results = {name: fn() for name, fn in list(globals().items()) if name.startswith('test_') and callable(fn)}
    failed = [name for name, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)} / {len(results)} tests passed" + (f", failed: {failed}" if failed else ''))
    sys.exit(1 if failed else 0)
