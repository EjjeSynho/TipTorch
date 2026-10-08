"""
CPU checks for the TIPTOP bridge helpers in `tiptorch.tools.tiptop_integration`: jitter covariance algebra and TipTorch's jitter
kernel rotation, nm² to mas conversion, the batched PSF metrics and the pupil / PSD-grid helpers. No AO configuration or data
cache is needed. The P3-style PSD members of `TipTorch` (TiltFilter, ExtraErrorPSD, FocusErrorPSD, WindShakePSD) are checked
against P3 itself in `tests/PSD_comparison_with_P3.py`. Run with `python tests/test_tiptop_integration.py` (the TipTop env has no pytest).
"""

from types import SimpleNamespace

import numpy as np
import torch

from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.tools.tiptop_integration import (
    SIGMA_TO_FWHM, circular_pupil, combine_zero_centered_jitters, fwhm_to_jitter, interpolate_curves, jitter_params_to_cov,
    pad_PSD_to_even, PSF_encircled_energy, PSF_ensquared_energy, PSF_FWHM, PSF_radial_profile, tiptilt_covariance_to_jitter,
)

torch.set_default_dtype(torch.float64)


def _gaussian_stack(sigmas, size=65, shape=(2, 3)):
    ''' PSFs [*shape, size, size]: elliptical Gaussians with (sigma_x, sigma_y) pixel widths, rotated by a few degrees '''
    y, x = torch.meshgrid(*[torch.arange(size) - size//2]*2, indexing='ij')
    sx, sy = (torch.as_tensor(s).view(*shape, 1, 1) for s in sigmas)
    return torch.exp(-0.5 * ((x/sx)**2 + (y/sy)**2))


def test_two_rotated_jitters_add_covariance():
    a = tuple(torch.tensor(v) for v in ([4., 3.], [2., 1.], [20., -35.]))
    b = tuple(torch.tensor(v) for v in ([3., 2.], [1., 0.5], [-35., 70.]))
    combined = combine_zero_centered_jitters(*a, *b)
    torch.testing.assert_close(jitter_params_to_cov(*combined), jitter_params_to_cov(*a) + jitter_params_to_cov(*b), rtol=1e-12, atol=1e-12)


def test_tiptorch_jitter_kernel_respects_signed_rotation():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 9), torch.linspace(-1, 1, 9), indexing='ij')
    context = SimpleNamespace(U=xx, V=yy, u_max=torch.tensor(0.05), jitter_norm_fact=torch.tensor((2*np.pi)**2))
    sigma = jitter_params_to_cov(torch.tensor(3.), torch.tensor(1.), torch.tensor(-35.))
    expected = torch.exp(-0.5 * (2*np.pi)**2 * context.u_max * (sigma[0, 0]*xx.square() + 2*sigma[0, 1]*xx*yy + sigma[1, 1]*yy.square()))
    actual = TipTorch.JitterKernel(context, torch.tensor(3.), torch.tensor(1.), torch.tensor(-35.))
    torch.testing.assert_close(actual.squeeze(), expected, rtol=1e-6, atol=1e-6)


def test_covariance_to_jitter_units():
    ''' A pure tilt of 10 nm RMS on an 8 m telescope moves the image by 4e-9*10/8 rad = 1.031 mas '''
    cov = torch.diag(torch.tensor([100.0, 1e-6]))[None]
    Jx, Jy, _ = tiptilt_covariance_to_jitter(cov, 8.0)
    np.testing.assert_allclose(Jx.item(), 4e-9*10/8 * 180*3600*1000/np.pi, rtol=1e-9)
    assert Jy.item() < 1e-2 * Jx.item()


def test_telescope_jitter_conversion():
    major, minor, angle = fwhm_to_jitter([10., 5., np.pi/4], device='cpu', dtype=torch.float64, count=2)
    np.testing.assert_allclose([major[0], minor[0]], np.array([10., 5.]) / SIGMA_TO_FWHM, rtol=1e-8)
    np.testing.assert_allclose(angle.numpy(), [45., 45.])


def test_batched_fwhm_of_elliptical_gaussians():
    sx = torch.tensor([[3., 4., 5.], [2.5, 6., 3.5]])
    sy = torch.tensor([[2., 4., 3.], [2.5, 2., 7.0]])
    fx, fy = PSF_FWHM(_gaussian_stack((sx, sy)), 2.0)
    assert fx.shape == fy.shape == (2, 3)
    np.testing.assert_allclose(fx.numpy(), 2.0 * SIGMA_TO_FWHM * torch.maximum(sx, sy).numpy(), rtol=0.03)
    np.testing.assert_allclose(fy.numpy(), 2.0 * SIGMA_TO_FWHM * torch.minimum(sx, sy).numpy(), rtol=0.03)


def test_batched_energy_metrics():
    PSFs = _gaussian_stack((torch.full((2, 3), 4.), torch.full((2, 3), 4.)), size=129)
    radii, EE = PSF_encircled_energy(PSFs, 1.0)
    assert EE.shape == (2, 3, radii.numel()) and torch.all(EE[..., -1] == 1)
    r = 6.0 # bin r collects the pixels with round(distance) == r, i.e. up to a continuous radius of r + 0.5 (as P3's radial_profile)
    np.testing.assert_allclose(interpolate_curves(radii, EE, r).numpy(), 1 - np.exp(-(r + 0.5)**2 / (2*4.**2)), rtol=0.03)

    square = PSF_ensquared_energy(PSFs)
    assert square.shape[:2] == (2, 3) and torch.all(square[..., 0] < square[..., 1]) and torch.allclose(square[..., -1], torch.ones(2, 3))

    radii, profile = PSF_radial_profile(PSFs, 1.0)
    expected = torch.exp(-0.5 * (radii/4.)**2)
    np.testing.assert_allclose(profile[0, 0, :20].numpy() / profile[0, 0, 0].item(), expected[:20].numpy(), rtol=0.1) # annulus averages of a curved profile


def test_pupil_and_psd_padding():
    pupil = circular_pupil(64, 0.25, device='cpu', dtype=torch.float64)
    np.testing.assert_allclose(pupil.sum().item(), np.pi * 32**2 * (1 - 0.25**2), rtol=0.02)
    PSD = torch.zeros(2, 1, 11, 11)
    PSD[..., 5, 5] = 1
    padded = pad_PSD_to_even(PSD)
    assert padded.shape == (2, 1, 12, 12) and padded[0, 0, 6, 6] == 1 and padded.sum() == 2


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
    print('TIPTOP integration checks passed')
