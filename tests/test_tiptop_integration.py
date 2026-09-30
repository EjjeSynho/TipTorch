"""Small CPU checks for the TIPTOP Gaussian and tilt conversions."""

from types import SimpleNamespace

import numpy as np
from scipy.special import jv
import torch

from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.tools.tiptop_integration import (
    combine_zero_centered_jitters, fwhm_to_jitter,
    extra_error_psd, jitter_params_to_cov, psf_fwhm,
    tiptilt_rejection_filter, wind_shake_psd,
)


def test_two_rotated_jitters_add_covariance():
    a = tuple(torch.tensor(v, dtype=torch.float64) for v in ([4., 3.], [2., 1.], [20., -35.]))
    b = tuple(torch.tensor(v, dtype=torch.float64) for v in ([3., 2.], [1., 0.5], [-35., 70.]))
    combined = combine_zero_centered_jitters(*a, *b)
    expected = jitter_params_to_cov(*a) + jitter_params_to_cov(*b)
    torch.testing.assert_close(jitter_params_to_cov(*combined), expected, rtol=1e-12, atol=1e-12)


def test_tiptorch_jitter_kernel_respects_signed_rotation():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 9, dtype=torch.float64), torch.linspace(-1, 1, 9, dtype=torch.float64), indexing='ij')
    context = SimpleNamespace(U=xx, V=yy, u_max=torch.tensor(0.05, dtype=torch.float64), jitter_norm_fact=torch.tensor((2 * np.pi)**2, dtype=torch.float64))
    sigma = jitter_params_to_cov(torch.tensor(3.), torch.tensor(1.), torch.tensor(-35.)).double()
    expected = torch.exp(-0.5 * (2 * np.pi)**2 * context.u_max * (sigma[0, 0] * xx.square() + 2 * sigma[0, 1] * xx * yy + sigma[1, 1] * yy.square()))
    actual = TipTorch.JitterKernel(context, torch.tensor(3.), torch.tensor(1.), torch.tensor(-35.))
    torch.testing.assert_close(actual.squeeze(), expected, rtol=1e-6, atol=1e-6)


def test_telescope_jitter_units_and_psf_width():
    major, minor, angle = fwhm_to_jitter([10., 5., np.pi / 4], device='cpu', dtype=torch.float64, count=2)
    np.testing.assert_allclose([major[0], minor[0]], np.array([10., 5.]) / 2.354820045, rtol=1e-8)
    np.testing.assert_allclose(angle.numpy(), [45., 45.])
    y, x = np.mgrid[-32:33, -32:33]
    image = np.exp(-0.5 * ((x / 3)**2 + (y / 2)**2))
    widths = psf_fwhm(image, 1.)
    np.testing.assert_allclose(widths, [2.35482 * 3, 2.35482 * 2], rtol=0.06)


def test_tilt_filter_rejects_origin():
    half = torch.tensor([[[0., 0.1, 0.2],
                          [0., 0.1, 0.2],
                          [0., 0.1, 0.2]]], dtype=torch.float64)
    
    context = SimpleNamespace(D=torch.tensor(8.), k=half, half_PSD_to_full=lambda value: value)
    result = tiptilt_rejection_filter(context)
    assert torch.all(result[..., 0] < 1e-9)
    assert torch.all((result >= 0) & (result <= 1))
    x = np.pi * 8 * 0.2
    expected = 1 - (2 * jv(1, x) / x)**2 - (4 * jv(2, x) / x)**2
    np.testing.assert_allclose(result[0, 0, 2], expected, rtol=1e-6)


def test_extra_error_has_requested_rms():
    coordinates = torch.linspace(0, 0.5, 9, dtype=torch.float64)
    context = SimpleNamespace(D=torch.tensor(8.), k=coordinates[None, None, :], half_PSD_to_full=lambda value: value)
    spectrum = extra_error_psd(context, 60., -2., min_frequency=0.1)
    np.testing.assert_allclose(spectrum.sum(), 60.**2, rtol=1e-10)
    assert torch.all(spectrum[..., :2] == 0)


def test_wind_shake_psd_is_finite_and_rejected_by_loop():
    yy, xx = torch.meshgrid(torch.linspace(0, 0.2, 5), torch.linspace(0, 0.2, 5), indexing='ij')
    context = SimpleNamespace(D=torch.tensor(8.), k=torch.hypot(xx, yy), mask_corrected=torch.ones_like(xx), half_PSD_to_full=lambda value: value)
    frequencies = np.linspace(0.1, 100, 200)
    source = np.vstack((frequencies, 1 / (1 + frequencies**2), 1 / (1 + frequencies**2)))
    low_gain  = wind_shake_psd(context, source, 200, 0.1, 2)
    high_gain = wind_shake_psd(context, source, 200, 0.5, 2)
    assert torch.isfinite(low_gain).all()
    assert torch.isfinite(high_gain).all()
    assert low_gain.sum() > high_gain.sum() > 0


if __name__ == '__main__':
    test_two_rotated_jitters_add_covariance()
    test_tiptorch_jitter_kernel_respects_signed_rotation()
    test_telescope_jitter_units_and_psf_width()
    test_tilt_filter_rejects_origin()
    test_extra_error_has_requested_rms()
    test_wind_shake_psd_is_finite_and_rejected_by_loop()
    print('TIPTOP integration numerical checks passed')
