"""
Helpers for the TIPTOP to TipTorch bridge (`TIPTOP/tiptop/TipTop_integration.py`).

Jitter widths are Gaussian standard deviations in [mas] with the ellipse angle in [deg], i.e. TipTorch's (Jx, Jy, Jxy)
convention. A convolution of independent, zero-centered Gaussian jitters adds their covariance matrices.
PSF metrics are batched PyTorch versions of P3's getFWHM(method='contour'), getEncircledEnergy, getEnsquaredEnergy and
radial_profile: they take PSF stacks [..., H, W] and evaluate all PSFs in one pass.
"""

import math
import numpy as np
import torch
from torch.nn.functional import grid_sample, pad

from tiptorch.tools.utils import cov_to_jitter_params, mask_circle

SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
FWHM_TO_SIGMA = 1.0 / SIGMA_TO_FWHM
RAD_TO_MAS    = 180.0 * 3600.0 * 1000.0 / math.pi


# ------------------------------------------ Jitter algebra ------------------------------------------
def jitter_params_to_cov(Jx, Jy, Jxy_deg):
    """ Covariance [..., 2, 2] of a Gaussian with major/minor sigmas Jx, Jy and major axis angle Jxy_deg (counterclockwise from x) """
    Jx, Jy, angle = torch.broadcast_tensors(Jx, Jy, torch.deg2rad(Jxy_deg))
    c, s = torch.cos(angle), torch.sin(angle)
    xx = (Jx*c).square() + (Jy*s).square()
    yy = (Jx*s).square() + (Jy*c).square()
    xy = (Jx.square() - Jy.square()) * s * c
    return torch.stack((torch.stack((xx, xy), -1), torch.stack((xy, yy), -1)), -2)


def combine_zero_centered_jitters(Jx1, Jy1, Jxy1_deg, Jx2, Jy2, Jxy2_deg):
    """ Single Gaussian equivalent to the convolution of two zero-centered Gaussian jitters """
    return cov_to_jitter_params(jitter_params_to_cov(Jx1, Jy1, Jxy1_deg) + jitter_params_to_cov(Jx2, Jy2, Jxy2_deg))


def tiptilt_covariance_to_jitter(covariance_nm2, D):
    """ MASTSEL tip/tilt residual covariance [N, 2, 2] (Zernike OPD coefficients, nm²) to image jitter sigmas [mas] and angle [deg] """
    scale = 4.0e-9 * RAD_TO_MAS / D # a 1 nm tilt coefficient displaces the image by 4e-9/D rad
    return cov_to_jitter_params(covariance_nm2 * scale**2)


def fwhm_to_jitter(fwhm_mas, *, device, dtype, count):
    """ TIPTOP's telescope jitter_FWHM (a scalar [mas] or [FWHM_x, FWHM_y, angle_rad]) as per-source sigmas [mas] and angle [deg] """
    values = torch.as_tensor(fwhm_mas, device=device, dtype=dtype).flatten()
    if values.numel() == 1:
        Jx, Jy, angle = values[0]*FWHM_TO_SIGMA, values[0]*FWHM_TO_SIGMA, values.new_zeros(())
    elif values.numel() == 3:
        Jx, Jy, angle = values[0]*FWHM_TO_SIGMA, values[1]*FWHM_TO_SIGMA, torch.rad2deg(values[2])
    else:
        raise ValueError('jitter_FWHM must be a scalar or [FWHM_x, FWHM_y, angle_rad]')
    return tuple(v.expand(count) for v in (Jx, Jy, angle))


# ----------------------------------------- Pupil and PSD grids -----------------------------------------
def circular_pupil(resolution, obscuration_ratio, *, device, dtype):
    """ Annular pupil mask [N, N] used when a TIPTOP config has no PathPupil """
    N = int(resolution)
    pupil = mask_circle(N, N/2) - mask_circle(N, N/2 * obscuration_ratio)
    return torch.as_tensor(pupil, device=device, dtype=dtype)


def pad_PSD_to_even(PSD):
    """ Prepend a zero Nyquist row and column to an odd, DC-centered PSD [..., n, n] so that the DC stays at n//2 on MASTSEL's even grids """
    return pad(PSD, (1, 0, 1, 0)) if PSD.shape[-1] % 2 else PSD


# ----------------------------------------- Batched PSF metrics -----------------------------------------
def _peak_position(PSFs):
    """ Integer (y, x) peak coordinates [...] of a PSF stack [..., H, W] """
    ids = PSFs.flatten(-2).argmax(-1)
    return ids // PSFs.shape[-1], ids % PSFs.shape[-1]


def _radial_bins(PSFs, center):
    """ Integer radial bin [..., H, W] of every pixel about center = (y, x) [...], i.e. P3's radial_profile binning round(r) """
    H, W = PSFs.shape[-2:]
    y = torch.arange(H, device=PSFs.device, dtype=PSFs.dtype)[:, None] - center[0].to(PSFs.dtype)[..., None, None]
    x = torch.arange(W, device=PSFs.device, dtype=PSFs.dtype)[None, :] - center[1].to(PSFs.dtype)[..., None, None]
    return torch.round(torch.hypot(y, x)).long().expand(PSFs.shape)


def _bin_sums(PSFs, bins):
    """ Per-bin pixel sums and pixel counts [..., n_bins] computed with scatter_add over the flattened images """
    sums   = torch.zeros(*PSFs.shape[:-2], int(bins.max())+1, device=PSFs.device, dtype=PSFs.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_  (-1, bins.flatten(-2), PSFs.flatten(-2))
    counts.scatter_add_(-1, bins.flatten(-2), torch.ones_like(PSFs).flatten(-2))
    return sums, counts


def PSF_radial_profile(PSFs, pixel_scale_mas, center=None):
    """
    Bin radii [n_bins] in [mas] and azimuthally averaged profiles [..., n_bins] of unit-flux PSFs [..., H, W] about the peak
    (default) or about the (y, x) pixel given in `center`, as P3's radial_profile(normalize='total')
    """
    PSFs = PSFs / PSFs.sum(dim=(-2,-1), keepdim=True)
    sums, counts = _bin_sums(PSFs, _radial_bins(PSFs, _peak_position(PSFs) if center is None else center))
    radii = torch.arange(sums.shape[-1], device=PSFs.device, dtype=PSFs.dtype) * pixel_scale_mas
    return radii, sums / counts.clamp_min(1)


def PSF_radial_profile_polar(PSFs, pixel_scale_mas, step_mas, max_radius_mas, n_theta_max=180):
    """
    Radii [n_r] every step_mas up to max_radius_mas and the azimuthal means [..., n_r] of unit-flux PSFs [..., H, W] about their peaks,
    sampled by bicubic interpolation on polar rings of max(3, ceil(2 pi r / step)) points capped at n_theta_max, as P3's
    precompute_polar_grid / interpolate_2d (TIPTOP's Super_Sampling option 2): values are clipped at zero and scaled by (step / pixel_scale)²
    """
    PSFs = PSFs / PSFs.sum(dim=(-2,-1), keepdim=True)
    H, W = PSFs.shape[-2:]
    x = PSFs.reshape(-1, 1, H, W)
    B = x.shape[0]
    radii   = torch.arange(0, max_radius_mas, step_mas, device=x.device, dtype=x.dtype)          # [R]
    n_theta = torch.ceil(2*torch.pi * radii / step_mas).clamp(3, n_theta_max)                     # [R] points per ring (all coincide at r = 0)
    j       = torch.arange(n_theta_max, device=x.device, dtype=x.dtype)                           # [T]
    theta   = 2*torch.pi * j[None, :] / n_theta[:, None]                                          # [R, T]
    valid   = j[None, :] < n_theta[:, None]
    py, px  = _peak_position(x[:, 0])
    r_pix   = (radii / pixel_scale_mas)[:, None]

    xs, ys = px.view(B,1,1) + r_pix*torch.cos(theta), py.view(B,1,1) + r_pix*torch.sin(theta)    # [B, R, T] ring coordinates in pixels
    grid   = torch.stack((2*xs/(W-1) - 1, 2*ys/(H-1) - 1), -1)
    values = grid_sample(x, grid, mode='bicubic', padding_mode='zeros', align_corners=True)[:, 0]
    profile = (values * valid).sum(-1) / valid.sum(-1)
    return radii, profile.clamp_min(0).view(*PSFs.shape[:-2], -1) * (step_mas / pixel_scale_mas)**2


def resample_profile_cubic(radii, profiles, pixel_scale_mas, step_mas):
    """
    Resample profiles [..., n] sampled at the uniform radii [n] every step_mas from radii[0] to radii[-1] with a not-a-knot cubic spline,
    as P3's interpolate_1d (TIPTOP's Super_Sampling option 1): values are clipped at zero and scaled by (step / pixel_scale)²
    """
    n, h = radii.numel(), radii[1] - radii[0]
    y = profiles.reshape(-1, n).T # [n, B]
    A, rhs = torch.zeros(n, n, device=y.device, dtype=y.dtype), torch.zeros_like(y)
    i = torch.arange(1, n-1, device=y.device)
    A[i, i-1], A[i, i], A[i, i+1] = 1.0, 4.0, 1.0
    rhs[i] = 6/h**2 * (y[i+1] - 2*y[i] + y[i-1])
    A[0, :3], A[-1, -3:] = torch.tensor([1., -2., 1.], device=y.device, dtype=y.dtype), torch.tensor([1., -2., 1.], device=y.device, dtype=y.dtype) # not-a-knot ends
    M = torch.linalg.solve(A, rhs) # second derivatives at the knots [n, B]

    r = torch.arange(radii[0].item(), radii[-1].item(), step_mas, device=y.device, dtype=y.dtype)
    k = ((r - radii[0]) / h).floor().long().clamp(0, n-2)
    t = (r - radii[k])[:, None]
    b = (y[k+1] - y[k])/h - h*(2*M[k] + M[k+1])/6
    y_new = y[k] + b*t + M[k]/2*t**2 + (M[k+1] - M[k])/(6*h)*t**3
    return r, y_new.T.clamp_min(0).view(*profiles.shape[:-1], -1) * (step_mas / pixel_scale_mas)**2


def PSF_encircled_energy(PSFs, pixel_scale_mas, center=None):
    """
    Bin radii [n_bins] in [mas] and encircled energy curves [..., n_bins] normalized to their maximum (as P3's getEncircledEnergy),
    about the central pixel (H//2, W//2) by default
    """
    H, W = PSFs.shape[-2:]
    center = tuple(torch.tensor(c, device=PSFs.device) for c in (H//2, W//2)) if center is None else center
    EE = _bin_sums(PSFs, _radial_bins(PSFs, center))[0].cumsum(-1)
    radii = torch.arange(EE.shape[-1], device=PSFs.device, dtype=PSFs.dtype) * pixel_scale_mas
    return radii, EE / EE.amax(-1, keepdim=True)


def PSF_ensquared_energy(PSFs):
    """ Ensquared energy [..., n+1] in squares of half-side n = 0..min(H, W)//2 pixels centered on the peak (as P3's getEnsquaredEnergy) """
    H, W = PSFs.shape[-2:]
    S = pad(PSFs.cumsum(-1).cumsum(-2), (1, 0, 1, 0)).flatten(-2) # integral image, S[y, x] = sum of PSF[:y, :x]
    n = torch.arange(min(H, W)//2 + 1, device=PSFs.device)
    py, px = (c[..., None] for c in _peak_position(PSFs))
    y0, y1 = (py-n).clamp(0, H), (py+n+1).clamp(0, H)
    x0, x1 = (px-n).clamp(0, W), (px+n+1).clamp(0, W)
    at = lambda y, x: S.gather(-1, y*(W+1) + x)
    return (at(y1, x1) - at(y0, x1) - at(y1, x0) + at(y0, x0)) / PSFs.sum(dim=(-2,-1))[..., None]


def PSF_FWHM(PSFs, pixel_scale_mas, n_angles=64, radial_step=0.25):
    """
    Major and minor FWHM [mas] ([...], [...]) of PSFs [..., H, W] from their half-maximum contour, as P3's getFWHM(method='contour'):
    the contour is sampled along n_angles rays from the peak (bicubic interpolation every radial_step pixels), re-centered on its
    bounding box, and the FWHMs are twice its largest and smallest radii. Rays without a crossing extend to min(H, W)/2.
    Bicubic sampling matters: bilinear interpolation biases the diagonal rays of PSFs only a few pixels wide.
    """
    H, W = PSFs.shape[-2:]
    x = PSFs.reshape(-1, 1, H, W)
    B = x.shape[0]
    half   = x.flatten(-2).amax(-1).view(B, 1, 1) / 2
    py, px = _peak_position(x[:, 0])
    r_max  = min(H, W) / 2
    radii  = torch.arange(0, r_max, radial_step, device=x.device, dtype=x.dtype)                # [R]
    angles = torch.arange(n_angles, device=x.device, dtype=x.dtype) * (2*torch.pi / n_angles)  # [A]
    cos, sin = torch.cos(angles)[:, None], torch.sin(angles)[:, None]

    xs, ys = px.view(B,1,1) + radii*cos, py.view(B,1,1) + radii*sin # [B, A, R] ray coordinates in pixels
    grid   = torch.stack((2*xs/(W-1) - 1, 2*ys/(H-1) - 1), -1)
    values = grid_sample(x, grid, mode='bicubic', padding_mode='zeros', align_corners=True)[:, 0]

    below   = values <= half
    crossed = below.any(-1)
    k  = torch.where(crossed, below.int().argmax(-1), radii.numel()-1).clamp_min(1) # first sample under the half maximum
    v0 = values.gather(-1, (k-1)[..., None])[..., 0]
    v1 = values.gather(-1, k[..., None])[..., 0]
    frac = ((v0 - half[..., 0]) / (v0 - v1).clamp_min(1e-30)).clamp(0, 1)
    r  = torch.where(crossed, radii[k-1] + frac*radial_step, torch.full_like(frac, r_max)) # [B, A] contour radii about the peak

    cx, cy = r*cos[:, 0], r*sin[:, 0]
    cx = cx - (cx.amax(-1, keepdim=True) + cx.amin(-1, keepdim=True)) / 2 # re-center the contour on its bounding box
    cy = cy - (cy.amax(-1, keepdim=True) + cy.amin(-1, keepdim=True)) / 2
    rc = torch.hypot(cx, cy)
    return (2*rc.amax(-1)*pixel_scale_mas).view(PSFs.shape[:-2]), (2*rc.amin(-1)*pixel_scale_mas).view(PSFs.shape[:-2])


def interpolate_curves(radii, curves, r_query):
    """ Linear interpolation of batched curves [..., n] sampled at radii [n] at r_query (a scalar or [...]), clamped to the sampled range """
    r = torch.as_tensor(r_query, device=radii.device, dtype=radii.dtype).clamp(radii[0], radii[-1]).expand(curves.shape[:-1])
    k = torch.searchsorted(radii, r.contiguous()).clamp(1, radii.numel()-1)
    c0 = curves.gather(-1, (k-1)[..., None])[..., 0]
    c1 = curves.gather(-1, k[..., None])[..., 0]
    return c0 + (c1-c0) * (r - radii[k-1]) / (radii[k] - radii[k-1])
