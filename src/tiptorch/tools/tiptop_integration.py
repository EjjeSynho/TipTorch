"""Numerical helpers for the TIPTOP to TipTorch bridge.

Jitter widths are Gaussian standard deviations in milliarcseconds.  A
convolution of independent, centred Gaussian jitters adds their covariance
matrices; multiplying the Gaussian images would give a different result.
"""

import math

import numpy as np
from scipy.ndimage import map_coordinates
import torch

from tiptorch.tools.utils import cov_to_jitter_params


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
RAD_TO_MAS = 180.0 * 3600.0 * 1000.0 / math.pi


def jitter_params_to_cov(jx, jy, angle_deg):
    """Convert broadcastable major/minor sigmas and angle to covariance.

    The angle is measured counterclockwise from the x axis in degrees.
    """
    jx, jy, angle_deg = torch.broadcast_tensors(jx, jy, angle_deg)
    angle = torch.deg2rad(angle_deg)
    c, s = torch.cos(angle), torch.sin(angle)
    xx = jx.square() * c.square() + jy.square() * s.square()
    yy = jx.square() * s.square() + jy.square() * c.square()
    xy = (jx.square() - jy.square()) * s * c
    return torch.stack((torch.stack((xx, xy), -1),
                        torch.stack((xy, yy), -1)), -2)


def combine_zero_centered_jitters(jx1, jy1, angle1_deg,
                                   jx2, jy2, angle2_deg):
    """Return the single Gaussian equivalent of two jitter convolutions."""
    covariance = (jitter_params_to_cov(jx1, jy1, angle1_deg)
                  + jitter_params_to_cov(jx2, jy2, angle2_deg))
    return cov_to_jitter_params(covariance)


def tiptilt_covariance_to_jitter(covariance_nm2, telescope_diameter_m):
    """Convert MASTSEL tip/tilt covariance in nm² to image jitter in mas.

    MASTSEL's residual matrix describes Zernike tip/tilt OPD coefficients.
    Their image displacement is ``4 * coefficient / diameter`` radians.
    """
    if telescope_diameter_m <= 0:
        raise ValueError("Telescope diameter must be positive")
    scale = 4.0e-9 * RAD_TO_MAS / telescope_diameter_m
    return cov_to_jitter_params(covariance_nm2 * scale**2)


def fwhm_to_jitter(fwhm_mas, *, device, dtype, count):
    """Expand TIPTOP telescope jitter to source-wise sigma/angle tensors.

    A scalar is circular FWHM in mas.  A triple is x/y FWHM in mas plus
    ellipse angle in radians, matching ``residualToSpectrum`` in TIPTOP.
    """
    values = torch.as_tensor(fwhm_mas, device=device, dtype=dtype).flatten()
    if values.numel() == 1:
        jx = jy = values[0] * FWHM_TO_SIGMA
        angle = values.new_zeros(())
    elif values.numel() == 3:
        jx, jy = values[:2] * FWHM_TO_SIGMA
        angle = torch.rad2deg(values[2])
    else:
        raise ValueError("jitter_FWHM must be a scalar or [x, y, angle_rad]")
    if torch.any(values[:2] < 0):
        raise ValueError("jitter_FWHM must be nonnegative")
    return tuple(value.expand(count) for value in (jx, jy, angle))


def psf_fwhm(image, pixel_scale_mas):
    """Estimate major/minor FWHM from the half-height contour in mas."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("PSF must be a finite two-dimensional image")
    peak = np.unravel_index(np.argmax(image), image.shape)
    height = image[peak] / 2
    max_radius = min(image.shape) / 2
    radii = np.linspace(0, max_radius, max(128, int(max_radius * 16)))
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    points = []
    for angle in angles:
        ys = peak[0] + radii * np.sin(angle)
        xs = peak[1] + radii * np.cos(angle)
        values = map_coordinates(image, [ys, xs], order=1, mode='constant')
        below = np.flatnonzero(values <= height)
        if not len(below) or below[0] == 0:
            continue
        k = below[0]
        radius = np.interp(height, values[k-1:k+1][::-1], radii[k-1:k+1][::-1])
        points.append((radius * np.cos(angle), radius * np.sin(angle)))
    if len(points) < 8:
        return (2 * max_radius * pixel_scale_mas,) * 2
    points = np.asarray(points)
    design = np.column_stack((points[:, 0]**2,
                              points[:, 0] * points[:, 1],
                              points[:, 1]**2))
    a, b, c = np.linalg.lstsq(design, np.ones(len(points)), rcond=None)[0]
    eigenvalues = np.linalg.eigvalsh([[a, b / 2], [b / 2, c]])
    if np.any(eigenvalues <= 0):
        return (2 * max_radius * pixel_scale_mas,) * 2
    widths = 2 * pixel_scale_mas / np.sqrt(eigenvalues)
    return float(widths[0]), float(widths[1])


def psf_encircled_energy(image, pixel_scale_mas):
    """Return radial cumulative flux and bin radii in mas."""
    image = np.asarray(image, dtype=float)
    y, x = np.indices(image.shape)
    cy, cx = (np.asarray(image.shape) - 1) / 2
    distance = np.hypot(y - cy, x - cx)
    bins = np.floor(distance).astype(int)
    annular = np.bincount(bins.ravel(), weights=image.ravel())
    total = annular.sum()
    if total <= 0:
        raise ValueError("PSF total flux must be positive")
    return np.cumsum(annular) / total, (np.arange(len(annular)) + 0.5) * pixel_scale_mas


def psf_ensquared_energy(image):
    """Return cumulative flux in squares centred on the PSF peak."""
    image = np.asarray(image, dtype=float)
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    max_radius = min(cy, cx, image.shape[0] - cy - 1, image.shape[1] - cx - 1)
    total = image.sum()
    return np.array([
        image[cy-r:cy+r+1, cx-r:cx+r+1].sum() / total
        for r in range(max_radius + 1)])


def circular_pupil(resolution, obscuration_ratio, *, device, dtype):
    """Build the pupil used when a TIPTOP config has no pupil FITS path."""
    if resolution < 4 or not 0 <= obscuration_ratio < 1:
        raise ValueError("Invalid pupil resolution or obscuration ratio")
    coordinates = (torch.arange(resolution, device=device, dtype=dtype)
                   - (resolution - 1) / 2) / (resolution / 2)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    radius_squared = xx.square() + yy.square()
    return ((radius_squared <= 1)
            & (radius_squared >= obscuration_ratio**2)).to(dtype)


def tiptilt_rejection_filter(model):
    """P3-equivalent tilt rejection filter on TipTorch's full PSD grid."""
    x = torch.pi * model.D * model.k
    safe_x = torch.where(x.abs() < 1e-5, torch.ones_like(x), x)
    j0 = torch.special.bessel_j0(safe_x)
    j1 = torch.special.bessel_j1(safe_x)
    j2 = 2 * j1 / safe_x - j0
    j1_term = torch.where(x.abs() < 1e-5, torch.ones_like(x), 2 * j1 / safe_x)
    j2_term = torch.where(x.abs() < 1e-5, torch.zeros_like(x), 4 * j2 / safe_x)
    half = (1 - j1_term.square() - j2_term.square()).clamp(0, 1)
    return model.half_PSD_to_full(half)


def extra_error_psd(model, rms_nm, exponent=-2.0, min_frequency=0.0,
                    max_frequency=0.0):
    """Piston-filtered TIPTOP extra-error spectrum normalized to RMS nm²."""
    frequency = model.half_PSD_to_full(model.k).real
    x = torch.pi * model.D * frequency
    safe_x = torch.where(x.abs() < 1e-5, torch.ones_like(x), x)
    aperture = torch.where(x.abs() < 1e-5, torch.ones_like(x),
                           2 * torch.special.bessel_j1(safe_x) / safe_x)
    piston = (1 - aperture.square()).clamp_min(0)
    spectrum = frequency.clamp_min(1e-9).pow(exponent) * piston
    spectrum = torch.where(frequency >= min_frequency, spectrum, 0)
    if max_frequency > 0:
        spectrum = torch.where(frequency <= max_frequency, spectrum, 0)
    spectrum = torch.where(x.abs() < 1e-5, 0, spectrum)
    power = spectrum.sum(dim=(-2, -1), keepdim=True)
    if torch.any(power <= 0):
        raise ValueError("Extra-error frequency range contains no PSD samples")
    return spectrum * (rms_nm**2 / power)


def wind_shake_psd(model, vibration_data, frame_rate, loop_gain, delay_steps):
    """Residual wind-shake PSD in nm², following TIPTOP's RTC integration."""
    data = np.asarray(vibration_data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3 or frame_rate <= 0:
        raise ValueError("Wind-shake data must contain frequency, tip and tilt PSDs")
    frequencies = np.linspace(0.1, frame_rate / 2, int(5 * frame_rate))
    tip = np.interp(frequencies, data[0], data[1], left=0, right=0)
    tilt = np.interp(frequencies, data[0], data[2], left=0, right=0)
    z = np.exp(-2j * np.pi * frequencies / frame_rate)
    integrator = loop_gain / (1 - z**-1)
    rejection = 1 / (1 + integrator * z**-delay_steps)
    power_nm2 = abs(np.sum(rejection**2 * (tip + tilt))
                    * (frequencies[1] - frequencies[0]))

    tilt_shape = (1 - tiptilt_rejection_filter(model)).clamp_min(0)
    frequency = model.half_PSD_to_full(model.k).real
    x = torch.pi * model.D * frequency
    safe_x = torch.where(x.abs() < 1e-5, torch.ones_like(x), x)
    aperture = torch.where(x.abs() < 1e-5, torch.ones_like(x),
                           2 * torch.special.bessel_j1(safe_x) / safe_x)
    piston = (1 - aperture.square()).clamp_min(0)
    shape = tilt_shape * piston * model.mask_corrected
    total = shape.sum(dim=(-2, -1), keepdim=True)
    if torch.any(total <= 0):
        raise ValueError("Wind-shake PSD has no corrected spatial-frequency samples")
    return shape * (power_nm2 / total)
