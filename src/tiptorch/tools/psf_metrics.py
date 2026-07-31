"""Differentiable image-quality metrics for point-spread functions.

All functions in this module operate on the final two dimensions of a tensor,
so inputs can be individual PSFs ``(H, W)`` or stacks such as
``(batch, wavelength, H, W)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn


def _validate_psf(psf: torch.Tensor, name: str) -> None:
    if not isinstance(psf, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not psf.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if psf.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions (..., H, W).")
    if psf.shape[-2] == 0 or psf.shape[-1] == 0:
        raise ValueError(f"{name} must have non-empty spatial dimensions.")


def psf_peak(psf: torch.Tensor) -> torch.Tensor:
    """Return the peak value of every PSF.

    ``torch.amax`` is differentiable (with a valid subgradient at ties), so this
    computes the exact sampled peak rather than a detached or NumPy value.
    """

    _validate_psf(psf, "psf")
    return psf.amax(dim=(-2, -1))


def psf_fwhm(
    psf: torch.Tensor,
    *,
    temperature: float = 0.02,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a differentiable equivalent-circular FWHM in pixels.

    The hard set of pixels above half maximum is replaced by a sigmoid:

    ``sigmoid((PSF / peak - 0.5) / temperature)``.

    Its area is converted to the diameter of a circle with the same area. This
    is a smooth approximation to the ordinary half-maximum width, remains
    meaningful for mildly asymmetric PSFs, and avoids a non-differentiable
    profile fit. Smaller temperatures approach the hard half-maximum contour.
    """

    _validate_psf(psf, "psf")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    peak = psf_peak(psf)
    positive_peak = peak.clamp_min(0.0)
    normalized = psf / positive_peak.clamp_min(eps)[..., None, None]
    half_max_support = torch.sigmoid((normalized - 0.5) / temperature)
    area = half_max_support.sum(dim=(-2, -1))

    # Make the all-zero-PSF limit exactly zero without evaluating sqrt at zero.
    validity = positive_peak / (positive_peak + eps)
    equivalent_diameter = 2.0 * torch.sqrt(
        area.clamp_min(eps) / psf.new_tensor(torch.pi)
    )
    return equivalent_diameter * validity


def psf_encircled_energy(
    psf: torch.Tensor,
    radius: float | Sequence[float] | torch.Tensor,
    *,
    center: Optional[tuple[float, float]] = None,
    edge_softness: float = 0.25,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return encircled energy inside one or more radii.

    Parameters
    ----------
    psf
        Tensor ending in ``(H, W)``.
    radius
        Aperture radius or radii in pixels. Multiple radii add a final output
        dimension.
    center
        Aperture center as ``(y, x)``. The geometric image center is used by
        default.
    edge_softness
        Width in pixels of the sigmoid aperture edge. It must be positive.
    normalize
        If true, return the fraction of total PSF flux; otherwise return the
        enclosed flux.

    Notes
    -----
    The circular aperture is ``sigmoid((radius - distance) / edge_softness)``.
    Consequently EE is differentiable with respect to every PSF pixel (and also
    with respect to tensor-valued radii).
    """

    _validate_psf(psf, "psf")
    if edge_softness <= 0:
        raise ValueError("edge_softness must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    radii = torch.as_tensor(radius, device=psf.device, dtype=psf.dtype)
    scalar_radius = radii.ndim == 0
    radii = radii.reshape(-1)
    if radii.numel() == 0:
        raise ValueError("radius must contain at least one value.")
    if bool((radii < 0).any()):
        raise ValueError("radius values must be non-negative.")

    height, width = psf.shape[-2:]
    if center is None:
        center_y = psf.new_tensor((height - 1) / 2.0)
        center_x = psf.new_tensor((width - 1) / 2.0)
    else:
        if len(center) != 2:
            raise ValueError("center must be a (y, x) pair.")
        center_y = torch.as_tensor(center[0], device=psf.device, dtype=psf.dtype)
        center_x = torch.as_tensor(center[1], device=psf.device, dtype=psf.dtype)

    y, x = torch.meshgrid(
        torch.arange(height, device=psf.device, dtype=psf.dtype),
        torch.arange(width, device=psf.device, dtype=psf.dtype),
        indexing="ij",
    )
    distance = torch.sqrt((y - center_y).square() + (x - center_x).square())
    aperture = torch.sigmoid(
        (radii[:, None, None] - distance[None, ...]) / edge_softness
    )

    enclosed = torch.einsum("...hw,rhw->...r", psf, aperture)
    if normalize:
        total = psf.sum(dim=(-2, -1)).unsqueeze(-1)
        # Preserve the denominator sign while keeping it away from zero.
        safe_total = torch.where(
            total.abs() < eps,
            torch.where(total < 0, -eps, eps),
            total,
        )
        enclosed = enclosed / safe_total

    return enclosed.squeeze(-1) if scalar_radius else enclosed


def _metric_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    relative: bool,
    eps: float,
) -> torch.Tensor:
    difference = (prediction - target).abs()
    if relative:
        difference = difference / target.abs().clamp_min(eps)
    return difference


def psf_mismatch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ee_radius: Optional[float | Sequence[float] | torch.Tensor] = None,
    fwhm_temperature: float = 0.02,
    ee_edge_softness: float = 0.25,
    relative: bool = True,
    reduction: str = "none",
    peak_weight: float = 1.0,
    fwhm_weight: float = 1.0,
    ee_weight: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute peak, FWHM, and encircled-energy mismatch between PSFs.

    ``target`` is the reference for relative differences. For a sequence of EE
    radii, the EE mismatch is averaged over the radius dimension. The returned
    dictionary contains the three components and their weighted sum, ``total``.
    With ``reduction="none"``, each value has the input's leading dimensions.
    """

    _validate_psf(prediction, "prediction")
    _validate_psf(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device.")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    # Follow normal PyTorch arithmetic semantics for mixed floating-point
    # inputs. These casts stay in the autograd graph, so a gradient computed in
    # the promoted dtype is propagated back to the original prediction tensor.
    common_dtype = torch.promote_types(prediction.dtype, target.dtype)
    prediction = prediction.to(dtype=common_dtype)
    target = target.to(dtype=common_dtype)

    if ee_radius is None:
        ee_radius = (min(prediction.shape[-2:]) - 1) / 4.0

    peak = _metric_difference(
        psf_peak(prediction), psf_peak(target), relative=relative, eps=eps
    )
    fwhm = _metric_difference(
        psf_fwhm(prediction, temperature=fwhm_temperature, eps=eps),
        psf_fwhm(target, temperature=fwhm_temperature, eps=eps),
        relative=relative,
        eps=eps,
    )
    prediction_ee = psf_encircled_energy(
        prediction,
        ee_radius,
        edge_softness=ee_edge_softness,
        eps=eps,
    )
    target_ee = psf_encircled_energy(
        target,
        ee_radius,
        edge_softness=ee_edge_softness,
        eps=eps,
    )
    ee = _metric_difference(
        prediction_ee, target_ee, relative=relative, eps=eps
    )
    if ee.ndim == prediction.ndim - 1:
        # A sequence of radii appends one dimension to the per-PSF metric.
        ee = ee.mean(dim=-1)

    components = {"peak": peak, "fwhm": fwhm, "ee": ee}
    components["total"] = (
        peak_weight * peak + fwhm_weight * fwhm + ee_weight * ee
    )

    if reduction != "none":
        reduce = torch.mean if reduction == "mean" else torch.sum
        components = {name: reduce(value) for name, value in components.items()}
    return components


class PSFMismatchLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`psf_mismatch`.

    By default, ``forward`` returns a scalar mean of the weighted peak, FWHM,
    and EE mismatches. Set ``return_components=True`` to receive the full metric
    dictionary instead.
    """

    def __init__(
        self,
        *,
        ee_radius: Optional[float | Sequence[float]] = None,
        fwhm_temperature: float = 0.02,
        ee_edge_softness: float = 0.25,
        relative: bool = True,
        reduction: str = "mean",
        peak_weight: float = 1.0,
        fwhm_weight: float = 1.0,
        ee_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.ee_radius = ee_radius
        self.fwhm_temperature = fwhm_temperature
        self.ee_edge_softness = ee_edge_softness
        self.relative = relative
        self.reduction = reduction
        self.peak_weight = peak_weight
        self.fwhm_weight = fwhm_weight
        self.ee_weight = ee_weight
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        components = psf_mismatch(
            prediction,
            target,
            ee_radius=self.ee_radius,
            fwhm_temperature=self.fwhm_temperature,
            ee_edge_softness=self.ee_edge_softness,
            relative=self.relative,
            reduction=self.reduction,
            peak_weight=self.peak_weight,
            fwhm_weight=self.fwhm_weight,
            ee_weight=self.ee_weight,
            eps=self.eps,
        )
        return components if return_components else components["total"]
