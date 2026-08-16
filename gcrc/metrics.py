"""Small tensor-only image-quality diagnostics used by the examples."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    mse = float((original - reconstruction).square().mean())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim(original: torch.Tensor, reconstruction: torch.Tensor, window: int = 7) -> float:
    """Return the same local SSIM approximation used in the paper scripts."""
    if window % 2 == 0:
        raise ValueError("window must be odd")
    padding = window // 2
    mu_x = F.avg_pool2d(original, window, 1, padding)
    mu_y = F.avg_pool2d(reconstruction, window, 1, padding)
    sigma_x = F.avg_pool2d(original * original, window, 1, padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(reconstruction * reconstruction, window, 1, padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(original * reconstruction, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return float(value.mean())
