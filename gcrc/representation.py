"""Patch/token representation utilities without dataset or checkpoint paths."""

from __future__ import annotations

import math

import torch


def patchify(images: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """Convert ``[B,C,H,W]`` images into ``[B,N,C*p*p]`` patch vectors."""
    if images.ndim != 4:
        raise ValueError("images must have shape [B,C,H,W]")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.view(batch, channels, grid_h, patch_size, grid_w, patch_size)
    return patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(
        batch, grid_h * grid_w, channels * patch_size * patch_size
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """Convert patch vectors back to ``[B,C,H,W]`` images in ``[0,1]``."""
    if patches.ndim != 3:
        raise ValueError("patches must have shape [B,N,D]")
    batch, tokens, dim = patches.shape
    channels = 3
    if patch_size is None:
        patch_size = int(round(math.sqrt(dim / channels)))
        if channels * patch_size * patch_size != dim:
            raise ValueError(f"cannot infer patch size from patch dimension {dim}")
    if dim % (channels * patch_size * patch_size) != 0:
        raise ValueError("patch dimension is incompatible with patch_size")
    if height is None or width is None:
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:
            raise ValueError("height and width are required for a non-square token grid")
        height, width = side * patch_size, side * patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h * grid_w != tokens:
        raise ValueError("height/width do not match the number of tokens")
    x = patches.view(batch, grid_h, grid_w, channels, patch_size, patch_size)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(batch, channels, height, width).clamp(0, 1)


def fit_codebook(
    patches: torch.Tensor,
    vocab_size: int,
    iterations: int = 20,
    sample_size: int = 100_000,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fit a small k-means codebook from patch vectors.

    This helper trains a new codebook from caller-provided tensors. It does
    not read or write a dataset, checkpoint, or result directory.
    """
    if patches.ndim == 3:
        samples = patches.reshape(-1, patches.shape[-1])
    elif patches.ndim == 2:
        samples = patches
    else:
        raise ValueError("patches must have shape [B,N,D] or [M,D]")
    if vocab_size < 2 or len(samples) < vocab_size:
        raise ValueError("vocab_size must be at least 2 and no larger than the sample count")
    target_device = torch.device(device) if device is not None else samples.device
    samples = samples.detach().to(target_device, dtype=torch.float32)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    if len(samples) > sample_size:
        indices = torch.randperm(len(samples), generator=generator, device=target_device)[:sample_size]
        samples = samples[indices]
    centers = samples[torch.randperm(len(samples), generator=generator, device=target_device)[:vocab_size]].clone()
    for _ in range(iterations):
        distances = (
            samples.square().sum(dim=1, keepdim=True)
            - 2.0 * samples @ centers.t()
            + centers.square().sum(dim=1).unsqueeze(0)
        )
        labels = distances.argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, samples)
        counts = torch.bincount(labels, minlength=vocab_size).to(centers.dtype).unsqueeze(1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return centers.detach()


def quantize(
    images: torch.Tensor,
    codebook: torch.Tensor,
    patch_size: int = 4,
    batch_size: int = 512,
) -> torch.Tensor:
    """Quantize images into nearest codebook indices."""
    patches = patchify(images, patch_size=patch_size)
    codebook = codebook.to(device=patches.device, dtype=patches.dtype)
    outputs = []
    for start in range(0, len(patches), batch_size):
        chunk = patches[start : start + batch_size]
        distances = (
            chunk.square().sum(dim=-1, keepdim=True)
            - 2.0 * chunk @ codebook.t()
            + codebook.square().sum(dim=-1).view(1, 1, -1)
        )
        outputs.append(distances.argmin(dim=-1))
    return torch.cat(outputs, dim=0)
