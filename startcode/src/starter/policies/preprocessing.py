"""Shared observation transforms at the offline, online and serving boundaries."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def canonicalize_quaternions(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).copy()
    if values.shape[-1] != 16 or not np.isfinite(values).all():
        raise ValueError("Expected finite 16-D end-effector vectors")
    for start in (3, 11):
        q = values[..., start:start + 4]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        q = q / np.maximum(norm, 1e-8)
        q = np.where(norm > 1e-8, q, np.array([1, 0, 0, 0], np.float32))
        # Break ties at qw=0 as well: q and -q must map to the same vector.
        leading = np.take_along_axis(q, (np.abs(q) > 1e-8).argmax(-1)[..., None], axis=-1)
        values[..., start:start + 4] = q * np.where(leading < 0, -1, 1)
    return values


def canonicalize_state(state: torch.Tensor) -> torch.Tensor:
    values = state.clone()
    if values.shape[-1] != 16 or not torch.isfinite(values).all():
        raise ValueError("Expected finite 16-D end-effector state")
    for start in (3, 11):
        q = values[..., start:start + 4]
        norm = q.norm(dim=-1, keepdim=True)
        identity = q.new_tensor([1, 0, 0, 0])
        q = torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)
        index = (q.abs() > 1e-8).long().argmax(dim=-1, keepdim=True)
        leading = q.gather(-1, index)
        values[..., start:start + 4] = q * torch.where(leading < 0, -1, 1)
    return values


def prepare_image_batch(images: torch.Tensor, size: int) -> torch.Tensor:
    """Raw BCHW RGB to normalized square images, also used during rollouts."""
    was_uint8 = images.dtype == torch.uint8
    images = images.float()
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError("Expected finite BCHW RGB images")
    if was_uint8 or images.max() > 1:
        images = images / 255.0
    if images.shape[-2:] != (size, size):
        images = F.interpolate(images, size=(size, size), mode="area")
    return images.sub(0.5).div(0.5)


def prepare_image(image, size: int = 96) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(image).copy()) if not isinstance(image, torch.Tensor) else image.detach()
    if tensor.ndim != 3:
        raise ValueError("Expected an HWC or CHW RGB image")
    if tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    return prepare_image_batch(tensor.unsqueeze(0), size)[0]
