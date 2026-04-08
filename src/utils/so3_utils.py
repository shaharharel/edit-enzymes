"""SO(3) diffusion utilities for backbone generation.

Implements the IGSO3 distribution and score functions for diffusion
over rotations, plus standard R3 diffusion for translations.
"""

import torch
import numpy as np
import math
from typing import Tuple


def random_rotation_matrix(batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
    """Sample uniform random rotation matrices.

    Uses QR decomposition of random Gaussian matrices.

    Args:
        batch_size: Number of rotations to sample
        device: Device for output tensor

    Returns:
        (batch_size, 3, 3) rotation matrices
    """
    z = torch.randn(batch_size, 3, 3, device=device)
    q, r = torch.linalg.qr(z)
    # Ensure proper rotations (det=+1)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-2)
    # Fix determinant
    det = torch.det(q)
    q[det < 0] = -q[det < 0]
    return q


def axis_angle_to_rotation(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: (..., 3) axis-angle vectors (direction = axis, norm = angle)

    Returns:
        (..., 3, 3) rotation matrices
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / (angle + 1e-8)  # (..., 3)

    cos_a = torch.cos(angle).unsqueeze(-1)  # (..., 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (..., 1, 1)

    # Skew-symmetric matrix
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    I = torch.eye(3, device=axis_angle.device).expand_as(K)

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = I + sin_a * K + (1 - cos_a) * (K @ K)

    # Handle zero angle (identity)
    zero_mask = (angle.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    R = torch.where(zero_mask, I, R)

    return R


def rotation_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle.

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        (..., 3) axis-angle vectors
    """
    # Angle from trace: cos(θ) = (tr(R) - 1) / 2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)  # (...)

    # Axis from skew-symmetric part: [R - R^T] / (2 sin θ)
    skew = (R - R.transpose(-1, -2)) / (2 * torch.sin(angle).unsqueeze(-1).unsqueeze(-1) + 1e-8)
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)

    return axis * angle.unsqueeze(-1)


# =============================================================================
# R3 Diffusion (for translations / coordinates)
# =============================================================================

def r3_forward_diffusion(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: 'DiffusionSchedule' = None,
    sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion in R3: q(x_t | x_0) = N(α_t * x_0, σ_t^2 * I).

    For a simple variance-exploding schedule: x_t = x_0 + σ_t * ε

    Args:
        x0: (..., 3) clean coordinates
        t: (...,) or scalar diffusion time in [0, 1]
        schedule: Optional noise schedule (uses linear if None)
        sigma: Maximum noise level

    Returns:
        xt: (..., 3) noised coordinates
        noise: (..., 3) the noise that was added
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, device=x0.device, dtype=x0.dtype)

    # Expand t to match x0 shape
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)

    sigma_t = t * sigma
    noise = torch.randn_like(x0)
    xt = x0 + sigma_t * noise

    return xt, noise


def r3_score(
    xt: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Score function for R3 diffusion: ∇_xt log q(xt | x0).

    For Gaussian: score = -(xt - x0) / σ_t^2

    Args:
        xt: Noised coordinates
        x0: Clean coordinates
        t: Diffusion time
        sigma: Maximum noise level

    Returns:
        Score (same shape as xt)
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, device=xt.device, dtype=xt.dtype)
    while t.dim() < xt.dim():
        t = t.unsqueeze(-1)

    sigma_t = t * sigma
    return -(xt - x0) / (sigma_t ** 2 + 1e-8)


# =============================================================================
# SO(3) Diffusion (for rotations)
# =============================================================================

def so3_sample_tangent(shape: tuple, device: str = 'cpu') -> torch.Tensor:
    """Sample from isotropic Gaussian on the tangent space of SO(3).

    Args:
        shape: Batch shape, will append (3,) for axis-angle
        device: Device

    Returns:
        (*shape, 3) tangent vectors (axis-angle)
    """
    return torch.randn(*shape, 3, device=device)


def so3_expmap(tangent: torch.Tensor) -> torch.Tensor:
    """Exponential map: tangent space → SO(3).

    Args:
        tangent: (..., 3) tangent vectors (axis-angle)

    Returns:
        (..., 3, 3) rotation matrices
    """
    return axis_angle_to_rotation(tangent)


def so3_logmap(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: SO(3) → tangent space.

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        (..., 3) tangent vectors (axis-angle)
    """
    return rotation_to_axis_angle(R)


def so3_forward_diffusion(
    R0: torch.Tensor,
    t: torch.Tensor,
    sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion on SO(3).

    Applies random perturbation in tangent space: R_t = R_0 @ exp(σ_t * ε)

    Args:
        R0: (..., 3, 3) clean rotations
        t: Scalar or (...,) diffusion time in [0, 1]
        sigma: Maximum noise level (in radians)

    Returns:
        Rt: (..., 3, 3) noised rotations
        tangent_noise: (..., 3) the tangent noise applied
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, device=R0.device, dtype=R0.dtype)

    batch_shape = R0.shape[:-2]
    sigma_t = t * sigma
    while sigma_t.dim() < len(batch_shape):
        sigma_t = sigma_t.unsqueeze(-1)

    tangent_noise = torch.randn(*batch_shape, 3, device=R0.device)
    scaled_noise = sigma_t.unsqueeze(-1) * tangent_noise

    delta_R = so3_expmap(scaled_noise)
    Rt = R0 @ delta_R

    return Rt, tangent_noise
