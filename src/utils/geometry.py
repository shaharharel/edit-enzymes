"""Rigid body transforms, frame operations, and RMSD computation."""

import torch
import numpy as np
from typing import Tuple


def rigid_from_3_points(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a local coordinate frame from three points (e.g., N, CA, C).

    The frame is defined as:
    - Origin at p2 (CA)
    - X-axis along p2 → p1 (CA → N)
    - Y-axis in the plane of p1-p2-p3, perpendicular to X
    - Z-axis = X cross Y

    Args:
        p1, p2, p3: Points of shape (..., 3). Typically N, CA, C.

    Returns:
        rotation: (..., 3, 3) rotation matrix
        translation: (..., 3) translation vector (= p2)
    """
    v1 = p1 - p2  # CA → N
    v2 = p3 - p2  # CA → C

    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)

    rotation = torch.stack([e1, e2, e3], dim=-1)  # (..., 3, 3)
    translation = p2

    return rotation, translation


def apply_rigid(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Apply rigid body transform: R @ x + t.

    Args:
        rotation: (..., 3, 3)
        translation: (..., 3)
        points: (..., 3) or (..., N, 3)

    Returns:
        Transformed points, same shape as input.
    """
    if points.dim() == rotation.dim() - 1:
        # points is (..., 3), rotation is (..., 3, 3)
        return torch.einsum('...ij,...j->...i', rotation, points) + translation
    else:
        # points is (..., N, 3), rotation is (..., 3, 3)
        return torch.einsum('...ij,...nj->...ni', rotation, points) + translation.unsqueeze(-2)


def invert_rigid(
    rotation: torch.Tensor, translation: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a rigid body transform: (R^T, -R^T @ t).

    Args:
        rotation: (..., 3, 3)
        translation: (..., 3)

    Returns:
        inv_rotation: (..., 3, 3)
        inv_translation: (..., 3)
    """
    inv_rotation = rotation.transpose(-1, -2)
    inv_translation = -torch.einsum('...ij,...j->...i', inv_rotation, translation)
    return inv_rotation, inv_translation


def compose_rigid(
    r1: torch.Tensor, t1: torch.Tensor,
    r2: torch.Tensor, t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two rigid transforms: (R1 @ R2, R1 @ t2 + t1).

    Args:
        r1, t1: First transform (..., 3, 3), (..., 3)
        r2, t2: Second transform (..., 3, 3), (..., 3)

    Returns:
        Composed rotation (..., 3, 3) and translation (..., 3).
    """
    rotation = torch.matmul(r1, r2)
    translation = torch.einsum('...ij,...j->...i', r1, t2) + t1
    return rotation, translation


def kabsch_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RMSD after optimal superposition (Kabsch algorithm).

    Args:
        coords1: (N, 3) reference coordinates
        coords2: (N, 3) mobile coordinates
        mask: (N,) optional boolean mask for valid atoms

    Returns:
        rmsd: scalar RMSD value
        rotation: (3, 3) optimal rotation
        translation: (3,) optimal translation (apply to coords2)
    """
    if mask is not None:
        c1 = coords1[mask]
        c2 = coords2[mask]
    else:
        c1, c2 = coords1, coords2

    # Center
    centroid1 = c1.mean(dim=0)
    centroid2 = c2.mean(dim=0)
    c1_centered = c1 - centroid1
    c2_centered = c2 - centroid2

    # Cross-covariance matrix
    H = c2_centered.T @ c1_centered

    # SVD
    U, S, Vh = torch.linalg.svd(H)

    # Correct for reflection
    d = torch.det(Vh.T @ U.T)
    sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=coords1.device))

    rotation = Vh.T @ sign_matrix @ U.T
    translation = centroid1 - rotation @ centroid2

    # Compute RMSD
    aligned = (rotation @ c2.T).T + translation
    if mask is not None:
        diff = aligned - coords1[mask]
    else:
        diff = aligned - coords1
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())

    return rmsd, rotation, translation


def backbone_frames(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract per-residue frames from backbone coordinates.

    Args:
        coords: (L, 4, 3) backbone coordinates [N, CA, C, O]

    Returns:
        rotations: (L, 3, 3) per-residue rotation matrices
        translations: (L, 3) per-residue translations (CA positions)
    """
    n_coords = coords[:, 0]   # N
    ca_coords = coords[:, 1]  # CA
    c_coords = coords[:, 2]   # C
    return rigid_from_3_points(n_coords, ca_coords, c_coords)


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance matrix.

    Args:
        coords: (N, 3) coordinates

    Returns:
        dist: (N, N) pairwise distances
    """
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    return torch.norm(diff, dim=-1)


def bond_length_loss(coords: torch.Tensor) -> torch.Tensor:
    """Compute loss for deviation from ideal backbone bond lengths.

    Args:
        coords: (L, 4, 3) backbone coordinates [N, CA, C, O]

    Returns:
        Scalar loss (mean squared deviation from ideal bond lengths).
    """
    from src.utils.protein_constants import BOND_LENGTHS

    loss = torch.tensor(0.0, device=coords.device)
    n_terms = 0

    L = coords.shape[0]

    # N-CA bonds within residues
    d_n_ca = torch.norm(coords[:, 1] - coords[:, 0], dim=-1)
    loss = loss + ((d_n_ca - BOND_LENGTHS[('N', 'CA')]) ** 2).sum()
    n_terms += L

    # CA-C bonds within residues
    d_ca_c = torch.norm(coords[:, 2] - coords[:, 1], dim=-1)
    loss = loss + ((d_ca_c - BOND_LENGTHS[('CA', 'C')]) ** 2).sum()
    n_terms += L

    # C-O bonds within residues
    d_c_o = torch.norm(coords[:, 3] - coords[:, 2], dim=-1)
    loss = loss + ((d_c_o - BOND_LENGTHS[('C', 'O')]) ** 2).sum()
    n_terms += L

    # C-N peptide bonds between consecutive residues
    if L > 1:
        d_c_n = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1)
        loss = loss + ((d_c_n - BOND_LENGTHS[('C', 'N')]) ** 2).sum()
        n_terms += L - 1

    return loss / max(n_terms, 1)
