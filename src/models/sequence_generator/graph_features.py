"""Functions to convert backbone structures to enhanced graph features.

Builds on ProteinBackbone.to_graph() with additional features for
sequence design: one-hot AA encoding, virtual CB positions, and
relative orientation features on edges.
"""

import numpy as np
import torch
from typing import Optional

from src.data.protein_structure import ProteinBackbone, ProteinGraph
from src.utils.protein_constants import NUM_AA, AA_1_INDEX


def _estimate_cb_position(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Estimate virtual CB position from backbone N, CA, C atoms.

    Uses the ideal tetrahedral geometry to place CB.  The CB direction
    is approximately along -(N-CA + C-CA) normalized, rotated into the
    plane perpendicular to the N-CA-C bisector.

    Args:
        n: (L, 3) N atom coordinates
        ca: (L, 3) CA atom coordinates
        c: (L, 3) C atom coordinates

    Returns:
        cb: (L, 3) estimated CB coordinates
    """
    # Vectors from CA
    v1 = n - ca  # CA -> N
    v2 = c - ca  # CA -> C

    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)

    # CB is roughly in the opposite direction of the bisector of N-CA-C
    bisector = v1_norm + v2_norm
    bisector = bisector / (np.linalg.norm(bisector, axis=-1, keepdims=True) + 1e-8)

    # Cross product for the perpendicular direction
    perp = np.cross(v1_norm, v2_norm)
    perp = perp / (np.linalg.norm(perp, axis=-1, keepdims=True) + 1e-8)

    # CB placement: ideal CB-CA bond length ~1.52 A, tetrahedral angle
    cb_direction = -bisector * np.sin(np.radians(54.75)) + perp * np.cos(np.radians(54.75))
    cb = ca + 1.521 * cb_direction

    return cb.astype(np.float32)


def _compute_orientation_features(
    coords: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Compute relative orientation features for edges.

    For each edge (i, j), computes rotation-derived features that
    capture the relative orientation of the local frames at residues
    i and j.  Uses the 9 elements of the relative rotation matrix
    (flattened), which is SE(3)-informative.

    Args:
        coords: (L, 4, 3) backbone coordinates [N, CA, C, O]
        edge_index: (2, E) edge indices

    Returns:
        orientation_features: (E, 9) relative rotation matrix elements
    """
    n_coords = coords[:, 0]
    ca_coords = coords[:, 1]
    c_coords = coords[:, 2]

    # Build local frames at each residue
    L = coords.shape[0]
    frames = np.zeros((L, 3, 3), dtype=np.float32)

    for i in range(L):
        v1 = n_coords[i] - ca_coords[i]
        v2 = c_coords[i] - ca_coords[i]

        e1 = v1 / (np.linalg.norm(v1) + 1e-8)
        u2 = v2 - np.dot(v2, e1) * e1
        e2 = u2 / (np.linalg.norm(u2) + 1e-8)
        e3 = np.cross(e1, e2)

        frames[i] = np.stack([e1, e2, e3], axis=-1)  # (3, 3)

    # Relative rotation: R_j^T @ R_i
    src = edge_index[0]
    dst = edge_index[1]

    # R_rel = R_dst^T @ R_src
    frames_src = frames[src]  # (E, 3, 3)
    frames_dst = frames[dst]  # (E, 3, 3)

    rel_rot = np.einsum('eji,ejk->eik', frames_dst, frames_src)  # (E, 3, 3)
    orientation_feats = rel_rot.reshape(-1, 9)  # (E, 9)

    return orientation_feats.astype(np.float32)


def _encode_sequence_onehot(
    sequence: Optional[str],
    length: int,
    fixed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One-hot encode amino acid sequence for known/fixed residues.

    For positions where the residue is known (fixed_mask=True) or all
    positions if no mask is provided, encode the amino acid as a one-hot
    vector.  Unknown positions get a zero vector.

    Args:
        sequence: Amino acid sequence (one-letter codes), or None
        length: Number of residues
        fixed_mask: (L,) boolean mask; True = known/fixed residue

    Returns:
        onehot: (L, 20) one-hot encoding (zeros for unknown positions)
    """
    onehot = np.zeros((length, NUM_AA), dtype=np.float32)

    if sequence is None:
        return onehot

    for i, aa in enumerate(sequence[:length]):
        if fixed_mask is not None and not fixed_mask[i]:
            continue
        idx = AA_1_INDEX.get(aa, -1)
        if idx >= 0:
            onehot[i, idx] = 1.0

    return onehot


def backbone_to_graph_features(
    backbone: ProteinBackbone,
    k: int = 30,
    fixed_mask: Optional[np.ndarray] = None,
) -> ProteinGraph:
    """Convert a ProteinBackbone to an enhanced ProteinGraph.

    Augments the base ProteinGraph from backbone.to_graph() with:
    - One-hot amino acid encoding for known/fixed residues (dim 20)
    - Virtual CB position relative to CA (dim 3)
    - Relative orientation features on edges (dim 9)

    Args:
        backbone: Input protein backbone
        k: Number of nearest neighbors for graph construction
        fixed_mask: (L,) boolean mask for fixed/known residues

    Returns:
        Enhanced ProteinGraph with richer node and edge features.
        Node features: base (23) + AA one-hot (20) + CB offset (3) = 46
        Edge features: base (8) + orientation (9) = 17
    """
    # Start from the base graph
    base_graph = backbone.to_graph(k=k)

    L = backbone.length

    # --- Enhanced node features ---
    # One-hot AA encoding
    aa_onehot = _encode_sequence_onehot(
        backbone.sequence, L, fixed_mask
    )  # (L, 20)

    # Virtual CB offset from CA
    n_coords = backbone.coords[:, 0]
    ca_coords = backbone.coords[:, 1]
    c_coords = backbone.coords[:, 2]
    cb_coords = _estimate_cb_position(n_coords, ca_coords, c_coords)
    cb_offset = cb_coords - ca_coords  # (L, 3)

    # Concatenate enhanced node features
    enhanced_node_features = torch.cat([
        base_graph.node_features,                           # (L, 23)
        torch.tensor(aa_onehot, dtype=torch.float32),       # (L, 20)
        torch.tensor(cb_offset, dtype=torch.float32),       # (L, 3)
    ], dim=-1)  # (L, 46)

    # --- Enhanced edge features ---
    orientation_feats = _compute_orientation_features(
        backbone.coords, base_graph.edge_index.numpy()
    )  # (E, 9)

    enhanced_edge_features = torch.cat([
        base_graph.edge_features,                               # (E, 8)
        torch.tensor(orientation_feats, dtype=torch.float32),   # (E, 9)
    ], dim=-1)  # (E, 17)

    return ProteinGraph(
        node_features=enhanced_node_features,
        edge_index=base_graph.edge_index,
        edge_features=enhanced_edge_features,
        coords=base_graph.coords,
        mask=base_graph.mask,
    )
