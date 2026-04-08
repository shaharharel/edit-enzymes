"""Core protein structure representations.

ProteinBackbone and ProteinGraph are the interface between the backbone
generator (produces backbones) and the sequence generator (consumes graphs).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ProteinGraph:
    """Graph representation of a protein backbone for message passing.

    Attributes:
        node_features: (L, d_node) per-residue features
        edge_index: (2, E) edge indices
        edge_features: (E, d_edge) per-edge features (distances, orientations)
        coords: (L, 3) CA coordinates
        mask: (L,) boolean mask for valid residues
    """
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    coords: torch.Tensor
    mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(self, device: str) -> 'ProteinGraph':
        return ProteinGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            coords=self.coords.to(device),
            mask=self.mask.to(device),
        )


@dataclass
class ProteinBackbone:
    """Protein backbone representation.

    Attributes:
        coords: (L, 4, 3) backbone atom coordinates [N, CA, C, O]
        sequence: Optional amino acid sequence (one-letter codes)
        residue_mask: (L,) boolean mask for real vs padding residues
        chain_id: Optional chain identifier
        pdb_id: Optional PDB identifier
    """
    coords: np.ndarray  # (L, 4, 3)
    sequence: Optional[str] = None
    residue_mask: Optional[np.ndarray] = None
    chain_id: Optional[str] = None
    pdb_id: Optional[str] = None

    def __post_init__(self):
        assert self.coords.ndim == 3 and self.coords.shape[1] == 4 and self.coords.shape[2] == 3, \
            f"Expected coords shape (L, 4, 3), got {self.coords.shape}"
        if self.residue_mask is None:
            self.residue_mask = np.ones(self.coords.shape[0], dtype=bool)

    @property
    def length(self) -> int:
        return self.coords.shape[0]

    @property
    def ca_coords(self) -> np.ndarray:
        """CA coordinates (L, 3)."""
        return self.coords[:, 1]

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert coordinates to torch tensor (L, 4, 3)."""
        return torch.tensor(self.coords, dtype=torch.float32, device=device)

    def to_graph(self, k: int = 30) -> ProteinGraph:
        """Build k-NN graph on CA atoms.

        Args:
            k: Number of nearest neighbors per node.

        Returns:
            ProteinGraph with node/edge features suitable for message passing.
        """
        ca = self.ca_coords  # (L, 3)
        L = self.length
        k = min(k, L - 1)

        # Pairwise distances between CA atoms
        diff = ca[:, None] - ca[None, :]  # (L, L, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (L, L)

        # k-NN edges
        src_list, dst_list = [], []
        for i in range(L):
            neighbors = np.argsort(dist[i])[1:k + 1]  # exclude self
            for j in neighbors:
                src_list.append(i)
                dst_list.append(j)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)  # (2, E)

        # Node features: positional encoding + local geometry
        node_features = self._compute_node_features()  # (L, d_node)

        # Edge features: relative position, distance, sequential separation
        edge_features = self._compute_edge_features(edge_index)  # (E, d_edge)

        mask = torch.tensor(self.residue_mask, dtype=torch.bool)
        ca_tensor = torch.tensor(ca, dtype=torch.float32)

        return ProteinGraph(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_features=torch.tensor(edge_features, dtype=torch.float32),
            coords=ca_tensor,
            mask=mask,
        )

    def _compute_node_features(self) -> np.ndarray:
        """Compute per-residue node features.

        Features:
        - Sinusoidal positional encoding (dim 16)
        - Local backbone geometry (bond lengths, angles) (dim 6)
        - Relative position in chain (dim 1)
        Total: 23
        """
        L = self.length

        # Sinusoidal positional encoding
        pos_enc = _sinusoidal_encoding(np.arange(L), dim=16)  # (L, 16)

        # Local geometry features
        local_geom = np.zeros((L, 6), dtype=np.float32)

        # N-CA distance
        local_geom[:, 0] = np.linalg.norm(
            self.coords[:, 1] - self.coords[:, 0], axis=-1
        )
        # CA-C distance
        local_geom[:, 1] = np.linalg.norm(
            self.coords[:, 2] - self.coords[:, 1], axis=-1
        )
        # C-O distance
        local_geom[:, 2] = np.linalg.norm(
            self.coords[:, 3] - self.coords[:, 2], axis=-1
        )

        # N-CA-C angle
        for i in range(L):
            v1 = self.coords[i, 0] - self.coords[i, 1]
            v2 = self.coords[i, 2] - self.coords[i, 1]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            local_geom[i, 3] = np.arccos(np.clip(cos_angle, -1, 1))

        # CA-C-N angle (peptide bond angle, for residues with a next)
        if L > 1:
            for i in range(L - 1):
                v1 = self.coords[i, 1] - self.coords[i, 2]
                v2 = self.coords[i + 1, 0] - self.coords[i, 2]
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                )
                local_geom[i, 4] = np.arccos(np.clip(cos_angle, -1, 1))

        # Relative position [0, 1]
        rel_pos = np.arange(L, dtype=np.float32) / max(L - 1, 1)

        return np.concatenate(
            [pos_enc, local_geom, rel_pos[:, None]], axis=-1
        )  # (L, 23)

    def _compute_edge_features(self, edge_index: np.ndarray) -> np.ndarray:
        """Compute per-edge features.

        Features:
        - Relative CA position (3D, in local frame of source) (dim 3)
        - CA-CA distance (dim 1)
        - Sequential separation (normalized, dim 1)
        - Unit direction vector (dim 3)
        Total: 8
        """
        src = edge_index[0]
        dst = edge_index[1]
        E = len(src)

        ca = self.ca_coords

        # Relative position
        rel_pos = ca[dst] - ca[src]  # (E, 3)

        # Distance
        dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)  # (E, 1)

        # Unit direction
        unit_dir = rel_pos / (dist + 1e-8)  # (E, 3)

        # Sequential separation (normalized by sequence length)
        seq_sep = np.abs(dst - src).astype(np.float32) / max(self.length, 1)
        seq_sep = seq_sep[:, None]  # (E, 1)

        return np.concatenate(
            [rel_pos, dist, seq_sep, unit_dir], axis=-1
        ).astype(np.float32)  # (E, 8)

    def to_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to per-residue rigid body frames.

        Returns:
            rotations: (L, 3, 3) rotation matrices
            translations: (L, 3) translation vectors (CA positions)
        """
        from src.utils.geometry import rigid_from_3_points
        coords_t = torch.tensor(self.coords, dtype=torch.float32)
        rot, trans = rigid_from_3_points(
            coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]
        )
        return rot.numpy(), trans.numpy()


def _sinusoidal_encoding(positions: np.ndarray, dim: int = 16) -> np.ndarray:
    """Sinusoidal positional encoding."""
    pe = np.zeros((len(positions), dim), dtype=np.float32)
    for i in range(0, dim, 2):
        freq = 1.0 / (10000 ** (i / dim))
        pe[:, i] = np.sin(positions * freq)
        if i + 1 < dim:
            pe[:, i + 1] = np.cos(positions * freq)
    return pe
