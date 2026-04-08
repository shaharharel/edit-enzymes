"""Invariant Point Attention (IPA) from AlphaFold2 / FrameDiff.

Unlike EGNN which only sees pairwise distances, IPA operates on local
reference frames and can reason about orientations and local geometry.
Each residue has a rigid body frame (rotation + translation) built from
N-CA-C atoms, and attention computes queries/keys/values as 3D points
in these local frames.

Three attention components computed simultaneously:
1. Standard scalar QKV attention
2. Pair feature bias (pairwise residue relationships)
3. 3D point attention (query/key/value projected as points in local frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention layer.

    Args:
        node_dim: Dimension of single representation (per-residue features)
        pair_dim: Dimension of pair representation (pairwise features)
        n_heads: Number of attention heads
        n_query_points: Number of 3D query points per head
        n_value_points: Number of 3D value points per head
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_dim: int = 256,
        pair_dim: int = 64,
        n_heads: int = 8,
        n_query_points: int = 4,
        n_value_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points

        head_dim = node_dim // n_heads

        # Scalar QKV projections
        self.q_scalar = nn.Linear(node_dim, n_heads * head_dim, bias=False)
        self.k_scalar = nn.Linear(node_dim, n_heads * head_dim, bias=False)
        self.v_scalar = nn.Linear(node_dim, n_heads * head_dim, bias=False)

        # Point QKV projections (3D coordinates per point per head)
        self.q_points = nn.Linear(node_dim, n_heads * n_query_points * 3, bias=False)
        self.k_points = nn.Linear(node_dim, n_heads * n_query_points * 3, bias=False)
        self.v_points = nn.Linear(node_dim, n_heads * n_value_points * 3, bias=False)

        # Pair bias
        self.pair_bias = nn.Linear(pair_dim, n_heads, bias=False)

        # Learnable weight for point attention (one per head)
        self.head_weights = nn.Parameter(torch.zeros(n_heads))

        # Output projection
        # Output = concat(scalar_values, pair_values, point_values_norm)
        output_dim = n_heads * (head_dim + pair_dim + n_value_points * (3 + 1))
        self.output_proj = nn.Linear(output_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)

        # Scale factors
        self._scalar_scale = 1.0 / math.sqrt(head_dim)
        self._point_scale = 1.0 / math.sqrt(3 * n_query_points)

    def forward(
        self,
        node_features: torch.Tensor,
        pair_features: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: (B, L, node_dim) per-residue features
            pair_features: (B, L, L, pair_dim) pairwise features
            rotations: (B, L, 3, 3) per-residue rotation matrices
            translations: (B, L, 3) per-residue translations (CA positions)
            mask: (B, L) boolean mask for valid residues

        Returns:
            updated_features: (B, L, node_dim)
        """
        B, L, _ = node_features.shape
        H = self.n_heads
        head_dim = self.node_dim // H

        residual = node_features

        # 1. Scalar attention
        q_s = self.q_scalar(node_features).view(B, L, H, head_dim)
        k_s = self.k_scalar(node_features).view(B, L, H, head_dim)
        v_s = self.v_scalar(node_features).view(B, L, H, head_dim)

        # Scalar attention logits: (B, H, L, L)
        attn_scalar = torch.einsum('bihd,bjhd->bhij', q_s, k_s) * self._scalar_scale

        # 2. Point attention
        q_pts = self.q_points(node_features).view(B, L, H, self.n_query_points, 3)
        k_pts = self.k_points(node_features).view(B, L, H, self.n_query_points, 3)
        v_pts = self.v_points(node_features).view(B, L, H, self.n_value_points, 3)

        # Transform points to global frame: R @ p + t
        q_pts_global = torch.einsum(
            'blij,blhpj->blhpi', rotations, q_pts
        ) + translations[:, :, None, None, :]
        k_pts_global = torch.einsum(
            'blij,blhpj->blhpi', rotations, k_pts
        ) + translations[:, :, None, None, :]
        v_pts_global = torch.einsum(
            'blij,blhpj->blhpi', rotations, v_pts
        ) + translations[:, :, None, None, :]

        # Point attention logits: sum of squared distances over query points
        # (B, H, L_q, L_k)
        q_pts_expanded = q_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, Pq, 3)
        k_pts_expanded = k_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, Pq, 3)

        pt_diff = q_pts_expanded[:, :, :, None, :, :] - k_pts_expanded[:, :, None, :, :, :]
        # (B, H, L, L, Pq, 3)
        pt_dist_sq = (pt_diff ** 2).sum(dim=(-1, -2))  # (B, H, L, L)

        # Learnable per-head weight
        w = F.softplus(self.head_weights)  # (H,)
        attn_points = -0.5 * w[None, :, None, None] * pt_dist_sq * self._point_scale

        # 3. Pair bias
        attn_pair = self.pair_bias(pair_features).permute(0, 3, 1, 2)  # (B, H, L, L)

        # Combine attention logits
        attn_logits = attn_scalar + attn_points + attn_pair

        # Mask
        if mask is not None:
            mask_2d = mask[:, None, None, :].float()  # (B, 1, 1, L)
            attn_logits = attn_logits - (1 - mask_2d) * 1e9

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate scalar values
        out_scalar = torch.einsum('bhij,bjhd->bihd', attn_weights, v_s)
        out_scalar = out_scalar.reshape(B, L, H * head_dim)

        # Aggregate point values (in global frame, then transform back to local)
        v_pts_perm = v_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, Pv, 3)
        out_pts_global = torch.einsum(
            'bhij,bhjpd->bhipd', attn_weights, v_pts_perm
        )  # (B, H, L, Pv, 3)
        out_pts_global = out_pts_global.permute(0, 2, 1, 3, 4)  # (B, L, H, Pv, 3)

        # Transform back to local frame: R^T @ (p - t)
        out_pts_local = out_pts_global - translations[:, :, None, None, :]
        rot_T = rotations.transpose(-1, -2)
        out_pts_local = torch.einsum(
            'blij,blhpj->blhpi', rot_T, out_pts_local
        )  # (B, L, H, Pv, 3)

        # Point norms (invariant)
        out_pts_norm = torch.norm(out_pts_local, dim=-1)  # (B, L, H, Pv)

        out_pts_flat = out_pts_local.reshape(B, L, H * self.n_value_points * 3)
        out_pts_norm_flat = out_pts_norm.reshape(B, L, H * self.n_value_points)

        # Aggregate pair features
        pair_expanded = pair_features.unsqueeze(1).expand(-1, H, -1, -1, -1)  # (B, H, L, L, pair_dim)
        out_pair = torch.einsum(
            'bhij,bhijd->bhid',
            attn_weights,
            pair_expanded,
        )  # (B, H, L, pair_dim)
        out_pair = out_pair.permute(0, 2, 1, 3).reshape(B, L, H * self.pair_dim)

        # Concatenate and project
        output = torch.cat([out_scalar, out_pair, out_pts_flat, out_pts_norm_flat], dim=-1)
        output = self.output_proj(output)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class IPAStack(nn.Module):
    """Stack of IPA layers with optional transition blocks.

    Args:
        node_dim: Dimension of single representation
        pair_dim: Dimension of pair representation
        n_heads: Number of attention heads
        n_query_points: Number of 3D query points per head
        n_value_points: Number of 3D value points per head
        n_layers: Number of IPA layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_dim: int = 256,
        pair_dim: int = 64,
        n_heads: int = 8,
        n_query_points: int = 4,
        n_value_points: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(
                node_dim=node_dim,
                pair_dim=pair_dim,
                n_heads=n_heads,
                n_query_points=n_query_points,
                n_value_points=n_value_points,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Transition blocks (feedforward) between IPA layers
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(node_dim),
                nn.Linear(node_dim, node_dim * 4),
                nn.GELU(),
                nn.Linear(node_dim * 4, node_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        pair_features: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all IPA layers.

        Args:
            node_features: (B, L, node_dim)
            pair_features: (B, L, L, pair_dim)
            rotations: (B, L, 3, 3)
            translations: (B, L, 3)
            mask: (B, L) boolean mask

        Returns:
            Updated node features (B, L, node_dim)
        """
        h = node_features
        for ipa, transition in zip(self.ipa_layers, self.transitions):
            h = ipa(h, pair_features, rotations, translations, mask)
            h = h + transition(h)
        return h
