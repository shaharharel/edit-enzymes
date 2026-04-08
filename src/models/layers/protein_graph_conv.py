"""Message passing layers on protein graphs.

Edge-conditioned graph convolution for processing protein backbone graphs.
Used by the sequence generator (ProteinMPNN-style).
"""

import torch
import torch.nn as nn
from typing import Optional


class ProteinGraphConv(nn.Module):
    """Edge-conditioned message passing on protein graphs.

    Messages are computed as: m_ij = MLP([h_i, h_j, e_ij])
    Aggregation: agg_i = sum(m_ij for j in N(i))
    Update: h_i' = h_i + MLP([h_i, agg_i])

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension for MLPs
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E) source and destination indices
            edge_features: (E, edge_dim)
            node_mask: (N,) optional boolean mask

        Returns:
            updated_features: (N, node_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        N = node_features.shape[0]

        # Compute messages
        msg_input = torch.cat([
            node_features[src],
            node_features[dst],
            edge_features,
        ], dim=-1)
        messages = self.message_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate
        agg = torch.zeros(N, messages.shape[-1], device=node_features.device)
        agg.index_add_(0, dst, messages)

        # Update
        update_input = torch.cat([node_features, agg], dim=-1)
        updated = node_features + self.update_mlp(update_input)  # residual
        updated = self.layer_norm(updated)

        if node_mask is not None:
            updated = updated * node_mask.unsqueeze(-1).float()

        return updated


class ProteinGraphConvStack(nn.Module):
    """Stack of ProteinGraphConv layers.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ProteinGraphConv(node_dim, edge_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = node_features
        for layer in self.layers:
            h = layer(h, edge_index, edge_features, node_mask)
        return h
