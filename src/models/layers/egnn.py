"""E(n)-Equivariant Graph Neural Network (EGNN).

Implements the EGNN layer from Satorras et al. (2021). Updates both node features
and positions equivariantly using only pairwise distances (no frames needed).

Simpler than IPA but only sees pairwise distances, not orientations.
"""

import torch
import torch.nn as nn
from typing import Optional


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant message passing layer.

    Updates node features h and positions x:
        m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
        x_i' = x_i + C * Σ_j (x_i - x_j) * φ_x(m_ij)
        m_i  = Σ_j m_ij
        h_i' = φ_h(h_i, m_i)

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge attributes (0 if none)
        hidden_dim: Hidden dimension for MLPs
        update_coords: Whether to update coordinates (set False for last layer if only features needed)
        coord_scale: Scale factor for coordinate updates (prevents large jumps)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        update_coords: bool = True,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.update_coords = update_coords
        self.coord_scale = coord_scale

        # Edge message MLP: (h_i, h_j, ||x_i-x_j||^2, edge_attr) → message
        edge_input_dim = 2 * node_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP: (h_i, aggregated_messages) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Coordinate update MLP: message → scalar weight for displacement
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        # Layer norm for stability
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass.

        Args:
            h: (N, node_dim) node features
            x: (N, 3) node positions
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) optional edge attributes
            node_mask: (N,) optional boolean mask

        Returns:
            h_out: (N, node_dim) updated node features
            x_out: (N, 3) updated positions
        """
        src, dst = edge_index[0], edge_index[1]

        # Compute pairwise displacements and squared distances
        rel_pos = x[src] - x[dst]  # (E, 3)
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build edge input
        edge_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Edge messages
        messages = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Aggregate messages (sum)
        agg = torch.zeros(h.shape[0], messages.shape[-1], device=h.device)
        agg.index_add_(0, dst, messages)

        # Update node features
        h_input = torch.cat([h, agg], dim=-1)
        h_out = h + self.node_mlp(h_input)  # residual
        h_out = self.node_norm(h_out)

        # Update coordinates
        if self.update_coords:
            coord_weights = self.coord_mlp(messages)  # (E, 1)
            weighted_displacements = rel_pos * coord_weights  # (E, 3)

            coord_update = torch.zeros_like(x)
            coord_update.index_add_(0, dst, weighted_displacements)

            x_out = x + self.coord_scale * coord_update
        else:
            x_out = x

        # Apply mask
        if node_mask is not None:
            mask = node_mask.unsqueeze(-1).float()
            h_out = h_out * mask
            x_out = x * (1 - mask) + x_out * mask

        return h_out, x_out


class EGNNStack(nn.Module):
    """Stack of EGNN layers.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge attributes
        hidden_dim: Hidden dimension for MLPs
        n_layers: Number of EGNN layers
        coord_scale: Scale factor for coordinate updates
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 4,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                update_coords=True,
                coord_scale=coord_scale,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass through all layers.

        Returns:
            h: (N, node_dim) final node features
            x: (N, 3) final positions
        """
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, node_mask)
        return h, x
