"""Template-conditioned SE(3)-equivariant backbone diffusion model.

This is the core backbone generator. It takes a template backbone and produces
small deviations from it, conditioned on catalytic geometry constraints.

Key design choices:
- Template conditioning: Start reverse diffusion from noised template (not pure noise)
- Noise level controls deviation magnitude from template
- Supports both EGNN and IPA as the equivariant backbone
- Catalytic constraints injected as special node features + loss terms
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.data.protein_structure import ProteinBackbone, ProteinGraph
from src.data.catalytic_constraints import ActiveSiteSpec
from src.models.backbone_generator.base import AbstractBackboneGenerator
from src.models.backbone_generator.noise_schedule import DiffusionSchedule
from src.models.layers.egnn import EGNNStack
from src.utils.geometry import backbone_frames, bond_length_loss
from src.utils.so3_utils import r3_forward_diffusion
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for the backbone diffusion model."""
    # Architecture
    equivariant_backbone: Literal['egnn', 'ipa'] = 'egnn'
    node_dim: int = 256
    edge_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 6
    dropout: float = 0.1

    # IPA-specific
    n_heads: int = 8
    n_query_points: int = 4
    n_value_points: int = 4
    pair_dim: int = 64

    # Diffusion
    schedule_type: Literal['linear', 'cosine', 'polynomial'] = 'cosine'
    T: int = 1000
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    constraint_loss_weight: float = 10.0
    bond_loss_weight: float = 1.0

    # Template conditioning
    template_noise_scale: float = 0.1  # default noise for template conditioning


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep.

        Args:
            t: (B,) timestep values in [0, 1]

        Returns:
            (B, dim) timestep embedding
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class ConstraintEncoder(nn.Module):
    """Encode catalytic constraint information into node features.

    Catalytic residues get special features indicating their role,
    required geometry, and fixed status.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        # Role embedding (6 roles + non-catalytic)
        self.role_embedding = nn.Embedding(7, output_dim // 4)
        # Constraint feature projection
        self.constraint_proj = nn.Sequential(
            nn.Linear(output_dim // 4 + 3 + 1, output_dim),  # role + position + is_catalytic
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        n_residues: int,
        constraint_mask: torch.Tensor,
        constraint_positions: Optional[torch.Tensor] = None,
        constraint_roles: Optional[torch.Tensor] = None,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Encode constraints as per-residue features.

        Args:
            n_residues: Total number of residues
            constraint_mask: (L,) boolean mask for catalytic residues
            constraint_positions: (N_cat, 3) target positions for catalytic residues
            constraint_roles: (N_cat,) role indices for catalytic residues
            device: Device

        Returns:
            (L, output_dim) constraint features
        """
        B = 1  # handled per-sample
        output_dim = self.constraint_proj[-1].out_features

        # Default: all non-catalytic
        is_catalytic = constraint_mask.float().unsqueeze(-1)  # (L, 1)
        role_ids = torch.zeros(n_residues, dtype=torch.long, device=device)  # 0 = non-catalytic

        if constraint_roles is not None:
            cat_indices = constraint_mask.nonzero(as_tuple=True)[0]
            for i, idx in enumerate(cat_indices):
                if i < len(constraint_roles):
                    role_ids[idx] = constraint_roles[i] + 1  # shift by 1 (0 = non-catalytic)

        role_emb = self.role_embedding(role_ids)  # (L, output_dim//4)

        # Target positions (zeros for non-catalytic)
        target_pos = torch.zeros(n_residues, 3, device=device)
        if constraint_positions is not None:
            cat_indices = constraint_mask.nonzero(as_tuple=True)[0]
            for i, idx in enumerate(cat_indices):
                if i < len(constraint_positions):
                    target_pos[idx] = constraint_positions[i]

        features = torch.cat([role_emb, target_pos, is_catalytic], dim=-1)
        return self.constraint_proj(features)


class SE3BackboneDiffusion(AbstractBackboneGenerator):
    """Template-conditioned SE(3)-equivariant backbone diffusion model.

    Generates protein backbones by denoising from a noised template,
    conditioned on catalytic geometry constraints.

    Supports both EGNN and IPA as the equivariant backbone network.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Noise schedule
        self.schedule = DiffusionSchedule(
            schedule_type=config.schedule_type,
            T=config.T,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )

        # Timestep embedding
        self.time_embed = TimestepEmbedding(config.node_dim)

        # Constraint encoder
        self.constraint_encoder = ConstraintEncoder(config.node_dim)

        # Input projection (node features + timestep + constraint → hidden)
        # Node features from ProteinGraph: dim 23 (see protein_structure.py)
        self.input_proj = nn.Linear(23 + config.node_dim + config.node_dim, config.node_dim)

        # Edge feature projection
        # Edge features from ProteinGraph: dim 8 (see protein_structure.py)
        self.edge_proj = nn.Linear(8, config.edge_dim)

        # Equivariant backbone
        if config.equivariant_backbone == 'egnn':
            self.backbone = EGNNStack(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                coord_scale=0.1,
            )
        elif config.equivariant_backbone == 'ipa':
            from src.models.layers.invariant_point_attention import IPAStack
            self.backbone = IPAStack(
                node_dim=config.node_dim,
                pair_dim=config.pair_dim,
                n_heads=config.n_heads,
                n_query_points=config.n_query_points,
                n_value_points=config.n_value_points,
                n_layers=config.n_layers,
                dropout=config.dropout,
            )
            # Pair feature computation for IPA
            self.pair_proj = nn.Sequential(
                nn.Linear(config.edge_dim + 1, config.pair_dim),
                nn.GELU(),
                nn.Linear(config.pair_dim, config.pair_dim),
            )
        else:
            raise ValueError(f"Unknown backbone: {config.equivariant_backbone}")

        # Output: predict coordinate displacement from noisy to clean
        self.output_proj = nn.Sequential(
            nn.Linear(config.node_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 4 * 3),  # 4 backbone atoms * 3 coords
        )

    def _prepare_inputs(
        self,
        noisy_coords: torch.Tensor,
        timestep: torch.Tensor,
        graph: ProteinGraph,
        constraint_mask: Optional[torch.Tensor] = None,
        constraint_positions: Optional[torch.Tensor] = None,
        constraint_roles: Optional[torch.Tensor] = None,
    ):
        """Prepare model inputs from raw data."""
        L = noisy_coords.shape[0]
        device = noisy_coords.device

        # Timestep embedding (broadcast to all residues)
        t_emb = self.time_embed(timestep)  # (1, node_dim) or (node_dim,)
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        t_emb = t_emb.expand(L, -1)  # (L, node_dim)

        # Constraint features
        if constraint_mask is None:
            constraint_mask = torch.zeros(L, dtype=torch.bool, device=device)
        c_emb = self.constraint_encoder(
            L, constraint_mask, constraint_positions, constraint_roles, device
        )  # (L, node_dim)

        # Combine node features
        node_feat = torch.cat([graph.node_features, t_emb, c_emb], dim=-1)
        node_feat = self.input_proj(node_feat)  # (L, node_dim)

        # Edge features
        edge_feat = self.edge_proj(graph.edge_features)  # (E, edge_dim)

        return node_feat, edge_feat

    def denoise_step(
        self,
        noisy_coords: torch.Tensor,
        timestep: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoising step. Predicts displacement from noisy to clean.

        Returns predicted clean coordinates (x0 parameterization).
        """
        L = noisy_coords.shape[0]

        # Use CA positions for the equivariant network
        ca_noisy = noisy_coords[:, 1]  # (L, 3)

        if self.config.equivariant_backbone == 'egnn':
            # Center coordinates to avoid numerical instability
            centroid = ca_noisy.mean(dim=0, keepdim=True)
            ca_centered = ca_noisy - centroid

            h_out, x_out_centered = self.backbone(
                node_features, ca_centered, edge_index, edge_features,
                node_mask=constraint_mask,
            )
            x_out = x_out_centered + centroid  # uncenter
        elif self.config.equivariant_backbone == 'ipa':
            # IPA needs frames and pair features
            from src.utils.geometry import rigid_from_3_points
            rotations, translations = rigid_from_3_points(
                noisy_coords[:, 0], noisy_coords[:, 1], noisy_coords[:, 2]
            )
            # Build pair features from edges
            pair_features = self._build_pair_features(
                edge_index, edge_features, L, noisy_coords.device
            )
            # IPA expects batch dimension
            h_out = self.backbone(
                node_features.unsqueeze(0),
                pair_features.unsqueeze(0),
                rotations.unsqueeze(0),
                translations.unsqueeze(0),
            ).squeeze(0)
            x_out = ca_noisy  # IPA doesn't update coords directly

        # Predict displacement for all 4 backbone atoms
        displacement = self.output_proj(h_out)  # (L, 12)
        displacement = displacement.view(L, 4, 3)

        # Clamp displacement to prevent coordinate explosion
        max_displacement = 10.0  # Angstroms
        displacement = displacement.clamp(-max_displacement, max_displacement)

        # For EGNN, also use the coordinate update for CA
        if self.config.equivariant_backbone == 'egnn':
            ca_displacement = (x_out - ca_noisy).clamp(-max_displacement, max_displacement)
            displacement[:, 1] = displacement[:, 1] + ca_displacement

        predicted_clean = noisy_coords + displacement
        return predicted_clean

    def _build_pair_features(
        self, edge_index: torch.Tensor, edge_features: torch.Tensor,
        L: int, device: torch.device,
    ) -> torch.Tensor:
        """Build dense pair feature matrix from sparse edge features (for IPA)."""
        pair_dim = self.config.pair_dim

        # Initialize with distance-based features
        pair = torch.zeros(L, L, pair_dim, device=device)

        if edge_features is not None:
            src, dst = edge_index[0], edge_index[1]
            # Add distance feature
            dist = edge_features[:, 3:4]  # distance is at index 3
            edge_with_dist = torch.cat([edge_features, dist], dim=-1)
            projected = self.pair_proj(edge_with_dist[:, :self.config.edge_dim + 1])
            pair[src, dst] = projected

        return pair

    def training_step(self, batch, batch_idx):
        """Training step: noise backbone, denoise, compute loss."""
        clean_coords = batch['coords']  # (B, L_padded, 4, 3)
        lengths = batch['length']  # (B,)
        constraint_mask = batch.get('constraint_mask')
        constraint_positions = batch.get('constraint_positions')
        template_coords = batch.get('template_coords')

        B = clean_coords.shape[0]
        device = clean_coords.device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(B):
            L_i = int(lengths[i].item())
            coords_i = clean_coords[i, :L_i]  # (L_i, 4, 3) — unpadded

            # Sample timestep
            t = self.schedule.sample_timestep(1, device=device)

            # Forward diffusion
            sigma_t = self.schedule.sigma_continuous(t)

            # If we have a template, noise from template (conditioned generation)
            if template_coords is not None:
                base = template_coords[i, :L_i]
            else:
                base = coords_i

            noise = torch.randn_like(coords_i) * sigma_t.view(1, 1, 1)
            noisy = base + noise

            # Build graph from noisy backbone (unpadded only)
            backbone = ProteinBackbone(coords=noisy.detach().cpu().numpy())
            graph = backbone.to_graph(k=min(30, L_i - 1)).to(device)

            # Prepare inputs (unpadded)
            mask_i = constraint_mask[i, :L_i] if constraint_mask is not None else None
            pos_i = constraint_positions[i] if constraint_positions is not None else None

            node_feat, edge_feat = self._prepare_inputs(
                noisy, t, graph, mask_i, pos_i,
            )

            # Denoise
            predicted_clean = self.denoise_step(
                noisy, t, node_feat, graph.edge_index, edge_feat, mask_i,
            )

            # MSE loss on predicted vs actual clean coords (unpadded)
            recon_loss = nn.functional.mse_loss(predicted_clean, coords_i)
            total_loss = total_loss + recon_loss

            # Bond length regularization
            bl_loss = bond_length_loss(predicted_clean)
            total_loss = total_loss + self.config.bond_loss_weight * bl_loss

            # Constraint loss (catalytic residue positions)
            if mask_i is not None and pos_i is not None:
                cat_indices = mask_i.nonzero(as_tuple=True)[0]
                if len(cat_indices) > 0:
                    pred_cat = predicted_clean[cat_indices, 1]  # CA of catalytic
                    target_cat = pos_i[:len(cat_indices)]
                    c_loss = nn.functional.mse_loss(pred_cat, target_cat)
                    total_loss = total_loss + self.config.constraint_loss_weight * c_loss

        total_loss = total_loss / B

        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Same as training but with self.log('val_loss', ...)
        clean_coords = batch['coords']
        lengths = batch['length']
        B = clean_coords.shape[0]
        device = clean_coords.device

        total_loss = torch.tensor(0.0, device=device)
        for i in range(B):
            L_i = int(lengths[i].item())
            coords_i = clean_coords[i, :L_i]  # unpadded
            t = torch.tensor([0.5], device=device)  # fixed t for validation
            sigma_t = self.schedule.sigma_continuous(t)
            noise = torch.randn_like(coords_i) * sigma_t.view(1, 1, 1)
            noisy = coords_i + noise

            backbone = ProteinBackbone(coords=noisy.detach().cpu().numpy())
            graph = backbone.to_graph(k=min(30, L_i - 1)).to(device)

            node_feat, edge_feat = self._prepare_inputs(noisy, t, graph)
            predicted = self.denoise_step(
                noisy, t, node_feat, graph.edge_index, edge_feat,
            )
            total_loss = total_loss + nn.functional.mse_loss(predicted, coords_i)

        total_loss = total_loss / B
        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss

    @torch.no_grad()
    def sample(
        self,
        spec: ActiveSiteSpec,
        n_residues: int,
        n_steps: int = 100,
        device: str = 'cpu',
    ) -> ProteinBackbone:
        """Generate a backbone via reverse diffusion from template.

        Starting from noised template, iteratively denoise to produce
        a backbone that satisfies catalytic constraints while staying
        close to the template.
        """
        self.eval()

        # Initialize from template with noise
        if spec.has_template:
            template = torch.tensor(
                spec.template_backbone[:n_residues],
                dtype=torch.float32, device=device,
            )
            noise_scale = spec.noise_level
        else:
            # No template: start from random (extended chain)
            template = self._make_extended_chain(n_residues, device)
            noise_scale = 1.0

        # Add initial noise
        x = template + noise_scale * self.config.sigma_max * torch.randn_like(template)

        # Constraint info
        constraint_mask = torch.tensor(
            spec.get_fixed_mask(n_residues), dtype=torch.bool, device=device
        )
        constraint_positions = None
        if spec.constraint.residues:
            positions = []
            for res in spec.constraint.residues:
                if 'CA' in res.atom_positions:
                    positions.append(
                        torch.tensor(res.atom_positions['CA'], dtype=torch.float32, device=device)
                    )
            if positions:
                constraint_positions = torch.stack(positions)

        # Reverse diffusion
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_current = timesteps[i].unsqueeze(0)
            t_next = timesteps[i + 1].unsqueeze(0)

            sigma_current = self.schedule.sigma_continuous(t_current)
            sigma_next = self.schedule.sigma_continuous(t_next)

            # Build graph
            backbone = ProteinBackbone(coords=x.detach().cpu().numpy())
            L = x.shape[0]
            graph = backbone.to_graph(k=min(30, L - 1)).to(device)

            # Prepare and denoise
            node_feat, edge_feat = self._prepare_inputs(
                x, t_current, graph, constraint_mask, constraint_positions,
            )
            x0_pred = self.denoise_step(
                x, t_current, node_feat, graph.edge_index, edge_feat, constraint_mask,
            )

            # DDPM-style step: interpolate between prediction and current
            if i < n_steps - 1:
                alpha = sigma_next / (sigma_current + 1e-8)
                x = alpha * x + (1 - alpha) * x0_pred
                # Add small noise for stochasticity
                x = x + sigma_next * 0.1 * torch.randn_like(x)
            else:
                x = x0_pred

            # Enforce catalytic constraints (project back)
            if constraint_positions is not None:
                cat_indices = constraint_mask.nonzero(as_tuple=True)[0]
                for j, idx in enumerate(cat_indices):
                    if j < len(constraint_positions):
                        # Soft projection toward target
                        target = constraint_positions[j]
                        x[idx, 1] = 0.8 * x[idx, 1] + 0.2 * target

        return ProteinBackbone(
            coords=x.detach().cpu().numpy(),
            pdb_id='generated',
        )

    def _make_extended_chain(self, n_residues: int, device: str) -> torch.Tensor:
        """Create an extended chain as starting point (no template).

        Uses a zigzag pattern to avoid degenerate collinear geometry.
        """
        from src.utils.protein_constants import BOND_LENGTHS, CA_CA_DISTANCE
        import math

        coords = torch.zeros(n_residues, 4, 3, device=device)
        for i in range(n_residues):
            # Place CA along x-axis with slight zigzag in y
            y_offset = 0.5 * ((-1) ** i)
            coords[i, 1, 0] = i * CA_CA_DISTANCE
            coords[i, 1, 1] = y_offset

            # Place N before CA with offset in z for non-degenerate frame
            coords[i, 0, 0] = coords[i, 1, 0] - BOND_LENGTHS[('N', 'CA')]
            coords[i, 0, 1] = y_offset + 0.3
            coords[i, 0, 2] = 0.2

            # Place C after CA
            coords[i, 2, 0] = coords[i, 1, 0] + BOND_LENGTHS[('CA', 'C')]
            coords[i, 2, 1] = y_offset - 0.3
            coords[i, 2, 2] = -0.2

            # Place O above C
            coords[i, 3, 0] = coords[i, 2, 0]
            coords[i, 3, 1] = coords[i, 2, 1] + BOND_LENGTHS[('C', 'O')]
            coords[i, 3, 2] = coords[i, 2, 2]

        return coords

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }
