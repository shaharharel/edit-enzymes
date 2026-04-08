"""Abstract base class for backbone generators."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import pytorch_lightning as pl

from src.data.protein_structure import ProteinBackbone
from src.data.catalytic_constraints import ActiveSiteSpec


class AbstractBackboneGenerator(pl.LightningModule, ABC):
    """Base class for protein backbone generators.

    All backbone generators must implement:
    - denoise_step: Single denoising step (used in training)
    - sample: Full generation from noise or template
    """

    @abstractmethod
    def denoise_step(
        self,
        noisy_coords: torch.Tensor,
        timestep: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoising step.

        Args:
            noisy_coords: (B, L, 4, 3) noised backbone coordinates
            timestep: (B,) diffusion timestep
            node_features: (B, L, d) per-residue features
            edge_index: (2, E) graph edges
            edge_features: (E, d_edge) edge features
            constraint_mask: (B, L) mask for catalytic residue positions

        Returns:
            predicted_clean: (B, L, 4, 3) predicted clean coordinates
                (or predicted noise, depending on parameterization)
        """
        ...

    @abstractmethod
    def sample(
        self,
        spec: ActiveSiteSpec,
        n_residues: int,
        n_steps: int = 100,
        device: str = 'cpu',
    ) -> ProteinBackbone:
        """Generate a backbone via reverse diffusion.

        Args:
            spec: Active site specification with template and constraints
            n_residues: Number of residues to generate
            n_steps: Number of reverse diffusion steps
            device: Device for computation

        Returns:
            Generated ProteinBackbone
        """
        ...
