"""DDPO trajectory and policy abstractions.

DDPO (Denoising Diffusion Policy Optimization) treats the diffusion
denoising process as an MDP:
- State: noisy structure x_t at timestep t
- Action: denoised structure x_{t-1}
- Policy: π(x_{t-1} | x_t, θ) = N(x_{t-1}; μ_θ(x_t, t), σ_t²)
- Reward: only at final step (Rosetta score of x_0)

The trajectory stores the full denoising path for PPO replay.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import numpy as np

from src.data.protein_structure import ProteinBackbone
from src.data.catalytic_constraints import ActiveSiteSpec


@dataclass
class DDPOTrajectory:
    """Stores a denoising trajectory for DDPO replay.

    For V1 (our custom model): stores per-step (x_t, noise, t, mean, std).
    For V3 (RFD3): stores the final design + pipeline data for loss proxy.
    """
    # The generated design
    design: Any  # ProteinBackbone for v1, CIF path for v3

    # Old log_prob (scalar, sum of per-step log_probs, detached)
    old_log_prob: float = 0.0

    # V1: per-step trajectory data (true DDPO)
    states: Optional[List[torch.Tensor]] = None      # x_t at each step t
    actions: Optional[List[torch.Tensor]] = None      # x_{t-1} (the next state)
    timesteps: Optional[List[float]] = None           # t values
    means: Optional[List[torch.Tensor]] = None        # μ_θ(x_t, t) predictions
    stds: Optional[List[float]] = None                # σ_t noise schedule values

    # V3: proxy data for Docker-based replay
    pipeline_data: Optional[Any] = None  # Featurized data for model.forward()
    old_loss: Optional[float] = None     # Diffusion loss as -log_prob proxy

    @property
    def n_steps(self) -> int:
        if self.timesteps is not None:
            return len(self.timesteps)
        return 0


class DiffusionPolicyBase(ABC):
    """Abstract interface for DDPO-compatible diffusion policies.

    Both V1 (our custom EGNN/IPA) and V3 (RFD3) implement this.
    The DDPOTrainer works with any implementation.
    """

    @abstractmethod
    def generate_with_trajectory(
        self,
        spec: ActiveSiteSpec,
        n_residues: int,
        n_steps: int = 50,
        device: str = 'cpu',
    ) -> Tuple[Any, DDPOTrajectory]:
        """Generate a design, storing trajectory for later PPO replay.

        Args:
            spec: Active site specification with constraints
            n_residues: Number of residues to generate
            n_steps: Number of denoising steps
            device: Device for computation

        Returns:
            design: The generated design (ProteinBackbone for v1, path for v3)
            trajectory: DDPOTrajectory with stored denoising path
        """
        ...

    @abstractmethod
    def compute_log_prob(
        self,
        trajectory: DDPOTrajectory,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Compute log_prob of trajectory under CURRENT model weights.

        This is called during PPO update with gradients enabled.
        Returns a scalar tensor that is differentiable w.r.t. model params.

        For V1: replay trajectory step by step, sum per-step Gaussian log_probs.
        For V3: run model.forward() + loss() on the design (log_prob proxy).
        """
        ...

    @abstractmethod
    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return parameters that should be updated by the optimizer."""
        ...

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model weights to disk."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model weights from disk."""
        ...
