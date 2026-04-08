"""Variance schedules for backbone diffusion.

Supports linear, cosine, and polynomial schedules for controlling
the noise level during forward and reverse diffusion.
"""

import torch
import numpy as np
import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class DiffusionSchedule:
    """Diffusion noise schedule.

    Defines how noise increases from t=0 (clean) to t=1 (full noise).

    Attributes:
        schedule_type: Type of schedule ('linear', 'cosine', 'polynomial')
        T: Number of discrete diffusion steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        s: Offset for cosine schedule (prevents singularity at t=0)
        power: Exponent for polynomial schedule
    """
    schedule_type: Literal['linear', 'cosine', 'polynomial'] = 'cosine'
    T: int = 1000
    sigma_min: float = 0.01
    sigma_max: float = 10.0
    s: float = 0.008  # cosine schedule offset
    power: float = 2.0  # polynomial schedule exponent

    def __post_init__(self):
        # Pre-compute discrete schedule values
        self._sigmas = self._compute_sigmas()
        self._alphas = self._compute_alphas()

    def _compute_sigmas(self) -> torch.Tensor:
        """Compute sigma values for each timestep."""
        t = torch.linspace(0, 1, self.T + 1)

        if self.schedule_type == 'linear':
            sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * t

        elif self.schedule_type == 'cosine':
            f_t = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
            f_0 = math.cos(self.s / (1 + self.s) * math.pi / 2) ** 2
            alpha_bar = f_t / f_0
            alpha_bar = torch.clamp(alpha_bar, 1e-5, 1.0)
            sigmas = torch.sqrt((1 - alpha_bar) / alpha_bar) * self.sigma_min
            sigmas = torch.clamp(sigmas, self.sigma_min, self.sigma_max)

        elif self.schedule_type == 'polynomial':
            sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * (t ** self.power)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return sigmas

    def _compute_alphas(self) -> torch.Tensor:
        """Compute alpha values (signal retention) from sigmas."""
        return 1.0 / (1.0 + self._sigmas ** 2)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level at continuous time t in [0, 1].

        Args:
            t: (...,) time values in [0, 1]

        Returns:
            sigma: (...,) noise levels
        """
        # Interpolate from pre-computed schedule
        t_idx = (t * self.T).long().clamp(0, self.T)
        return self._sigmas.to(t.device)[t_idx]

    def sigma_continuous(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level at continuous time (no discretization)."""
        if self.schedule_type == 'linear':
            return self.sigma_min + (self.sigma_max - self.sigma_min) * t
        elif self.schedule_type == 'cosine':
            f_t = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
            f_0 = math.cos(self.s / (1 + self.s) * math.pi / 2) ** 2
            alpha_bar = f_t / f_0
            alpha_bar = torch.clamp(alpha_bar, 1e-5, 1.0)
            return torch.sqrt((1 - alpha_bar) / alpha_bar) * self.sigma_min
        elif self.schedule_type == 'polynomial':
            return self.sigma_min + (self.sigma_max - self.sigma_min) * (t ** self.power)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Get signal retention factor at time t."""
        sigma_t = self.sigma_continuous(t)
        return 1.0 / (1.0 + sigma_t ** 2)

    def sample_timestep(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """Sample random timesteps uniformly in [0, 1].

        Args:
            batch_size: Number of timesteps to sample
            device: Device for output tensor

        Returns:
            (batch_size,) tensor of timesteps
        """
        return torch.rand(batch_size, device=device)

    def get_discrete_sigmas(self) -> torch.Tensor:
        """Get the full pre-computed sigma schedule."""
        return self._sigmas

    def __repr__(self) -> str:
        return (
            f"DiffusionSchedule({self.schedule_type}, T={self.T}, "
            f"σ=[{self.sigma_min:.3f}, {self.sigma_max:.3f}])"
        )
