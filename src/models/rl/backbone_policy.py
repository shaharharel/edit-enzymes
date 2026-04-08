"""Backbone generator wrapped as an RL policy (REINFORCE).

Wraps SE3BackboneDiffusion to support:
- Generation with log-probability tracking
- REINFORCE policy gradient updates
- Learned value baseline for variance reduction
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from src.data.catalytic_constraints import ActiveSiteSpec
from src.data.protein_structure import ProteinBackbone, ProteinGraph
from src.models.backbone_generator.diffusion_model import SE3BackboneDiffusion, DiffusionConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BackboneValueBaseline(nn.Module):
    """Learned value baseline for REINFORCE variance reduction.

    Predicts expected reward from the active site specification,
    encoded as a fixed-size feature vector.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, spec_features: torch.Tensor) -> torch.Tensor:
        """Predict expected reward from specification features.

        Args:
            spec_features: (B, input_dim) specification feature vector.

        Returns:
            (B,) predicted baseline values.
        """
        return self.feature_encoder(spec_features).squeeze(-1)


class BackbonePolicy(nn.Module):
    """RL policy wrapper around SE3BackboneDiffusion.

    Uses REINFORCE with a learned baseline for policy gradient updates.
    The log-probability of a generated backbone is computed as the sum
    of per-step denoising log-probabilities under a Gaussian model.
    """

    def __init__(
        self,
        generator: SE3BackboneDiffusion,
        baseline_input_dim: int = 64,
        baseline_hidden_dim: int = 128,
        learning_rate: float = 1e-5,
        baseline_learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            generator: Pre-trained backbone diffusion model.
            baseline_input_dim: Feature dimension for value baseline input.
            baseline_hidden_dim: Hidden dimension for value baseline MLP.
            learning_rate: Learning rate for policy (generator) updates.
            baseline_learning_rate: Learning rate for baseline updates.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        super().__init__()
        self.generator = generator
        self.baseline = BackboneValueBaseline(baseline_input_dim, baseline_hidden_dim)
        self.learning_rate = learning_rate
        self.baseline_learning_rate = baseline_learning_rate
        self.max_grad_norm = max_grad_norm

    def encode_spec(self, spec: ActiveSiteSpec, device: str = 'cpu') -> torch.Tensor:
        """Encode an ActiveSiteSpec into a fixed-size feature vector for the baseline.

        Args:
            spec: Active site specification.
            device: Target device.

        Returns:
            (1, baseline_input_dim) feature vector.
        """
        input_dim = self.baseline.feature_encoder[0].in_features
        features = torch.zeros(1, input_dim, device=device)

        # Encode basic properties
        features[0, 0] = float(spec.noise_level)
        features[0, 1] = float(spec.constraint.num_residues)
        features[0, 2] = float(len(spec.fixed_residue_indices))
        features[0, 3] = 1.0 if spec.has_template else 0.0

        # Encode catalytic residue positions (if available)
        for i, res in enumerate(spec.constraint.residues):
            if i >= 10:
                break
            if 'CA' in res.atom_positions:
                ca = res.atom_positions['CA']
                offset = 4 + i * 3
                if offset + 3 <= input_dim:
                    features[0, offset:offset + 3] = torch.tensor(
                        ca, dtype=torch.float32, device=device,
                    )

        return features

    def generate_with_log_prob(
        self,
        spec: ActiveSiteSpec,
        n_residues: int,
        n_steps: int = 50,
        device: str = 'cpu',
    ) -> Tuple[ProteinBackbone, torch.Tensor]:
        """Generate a backbone and compute its log-probability under the policy.

        The log-probability is the sum of per-step Gaussian log-probs, where
        at each denoising step the model predicts x0 and the next state is
        sampled from N(interpolated_mean, sigma_next * noise_scale).

        Args:
            spec: Active site specification with template and constraints.
            n_residues: Number of residues to generate.
            n_steps: Number of reverse diffusion steps.
            device: Device for computation.

        Returns:
            backbone: Generated ProteinBackbone.
            log_prob: Scalar log-probability of the generation trajectory.
        """
        self.generator.eval()

        config = self.generator.config

        # Initialize from template with noise (same as generator.sample)
        if spec.has_template:
            template = torch.tensor(
                spec.template_backbone[:n_residues],
                dtype=torch.float32, device=device,
            )
            noise_scale = spec.noise_level
        else:
            template = self.generator._make_extended_chain(n_residues, device)
            noise_scale = 1.0

        # Add initial noise
        x = template + noise_scale * config.sigma_max * torch.randn_like(template)

        # Constraint info
        constraint_mask = torch.tensor(
            spec.get_fixed_mask(n_residues), dtype=torch.bool, device=device,
        )
        constraint_positions = None
        if spec.constraint.residues:
            positions = []
            for res in spec.constraint.residues:
                if 'CA' in res.atom_positions:
                    positions.append(
                        torch.tensor(
                            res.atom_positions['CA'],
                            dtype=torch.float32, device=device,
                        )
                    )
            if positions:
                constraint_positions = torch.stack(positions)

        # Reverse diffusion with log-prob tracking
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        total_log_prob = torch.tensor(0.0, device=device)
        stochastic_noise_scale = 0.1  # matches generator.sample

        for i in range(n_steps):
            t_current = timesteps[i].unsqueeze(0)
            t_next = timesteps[i + 1].unsqueeze(0)

            sigma_current = self.generator.schedule.sigma_continuous(t_current)
            sigma_next = self.generator.schedule.sigma_continuous(t_next)

            # Build graph
            backbone_tmp = ProteinBackbone(coords=x.detach().cpu().numpy())
            L = x.shape[0]
            graph = backbone_tmp.to_graph(k=min(30, L - 1)).to(device)

            # Prepare and denoise (with gradient tracking)
            node_feat, edge_feat = self.generator._prepare_inputs(
                x, t_current, graph, constraint_mask, constraint_positions,
            )
            x0_pred = self.generator.denoise_step(
                x, t_current, node_feat, graph.edge_index, edge_feat,
                constraint_mask,
            )

            # DDPM-style step
            if i < n_steps - 1:
                alpha = sigma_next / (sigma_current + 1e-8)
                mean = alpha * x + (1 - alpha) * x0_pred
                std = sigma_next * stochastic_noise_scale

                # Sample next state
                noise = torch.randn_like(x)
                x_next = mean + std * noise

                # Log-probability: sum of Gaussian log-probs over all coords
                # log p(x_next | mean, std) = -0.5 * sum((x_next - mean)^2 / std^2)
                # We omit the constant normalization term as it doesn't affect gradients
                log_prob_step = -0.5 * torch.sum((noise) ** 2)
                total_log_prob = total_log_prob + log_prob_step

                x = x_next.detach()
                x.requires_grad_(True)
            else:
                x = x0_pred

            # Enforce catalytic constraints (soft projection)
            if constraint_positions is not None:
                cat_indices = constraint_mask.nonzero(as_tuple=True)[0]
                x_data = x.detach()
                for j, idx in enumerate(cat_indices):
                    if j < len(constraint_positions):
                        target = constraint_positions[j]
                        x_data[idx, 1] = 0.8 * x_data[idx, 1] + 0.2 * target
                x = x_data
                x.requires_grad_(True)

        backbone = ProteinBackbone(
            coords=x.detach().cpu().numpy(),
            pdb_id='rl_generated',
        )

        return backbone, total_log_prob

    def compute_loss(
        self,
        reward: torch.Tensor,
        log_prob: torch.Tensor,
        spec_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute REINFORCE loss with learned baseline.

        Args:
            reward: Scalar reward for the generated backbone.
            log_prob: Log-probability of the generation trajectory.
            spec_features: (1, input_dim) features for baseline prediction.

        Returns:
            policy_loss: REINFORCE loss for the generator.
            baseline_loss: MSE loss for the value baseline.
            advantage: reward - baseline (for logging).
        """
        # Baseline prediction
        baseline_value = self.baseline(spec_features)

        # Advantage = reward - baseline
        advantage = reward - baseline_value.detach()

        # REINFORCE: loss = -advantage * log_prob
        policy_loss = -(advantage * log_prob)

        # Baseline loss: MSE between predicted baseline and actual reward
        baseline_loss = nn.functional.mse_loss(baseline_value, reward.unsqueeze(0))

        return policy_loss, baseline_loss, advantage

    def get_optimizers(self):
        """Create separate optimizers for policy and baseline.

        Returns:
            Tuple of (policy_optimizer, baseline_optimizer).
        """
        policy_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
        )
        baseline_optimizer = torch.optim.Adam(
            self.baseline.parameters(),
            lr=self.baseline_learning_rate,
        )
        return policy_optimizer, baseline_optimizer
