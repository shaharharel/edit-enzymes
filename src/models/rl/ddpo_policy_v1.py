"""DDPO policy for our custom SE3BackboneDiffusion (V1).

True per-step DDPO: stores the full denoising trajectory during generation,
then replays it through the updated model to compute new log_probs.

Per-step log_prob:
    log π(x_{t-1} | x_t, θ) = log N(x_{t-1}; μ_θ(x_t, t), σ_t²)
                              = -0.5 * Σ_d ((x_{t-1,d} - μ_d)² / σ_t²) - D/2 * log(2πσ_t²)

Total trajectory log_prob: Σ_t log π(x_{t-1} | x_t, θ)
"""

import torch
import numpy as np
from typing import Iterator, Tuple

from src.data.protein_structure import ProteinBackbone
from src.data.catalytic_constraints import ActiveSiteSpec
from src.models.backbone_generator.diffusion_model import SE3BackboneDiffusion
from src.models.rl.ddpo_trajectory import DDPOTrajectory, DiffusionPolicyBase
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DDPOPolicyV1(DiffusionPolicyBase):
    """DDPO policy wrapping our custom SE3BackboneDiffusion.

    During generation: stores {x_t, x_{t-1}, t, μ_θ, σ_t} at each step.
    During replay: re-runs model forward at stored (x_t, t) to get new μ_θ,
    then computes Gaussian log_prob.
    """

    def __init__(self, generator: SE3BackboneDiffusion):
        self.generator = generator

    def generate_with_trajectory(
        self, spec: ActiveSiteSpec, n_residues: int,
        n_steps: int = 50, device: str = 'cpu',
    ) -> Tuple[ProteinBackbone, DDPOTrajectory]:
        """Generate backbone via reverse diffusion, storing trajectory."""
        self.generator.eval()

        # Initialize from template
        if spec.has_template:
            template = torch.tensor(
                spec.template_backbone[:n_residues],
                dtype=torch.float32, device=device,
            )
        else:
            template = self.generator._make_extended_chain(n_residues, device)

        noise_scale = spec.noise_level if spec.has_template else 1.0
        x = template + noise_scale * self.generator.config.sigma_max * torch.randn_like(template)

        # Constraint info
        constraint_mask = torch.tensor(
            spec.get_fixed_mask(n_residues), dtype=torch.bool, device=device
        )

        # Storage for trajectory
        states = [x.detach().clone()]
        actions = []
        timesteps_list = []
        means_list = []
        stds_list = []
        total_log_prob = 0.0

        # Reverse diffusion with trajectory tracking
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                t_current = ts[i].unsqueeze(0)
                t_next = ts[i + 1].unsqueeze(0)

                sigma_current = self.generator.schedule.sigma_continuous(t_current)
                sigma_next = self.generator.schedule.sigma_continuous(t_next)

                # Build graph and prepare inputs
                backbone = ProteinBackbone(coords=x.detach().cpu().numpy())
                L = x.shape[0]
                graph = backbone.to_graph(k=min(30, L - 1)).to(device)
                node_feat, edge_feat = self.generator._prepare_inputs(
                    x, t_current, graph, constraint_mask,
                )

                # Predict clean coordinates
                x0_pred = self.generator.denoise_step(
                    x, t_current, node_feat, graph.edge_index, edge_feat, constraint_mask,
                )

                # Compute step mean and std
                if i < n_steps - 1:
                    alpha = sigma_next / (sigma_current + 1e-8)
                    mean = alpha * x + (1 - alpha) * x0_pred
                    std = float(sigma_next * 0.1)  # noise scale for stochasticity
                    noise = torch.randn_like(x) * std
                    x_next = mean + noise
                else:
                    mean = x0_pred
                    std = 1e-8  # deterministic final step
                    x_next = x0_pred

                # Per-step log_prob: log N(x_next; mean, std²)
                if std > 1e-7:
                    diff = (x_next - mean) / std
                    step_log_prob = -0.5 * torch.sum(diff ** 2).item()
                    # Omit constant terms (they cancel in ratio)
                else:
                    step_log_prob = 0.0

                total_log_prob += step_log_prob

                # Store trajectory
                timesteps_list.append(float(t_current))
                means_list.append(mean.detach().clone())
                stds_list.append(std)
                actions.append(x_next.detach().clone())

                x = x_next
                states.append(x.detach().clone())

        backbone = ProteinBackbone(
            coords=x.detach().cpu().numpy(),
            pdb_id='ddpo_generated',
        )

        trajectory = DDPOTrajectory(
            design=backbone,
            old_log_prob=total_log_prob,
            states=states,
            actions=actions,
            timesteps=timesteps_list,
            means=means_list,
            stds=stds_list,
        )

        return backbone, trajectory

    def compute_log_prob(
        self, trajectory: DDPOTrajectory, device: str = 'cpu',
    ) -> torch.Tensor:
        """Replay trajectory through CURRENT model weights with gradients.

        Returns differentiable log_prob (sum of per-step Gaussian log_probs).
        """
        self.generator.train()

        total_log_prob = torch.tensor(0.0, device=device)

        for i in range(trajectory.n_steps):
            x_t = trajectory.states[i].to(device)      # state at step i
            x_next = trajectory.actions[i].to(device)   # action taken (x_{t-1})
            t = torch.tensor([trajectory.timesteps[i]], device=device)
            std = trajectory.stds[i]

            if std < 1e-7:
                continue  # skip deterministic steps

            # Forward through model WITH gradients
            L = x_t.shape[0]
            backbone = ProteinBackbone(coords=x_t.detach().cpu().numpy())
            graph = backbone.to_graph(k=min(30, L - 1)).to(device)
            node_feat, edge_feat = self.generator._prepare_inputs(x_t, t, graph)

            x0_pred = self.generator.denoise_step(
                x_t, t, node_feat, graph.edge_index, edge_feat,
            )

            # Compute mean (same formula as generation)
            t_current = trajectory.timesteps[i]
            if i < trajectory.n_steps - 1:
                t_next = trajectory.timesteps[i + 1] if i + 1 < len(trajectory.timesteps) else 0.0
                sigma_current = self.generator.schedule.sigma_continuous(
                    torch.tensor([t_current], device=device)
                )
                sigma_next = self.generator.schedule.sigma_continuous(
                    torch.tensor([t_next], device=device)
                )
                alpha = sigma_next / (sigma_current + 1e-8)
                mean = alpha * x_t + (1 - alpha) * x0_pred  # differentiable
            else:
                mean = x0_pred

            # Gaussian log_prob: -0.5 * Σ((x_next - mean)² / std²)
            diff = (x_next.detach() - mean) / std  # x_next is fixed, mean has grad
            step_log_prob = -0.5 * torch.sum(diff ** 2)

            total_log_prob = total_log_prob + step_log_prob

        return total_log_prob

    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        return (p for p in self.generator.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str):
        torch.save(self.generator.state_dict(), path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.generator.load_state_dict(state, strict=False)
