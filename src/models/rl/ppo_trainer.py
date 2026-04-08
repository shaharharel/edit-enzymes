"""RL Trainer: PPO-based optimization loop for enzyme design.

Orchestrates the full RL loop:
1. Sample backbone (BackbonePolicy)
2. Design sequence (SequencePolicy)
3. Score design (RewardFunction)
4. Update policies with credit-assigned rewards

Uses GAE for advantage estimation and alternating updates between
backbone (REINFORCE) and sequence (PPO) policies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data.catalytic_constraints import ActiveSiteSpec
from src.data.protein_structure import ProteinBackbone, ProteinGraph
from src.models.rl.backbone_policy import BackbonePolicy
from src.models.rl.sequence_policy import SequencePolicy
from src.models.rl.reward import RewardFunction
from src.models.sequence_generator.graph_features import backbone_to_graph_features
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RolloutEntry:
    """Single rollout entry storing a complete design trajectory."""
    backbone: ProteinBackbone
    sequence: str
    backbone_log_prob: torch.Tensor
    sequence_log_prob: torch.Tensor
    sequence_entropy: torch.Tensor
    sequence_value: torch.Tensor
    rewards: Dict[str, torch.Tensor]
    graph: ProteinGraph
    fixed_mask: Optional[torch.Tensor] = None
    sequence_indices: Optional[torch.Tensor] = None


class RolloutBuffer:
    """Buffer for collecting RL trajectories."""

    def __init__(self):
        self.entries: List[RolloutEntry] = []

    def add(self, entry: RolloutEntry):
        self.entries.append(entry)

    def clear(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)

    def get_backbone_rewards(self) -> torch.Tensor:
        """Get backbone rewards as a tensor."""
        return torch.stack([e.rewards['backbone_reward'] for e in self.entries])

    def get_sequence_rewards(self) -> torch.Tensor:
        """Get sequence rewards as a tensor."""
        return torch.stack([e.rewards['sequence_reward'] for e in self.entries])

    def get_total_rewards(self) -> torch.Tensor:
        """Get total rewards as a tensor."""
        return torch.stack([e.rewards['total'] for e in self.entries])


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    For enzyme design, each rollout is a single-step episode (one design
    per rollout), so GAE simplifies to advantage = reward - value.
    We support multi-step for future extensions.

    Args:
        rewards: (N,) rewards for each rollout.
        values: (N,) predicted values for each rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: (N,) GAE advantage estimates.
        returns: (N,) discounted returns (advantage + value).
    """
    N = len(rewards)
    advantages = torch.zeros(N, device=rewards.device)
    returns = torch.zeros(N, device=rewards.device)

    # For single-step episodes, advantage = reward - value
    # With GAE, this is the same since there's no future reward
    advantages = rewards - values.detach()
    returns = rewards

    return advantages, returns


class RLTrainer:
    """Full RL training loop for enzyme design optimization.

    Alternates between:
    1. Rollout collection: generate backbones + sequences, score them
    2. Backbone policy update: REINFORCE with learned baseline
    3. Sequence policy update: PPO with clipped objective
    """

    def __init__(
        self,
        backbone_policy: BackbonePolicy,
        sequence_policy: SequencePolicy,
        reward_fn: RewardFunction,
        spec: ActiveSiteSpec,
        n_residues: int = 100,
        n_diffusion_steps: int = 50,
        sampling_temperature: float = 0.3,
        # RL hyperparameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # Training
        rollouts_per_update: int = 8,
        ppo_epochs: int = 4,
        backbone_update_frequency: int = 2,
        max_grad_norm: float = 1.0,
        feature_dim: int = 256,
        device: str = 'cpu',
    ):
        """
        Args:
            backbone_policy: Backbone generator as RL policy.
            sequence_policy: Sequence generator as RL policy.
            reward_fn: Reward function combining scoring and constraints.
            spec: Active site specification for generation.
            n_residues: Number of residues to generate.
            n_diffusion_steps: Number of diffusion steps for backbone generation.
            sampling_temperature: Temperature for sequence sampling.
            gamma: Discount factor for GAE.
            gae_lambda: Lambda for GAE.
            rollouts_per_update: Number of rollouts before each update.
            ppo_epochs: Number of PPO epochs per update.
            backbone_update_frequency: Update backbone every N iterations.
            max_grad_norm: Maximum gradient norm for clipping.
            feature_dim: Feature dimension for scoring models.
            device: Device for computation.
        """
        self.backbone_policy = backbone_policy.to(device)
        self.sequence_policy = sequence_policy.to(device)
        self.reward_fn = reward_fn.to(device)
        self.spec = spec
        self.n_residues = n_residues
        self.n_diffusion_steps = n_diffusion_steps
        self.sampling_temperature = sampling_temperature
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollouts_per_update = rollouts_per_update
        self.ppo_epochs = ppo_epochs
        self.backbone_update_frequency = backbone_update_frequency
        self.max_grad_norm = max_grad_norm
        self.feature_dim = feature_dim
        self.device = device

        # Create optimizers
        self.bb_policy_opt, self.bb_baseline_opt = backbone_policy.get_optimizers()
        self.seq_opt = sequence_policy.get_optimizer()

        # Buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.best_reward = float('-inf')
        self.best_designs: List[Tuple[ProteinBackbone, str, float]] = []
        self.history: List[Dict[str, float]] = []

    def _sanitize_backbone(self, backbone: ProteinBackbone) -> ProteinBackbone:
        """Replace NaN/Inf values in backbone coordinates with template values.

        Untrained models may produce NaN coordinates. This ensures
        downstream components always receive valid geometry.

        Args:
            backbone: Generated backbone, possibly with NaN coords.

        Returns:
            Backbone with all NaN/Inf values replaced.
        """
        coords = backbone.coords.copy()
        bad_mask = ~np.isfinite(coords)
        if bad_mask.any():
            logger.debug(
                f"Sanitizing backbone: {bad_mask.sum()} NaN/Inf values found"
            )
            # Fall back to template if available
            if self.spec.has_template:
                template = self.spec.template_backbone[:coords.shape[0]]
                coords[bad_mask] = template[bad_mask]
            else:
                coords[bad_mask] = 0.0
        return ProteinBackbone(
            coords=coords,
            pdb_id=backbone.pdb_id,
            sequence=backbone.sequence,
        )

    def collect_rollout(self) -> RolloutEntry:
        """Collect a single rollout: backbone -> sequence -> score.

        Returns:
            RolloutEntry with all trajectory information.
        """
        device = self.device

        # Step 1: Generate backbone
        backbone, bb_log_prob = self.backbone_policy.generate_with_log_prob(
            self.spec, self.n_residues, self.n_diffusion_steps, device,
        )

        # Sanitize backbone (replace NaN from untrained models)
        backbone = self._sanitize_backbone(backbone)

        # Step 2: Build graph features for sequence design
        fixed_mask_np = self.spec.get_fixed_mask(backbone.length)
        graph = backbone_to_graph_features(
            backbone, k=min(30, backbone.length - 1), fixed_mask=fixed_mask_np,
        ).to(device)
        fixed_mask = torch.tensor(fixed_mask_np, dtype=torch.bool, device=device)

        # Sanitize graph features (clamp any remaining NaN/Inf)
        graph = ProteinGraph(
            node_features=torch.nan_to_num(graph.node_features, nan=0.0, posinf=1.0, neginf=-1.0),
            edge_index=graph.edge_index,
            edge_features=torch.nan_to_num(graph.edge_features, nan=0.0, posinf=1.0, neginf=-1.0),
            coords=torch.nan_to_num(graph.coords, nan=0.0),
            mask=graph.mask,
        )

        # Step 3: Design sequence
        sequence, seq_log_prob, seq_entropy, seq_value = (
            self.sequence_policy.sample_with_log_prob(
                graph, fixed_mask, self.sampling_temperature,
            )
        )

        # Step 4: Create synthetic features for scoring
        # In a full system, these would be computed from the structure+sequence
        # For now, use a simple encoding
        features = self._encode_design_features(backbone, sequence, device)

        # Step 5: Compute rewards
        rewards = self.reward_fn.compute(backbone, sequence, features)

        # Convert sequence to indices for PPO re-evaluation
        from src.utils.protein_constants import AA_1_INDEX
        seq_indices = torch.tensor(
            [AA_1_INDEX.get(aa, 0) for aa in sequence],
            dtype=torch.long, device=device,
        )

        entry = RolloutEntry(
            backbone=backbone,
            sequence=sequence,
            backbone_log_prob=bb_log_prob,
            sequence_log_prob=seq_log_prob,
            sequence_entropy=seq_entropy,
            sequence_value=seq_value,
            rewards=rewards,
            graph=graph,
            fixed_mask=fixed_mask,
            sequence_indices=seq_indices,
        )

        return entry

    def _encode_design_features(
        self,
        backbone: ProteinBackbone,
        sequence: str,
        device: str,
    ) -> torch.Tensor:
        """Encode a design (backbone + sequence) into a feature vector for scoring.

        Simple encoding: statistical summaries of backbone geometry + sequence
        composition. In production, this would use a learned feature extractor.

        Args:
            backbone: Generated backbone.
            sequence: Designed sequence.
            device: Target device.

        Returns:
            (1, feature_dim) feature tensor.
        """
        features = torch.zeros(1, self.feature_dim, device=device)

        # Backbone geometry statistics
        ca = backbone.ca_coords  # (L, 3)
        if len(ca) > 1:
            ca_dists = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
            features[0, 0] = float(np.mean(ca_dists))
            features[0, 1] = float(np.std(ca_dists))
            features[0, 2] = float(np.min(ca_dists))
            features[0, 3] = float(np.max(ca_dists))

        # Radius of gyration
        centroid = ca.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((ca - centroid) ** 2, axis=-1)))
        features[0, 4] = float(rg)

        # Sequence composition (amino acid frequencies)
        from src.utils.protein_constants import AA_1_INDEX
        for aa in sequence:
            idx = AA_1_INDEX.get(aa, -1)
            if 0 <= idx < 20:
                features[0, 10 + idx] += 1.0 / max(len(sequence), 1)

        # Sequence length
        features[0, 5] = float(len(sequence)) / 100.0

        return features

    def update_backbone_policy(self, iteration: int) -> Dict[str, float]:
        """Update backbone policy using REINFORCE.

        Args:
            iteration: Current training iteration.

        Returns:
            Dict with loss metrics.
        """
        if len(self.buffer) == 0:
            return {}

        # Only update every N iterations
        if iteration % self.backbone_update_frequency != 0:
            return {'backbone_skipped': 1.0}

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_baseline_loss = torch.tensor(0.0, device=self.device)
        total_advantage = 0.0
        has_policy_grad = False

        for entry in self.buffer.entries:
            spec_features = self.backbone_policy.encode_spec(
                self.spec, self.device,
            )

            policy_loss, baseline_loss, advantage = (
                self.backbone_policy.compute_loss(
                    entry.rewards['backbone_reward'],
                    entry.backbone_log_prob,
                    spec_features,
                )
            )
            total_policy_loss = total_policy_loss + policy_loss
            total_baseline_loss = total_baseline_loss + baseline_loss
            total_advantage += advantage.item()

            if entry.backbone_log_prob.requires_grad:
                has_policy_grad = True

        n = len(self.buffer)
        total_policy_loss = total_policy_loss / n
        total_baseline_loss = total_baseline_loss / n

        # Update policy (only if log_prob has gradients)
        if has_policy_grad and total_policy_loss.requires_grad:
            self.bb_policy_opt.zero_grad()
            total_policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(
                self.backbone_policy.generator.parameters(),
                self.max_grad_norm,
            )
            self.bb_policy_opt.step()

        # Update baseline (always has gradients)
        self.bb_baseline_opt.zero_grad()
        total_baseline_loss.backward()
        self.bb_baseline_opt.step()

        return {
            'backbone_policy_loss': total_policy_loss.item(),
            'backbone_baseline_loss': total_baseline_loss.item(),
            'backbone_advantage': total_advantage / n,
        }

    def update_sequence_policy(self) -> Dict[str, float]:
        """Update sequence policy using PPO.

        Runs multiple epochs of PPO updates over the collected rollouts.

        Returns:
            Dict with loss metrics.
        """
        if len(self.buffer) == 0:
            return {}

        all_metrics: Dict[str, List[float]] = {
            'seq_policy_loss': [],
            'seq_value_loss': [],
            'seq_entropy': [],
            'seq_approx_kl': [],
        }

        # Compute advantages using GAE
        seq_rewards = self.buffer.get_sequence_rewards()
        seq_values = torch.stack(
            [e.sequence_value.squeeze().detach() for e in self.buffer.entries],
        )
        advantages, returns = compute_gae(
            seq_rewards, seq_values, self.gamma, self.gae_lambda,
        )

        # Normalize advantages
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        for epoch in range(self.ppo_epochs):
            for i, entry in enumerate(self.buffer.entries):
                # Re-evaluate actions under current policy
                log_prob_new, entropy_new, value_new = (
                    self.sequence_policy.evaluate_actions(
                        entry.graph,
                        entry.sequence_indices,
                        entry.fixed_mask,
                        self.sampling_temperature,
                    )
                )

                # PPO loss
                loss, info = self.sequence_policy.compute_ppo_loss(
                    log_prob_new=log_prob_new,
                    log_prob_old=entry.sequence_log_prob.detach(),
                    advantage=advantages[i],
                    entropy=entropy_new,
                    value=value_new,
                    returns=returns[i].unsqueeze(0),
                )

                # Update
                self.seq_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.sequence_policy.parameters(),
                    self.max_grad_norm,
                )
                self.seq_opt.step()

                all_metrics['seq_policy_loss'].append(info['policy_loss'])
                all_metrics['seq_value_loss'].append(info['value_loss'])
                all_metrics['seq_entropy'].append(info['entropy'])
                all_metrics['seq_approx_kl'].append(info['approx_kl'])

        # Average metrics
        return {k: float(np.mean(v)) for k, v in all_metrics.items() if v}

    def train_step(self, iteration: int) -> Dict[str, float]:
        """Execute one full RL training step.

        1. Collect rollouts
        2. Update backbone policy (REINFORCE)
        3. Update sequence policy (PPO)
        4. Track best designs

        Args:
            iteration: Current training iteration number.

        Returns:
            Dict with all metrics for logging.
        """
        self.buffer.clear()

        # Collect rollouts
        rollout_rewards = []
        for r in range(self.rollouts_per_update):
            entry = self.collect_rollout()
            self.buffer.add(entry)
            total_reward = entry.rewards['total'].item()
            rollout_rewards.append(total_reward)

            logger.info(
                f"  Rollout {r+1}/{self.rollouts_per_update}: "
                f"total_reward={total_reward:.4f}, "
                f"backbone_reward={entry.rewards['backbone_reward'].item():.4f}, "
                f"sequence_reward={entry.rewards['sequence_reward'].item():.4f}"
            )

            # Track best designs
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_designs.append(
                    (entry.backbone, entry.sequence, total_reward)
                )
                # Keep only top 10
                self.best_designs.sort(key=lambda x: x[2], reverse=True)
                self.best_designs = self.best_designs[:10]

        # Update policies
        bb_metrics = self.update_backbone_policy(iteration)
        seq_metrics = self.update_sequence_policy()

        # Aggregate metrics
        metrics = {
            'mean_reward': float(np.mean(rollout_rewards)),
            'max_reward': float(np.max(rollout_rewards)),
            'min_reward': float(np.min(rollout_rewards)),
            'best_reward_ever': self.best_reward,
            **bb_metrics,
            **seq_metrics,
        }

        # Add per-objective averages
        for key in ['constraint_satisfaction', 'geometry_feasibility',
                     'backbone_reward', 'sequence_reward']:
            values = [
                e.rewards[key].item()
                for e in self.buffer.entries
                if key in e.rewards
            ]
            if values:
                metrics[f'mean_{key}'] = float(np.mean(values))

        self.history.append(metrics)

        return metrics

    def train(self, n_iterations: int) -> List[Dict[str, float]]:
        """Run the full RL training loop.

        Args:
            n_iterations: Number of RL training iterations.

        Returns:
            List of per-iteration metrics.
        """
        logger.info(
            f"Starting RL training: {n_iterations} iterations, "
            f"{self.rollouts_per_update} rollouts/iter, "
            f"{self.ppo_epochs} PPO epochs"
        )

        for iteration in range(n_iterations):
            logger.info(f"\n=== RL Iteration {iteration + 1}/{n_iterations} ===")

            metrics = self.train_step(iteration)

            logger.info(
                f"Iteration {iteration + 1}: "
                f"mean_reward={metrics['mean_reward']:.4f}, "
                f"max_reward={metrics['max_reward']:.4f}, "
                f"best_ever={metrics['best_reward_ever']:.4f}"
            )

            if 'seq_policy_loss' in metrics:
                logger.info(
                    f"  PPO: policy_loss={metrics['seq_policy_loss']:.4f}, "
                    f"entropy={metrics.get('seq_entropy', 0):.4f}"
                )

            if 'backbone_policy_loss' in metrics:
                logger.info(
                    f"  REINFORCE: loss={metrics['backbone_policy_loss']:.4f}, "
                    f"advantage={metrics.get('backbone_advantage', 0):.4f}"
                )

        logger.info(
            f"\nTraining complete. Best reward: {self.best_reward:.4f}, "
            f"Top designs: {len(self.best_designs)}"
        )

        return self.history

    def get_best_designs(
        self, top_k: int = 5,
    ) -> List[Tuple[ProteinBackbone, str, float]]:
        """Get the top-k best designs found during training.

        Args:
            top_k: Number of designs to return.

        Returns:
            List of (backbone, sequence, reward) tuples.
        """
        return self.best_designs[:top_k]
