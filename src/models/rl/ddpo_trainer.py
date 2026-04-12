"""DDPO Trainer: backend-agnostic PPO for diffusion models.

Works with any DiffusionPolicyBase implementation (V1 custom or V3 RFD3).
Handles: rollout collection, advantage computation, PPO update dispatch.

Algorithm (DDPO-IS from Black et al. 2023):
1. Generate designs, storing trajectories
2. Score with reward function
3. Compute advantages (reward - baseline, normalized)
4. For K PPO epochs:
   a. Compute new_log_prob by replaying trajectory (with grad)
   b. ratio = exp(new_log_prob - old_log_prob)
   c. Clipped surrogate loss
   d. Backward + step
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.models.rl.ddpo_trajectory import DDPOTrajectory, DiffusionPolicyBase
from src.data.catalytic_constraints import ActiveSiteSpec
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DDPOTrainer:
    """DDPO training loop for enzyme design.

    Backend-agnostic: works with V1 (local model) or V3 (Docker).
    """

    def __init__(
        self,
        policy: DiffusionPolicyBase,
        reward_fn: Callable,
        spec: ActiveSiteSpec,
        n_residues: int = 100,
        n_denoising_steps: int = 50,
        # DDPO hyperparams
        rollouts_per_update: int = 8,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 10.0,
        # Baseline
        baseline_momentum: float = 0.95,
        # Device
        device: str = 'cpu',
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.spec = spec
        self.n_residues = n_residues
        self.n_denoising_steps = n_denoising_steps
        self.rollouts_per_update = rollouts_per_update
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            policy.get_trainable_parameters(), lr=learning_rate,
        )

        # Running baseline
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum

        # Tracking
        self.reward_history = []
        self.loss_history = []
        self.best_reward = float('-inf')

    def collect_rollouts(self) -> List[Dict]:
        """Generate designs, score them, return rollouts."""
        rollouts = []

        for i in range(self.rollouts_per_update):
            try:
                design, trajectory = self.policy.generate_with_trajectory(
                    self.spec, self.n_residues, self.n_denoising_steps, self.device,
                )
                reward = self.reward_fn(design)
                rollouts.append({
                    'trajectory': trajectory,
                    'reward': reward,
                    'design': design,
                })
            except Exception as e:
                logger.warning(f"Rollout {i} failed: {e}")

        return rollouts

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute normalized advantages with running baseline."""
        mean_r = np.mean(rewards)
        self.baseline = (
            self.baseline_momentum * self.baseline +
            (1 - self.baseline_momentum) * mean_r
        )
        std_r = max(np.std(rewards), 1e-8)
        return [(r - self.baseline) / std_r for r in rewards]

    def ppo_update(
        self, rollouts: List[Dict], advantages: List[float],
    ) -> Dict[str, float]:
        """PPO update: replay trajectories, compute ratio, clipped loss."""
        total_loss = 0.0
        n_updates = 0
        approx_kl = 0.0

        for epoch in range(self.ppo_epochs):
            for rollout, advantage in zip(rollouts, advantages):
                trajectory = rollout['trajectory']

                self.optimizer.zero_grad()

                # Compute NEW log_prob under current model (with gradients)
                new_log_prob = self.policy.compute_log_prob(trajectory, self.device)

                # PPO ratio
                old_log_prob = trajectory.old_log_prob
                if isinstance(old_log_prob, torch.Tensor):
                    old_log_prob = old_log_prob.detach()
                else:
                    old_log_prob = torch.tensor(old_log_prob, device=self.device)

                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(torch.clamp(log_ratio, -10, 10))

                # Clipped surrogate
                adv = torch.tensor(advantage, device=self.device, dtype=torch.float32)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * adv
                ppo_loss = -torch.min(surr1, surr2)

                # Backward
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.get_trainable_parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += ppo_loss.item()
                approx_kl += log_ratio.detach().abs().item()
                n_updates += 1

        return {
            'ppo_loss': total_loss / max(n_updates, 1),
            'approx_kl': approx_kl / max(n_updates, 1),
            'n_updates': n_updates,
        }

    def train_step(self, iteration: int) -> Dict[str, float]:
        """One full DDPO iteration."""
        t0 = time.time()

        # 1. Collect rollouts
        rollouts = self.collect_rollouts()
        if not rollouts:
            return {'error': 'no rollouts'}

        rewards = [r['reward'] for r in rollouts]
        t_collect = time.time() - t0

        # 2. Compute advantages
        advantages = self.compute_advantages(rewards)

        # 3. PPO update
        t_update = time.time()
        metrics = self.ppo_update(rollouts, advantages)
        t_update = time.time() - t_update

        # 4. Track
        mean_reward = np.mean(rewards)
        best_reward = max(rewards)
        self.reward_history.append(mean_reward)
        self.loss_history.append(metrics['ppo_loss'])

        if best_reward > self.best_reward:
            self.best_reward = best_reward

        metrics.update({
            'mean_reward': mean_reward,
            'best_reward': best_reward,
            'best_ever': self.best_reward,
            'baseline': self.baseline,
            't_collect': t_collect,
            't_update': t_update,
        })

        return metrics

    def save_results(self, path: str):
        """Save training history."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f'{path}/reward_history.json', 'w') as f:
            json.dump(self.reward_history, f)
        with open(f'{path}/loss_history.json', 'w') as f:
            json.dump(self.loss_history, f)
