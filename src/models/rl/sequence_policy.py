"""Sequence generator wrapped as an RL policy (PPO).

Wraps ProteinMPNNModel to support:
- Sampling with log-probability and entropy tracking
- PPO clipped objective updates
- Value head for advantage estimation
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.protein_structure import ProteinGraph
from src.models.sequence_generator.mpnn_model import ProteinMPNNModel, MPNNConfig
from src.utils.protein_constants import NUM_AA, AA_3TO1, AA_LIST
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SequenceValueHead(nn.Module):
    """Value head for PPO advantage estimation.

    Predicts expected reward from the encoder output of ProteinMPNN.
    Uses mean-pooling over residue embeddings to produce a single value.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, encoder_out: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict value from encoder embeddings.

        Args:
            encoder_out: (L, hidden_dim) per-residue encoder embeddings.
            mask: (L,) boolean mask for valid residues.

        Returns:
            (1,) predicted value.
        """
        # Mean-pool over valid residues
        valid = encoder_out[mask]
        if valid.shape[0] == 0:
            valid = encoder_out
        pooled = valid.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        return self.value_net(pooled).squeeze(-1)  # (1,)


class SequencePolicy(nn.Module):
    """RL policy wrapper around ProteinMPNNModel using PPO.

    Supports sampling with log-probability tracking, entropy computation,
    and PPO clipped surrogate objective for policy updates.
    """

    def __init__(
        self,
        model: ProteinMPNNModel,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: Pre-trained ProteinMPNN sequence generator.
            learning_rate: Learning rate for PPO updates.
            clip_epsilon: PPO clipping parameter.
            entropy_coeff: Coefficient for entropy bonus.
            value_coeff: Coefficient for value function loss.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        super().__init__()
        self.model = model
        self.value_head = SequenceValueHead(model.config.hidden_dim)
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

    def sample_with_log_prob(
        self,
        graph: ProteinGraph,
        fixed_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a sequence and compute log-probability and entropy.

        Uses autoregressive sampling (like ProteinMPNNModel.sample) but
        tracks log-probabilities and entropy for PPO updates.

        Args:
            graph: ProteinGraph with structural features.
            fixed_mask: (L,) boolean mask; True = fixed residue.
            temperature: Sampling temperature.

        Returns:
            sequence: Sampled amino acid sequence string.
            log_prob: Sum of per-position log-probabilities (scalar).
            entropy: Mean per-position entropy (scalar).
            value: Predicted value from value head (scalar).
        """
        L = graph.num_nodes
        device = graph.node_features.device

        # Encode structure
        encoder_out = self.model.encode(graph)

        # Value prediction
        value = self.value_head(encoder_out, graph.mask)

        # Initialize
        sampled_indices = torch.zeros(L, dtype=torch.long, device=device)
        total_log_prob = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        n_designed = 0

        # Extract fixed AAs from graph features if available
        fixed_aa = None
        if fixed_mask is not None and graph.node_features.shape[-1] >= 43:
            aa_onehot = graph.node_features[:, 23:43]
            fixed_aa = aa_onehot.argmax(dim=-1)
            has_fixed = aa_onehot.sum(dim=-1) > 0.5
            fixed_mask = fixed_mask & has_fixed

        # Autoregressive generation with log-prob tracking
        for i in range(L):
            if fixed_mask is not None and fixed_mask[i]:
                sampled_indices[i] = fixed_aa[i]
                continue

            # Build decoder input from current partial sequence
            prev_aa = torch.full(
                (L,), self.model.sos_idx, dtype=torch.long, device=device,
            )
            if i > 0:
                prev_aa[1:i + 1] = sampled_indices[:i]

            decoder_out = self.model.decode(encoder_out, graph, prev_aa)
            logits_i = self.model.output_head(decoder_out[i])  # (20,)

            # Temperature-scaled sampling with numerical stability
            scaled_logits = logits_i / max(temperature, 1e-6)
            # Clamp to prevent overflow in softmax
            scaled_logits = scaled_logits.clamp(-50.0, 50.0)
            probs = F.softmax(scaled_logits, dim=-1)
            log_probs = F.log_softmax(scaled_logits, dim=-1)

            # Ensure valid probability distribution
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum()

            # Sample
            if temperature < 1e-6:
                action = logits_i.argmax()
            else:
                action = torch.multinomial(probs, 1).squeeze()

            sampled_indices[i] = action

            # Accumulate log-prob and entropy
            total_log_prob = total_log_prob + log_probs[action]
            position_entropy = -(probs * log_probs).sum()
            total_entropy = total_entropy + position_entropy
            n_designed += 1

        # Average entropy over designed positions
        if n_designed > 0:
            total_entropy = total_entropy / n_designed

        # Convert indices to one-letter codes
        idx_to_aa = {i: AA_3TO1[aa] for i, aa in enumerate(AA_LIST)}
        sequence = ''.join(
            idx_to_aa.get(sampled_indices[i].item(), 'X')
            for i in range(L)
        )

        return sequence, total_log_prob, total_entropy, value

    def evaluate_actions(
        self,
        graph: ProteinGraph,
        actions: torch.Tensor,
        fixed_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given actions under current policy.

        Used during PPO update to compute the ratio pi(a|s) / pi_old(a|s).

        Args:
            graph: ProteinGraph with structural features.
            actions: (L,) amino acid indices of the sampled sequence.
            fixed_mask: (L,) boolean mask; True = fixed residue.
            temperature: Sampling temperature.

        Returns:
            log_prob: Sum of per-position log-probabilities (scalar).
            entropy: Mean per-position entropy (scalar).
            value: Predicted value (scalar).
        """
        L = graph.num_nodes
        device = graph.node_features.device

        # Encode
        encoder_out = self.model.encode(graph)
        value = self.value_head(encoder_out, graph.mask)

        # Use teacher forcing to get all logits at once
        logits = self.model.forward(graph, fixed_mask, true_sequence=actions)  # (L, 20)
        scaled_logits = logits / max(temperature, 1e-6)

        log_probs = F.log_softmax(scaled_logits, dim=-1)  # (L, 20)
        probs = F.softmax(scaled_logits, dim=-1)

        # Determine design mask
        if fixed_mask is not None:
            design_mask = ~fixed_mask & graph.mask
        else:
            design_mask = graph.mask

        # Gather log-probs for taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # (L,)
        total_log_prob = action_log_probs[design_mask].sum()

        # Entropy
        position_entropy = -(probs * log_probs).sum(dim=-1)  # (L,)
        n_designed = design_mask.sum().clamp(min=1)
        mean_entropy = position_entropy[design_mask].sum() / n_designed

        return total_log_prob, mean_entropy, value

    def compute_ppo_loss(
        self,
        log_prob_new: torch.Tensor,
        log_prob_old: torch.Tensor,
        advantage: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute PPO clipped surrogate loss.

        Args:
            log_prob_new: Log-prob under current policy.
            log_prob_old: Log-prob under old policy (from rollout).
            advantage: GAE advantage estimate.
            entropy: Mean entropy of current policy.
            value: Predicted value under current policy.
            returns: Discounted returns (target for value function).

        Returns:
            total_loss: Combined policy + value + entropy loss.
            info: Dict with individual loss components for logging.
        """
        # Ensure all inputs are scalar
        log_prob_new = log_prob_new.squeeze()
        log_prob_old = log_prob_old.squeeze()
        advantage = advantage.squeeze()
        entropy = entropy.squeeze()
        value = value.squeeze()
        returns = returns.squeeze()

        # Policy ratio
        ratio = torch.exp(log_prob_new - log_prob_old.detach())

        # Clipped surrogate
        surr1 = ratio * advantage.detach()
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        ) * advantage.detach()
        policy_loss = -torch.min(surr1, surr2)

        # Value loss
        value_loss = F.mse_loss(value, returns)

        # Entropy bonus (negative because we maximize entropy)
        entropy_loss = -entropy

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': (log_prob_old - log_prob_new).detach().item(),
            'clip_fraction': (
                (torch.abs(ratio - 1.0) > self.clip_epsilon).float().item()
            ),
        }

        return total_loss, info

    def get_optimizer(self):
        """Create optimizer for all policy parameters (model + value head).

        Returns:
            Adam optimizer.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
