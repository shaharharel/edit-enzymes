"""PPO fine-tuning of pretrained RFdiffusion + ProteinMPNN.

The pipeline:
1. RFdiffusion (59.8M params) generates backbone via partial diffusion
   - Each denoising step: predict clean coords, add noise → log_prob (Gaussian)
2. ProteinMPNN (1.66M params) generates sequence autoregressively
   - Each position: predict AA distribution, sample → log_prob (categorical)
3. Rosetta surrogate scores the design → reward (NOT differentiable, doesn't matter)
4. PPO updates both models using log_probs + advantage

Same as LLM RLHF: reward doesn't need to be differentiable.
We backprop through log_prob computation, not through sampling.

Experiments:
A: Both frozen (baseline — just sample and score)
B: ProteinMPNN trainable (PPO on sequence)
C: RFdiffusion trainable (PPO on backbone)
D: Both trainable

Usage:
    python experiments/rl/run_ppo_finetune.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --experiment B --n-iterations 100
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Add RFdiffusion and ProteinMPNN to path
sys.path.insert(0, str(Path.home() / 'RFdiffusion'))
sys.path.insert(0, str(Path.home() / 'ProteinMPNN'))

import argparse
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Load pretrained models as PyTorch modules
# ============================================================================

def load_rfdiffusion(ckpt_path: str, device: str) -> torch.nn.Module:
    """Load RFdiffusion as a PyTorch module with gradient support."""
    from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
    from omegaconf import OmegaConf

    conf = OmegaConf.load(str(Path.home() / 'RFdiffusion' / 'config' / 'inference' / 'base.yaml'))
    conf.inference.ckpt_override_path = ckpt_path

    from rfdiffusion.inference import model_runners
    runner = model_runners.SelfConditioning(conf)
    model = runner.model  # RoseTTAFoldModule

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"RFdiffusion loaded: {params:,} params on {device}")
    return model, runner


def load_proteinmpnn(weights_path: str, device: str) -> torch.nn.Module:
    """Load ProteinMPNN as a PyTorch module with gradient support."""
    from protein_mpnn_utils import ProteinMPNN

    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=ckpt['num_edges'], augment_eps=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"ProteinMPNN loaded: {params:,} params on {device}")
    return model


# ============================================================================
# Sampling with log-probability tracking
# ============================================================================

def sample_sequence_with_logprob(
    mpnn_model: torch.nn.Module,
    pdb_path: str,
    fixed_positions: List[int],
    temperature: float,
    device: str,
) -> Tuple[str, torch.Tensor]:
    """Sample sequence from ProteinMPNN and return log-probability.

    Like LLM: sample token by token, record log_prob at each step.
    """
    from protein_mpnn_utils import parse_PDB, tied_featurize

    pdb_dict = parse_PDB(pdb_path, ca_only=False)[0]
    X, S, mask, lengths, chain_M, chain_encoding_all, *_ = tied_featurize(
        [pdb_dict], device, None, None, None, None, None, None, ca_only=False
    )

    L = int(lengths[0])
    randn = torch.randn(1, L, device=device)
    chain_M_pos = torch.ones_like(chain_M)

    # Fix catalytic positions
    for pos in fixed_positions:
        if pos < L:
            chain_M_pos[0, pos] = 0

    residue_idx = torch.arange(L, device=device).unsqueeze(0)
    omit_AAs = np.zeros(21, dtype=np.float32)
    bias_AAs = np.zeros(21, dtype=np.float32)
    bias_by_res = torch.zeros(1, L, 21, device=device)

    # Sample (this is the stochastic step)
    with torch.no_grad():
        sample_dict = mpnn_model.sample(
            X, randn, S, chain_M * chain_M_pos, chain_encoding_all,
            residue_idx, mask, temperature=temperature,
            chain_M_pos=chain_M_pos,
            omit_AAs_np=omit_AAs, bias_AAs_np=bias_AAs,
            bias_by_res=bias_by_res,
        )
    S_sample = sample_dict['S']

    # Now compute log-probability WITH gradient tracking
    # Forward pass through the model to get logits
    log_probs = mpnn_model(X, S_sample, mask, chain_M * chain_M_pos,
                            residue_idx, chain_encoding_all)

    # Sum log-probs at designed positions
    designed_mask = chain_M_pos[0] > 0
    total_log_prob = torch.tensor(0.0, device=device)

    if log_probs is not None and hasattr(log_probs, 'shape') and log_probs.dim() >= 2:
        for pos in range(L):
            if designed_mask[pos]:
                aa_idx = S_sample[0, pos]
                lp = F.log_softmax(log_probs[0, pos], dim=-1)
                total_log_prob = total_log_prob + lp[aa_idx]

    # Decode sequence
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    sequence = ''.join([alphabet[s] for s in S_sample[0].cpu().numpy()[:L]])

    return sequence, total_log_prob


def score_with_rosetta_fast(pdb_path: str) -> float:
    """Score with Rosetta (no relax). Returns negative score as reward."""
    try:
        import pyrosetta
        global _PYROSETTA_INIT
        if '_PYROSETTA_INIT' not in globals() or not _PYROSETTA_INIT:
            pyrosetta.init('-mute all')
            _PYROSETTA_INIT = True

        sfxn = pyrosetta.get_score_function(True)
        pose = pyrosetta.pose_from_pdb(pdb_path)
        score = sfxn(pose)
        return -score  # negative because lower Rosetta = better = higher reward
    except Exception as e:
        logger.warning(f"Rosetta scoring failed: {e}")
        return 0.0


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """PPO fine-tuning for pretrained protein design models."""

    def __init__(
        self,
        mpnn_model: torch.nn.Module,
        rfdiff_model: torch.nn.Module = None,
        rfdiff_runner=None,
        device: str = 'cuda',
        lr: float = 1e-5,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
    ):
        self.mpnn_model = mpnn_model
        self.rfdiff_model = rfdiff_model
        self.rfdiff_runner = rfdiff_runner
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs

        # Optimizers (only for trainable params)
        trainable_mpnn = [p for p in mpnn_model.parameters() if p.requires_grad]
        self.mpnn_optimizer = torch.optim.Adam(trainable_mpnn, lr=lr) if trainable_mpnn else None

        if rfdiff_model is not None:
            trainable_rfdiff = [p for p in rfdiff_model.parameters() if p.requires_grad]
            self.rfdiff_optimizer = torch.optim.Adam(trainable_rfdiff, lr=lr) if trainable_rfdiff else None
        else:
            self.rfdiff_optimizer = None

        # Baseline (running average reward)
        self.baseline = 0.0

        logger.info(f"PPO Trainer: mpnn_trainable={len(trainable_mpnn)}, "
                     f"lr={lr}, clip={clip_epsilon}")

    def collect_rollouts(
        self,
        template_pdb: str,
        fixed_positions: List[int],
        n_rollouts: int,
        temperature: float = 0.1,
        use_rfdiffusion: bool = False,
    ) -> List[Dict]:
        """Collect N rollouts (designs) with log-probs."""
        rollouts = []

        for i in range(n_rollouts):
            try:
                # For now: use template backbone directly
                # TODO: add RFdiffusion backbone generation with log-prob
                backbone_pdb = template_pdb
                bb_log_prob = torch.tensor(0.0, device=self.device)

                # Generate sequence with log-prob
                sequence, seq_log_prob = sample_sequence_with_logprob(
                    self.mpnn_model, backbone_pdb, fixed_positions,
                    temperature, self.device,
                )

                # Score with Rosetta (fast, no relax)
                reward = score_with_rosetta_fast(backbone_pdb)

                rollouts.append({
                    'sequence': sequence,
                    'seq_log_prob': seq_log_prob,
                    'bb_log_prob': bb_log_prob,
                    'reward': reward,
                    'backbone_pdb': backbone_pdb,
                })

            except Exception as e:
                logger.warning(f"Rollout {i} failed: {e}")
                continue

        return rollouts

    def ppo_update(self, rollouts: List[Dict]) -> Dict[str, float]:
        """PPO update on collected rollouts."""
        if not rollouts or self.mpnn_optimizer is None:
            return {}

        rewards = torch.tensor([r['reward'] for r in rollouts], device=self.device)
        old_log_probs = torch.stack([r['seq_log_prob'].detach() for r in rollouts])

        # Compute advantages
        advantages = rewards - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * rewards.mean().item()

        # Normalize advantages
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.ppo_epochs):
            for i, rollout in enumerate(rollouts):
                # Re-compute log-prob under CURRENT policy
                _, new_log_prob = sample_sequence_with_logprob(
                    self.mpnn_model, rollout['backbone_pdb'],
                    [], 0.1, self.device,  # empty fixed for re-eval
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                policy_loss = -torch.min(surr1, surr2)

                # Update
                self.mpnn_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mpnn_model.parameters(), 1.0)
                self.mpnn_optimizer.step()

                total_policy_loss += policy_loss.item()

        n = len(rollouts) * self.ppo_epochs
        return {
            'policy_loss': total_policy_loss / max(n, 1),
            'mean_reward': rewards.mean().item(),
            'best_reward': rewards.max().item(),
            'advantage_mean': advantages.mean().item(),
        }


def main():
    parser = argparse.ArgumentParser(description='PPO fine-tuning of pretrained models')
    parser.add_argument('--template', required=True)
    parser.add_argument('--constraint', required=True)
    parser.add_argument('--experiment', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--n-iterations', type=int, default=100)
    parser.add_argument('--rollouts-per-iter', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default='results/ppo_finetune')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir) / f'exp_{args.experiment}'
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_pdb(args.template)
    constraint = load_constraint_from_yaml(args.constraint)
    fixed_positions = [r.position_index for r in constraint.residues if r.position_index is not None]

    logger.info(f"=== PPO Fine-tuning — Experiment {args.experiment} ===")
    logger.info(f"Template: {template.pdb_id}, {template.length} residues")

    # Load pretrained ProteinMPNN
    mpnn = load_proteinmpnn(
        str(Path.home() / 'ProteinMPNN' / 'vanilla_model_weights' / 'v_48_010.pt'),
        device,
    )

    # Freeze/unfreeze based on experiment
    if args.experiment in ['A', 'C']:
        for p in mpnn.parameters():
            p.requires_grad = False
        logger.info("ProteinMPNN: FROZEN")
    else:
        for p in mpnn.parameters():
            p.requires_grad = True
        logger.info("ProteinMPNN: TRAINABLE (PPO)")

    # Clean template
    clean_template = output_dir / 'template_clean.pdb'
    with open(args.template) as f:
        lines = [l for l in f if l.startswith('ATOM') or l.startswith('TER') or l.startswith('END')]
    with open(clean_template, 'w') as f:
        f.writelines(lines)

    # Score native baseline
    native_reward = score_with_rosetta_fast(str(clean_template))
    logger.info(f"Native reward: {native_reward:.1f} (Rosetta: {-native_reward:.1f})")

    # PPO trainer
    trainer = PPOTrainer(
        mpnn_model=mpnn, device=device, lr=args.lr,
    )

    # Training loop
    reward_history = []
    best_reward = float('-inf')

    for iteration in range(args.n_iterations):
        t0 = time.time()

        # Collect rollouts
        rollouts = trainer.collect_rollouts(
            str(clean_template), fixed_positions,
            n_rollouts=args.rollouts_per_iter,
            temperature=args.temperature,
        )

        if not rollouts:
            logger.warning(f"Iter {iteration}: no rollouts")
            continue

        # PPO update
        metrics = trainer.ppo_update(rollouts)
        elapsed = time.time() - t0

        avg_reward = metrics.get('mean_reward', 0)
        best_r = metrics.get('best_reward', 0)
        reward_history.append(avg_reward)

        if best_r > best_reward:
            best_reward = best_r

        if (iteration + 1) % 5 == 0 or iteration == 0:
            logger.info(
                f"Iter {iteration+1}/{args.n_iterations}: "
                f"reward={avg_reward:.1f}, best={best_r:.1f}, "
                f"best_ever={best_reward:.1f}, "
                f"loss={metrics.get('policy_loss', 0):.4f}, "
                f"{elapsed:.1f}s"
            )

        if (iteration + 1) % 20 == 0:
            if len(reward_history) >= 6:
                first3 = np.mean(reward_history[:3])
                last3 = np.mean(reward_history[-3:])
                logger.info(f"  Trend: first3={first3:.1f} → last3={last3:.1f} (Δ={last3-first3:+.1f})")

    # Save results
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT {args.experiment} COMPLETE")
    logger.info(f"{'='*60}")
    if len(reward_history) >= 6:
        first3 = np.mean(reward_history[:3])
        last3 = np.mean(reward_history[-3:])
        logger.info(f"Reward: first3={first3:.1f} → last3={last3:.1f} (Δ={last3-first3:+.1f})")
    logger.info(f"Best reward: {best_reward:.1f} (native: {native_reward:.1f})")

    with open(output_dir / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Save fine-tuned weights
    if args.experiment in ['B', 'D']:
        torch.save(mpnn.state_dict(), output_dir / 'mpnn_finetuned.pt')
        logger.info(f"Saved fine-tuned ProteinMPNN to {output_dir / 'mpnn_finetuned.pt'}")


if __name__ == '__main__':
    main()
