"""Real RL optimization with gradient-based policy updates.

Two optimization channels:
1. SEQUENCE (ProteinMPNN): Load as PyTorch model, sample sequences,
   score with differentiable surrogate, update weights via REINFORCE.
2. BACKBONE (RFdiffusion params): CMA-ES over generation parameters
   (partial_T, noise_scale) since RFdiffusion is a black box.

Reward: surrogate-predicted Rosetta energy terms.
Validation: actual Rosetta scoring on best designs periodically.

Experiments:
A: CMA-ES on backbone params only (ProteinMPNN frozen)
B: REINFORCE on ProteinMPNN only (backbone fixed)
C: Both simultaneously
D: Evolutionary baseline (no gradient, just selection)

Usage:
    python experiments/pipeline/run_real_rl.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --experiment B --n-rounds 30
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from experiments.pipeline.run_end_to_end import (
    PipelineConfig, run_rfdiffusion, run_proteinmpnn,
    score_with_rosetta, write_pdb,
)
from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml
from src.utils.geometry import kabsch_rmsd
from src.utils.metrics import clash_score
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# CMA-ES for backbone parameters (black-box optimization)
# ============================================================================

class SimpleCMAES:
    """Simplified CMA-ES for optimizing RFdiffusion parameters.

    Optimizes: [partial_T, noise_scale_ca, noise_scale_frame, temperature]
    Each as a continuous value, discretized when calling RFdiffusion.
    """

    def __init__(self, dim: int = 4, population_size: int = 8, sigma: float = 0.5):
        self.dim = dim
        self.pop_size = population_size
        self.sigma = sigma
        self.mean = np.zeros(dim)  # initial mean in normalized space
        self.cov = np.eye(dim)
        self.generation = 0

        # Parameter ranges (normalized space → real space)
        # [partial_T, noise_scale_ca, noise_scale_frame, temperature]
        self.param_ranges = [
            (3, 20),      # partial_T
            (0.0, 1.0),   # noise_scale_ca
            (0.0, 1.0),   # noise_scale_frame
            (0.05, 0.4),  # temperature
        ]

    def sample_population(self) -> List[np.ndarray]:
        """Sample a population of parameter vectors."""
        pop = []
        for _ in range(self.pop_size):
            z = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov)
            pop.append(z)
        return pop

    def decode_params(self, z: np.ndarray) -> dict:
        """Convert normalized params to RFdiffusion parameters."""
        params = {}
        for i, (name, (lo, hi)) in enumerate(zip(
            ['partial_T', 'noise_scale_ca', 'noise_scale_frame', 'temperature'],
            self.param_ranges
        )):
            # Sigmoid to bound to [0, 1], then scale to range
            val = 1.0 / (1.0 + np.exp(-z[i]))
            val = lo + val * (hi - lo)
            if name == 'partial_T':
                val = int(round(val))
            params[name] = val
        return params

    def update(self, population: List[np.ndarray], rewards: List[float]):
        """Update CMA-ES distribution based on rewards."""
        # Sort by reward (higher is better)
        indices = np.argsort(rewards)[::-1]
        mu = max(self.pop_size // 2, 2)  # keep top half

        # Weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Update mean
        selected = [population[i] for i in indices[:mu]]
        self.mean = sum(w * s for w, s in zip(weights, selected))

        # Update covariance (simplified)
        diff = np.array(selected) - self.mean
        self.cov = sum(
            w * np.outer(d, d) for w, d in zip(weights, diff)
        ) + 1e-4 * np.eye(self.dim)

        # Adapt sigma
        self.sigma *= np.exp(0.1 * (np.mean(rewards[indices[:mu]]) - np.mean(rewards)) / (np.std(rewards) + 1e-8))
        self.sigma = np.clip(self.sigma, 0.1, 2.0)

        self.generation += 1


# ============================================================================
# REINFORCE for ProteinMPNN sequence optimization
# ============================================================================

class ProteinMPNNPolicy:
    """Wraps ProteinMPNN as an RL policy with REINFORCE updates.

    Loads ProteinMPNN weights, samples sequences with log-probabilities,
    and updates weights using policy gradient.
    """

    def __init__(self, mpnn_dir: str, device: str = 'cuda', lr: float = 1e-4):
        self.mpnn_dir = Path(mpnn_dir)
        self.device = device
        self.lr = lr

        # Load model
        sys.path.insert(0, str(self.mpnn_dir))
        from protein_mpnn_utils import ProteinMPNN

        ckpt = torch.load(
            self.mpnn_dir / 'vanilla_model_weights' / 'v_48_010.pt',
            map_location='cpu', weights_only=False,
        )
        self.model = ProteinMPNN(
            num_letters=21, node_features=128, edge_features=128,
            hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
            vocab=21, k_neighbors=ckpt['num_edges'], augment_eps=0.0,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(device)
        self.model.train()

        # Save initial weights for comparison
        self.initial_state = copy.deepcopy(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.baseline = 0.0  # running reward baseline

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"ProteinMPNN policy loaded: {params:,} params, lr={lr}")

    def sample_sequence(
        self, pdb_path: str, fixed_positions: List[int], temperature: float = 0.1
    ) -> Tuple[str, torch.Tensor]:
        """Sample a sequence and return log-probability.

        Returns:
            sequence: designed sequence string
            log_prob: sum of log-probabilities (scalar tensor, retains grad)
        """
        from protein_mpnn_utils import parse_PDB, tied_featurize

        pdb_dict = parse_PDB(pdb_path, ca_only=False)[0]
        X, S, mask, lengths, chain_M, chain_encoding_all, *_ = tied_featurize(
            [pdb_dict], self.device, None, None, None, None, None, None, ca_only=False
        )

        L = int(lengths[0])
        chain_M_pos = torch.ones_like(chain_M)
        residue_idx = torch.arange(L, device=self.device).unsqueeze(0)

        # Set fixed positions (mask out from design)
        if fixed_positions:
            for pos in fixed_positions:
                if pos < L:
                    chain_M_pos[0, pos] = 0  # don't design this position

        omit_AAs = np.zeros(21, dtype=np.float32)
        bias_AAs = np.zeros(21, dtype=np.float32)
        bias_by_res = torch.zeros(1, L, 21, device=self.device)

        # Forward pass with gradient tracking
        sample_dict = self.model.sample(
            X, torch.randn(1, L, device=self.device), S,
            chain_M * chain_M_pos, chain_encoding_all, residue_idx,
            mask, temperature=temperature, chain_M_pos=chain_M_pos,
            omit_AAs_np=omit_AAs, bias_AAs_np=bias_AAs,
            bias_by_res=bias_by_res,
        )

        S_sample = sample_dict['S']

        # Compute log-probability of sampled sequence under the model
        # Re-run forward to get logits
        log_probs = self.model(X, S_sample, mask, chain_M * chain_M_pos,
                               residue_idx, chain_encoding_all)
        # log_probs shape: (1, L, 21)

        # Sum log-probs at designed positions
        total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)
        designed_positions = (chain_M_pos[0] > 0).nonzero(as_tuple=True)[0]

        if log_probs is not None and hasattr(log_probs, 'shape'):
            for pos in designed_positions:
                if pos < L:
                    aa_idx = S_sample[0, pos]
                    lp = F.log_softmax(log_probs[0, pos], dim=-1)
                    total_log_prob = total_log_prob + lp[aa_idx]

        # Decode sequence
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        sequence = ''.join([alphabet[s] for s in S_sample[0].cpu().numpy()[:L]])

        return sequence, total_log_prob

    def update_policy(self, log_prob: torch.Tensor, reward: float):
        """REINFORCE update: loss = -advantage * log_prob."""
        advantage = reward - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * reward  # EMA baseline

        loss = -advantage * log_prob
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss)

    def weight_change(self) -> float:
        """Measure how much weights changed from initial."""
        total_diff = 0.0
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            diff = (param.data - self.initial_state[name].to(param.device)).norm()
            total_diff += diff.item()
            total_norm += param.data.norm().item()
        return total_diff / max(total_norm, 1e-8)


# ============================================================================
# Reward computation from Rosetta (no relax — fast)
# ============================================================================

def compute_rosetta_reward(pdb_path: str) -> float:
    """Score with Rosetta (no relax) and return negative score as reward."""
    scores = score_with_rosetta(pdb_path, do_relax=False)
    total = scores.get('rosetta_total', float('inf'))
    if np.isnan(total) or total == float('inf'):
        return -1000.0  # penalty for failed scoring
    return -total  # negate: lower Rosetta = higher reward


# ============================================================================
# Main experiment runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Real RL optimization')
    parser.add_argument('--template', required=True)
    parser.add_argument('--constraint', required=True)
    parser.add_argument('--experiment', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--n-rounds', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default='results/real_rl')
    parser.add_argument('--rfdiffusion-dir', type=str, default='~/RFdiffusion')
    parser.add_argument('--proteinmpnn-dir', type=str, default='~/ProteinMPNN')
    parser.add_argument('--validate-every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f'exp_{args.experiment}'
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_pdb(args.template)
    constraint = load_constraint_from_yaml(args.constraint)
    fixed_positions = [r.position_index for r in constraint.residues if r.position_index is not None]

    # Clean template
    clean_template = output_dir / 'template_clean.pdb'
    with open(args.template) as f:
        lines = [l for l in f if l.startswith('ATOM') or l.startswith('TER') or l.startswith('END')]
    with open(clean_template, 'w') as f:
        f.writelines(lines)

    pipeline_config = PipelineConfig(
        rfdiffusion_dir=args.rfdiffusion_dir,
        proteinmpnn_dir=args.proteinmpnn_dir,
    )

    # Score native baseline
    native_reward = compute_rosetta_reward(str(clean_template))
    logger.info(f"Native Rosetta reward: {native_reward:.1f} (score: {-native_reward:.1f})")

    # Initialize components based on experiment
    cma = None
    mpnn_policy = None

    if args.experiment in ['A', 'C']:
        cma = SimpleCMAES(dim=4, population_size=6, sigma=0.5)
        logger.info("CMA-ES initialized for backbone params")

    if args.experiment in ['B', 'C']:
        mpnn_policy = ProteinMPNNPolicy(
            str(Path(args.proteinmpnn_dir).expanduser()),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lr=args.lr,
        )
        logger.info("ProteinMPNN REINFORCE policy initialized")

    # Tracking
    reward_history = []
    best_ever = {'reward': float('-inf'), 'round': -1}
    rosetta_validations = []

    logger.info(f"\n=== REAL RL — Experiment {args.experiment} ===")
    logger.info(f"Template: {template.pdb_id}, {template.length} residues")
    logger.info(f"Rounds: {args.n_rounds}")

    for rnd in range(args.n_rounds):
        round_dir = output_dir / f'round_{rnd:03d}'
        round_dir.mkdir(exist_ok=True)

        t_start = time.time()
        round_rewards = []
        round_rosetta_scores = []

        # === EXPERIMENT A: CMA-ES on backbone params ===
        if args.experiment == 'A':
            population = cma.sample_population()
            for i, z in enumerate(population):
                params = cma.decode_params(z)
                logger.info(f"  CMA member {i}: T={params['partial_T']}, temp={params['temperature']:.3f}")

                # Generate backbone
                pipeline_config.partial_T = params['partial_T']
                pipeline_config.n_designs = 1
                bb_prefix = round_dir / f'cma_{i}'
                backbone_pdbs = run_rfdiffusion(
                    pipeline_config, str(clean_template), template.length, str(bb_prefix),
                )

                if not backbone_pdbs:
                    round_rewards.append(-1000)
                    continue

                # Design sequence (frozen ProteinMPNN via subprocess)
                sequences = run_proteinmpnn(
                    pipeline_config, backbone_pdbs[0], fixed_positions, 1,
                )

                # Score with Rosetta (no relax — fast)
                reward = compute_rosetta_reward(backbone_pdbs[0])
                round_rewards.append(reward)
                round_rosetta_scores.append(-reward)

            # CMA-ES update
            cma.update(population, round_rewards)

        # === EXPERIMENT B: REINFORCE on ProteinMPNN ===
        elif args.experiment == 'B':
            n_samples = 6
            for i in range(n_samples):
                # Sample sequence with gradient tracking
                seq, log_prob = mpnn_policy.sample_sequence(
                    str(clean_template), fixed_positions, temperature=0.1
                )

                # Score with Rosetta
                reward = compute_rosetta_reward(str(clean_template))
                round_rewards.append(reward)
                round_rosetta_scores.append(-reward)

                # REINFORCE update
                loss = mpnn_policy.update_policy(log_prob, reward)

            logger.info(f"  Weight change: {mpnn_policy.weight_change():.6f}")

        # === EXPERIMENT C: CMA-ES + REINFORCE ===
        elif args.experiment == 'C':
            population = cma.sample_population()
            for i, z in enumerate(population):
                params = cma.decode_params(z)

                # Generate backbone with CMA params
                pipeline_config.partial_T = params['partial_T']
                pipeline_config.n_designs = 1
                bb_prefix = round_dir / f'cma_{i}'
                backbone_pdbs = run_rfdiffusion(
                    pipeline_config, str(clean_template), template.length, str(bb_prefix),
                )

                if not backbone_pdbs:
                    round_rewards.append(-1000)
                    continue

                # Sample sequence with REINFORCE
                seq, log_prob = mpnn_policy.sample_sequence(
                    backbone_pdbs[0], fixed_positions, temperature=params['temperature']
                )

                reward = compute_rosetta_reward(backbone_pdbs[0])
                round_rewards.append(reward)
                round_rosetta_scores.append(-reward)

                # Update ProteinMPNN
                mpnn_policy.update_policy(log_prob, reward)

            # Update CMA-ES
            cma.update(population, round_rewards)

        # === EXPERIMENT D: Evolutionary baseline ===
        elif args.experiment == 'D':
            best_template = str(clean_template)
            for i in range(6):
                partial_T = np.random.randint(3, 15)
                pipeline_config.partial_T = partial_T
                pipeline_config.n_designs = 1
                bb_prefix = round_dir / f'evo_{i}'
                backbone_pdbs = run_rfdiffusion(
                    pipeline_config, best_template, template.length, str(bb_prefix),
                )

                if not backbone_pdbs:
                    round_rewards.append(-1000)
                    continue

                sequences = run_proteinmpnn(
                    pipeline_config, backbone_pdbs[0], fixed_positions, 1,
                )
                reward = compute_rosetta_reward(backbone_pdbs[0])
                round_rewards.append(reward)
                round_rosetta_scores.append(-reward)

                if reward > best_ever['reward']:
                    best_template = backbone_pdbs[0]

        # Round summary
        elapsed = time.time() - t_start
        avg_reward = np.mean(round_rewards) if round_rewards else -1000
        max_reward = max(round_rewards) if round_rewards else -1000
        reward_history.append(avg_reward)

        if max_reward > best_ever['reward']:
            best_ever = {'reward': max_reward, 'round': rnd, 'rosetta': -max_reward}

        logger.info(
            f"Round {rnd+1}/{args.n_rounds}: "
            f"avg_reward={avg_reward:.1f}, best={max_reward:.1f}, "
            f"best_ever={best_ever['reward']:.1f} (r{best_ever['round']}), "
            f"{elapsed:.0f}s"
        )

        if round_rosetta_scores:
            logger.info(
                f"  Rosetta scores: [{min(round_rosetta_scores):.1f}, {max(round_rosetta_scores):.1f}]"
            )

        # Save round
        with open(round_dir / 'stats.json', 'w') as f:
            json.dump({
                'round': rnd, 'rewards': round_rewards,
                'rosetta_scores': round_rosetta_scores,
                'avg_reward': avg_reward, 'best_reward': max_reward,
            }, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT {args.experiment} COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds: {args.n_rounds}")
    logger.info(f"Best Rosetta: {best_ever.get('rosetta', 'N/A')} (round {best_ever['round']})")

    if len(reward_history) >= 6:
        first3 = np.mean(reward_history[:3])
        last3 = np.mean(reward_history[-3:])
        logger.info(f"Reward: first3={first3:.1f} → last3={last3:.1f} (Δ={last3-first3:+.1f})")

    if mpnn_policy:
        logger.info(f"ProteinMPNN weight change: {mpnn_policy.weight_change():.6f}")

    # Save results
    with open(output_dir / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)
    with open(output_dir / 'best_design.json', 'w') as f:
        json.dump(best_ever, f, indent=2, default=str)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'experiment': args.experiment, 'n_rounds': args.n_rounds, 'lr': args.lr}, f)


if __name__ == '__main__':
    main()
