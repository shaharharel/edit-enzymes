"""RL-guided enzyme design optimization.

Optimizes design parameters (partial_T, temperature, noise) using
surrogate scores as reward, with periodic Rosetta validation.

Progressive unfreezing experiments:
A: Frozen generators — optimize search params only
B: Unfreeze ProteinMPNN — optimize sequence sampling
C: Unfreeze RFdiffusion — optimize backbone generation
D: Unfreeze both — full end-to-end

Usage:
    python experiments/pipeline/run_rl_optimization.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --n-rounds 50 \
        --experiment A
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from experiments.pipeline.run_end_to_end import (
    PipelineConfig, run_rfdiffusion, run_proteinmpnn,
    score_with_surrogate, score_with_rosetta,
    write_pdb, _compute_esm_embedding,
)
from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLConfig:
    """RL optimization configuration."""
    # Pipeline
    template_pdb: str = ''
    constraint_yaml: str = ''
    rfdiffusion_dir: str = '~/RFdiffusion'
    proteinmpnn_dir: str = '~/ProteinMPNN'
    surrogate_dir: str = 'results/surrogates'
    output_dir: str = 'results/rl'

    # Search space (what RL optimizes)
    partial_T_range: Tuple[int, int] = (3, 20)
    temperature_range: Tuple[float, float] = (0.05, 0.5)
    n_designs_per_round: int = 5
    n_sequences_per_backbone: int = 4

    # RL params
    n_rounds: int = 50
    top_k: int = 3  # keep top K per round
    exploration_rate: float = 0.3  # probability of random exploration
    learning_rate: float = 0.1  # for parameter updates

    # Validation
    rosetta_validate_every: int = 10  # run Rosetta every N rounds
    rosetta_relax: bool = True

    # Experiment type
    experiment: str = 'A'  # A, B, C, D


class SearchParamOptimizer:
    """Optimizes search parameters (partial_T, temperature) using bandit-style RL.

    Maintains a distribution over parameter values and updates based on rewards.
    Simple but effective for this low-dimensional search space.
    """

    def __init__(self, config: RLConfig):
        self.config = config

        # Parameter distributions (mean, std)
        self.partial_T_mean = 10.0
        self.partial_T_std = 3.0
        self.temp_mean = 0.15
        self.temp_std = 0.05

        # History for learning
        self.history = []  # (params, reward) pairs

    def sample_params(self) -> dict:
        """Sample search parameters."""
        if np.random.random() < self.config.exploration_rate:
            # Explore: uniform random
            partial_T = np.random.randint(*self.config.partial_T_range)
            temperature = np.random.uniform(*self.config.temperature_range)
        else:
            # Exploit: sample from learned distribution
            partial_T = int(np.clip(
                np.random.normal(self.partial_T_mean, self.partial_T_std),
                *self.config.partial_T_range
            ))
            temperature = float(np.clip(
                np.random.normal(self.temp_mean, self.temp_std),
                *self.config.temperature_range
            ))

        return {'partial_T': partial_T, 'temperature': temperature}

    def update(self, params: dict, reward: float):
        """Update parameter distributions based on reward."""
        self.history.append((params, reward))

        # Use exponential moving average toward good parameters
        if reward > self._baseline_reward():
            lr = self.config.learning_rate
            self.partial_T_mean += lr * (params['partial_T'] - self.partial_T_mean)
            self.temp_mean += lr * (params['temperature'] - self.temp_mean)

            # Reduce std (exploit more)
            self.partial_T_std *= 0.98
            self.temp_std *= 0.98
        else:
            # Increase std (explore more)
            self.partial_T_std = min(self.partial_T_std * 1.01, 5.0)
            self.temp_std = min(self.temp_std * 1.01, 0.1)

    def _baseline_reward(self) -> float:
        """Running average reward as baseline."""
        if not self.history:
            return 0.0
        recent = [r for _, r in self.history[-10:]]
        return np.mean(recent)

    def get_stats(self) -> dict:
        return {
            'partial_T_mean': self.partial_T_mean,
            'partial_T_std': self.partial_T_std,
            'temp_mean': self.temp_mean,
            'temp_std': self.temp_std,
            'n_updates': len(self.history),
            'baseline_reward': self._baseline_reward(),
        }


def compute_reward(scores: dict) -> float:
    """Compute scalar reward from surrogate scores.

    Reward = negative of predicted total ΔΔG (lower ΔΔG = higher reward)
    + bonuses for good individual terms.
    """
    reward = 0.0
    n = 0

    # Primary: total ΔΔG (lower is better → negate)
    if 'surrogate_total_ddg' in scores and not np.isnan(scores['surrogate_total_ddg']):
        reward -= scores['surrogate_total_ddg']
        n += 1

    # Bonus for favorable individual terms
    term_weights = {
        'surrogate_d_fa_atr': -0.5,  # attraction: more negative = better
        'surrogate_d_fa_rep': -1.0,  # repulsion: penalize high values
        'surrogate_d_fa_sol': -0.3,  # solvation
        'surrogate_d_fa_elec': -0.3,  # electrostatics
        'surrogate_d_hbond_sc': -0.5,  # H-bonds: more negative = better
    }

    for term, weight in term_weights.items():
        if term in scores and not np.isnan(scores[term]):
            reward += weight * scores[term]
            n += 1

    return reward / max(n, 1)


def run_one_round(
    config: RLConfig,
    pipeline_config: PipelineConfig,
    template,
    constraint,
    fixed_positions: List[int],
    params: dict,
    round_dir: Path,
) -> List[dict]:
    """Run one design round with given parameters."""
    pipeline_config.partial_T = params['partial_T']
    pipeline_config.sampling_temperature = params['temperature']
    pipeline_config.n_designs = config.n_designs_per_round
    pipeline_config.n_sequences_per_backbone = config.n_sequences_per_backbone

    # Clean template
    clean_template = round_dir / 'template_clean.pdb'
    with open(config.template_pdb) as f:
        lines = [l for l in f if l.startswith('ATOM') or l.startswith('TER') or l.startswith('END')]
    with open(clean_template, 'w') as f:
        f.writelines(lines)

    # Generate backbones
    t0 = time.time()
    backbone_prefix = round_dir / 'backbone'
    backbone_pdbs = run_rfdiffusion(
        pipeline_config, str(clean_template), template.length, str(backbone_prefix),
    )
    t_backbone = time.time() - t0

    if not backbone_pdbs:
        return []

    # Design sequences
    t0 = time.time()
    designs = []
    for bb_pdb in backbone_pdbs:
        sequences = run_proteinmpnn(
            pipeline_config, bb_pdb, fixed_positions,
            config.n_sequences_per_backbone,
        )
        for seq in sequences:
            designs.append({'backbone_pdb': bb_pdb, 'sequence': seq})
    t_sequence = time.time() - t0

    # Score with surrogates (fast)
    t0 = time.time()
    results = []
    for design in designs:
        surr_scores = score_with_surrogate(
            design['backbone_pdb'], design['sequence'],
            surrogate_dir=config.surrogate_dir,
        )

        # Structural metrics
        bb = load_pdb(design['backbone_pdb'])
        from src.utils.geometry import kabsch_rmsd
        min_len = min(bb.length, template.length)
        ca_gen = torch.tensor(bb.ca_coords[:min_len], dtype=torch.float32)
        ca_tmpl = torch.tensor(template.ca_coords[:min_len], dtype=torch.float32)
        rmsd, _, _ = kabsch_rmsd(ca_tmpl, ca_gen)

        reward = compute_reward(surr_scores)

        result = {
            'backbone_pdb': design['backbone_pdb'],
            'sequence': design['sequence'][:30] + '...',
            'full_sequence': design['sequence'],
            'template_rmsd': float(rmsd),
            'reward': reward,
            'partial_T': params['partial_T'],
            'temperature': params['temperature'],
            **surr_scores,
        }
        results.append(result)
    t_scoring = time.time() - t0

    logger.info(
        f"  backbone={t_backbone:.0f}s, sequence={t_sequence:.0f}s, "
        f"scoring={t_scoring:.0f}s, designs={len(results)}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='RL-guided enzyme design')
    parser.add_argument('--template', required=True)
    parser.add_argument('--constraint', required=True)
    parser.add_argument('--n-rounds', type=int, default=50)
    parser.add_argument('--n-designs', type=int, default=5)
    parser.add_argument('--experiment', choices=['A', 'B', 'C', 'D'], default='A')
    parser.add_argument('--output-dir', type=str, default='results/rl')
    parser.add_argument('--rfdiffusion-dir', type=str, default='~/RFdiffusion')
    parser.add_argument('--proteinmpnn-dir', type=str, default='~/ProteinMPNN')
    parser.add_argument('--surrogate-dir', type=str, default='results/surrogates')
    parser.add_argument('--validate-every', type=int, default=10)
    args = parser.parse_args()

    config = RLConfig(
        template_pdb=args.template,
        constraint_yaml=args.constraint,
        n_rounds=args.n_rounds,
        n_designs_per_round=args.n_designs,
        experiment=args.experiment,
        output_dir=args.output_dir,
        rfdiffusion_dir=args.rfdiffusion_dir,
        proteinmpnn_dir=args.proteinmpnn_dir,
        surrogate_dir=args.surrogate_dir,
        rosetta_validate_every=args.validate_every,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template and constraints
    template = load_pdb(config.template_pdb)
    constraint = load_constraint_from_yaml(config.constraint_yaml)
    fixed_positions = [r.position_index for r in constraint.residues if r.position_index is not None]

    logger.info(f"=== RL Optimization: Experiment {config.experiment} ===")
    logger.info(f"Template: {template.pdb_id}, {template.length} residues")
    logger.info(f"Catalytic: {len(fixed_positions)} fixed positions")
    logger.info(f"Rounds: {config.n_rounds}, designs/round: {config.n_designs_per_round}")

    # Pipeline config (shared)
    pipeline_config = PipelineConfig(
        rfdiffusion_dir=config.rfdiffusion_dir,
        proteinmpnn_dir=config.proteinmpnn_dir,
        surrogate_checkpoint=config.surrogate_dir,
    )

    # Initialize RL optimizer
    optimizer = SearchParamOptimizer(config)

    # Track results
    all_results = []
    best_ever = {'reward': float('-inf')}
    reward_history = []

    for round_num in range(config.n_rounds):
        round_dir = output_dir / f'round_{round_num:03d}'
        round_dir.mkdir(exist_ok=True)

        # Sample parameters
        params = optimizer.sample_params()

        logger.info(f"\n--- Round {round_num+1}/{config.n_rounds} | "
                    f"partial_T={params['partial_T']}, temp={params['temperature']:.3f} ---")

        # Run design round
        results = run_one_round(
            config, pipeline_config, template, constraint,
            fixed_positions, params, round_dir,
        )

        if not results:
            logger.warning("No designs generated, skipping")
            continue

        # Find best in this round
        round_best = max(results, key=lambda x: x['reward'])
        round_avg_reward = np.mean([r['reward'] for r in results])

        # Update RL
        optimizer.update(params, round_avg_reward)
        reward_history.append(round_avg_reward)

        # Track overall best
        if round_best['reward'] > best_ever['reward']:
            best_ever = round_best.copy()
            best_ever['round'] = round_num

        logger.info(
            f"  Best reward: {round_best['reward']:.3f}, "
            f"avg: {round_avg_reward:.3f}, "
            f"RMSD: {round_best['template_rmsd']:.2f}Å"
        )
        stats = optimizer.get_stats()
        logger.info(
            f"  Params: T_mean={stats['partial_T_mean']:.1f}±{stats['partial_T_std']:.1f}, "
            f"temp_mean={stats['temp_mean']:.3f}±{stats['temp_std']:.3f}"
        )

        all_results.extend(results)

        # Periodic Rosetta validation
        if (round_num + 1) % config.rosetta_validate_every == 0:
            logger.info(f"\n  === Rosetta Validation (round {round_num+1}) ===")
            rosetta_scores = score_with_rosetta(
                round_best['backbone_pdb'], do_relax=config.rosetta_relax
            )
            if 'rosetta_total' in rosetta_scores:
                logger.info(f"  Rosetta total: {rosetta_scores['rosetta_total']:.1f}")
                logger.info(f"  Surrogate predicted: {round_best.get('surrogate_total_ddg', 'N/A')}")

        # Save round results
        with open(round_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RL OPTIMIZATION COMPLETE — Experiment {config.experiment}")
    logger.info(f"{'='*60}")
    logger.info(f"Rounds: {config.n_rounds}")
    logger.info(f"Total designs: {len(all_results)}")
    logger.info(f"Best ever: reward={best_ever['reward']:.3f}, "
                f"RMSD={best_ever.get('template_rmsd', 'N/A')}, "
                f"round={best_ever.get('round', 'N/A')}")

    # Reward trend
    if reward_history:
        first_5 = np.mean(reward_history[:5])
        last_5 = np.mean(reward_history[-5:])
        logger.info(f"Reward trend: first 5 avg={first_5:.3f} → last 5 avg={last_5:.3f}")
        improvement = last_5 - first_5
        logger.info(f"Improvement: {improvement:+.3f} ({'↑' if improvement > 0 else '↓'})")

    # Save everything
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / 'best_design.json', 'w') as f:
        json.dump(best_ever, f, indent=2, default=str)
    with open(output_dir / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)
    with open(output_dir / 'optimizer_stats.json', 'w') as f:
        json.dump(optimizer.get_stats(), f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
