"""Diagnostic RL experiments to prove optimization works.

Goal: clearly show that RL can optimize ACTUAL Rosetta scores (not just surrogates).
Track Rosetta on EVERY design, not just validation rounds.

Experiments:
1. SEQUENCE_ONLY: Fix backbone (partial_T=0), optimize ProteinMPNN temperature
   → Shows if sequence optimization alone improves Rosetta scores
2. BACKBONE_ONLY: Fix sequence strategy (temp=0.1), optimize partial_T with exploration
   → Shows if backbone variation helps
3. BOTH_EXPLORE: Optimize both with forced exploration (min partial_T=5, epsilon=0.4)
   → Shows if joint optimization with exploration beats individual
4. BOTH_EXPLOIT: Optimize both with exploitation (epsilon=0.1)
   → Shows if convergence happens with less exploration
5. EVOLUTIONARY: Keep top-K designs, re-generate from best backbone each round
   → Evolutionary strategy instead of independent rounds

Each experiment: 30 rounds, score ALL designs with Rosetta (no surrogate shortcut).
Track: Rosetta score per round (best, mean, worst), RMSD, catalytic recovery.

Usage:
    python experiments/pipeline/run_rl_diagnostics.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml
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
from typing import Dict, List, Optional

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


def score_design_with_rosetta(backbone_pdb: str, do_relax: bool = True) -> dict:
    """Score one design with actual Rosetta. Returns all energy terms."""
    scores = score_with_rosetta(backbone_pdb, do_relax=do_relax)
    return scores


def run_diagnostic_round(
    template_pdb: str,
    template,
    fixed_positions: List[int],
    pipeline_config: PipelineConfig,
    partial_T: int,
    temperature: float,
    n_designs: int,
    n_seqs: int,
    round_dir: Path,
    use_template_as_backbone: bool = False,
    rosetta_relax: bool = True,
) -> List[dict]:
    """Run one round: generate backbones → sequences → score with Rosetta."""
    round_dir.mkdir(exist_ok=True)

    pipeline_config.partial_T = partial_T
    pipeline_config.sampling_temperature = temperature
    pipeline_config.n_designs = n_designs
    pipeline_config.n_sequences_per_backbone = n_seqs

    # Generate backbones (or use template)
    t0 = time.time()
    if use_template_as_backbone or partial_T == 0:
        # Use template directly — sequence-only optimization
        backbone_pdbs = [template_pdb] * n_designs
        t_backbone = 0
    else:
        backbone_prefix = round_dir / 'backbone'
        backbone_pdbs = run_rfdiffusion(
            pipeline_config, template_pdb, template.length, str(backbone_prefix),
        )
        t_backbone = time.time() - t0

    if not backbone_pdbs:
        return []

    # Design sequences
    t0 = time.time()
    designs = []
    for bb_pdb in backbone_pdbs:
        sequences = run_proteinmpnn(
            pipeline_config, bb_pdb, fixed_positions, n_seqs,
        )
        for seq in sequences:
            designs.append({'backbone_pdb': bb_pdb, 'sequence': seq})
    t_sequence = time.time() - t0

    # Score ALL designs with Rosetta (the ground truth)
    t0 = time.time()
    results = []
    for i, design in enumerate(designs):
        rosetta_scores = score_design_with_rosetta(
            design['backbone_pdb'], do_relax=rosetta_relax
        )

        # Structural metrics
        bb = load_pdb(design['backbone_pdb'])
        min_len = min(bb.length, template.length)
        ca_gen = torch.tensor(bb.ca_coords[:min_len], dtype=torch.float32)
        ca_tmpl = torch.tensor(template.ca_coords[:min_len], dtype=torch.float32)
        rmsd_val, _, _ = kabsch_rmsd(ca_tmpl, ca_gen)
        clash_val = clash_score(torch.tensor(bb.coords, dtype=torch.float32))

        # Catalytic recovery
        cat_recovery = 1.0
        if fixed_positions and template.sequence and design['sequence']:
            matches = sum(
                1 for p in fixed_positions
                if p < len(design['sequence']) and p < len(template.sequence)
                and design['sequence'][p] == template.sequence[p]
            )
            cat_recovery = matches / max(len(fixed_positions), 1)

        result = {
            'backbone_pdb': design['backbone_pdb'],
            'sequence': design['sequence'],
            'template_rmsd': float(rmsd_val),
            'clash_score': float(clash_val),
            'catalytic_recovery': cat_recovery,
            'partial_T': partial_T,
            'temperature': temperature,
            **rosetta_scores,
        }
        results.append(result)

    t_scoring = time.time() - t0

    logger.info(
        f"  backbone={t_backbone:.0f}s, sequence={t_sequence:.0f}s, "
        f"rosetta={t_scoring:.0f}s ({len(results)} designs)"
    )

    return results


def run_experiment(
    name: str,
    template_pdb: str,
    template,
    constraint,
    fixed_positions: List[int],
    pipeline_config: PipelineConfig,
    output_dir: Path,
    n_rounds: int = 30,
    n_designs: int = 3,
    n_seqs: int = 3,
    rosetta_relax: bool = True,
    strategy: str = 'sequence_only',
) -> dict:
    """Run one diagnostic experiment."""
    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {name}")
    logger.info(f"Strategy: {strategy}, {n_rounds} rounds, {n_designs}×{n_seqs} designs")
    logger.info(f"{'='*60}")

    all_results = []
    round_stats = []  # per-round Rosetta summary

    # Track best design for evolutionary strategy
    best_backbone_pdb = template_pdb
    best_rosetta = float('inf')

    # Parameter state
    partial_T = 10
    temperature = 0.1
    epsilon = 0.3  # exploration rate

    for rnd in range(n_rounds):
        round_dir = exp_dir / f'round_{rnd:03d}'

        # Choose parameters based on strategy
        if strategy == 'sequence_only':
            partial_T = 0  # no backbone change
            # Explore temperature
            if np.random.random() < epsilon:
                temperature = np.random.uniform(0.05, 0.5)
            else:
                temperature = max(0.05, temperature + np.random.normal(0, 0.02))

        elif strategy == 'backbone_only':
            temperature = 0.1  # fixed sequence strategy
            # Explore partial_T with minimum floor
            if np.random.random() < epsilon:
                partial_T = np.random.randint(3, 20)
            else:
                partial_T = max(3, min(20, partial_T + np.random.randint(-2, 3)))

        elif strategy == 'both_explore':
            epsilon = 0.4  # high exploration
            if np.random.random() < epsilon:
                partial_T = np.random.randint(5, 20)  # min 5 — force exploration
                temperature = np.random.uniform(0.05, 0.4)
            else:
                partial_T = max(5, min(20, partial_T + np.random.randint(-2, 3)))
                temperature = max(0.05, min(0.4, temperature + np.random.normal(0, 0.03)))

        elif strategy == 'both_exploit':
            epsilon = 0.1  # low exploration
            if np.random.random() < epsilon:
                partial_T = np.random.randint(3, 15)
                temperature = np.random.uniform(0.05, 0.3)
            else:
                partial_T = max(3, min(15, partial_T + np.random.randint(-1, 2)))
                temperature = max(0.05, min(0.3, temperature + np.random.normal(0, 0.01)))

        elif strategy == 'evolutionary':
            # Use best backbone from previous round as template
            partial_T = np.random.randint(3, 12)
            temperature = np.random.uniform(0.05, 0.2)

        logger.info(f"\n--- Round {rnd+1}/{n_rounds} | T={partial_T}, temp={temperature:.3f} ---")

        # For evolutionary: use best backbone as starting point
        use_template = template_pdb
        if strategy == 'evolutionary' and rnd > 0 and best_backbone_pdb != template_pdb:
            use_template = best_backbone_pdb

        results = run_diagnostic_round(
            use_template, template, fixed_positions, pipeline_config,
            partial_T, temperature, n_designs, n_seqs, round_dir,
            use_template_as_backbone=(partial_T == 0),
            rosetta_relax=rosetta_relax,
        )

        if not results:
            continue

        # Rosetta scores for this round
        rosetta_scores = [r.get('rosetta_total', float('inf')) for r in results]
        valid_scores = [s for s in rosetta_scores if not np.isnan(s) and s != float('inf')]

        if valid_scores:
            round_best = min(valid_scores)
            round_mean = np.mean(valid_scores)
            round_worst = max(valid_scores)

            # Track best overall
            if round_best < best_rosetta:
                best_rosetta = round_best
                # Find the best design and save its backbone
                best_idx = rosetta_scores.index(round_best)
                best_backbone_pdb = results[best_idx]['backbone_pdb']

            stats = {
                'round': rnd,
                'partial_T': partial_T,
                'temperature': temperature,
                'rosetta_best': round_best,
                'rosetta_mean': round_mean,
                'rosetta_worst': round_worst,
                'best_ever': best_rosetta,
                'n_designs': len(valid_scores),
                'mean_rmsd': np.mean([r['template_rmsd'] for r in results]),
                'mean_cat_recovery': np.mean([r['catalytic_recovery'] for r in results]),
            }
            round_stats.append(stats)

            logger.info(
                f"  Rosetta: best={round_best:.1f}, mean={round_mean:.1f}, "
                f"best_ever={best_rosetta:.1f}, RMSD={stats['mean_rmsd']:.2f}Å"
            )
        else:
            logger.warning(f"  No valid Rosetta scores this round")

        all_results.extend(results)

        # Save round results
        with open(round_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Save experiment summary
    with open(exp_dir / 'round_stats.json', 'w') as f:
        json.dump(round_stats, f, indent=2)
    with open(exp_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    if round_stats:
        first3 = np.mean([s['rosetta_best'] for s in round_stats[:3]])
        last3 = np.mean([s['rosetta_best'] for s in round_stats[-3:]])
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {name} COMPLETE")
        logger.info(f"  Rounds: {len(round_stats)}")
        logger.info(f"  Rosetta best: first3_avg={first3:.1f} → last3_avg={last3:.1f} (Δ={last3-first3:+.1f})")
        logger.info(f"  Best ever: {best_rosetta:.1f}")
        logger.info(f"{'='*60}")

    return {
        'name': name,
        'strategy': strategy,
        'round_stats': round_stats,
        'best_rosetta': best_rosetta,
    }


def main():
    parser = argparse.ArgumentParser(description='RL diagnostic experiments')
    parser.add_argument('--template', required=True)
    parser.add_argument('--constraint', required=True)
    parser.add_argument('--n-rounds', type=int, default=30)
    parser.add_argument('--n-designs', type=int, default=3)
    parser.add_argument('--n-seqs', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default='results/diagnostics')
    parser.add_argument('--rfdiffusion-dir', type=str, default='~/RFdiffusion')
    parser.add_argument('--proteinmpnn-dir', type=str, default='~/ProteinMPNN')
    parser.add_argument('--no-relax', action='store_true', help='Skip Rosetta relax (faster but noisier)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
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

    # Score native template first (baseline)
    logger.info("=== Scoring native template (baseline) ===")
    native_scores = score_with_rosetta(str(clean_template), do_relax=not args.no_relax)
    native_rosetta = native_scores.get('rosetta_total', float('nan'))
    logger.info(f"Native template Rosetta: {native_rosetta:.1f} REU")

    # Run experiments
    experiments = [
        ('1_sequence_only', 'sequence_only'),
        ('2_backbone_only', 'backbone_only'),
        ('3_both_explore', 'both_explore'),
        ('4_both_exploit', 'both_exploit'),
        ('5_evolutionary', 'evolutionary'),
    ]

    all_summaries = []
    for name, strategy in experiments:
        summary = run_experiment(
            name=name,
            template_pdb=str(clean_template),
            template=template,
            constraint=constraint,
            fixed_positions=fixed_positions,
            pipeline_config=pipeline_config,
            output_dir=output_dir,
            n_rounds=args.n_rounds,
            n_designs=args.n_designs,
            n_seqs=args.n_seqs,
            rosetta_relax=not args.no_relax,
            strategy=strategy,
        )
        all_summaries.append(summary)

    # Final comparison
    logger.info(f"\n{'='*60}")
    logger.info(f"DIAGNOSTIC RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Native template: {native_rosetta:.1f} REU")
    logger.info(f"")
    for s in all_summaries:
        if s['round_stats']:
            first3 = np.mean([r['rosetta_best'] for r in s['round_stats'][:3]])
            last3 = np.mean([r['rosetta_best'] for r in s['round_stats'][-3:]])
            logger.info(
                f"  {s['name']:25s} | best={s['best_rosetta']:8.1f} | "
                f"first3={first3:8.1f} → last3={last3:8.1f} | Δ={last3-first3:+.1f}"
            )

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'native_rosetta': native_rosetta,
            'experiments': [{
                'name': s['name'],
                'strategy': s['strategy'],
                'best_rosetta': s['best_rosetta'],
            } for s in all_summaries],
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
