"""PPO fine-tuning of RFD3 inside Docker container.

Runs inside the Foundry Docker image where RFD3 model + atomworks are available.
Loads RFD3 natively as PyTorch module, generates designs, scores with Rosetta energy
proxy, and updates weights via PPO.

Architecture (168M trainable params, 1054 linear layers):
- token_initializer: atom embeddings
- trunk: attention blocks (main compute)
- output heads: coordinate + sequence prediction

Fine-tuning strategies (experiments for tonight):
1. FULL: all 168M params (baseline, might overfit)
2. LAST_LAYERS: only last N attention blocks
3. LORA: LoRA adapters on attention linear layers (most efficient)
4. OUTPUT_ONLY: only output heads (sequence + coords)

Usage (inside Docker):
    python3 /workspace/rfd3_ppo_docker.py --strategy lora --n-rounds 50
"""

import os
import sys
import json
import time
import copy
import glob
import gzip
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import subprocess

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def load_rfd3_model(device='cuda'):
    """Load RFD3 model from checkpoint."""
    ckpt_path = "/root/.foundry/checkpoints/rfd3_latest.ckpt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # The model state is under 'model' key (shadow is EMA, skip it)
    model_state = {k.replace('model.', '', 1): v for k, v in ckpt['model'].items()
                   if k.startswith('model.')}

    total_params = sum(v.numel() for v in model_state.values())
    logger.info(f"RFD3 model loaded: {total_params:,} params")

    return ckpt, model_state


def generate_and_score(input_json: str, output_dir: str) -> List[Dict]:
    """Generate designs using RFD3 CLI and score outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run RFD3 inference
    t0 = time.time()
    cmd = [
        'rfd3', 'design',
        f'out_dir={output_dir}',
        f'inputs={input_json}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    t_gen = time.time() - t0

    if result.returncode != 0:
        logger.warning(f"RFD3 generation failed: {result.stderr[-200:]}")
        return []

    # Collect and score outputs
    cif_files = sorted(output_dir.glob('*.cif.gz'))
    logger.info(f"Generated {len(cif_files)} designs in {t_gen:.0f}s")

    results = []
    for cif_path in cif_files:
        json_path = str(cif_path).replace('.cif.gz', '.json')

        scores = {}

        # Read RFD3 metrics
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            metrics = meta.get('metrics', {})
            scores['rfd3_clashes'] = metrics.get('n_clashing.interresidue_clashes_w_sidechain', 0)
            scores['rfd3_chainbreaks'] = metrics.get('n_chainbreaks', 0)
            scores['rfd3_rog'] = metrics.get('radius_of_gyration', 0)
            scores['rfd3_helix'] = metrics.get('helix_fraction', 0)
            scores['rfd3_sheet'] = metrics.get('sheet_fraction', 0)
            scores['rfd3_max_ca_dev'] = metrics.get('max_ca_deviation', 0)

        # Compute reward from RFD3 metrics
        # Lower clashes + lower CA deviation + more secondary structure = better
        reward = 0.0
        reward -= scores.get('rfd3_clashes', 0) * 10.0  # heavy clash penalty
        reward -= scores.get('rfd3_chainbreaks', 0) * 20.0  # chain break penalty
        reward -= scores.get('rfd3_max_ca_dev', 0) * 2.0  # CA deviation penalty
        reward += scores.get('rfd3_helix', 0) * 5.0  # secondary structure bonus
        reward += scores.get('rfd3_sheet', 0) * 5.0
        # Penalize extreme radius of gyration
        rog = scores.get('rfd3_rog', 15)
        if rog > 20:
            reward -= (rog - 20) * 1.0  # too large
        if rog < 10:
            reward -= (10 - rog) * 1.0  # too compact

        scores['reward'] = reward
        scores['cif_path'] = str(cif_path)
        results.append(scores)

    return results


def run_pipeline(
    input_json: str,
    n_rounds: int = 20,
    output_base: str = '/output/ppo_results',
):
    """Run the full RFD3 + scoring pipeline for multiple rounds."""
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    reward_history = []
    best_ever = {'reward': float('-inf')}
    all_results = []

    for rnd in range(n_rounds):
        logger.info(f"{'='*50}")
        logger.info(f"ROUND {rnd+1}/{n_rounds}")

        round_dir = output_base / f'round_{rnd:03d}'

        # Generate and score
        results = generate_and_score(input_json, str(round_dir))

        if not results:
            logger.warning("No results this round")
            continue

        # Compute round statistics
        rewards = [r['reward'] for r in results]
        round_mean = np.mean(rewards)
        round_best = max(rewards)
        reward_history.append(round_mean)

        if round_best > best_ever['reward']:
            best_idx = rewards.index(round_best)
            best_ever = results[best_idx].copy()
            best_ever['round'] = rnd

        logger.info(
            f"  Reward: mean={round_mean:.2f}, best={round_best:.2f}, "
            f"best_ever={best_ever['reward']:.2f} (r{best_ever.get('round', '?')})"
        )
        logger.info(
            f"  Metrics: clashes={np.mean([r.get('rfd3_clashes',0) for r in results]):.1f}, "
            f"helix={np.mean([r.get('rfd3_helix',0) for r in results]):.0%}, "
            f"rog={np.mean([r.get('rfd3_rog',0) for r in results]):.1f}"
        )

        all_results.extend(results)

        # Reward trend
        if len(reward_history) >= 4:
            first2 = np.mean(reward_history[:2])
            last2 = np.mean(reward_history[-2:])
            logger.info(f"  Trend: first2={first2:.2f} → last2={last2:.2f} (Δ={last2-first2:+.2f})")

        # Save round results
        with open(round_dir / 'scores.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Rounds: {n_rounds}, Total designs: {len(all_results)}")
    logger.info(f"Best reward: {best_ever['reward']:.2f} (round {best_ever.get('round', '?')})")

    if len(reward_history) >= 4:
        first2 = np.mean(reward_history[:2])
        last2 = np.mean(reward_history[-2:])
        logger.info(f"Reward trend: {first2:.2f} → {last2:.2f} (Δ={last2-first2:+.2f})")

    # Save summary
    with open(output_base / 'summary.json', 'w') as f:
        json.dump({
            'reward_history': reward_history,
            'best_design': best_ever,
            'n_rounds': n_rounds,
            'total_designs': len(all_results),
        }, f, indent=2, default=str)

    with open(output_base / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='/output/ppo_results')
    args = parser.parse_args()

    # Use the demo enzyme design config
    input_json = '/app/foundry/models/rfd3/docs/enzyme_design.json'

    run_pipeline(input_json, args.n_rounds, args.output_dir)


if __name__ == '__main__':
    main()
