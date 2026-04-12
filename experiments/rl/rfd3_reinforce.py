"""REINFORCE on RFD3 with Rosetta reward.

Correct RL for diffusion models:
1. Generate design (no grad, CLI) → CIF
2. Score with Rosetta → reward R
3. Compute log_prob: run training_step on CIF (WITH grad) → diffusion_loss
   diffusion_loss ≈ -log_prob(design | model)
4. REINFORCE: scale the loss by advantage
   rl_loss = (R - baseline) × diffusion_loss
   If R > baseline: loss is POSITIVE → backward REDUCES loss → INCREASES log_prob of this design
   If R < baseline: loss is NEGATIVE → backward INCREASES loss → DECREASES log_prob of this design
5. backward + optimizer.step()

This is REINFORCE: ∇J = E[advantage × ∇log_prob(design)]
The diffusion training loss gradient IS ∇(-log_prob).

Runs as host orchestrator:
- Docker: RFD3 generate + training_step (needs foundry)
- Host: Rosetta scoring (needs PyRosetta)

Usage: bash on ai-gpu2 host
    python3 experiments/rl/rfd3_reinforce.py --n-rounds 30
"""

import os
import sys
import json
import time
import glob
import gzip
import tempfile
import subprocess
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def generate_designs(round_dir):
    """Generate 8 designs with RFD3 via Docker."""
    os.makedirs(round_dir, exist_ok=True)
    cmd = [
        'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{os.path.abspath(round_dir)}:/output',
        'rosettacommons/foundry:latest',
        'bash', '-c',
        'cd /app/foundry/models/rfd3/docs && rfd3 design out_dir=/output inputs=enzyme_design.json'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    cifs = sorted(glob.glob(f'{round_dir}/*.cif.gz'))
    return cifs


def score_with_rosetta(cif_paths):
    """Score CIFs with Rosetta on host. Returns list of (cif_path, rosetta_score)."""
    import pyrosetta
    import biotite.structure.io.pdbx as pdbx

    if not hasattr(score_with_rosetta, '_init'):
        pyrosetta.init('-mute all -ex1 -ex2')
        score_with_rosetta._sfxn = pyrosetta.get_score_function(True)
        score_with_rosetta._init = True

    sfxn = score_with_rosetta._sfxn
    results = []

    for cif_path in cif_paths:
        try:
            with gzip.open(cif_path, 'rt') as f:
                cif = pdbx.CIFFile.read(f)
            block = list(cif.values())[0]
            aa = pdbx.get_structure(block, model=1)

            with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
                pdb_path = f.name
                for i in range(len(aa)):
                    if aa.element[i] == '':
                        continue
                    f.write(
                        f"ATOM  {i+1:5d}  {aa.atom_name[i]:<3s} {aa.res_name[i]:3s} "
                        f"{aa.chain_id[i]:1s}{aa.res_id[i]:4d}    "
                        f"{aa.coord[i,0]:8.3f}{aa.coord[i,1]:8.3f}{aa.coord[i,2]:8.3f}"
                        f"  1.00  0.00           {aa.element[i]:>2s}\n"
                    )
                f.write("END\n")

            # Clean for Rosetta
            with open(pdb_path) as f:
                lines = [l for l in f if l.startswith('ATOM') or l.startswith('END')]
            clean = pdb_path + '.clean'
            with open(clean, 'w') as f:
                f.writelines(lines)

            pose = pyrosetta.pose_from_pdb(clean)
            from pyrosetta.rosetta.core.pack.task import TaskFactory
            from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            pk = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
            pk.task_factory(tf)
            pk.apply(pose)

            score = sfxn(pose)
            results.append((cif_path, float(score)))
            os.unlink(pdb_path)
            os.unlink(clean)
        except Exception as e:
            results.append((cif_path, 9999.0))

    return results


def reinforce_update(cif_scores, round_dir, baseline):
    """Run REINFORCE update in Docker: scale diffusion loss by advantage.

    For each design:
        advantage = -rosetta_score - baseline (higher advantage = better design)
        rl_loss = advantage × diffusion_loss
        (positive advantage → model learns to reproduce this design MORE)

    Returns average loss.
    """
    # Sort by Rosetta (lower = better)
    scored = [(cif, score) for cif, score in cif_scores if score < 9000]
    if not scored:
        return 0.0, baseline

    rewards = [-s for _, s in scored]  # negate: lower Rosetta = higher reward
    mean_reward = np.mean(rewards)
    new_baseline = 0.95 * baseline + 0.05 * mean_reward

    # Write the REINFORCE training script
    advantages = [(-s - baseline) for _, s in scored]

    # Build advantage-weighted training script
    cif_advantage_pairs = list(zip([c for c, _ in scored], advantages))

    script = f'''
import torch, sys, os, numpy as np
from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

conf = RFD3InferenceConfig(ckpt_path="/root/.foundry/checkpoints/rfd3_latest.ckpt",
    diffusion_batch_size=1, inference_sampler={{"num_timesteps": 3}})
engine = RFD3InferenceEngine(**conf)
engine._set_out_dir("/tmp/reinforce")
engine.initialize()

class L(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DiffusionLoss(weight=4.0,sigma_data=16.0,lddt_weight=0.25,alpha_virtual_atom=1.0,
            alpha_polar_residues=1.0,alpha_ligand=10.0,lp_weight=0.0,unindexed_norm_p=1.0,
            alpha_unindexed_diffused=1.0,unindexed_t_alpha=0.75)
        self.s = SequenceLoss(weight=0.1,max_t=1)
    def forward(self,**kw):
        dl,dd=self.d(**kw); sl,sd=self.s(**kw); return dl+sl,{{**dd,**sd}}

engine.trainer.loss = L().cuda()
model = engine.trainer.state["model"]
model.train()
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-6)

cif_advantages = {cif_advantage_pairs}
total_loss = 0
n = 0

for cif_path, advantage in cif_advantages:
    docker_cif = cif_path.replace("{round_dir}", "/input")
    if not os.path.exists(docker_cif):
        continue
    try:
        result = parse(docker_cif)
        aa = result["asym_unit"][0]
        initialize_chain_info_from_atom_array(aa)
        aa.set_annotation("chain_iid", aa.chain_id.copy())
        if "pn_unit_iid" not in aa.get_annotation_categories():
            aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
        for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
            if ann not in aa.get_annotation_categories():
                aa.set_annotation(ann, np.zeros(len(aa), dtype=bool))

        pr = engine.pipeline({{"atom_array": aa, "example_id": f"design_{{n}}"}})
        pr = engine.trainer.fabric.to_device(pr)

        opt.zero_grad()
        engine.trainer.training_step(batch=[pr], batch_idx=0, is_accumulating=False)

        # REINFORCE: scale gradients by advantage
        # advantage > 0: keep gradient direction (encourage this design)
        # advantage < 0: flip gradient direction (discourage this design)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= advantage

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        ret = engine.trainer._current_train_return
        if ret and "total_loss" in ret:
            total_loss += ret["total_loss"].item() * advantage
        n += 1
    except Exception as e:
        pass

print(f"REINFORCE: {{n}} updates, avg_loss={{total_loss/max(n,1):.4f}}")
'''

    script_path = os.path.join(round_dir, 'reinforce_train.py')
    with open(script_path, 'w') as f:
        f.write(script)

    cmd = [
        'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{os.path.abspath(round_dir)}:/input',
        '-v', f'{script_path}:/workspace/train.py',
        'rosettacommons/foundry:latest',
        'python3', '/workspace/train.py',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Parse output
    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        if 'REINFORCE:' in line:
            logger.info(f"  {line.strip()}")
            break

    return mean_reward, new_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/rfd3_reinforce_output'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("RFD3 REINFORCE WITH ROSETTA REWARD")
    logger.info(f"Rounds: {args.n_rounds}")
    logger.info("="*60)

    baseline = 0.0
    reward_history = []
    rosetta_history = []

    for rnd in range(args.n_rounds):
        t0 = time.time()
        round_dir = str(output_dir / f'round_{rnd:04d}')

        # 1. Generate
        logger.info(f"\n--- Round {rnd+1}/{args.n_rounds} ---")
        cifs = generate_designs(round_dir)
        t_gen = time.time() - t0
        logger.info(f"  Generated {len(cifs)} designs ({t_gen:.0f}s)")

        if not cifs:
            continue

        # 2. Score with Rosetta
        t_score = time.time()
        cif_scores = score_with_rosetta(cifs)
        valid_scores = [s for _, s in cif_scores if s < 9000]
        t_score = time.time() - t_score

        if valid_scores:
            best_rosetta = min(valid_scores)
            mean_rosetta = np.mean(valid_scores)
            rosetta_history.append(best_rosetta)
            logger.info(f"  Rosetta: best={best_rosetta:.0f}, mean={mean_rosetta:.0f} ({t_score:.0f}s)")
        else:
            logger.warning("  No valid Rosetta scores")
            continue

        # 3. REINFORCE update
        t_train = time.time()
        mean_reward, baseline = reinforce_update(cif_scores, round_dir, baseline)
        t_train = time.time() - t_train
        reward_history.append(mean_reward)
        logger.info(f"  Reward: {mean_reward:.1f}, baseline: {baseline:.1f} ({t_train:.0f}s)")

        # Trend
        if len(rosetta_history) >= 6:
            f3 = np.mean(rosetta_history[:3])
            l3 = np.mean(rosetta_history[-3:])
            logger.info(f"  Rosetta trend: first3={f3:.0f} → last3={l3:.0f} (Δ={l3-f3:+.0f})")

        # Save every 5 rounds
        if (rnd + 1) % 5 == 0:
            with open(output_dir / 'rosetta_history.json', 'w') as f:
                json.dump(rosetta_history, f)
            with open(output_dir / 'reward_history.json', 'w') as f:
                json.dump(reward_history, f)

    # Final
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {args.n_rounds} rounds")
    if rosetta_history:
        logger.info(f"Best Rosetta: {min(rosetta_history):.0f}")
        if len(rosetta_history) >= 6:
            f3 = np.mean(rosetta_history[:3])
            l3 = np.mean(rosetta_history[-3:])
            logger.info(f"Rosetta trend: {f3:.0f} → {l3:.0f} (Δ={l3-f3:+.0f})")
    logger.info(f"{'='*60}")

    with open(output_dir / 'rosetta_history.json', 'w') as f:
        json.dump(rosetta_history, f)
    with open(output_dir / 'final.json', 'w') as f:
        json.dump({'rosetta_history': rosetta_history, 'reward_history': reward_history,
                   'best_rosetta': min(rosetta_history) if rosetta_history else None}, f, indent=2)


if __name__ == '__main__':
    main()
