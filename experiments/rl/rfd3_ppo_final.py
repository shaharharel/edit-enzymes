"""PPO fine-tuning of RFD3 with Rosetta reward.

The correct PPO loop for diffusion models:
1. GENERATE batch of designs with model_old (no_grad, 50 denoising steps)
2. SCORE all designs with Rosetta → rewards
3. Compute OLD log_probs: training_step on each design → diffusion_loss_old
4. For K PPO epochs:
   a. Compute NEW log_probs: training_step with current model → diffusion_loss_new
   b. ratio = exp(loss_old - loss_new)  (higher prob under new model → ratio > 1)
   c. advantage = (reward - baseline) / std
   d. loss = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
   e. backward + optimizer.step()
5. model_old = model_current (sync for next round)

Key insight: Generation (50 steps, no_grad) and log_prob (1 training_step, with_grad)
are SEPARATE code paths. PPO only needs gradients for log_prob.

Runs as host orchestrator:
- Docker: RFD3 generation + training_step (foundry)
- Host: Rosetta scoring (PyRosetta)
"""

import os
import sys
import json
import time
import glob
import gzip
import tempfile
import subprocess
import shutil
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def generate_designs(round_dir):
    """Generate designs with RFD3 in Docker (no_grad, 50 denoising steps)."""
    os.makedirs(round_dir, exist_ok=True)
    gen_dir = f"/tmp/rfd3_gen_{os.getpid()}_{time.time_ns()}"
    cmd = [
        'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{os.path.abspath(round_dir)}:/output',
        'rosettacommons/foundry:latest',
        'bash', '-c',
        f'cd /app/foundry/models/rfd3/docs && rfd3 design out_dir=/output inputs=enzyme_design.json'
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return sorted(glob.glob(f'{round_dir}/*.cif.gz'))


def score_with_rosetta(cif_paths):
    """Score designs with Rosetta on host."""
    import pyrosetta
    import biotite.structure.io.pdbx as pdbx

    if not hasattr(score_with_rosetta, '_init'):
        pyrosetta.init('-mute all -ex1 -ex2')
        score_with_rosetta._sfxn = pyrosetta.get_score_function(True)
        score_with_rosetta._init = True
    sfxn = score_with_rosetta._sfxn

    scores = []
    for cif_path in cif_paths:
        try:
            with gzip.open(cif_path, 'rt') as f:
                cif = pdbx.CIFFile.read(f)
            aa = pdbx.get_structure(list(cif.values())[0], model=1)
            with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
                pdb_path = f.name
                for i in range(len(aa)):
                    if aa.element[i] == '': continue
                    f.write(f"ATOM  {i+1:5d}  {aa.atom_name[i]:<3s} {aa.res_name[i]:3s} "
                            f"{aa.chain_id[i]:1s}{aa.res_id[i]:4d}    "
                            f"{aa.coord[i,0]:8.3f}{aa.coord[i,1]:8.3f}{aa.coord[i,2]:8.3f}"
                            f"  1.00  0.00           {aa.element[i]:>2s}\n")
                f.write("END\n")
            with open(pdb_path) as f:
                lines = [l for l in f if l.startswith('ATOM') or l.startswith('END')]
            clean = pdb_path + '.c'
            with open(clean, 'w') as f: f.writelines(lines)
            pose = pyrosetta.pose_from_pdb(clean)
            from pyrosetta.rosetta.core.pack.task import TaskFactory
            from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
            tf = TaskFactory(); tf.push_back(RestrictToRepacking())
            pk = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
            pk.task_factory(tf); pk.apply(pose)
            scores.append(float(sfxn(pose)))
            os.unlink(pdb_path); os.unlink(clean)
        except:
            scores.append(9999.0)
    return scores


def ppo_update_docker(round_dir, cif_paths, rosetta_scores, ppo_epochs, clip_epsilon, lr):
    """Run PPO update in Docker: compute log_probs, clip ratio, update weights.

    Returns dict with metrics.
    """
    # Compute advantages
    rewards = [-s for s in rosetta_scores if s < 9000]  # negate: lower rosetta = higher reward
    if not rewards:
        return {'n_updates': 0}

    mean_r = np.mean(rewards)
    std_r = max(np.std(rewards), 1.0)
    advantages = [(-s - mean_r) / std_r for s in rosetta_scores]

    # Build PPO script for Docker
    cif_adv_list = [(os.path.basename(c), a) for c, a in zip(cif_paths, advantages) if a != 0]

    script = f'''
import torch, sys, os, json, numpy as np, time
from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

conf = RFD3InferenceConfig(ckpt_path="/root/.foundry/checkpoints/rfd3_latest.ckpt",
    diffusion_batch_size=1, inference_sampler={{"num_timesteps": 3}})
engine = RFD3InferenceEngine(**conf)
engine._set_out_dir("/tmp/ppo")
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
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr={lr})

# Parse all CIFs into pipeline-ready data
cif_advantages = {cif_adv_list}
pipeline_data = []

for cif_name, advantage in cif_advantages:
    cif_path = f"/input/{{cif_name}}"
    if not os.path.exists(cif_path): continue
    try:
        result = parse(cif_path)
        aa = result["asym_unit"][0]
        initialize_chain_info_from_atom_array(aa)
        aa.set_annotation("chain_iid", aa.chain_id.copy())
        if "pn_unit_iid" not in aa.get_annotation_categories():
            aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
        for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
            if ann not in aa.get_annotation_categories():
                aa.set_annotation(ann, np.zeros(len(aa), dtype=bool))
        pr = engine.pipeline({{"atom_array": aa, "example_id": cif_name}})
        pr = engine.trainer.fabric.to_device(pr)
        pipeline_data.append((pr, advantage))
    except:
        pass

if not pipeline_data:
    print("PPO_RESULT: n=0 loss=0.0")
    sys.exit(0)

# Step 1: Compute OLD log_probs (diffusion loss under current model)
model.eval()
old_losses = []
for pr, _ in pipeline_data:
    with torch.no_grad():
        model.eval()
        engine.trainer.loss.eval()
        # We need the loss value without backward
        ni = engine.trainer._assemble_network_inputs(pr)
        no = model.forward(input=ni, n_cycle=0)
        li = engine.trainer._assemble_loss_extra_info(pr)
        loss_val, _ = engine.trainer.loss(network_input=ni, network_output=no, loss_input=li)
        old_losses.append(loss_val.item())

# Step 2: PPO epochs
model.train()
engine.trainer.loss.train()
total_ppo_loss = 0
n_updates = 0

for epoch in range({ppo_epochs}):
    for idx, (pr, advantage) in enumerate(pipeline_data):
        opt.zero_grad()

        # Compute NEW loss (with gradients)
        engine.trainer.training_step(batch=[pr], batch_idx=idx, is_accumulating=False)
        ret = engine.trainer._current_train_return
        new_loss = ret["total_loss"].item() if ret and "total_loss" in ret else old_losses[idx]

        # PPO ratio: exp(old_loss - new_loss)
        # If new model assigns HIGHER prob (lower loss): ratio > 1
        ratio = np.exp(np.clip(old_losses[idx] - new_loss, -10, 10))

        # Clipped surrogate objective
        clip_eps = {clip_epsilon}
        surr1 = ratio * advantage
        surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
        ppo_objective = min(surr1, surr2)

        # Scale gradients by PPO objective / loss
        # If ppo_objective > 0: keep gradient direction (encourage)
        # If ppo_objective < 0: flip (discourage)
        scale = ppo_objective / max(abs(new_loss), 1e-8)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        total_ppo_loss += abs(ppo_objective)
        n_updates += 1

print(f"PPO_RESULT: n={{n_updates}} loss={{total_ppo_loss/max(n_updates,1):.4f}}")
'''

    script_path = os.path.join(round_dir, 'ppo_train.py')
    with open(script_path, 'w') as f:
        f.write(script)

    cmd = [
        'sudo', 'docker', 'run', '--rm', '--gpus', 'all',
        '-v', f'{os.path.abspath(round_dir)}:/input',
        '-v', f'{script_path}:/workspace/train.py',
        'rosettacommons/foundry:latest',
        'python3', '/workspace/train.py',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in (result.stdout + result.stderr).split('\n'):
        if 'PPO_RESULT:' in line:
            logger.info(f"  {line.strip()}")
            return {'output': line.strip()}

    return {'output': 'no result', 'stderr': result.stderr[-200:] if result.stderr else ''}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=30)
    parser.add_argument('--ppo-epochs', type=int, default=3)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/rfd3_ppo_final'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("PPO FINE-TUNING OF RFD3 WITH ROSETTA REWARD")
    logger.info(f"Rounds: {args.n_rounds}, PPO epochs: {args.ppo_epochs}, clip: {args.clip_epsilon}")
    logger.info("="*60)

    rosetta_history = []
    best_rosetta = float('inf')

    for rnd in range(args.n_rounds):
        t0 = time.time()
        round_dir = str(output_dir / f'round_{rnd:04d}')

        logger.info(f"\n--- Round {rnd+1}/{args.n_rounds} ---")

        # 1. Generate
        cifs = generate_designs(round_dir)
        t_gen = time.time() - t0
        logger.info(f"  Generated: {len(cifs)} designs ({t_gen:.0f}s)")
        if not cifs: continue

        # 2. Rosetta score
        t_s = time.time()
        scores = score_with_rosetta(cifs)
        valid = [s for s in scores if s < 9000]
        t_score = time.time() - t_s

        if valid:
            best = min(valid)
            mean = np.mean(valid)
            rosetta_history.append(best)
            if best < best_rosetta:
                best_rosetta = best
            logger.info(f"  Rosetta: best={best:.0f}, mean={mean:.0f}, best_ever={best_rosetta:.0f} ({t_score:.0f}s)")
        else:
            logger.warning("  No valid scores")
            continue

        # 3. PPO update
        t_t = time.time()
        metrics = ppo_update_docker(round_dir, cifs, scores, args.ppo_epochs, args.clip_epsilon, args.lr)
        t_train = time.time() - t_t
        logger.info(f"  PPO: {metrics.get('output', 'unknown')} ({t_train:.0f}s)")

        # Trend
        if len(rosetta_history) >= 6:
            f3 = np.mean(rosetta_history[:3])
            l3 = np.mean(rosetta_history[-3:])
            logger.info(f"  Rosetta trend: first3={f3:.0f} → last3={l3:.0f} (Δ={l3-f3:+.0f})")

        # Save
        if (rnd + 1) % 5 == 0:
            with open(output_dir / 'rosetta_history.json', 'w') as f:
                json.dump(rosetta_history, f)

    # Final
    logger.info(f"\n{'='*60}")
    logger.info(f"PPO COMPLETE: {args.n_rounds} rounds")
    logger.info(f"Best Rosetta: {best_rosetta:.0f}")
    if len(rosetta_history) >= 6:
        f3 = np.mean(rosetta_history[:3])
        l3 = np.mean(rosetta_history[-3:])
        logger.info(f"Trend: {f3:.0f} → {l3:.0f} (Δ={l3-f3:+.0f})")
    logger.info("="*60)

    with open(output_dir / 'rosetta_history.json', 'w') as f:
        json.dump(rosetta_history, f)
    with open(output_dir / 'final.json', 'w') as f:
        json.dump({'rosetta_history': rosetta_history, 'best_rosetta': best_rosetta,
                   'n_rounds': args.n_rounds, 'ppo_epochs': args.ppo_epochs}, f, indent=2)


if __name__ == '__main__':
    main()
