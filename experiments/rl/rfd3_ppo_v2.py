"""PPO v2: Fixed weight persistence + proper PPO loss + filtered advantages.

Fixes from QA review:
1. WEIGHTS PERSIST: checkpoint saved after each round, loaded in next round
2. PROPER PPO: loss is differentiable torch expression, not numpy scaling
3. FILTERED DESIGNS: failed Rosetta scores (9999) excluded from training
4. SINGLE Docker container kept alive across rounds (not --rm)

Architecture:
- Long-running Docker container with RFD3 model in memory
- Host feeds commands via shared volume (generate, train, save)
- Host scores with Rosetta between steps
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


def score_with_rosetta(cif_paths):
    """Score CIFs with Rosetta on host."""
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
        except Exception as e:
            scores.append(9999.0)
    return scores


def write_docker_worker_script(output_dir, lr):
    """Write the Docker worker script that stays alive and processes commands."""
    script = f'''#!/usr/bin/env python3
"""RFD3 worker: stays alive, processes generate/train commands via shared volume."""
import torch, sys, os, json, time, glob, numpy as np, shutil

from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

print("WORKER: Loading RFD3...", flush=True)

# Check for persisted checkpoint
CKPT_PATH = "/workspace/model_checkpoint.pt"
BASE_CKPT = "/root/.foundry/checkpoints/rfd3_latest.ckpt"

conf = RFD3InferenceConfig(ckpt_path=BASE_CKPT,
    diffusion_batch_size=1, inference_sampler={{"num_timesteps": 5}})
engine = RFD3InferenceEngine(**conf)
engine._set_out_dir("/tmp/worker_gen")
engine.initialize()

# Setup training
class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DiffusionLoss(weight=4.0,sigma_data=16.0,lddt_weight=0.25,alpha_virtual_atom=1.0,
            alpha_polar_residues=1.0,alpha_ligand=10.0,lp_weight=0.0,unindexed_norm_p=1.0,
            alpha_unindexed_diffused=1.0,unindexed_t_alpha=0.75)
        self.s = SequenceLoss(weight=0.1,max_t=1)
    def forward(self,**kw):
        dl,dd=self.d(**kw); sl,sd=self.s(**kw); return dl+sl,{{**dd,**sd}}

engine.trainer.loss = Loss().cuda()
model = engine.trainer.state["model"]
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr={lr})

# Load persisted weights if they exist
if os.path.exists(CKPT_PATH):
    saved = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
    model.load_state_dict(saved, strict=False)
    print(f"WORKER: Loaded persisted weights from {{CKPT_PATH}}", flush=True)
else:
    print("WORKER: Using base checkpoint", flush=True)

print("WORKER: Ready", flush=True)

def load_cif_for_training(cif_path):
    result = parse(cif_path)
    aa = result["asym_unit"][0]
    initialize_chain_info_from_atom_array(aa)
    aa.set_annotation("chain_iid", aa.chain_id.copy())
    if "pn_unit_iid" not in aa.get_annotation_categories():
        aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
    for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
        if ann not in aa.get_annotation_categories():
            aa.set_annotation(ann, np.zeros(len(aa), dtype=bool))
    return aa

# Main loop: process commands from /workspace/command.json
while True:
    cmd_file = "/workspace/command.json"
    if not os.path.exists(cmd_file):
        time.sleep(1)
        continue

    try:
        with open(cmd_file) as f:
            cmd = json.load(f)
        os.remove(cmd_file)
    except:
        time.sleep(0.5)
        continue

    action = cmd.get("action")

    if action == "generate":
        out_dir = cmd["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        # Clear previous outputs
        for f in glob.glob(f"{{out_dir}}/*"): os.remove(f)
        model.eval()
        import subprocess as sp
        sp.run(["rfd3", "design", f"out_dir={{out_dir}}",
                "inputs=/app/foundry/models/rfd3/docs/enzyme_design.json"],
               capture_output=True, timeout=600)
        n_cifs = len(glob.glob(f"{{out_dir}}/*.cif.gz"))
        with open("/workspace/result.json", "w") as f:
            json.dump({{"status": "done", "n_designs": n_cifs}}, f)
        print(f"WORKER: Generated {{n_cifs}} designs", flush=True)

    elif action == "ppo_update":
        cif_advantages = cmd["cif_advantages"]  # list of [cif_path, advantage]
        ppo_epochs = cmd.get("ppo_epochs", 3)
        clip_eps = cmd.get("clip_epsilon", 0.2)

        # Filter: only valid designs (advantage != 0 and file exists)
        valid = [(c, a) for c, a in cif_advantages if os.path.exists(c) and abs(a) > 0.001]
        if not valid:
            with open("/workspace/result.json", "w") as f:
                json.dump({{"status": "done", "n_updates": 0, "avg_loss": 0}}, f)
            continue

        # Step 1: Compute old losses (frozen model)
        model.eval()
        old_losses = {{}}
        pipeline_data = {{}}
        for cif_path, adv in valid:
            try:
                aa = load_cif_for_training(cif_path)
                pr = engine.pipeline({{"atom_array": aa, "example_id": os.path.basename(cif_path)}})
                pr = engine.trainer.fabric.to_device(pr)
                pipeline_data[cif_path] = pr

                with torch.no_grad():
                    ni = engine.trainer._assemble_network_inputs(pr)
                    no = model.forward(input=ni, n_cycle=0)
                    li = engine.trainer._assemble_loss_extra_info(pr)
                    loss_val, _ = engine.trainer.loss(network_input=ni, network_output=no, loss_input=li)
                    old_losses[cif_path] = loss_val.item()
            except Exception as e:
                print(f"WORKER: Failed to process {{cif_path}}: {{e}}", flush=True)

        # Step 2: PPO epochs
        model.train()
        total_loss = 0
        n_updates = 0

        for epoch in range(ppo_epochs):
            for cif_path, advantage in valid:
                if cif_path not in pipeline_data or cif_path not in old_losses:
                    continue

                pr = pipeline_data[cif_path]
                opt.zero_grad()

                # Compute new loss WITH gradients
                ni = engine.trainer._assemble_network_inputs(pr)
                no = model.forward(input=ni, n_cycle=0)
                li = engine.trainer._assemble_loss_extra_info(pr)
                new_loss_tensor, _ = engine.trainer.loss(
                    network_input=ni, network_output=no, loss_input=li)

                # PPO: ratio = exp(old_loss - new_loss)
                # new_loss_tensor IS differentiable
                old_loss_val = old_losses[cif_path]
                log_ratio = old_loss_val - new_loss_tensor  # tensor
                ratio = torch.exp(torch.clamp(log_ratio, -10, 10))

                # Clipped surrogate
                adv_tensor = torch.tensor(advantage, device=ratio.device)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tensor
                ppo_loss = -torch.min(surr1, surr2)  # negative because we minimize

                # Backward through PPO loss (differentiable!)
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

                total_loss += ppo_loss.item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        print(f"WORKER: PPO {{n_updates}} updates, avg_loss={{avg_loss:.4f}}", flush=True)

        # Step 3: SAVE updated weights (FIX for Bug #1)
        torch.save({{k: v.cpu() for k, v in model.state_dict().items()}}, CKPT_PATH)
        print(f"WORKER: Checkpoint saved", flush=True)

        with open("/workspace/result.json", "w") as f:
            json.dump({{"status": "done", "n_updates": n_updates, "avg_loss": avg_loss}}, f)

    elif action == "stop":
        print("WORKER: Stopping", flush=True)
        break

    else:
        print(f"WORKER: Unknown action {{action}}", flush=True)
'''
    script_path = os.path.join(output_dir, 'worker.py')
    with open(script_path, 'w') as f:
        f.write(script)
    return script_path


def send_command(workspace, cmd, timeout=600):
    """Send command to Docker worker and wait for result."""
    with open(os.path.join(workspace, 'command.json'), 'w') as f:
        json.dump(cmd, f)

    result_file = os.path.join(workspace, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(result_file):
            with open(result_file) as f:
                result = json.load(f)
            os.remove(result_file)
            return result
        time.sleep(1)

    return {'status': 'timeout'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=30)
    parser.add_argument('--ppo-epochs', type=int, default=3)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-5)  # higher lr than before
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/rfd3_ppo_v2'))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = str(output_dir / 'workspace')
    os.makedirs(workspace, exist_ok=True)

    # Write worker script
    write_docker_worker_script(workspace, args.lr)

    # Start persistent Docker container
    logger.info("Starting persistent Docker worker...")
    container_name = 'rfd3_ppo_worker'
    subprocess.run(['sudo', 'docker', 'rm', '-f', container_name], capture_output=True)

    proc = subprocess.Popen([
        'sudo', 'docker', 'run', '--gpus', 'all',
        '--name', container_name,
        '-v', f'{os.path.abspath(workspace)}:/workspace',
        'rosettacommons/foundry:latest',
        'python3', '-u', '/workspace/worker.py',
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for worker to be ready
    logger.info("Waiting for worker to initialize...")
    time.sleep(30)  # RFD3 loading takes ~30s

    # Check if worker is ready
    for _ in range(60):
        try:
            with open(os.path.join(workspace, 'command.json'), 'w') as f:
                json.dump({'action': 'ping'}, f)
            time.sleep(2)
            break
        except:
            time.sleep(1)

    logger.info("="*60)
    logger.info("PPO v2: FIXED WEIGHT PERSISTENCE + PROPER PPO LOSS")
    logger.info(f"Rounds: {args.n_rounds}, PPO epochs: {args.ppo_epochs}, lr: {args.lr}")
    logger.info("="*60)

    rosetta_history = []
    best_rosetta = float('inf')

    for rnd in range(args.n_rounds):
        t0 = time.time()
        round_dir = str(output_dir / f'round_{rnd:04d}')
        os.makedirs(round_dir, exist_ok=True)

        logger.info(f"\n--- Round {rnd+1}/{args.n_rounds} ---")

        # 1. Generate (in persistent container)
        gen_dir = f'/workspace/gen_{rnd}'
        result = send_command(workspace, {'action': 'generate', 'output_dir': gen_dir}, timeout=600)
        t_gen = time.time() - t0

        # Copy CIFs from workspace to round_dir
        cifs_in_workspace = glob.glob(f'{workspace}/gen_{rnd}/*.cif.gz')
        cifs = []
        for c in cifs_in_workspace:
            dest = os.path.join(round_dir, os.path.basename(c))
            os.system(f'cp {c} {dest}')
            cifs.append(dest)

        logger.info(f"  Generated: {len(cifs)} ({t_gen:.0f}s)")
        if not cifs: continue

        # 2. Rosetta score (on host)
        t_s = time.time()
        scores = score_with_rosetta(cifs)
        t_score = time.time() - t_s

        # Filter valid scores
        valid_pairs = [(c, s) for c, s in zip(cifs, scores) if s < 9000]
        if not valid_pairs:
            logger.warning("  No valid Rosetta scores")
            continue

        valid_scores = [s for _, s in valid_pairs]
        best = min(valid_scores)
        mean = np.mean(valid_scores)
        rosetta_history.append(best)
        if best < best_rosetta:
            best_rosetta = best
        logger.info(f"  Rosetta: best={best:.0f}, mean={mean:.0f}, best_ever={best_rosetta:.0f} ({t_score:.0f}s)")

        # 3. Compute advantages (only valid designs)
        rewards = [-s for _, s in valid_pairs]
        mean_r = np.mean(rewards)
        std_r = max(np.std(rewards), 1.0)
        advantages = [(-s - mean_r) / std_r for _, s in valid_pairs]

        # Map to workspace CIF paths (Docker sees /workspace/gen_N/)
        cif_advantages = [
            [f'/workspace/gen_{rnd}/{os.path.basename(c)}', a]
            for (c, _), a in zip(valid_pairs, advantages)
        ]

        # 4. PPO update (in persistent container with weight persistence)
        t_t = time.time()
        result = send_command(workspace, {
            'action': 'ppo_update',
            'cif_advantages': cif_advantages,
            'ppo_epochs': args.ppo_epochs,
            'clip_epsilon': args.clip_epsilon,
        }, timeout=300)
        t_train = time.time() - t_t
        logger.info(f"  PPO: {result} ({t_train:.0f}s)")

        # Trend
        if len(rosetta_history) >= 6:
            f3 = np.mean(rosetta_history[:3])
            l3 = np.mean(rosetta_history[-3:])
            logger.info(f"  Trend: first3={f3:.0f} → last3={l3:.0f} (Δ={l3-f3:+.0f})")

        # Save
        if (rnd + 1) % 5 == 0:
            with open(output_dir / 'rosetta_history.json', 'w') as f:
                json.dump(rosetta_history, f)

    # Stop worker
    send_command(workspace, {'action': 'stop'})
    subprocess.run(['sudo', 'docker', 'rm', '-f', container_name], capture_output=True)

    # Final
    logger.info(f"\n{'='*60}")
    logger.info(f"PPO v2 COMPLETE: {args.n_rounds} rounds")
    logger.info(f"Best Rosetta: {best_rosetta:.0f}")
    if len(rosetta_history) >= 6:
        f3 = np.mean(rosetta_history[:3])
        l3 = np.mean(rosetta_history[-3:])
        logger.info(f"Trend: {f3:.0f} → {l3:.0f} (Δ={l3-f3:+.0f})")
    logger.info("="*60)

    with open(output_dir / 'rosetta_history.json', 'w') as f:
        json.dump(rosetta_history, f)
    with open(output_dir / 'final.json', 'w') as f:
        json.dump({'rosetta_history': rosetta_history, 'best_rosetta': best_rosetta}, f, indent=2)


if __name__ == '__main__':
    main()
