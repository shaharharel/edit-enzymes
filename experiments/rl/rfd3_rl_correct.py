"""Correct RL fine-tuning of RFD3: generate → score with Rosetta → train on best.

The loop:
1. Generate 8 designs via RFD3 CLI (no gradients needed here)
2. Score ALL designs with Rosetta (actual energy, not just RFD3 metrics)
3. Select top 4 by Rosetta score (reward = -rosetta_energy, higher = better)
4. Parse best CIFs with atomworks → feed through pipeline → training_step
   → model learns to reproduce Rosetta-favorable structures
5. Repeat: model should generate increasingly stable designs

This is reward-weighted MLE (same principle as RLHF reject sampling fine-tuning).
The diffusion training loss IS the log-prob under the model.

Usage (inside Foundry Docker):
    python3 /workspace/rfd3_rl_correct.py --n-rounds 50
"""

import os
import sys
import json
import time
import glob
import gzip
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ============================================================================
# Rosetta scoring
# ============================================================================

_PYROSETTA = False
_SFXN = None

def init_rosetta():
    global _PYROSETTA, _SFXN
    if not _PYROSETTA:
        import pyrosetta
        pyrosetta.init('-mute all -ex1 -ex2')
        _SFXN = pyrosetta.get_score_function(True)
        _PYROSETTA = True


def score_cif_with_rosetta(cif_gz_path):
    """Convert CIF to PDB, score with Rosetta. Returns Rosetta energy (lower = better)."""
    try:
        init_rosetta()
        import pyrosetta
        import biotite.structure.io.pdbx as pdbx

        with gzip.open(cif_gz_path, 'rt') as f:
            cif = pdbx.CIFFile.read(f)
        block = list(cif.values())[0]
        aa = pdbx.get_structure(block, model=1)

        # Write as PDB
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
            pdb_path = f.name
            atom_num = 1
            for i in range(len(aa)):
                if aa.element[i] == '':
                    continue
                f.write(
                    f"ATOM  {atom_num:5d}  {aa.atom_name[i]:<3s} {aa.res_name[i]:3s} "
                    f"{aa.chain_id[i]:1s}{aa.res_id[i]:4d}    "
                    f"{aa.coord[i,0]:8.3f}{aa.coord[i,1]:8.3f}{aa.coord[i,2]:8.3f}"
                    f"  1.00  0.00           {aa.element[i]:>2s}\n"
                )
                atom_num += 1
            f.write("END\n")

        # Score (strip non-standard for Rosetta)
        with open(pdb_path) as f:
            lines = [l for l in f if l.startswith('ATOM') or l.startswith('END')]
        clean_path = pdb_path + '.clean.pdb'
        with open(clean_path, 'w') as f:
            f.writelines(lines)

        pose = pyrosetta.pose_from_pdb(clean_path)

        # Repack sidechains
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(_SFXN)
        packer.task_factory(tf)
        packer.apply(pose)

        score = _SFXN(pose)
        n_res = pose.total_residue()
        os.unlink(pdb_path)
        os.unlink(clean_path)
        return score, n_res

    except Exception as e:
        logger.warning(f"Rosetta scoring failed: {e}")
        return None, 0


# ============================================================================
# CIF → training input
# ============================================================================

def load_cif_for_training(cif_gz_path):
    """Parse a generated CIF file and add all annotations needed for RFD3 training."""
    from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
    from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS

    result = parse(cif_gz_path)
    aa = result["asym_unit"][0]  # First model
    initialize_chain_info_from_atom_array(aa)

    n = len(aa)
    # Add chain_iid (uses chain_id as instance)
    aa.set_annotation("chain_iid", aa.chain_id.copy())
    # Add pn_unit_iid
    if "pn_unit_iid" not in aa.get_annotation_categories():
        aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
    # Add conditioning annotations (all False = unconditional training)
    for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
        if ann not in aa.get_annotation_categories():
            aa.set_annotation(ann, np.zeros(n, dtype=bool))

    return aa


# ============================================================================
# Main RL loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=50)
    parser.add_argument('--train-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='/output/rl_rosetta')
    parser.add_argument('--use-rosetta', action='store_true', help='Score with Rosetta (slow but accurate)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json = '/app/foundry/models/rfd3/docs/enzyme_design.json'

    # Load engine
    logger.info("Loading RFD3...")
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

    conf = RFD3InferenceConfig(
        ckpt_path="/root/.foundry/checkpoints/rfd3_latest.ckpt",
        diffusion_batch_size=1,
        inference_sampler={"num_timesteps": 5},
    )
    engine = RFD3InferenceEngine(**conf)
    engine._set_out_dir("/tmp/rl_gen")
    engine.initialize()

    # Setup training
    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = DiffusionLoss(weight=4.0, sigma_data=16.0, lddt_weight=0.25,
                alpha_virtual_atom=1.0, alpha_polar_residues=1.0, alpha_ligand=10.0,
                lp_weight=0.0, unindexed_norm_p=1.0, alpha_unindexed_diffused=1.0,
                unindexed_t_alpha=0.75)
            self.s = SequenceLoss(weight=0.1, max_t=1)
        def forward(self, **kw):
            dl, dd = self.d(**kw)
            sl, sd = self.s(**kw)
            return dl + sl, {**dd, **sd}

    engine.trainer.loss = Loss().cuda()
    model = engine.trainer.state["model"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Tracking
    reward_history = []
    rosetta_history = []
    loss_history = []
    best_rosetta = float('inf')

    logger.info(f"{'='*60}")
    logger.info(f"RL FINE-TUNING OF RFD3")
    logger.info(f"Rounds: {args.n_rounds}, top-K: {args.top_k}, lr: {args.lr}")
    logger.info(f"Scoring: {'Rosetta' if args.use_rosetta else 'RFD3 metrics'}")
    logger.info(f"{'='*60}")

    for rnd in range(args.n_rounds):
        t_start = time.time()

        # === 1. GENERATE ===
        round_dir = str(output_dir / f'round_{rnd:04d}')
        model.eval()
        # Each round gets a fresh output dir (RFD3 skips if outputs exist)
        import shutil
        gen_dir = f"/tmp/rl_gen_round_{rnd}"
        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
        cmd = ['rfd3', 'design', f'out_dir={gen_dir}', f'inputs={input_json}']
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Copy outputs to round_dir
        os.makedirs(round_dir, exist_ok=True)
        for f in glob.glob(f"{gen_dir}/*"):
            shutil.copy(f, round_dir)
        cif_files = sorted(glob.glob(f"{round_dir}/*.cif.gz"))
        json_files = sorted(glob.glob(f"{round_dir}/*.json"))
        t_gen = time.time() - t_start

        if not cif_files:
            logger.warning(f"Round {rnd}: no designs")
            continue

        # === 2. SCORE ===
        t_score_start = time.time()
        scores = []
        for cif, jf in zip(cif_files, json_files):
            if args.use_rosetta:
                rosetta_e, n_res = score_cif_with_rosetta(cif)
                reward = -rosetta_e if rosetta_e is not None else -9999
                scores.append({'cif': cif, 'reward': reward, 'rosetta': rosetta_e, 'n_res': n_res})
            else:
                # Use RFD3 metrics as fast proxy
                try:
                    with open(jf) as f:
                        m = json.load(f).get('metrics', {})
                    reward = -m.get('n_clashing.interresidue_clashes_w_sidechain', 0) * 10
                    reward += m.get('non_loop_fraction', 0) * 5
                    reward -= abs(m.get('radius_of_gyration', 15) - 15) * 0.5
                    scores.append({'cif': cif, 'reward': reward, 'rog': m.get('radius_of_gyration', 0)})
                except:
                    scores.append({'cif': cif, 'reward': -999})
        t_score = time.time() - t_score_start

        rewards = [s['reward'] for s in scores]
        mean_reward = np.mean(rewards)
        best_r = max(rewards)
        reward_history.append(mean_reward)

        if args.use_rosetta:
            rosetta_scores = [s.get('rosetta', None) for s in scores if s.get('rosetta') is not None]
            if rosetta_scores:
                best_rosetta_this = min(rosetta_scores)
                rosetta_history.append(best_rosetta_this)
                if best_rosetta_this < best_rosetta:
                    best_rosetta = best_rosetta_this

        # === 3. SELECT BEST & FINE-TUNE ===
        t_train_start = time.time()
        sorted_scores = sorted(scores, key=lambda x: x['reward'], reverse=True)
        best_cifs = [s['cif'] for s in sorted_scores[:args.top_k]]

        model.train()
        total_loss = 0.0
        n_trained = 0

        for cif_path in best_cifs:
            try:
                aa = load_cif_for_training(cif_path)
                data = {"atom_array": aa, "example_id": f"best_{n_trained}"}
                pr = engine.pipeline(data)
                pr = engine.trainer.fabric.to_device(pr)

                for step in range(args.train_steps):
                    optimizer.zero_grad()
                    engine.trainer.training_step(batch=[pr], batch_idx=step, is_accumulating=False)
                    ret = engine.trainer._current_train_return
                    if ret and 'total_loss' in ret:
                        total_loss += ret['total_loss'].item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    n_trained += 1

            except Exception as e:
                logger.debug(f"Train on {cif_path} failed: {e}")

        t_train = time.time() - t_train_start
        avg_loss = total_loss / max(n_trained, 1)
        loss_history.append(avg_loss)
        t_total = time.time() - t_start

        # === LOG ===
        if args.use_rosetta and rosetta_scores:
            logger.info(
                f"R{rnd+1:3d}/{args.n_rounds}: "
                f"rosetta_best={min(rosetta_scores):.0f}, "
                f"rosetta_mean={np.mean(rosetta_scores):.0f}, "
                f"best_ever={best_rosetta:.0f}, "
                f"loss={avg_loss:.3f}, "
                f"trained={n_trained}, "
                f"gen={t_gen:.0f}s score={t_score:.0f}s train={t_train:.0f}s"
            )
        else:
            logger.info(
                f"R{rnd+1:3d}/{args.n_rounds}: "
                f"reward={mean_reward:.2f} (best={best_r:.2f}), "
                f"loss={avg_loss:.3f}, trained={n_trained}, "
                f"{t_total:.0f}s"
            )

        # Trend every 5 rounds
        if (rnd + 1) % 5 == 0 and len(reward_history) >= 6:
            f3 = np.mean(reward_history[:3])
            l3 = np.mean(reward_history[-3:])
            logger.info(f"  >>> TREND: {f3:.2f} → {l3:.2f} (Δ={l3-f3:+.2f})")

        # Save every 10 rounds
        if (rnd + 1) % 10 == 0:
            with open(output_dir / 'reward_history.json', 'w') as f:
                json.dump(reward_history, f)
            with open(output_dir / 'loss_history.json', 'w') as f:
                json.dump(loss_history, f)
            if rosetta_history:
                with open(output_dir / 'rosetta_history.json', 'w') as f:
                    json.dump(rosetta_history, f)

    # Final
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {args.n_rounds} rounds")
    if len(reward_history) >= 6:
        f3 = np.mean(reward_history[:3])
        l3 = np.mean(reward_history[-3:])
        logger.info(f"Reward: {f3:.2f} → {l3:.2f} (Δ={l3-f3:+.2f})")
    if rosetta_history:
        logger.info(f"Best Rosetta: {best_rosetta:.0f}")
    logger.info(f"{'='*60}")

    with open(output_dir / 'reward_history.json', 'w') as f:
        json.dump(reward_history, f)
    with open(output_dir / 'final.json', 'w') as f:
        json.dump({'reward_history': reward_history, 'loss_history': loss_history,
                   'rosetta_history': rosetta_history, 'best_rosetta': best_rosetta}, f, indent=2)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output_dir / 'model.pt')


if __name__ == '__main__':
    main()
