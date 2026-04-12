"""DDPO V1: True per-step PPO on our custom SE3BackboneDiffusion.

This is the mathematically correct DDPO implementation using our own
differentiable diffusion model. Tests on T4/V100/MPS.

Usage:
    python experiments/rl/run_ddpo_v1.py \
        --template data/pdb_clean/2RKX.pdb \
        --constraint data/catalytic_sites/2RKX.yaml \
        --n-iterations 50 --device cuda
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import numpy as np

from src.data.pdb_loader import load_pdb
from src.data.catalytic_constraints import load_constraint_from_yaml, ActiveSiteSpec
from src.data.protein_structure import ProteinBackbone
from src.models.backbone_generator.diffusion_model import SE3BackboneDiffusion, DiffusionConfig
from src.models.rl.ddpo_policy_v1 import DDPOPolicyV1
from src.models.rl.ddpo_trainer import DDPOTrainer
from src.utils.metrics import bond_geometry_metrics, clash_score
from src.utils.geometry import kabsch_rmsd
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_reward_fn(template: ProteinBackbone):
    """Create reward function based on structural quality.

    For V1 (no Rosetta in the loop), we use structural metrics:
    - Bond geometry quality
    - Clash score
    - RMSD to template
    - Compactness
    """
    def reward_fn(design: ProteinBackbone) -> float:
        coords = torch.tensor(design.coords, dtype=torch.float32)
        reward = 0.0

        # Bond geometry (lower deviation = better)
        geom = bond_geometry_metrics(coords)
        bond_dev = geom.get('n_ca_bond_deviation', 1.0)
        reward -= bond_dev * 10.0

        # Clash score (lower = better)
        clash = clash_score(coords)
        reward -= clash * 20.0

        # RMSD to template (sweet spot 0.5-2.0Å)
        min_len = min(design.length, template.length)
        if min_len > 5:
            ca_gen = torch.tensor(design.ca_coords[:min_len], dtype=torch.float32)
            ca_tmpl = torch.tensor(template.ca_coords[:min_len], dtype=torch.float32)
            try:
                rmsd, _, _ = kabsch_rmsd(ca_tmpl, ca_gen)
                rmsd_val = float(rmsd)
                if 0.5 < rmsd_val < 2.0:
                    reward += 2.0  # sweet spot
                elif rmsd_val > 3.0:
                    reward -= (rmsd_val - 3.0) * 2.0
            except:
                pass

        # Compactness (radius of gyration)
        ca = design.ca_coords
        if len(ca) > 5:
            centroid = ca.mean(axis=0)
            rog = np.sqrt(np.mean(np.sum((ca - centroid) ** 2, axis=-1)))
            if 10 < rog < 20:
                reward += 1.0
            else:
                reward -= abs(rog - 15) * 0.2

        return reward

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description='DDPO V1')
    parser.add_argument('--template', required=True)
    parser.add_argument('--constraint', required=True)
    parser.add_argument('--n-iterations', type=int, default=50)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='results/ddpo_v1')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_pdb(args.template)
    constraint = load_constraint_from_yaml(args.constraint)

    logger.info("="*60)
    logger.info("DDPO V1: True Per-Step PPO on Custom Diffusion Model")
    logger.info(f"Template: {template.pdb_id}, {template.length} residues")
    logger.info(f"Device: {args.device}")
    logger.info(f"Iterations: {args.n_iterations}, rollouts: {args.rollouts}")
    logger.info(f"PPO epochs: {args.ppo_epochs}, denoising steps: {args.n_steps}")
    logger.info("="*60)

    # Build model
    config = DiffusionConfig(
        equivariant_backbone='egnn',
        node_dim=128, hidden_dim=128, n_layers=3,
    )
    generator = SE3BackboneDiffusion(config).to(args.device)

    # Load pretrained checkpoint if available
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        generator.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        ckpt_dir = Path('results/backbone/checkpoints')
        egnn_ckpts = sorted(ckpt_dir.glob('egnn-*.ckpt'))
        if egnn_ckpts:
            ckpt = torch.load(egnn_ckpts[-1], map_location=args.device, weights_only=False)
            generator.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
            logger.info(f"Loaded pretrained: {egnn_ckpts[-1]}")

    for p in generator.parameters():
        p.requires_grad_(True)

    params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {params:,}")

    # DDPO policy
    policy = DDPOPolicyV1(generator)

    # Active site spec
    spec = ActiveSiteSpec(
        constraint=constraint,
        template_backbone=template.coords,
        fixed_residue_indices=[
            r.position_index for r in constraint.residues
            if r.position_index is not None
        ],
        noise_level=0.1,
    )

    # Reward function
    reward_fn = make_reward_fn(template)

    # DDPO trainer
    n_residues = min(template.length, 100)
    trainer = DDPOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        spec=spec,
        n_residues=n_residues,
        n_denoising_steps=args.n_steps,
        rollouts_per_update=args.rollouts,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=0.2,
        learning_rate=args.lr,
        device=args.device,
    )

    # Training loop
    for iteration in range(args.n_iterations):
        metrics = trainer.train_step(iteration)

        if 'error' in metrics:
            logger.warning(f"Iter {iteration}: {metrics['error']}")
            continue

        logger.info(
            f"Iter {iteration+1}/{args.n_iterations}: "
            f"reward={metrics['mean_reward']:.3f} (best={metrics['best_reward']:.3f}), "
            f"ppo_loss={metrics['ppo_loss']:.4f}, "
            f"kl={metrics['approx_kl']:.4f}, "
            f"best_ever={metrics['best_ever']:.3f}, "
            f"collect={metrics['t_collect']:.1f}s update={metrics['t_update']:.1f}s"
        )

        # Trend every 5 iterations
        if (iteration + 1) % 5 == 0 and len(trainer.reward_history) >= 6:
            f3 = np.mean(trainer.reward_history[:3])
            l3 = np.mean(trainer.reward_history[-3:])
            logger.info(f"  >>> TREND: {f3:.3f} → {l3:.3f} (Δ={l3-f3:+.3f})")

        # Save every 10 iterations
        if (iteration + 1) % 10 == 0:
            trainer.save_results(str(output_dir))
            policy.save_checkpoint(str(output_dir / f'model_iter_{iteration+1}.pt'))

    # Final
    logger.info(f"\n{'='*60}")
    logger.info(f"DDPO V1 COMPLETE: {args.n_iterations} iterations")
    if len(trainer.reward_history) >= 6:
        f3 = np.mean(trainer.reward_history[:3])
        l3 = np.mean(trainer.reward_history[-3:])
        logger.info(f"Reward: {f3:.3f} → {l3:.3f} (Δ={l3-f3:+.3f})")
    logger.info(f"Best reward: {trainer.best_reward:.3f}")
    logger.info("="*60)

    trainer.save_results(str(output_dir))
    policy.save_checkpoint(str(output_dir / 'model_final.pt'))


if __name__ == '__main__':
    main()
