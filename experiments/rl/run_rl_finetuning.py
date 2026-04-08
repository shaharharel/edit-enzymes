"""Main script for RL-based enzyme design finetuning.

Usage:
    # With pretrained models:
    python -m experiments.rl.run_rl_finetuning \
        --backbone-ckpt path/to/backbone.ckpt \
        --sequence-ckpt path/to/sequence.ckpt

    # Synthetic mode (no pretrained models needed):
    python -m experiments.rl.run_rl_finetuning --synthetic --n-iterations 3

    # With custom config:
    python -m experiments.rl.run_rl_finetuning --synthetic \
        --n-residues 50 --n-iterations 5 --rollouts-per-update 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.rl.configs import RLExperimentConfig
from src.data.catalytic_constraints import (
    ActiveSiteSpec,
    CatalyticConstraint,
    CatalyticResidue,
)
from src.data.protein_structure import ProteinBackbone
from src.models.backbone_generator.diffusion_model import (
    DiffusionConfig,
    SE3BackboneDiffusion,
)
from src.models.rl.backbone_policy import BackbonePolicy
from src.models.rl.ppo_trainer import RLTrainer
from src.models.rl.reward import RewardFunction
from src.models.rl.sequence_policy import SequencePolicy
from src.models.scoring.multi_objective import MultiObjectiveScorer
from src.models.scoring.stability_scorer import StabilityScorerMLP
from src.models.scoring.packing_scorer import PackingScorerMLP
from src.models.sequence_generator.mpnn_model import MPNNConfig, ProteinMPNNModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_synthetic_spec(n_residues: int) -> ActiveSiteSpec:
    """Create a synthetic active site specification for testing.

    Builds a simple catalytic triad (Ser-His-Asp) with a random
    template backbone.

    Args:
        n_residues: Number of residues in the template.

    Returns:
        ActiveSiteSpec with synthetic constraint and template.
    """
    from src.utils.protein_constants import CA_CA_DISTANCE, BOND_LENGTHS

    # Create catalytic residues at specific positions
    cat_positions = [10, 25, 40] if n_residues >= 50 else [2, 5, 8]

    residues = [
        CatalyticResidue(
            residue_type='SER',
            atom_positions={'CA': np.array([cat_positions[0] * CA_CA_DISTANCE, 0.5, 0.0])},
            role='nucleophile',
            position_index=cat_positions[0],
        ),
        CatalyticResidue(
            residue_type='HIS',
            atom_positions={'CA': np.array([cat_positions[1] * CA_CA_DISTANCE, -0.5, 0.0])},
            role='general_base',
            position_index=cat_positions[1],
        ),
        CatalyticResidue(
            residue_type='ASP',
            atom_positions={'CA': np.array([cat_positions[2] * CA_CA_DISTANCE, 0.3, 0.2])},
            role='general_acid',
            position_index=cat_positions[2],
        ),
    ]

    constraint = CatalyticConstraint(
        residues=residues,
        pairwise_distances={(0, 1): 5.0, (1, 2): 4.5},
        fold_family='TIM_barrel',
    )

    # Create a simple template backbone
    template = np.zeros((n_residues, 4, 3), dtype=np.float32)
    for i in range(n_residues):
        y_offset = 0.5 * ((-1) ** i)
        # CA
        template[i, 1, 0] = i * CA_CA_DISTANCE
        template[i, 1, 1] = y_offset
        # N
        template[i, 0, 0] = template[i, 1, 0] - BOND_LENGTHS[('N', 'CA')]
        template[i, 0, 1] = y_offset + 0.3
        template[i, 0, 2] = 0.2
        # C
        template[i, 2, 0] = template[i, 1, 0] + BOND_LENGTHS[('CA', 'C')]
        template[i, 2, 1] = y_offset - 0.3
        template[i, 2, 2] = -0.2
        # O
        template[i, 3, 0] = template[i, 2, 0]
        template[i, 3, 1] = template[i, 2, 1] + BOND_LENGTHS[('C', 'O')]
        template[i, 3, 2] = template[i, 2, 2]

    spec = ActiveSiteSpec(
        constraint=constraint,
        template_backbone=template,
        fixed_residue_indices=cat_positions,
        noise_level=0.1,
    )

    return spec


def create_synthetic_components(
    config: RLExperimentConfig,
) -> tuple:
    """Create synthetic (untrained) model components for testing.

    Returns:
        Tuple of (backbone_generator, sequence_generator, scorer).
    """
    # Backbone generator
    diff_config = DiffusionConfig(
        equivariant_backbone=config.equivariant_backbone,
        node_dim=config.backbone_node_dim,
        edge_dim=config.backbone_edge_dim,
        hidden_dim=config.backbone_hidden_dim,
        n_layers=config.backbone_n_layers,
        T=config.backbone_T,
        schedule_type=config.backbone_schedule_type,
    )
    backbone_gen = SE3BackboneDiffusion(diff_config)

    # Sequence generator
    mpnn_config = MPNNConfig(
        hidden_dim=config.mpnn_hidden_dim,
        encoder_layers=config.mpnn_encoder_layers,
        decoder_layers=config.mpnn_decoder_layers,
    )
    seq_gen = ProteinMPNNModel(mpnn_config)

    # Scoring models
    scorers = {}
    for name in config.scorer_weights:
        if name == 'stability':
            scorers[name] = StabilityScorerMLP(input_dim=config.feature_dim)
        elif name == 'packing':
            scorers[name] = PackingScorerMLP(input_dim=config.feature_dim)
        else:
            # Default to stability scorer architecture
            scorers[name] = StabilityScorerMLP(input_dim=config.feature_dim)

    scorer = MultiObjectiveScorer(scorers, config.scorer_weights)

    return backbone_gen, seq_gen, scorer


def build_trainer(
    config: RLExperimentConfig,
    backbone_gen: SE3BackboneDiffusion,
    seq_gen: ProteinMPNNModel,
    scorer: MultiObjectiveScorer,
    spec: ActiveSiteSpec,
) -> RLTrainer:
    """Build the RL trainer from components.

    Args:
        config: Experiment configuration.
        backbone_gen: Backbone generator.
        seq_gen: Sequence generator.
        scorer: Multi-objective scorer.
        spec: Active site specification.

    Returns:
        Configured RLTrainer.
    """
    # Wrap generators as RL policies
    backbone_policy = BackbonePolicy(
        generator=backbone_gen,
        learning_rate=config.backbone_lr,
        baseline_learning_rate=config.baseline_lr,
        max_grad_norm=config.max_grad_norm,
    )

    sequence_policy = SequencePolicy(
        model=seq_gen,
        learning_rate=config.sequence_lr,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_coeff,
        value_coeff=config.value_coeff,
        max_grad_norm=config.max_grad_norm,
    )

    # Create reward function
    reward_fn = RewardFunction(
        scorer=scorer,
        constraint=spec.constraint,
        backbone_reward_weight=config.backbone_reward_weight,
        sequence_reward_weight=config.sequence_reward_weight,
        constraint_reward_scale=config.constraint_reward_scale,
        geometry_reward_scale=config.geometry_reward_scale,
    )

    # Create trainer
    trainer = RLTrainer(
        backbone_policy=backbone_policy,
        sequence_policy=sequence_policy,
        reward_fn=reward_fn,
        spec=spec,
        n_residues=config.n_residues,
        n_diffusion_steps=config.n_diffusion_steps,
        sampling_temperature=config.sampling_temperature,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        rollouts_per_update=config.rollouts_per_update,
        ppo_epochs=config.ppo_epochs,
        backbone_update_frequency=config.backbone_update_frequency,
        max_grad_norm=config.max_grad_norm,
        feature_dim=config.feature_dim,
        device=config.device,
    )

    return trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='RL finetuning for enzyme design',
    )

    # Mode
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic (untrained) components for testing',
    )

    # Model paths
    parser.add_argument('--backbone-ckpt', type=str, default=None)
    parser.add_argument('--sequence-ckpt', type=str, default=None)

    # Key hyperparameters (override config defaults)
    parser.add_argument('--n-residues', type=int, default=None)
    parser.add_argument('--n-iterations', type=int, default=None)
    parser.add_argument('--rollouts-per-update', type=int, default=None)
    parser.add_argument('--ppo-epochs', type=int, default=None)
    parser.add_argument('--n-diffusion-steps', type=int, default=None)
    parser.add_argument('--sampling-temperature', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--results-dir', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = RLExperimentConfig()

    # Apply CLI overrides
    if args.n_residues is not None:
        config.n_residues = args.n_residues
    if args.n_iterations is not None:
        config.n_iterations = args.n_iterations
    if args.rollouts_per_update is not None:
        config.rollouts_per_update = args.rollouts_per_update
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.n_diffusion_steps is not None:
        config.n_diffusion_steps = args.n_diffusion_steps
    if args.sampling_temperature is not None:
        config.sampling_temperature = args.sampling_temperature
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.backbone_ckpt is not None:
        config.backbone_checkpoint = args.backbone_ckpt
    if args.sequence_ckpt is not None:
        config.sequence_checkpoint = args.sequence_ckpt

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    logger.info(f"RL Finetuning Config: {config}")

    # Create or load components
    if args.synthetic:
        logger.info("Using synthetic (untrained) components for testing")
        backbone_gen, seq_gen, scorer = create_synthetic_components(config)
        spec = create_synthetic_spec(config.n_residues)
    else:
        if not config.backbone_checkpoint or not config.sequence_checkpoint:
            logger.error(
                "Pretrained model paths required. "
                "Use --synthetic for testing without pretrained models."
            )
            sys.exit(1)

        # Load pretrained models
        logger.info(f"Loading backbone generator from {config.backbone_checkpoint}")
        backbone_gen = SE3BackboneDiffusion.load_from_checkpoint(
            config.backbone_checkpoint,
        )

        logger.info(f"Loading sequence generator from {config.sequence_checkpoint}")
        seq_gen = ProteinMPNNModel.load_from_checkpoint(
            config.sequence_checkpoint,
        )

        # Load scorers
        scorers = {}
        for name, path in config.scorer_checkpoints.items():
            logger.info(f"Loading {name} scorer from {path}")
            if name == 'stability':
                scorers[name] = StabilityScorerMLP.load_from_checkpoint(path)
            elif name == 'packing':
                scorers[name] = PackingScorerMLP.load_from_checkpoint(path)
        scorer = MultiObjectiveScorer(scorers, config.scorer_weights)

        # Create spec (would normally come from a constraint file)
        spec = create_synthetic_spec(config.n_residues)

    # Build trainer
    trainer = build_trainer(config, backbone_gen, seq_gen, scorer, spec)

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Run training
    history = trainer.train(config.n_iterations)

    # Save results
    results_path = os.path.join(config.results_dir, 'training_history.json')
    serializable_history = [
        {k: float(v) for k, v in step.items()} for step in history
    ]
    with open(results_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Training history saved to {results_path}")

    # Save best designs
    best_designs = trainer.get_best_designs(config.save_top_k)
    for i, (backbone, sequence, reward) in enumerate(best_designs):
        design_path = os.path.join(config.results_dir, f'best_design_{i}.npz')
        np.savez(
            design_path,
            coords=backbone.coords,
            sequence=np.array(list(sequence)),
            reward=np.array([reward]),
        )
        logger.info(
            f"Design {i}: reward={reward:.4f}, "
            f"length={backbone.length}, "
            f"sequence={sequence[:20]}..."
        )

    logger.info("RL finetuning complete.")


if __name__ == '__main__':
    main()
