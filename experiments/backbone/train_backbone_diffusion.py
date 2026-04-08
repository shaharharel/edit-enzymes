"""Training script for backbone diffusion model.

Usage:
    python experiments/backbone/train_backbone_diffusion.py --backbone egnn
    python experiments/backbone/train_backbone_diffusion.py --backbone ipa --template data/pdb/1TIM.pdb
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from experiments.backbone.configs import BackboneExperimentConfig
from src.data.pdb_loader import load_pdb, load_pdb_all_chains
from src.data.dataset_builders import BackboneDiffusionDataset, create_dataloaders
from src.data.catalytic_constraints import ActiveSiteSpec, CatalyticConstraint, load_constraint_from_yaml
from src.data.protein_structure import ProteinBackbone
from src.models.backbone_generator.diffusion_model import SE3BackboneDiffusion, DiffusionConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_training_data(config: BackboneExperimentConfig):
    """Load PDB structures for training."""
    data_dir = Path(config.data_dir)
    pdb_files = sorted(data_dir.glob('*.pdb'))

    if not pdb_files:
        logger.warning(f"No PDB files found in {data_dir}. Using synthetic data.")
        return _make_synthetic_data(config)

    backbones = []
    for pdb_path in pdb_files:
        try:
            chains = load_pdb_all_chains(str(pdb_path))
            for bb in chains:
                if config.min_length <= bb.length <= config.max_length:
                    backbones.append(bb)
        except Exception as e:
            logger.warning(f"Failed to load {pdb_path}: {e}")

    logger.info(f"Loaded {len(backbones)} chains from {len(pdb_files)} PDB files")
    return backbones


def _make_synthetic_data(config: BackboneExperimentConfig, n_samples: int = 100):
    """Generate synthetic backbone data for testing the pipeline."""
    import numpy as np
    from src.utils.protein_constants import CA_CA_DISTANCE, BOND_LENGTHS

    backbones = []
    for i in range(n_samples):
        L = np.random.randint(config.min_length, min(config.max_length, 100))
        coords = np.zeros((L, 4, 3), dtype=np.float32)

        for j in range(L):
            # Extended chain with small random perturbations
            coords[j, 1, 0] = j * CA_CA_DISTANCE + np.random.normal(0, 0.1)
            coords[j, 1, 1] = np.random.normal(0, 0.3)
            coords[j, 1, 2] = np.random.normal(0, 0.3)

            coords[j, 0] = coords[j, 1] + np.array([-BOND_LENGTHS[('N', 'CA')], 0.2, 0]) + np.random.normal(0, 0.05, 3)
            coords[j, 2] = coords[j, 1] + np.array([BOND_LENGTHS[('CA', 'C')], -0.2, 0]) + np.random.normal(0, 0.05, 3)
            coords[j, 3] = coords[j, 2] + np.array([0, BOND_LENGTHS[('C', 'O')], 0]) + np.random.normal(0, 0.05, 3)

        backbones.append(ProteinBackbone(coords=coords, pdb_id=f'synthetic_{i}'))

    logger.info(f"Created {n_samples} synthetic backbones")
    return backbones


def main():
    parser = argparse.ArgumentParser(description='Train backbone diffusion model')
    parser.add_argument('--backbone', choices=['egnn', 'ipa'], default='egnn',
                        help='Equivariant backbone architecture')
    parser.add_argument('--template', type=str, default=None,
                        help='Template PDB for conditioned generation')
    parser.add_argument('--constraint', type=str, default=None,
                        help='Catalytic constraint YAML file')
    parser.add_argument('--fold-family', type=str, default=None,
                        help='Fold family constraint (e.g., TIM_barrel)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Build config
    config = BackboneExperimentConfig(
        equivariant_backbone=args.backbone,
        fold_family=args.fold_family,
        template_pdb=args.template,
        seed=args.seed,
    )
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    pl.seed_everything(config.seed)

    # Load data
    backbones = load_training_data(config)

    if not backbones:
        logger.error("No training data available")
        return

    # Build specs (if template/constraint provided)
    specs = None
    if args.constraint:
        constraint = load_constraint_from_yaml(args.constraint)
        template_bb = None
        if args.template:
            template_bb = load_pdb(args.template).coords
        specs = [
            ActiveSiteSpec(
                constraint=constraint,
                template_backbone=template_bb,
                fixed_residue_indices=[
                    r.position_index for r in constraint.residues
                    if r.position_index is not None
                ],
                noise_level=config.template_noise_scale,
            )
        ] * len(backbones)

    # Split data
    n_train = int(0.9 * len(backbones))
    train_backbones = backbones[:n_train]
    val_backbones = backbones[n_train:]

    train_specs = specs[:n_train] if specs else None
    val_specs = specs[n_train:] if specs else None

    # Create datasets
    train_dataset = BackboneDiffusionDataset(
        train_backbones, train_specs, max_length=config.max_length,
    )
    val_dataset = BackboneDiffusionDataset(
        val_backbones, val_specs, max_length=config.max_length,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Create model
    model_config = DiffusionConfig(
        equivariant_backbone=config.equivariant_backbone,
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        n_heads=config.n_heads,
        n_query_points=config.n_query_points,
        n_value_points=config.n_value_points,
        pair_dim=config.pair_dim,
        schedule_type=config.schedule_type,
        T=config.T,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        constraint_loss_weight=config.constraint_loss_weight,
        bond_loss_weight=config.bond_loss_weight,
        template_noise_scale=config.template_noise_scale,
    )
    model = SE3BackboneDiffusion(model_config)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.equivariant_backbone.upper()}, {param_count:,} parameters")

    # Callbacks
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=f'{config.equivariant_backbone}-{{epoch}}-{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
        ),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        default_root_dir=config.results_dir,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
    )

    logger.info(f"Starting training: {config.max_epochs} epochs, "
                f"batch_size={config.batch_size}")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == '__main__':
    main()
