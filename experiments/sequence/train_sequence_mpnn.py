"""Training script for ProteinMPNN-style sequence generator.

Usage:
    python experiments/sequence/train_sequence_mpnn.py
    python experiments/sequence/train_sequence_mpnn.py --epochs 100 --hidden-dim 256
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from experiments.sequence.configs import SequenceExperimentConfig
from src.data.dataset_builders import SequenceDesignDataset, create_dataloaders
from src.data.protein_structure import ProteinBackbone
from src.models.sequence_generator import ProteinMPNNModel, MPNNConfig, backbone_to_graph_features
from src.utils.logging import get_logger
from src.utils.protein_constants import CA_CA_DISTANCE, BOND_LENGTHS, AA_LIST, AA_3TO1

logger = get_logger(__name__)


def load_training_data(config: SequenceExperimentConfig):
    """Load PDB structures for training."""
    data_dir = Path(config.data_dir)
    pdb_files = sorted(data_dir.glob('*.pdb'))

    if not pdb_files:
        logger.warning(f"No PDB files found in {data_dir}. Using synthetic data.")
        return _make_synthetic_data(config)

    from src.data.pdb_loader import load_pdb_all_chains

    backbones = []
    for pdb_path in pdb_files:
        try:
            chains = load_pdb_all_chains(str(pdb_path))
            for bb in chains:
                if (config.min_length <= bb.length <= config.max_length
                        and bb.sequence is not None):
                    backbones.append(bb)
        except Exception as e:
            logger.warning(f"Failed to load {pdb_path}: {e}")

    logger.info(f"Loaded {len(backbones)} chains with sequences from {len(pdb_files)} PDB files")

    if not backbones:
        logger.warning("No valid chains found. Falling back to synthetic data.")
        return _make_synthetic_data(config)

    return backbones


def _make_synthetic_data(config: SequenceExperimentConfig, n_samples: int = 100):
    """Generate synthetic backbone data with random sequences for testing."""
    aa_letters = [AA_3TO1[aa] for aa in AA_LIST]
    backbones = []

    for i in range(n_samples):
        L = np.random.randint(config.min_length, min(config.max_length, 80))
        coords = np.zeros((L, 4, 3), dtype=np.float32)

        for j in range(L):
            coords[j, 1, 0] = j * CA_CA_DISTANCE + np.random.normal(0, 0.1)
            coords[j, 1, 1] = np.random.normal(0, 0.3)
            coords[j, 1, 2] = np.random.normal(0, 0.3)

            coords[j, 0] = coords[j, 1] + np.array(
                [-BOND_LENGTHS[('N', 'CA')], 0.2, 0]
            ) + np.random.normal(0, 0.05, 3)
            coords[j, 2] = coords[j, 1] + np.array(
                [BOND_LENGTHS[('CA', 'C')], -0.2, 0]
            ) + np.random.normal(0, 0.05, 3)
            coords[j, 3] = coords[j, 2] + np.array(
                [0, BOND_LENGTHS[('C', 'O')], 0]
            ) + np.random.normal(0, 0.05, 3)

        sequence = ''.join(np.random.choice(aa_letters, size=L))

        backbones.append(ProteinBackbone(
            coords=coords,
            sequence=sequence,
            pdb_id=f'synthetic_{i}',
        ))

    logger.info(f"Created {n_samples} synthetic backbones with random sequences")
    return backbones


def main():
    parser = argparse.ArgumentParser(description='Train ProteinMPNN sequence generator')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--encoder-layers', type=int, default=None)
    parser.add_argument('--decoder-layers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=None)
    args = parser.parse_args()

    # Build config
    config = SequenceExperimentConfig(seed=args.seed)
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.encoder_layers:
        config.encoder_layers = args.encoder_layers
    if args.decoder_layers:
        config.decoder_layers = args.decoder_layers
    if args.temperature:
        config.sample_temperature = args.temperature

    pl.seed_everything(config.seed)

    # Load data
    backbones = load_training_data(config)
    if not backbones:
        logger.error("No training data available")
        return

    # Split data
    n_train = int(0.9 * len(backbones))
    train_backbones = backbones[:n_train]
    val_backbones = backbones[n_train:]

    if not val_backbones:
        val_backbones = train_backbones[-5:]

    # Create datasets
    train_dataset = SequenceDesignDataset(
        train_backbones, max_length=config.max_length,
    )
    val_dataset = SequenceDesignDataset(
        val_backbones, max_length=config.max_length,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Create model
    model_config = MPNNConfig(
        node_input_dim=config.node_input_dim,
        edge_input_dim=config.edge_input_dim,
        hidden_dim=config.hidden_dim,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
    )
    model = ProteinMPNNModel(model_config)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ProteinMPNN model: {param_count:,} parameters")

    # Callbacks
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename='mpnn-{epoch}-{val_loss:.4f}-{val_recovery:.4f}',
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
                f"batch_size={config.batch_size}, hidden_dim={config.hidden_dim}")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {callbacks[0].best_model_path}")

    # Sample a sequence from the first validation backbone
    if val_backbones:
        bb = val_backbones[0]
        graph = backbone_to_graph_features(bb, k=config.k_neighbors)
        sampled_seq = model.sample(graph, temperature=config.sample_temperature)
        logger.info(f"Sample sequence (L={bb.length}): {sampled_seq}")
        if bb.sequence:
            matches = sum(
                1 for a, b in zip(sampled_seq, bb.sequence) if a == b
            )
            logger.info(f"Recovery: {matches}/{bb.length} = {matches/bb.length:.2%}")


if __name__ == '__main__':
    main()
