"""Training script for scoring surrogate models.

Trains MLP surrogates that approximate Rosetta energy computations.
Can use either pre-computed Rosetta features or synthetic data for testing.

Usage:
    conda run -n quris python experiments/scoring/train_scoring_models.py
    conda run -n quris python experiments/scoring/train_scoring_models.py --synthetic
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataset_builders import ScoringDataset, create_dataloaders
from src.models.scoring import (
    StabilityScorerMLP,
    PackingScorerMLP,
    DesolvationScorerMLP,
    ActivityScorerMLP,
    MultiObjectiveScorer,
)
from experiments.scoring.configs import ScoringExperimentConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCORER_CLASSES = {
    'stability': StabilityScorerMLP,
    'packing': PackingScorerMLP,
    'desolvation': DesolvationScorerMLP,
    'activity': ActivityScorerMLP,
}


def generate_synthetic_data(
    n_samples: int,
    input_dim: int,
    seed: int = 42,
) -> tuple:
    """Generate synthetic training data for testing the pipeline.

    Creates random features with correlated synthetic targets that
    mimic the structure of real Rosetta scores.

    Returns:
        (features, targets) where features is (N, d) tensor and
        targets is a dict of (N,) tensors.
    """
    rng = np.random.RandomState(seed)

    features = torch.tensor(rng.randn(n_samples, input_dim), dtype=torch.float32)

    # Create synthetic targets with some structure (linear + noise)
    W = torch.tensor(rng.randn(input_dim, 4), dtype=torch.float32) * 0.1
    base_scores = features @ W  # (N, 4)

    targets = {
        'stability': base_scores[:, 0] + torch.randn(n_samples) * 0.5,
        'packing': torch.sigmoid(base_scores[:, 1]) + torch.randn(n_samples) * 0.1,
        'desolvation': base_scores[:, 2] + torch.randn(n_samples) * 0.3,
        'activity': torch.relu(base_scores[:, 3]) + torch.randn(n_samples) * 0.2,
    }

    return features, targets


def load_cached_data(config: ScoringExperimentConfig) -> tuple:
    """Load pre-computed features from cache directory.

    Expects cache_dir to contain features.pt and targets.pt files,
    or a features.csv file with columns for each score.

    Returns:
        (features, targets) tuple.
    """
    import pandas as pd

    cache_dir = Path(config.cache_dir)

    # Try loading torch tensors first
    features_path = cache_dir / 'features.pt'
    targets_path = cache_dir / 'targets.pt'
    if features_path.exists() and targets_path.exists():
        features = torch.load(features_path, weights_only=False)
        targets = torch.load(targets_path, weights_only=False)
        logger.info(f"Loaded {features.shape[0]} samples from cache")
        return features, targets

    # Try CSV
    csv_path = config.features_csv or (cache_dir / 'rosetta_scores.csv')
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # Expect feature columns to be prefixed with 'feat_'
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        if not feat_cols:
            raise ValueError(f"No feature columns (feat_*) found in {csv_path}")

        features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        targets = {}
        for scorer_name in config.scorers:
            if scorer_name in df.columns:
                targets[scorer_name] = torch.tensor(
                    df[scorer_name].values, dtype=torch.float32
                )

        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        return features, targets

    raise FileNotFoundError(
        f"No cached data found in {cache_dir}. "
        f"Run scripts/data_prep/compute_rosetta_scores.py first, "
        f"or use --synthetic for testing."
    )


def train_scorer(
    scorer_name: str,
    features: torch.Tensor,
    targets: dict,
    config: ScoringExperimentConfig,
) -> pl.LightningModule:
    """Train a single scoring model.

    Args:
        scorer_name: Name of the scorer to train
        features: (N, d) feature tensor
        targets: Dict of (N,) target tensors
        config: Experiment configuration

    Returns:
        Trained model.
    """
    if scorer_name not in SCORER_CLASSES:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    if scorer_name not in targets:
        logger.warning(f"No target data for {scorer_name}, skipping")
        return None

    logger.info(f"Training {scorer_name} scorer...")

    # Create model
    model = SCORER_CLASSES[scorer_name](
        input_dim=config.input_dim,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Split data
    n = features.shape[0]
    n_val = int(n * config.val_fraction)
    n_train = n - n_val

    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_targets = {k: v[train_idx] for k, v in targets.items()}
    val_targets = {k: v[val_idx] for k, v in targets.items()}

    train_dataset = ScoringDataset(features[train_idx], train_targets)
    val_dataset = ScoringDataset(features[val_idx], val_targets)

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Callbacks
    checkpoint_dir = Path(config.checkpoint_dir) / scorer_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f'{scorer_name}-{{epoch:03d}}-{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            mode='min',
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        default_root_dir=config.results_dir,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_path = callbacks[0].best_model_path
    if best_path:
        logger.info(f"Best checkpoint: {best_path}")
        model = SCORER_CLASSES[scorer_name].load_from_checkpoint(best_path)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train scoring surrogate models')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--scorers', nargs='+', default=None,
                        help='Which scorers to train (default: all)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    config = ScoringExperimentConfig()

    if args.synthetic:
        config.use_synthetic = True
    if args.scorers:
        config.scorers = args.scorers
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    pl.seed_everything(config.seed)

    # Load data
    if config.use_synthetic:
        logger.info("Generating synthetic data for testing...")
        features, targets = generate_synthetic_data(
            config.synthetic_n_samples, config.input_dim, config.seed,
        )
    else:
        features, targets = load_cached_data(config)

    logger.info(f"Data shape: {features.shape}")
    logger.info(f"Targets: {list(targets.keys())}")

    # Train each scorer
    trained_models = {}
    for scorer_name in config.scorers:
        model = train_scorer(scorer_name, features, targets, config)
        if model is not None:
            trained_models[scorer_name] = model

    # Build multi-objective scorer
    if len(trained_models) > 1:
        weights = {
            'stability': config.stability_weight,
            'packing': config.packing_weight,
            'desolvation': config.desolvation_weight,
            'activity': config.activity_weight,
        }
        # Only include weights for trained scorers
        weights = {k: v for k, v in weights.items() if k in trained_models}

        multi_scorer = MultiObjectiveScorer(trained_models, weights)

        # Quick test
        test_features = features[:8]
        with torch.no_grad():
            multi_scores = multi_scorer(test_features)
        logger.info(f"Multi-objective test scores: "
                     f"total={multi_scores['total'].mean():.4f}")
        for name in trained_models:
            logger.info(f"  {name}: {multi_scores[name].mean():.4f}")

        # Save multi-objective scorer
        multi_path = Path(config.checkpoint_dir) / 'multi_objective_scorer.pt'
        multi_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(multi_scorer.state_dict(), multi_path)
        logger.info(f"Saved multi-objective scorer to {multi_path}")

    logger.info("Scoring model training complete.")


if __name__ == '__main__':
    main()
