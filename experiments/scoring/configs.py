"""Configuration for scoring model experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScoringExperimentConfig:
    """Configuration for scoring surrogate model training."""

    # Data
    data_dir: str = 'data/pdb'
    cache_dir: str = 'cache/rosetta_features'
    features_csv: Optional[str] = None  # pre-computed features CSV
    input_dim: int = 256

    # Which scorers to train
    scorers: List[str] = field(default_factory=lambda: [
        'stability', 'packing', 'desolvation', 'activity',
    ])

    # Model architecture
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 15
    val_fraction: float = 0.15

    # Multi-objective weights (for combined scoring)
    stability_weight: float = 1.0
    packing_weight: float = 0.5
    desolvation_weight: float = 0.3
    activity_weight: float = 1.0

    # Infrastructure
    num_workers: int = 4
    seed: int = 42
    results_dir: str = 'results/scoring'
    checkpoint_dir: str = 'results/scoring/checkpoints'

    # Synthetic data (for testing without Rosetta)
    use_synthetic: bool = False
    synthetic_n_samples: int = 2000
