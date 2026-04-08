"""Configuration for backbone diffusion experiments."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class BackboneExperimentConfig:
    """Configuration for backbone generator training."""

    # Data
    data_dir: str = 'data/pdb'
    cache_dir: str = 'cache/structure_features'
    catalytic_sites_dir: str = 'data/catalytic_sites'
    max_length: int = 256
    min_length: int = 50

    # Model architecture
    equivariant_backbone: Literal['egnn', 'ipa'] = 'egnn'
    node_dim: int = 256
    edge_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 6
    dropout: float = 0.1

    # IPA-specific
    n_heads: int = 8
    n_query_points: int = 4
    n_value_points: int = 4
    pair_dim: int = 64

    # Diffusion
    schedule_type: Literal['linear', 'cosine', 'polynomial'] = 'cosine'
    T: int = 1000
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    # Training
    batch_size: int = 8
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    constraint_loss_weight: float = 10.0
    bond_loss_weight: float = 1.0
    template_noise_scale: float = 0.1

    # Sampling
    n_sampling_steps: int = 100

    # Infrastructure
    num_workers: int = 4
    seed: int = 42
    results_dir: str = 'results/backbone'
    checkpoint_dir: str = 'results/backbone/checkpoints'

    # Fold family constraint (optional)
    fold_family: Optional[str] = None
    template_pdb: Optional[str] = None
