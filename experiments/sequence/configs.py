"""Configuration for sequence design experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SequenceExperimentConfig:
    """Configuration for sequence generator training."""

    # Data
    data_dir: str = 'data/pdb'
    max_length: int = 256
    min_length: int = 20
    k_neighbors: int = 30

    # Model architecture
    node_input_dim: int = 46
    edge_input_dim: int = 17
    hidden_dim: int = 128
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    # Sampling
    sample_temperature: float = 0.1

    # Infrastructure
    num_workers: int = 4
    seed: int = 42
    results_dir: str = 'results/sequence'
    checkpoint_dir: str = 'results/sequence/checkpoints'

    # Optional: fixed catalytic residues
    constraint_yaml: Optional[str] = None
