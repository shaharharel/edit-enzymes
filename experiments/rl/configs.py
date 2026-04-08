"""Configuration for RL finetuning experiments."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


@dataclass
class RLExperimentConfig:
    """Configuration for RL-based enzyme design optimization.

    Covers pretrained model paths, RL hyperparameters, reward weights,
    and training loop settings.
    """

    # Pretrained model paths
    backbone_checkpoint: Optional[str] = None
    sequence_checkpoint: Optional[str] = None
    scorer_checkpoints: Dict[str, str] = field(default_factory=dict)

    # Backbone generator config (used when --synthetic)
    equivariant_backbone: Literal['egnn', 'ipa'] = 'egnn'
    backbone_node_dim: int = 256
    backbone_edge_dim: int = 64
    backbone_hidden_dim: int = 256
    backbone_n_layers: int = 6
    backbone_T: int = 1000
    backbone_schedule_type: Literal['linear', 'cosine', 'polynomial'] = 'cosine'

    # Sequence generator config (used when --synthetic)
    mpnn_hidden_dim: int = 128
    mpnn_encoder_layers: int = 3
    mpnn_decoder_layers: int = 3

    # RL hyperparameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5

    # Learning rates
    backbone_lr: float = 1e-5
    sequence_lr: float = 1e-5
    baseline_lr: float = 1e-4

    # Reward weights
    backbone_reward_weight: float = 1.0
    sequence_reward_weight: float = 1.0
    constraint_reward_scale: float = 5.0
    geometry_reward_scale: float = 2.0
    scorer_weights: Dict[str, float] = field(default_factory=lambda: {
        'stability': 1.0,
        'packing': 0.5,
    })

    # Training loop
    n_iterations: int = 100
    rollouts_per_update: int = 8
    ppo_epochs: int = 4
    backbone_update_frequency: int = 2
    max_grad_norm: float = 1.0

    # Generation
    n_residues: int = 100
    n_diffusion_steps: int = 50
    sampling_temperature: float = 0.3
    feature_dim: int = 256

    # Infrastructure
    device: str = 'cpu'
    seed: int = 42
    results_dir: str = 'results/rl'
    save_top_k: int = 10
