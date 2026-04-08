"""Stability scoring model -- predicts ddG (stability change).

Learned surrogate for Rosetta ddG calculations. Takes pre-computed
structural/sequence features and predicts the change in free energy
of unfolding upon mutation.
"""

import torch
import torch.nn as nn

from src.models.scoring.base import AbstractScoringModel


class StabilityScorerMLP(AbstractScoringModel):
    """MLP surrogate for ddG stability prediction.

    Architecture: [input_dim] -> [512] -> [256] -> [128] -> [1]
    with GELU activations and dropout between layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @property
    def score_name(self) -> str:
        return 'stability'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict ddG from features.

        Args:
            features: (B, input_dim) pre-computed feature vectors

        Returns:
            ddg: (B, 1) predicted ddG values
        """
        return self.mlp(features)
