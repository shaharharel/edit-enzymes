"""Packing quality scoring model.

Learned surrogate for Rosetta packing quality evaluation. Predicts
how well residues are packed in the protein core, approximating
the Rosetta PackStat score.
"""

import torch
import torch.nn as nn

from src.models.scoring.base import AbstractScoringModel


class PackingScorerMLP(AbstractScoringModel):
    """MLP surrogate for packing quality prediction.

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
        return 'packing'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict packing quality from features.

        Args:
            features: (B, input_dim) pre-computed feature vectors

        Returns:
            packing_score: (B, 1) predicted packing quality
        """
        return self.mlp(features)
