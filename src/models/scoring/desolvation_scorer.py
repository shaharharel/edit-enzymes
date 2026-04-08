"""Desolvation scoring model.

Learned surrogate for desolvation energy around the active site.
Predicts the energetic cost of removing water molecules from the
active site cavity upon substrate binding.
"""

import torch
import torch.nn as nn

from src.models.scoring.base import AbstractScoringModel


class DesolvationScorerMLP(AbstractScoringModel):
    """MLP surrogate for desolvation energy prediction.

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
        return 'desolvation'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict desolvation energy from features.

        Args:
            features: (B, input_dim) pre-computed feature vectors

        Returns:
            desolvation_score: (B, 1) predicted desolvation energy
        """
        return self.mlp(features)
