"""Activity scoring model.

Learned surrogate for enzymatic activity prediction. Predicts an
activity proxy score that can be refined with experimental data
(e.g., kcat, kcat/KM) once available.
"""

import torch
import torch.nn as nn

from src.models.scoring.base import AbstractScoringModel


class ActivityScorerMLP(AbstractScoringModel):
    """MLP surrogate for activity proxy prediction.

    Architecture: [input_dim] -> [512] -> [256] -> [128] -> [1]
    with GELU activations and dropout between layers.

    Initially trained on computational proxies (e.g., Rosetta ligand
    binding scores), can be fine-tuned on experimental activity data.
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
        return 'activity'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict activity proxy from features.

        Args:
            features: (B, input_dim) pre-computed feature vectors

        Returns:
            activity_score: (B, 1) predicted activity proxy
        """
        return self.mlp(features)
