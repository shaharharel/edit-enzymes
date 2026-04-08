"""Abstract base class for scoring models.

Scoring models are learned surrogates that approximate Rosetta/PROSS
computations as fast neural networks. Each scorer predicts a single
scalar value from pre-computed structural/sequence features.
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AbstractScoringModel(pl.LightningModule, ABC):
    """Base class for all scoring surrogate models.

    All scorers must implement:
    - forward: Map feature vector to scalar score prediction

    Provides shared training/validation logic with MSE loss.
    """

    def __init__(self, learning_rate: float = 1e-4, weight_decay: float = 1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict a scalar score from input features.

        Args:
            features: (B, d) pre-computed feature vectors

        Returns:
            scores: (B,) predicted scalar scores
        """
        ...

    @property
    @abstractmethod
    def score_name(self) -> str:
        """Name of the score this model predicts (used as key in batch dicts)."""
        ...

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Shared training step with MSE loss."""
        features = batch['features']
        targets = batch[self.score_name]

        predictions = self(features).squeeze(-1)
        loss = nn.functional.mse_loss(predictions, targets)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Shared validation step logging val_loss and val_mae."""
        features = batch['features']
        targets = batch[self.score_name]

        predictions = self(features).squeeze(-1)
        loss = nn.functional.mse_loss(predictions, targets)
        mae = torch.mean(torch.abs(predictions - targets))

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
            },
        }
