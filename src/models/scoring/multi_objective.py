"""Multi-objective scoring combiner.

Aggregates multiple scoring models into a single weighted objective.
Used during optimization to balance stability, packing, desolvation,
and activity criteria.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.models.scoring.base import AbstractScoringModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MultiObjectiveScorer(nn.Module):
    """Combines multiple scoring models into a weighted objective.

    Takes a dictionary of named AbstractScoringModel instances and
    corresponding weights, computes each score, and returns per-scorer
    values plus a weighted total.
    """

    def __init__(
        self,
        scorers: Dict[str, AbstractScoringModel],
        weights: Dict[str, float],
    ):
        """
        Args:
            scorers: Dict mapping score names to trained scoring models
            weights: Dict mapping score names to combination weights
        """
        super().__init__()
        self.scorers = nn.ModuleDict(scorers)
        self.weights = dict(weights)

        # Validate that all scorers have weights
        for name in scorers:
            if name not in self.weights:
                logger.warning(f"No weight for scorer '{name}', defaulting to 1.0")
                self.weights[name] = 1.0

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all scores and weighted total.

        Args:
            features: (B, d) pre-computed feature vectors

        Returns:
            Dict with per-scorer values (B,) and 'total' weighted sum (B,)
        """
        results = {}
        total = torch.zeros(features.shape[0], device=features.device)

        for name, scorer in self.scorers.items():
            score = scorer(features).squeeze(-1)  # (B,)
            results[name] = score
            total = total + self.weights[name] * score

        results['total'] = total
        return results

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update combination weights.

        Args:
            new_weights: Dict mapping score names to new weights
        """
        for name, weight in new_weights.items():
            if name in self.scorers:
                self.weights[name] = weight
                logger.info(f"Updated weight for '{name}': {weight}")
            else:
                logger.warning(f"Unknown scorer '{name}', skipping weight update")
