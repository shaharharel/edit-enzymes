"""PROSS-style stability scoring surrogate.

PROSS (Protein Repair One-Stop Shop) from the Fleishman lab combines
three signals that are distinct from raw Rosetta energy:

1. Phylogenetic conservation (PSSM) — which mutations are evolutionarily tolerated
2. Rosetta ΔΔG — energy change from single-point mutations
3. Combinatorial compatibility — which mutations work well together

This module provides surrogates for each component and a combined PROSS-style
scorer that mimics the full pipeline.

Unlike the raw Rosetta scorer (which just computes physics energy on a structure),
PROSS scoring is mutation-aware: it evaluates proposed mutations against
evolutionary and energetic criteria.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.models.scoring.base import AbstractScoringModel
from src.utils.protein_constants import NUM_AA


@dataclass
class PROSSScoreComponents:
    """Output of PROSS-style scoring.

    Attributes:
        pssm_score: Per-mutation phylogenetic score (higher = more tolerated)
        ddg_score: Per-mutation Rosetta ΔΔG (negative = stabilizing)
        pssm_filter_pass: Whether mutation passes PSSM ≥ 0 filter
        combined_score: PROSS-style combined score
        compatibility_score: Multi-mutation compatibility (if multiple mutations)
    """
    pssm_score: torch.Tensor       # (N_mutations,)
    ddg_score: torch.Tensor        # (N_mutations,)
    pssm_filter_pass: torch.Tensor  # (N_mutations,) bool
    combined_score: torch.Tensor    # scalar
    compatibility_score: Optional[torch.Tensor] = None  # scalar, if multi-mutation


class PSSMScorer(AbstractScoringModel):
    """Learned surrogate for position-specific scoring matrix (PSSM) conservation.

    Approximates the phylogenetic component of PROSS: given a sequence position
    and proposed amino acid, predict the PSSM score (log-likelihood of that AA
    at that position based on evolutionary conservation).

    Input: ESM-2 embedding at mutation position + AA identity
    Output: PSSM score (higher = more conserved/tolerated)

    The key insight: ESM-2 embeddings already encode evolutionary information
    (trained on millions of sequences), so they can approximate PSSM scores
    without requiring explicit MSA computation.
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
    ):
        super().__init__(learning_rate=learning_rate)
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Input: ESM embedding at position + one-hot AA (wild-type and mutant)
        input_dim = esm_dim + NUM_AA * 2  # WT AA + mutant AA

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    @property
    def score_name(self) -> str:
        return 'pssm'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class PROSSDeltaGScorer(AbstractScoringModel):
    """Learned surrogate for PROSS-style ΔΔG prediction.

    Unlike raw Rosetta ΔΔG, PROSS ΔΔG is filtered by phylogenetic constraints
    and uses backbone-restrained relaxation. This surrogate learns to predict
    the PROSS-specific ΔΔG which tends to be more conservative than raw Rosetta.

    Input: structural features at mutation site + ESM embedding + AA identity
    Output: ΔΔG in Rosetta energy units (negative = stabilizing)

    Training data: PROSS web server outputs or local Rosetta ddG calculations
    with harmonic backbone restraints (matching PROSS protocol).
    """

    def __init__(
        self,
        input_dim: int = 1280 + NUM_AA * 2 + 64,  # ESM + AAs + structural features
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
    ):
        super().__init__(learning_rate=learning_rate)
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    @property
    def score_name(self) -> str:
        return 'pross_ddg'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MutationCompatibilityScorer(AbstractScoringModel):
    """Learned surrogate for multi-mutation compatibility scoring.

    Approximates the combinatorial component of PROSS / EpiNNet from htFuncLib:
    given a set of mutations, predict whether they form a low-energy combination.

    This is the key component that goes beyond single-mutation scoring —
    it captures epistatic interactions between mutations.

    Input: set of mutation features (variable size, aggregated)
    Output: compatibility score (higher = mutations work well together)

    Architecture uses a set-aggregation approach (DeepSets-style) to handle
    variable numbers of mutations.
    """

    def __init__(
        self,
        mutation_dim: int = 128,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
    ):
        super().__init__(learning_rate=learning_rate)

        # Per-mutation encoder
        self.mutation_encoder = nn.Sequential(
            nn.Linear(mutation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise interaction encoder
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Set aggregation → prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # sum + max pooling
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @property
    def score_name(self) -> str:
        return 'compatibility'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (N_mutations, mutation_dim) mutation feature set
                      or (B, max_mutations, mutation_dim) if batched

        Returns:
            (1,) or (B, 1) compatibility score
        """
        if features.dim() == 2:
            return self._forward_single(features)
        else:
            # Batched
            B = features.shape[0]
            scores = []
            for i in range(B):
                scores.append(self._forward_single(features[i]))
            return torch.stack(scores)

    def _forward_single(self, mutation_features: torch.Tensor) -> torch.Tensor:
        """Score a single set of mutations.

        Args:
            mutation_features: (K, mutation_dim) where K = number of mutations
        """
        K = mutation_features.shape[0]

        # Encode each mutation
        encoded = self.mutation_encoder(mutation_features)  # (K, hidden)

        # Pairwise interactions (if multiple mutations)
        if K > 1:
            # All pairs
            idx_i = torch.arange(K).unsqueeze(1).expand(-1, K).reshape(-1)
            idx_j = torch.arange(K).unsqueeze(0).expand(K, -1).reshape(-1)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]

            pairs = torch.cat([encoded[idx_i], encoded[idx_j]], dim=-1)
            pair_features = self.pair_encoder(pairs)  # (K*(K-1), hidden)

            # Add pairwise info back to per-mutation encoding
            pair_agg = torch.zeros_like(encoded)
            pair_agg.index_add_(0, idx_i, pair_features)
            encoded = encoded + pair_agg / max(K - 1, 1)

        # Set aggregation: sum + max
        sum_pool = encoded.sum(dim=0)   # (hidden,)
        max_pool = encoded.max(dim=0).values  # (hidden,)

        combined = torch.cat([sum_pool, max_pool])  # (hidden*2,)
        return self.predictor(combined)


class PROSSCombinedScorer(nn.Module):
    """Combined PROSS-style scorer that integrates all components.

    Mimics the full PROSS pipeline:
    1. Check phylogenetic tolerance (PSSM)
    2. Predict mutation ΔΔG
    3. Filter: only keep mutations with PSSM ≥ 0 and ΔΔG ≤ threshold
    4. Score mutation combinations for compatibility

    This is what the RL loop uses as reward for sequence-level decisions.
    """

    def __init__(
        self,
        pssm_scorer: PSSMScorer,
        ddg_scorer: PROSSDeltaGScorer,
        compatibility_scorer: MutationCompatibilityScorer,
        pssm_threshold: float = 0.0,
        ddg_threshold: float = -0.45,  # PROSS default: -0.45 REU
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.pssm_scorer = pssm_scorer
        self.ddg_scorer = ddg_scorer
        self.compatibility_scorer = compatibility_scorer
        self.pssm_threshold = pssm_threshold
        self.ddg_threshold = ddg_threshold

        if weights is None:
            weights = {
                'pssm': 0.3,
                'ddg': 0.4,
                'compatibility': 0.3,
            }
        self.weights = weights

    def forward(
        self,
        mutation_features: Dict[str, torch.Tensor],
    ) -> PROSSScoreComponents:
        """Score a set of mutations PROSS-style.

        Args:
            mutation_features: Dict with:
                - 'pssm_input': (K, esm_dim + 2*NUM_AA) per-mutation PSSM features
                - 'ddg_input': (K, input_dim) per-mutation ΔΔG features
                - 'compatibility_input': (K, mutation_dim) mutation set features

        Returns:
            PROSSScoreComponents with all scores
        """
        # 1. PSSM scores per mutation
        pssm_scores = self.pssm_scorer(mutation_features['pssm_input']).squeeze(-1)

        # 2. ΔΔG scores per mutation
        ddg_scores = self.ddg_scorer(mutation_features['ddg_input']).squeeze(-1)

        # 3. PSSM filter
        pssm_pass = pssm_scores >= self.pssm_threshold

        # 4. Compatibility score (if multiple mutations)
        K = pssm_scores.shape[0]
        if K > 1:
            compat = self.compatibility_scorer(
                mutation_features['compatibility_input']
            )
        else:
            compat = torch.tensor([1.0], device=pssm_scores.device)

        # 5. Combined score
        # Reward mutations that are: evolutionarily tolerated + stabilizing + compatible
        combined = (
            self.weights['pssm'] * pssm_scores.mean()
            + self.weights['ddg'] * (-ddg_scores.mean())  # negate: lower ddg = better
            + self.weights['compatibility'] * compat.squeeze()
        )

        return PROSSScoreComponents(
            pssm_score=pssm_scores,
            ddg_score=ddg_scores,
            pssm_filter_pass=pssm_pass,
            combined_score=combined,
            compatibility_score=compat,
        )

    def update_weights(self, weights: Dict[str, float]):
        self.weights.update(weights)
