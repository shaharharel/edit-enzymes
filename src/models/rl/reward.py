"""Reward function for RL-based enzyme design.

Combines multi-objective scoring with catalytic constraint satisfaction
to produce separate reward signals for backbone and sequence policies.

Credit assignment:
- Backbone policy: geometry feasibility + constraint satisfaction
- Sequence policy: stability + packing + activity scores
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.catalytic_constraints import CatalyticConstraint
from src.data.protein_structure import ProteinBackbone
from src.models.scoring.multi_objective import MultiObjectiveScorer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RewardFunction(nn.Module):
    """Computes per-objective and total rewards for generated designs.

    Separates reward into backbone-level (geometric) and sequence-level
    (biophysical) components for credit assignment in the RL loop.
    """

    def __init__(
        self,
        scorer: MultiObjectiveScorer,
        constraint: CatalyticConstraint,
        backbone_reward_weight: float = 1.0,
        sequence_reward_weight: float = 1.0,
        constraint_reward_scale: float = 5.0,
        geometry_reward_scale: float = 2.0,
    ):
        """
        Args:
            scorer: Multi-objective scorer combining stability, packing, etc.
            constraint: Catalytic constraint specification.
            backbone_reward_weight: Overall weight for backbone reward.
            sequence_reward_weight: Overall weight for sequence reward.
            constraint_reward_scale: Scale factor for constraint satisfaction reward.
            geometry_reward_scale: Scale factor for geometry feasibility reward.
        """
        super().__init__()
        self.scorer = scorer
        self.constraint = constraint
        self.backbone_reward_weight = backbone_reward_weight
        self.sequence_reward_weight = sequence_reward_weight
        self.constraint_reward_scale = constraint_reward_scale
        self.geometry_reward_scale = geometry_reward_scale

    def compute(
        self,
        backbone: ProteinBackbone,
        sequence: str,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all reward components for a design.

        Args:
            backbone: Generated protein backbone.
            sequence: Designed amino acid sequence (one-letter codes).
            features: (1, d) pre-computed feature vector for scoring models.

        Returns:
            Dict with keys:
                - per-objective scores (from scorer)
                - 'constraint_satisfaction': geometric constraint score
                - 'geometry_feasibility': bond/angle quality score
                - 'backbone_reward': combined backbone-level reward
                - 'sequence_reward': combined sequence-level reward
                - 'total': overall reward
        """
        device = features.device

        # Score with multi-objective scorer
        if features.dim() == 1:
            features = features.unsqueeze(0)
        with torch.no_grad():
            scores = self.scorer(features)

        results = {}
        for name, value in scores.items():
            results[name] = value.squeeze()

        # Backbone-level rewards
        constraint_score = self.constraint_satisfaction(backbone)
        geometry_score = self.geometry_feasibility(backbone)

        results['constraint_satisfaction'] = torch.tensor(
            constraint_score, device=device, dtype=torch.float32,
        )
        results['geometry_feasibility'] = torch.tensor(
            geometry_score, device=device, dtype=torch.float32,
        )

        # Backbone reward: geometry + constraint satisfaction
        backbone_reward = (
            self.geometry_reward_scale * geometry_score
            + self.constraint_reward_scale * constraint_score
        )
        results['backbone_reward'] = torch.tensor(
            self.backbone_reward_weight * backbone_reward,
            device=device, dtype=torch.float32,
        )

        # Sequence reward: scorer total (stability + packing + activity, etc.)
        # The scorer 'total' already combines the individual scores with weights
        sequence_reward = scores['total'].squeeze().item()
        results['sequence_reward'] = torch.tensor(
            self.sequence_reward_weight * sequence_reward,
            device=device, dtype=torch.float32,
        )

        # Overall total
        results['total'] = results['backbone_reward'] + results['sequence_reward']

        return results

    def constraint_satisfaction(self, backbone: ProteinBackbone) -> float:
        """Score how well the backbone satisfies catalytic constraints.

        Computes the negative RMSD between generated and target catalytic
        residue CA positions. Higher (less negative) is better.

        Args:
            backbone: Generated protein backbone.

        Returns:
            Constraint satisfaction score (negative RMSD, so higher = better).
        """
        if not self.constraint.residues:
            return 0.0

        target_positions = []
        generated_positions = []

        for res in self.constraint.residues:
            if 'CA' not in res.atom_positions:
                continue
            target_ca = res.atom_positions['CA']
            target_positions.append(target_ca)

            # Use position_index if available, otherwise use the residue
            # order in the constraint list
            idx = res.position_index
            if idx is not None and idx < backbone.length:
                generated_positions.append(backbone.ca_coords[idx])
            else:
                # Fallback: cannot match this residue
                generated_positions.append(target_ca)

        if not target_positions:
            return 0.0

        target = np.stack(target_positions)
        generated = np.stack(generated_positions)

        rmsd = np.sqrt(np.mean(np.sum((target - generated) ** 2, axis=-1)))

        # Convert to reward: negative RMSD (higher = better)
        # Use exp(-rmsd) to bound reward in [0, 1]
        return float(np.exp(-rmsd))

    def geometry_feasibility(self, backbone: ProteinBackbone) -> float:
        """Score the geometric quality of a generated backbone.

        Checks bond lengths and angles against ideal values.
        Returns a score in [0, 1] where 1 = ideal geometry.

        Args:
            backbone: Generated protein backbone.

        Returns:
            Geometry feasibility score in [0, 1].
        """
        from src.utils.protein_constants import BOND_LENGTHS, CA_CA_DISTANCE

        coords = backbone.coords  # (L, 4, 3)
        L = backbone.length

        if L < 2:
            return 1.0

        violations = 0.0
        n_checks = 0

        # Check N-CA bond lengths
        n_ca_dist = np.linalg.norm(coords[:, 1] - coords[:, 0], axis=-1)
        ideal_n_ca = BOND_LENGTHS[('N', 'CA')]
        violations += np.sum((n_ca_dist - ideal_n_ca) ** 2)
        n_checks += L

        # Check CA-C bond lengths
        ca_c_dist = np.linalg.norm(coords[:, 2] - coords[:, 1], axis=-1)
        ideal_ca_c = BOND_LENGTHS[('CA', 'C')]
        violations += np.sum((ca_c_dist - ideal_ca_c) ** 2)
        n_checks += L

        # Check consecutive CA-CA distances
        ca_coords = backbone.ca_coords
        ca_ca_dist = np.linalg.norm(ca_coords[1:] - ca_coords[:-1], axis=-1)
        violations += np.sum((ca_ca_dist - CA_CA_DISTANCE) ** 2)
        n_checks += L - 1

        # Convert to score: exp(-mean_squared_violation)
        mean_violation = violations / max(n_checks, 1)
        return float(np.exp(-mean_violation))
