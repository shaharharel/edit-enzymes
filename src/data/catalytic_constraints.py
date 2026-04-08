"""Catalytic constraint representations for enzyme design.

Defines the core data structures that specify what the backbone generator
must satisfy: which residues form the active site, their geometry, and
optional fold family constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


@dataclass
class CatalyticResidue:
    """A single catalytic residue with its functional atoms and role.

    Attributes:
        residue_type: Three-letter amino acid code (e.g., 'HIS', 'SER')
        atom_positions: Key atom coordinates {atom_name: (3,) array}
        role: Functional role (e.g., 'nucleophile', 'general_base')
        position_index: Optional residue index in template (if known)
    """
    residue_type: str
    atom_positions: Dict[str, np.ndarray]
    role: str
    position_index: Optional[int] = None

    def get_ca_position(self) -> np.ndarray:
        """Return CA position if available."""
        if 'CA' in self.atom_positions:
            return self.atom_positions['CA']
        raise ValueError(f"CA not found in atom positions for {self.residue_type}")

    def to_tensor_dict(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert atom positions to tensors."""
        return {
            name: torch.tensor(pos, dtype=torch.float32, device=device)
            for name, pos in self.atom_positions.items()
        }


@dataclass
class CatalyticConstraint:
    """Complete catalytic constraint specification.

    Defines the geometry that the backbone generator must satisfy.

    Attributes:
        residues: List of catalytic residues with positions and roles
        pairwise_distances: Required distances between residue pairs
            {(idx_a, idx_b): distance_angstroms}
        ligand_pose: Optional ligand heavy atom coordinates (N_atoms, 3)
        fold_family: Optional fold family identifier (e.g., 'TIM_barrel')
    """
    residues: List[CatalyticResidue]
    pairwise_distances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    ligand_pose: Optional[np.ndarray] = None
    fold_family: Optional[str] = None

    @property
    def num_residues(self) -> int:
        return len(self.residues)

    def get_constraint_positions(self) -> np.ndarray:
        """Get CA positions of all catalytic residues as (N, 3) array."""
        positions = []
        for res in self.residues:
            positions.append(res.get_ca_position())
        return np.stack(positions)

    def compute_distance_violations(
        self, positions: np.ndarray, tolerance: float = 0.5
    ) -> Dict[Tuple[int, int], float]:
        """Check which pairwise distance constraints are violated.

        Args:
            positions: (N, 3) current positions of catalytic residues
            tolerance: Allowed deviation in Angstroms

        Returns:
            Dictionary of violated constraints with their deviations.
        """
        violations = {}
        for (i, j), target_dist in self.pairwise_distances.items():
            actual_dist = np.linalg.norm(positions[i] - positions[j])
            deviation = abs(actual_dist - target_dist)
            if deviation > tolerance:
                violations[(i, j)] = deviation
        return violations

    def constraint_loss(
        self, coords: torch.Tensor, residue_indices: List[int]
    ) -> torch.Tensor:
        """Compute differentiable constraint satisfaction loss.

        Args:
            coords: (L, 4, 3) backbone coordinates
            residue_indices: Which residue positions in coords correspond
                to the catalytic residues

        Returns:
            Scalar loss for constraint violation.
        """
        loss = torch.tensor(0.0, device=coords.device)

        # Pairwise distance constraints
        for (i, j), target_dist in self.pairwise_distances.items():
            pos_i = coords[residue_indices[i], 1]  # CA of residue i
            pos_j = coords[residue_indices[j], 1]  # CA of residue j
            actual_dist = torch.norm(pos_i - pos_j)
            loss = loss + (actual_dist - target_dist) ** 2

        # Position RMSD for catalytic residues with known positions
        for k, res in enumerate(self.residues):
            if 'CA' in res.atom_positions:
                target = torch.tensor(
                    res.atom_positions['CA'],
                    dtype=torch.float32,
                    device=coords.device,
                )
                actual = coords[residue_indices[k], 1]
                loss = loss + torch.sum((actual - target) ** 2)

        return loss


@dataclass
class ActiveSiteSpec:
    """Complete active site specification for enzyme design.

    Combines the catalytic constraint with template and masking information
    needed by both the backbone and sequence generators.

    Attributes:
        constraint: The catalytic geometry to satisfy
        template_backbone: Optional template backbone coords (L, 4, 3)
        fixed_residue_indices: Residue indices fixed during sequence design
        template_pdb_id: Source PDB identifier
        noise_level: How much deviation from template (0=exact, 1=full noise)
    """
    constraint: CatalyticConstraint
    template_backbone: Optional[np.ndarray] = None
    fixed_residue_indices: List[int] = field(default_factory=list)
    template_pdb_id: Optional[str] = None
    noise_level: float = 0.1  # default: small deviations from template

    @property
    def has_template(self) -> bool:
        return self.template_backbone is not None

    @property
    def num_residues(self) -> int:
        if self.template_backbone is not None:
            return self.template_backbone.shape[0]
        return 0

    def get_fixed_mask(self, length: int) -> np.ndarray:
        """Get boolean mask of fixed residues for sequence design.

        Args:
            length: Total sequence length

        Returns:
            (L,) boolean array, True for fixed residues.
        """
        mask = np.zeros(length, dtype=bool)
        for idx in self.fixed_residue_indices:
            if idx < length:
                mask[idx] = True
        return mask


def load_constraint_from_yaml(path: str) -> CatalyticConstraint:
    """Load a catalytic constraint from a YAML file.

    Expected format:
        fold_family: TIM_barrel  # optional
        residues:
          - type: HIS
            role: general_base
            position_index: 95
            atoms:
              CA: [x, y, z]
              NE2: [x, y, z]
          - type: ASP
            role: nucleophile
            ...
        pairwise_distances:
          - pair: [0, 1]
            distance: 5.2
        ligand_atoms:  # optional
          - [x, y, z]
          - ...
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    residues = []
    for res_data in data['residues']:
        atoms = {
            name: np.array(pos, dtype=np.float32)
            for name, pos in res_data['atoms'].items()
        }
        residues.append(CatalyticResidue(
            residue_type=res_data['type'],
            atom_positions=atoms,
            role=res_data['role'],
            position_index=res_data.get('position_index'),
        ))

    pairwise = {}
    for pd in data.get('pairwise_distances', []):
        pair = tuple(pd['pair'])
        pairwise[pair] = pd['distance']

    ligand = None
    if 'ligand_atoms' in data:
        ligand = np.array(data['ligand_atoms'], dtype=np.float32)

    return CatalyticConstraint(
        residues=residues,
        pairwise_distances=pairwise,
        ligand_pose=ligand,
        fold_family=data.get('fold_family'),
    )
