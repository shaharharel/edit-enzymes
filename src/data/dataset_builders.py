"""Training datasets for each system component.

Each dataset handles loading, preprocessing, and batching for its
respective training phase.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.protein_structure import ProteinBackbone
from src.data.catalytic_constraints import ActiveSiteSpec
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BackboneDiffusionDataset(Dataset):
    """Dataset for backbone diffusion training.

    Returns (noised_coords, clean_coords, constraint_info, timestep)
    for training the denoising network.

    In template-conditioned mode, noise is added to a template backbone.
    The model learns to recover the clean structure.
    """

    def __init__(
        self,
        backbones: List[ProteinBackbone],
        specs: Optional[List[ActiveSiteSpec]] = None,
        max_length: int = 256,
        noise_scale: float = 1.0,
    ):
        """
        Args:
            backbones: List of ProteinBackbone objects
            specs: Optional ActiveSiteSpec per backbone (for constraint conditioning)
            max_length: Maximum sequence length (pad/truncate)
            noise_scale: Overall noise scaling factor
        """
        self.backbones = backbones
        self.specs = specs
        self.max_length = max_length
        self.noise_scale = noise_scale

    def __len__(self) -> int:
        return len(self.backbones)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bb = self.backbones[idx]
        L = min(bb.length, self.max_length)

        # Coordinates (pad to max_length)
        coords = np.zeros((self.max_length, 4, 3), dtype=np.float32)
        coords[:L] = bb.coords[:L]
        coords_tensor = torch.tensor(coords)

        # Residue mask
        mask = np.zeros(self.max_length, dtype=bool)
        mask[:L] = True

        result = {
            'coords': coords_tensor,
            'mask': torch.tensor(mask),
            'length': torch.tensor(L),
        }

        # Constraint info
        if self.specs is not None and idx < len(self.specs):
            spec = self.specs[idx]
            constraint_mask = np.zeros(self.max_length, dtype=bool)
            for i in spec.fixed_residue_indices:
                if i < self.max_length:
                    constraint_mask[i] = True
            result['constraint_mask'] = torch.tensor(constraint_mask)

            # Template coords
            if spec.has_template:
                template = np.zeros((self.max_length, 4, 3), dtype=np.float32)
                t_len = min(spec.template_backbone.shape[0], self.max_length)
                template[:t_len] = spec.template_backbone[:t_len]
                result['template_coords'] = torch.tensor(template)

            # Catalytic residue target positions
            if spec.constraint.residues:
                positions = []
                for res in spec.constraint.residues:
                    if 'CA' in res.atom_positions:
                        positions.append(res.atom_positions['CA'])
                if positions:
                    result['constraint_positions'] = torch.tensor(
                        np.array(positions, dtype=np.float32)
                    )

        return result


class SequenceDesignDataset(Dataset):
    """Dataset for sequence generator training.

    Returns (backbone_features, true_sequence, fixed_mask) pairs
    for training the ProteinMPNN-style model.
    """

    def __init__(
        self,
        backbones: List[ProteinBackbone],
        fixed_masks: Optional[List[np.ndarray]] = None,
        max_length: int = 256,
    ):
        self.backbones = [
            bb for bb in backbones if bb.sequence is not None
        ]
        self.fixed_masks = fixed_masks
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.backbones)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from src.utils.protein_constants import AA_1_INDEX, NUM_AA

        bb = self.backbones[idx]
        L = min(bb.length, self.max_length)

        # Build graph
        graph = bb.to_graph(k=min(30, L - 1))

        # Encode sequence as indices
        seq_indices = np.zeros(self.max_length, dtype=np.int64)
        for i, aa in enumerate(bb.sequence[:L]):
            seq_indices[i] = AA_1_INDEX.get(aa, 0)

        # Mask
        mask = np.zeros(self.max_length, dtype=bool)
        mask[:L] = True

        # Fixed residue mask
        fixed = np.zeros(self.max_length, dtype=bool)
        if self.fixed_masks is not None and idx < len(self.fixed_masks):
            fm = self.fixed_masks[idx]
            fixed[:min(len(fm), self.max_length)] = fm[:self.max_length]

        return {
            'node_features': graph.node_features,
            'edge_index': graph.edge_index,
            'edge_features': graph.edge_features,
            'coords': graph.coords,
            'sequence': torch.tensor(seq_indices),
            'mask': torch.tensor(mask),
            'fixed_mask': torch.tensor(fixed),
        }


class ScoringDataset(Dataset):
    """Dataset for scoring model training.

    Returns (features, score_targets) for training surrogate models.
    Features are pre-computed structure+sequence representations.
    Targets are Rosetta energy terms or experimental measurements.
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ):
        """
        Args:
            features: (N, d) pre-computed feature vectors
            targets: Dict mapping score names to (N,) target tensors
        """
        self.features = features
        self.targets = targets
        self._n = features.shape[0]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {'features': self.features[idx]}
        for name, values in self.targets.items():
            result[name] = values[idx]
        return result


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
