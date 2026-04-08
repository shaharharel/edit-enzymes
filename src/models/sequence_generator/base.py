"""Abstract base class for sequence generators.

Defines the interface and shared training/validation logic for
ProteinMPNN-style sequence design models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.data.protein_structure import ProteinGraph
from src.utils.protein_constants import NUM_AA


class AbstractSequenceGenerator(pl.LightningModule, ABC):
    """Base class for protein sequence generators.

    All sequence generators must implement:
    - forward: Predict per-position AA logits from graph features
    - sample: Autoregressively generate a sequence string

    Shared functionality:
    - training_step: Cross-entropy loss on non-fixed positions
    - validation_step: Sequence recovery rate
    - configure_optimizers: AdamW + CosineAnnealingLR
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    @abstractmethod
    def forward(
        self,
        graph: ProteinGraph,
        fixed_mask: Optional[torch.Tensor] = None,
        true_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict per-position amino acid logits.

        Args:
            graph: ProteinGraph with node/edge features
            fixed_mask: (L,) boolean mask; True = fixed/known residues
            true_sequence: (L,) AA indices for teacher forcing in decoder

        Returns:
            logits: (L, 20) per-position amino acid logits
        """
        ...

    @abstractmethod
    def sample(
        self,
        graph: ProteinGraph,
        fixed_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> str:
        """Autoregressively sample a sequence.

        Args:
            graph: ProteinGraph with node/edge features
            fixed_mask: (L,) boolean mask; True = fixed residues
            temperature: Sampling temperature (lower = more greedy)

        Returns:
            Amino acid sequence string (one-letter codes)
        """
        ...

    def _unbatch(self, batch):
        """Squeeze batch dimension when batch_size=1."""
        return {k: v.squeeze(0) for k, v in batch.items()}

    def training_step(self, batch, batch_idx):
        """Cross-entropy loss on non-fixed positions."""
        batch = self._unbatch(batch)
        graph = ProteinGraph(
            node_features=batch['node_features'],
            edge_index=batch['edge_index'],
            edge_features=batch['edge_features'],
            coords=batch['coords'],
            mask=batch['mask'],
        )

        target = batch['sequence']      # (L,) int64 AA indices
        mask = batch['mask']            # (L,) bool, valid residues
        fixed_mask = batch['fixed_mask']  # (L,) bool, fixed residues

        logits = self.forward(graph, fixed_mask, true_sequence=target)  # (L, 20)

        # Only compute loss on non-fixed, valid positions
        design_mask = mask & ~fixed_mask  # positions we need to design

        if design_mask.any():
            loss = F.cross_entropy(
                logits[design_mask],
                target[design_mask],
            )
        else:
            # Fallback: loss on all valid positions
            loss = F.cross_entropy(logits[mask], target[mask])

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute sequence recovery rate."""
        batch = self._unbatch(batch)
        graph = ProteinGraph(
            node_features=batch['node_features'],
            edge_index=batch['edge_index'],
            edge_features=batch['edge_features'],
            coords=batch['coords'],
            mask=batch['mask'],
        )

        target = batch['sequence']
        mask = batch['mask']
        fixed_mask = batch['fixed_mask']

        logits = self.forward(graph, fixed_mask, true_sequence=target)

        # Loss
        design_mask = mask & ~fixed_mask
        if design_mask.any():
            loss = F.cross_entropy(logits[design_mask], target[design_mask])
        else:
            loss = F.cross_entropy(logits[mask], target[mask])

        # Sequence recovery rate (on designable positions)
        preds = logits.argmax(dim=-1)
        eval_mask = design_mask if design_mask.any() else mask
        correct = (preds[eval_mask] == target[eval_mask]).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recovery', correct, prog_bar=True)

        return {'val_loss': loss, 'val_recovery': correct}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
