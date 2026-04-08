"""ProteinMPNN-style sequence generator.

Encoder-decoder architecture on protein graphs:
- Encoder: Structure -> node embeddings via message passing
- Decoder: Autoregressive sequence prediction conditioned on encoder output

Reference: Dauparas et al., "Robust deep learning-based protein sequence
design using ProteinMPNN", Science 2022.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.protein_structure import ProteinGraph
from src.models.sequence_generator.base import AbstractSequenceGenerator
from src.models.layers.protein_graph_conv import ProteinGraphConvStack
from src.utils.protein_constants import NUM_AA, AA_LIST, AA_3TO1


@dataclass
class MPNNConfig:
    """Configuration for ProteinMPNN model."""

    # Feature dimensions (from graph_features: node=46, edge=17)
    node_input_dim: int = 46
    edge_input_dim: int = 17

    # Architecture
    hidden_dim: int = 128
    encoder_layers: int = 3
    decoder_layers: int = 3
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 200


class ProteinMPNNModel(AbstractSequenceGenerator):
    """ProteinMPNN-style sequence design model.

    Architecture:
    - Input projection: node features -> hidden_dim
    - Encoder: 3 ProteinGraphConvStack layers (structure -> embeddings)
    - Decoder: 3 autoregressive layers conditioned on previous AA + encoder
    - Output head: hidden_dim -> 20 amino acid logits

    During training, uses teacher forcing (true sequence as decoder input).
    During sampling, uses autoregressive left-to-right generation.
    """

    def __init__(self, config: MPNNConfig):
        super().__init__(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=config.max_epochs,
        )
        self.save_hyperparameters()
        self.config = config

        # Input projections
        self.node_proj = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(config.edge_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

        # Encoder: structure -> node embeddings
        self.encoder = ProteinGraphConvStack(
            node_dim=config.hidden_dim,
            edge_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.encoder_layers,
            dropout=config.dropout,
        )

        # AA embedding for decoder (previous residue input)
        # +1 for start-of-sequence token
        self.aa_embedding = nn.Embedding(NUM_AA + 1, config.hidden_dim)

        # Decoder: autoregressive layers conditioned on encoder + previous AA
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.decoder = ProteinGraphConvStack(
            node_dim=config.hidden_dim,
            edge_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.decoder_layers,
            dropout=config.dropout,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, NUM_AA),
        )

        # Index of start-of-sequence token
        self.sos_idx = NUM_AA

    def _build_causal_edge_mask(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Build causal mask for autoregressive decoding.

        Only allow messages from positions j <= i (left-to-right ordering).

        Args:
            edge_index: (2, E) edge indices
            num_nodes: Number of nodes (unused, kept for API)

        Returns:
            causal_mask: (E,) boolean mask for valid edges
        """
        src, dst = edge_index[0], edge_index[1]
        return src <= dst  # message from src to dst, src must be <= dst

    def encode(self, graph: ProteinGraph) -> torch.Tensor:
        """Encode backbone structure into per-residue embeddings.

        Args:
            graph: ProteinGraph with node/edge features

        Returns:
            encoder_out: (L, hidden_dim) structural embeddings
        """
        h = self.node_proj(graph.node_features)
        e = self.edge_proj(graph.edge_features)
        encoder_out = self.encoder(h, graph.edge_index, e, graph.mask)
        return encoder_out

    def decode(
        self,
        encoder_out: torch.Tensor,
        graph: ProteinGraph,
        prev_aa: torch.Tensor,
    ) -> torch.Tensor:
        """Decode sequence conditioned on encoder output and previous AAs.

        Args:
            encoder_out: (L, hidden_dim) encoder embeddings
            graph: ProteinGraph (for edges)
            prev_aa: (L,) previous AA indices (shifted sequence)

        Returns:
            decoder_out: (L, hidden_dim) decoder embeddings
        """
        aa_emb = self.aa_embedding(prev_aa)  # (L, hidden_dim)
        decoder_input = self.decoder_input_proj(
            torch.cat([encoder_out, aa_emb], dim=-1)
        )

        # Use causal masking on edges for autoregressive decoding
        causal_mask = self._build_causal_edge_mask(
            graph.edge_index, graph.num_nodes
        )
        causal_edge_index = graph.edge_index[:, causal_mask]

        e = self.edge_proj(graph.edge_features)
        causal_edge_features = e[causal_mask]

        decoder_out = self.decoder(
            decoder_input, causal_edge_index, causal_edge_features, graph.mask
        )
        return decoder_out

    def forward(
        self,
        graph: ProteinGraph,
        fixed_mask: Optional[torch.Tensor] = None,
        true_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            graph: ProteinGraph
            fixed_mask: (L,) boolean mask for fixed residues
            true_sequence: (L,) AA indices for teacher forcing

        Returns:
            logits: (L, 20) per-position amino acid logits
        """
        encoder_out = self.encode(graph)

        # Build shifted sequence for decoder input (teacher forcing)
        L = graph.num_nodes
        device = graph.node_features.device

        if true_sequence is not None:
            # Shift right: [SOS, aa_0, aa_1, ..., aa_{L-2}]
            prev_aa = torch.full((L,), self.sos_idx, dtype=torch.long, device=device)
            prev_aa[1:] = true_sequence[:-1]
        else:
            # All SOS tokens (no teacher forcing)
            prev_aa = torch.full((L,), self.sos_idx, dtype=torch.long, device=device)

        decoder_out = self.decode(encoder_out, graph, prev_aa)
        logits = self.output_head(decoder_out)  # (L, 20)

        return logits

    @torch.no_grad()
    def sample(
        self,
        graph: ProteinGraph,
        fixed_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> str:
        """Autoregressively sample a sequence left-to-right.

        Args:
            graph: ProteinGraph
            fixed_mask: (L,) boolean; True = use known AA from graph features
            temperature: Sampling temperature

        Returns:
            Amino acid sequence string (one-letter codes)
        """
        self.eval()

        L = graph.num_nodes
        device = graph.node_features.device

        # Encode structure (non-autoregressive)
        encoder_out = self.encode(graph)

        # Initialize sequence with SOS tokens
        current_seq = torch.full((L,), self.sos_idx, dtype=torch.long, device=device)
        sampled_indices = torch.zeros(L, dtype=torch.long, device=device)

        # If fixed_mask provided with known sequence in graph node features,
        # extract the fixed AA indices from the one-hot encoding in node features
        # (positions 23:43 in the enhanced features are the AA one-hot)
        fixed_aa = None
        if fixed_mask is not None and graph.node_features.shape[-1] >= 43:
            aa_onehot = graph.node_features[:, 23:43]  # (L, 20)
            fixed_aa = aa_onehot.argmax(dim=-1)  # (L,)
            has_fixed = aa_onehot.sum(dim=-1) > 0.5  # (L,)
            fixed_mask = fixed_mask & has_fixed

        # Autoregressive generation
        for i in range(L):
            if fixed_mask is not None and fixed_mask[i]:
                # Use the known/fixed residue
                sampled_indices[i] = fixed_aa[i]
            else:
                # Build decoder input from current sequence
                prev_aa = torch.full((L,), self.sos_idx, dtype=torch.long, device=device)
                if i > 0:
                    prev_aa[1:i + 1] = sampled_indices[:i]

                decoder_out = self.decode(encoder_out, graph, prev_aa)
                logits_i = self.output_head(decoder_out[i])  # (20,)

                # Temperature-scaled sampling
                if temperature < 1e-6:
                    sampled_indices[i] = logits_i.argmax()
                else:
                    probs = F.softmax(logits_i / temperature, dim=-1)
                    sampled_indices[i] = torch.multinomial(probs, 1).squeeze()

            # Update current sequence for next step
            current_seq[i] = sampled_indices[i]

        # Convert indices to one-letter codes
        idx_to_aa = {i: AA_3TO1[aa] for i, aa in enumerate(AA_LIST)}
        sequence = ''.join(
            idx_to_aa.get(sampled_indices[i].item(), 'X')
            for i in range(L)
        )

        return sequence
