"""Compute ESM-2 embeddings for all PDB sequences and cache them.

Runs on MPS (Apple Silicon) or CUDA if available, falls back to CPU.
Produces per-residue embeddings (L, 1280) and mean-pooled (1280,) for each
unique sequence, cached via FeatureCache for reuse across experiments.

Usage:
    python scripts/data_prep/compute_esm_embeddings.py
    python scripts/data_prep/compute_esm_embeddings.py --pdb-dir data/pdb --model esm2_t33_650M_UR50D
    python scripts/data_prep/compute_esm_embeddings.py --sequences-file data/sequences.txt
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import numpy as np

from src.data.pdb_loader import load_pdb
from src.utils.feature_cache import FeatureCache, CacheMetadata, get_sequence_hash
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    """Select best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    return device


def load_esm_model(model_name: str, device: str):
    """Load ESM-2 model and alphabet."""
    import esm

    model_loaders = {
        'esm2_t33_650M_UR50D': esm.pretrained.esm2_t33_650M_UR50D,
        'esm2_t30_150M_UR50D': esm.pretrained.esm2_t30_150M_UR50D,
        'esm2_t12_35M_UR50D': esm.pretrained.esm2_t12_35M_UR50D,
        'esm2_t6_8M_UR50D': esm.pretrained.esm2_t6_8M_UR50D,
    }

    if model_name not in model_loaders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_loaders.keys())}")

    logger.info(f"Loading {model_name}...")
    model, alphabet = model_loaders[model_name]()
    model = model.eval()

    # MPS doesn't support all ops in half precision, use float32
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_name}: {param_count:,} params on {device}")

    return model, alphabet


@torch.no_grad()
def compute_embedding(
    sequence: str,
    model,
    alphabet,
    device: str,
    repr_layer: int = 33,
) -> dict:
    """Compute ESM-2 embedding for a single sequence.

    Args:
        sequence: Amino acid sequence (one-letter codes)
        model: ESM-2 model
        alphabet: ESM alphabet
        device: Device string
        repr_layer: Which layer to extract representations from

    Returns:
        Dict with 'per_residue' (L, embed_dim) and 'mean_pooled' (embed_dim,) tensors
    """
    batch_converter = alphabet.get_batch_converter()

    # ESM expects list of (label, sequence) tuples
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Forward pass
    results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    representations = results["representations"][repr_layer]

    # Remove BOS and EOS tokens: tokens are [BOS, seq..., EOS]
    # Per-residue: (1, L+2, embed_dim) -> (L, embed_dim)
    per_residue = representations[0, 1:len(sequence) + 1].cpu()

    # Mean-pooled over residues
    mean_pooled = per_residue.mean(dim=0)

    return {
        'per_residue': per_residue,      # (L, 1280)
        'mean_pooled': mean_pooled,      # (1280,)
    }


def extract_sequences_from_pdbs(pdb_dir: str) -> dict:
    """Extract unique sequences from all PDB files.

    Returns:
        Dict mapping sequence_hash -> {sequence, pdb_ids, chain_ids}
    """
    pdb_dir = Path(pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))

    if not pdb_files:
        logger.warning(f"No PDB files found in {pdb_dir}")
        return {}

    sequences = {}  # hash -> info

    for pdb_path in pdb_files:
        try:
            bb = load_pdb(str(pdb_path))
            if bb.sequence is None:
                logger.warning(f"No sequence in {pdb_path.name}")
                continue

            seq = bb.sequence
            seq_hash = get_sequence_hash(seq)

            if seq_hash not in sequences:
                sequences[seq_hash] = {
                    'sequence': seq,
                    'pdb_ids': [],
                    'chain_ids': [],
                    'length': len(seq),
                }

            sequences[seq_hash]['pdb_ids'].append(bb.pdb_id)
            sequences[seq_hash]['chain_ids'].append(bb.chain_id or 'A')

        except Exception as e:
            logger.warning(f"Failed to load {pdb_path.name}: {e}")

    logger.info(f"Found {len(sequences)} unique sequences from {len(pdb_files)} PDB files")
    return sequences


def load_sequences_from_file(filepath: str) -> dict:
    """Load sequences from a text file (one per line, optionally with labels).

    Format: sequence or label\tsequence
    """
    sequences = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                label, seq = parts[0], parts[1]
            else:
                seq = parts[0]
                label = f"seq_{len(sequences)}"

            seq_hash = get_sequence_hash(seq)
            if seq_hash not in sequences:
                sequences[seq_hash] = {
                    'sequence': seq,
                    'pdb_ids': [label],
                    'chain_ids': ['A'],
                    'length': len(seq),
                }

    logger.info(f"Loaded {len(sequences)} unique sequences from {filepath}")
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Compute ESM-2 embeddings')
    parser.add_argument('--pdb-dir', type=str, default='data/pdb',
                        help='Directory with PDB files')
    parser.add_argument('--sequences-file', type=str, default=None,
                        help='Text file with sequences (alternative to PDB dir)')
    parser.add_argument('--cache-dir', type=str, default='cache/esm_embeddings',
                        help='Output cache directory')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D',
                                 'esm2_t12_35M_UR50D', 'esm2_t6_8M_UR50D'],
                        help='ESM-2 model variant')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length (skip longer)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cached')
    args = parser.parse_args()

    # Get device
    device = get_device()

    # Collect sequences
    if args.sequences_file:
        sequences = load_sequences_from_file(args.sequences_file)
    else:
        sequences = extract_sequences_from_pdbs(args.pdb_dir)

    if not sequences:
        logger.error("No sequences to process")
        return

    # Initialize cache
    cache = FeatureCache(args.cache_dir)
    logger.info(f"Cache at {args.cache_dir}, existing entries: {len(cache)}")

    # Filter: skip already cached (unless --force)
    to_compute = {}
    skipped = 0
    for seq_hash, info in sequences.items():
        cache_key = {'sequence_hash': seq_hash, 'model': args.model}
        if not args.force and cache.has(cache_key):
            skipped += 1
        elif info['length'] > args.max_length:
            logger.info(f"Skipping {info['pdb_ids'][0]} (length {info['length']} > {args.max_length})")
        else:
            to_compute[seq_hash] = info

    logger.info(f"To compute: {len(to_compute)}, already cached: {skipped}")

    if not to_compute:
        logger.info("Nothing to compute — all sequences already cached!")
        return

    # Load model
    model, alphabet = load_esm_model(args.model, device)
    repr_layer = model.num_layers  # last layer

    # Compute embeddings
    total = len(to_compute)
    start_time = time.time()

    for i, (seq_hash, info) in enumerate(to_compute.items()):
        seq = info['sequence']
        pdb_label = info['pdb_ids'][0]

        t0 = time.time()

        try:
            embedding = compute_embedding(seq, model, alphabet, device, repr_layer)

            # Cache: save per-residue and mean-pooled as a single dict of tensors
            cache_key = {'sequence_hash': seq_hash, 'model': args.model}
            cache_data = torch.cat([
                embedding['per_residue'],                          # (L, 1280)
                embedding['mean_pooled'].unsqueeze(0),             # (1, 1280)
            ], dim=0)  # (L+1, 1280) — last row is mean-pooled

            metadata = CacheMetadata(
                method=args.model,
                params={'repr_layer': repr_layer, 'length': len(seq)},
                source=f"{pdb_label} (chain {info['chain_ids'][0]})",
            )
            cache.save(cache_key, cache_data, metadata)

            elapsed = time.time() - t0
            logger.info(
                f"[{i+1}/{total}] {pdb_label}: {len(seq)} residues, "
                f"embedding {cache_data.shape}, {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"[{i+1}/{total}] Failed {pdb_label}: {e}")
            # Try to free GPU memory on failure
            if device != 'cpu':
                torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()
            continue

    total_time = time.time() - start_time
    logger.info(f"\nDone! Computed {total} embeddings in {total_time:.1f}s")
    logger.info(f"Cache now has {len(cache)} entries at {args.cache_dir}")


if __name__ == '__main__':
    main()
