"""Generate PROSS-style training labels using ESM-2 pseudo-likelihoods as PSSM proxy.

PROSS uses MSA-derived PSSMs to score mutations. Computing real MSAs (via HHblits)
is slow and requires large databases. Instead, we use ESM-2 masked marginal
probabilities as a PSSM proxy — ESM-2 was trained on millions of protein sequences
and its masked predictions correlate well with evolutionary conservation.

This script generates three types of labels:
1. ESM-2 pseudo-PSSM: P(aa | context) at each position via masked prediction
2. Mutation ΔΔG proxy: change in ESM-2 log-likelihood (ESM-IF style)
3. Position conservation: entropy of the ESM-2 probability distribution

These labels train the PSSMScorer and PROSSDeltaGScorer surrogates.

Usage:
    python scripts/data_prep/compute_pross_labels.py
    python scripts/data_prep/compute_pross_labels.py --pdb-dir data/pdb
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np

from src.data.pdb_loader import load_pdb
from src.utils.feature_cache import FeatureCache, CacheMetadata, get_sequence_hash
from src.utils.protein_constants import AA_1_INDEX, AA_LIST, AA_3TO1, NUM_AA
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def compute_esm_pseudo_pssm(
    sequence: str,
    model,
    alphabet,
    device: str,
) -> dict:
    """Compute ESM-2 pseudo-PSSM via masked marginal probabilities.

    For each position, mask it and predict the probability distribution
    over all 20 amino acids. This approximates a PSSM derived from MSA.

    Args:
        sequence: Amino acid sequence
        model: ESM-2 model
        alphabet: ESM alphabet
        device: Device

    Returns:
        Dict with:
            'pseudo_pssm': (L, 20) log-probabilities per position per AA
            'wt_log_prob': (L,) log-probability of wild-type AA at each position
            'entropy': (L,) Shannon entropy at each position
            'conservation': (L,) 1 - normalized entropy (higher = more conserved)
    """
    batch_converter = alphabet.get_batch_converter()
    L = len(sequence)

    # Tokenize the full sequence
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    mask_idx = alphabet.mask_idx

    # AA token mapping: find ESM token IDs for standard amino acids
    aa_token_ids = []
    for aa_1letter in [AA_3TO1[aa3] for aa3 in AA_LIST]:
        tok_id = alphabet.get_idx(aa_1letter)
        aa_token_ids.append(tok_id)
    aa_token_ids = torch.tensor(aa_token_ids, device=device)  # (20,)

    pseudo_pssm = torch.zeros(L, NUM_AA)
    wt_log_prob = torch.zeros(L)

    with torch.no_grad():
        # Batch masking: mask one position at a time
        # For efficiency, process in small batches
        batch_size = min(32, L)

        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            n_pos = end - start

            # Create masked copies
            masked_tokens = tokens.expand(n_pos, -1).clone()
            for i in range(n_pos):
                pos = start + i
                masked_tokens[i, pos + 1] = mask_idx  # +1 for BOS token

            # Forward pass
            results = model(masked_tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"]  # (n_pos, seq_len+2, vocab_size)

            for i in range(n_pos):
                pos = start + i
                # Get logits at masked position
                pos_logits = logits[i, pos + 1]  # (vocab_size,)

                # Extract probabilities for the 20 standard AAs
                aa_logits = pos_logits[aa_token_ids]
                aa_log_probs = F.log_softmax(aa_logits, dim=0)

                pseudo_pssm[pos] = aa_log_probs.cpu()

                # Wild-type log probability
                wt_aa = sequence[pos]
                wt_idx = AA_1_INDEX.get(wt_aa, 0)
                wt_log_prob[pos] = aa_log_probs[wt_idx].cpu()

    # Compute entropy and conservation
    probs = torch.exp(pseudo_pssm)  # (L, 20)
    entropy = -(probs * pseudo_pssm).sum(dim=-1)  # (L,)
    max_entropy = np.log(NUM_AA)
    conservation = 1.0 - (entropy / max_entropy)

    return {
        'pseudo_pssm': pseudo_pssm,       # (L, 20) log-probs
        'wt_log_prob': wt_log_prob,        # (L,)
        'entropy': entropy,                # (L,)
        'conservation': conservation,      # (L,) 0=variable, 1=conserved
    }


def compute_mutation_scores(
    pseudo_pssm: torch.Tensor,
    sequence: str,
) -> dict:
    """Compute per-mutation scores from the pseudo-PSSM.

    For each position, compute the score for every possible substitution:
    - PSSM score = log P(mutant_AA) - log P(wt_AA)
    - Positive = mutation is tolerated/favored
    - Negative = mutation is disfavored

    Args:
        pseudo_pssm: (L, 20) log-probabilities
        sequence: Wild-type sequence

    Returns:
        Dict with:
            'mutation_scores': (L, 20) score for each mutation at each position
            'best_mutations': (L,) index of best non-WT mutation at each position
            'n_tolerated': (L,) number of tolerated mutations (score ≥ 0) per position
    """
    L = len(sequence)

    wt_indices = torch.tensor([AA_1_INDEX.get(aa, 0) for aa in sequence])
    wt_scores = pseudo_pssm[torch.arange(L), wt_indices]  # (L,)

    # Mutation score = log P(mut) - log P(wt) (like PSSM score)
    mutation_scores = pseudo_pssm - wt_scores.unsqueeze(-1)  # (L, 20)

    # Mask out WT (score = 0 by definition, not a mutation)
    for i in range(L):
        mutation_scores[i, wt_indices[i]] = float('-inf')

    # Best non-WT mutation
    best_mutations = mutation_scores.argmax(dim=-1)

    # Number of tolerated mutations (PSSM ≥ 0, like PROSS filter)
    n_tolerated = (mutation_scores >= 0).sum(dim=-1)
    # Don't count the masked WT
    n_tolerated = n_tolerated.float()

    return {
        'mutation_scores': mutation_scores,  # (L, 20)
        'best_mutations': best_mutations,     # (L,)
        'n_tolerated': n_tolerated,           # (L,)
    }


def main():
    parser = argparse.ArgumentParser(description='Compute PROSS-style labels')
    parser.add_argument('--pdb-dir', type=str, default='data/pdb')
    parser.add_argument('--cache-dir', type=str, default='cache/pross_labels')
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Load ESM model
    import esm
    logger.info(f"Loading {args.model}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    logger.info("Model loaded")

    # Collect sequences from PDBs
    pdb_dir = Path(args.pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    cache = FeatureCache(args.cache_dir)

    logger.info(f"Processing {len(pdb_files)} PDB files")
    computed = 0
    skipped = 0

    for pdb_path in pdb_files:
        try:
            bb = load_pdb(str(pdb_path))
            if bb.sequence is None:
                continue

            pdb_id = pdb_path.stem.upper()
            seq_hash = get_sequence_hash(bb.sequence)
            cache_key = {'sequence_hash': seq_hash, 'method': 'pross_labels', 'model': args.model}

            if not args.force and cache.has(cache_key):
                skipped += 1
                continue

            t0 = time.time()

            # Compute pseudo-PSSM
            pssm_data = compute_esm_pseudo_pssm(bb.sequence, model, alphabet, device)

            # Compute mutation scores
            mut_data = compute_mutation_scores(pssm_data['pseudo_pssm'], bb.sequence)

            # Combine and cache
            cache_data = {
                'pseudo_pssm': pssm_data['pseudo_pssm'],          # (L, 20)
                'wt_log_prob': pssm_data['wt_log_prob'],           # (L,)
                'entropy': pssm_data['entropy'],                   # (L,)
                'conservation': pssm_data['conservation'],         # (L,)
                'mutation_scores': mut_data['mutation_scores'],    # (L, 20)
                'best_mutations': mut_data['best_mutations'],      # (L,)
                'n_tolerated': mut_data['n_tolerated'],            # (L,)
            }

            # Save as torch tensors
            cache_tensor = torch.stack([
                pssm_data['pseudo_pssm'],                          # (L, 20)
            ])  # Can't easily stack variable shapes, save as dict

            metadata = CacheMetadata(
                method='pross_labels',
                params={
                    'esm_model': args.model,
                    'length': len(bb.sequence),
                    'mean_conservation': float(pssm_data['conservation'].mean()),
                    'mean_tolerated': float(mut_data['n_tolerated'].float().mean()),
                },
                source=f"{pdb_id} (chain {bb.chain_id})",
            )

            # Save individual tensors for the main PSSM data
            cache.save(cache_key, {
                'pseudo_pssm': pssm_data['pseudo_pssm'].numpy().tolist(),
                'conservation': pssm_data['conservation'].numpy().tolist(),
                'mutation_scores': mut_data['mutation_scores'].numpy().tolist(),
                'n_tolerated': mut_data['n_tolerated'].numpy().tolist(),
            }, metadata)

            elapsed = time.time() - t0
            computed += 1

            mean_cons = pssm_data['conservation'].mean()
            mean_tol = mut_data['n_tolerated'].float().mean()
            logger.info(
                f"{pdb_id}: L={len(bb.sequence)}, "
                f"mean_conservation={mean_cons:.3f}, "
                f"mean_tolerated_mutations={mean_tol:.1f}/19, "
                f"{elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed {pdb_path.name}: {e}")
            if device != 'cpu':
                torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()

    logger.info(f"\nDone: {computed} computed, {skipped} cached")
    logger.info(f"Cache has {len(cache)} entries at {args.cache_dir}")


if __name__ == '__main__':
    main()
