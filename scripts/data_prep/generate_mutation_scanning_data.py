"""Generate informative mutation scanning data for scoring surrogate training.

Instead of exhaustive scanning (190K trivial mutations), we sample mutations
that are informative for training — near the decision boundary where the
surrogate's discrimination ability actually matters.

Sampling strategy:
1. PSSM-guided: mutations with borderline PSSM scores (near 0, the PROSS threshold)
2. Active-site adjacent: positions near catalytic residues (within 10Å)
3. ProteinMPNN-proposed: mutations the sequence generator would actually suggest
4. Conservation-weighted: more samples at moderately conserved positions

For each selected mutation, we compute:
- Rosetta ΔΔG via fast ddG protocol (score difference, no full relax)
- Per-residue energy decomposition
- Sidechain packing metrics

Usage:
    python scripts/data_prep/generate_mutation_scanning_data.py
    python scripts/data_prep/generate_mutation_scanning_data.py --n-mutations 5000 --n-workers 4
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.data.pdb_loader import load_pdb
from src.data.protein_structure import ProteinBackbone
from src.utils.feature_cache import FeatureCache, CacheMetadata, get_sequence_hash
from src.utils.protein_constants import AA_LIST, AA_3TO1, AA_1TO3, AA_1_INDEX, NUM_AA
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_pross_labels(cache_dir: str, sequence: str) -> Optional[dict]:
    """Load cached PROSS pseudo-PSSM labels for a sequence."""
    cache = FeatureCache(cache_dir)
    seq_hash = get_sequence_hash(sequence)
    key = {'sequence_hash': seq_hash, 'method': 'pross_labels', 'model': 'esm2_t33_650M_UR50D'}
    if cache.has(key):
        return cache.load(key)
    return None


def load_catalytic_positions(yaml_dir: str, pdb_id: str) -> List[int]:
    """Load catalytic residue positions from YAML."""
    yaml_path = Path(yaml_dir) / f'{pdb_id}.yaml'
    if not yaml_path.exists():
        return []
    from src.data.catalytic_constraints import load_constraint_from_yaml
    constraint = load_constraint_from_yaml(str(yaml_path))
    return [r.position_index for r in constraint.residues if r.position_index is not None]


def select_informative_mutations(
    backbone: ProteinBackbone,
    pross_data: Optional[dict],
    catalytic_positions: List[int],
    n_mutations: int = 200,
) -> List[Tuple[int, str, str]]:
    """Select informative mutations for a single protein.

    Returns list of (position, wt_aa, mut_aa) tuples.

    Sampling strategy:
    - 30% borderline PSSM (score between -1 and +1)
    - 25% active-site adjacent (within 10Å of catalytic residues)
    - 25% moderately conserved positions
    - 20% random (for coverage)
    """
    if backbone.sequence is None:
        return []

    L = backbone.length
    seq = backbone.sequence
    ca_coords = backbone.ca_coords  # (L, 3)

    mutations = []
    used = set()  # (pos, mut_aa) pairs already selected

    def add_mutation(pos: int, mut_aa_1: str):
        if pos >= L:
            return
        wt = seq[pos]
        if wt == mut_aa_1 or (pos, mut_aa_1) in used:
            return
        mutations.append((pos, wt, mut_aa_1))
        used.add((pos, mut_aa_1))

    # 1. Borderline PSSM mutations (30%)
    n_borderline = int(n_mutations * 0.30)
    if pross_data and 'mutation_scores' in pross_data:
        scores = np.array(pross_data['mutation_scores'])  # (L, 20)
        # Find mutations with PSSM score between -1 and +1 (borderline)
        borderline_positions = []
        for i in range(min(L, scores.shape[0])):
            for j in range(NUM_AA):
                if scores[i][j] != float('-inf') and -1.0 <= scores[i][j] <= 1.0:
                    aa_1 = AA_3TO1.get(AA_LIST[j], 'A')
                    borderline_positions.append((i, aa_1, abs(scores[i][j])))

        # Sort by closeness to 0 (hardest cases)
        borderline_positions.sort(key=lambda x: x[2])
        for pos, aa, _ in borderline_positions[:n_borderline]:
            add_mutation(pos, aa)

    # 2. Active-site adjacent mutations (25%)
    n_active = int(n_mutations * 0.25)
    if catalytic_positions and len(ca_coords) > 0:
        # Find positions within 10Å of any catalytic residue
        cat_coords = ca_coords[catalytic_positions] if catalytic_positions else ca_coords[:0]
        if len(cat_coords) > 0:
            adjacent = []
            for i in range(L):
                if i in catalytic_positions:
                    continue  # don't mutate catalytic residues themselves
                min_dist = np.min(np.linalg.norm(ca_coords[i] - cat_coords, axis=-1))
                if min_dist <= 10.0:
                    adjacent.append((i, min_dist))

            adjacent.sort(key=lambda x: x[1])  # closest first
            for pos, _ in adjacent:
                if len([m for m in mutations if m[0] == pos]) >= 3:
                    continue  # max 3 mutations per position
                # Pick 2-3 random non-WT amino acids
                wt = seq[pos]
                candidates = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt]
                np.random.shuffle(candidates)
                for aa in candidates[:2]:
                    add_mutation(pos, aa)
                    if len(mutations) >= n_borderline + n_active:
                        break
                if len(mutations) >= n_borderline + n_active:
                    break

    # 3. Moderately conserved positions (25%)
    n_conserved = int(n_mutations * 0.25)
    if pross_data and 'conservation' in pross_data:
        conservation = np.array(pross_data['conservation'])
        # Moderately conserved: 0.3-0.7 range
        moderate = [(i, conservation[i]) for i in range(min(L, len(conservation)))
                    if 0.3 <= conservation[i] <= 0.7]
        np.random.shuffle(moderate)
        for pos, _ in moderate:
            wt = seq[pos]
            candidates = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt]
            np.random.shuffle(candidates)
            for aa in candidates[:2]:
                add_mutation(pos, aa)
                if len(mutations) >= n_borderline + n_active + n_conserved:
                    break
            if len(mutations) >= n_borderline + n_active + n_conserved:
                break

    # 4. Random mutations for coverage (20%)
    n_random = n_mutations - len(mutations)
    for _ in range(n_random * 3):  # oversample, some will be duplicates
        pos = np.random.randint(0, L)
        wt = seq[pos]
        mut = np.random.choice([aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != wt])
        add_mutation(pos, mut)
        if len(mutations) >= n_mutations:
            break

    return mutations[:n_mutations]


def compute_fast_ddg(
    pdb_path: str,
    mutations: List[Tuple[int, str, str]],
) -> List[dict]:
    """Compute fast ΔΔG for a list of mutations using PyRosetta.

    Uses score-only protocol (no relax) for speed:
    - Mutate residue
    - Repack sidechains in 8Å shell
    - Score WT and mutant
    - ΔΔG = score(mutant) - score(WT)

    Args:
        pdb_path: Path to PDB file
        mutations: List of (position_0indexed, wt_aa_1letter, mut_aa_1letter)

    Returns:
        List of dicts with mutation info and scores
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        RestrictToRepacking,
        PreventRepackingRLT,
        OperateOnResidueSubset,
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        NeighborhoodResidueSelector,
        ResidueIndexSelector,
    )

    pyrosetta.init('-mute all -ex1 -ex2')
    sfxn = pyrosetta.get_score_function(True)

    pose = pyrosetta.pose_from_pdb(pdb_path)
    wt_score = sfxn(pose)

    results = []

    for pos_0, wt_aa, mut_aa in mutations:
        pos_1 = pos_0 + 1  # PyRosetta uses 1-indexed

        if pos_1 > pose.total_residue():
            continue

        # Check WT matches
        actual_wt = pose.residue(pos_1).name1()
        if actual_wt != wt_aa:
            # Numbering mismatch, skip
            continue

        try:
            # Create mutant pose
            mutant_pose = pose.clone()

            # Mutate
            mut_resname = AA_1TO3.get(mut_aa, 'ALA')
            pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                pos_1, mut_resname
            ).apply(mutant_pose)

            # Repack shell around mutation (8Å)
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())

            # Only repack near the mutation site
            mut_selector = ResidueIndexSelector(str(pos_1))
            nbr_selector = NeighborhoodResidueSelector(mut_selector, 8.0, True)

            prevent = PreventRepackingRLT()
            tf.push_back(OperateOnResidueSubset(prevent, nbr_selector, True))

            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
            packer.task_factory(tf)
            packer.apply(mutant_pose)

            mut_score = sfxn(mutant_pose)
            ddg = mut_score - wt_score

            # Per-residue energy at mutation site
            residue_energy = mutant_pose.energies().residue_total_energy(pos_1)
            wt_residue_energy = pose.energies().residue_total_energy(pos_1)

            results.append({
                'position': pos_0,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'wt_score': float(wt_score),
                'mut_score': float(mut_score),
                'ddg': float(ddg),
                'wt_residue_energy': float(wt_residue_energy),
                'mut_residue_energy': float(residue_energy),
                'residue_ddg': float(residue_energy - wt_residue_energy),
            })

        except Exception as e:
            logger.debug(f"Failed mutation {wt_aa}{pos_1}{mut_aa}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate mutation scanning data')
    parser.add_argument('--pdb-dir', type=str, default='data/pdb')
    parser.add_argument('--pross-cache', type=str, default='cache/pross_labels')
    parser.add_argument('--catalytic-dir', type=str, default='data/catalytic_sites')
    parser.add_argument('--output-cache', type=str, default='cache/mutation_scanning')
    parser.add_argument('--n-mutations-per-protein', type=int, default=150,
                        help='Informative mutations per protein (~150 × 41 = ~6K total)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)

    pdb_dir = Path(args.pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    cache = FeatureCache(args.output_cache)

    logger.info(f"Processing {len(pdb_files)} PDB files, {args.n_mutations_per_protein} mutations each")

    all_results = []
    total_mutations = 0
    total_time = 0

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()
        cache_key = {'pdb_id': pdb_id, 'method': 'mutation_scanning', 'n_per_protein': args.n_mutations_per_protein}

        if not args.force and cache.has(cache_key):
            cached = cache.load(cache_key)
            logger.info(f"{pdb_id}: {len(cached)} mutations (cached)")
            all_results.extend(cached)
            total_mutations += len(cached)
            continue

        try:
            bb = load_pdb(str(pdb_path))
            if bb.sequence is None:
                continue

            # Load PROSS labels
            pross_data = load_pross_labels(args.pross_cache, bb.sequence)

            # Load catalytic positions
            cat_positions = load_catalytic_positions(args.catalytic_dir, pdb_id)

            # Select informative mutations
            mutations = select_informative_mutations(
                bb, pross_data, cat_positions,
                n_mutations=args.n_mutations_per_protein,
            )

            if not mutations:
                logger.warning(f"{pdb_id}: no mutations selected")
                continue

            logger.info(f"{pdb_id}: scoring {len(mutations)} mutations...")
            t0 = time.time()

            # Compute ΔΔG
            results = compute_fast_ddg(str(pdb_path), mutations)

            elapsed = time.time() - t0
            total_time += elapsed

            # Add metadata
            for r in results:
                r['pdb_id'] = pdb_id
                r['seq_length'] = bb.length

            # Cache
            cache.save(cache_key, results, CacheMetadata(
                method='mutation_scanning',
                params={
                    'n_mutations': len(results),
                    'n_requested': len(mutations),
                    'strategy': 'informative_sampling',
                },
                source=pdb_id,
            ))

            all_results.extend(results)
            total_mutations += len(results)

            logger.info(
                f"{pdb_id}: {len(results)}/{len(mutations)} mutations scored, "
                f"ΔΔG range [{min(r['ddg'] for r in results):.1f}, {max(r['ddg'] for r in results):.1f}], "
                f"{elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"{pdb_id}: failed - {e}")

    # Save summary
    logger.info(f"\nDone: {total_mutations} mutations across {len(pdb_files)} proteins, {total_time:.0f}s total")

    if all_results:
        ddgs = [r['ddg'] for r in all_results]
        logger.info(f"ΔΔG distribution: mean={np.mean(ddgs):.2f}, std={np.std(ddgs):.2f}, "
                    f"range=[{np.min(ddgs):.1f}, {np.max(ddgs):.1f}]")
        n_stabilizing = sum(1 for d in ddgs if d < -0.45)
        n_neutral = sum(1 for d in ddgs if -0.45 <= d <= 0.45)
        n_destabilizing = sum(1 for d in ddgs if d > 0.45)
        logger.info(f"Stabilizing: {n_stabilizing} ({n_stabilizing/len(ddgs)*100:.0f}%), "
                    f"Neutral: {n_neutral} ({n_neutral/len(ddgs)*100:.0f}%), "
                    f"Destabilizing: {n_destabilizing} ({n_destabilizing/len(ddgs)*100:.0f}%)")


if __name__ == '__main__':
    main()
