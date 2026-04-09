"""Parallelized mutation scanning for ai-chem (16 cores, no GPU needed).

Runs informative mutation sampling across all proteins in parallel,
outputting batched results so training can start with partial data.

Smart sampling strategy (broad + decision-boundary focused):
- 25% borderline PSSM (score near 0, the PROSS filter threshold)
- 20% active-site adjacent (within 10Å of catalytic residues)
- 15% moderately conserved (conservation 0.3-0.7)
- 15% surface-exposed positions (high SASA proxy)
- 15% buried positions (low SASA proxy, where mutations are impactful)
- 10% random (for coverage and preventing overfitting to any category)

Spread evenly across all proteins to prevent overfitting to one case.

Outputs batched results every N proteins (configurable) so downstream
training can start with partial data.

Usage:
    python scripts/data_prep/remote_mutation_scanning.py --pdb-dir data/pdb --n-workers 16
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

from src.utils.protein_constants import AA_LIST, AA_3TO1, AA_1TO3, AA_1_INDEX, NUM_AA
from src.utils.logging import get_logger

logger = get_logger(__name__)


def select_broad_informative_mutations(
    sequence: str,
    ca_coords: np.ndarray,
    pross_data: Optional[dict],
    catalytic_positions: List[int],
    structural_data: Optional[dict],
    n_mutations: int = 2500,
) -> List[Tuple[int, str, str]]:
    """Select broadly informative mutations for one protein.

    Ensures diversity across mutation types and positions to prevent
    the surrogate from learning shortcuts.

    Returns list of (position_0indexed, wt_aa, mut_aa).
    """
    L = len(sequence)
    mutations = []
    used = set()

    def add(pos, mut_aa):
        if pos >= L:
            return False
        wt = sequence[pos]
        if wt == mut_aa or (pos, mut_aa) in used:
            return False
        mutations.append((pos, wt, mut_aa))
        used.add((pos, mut_aa))
        return True

    all_aas = 'ACDEFGHIKLMNPQRSTVWY'

    # Load data arrays
    pssm_scores = None
    conservation = None
    sasa_proxy = None

    if pross_data:
        if 'mutation_scores' in pross_data:
            pssm_scores = np.array(pross_data['mutation_scores'])
        if 'conservation' in pross_data:
            conservation = np.array(pross_data['conservation'])

    if structural_data:
        if 'sasa_proxy' in structural_data:
            sasa_proxy = np.array(structural_data['sasa_proxy'])

    # === 1. Borderline PSSM (25%) ===
    n1 = int(n_mutations * 0.25)
    if pssm_scores is not None:
        borderline = []
        for i in range(min(L, pssm_scores.shape[0])):
            for j in range(NUM_AA):
                score = pssm_scores[i][j]
                if score != float('-inf') and not np.isinf(score):
                    # Borderline: score between -2 and +2 (broader than strict -1 to +1)
                    if -2.0 <= score <= 2.0:
                        aa_1 = AA_3TO1.get(AA_LIST[j], 'A')
                        borderline.append((i, aa_1, abs(score)))
        borderline.sort(key=lambda x: x[2])  # closest to 0 first
        for pos, aa, _ in borderline[:n1 * 2]:
            add(pos, aa)
            if len(mutations) >= n1:
                break

    # === 2. Active-site adjacent (20%) ===
    n2 = n1 + int(n_mutations * 0.20)
    if catalytic_positions and len(ca_coords) > 0:
        valid_cat = [p for p in catalytic_positions if p < L]
        if valid_cat:
            cat_coords = ca_coords[valid_cat]
            adjacent = []
            for i in range(L):
                if i in valid_cat:
                    continue
                min_dist = np.min(np.linalg.norm(ca_coords[i] - cat_coords, axis=-1))
                if min_dist <= 12.0:  # broader radius: 12Å
                    adjacent.append((i, min_dist))
            adjacent.sort(key=lambda x: x[1])
            for pos, dist in adjacent:
                wt = sequence[pos]
                # More mutations for closer positions
                n_muts = 3 if dist < 6.0 else 2 if dist < 9.0 else 1
                candidates = [aa for aa in all_aas if aa != wt]
                np.random.shuffle(candidates)
                for aa in candidates[:n_muts]:
                    add(pos, aa)
                if len(mutations) >= n2:
                    break

    # === 3. Moderately conserved (15%) ===
    n3 = n2 + int(n_mutations * 0.15)
    if conservation is not None:
        moderate = [(i, conservation[i]) for i in range(min(L, len(conservation)))
                    if 0.3 <= conservation[i] <= 0.7]
        np.random.shuffle(moderate)
        for pos, _ in moderate:
            wt = sequence[pos]
            candidates = [aa for aa in all_aas if aa != wt]
            np.random.shuffle(candidates)
            for aa in candidates[:3]:
                add(pos, aa)
            if len(mutations) >= n3:
                break

    # === 4. Surface-exposed positions (15%) ===
    n4 = n3 + int(n_mutations * 0.15)
    if sasa_proxy is not None:
        # High SASA = surface-exposed
        surface = [(i, sasa_proxy[i]) for i in range(min(L, len(sasa_proxy)))
                   if sasa_proxy[i] > 0.5]
        surface.sort(key=lambda x: -x[1])  # most exposed first
        for pos, _ in surface:
            wt = sequence[pos]
            candidates = [aa for aa in all_aas if aa != wt]
            np.random.shuffle(candidates)
            for aa in candidates[:2]:
                add(pos, aa)
            if len(mutations) >= n4:
                break

    # === 5. Buried positions (15%) ===
    n5 = n4 + int(n_mutations * 0.15)
    if sasa_proxy is not None:
        buried = [(i, sasa_proxy[i]) for i in range(min(L, len(sasa_proxy)))
                  if sasa_proxy[i] < 0.3]
        buried.sort(key=lambda x: x[1])  # most buried first
        for pos, _ in buried:
            wt = sequence[pos]
            candidates = [aa for aa in all_aas if aa != wt]
            np.random.shuffle(candidates)
            for aa in candidates[:2]:
                add(pos, aa)
            if len(mutations) >= n5:
                break

    # === 6. Random for coverage (10%) ===
    n6 = n_mutations
    attempts = 0
    while len(mutations) < n6 and attempts < n6 * 5:
        pos = np.random.randint(0, L)
        wt = sequence[pos]
        mut = np.random.choice([aa for aa in all_aas if aa != wt])
        add(pos, mut)
        attempts += 1

    return mutations[:n_mutations]


def process_one_protein(args):
    """Process a single protein (for multiprocessing).

    Returns list of mutation result dicts.
    """
    pdb_path, pross_data, structural_data, catalytic_positions, n_mutations = args
    pdb_id = Path(pdb_path).stem.upper()

    try:
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

        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        wt_score = sfxn(pose)

        # Extract sequence and CA coords from pose
        sequence = pose.sequence()
        L = pose.total_residue()
        ca_coords = np.zeros((L, 3))
        for i in range(L):
            xyz = pose.residue(i + 1).xyz("CA")
            ca_coords[i] = [xyz.x, xyz.y, xyz.z]

        # Select mutations
        mutations = select_broad_informative_mutations(
            sequence, ca_coords, pross_data, catalytic_positions,
            structural_data, n_mutations=n_mutations,
        )

        results = []
        for pos_0, wt_aa, mut_aa in mutations:
            pos_1 = pos_0 + 1
            if pos_1 > L:
                continue

            actual_wt = pose.residue(pos_1).name1()
            if actual_wt != wt_aa:
                continue

            try:
                mutant_pose = pose.clone()
                mut_resname = {
                    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                }.get(mut_aa, 'ALA')

                pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                    pos_1, mut_resname
                ).apply(mutant_pose)

                # Repack shell
                tf = TaskFactory()
                tf.push_back(RestrictToRepacking())
                mut_selector = ResidueIndexSelector(str(pos_1))
                nbr_selector = NeighborhoodResidueSelector(mut_selector, 8.0, True)
                prevent = PreventRepackingRLT()
                tf.push_back(OperateOnResidueSubset(prevent, nbr_selector, True))

                packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
                packer.task_factory(tf)
                packer.apply(mutant_pose)

                mut_score = sfxn(mutant_pose)
                ddg = mut_score - wt_score

                # Per-residue energies
                res_e_mut = mutant_pose.energies().residue_total_energy(pos_1)
                res_e_wt = pose.energies().residue_total_energy(pos_1)

                results.append({
                    'pdb_id': pdb_id,
                    'position': pos_0,
                    'wt_aa': wt_aa,
                    'mut_aa': mut_aa,
                    'wt_score': float(wt_score),
                    'mut_score': float(mut_score),
                    'ddg': float(ddg),
                    'residue_ddg': float(res_e_mut - res_e_wt),
                    'seq_length': L,
                })

            except Exception:
                continue

        return pdb_id, results

    except Exception as e:
        return pdb_id, []


def main():
    parser = argparse.ArgumentParser(description='Parallel mutation scanning')
    parser.add_argument('--pdb-dir', type=str, default='data/pdb')
    parser.add_argument('--pross-cache', type=str, default='cache/pross_labels')
    parser.add_argument('--structural-cache', type=str, default='cache/structure_features')
    parser.add_argument('--catalytic-dir', type=str, default='data/catalytic_sites')
    parser.add_argument('--output-dir', type=str, default='cache/mutation_scanning')
    parser.add_argument('--n-mutations-per-protein', type=int, default=2500,
                        help='Mutations per protein (2500 × 41 ≈ 100K)')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Save results every N proteins')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    n_workers = args.n_workers or cpu_count()

    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    logger.info(f"Processing {len(pdb_files)} PDBs × {args.n_mutations_per_protein} mutations = "
                f"~{len(pdb_files) * args.n_mutations_per_protein:,} total, {n_workers} workers")

    # Load cached PROSS labels and structural features
    from src.utils.feature_cache import FeatureCache, get_sequence_hash
    from src.data.pdb_loader import load_pdb

    pross_cache = FeatureCache(args.pross_cache)
    struct_cache = FeatureCache(args.structural_cache)

    # Prepare work items
    work_items = []
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()

        # Check if already done
        batch_file = output_dir / f'{pdb_id}_mutations.json'
        if batch_file.exists():
            logger.info(f"{pdb_id}: already done, skipping")
            continue

        # Load PROSS data
        try:
            bb = load_pdb(str(pdb_path))
            seq_hash = get_sequence_hash(bb.sequence)
            pross_key = {'sequence_hash': seq_hash, 'method': 'pross_labels', 'model': 'esm2_t33_650M_UR50D'}
            pross_data = pross_cache.load(pross_key) if pross_cache.has(pross_key) else None
        except Exception:
            pross_data = None

        # Load structural data
        struct_key = {'pdb_id': pdb_id, 'method': 'structural_features'}
        structural_data = struct_cache.load(struct_key) if struct_cache.has(struct_key) else None

        # Load catalytic positions
        cat_positions = []
        yaml_path = Path(args.catalytic_dir) / f'{pdb_id}.yaml'
        if yaml_path.exists():
            try:
                from src.data.catalytic_constraints import load_constraint_from_yaml
                constraint = load_constraint_from_yaml(str(yaml_path))
                cat_positions = [r.position_index for r in constraint.residues if r.position_index is not None]
            except Exception:
                pass

        work_items.append((
            str(pdb_path), pross_data, structural_data, cat_positions,
            args.n_mutations_per_protein,
        ))

    if not work_items:
        logger.info("All proteins already processed!")
        return

    logger.info(f"Queued {len(work_items)} proteins for processing")

    # Process in parallel, save in batches
    all_results = []
    total_mutations = 0
    batch_num = 0
    start_time = time.time()

    # Use sequential processing per protein (PyRosetta init is per-process)
    # but we can use Pool for parallel proteins
    with Pool(processes=min(n_workers, len(work_items))) as pool:
        for i, (pdb_id, results) in enumerate(pool.imap_unordered(process_one_protein, work_items)):
            if results:
                # Save per-protein results
                batch_file = output_dir / f'{pdb_id}_mutations.json'
                with open(batch_file, 'w') as f:
                    json.dump(results, f)

                total_mutations += len(results)
                all_results.extend(results)

                ddgs = [r['ddg'] for r in results]
                logger.info(
                    f"[{i+1}/{len(work_items)}] {pdb_id}: {len(results)} mutations, "
                    f"ΔΔG=[{min(ddgs):.1f}, {max(ddgs):.1f}], "
                    f"total so far: {total_mutations:,}"
                )
            else:
                logger.warning(f"[{i+1}/{len(work_items)}] {pdb_id}: failed or empty")

            # Save batch summary periodically
            if (i + 1) % args.batch_size == 0 or i == len(work_items) - 1:
                batch_num += 1
                elapsed = time.time() - start_time
                summary = {
                    'batch': batch_num,
                    'proteins_processed': i + 1,
                    'total_mutations': total_mutations,
                    'elapsed_seconds': elapsed,
                    'mutations_per_second': total_mutations / max(elapsed, 1),
                }
                summary_file = output_dir / f'batch_{batch_num:03d}_summary.json'
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Batch {batch_num}: {total_mutations:,} mutations, {elapsed:.0f}s elapsed")

    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"\n=== COMPLETE ===")
    logger.info(f"Total: {total_mutations:,} mutations across {len(work_items)} proteins in {elapsed:.0f}s")

    if all_results:
        ddgs = [r['ddg'] for r in all_results]
        logger.info(f"ΔΔG: mean={np.mean(ddgs):.2f}, std={np.std(ddgs):.2f}, "
                    f"range=[{np.min(ddgs):.1f}, {np.max(ddgs):.1f}]")
        n_stab = sum(1 for d in ddgs if d < -0.45)
        n_neut = sum(1 for d in ddgs if -0.45 <= d <= 0.45)
        n_dest = sum(1 for d in ddgs if d > 0.45)
        logger.info(f"Stabilizing: {n_stab} ({n_stab/len(ddgs)*100:.0f}%), "
                    f"Neutral: {n_neut} ({n_neut/len(ddgs)*100:.0f}%), "
                    f"Destabilizing: {n_dest} ({n_dest/len(ddgs)*100:.0f}%)")

        # Save combined dataset
        combined_file = output_dir / 'all_mutations.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f)
        logger.info(f"Saved combined dataset to {combined_file}")


if __name__ == '__main__':
    main()
