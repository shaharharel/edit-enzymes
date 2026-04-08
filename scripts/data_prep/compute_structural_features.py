"""Compute structural features for all PDB structures and cache them.

Extracts backbone geometry, inter-residue distances, and local structural
properties. These features are used by scoring models and as input to the
sequence generator.

CPU-only computation, lightweight.

Usage:
    python scripts/data_prep/compute_structural_features.py
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
from src.data.protein_structure import ProteinBackbone
from src.utils.feature_cache import FeatureCache, CacheMetadata
from src.utils.metrics import bond_geometry_metrics, clash_score
from src.utils.geometry import pairwise_distances, backbone_frames
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_structural_features(bb: ProteinBackbone) -> dict:
    """Compute structural features for a single protein.

    Returns dict with:
        - ca_distances: (L, L) CA pairwise distance matrix
        - contact_map_8A: (L, L) binary contacts at 8Å threshold
        - contact_map_12A: (L, L) binary contacts at 12Å threshold
        - local_geometry: (L, 6) per-residue bond lengths and angles
        - relative_sasa_proxy: (L,) approximate relative SASA from CA distances
        - secondary_structure_proxy: (L,) local curvature as SS proxy
        - geometry_quality: dict of bond geometry metrics
        - clash_score: fraction of clashing CA pairs
    """
    coords_t = torch.tensor(bb.coords, dtype=torch.float32)
    ca = coords_t[:, 1]  # (L, 3)
    L = bb.length

    # Pairwise CA distances
    ca_dist = pairwise_distances(ca)  # (L, L)

    # Contact maps at different thresholds
    contact_8 = (ca_dist < 8.0).float()
    contact_12 = (ca_dist < 12.0).float()

    # Local geometry (bond lengths per residue)
    local_geom = torch.zeros(L, 6)
    local_geom[:, 0] = torch.norm(coords_t[:, 1] - coords_t[:, 0], dim=-1)  # N-CA
    local_geom[:, 1] = torch.norm(coords_t[:, 2] - coords_t[:, 1], dim=-1)  # CA-C
    local_geom[:, 2] = torch.norm(coords_t[:, 3] - coords_t[:, 2], dim=-1)  # C-O

    # N-CA-C angle
    for i in range(L):
        v1 = coords_t[i, 0] - coords_t[i, 1]
        v2 = coords_t[i, 2] - coords_t[i, 1]
        cos_a = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
        local_geom[i, 3] = torch.acos(cos_a.clamp(-1, 1))

    # Peptide bond angle (C-N across consecutive residues)
    if L > 1:
        for i in range(L - 1):
            v1 = coords_t[i, 1] - coords_t[i, 2]
            v2 = coords_t[i + 1, 0] - coords_t[i, 2]
            cos_a = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
            local_geom[i, 4] = torch.acos(cos_a.clamp(-1, 1))

    # Approximate relative SASA proxy: residues with fewer contacts are more exposed
    # Count neighbors within 10Å, normalize
    n_contacts = (ca_dist < 10.0).float().sum(dim=1) - 1  # exclude self
    max_contacts = n_contacts.max()
    sasa_proxy = 1.0 - (n_contacts / (max_contacts + 1e-8))  # higher = more exposed

    # Secondary structure proxy via local CA curvature
    # Measure deviation from linear chain using CA triplet angles
    ss_proxy = torch.zeros(L)
    if L >= 3:
        for i in range(1, L - 1):
            v1 = ca[i] - ca[i - 1]
            v2 = ca[i + 1] - ca[i]
            cos_a = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
            ss_proxy[i] = cos_a  # ~1.0 = straight (strand), ~0.5 = helix, ~-1 = turn

    # Geometry quality metrics
    geom_quality = bond_geometry_metrics(coords_t)
    cs = clash_score(coords_t)

    return {
        'ca_distances': ca_dist,           # (L, L)
        'contact_map_8A': contact_8,       # (L, L)
        'contact_map_12A': contact_12,     # (L, L)
        'local_geometry': local_geom,      # (L, 6)
        'sasa_proxy': sasa_proxy,          # (L,)
        'ss_proxy': ss_proxy,              # (L,)
        'geometry_quality': geom_quality,  # dict
        'clash_score': cs,                 # float
        'length': L,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute structural features')
    parser.add_argument('--pdb-dir', type=str, default='data/pdb')
    parser.add_argument('--cache-dir', type=str, default='cache/structure_features')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))

    if not pdb_files:
        logger.error(f"No PDB files in {pdb_dir}")
        return

    cache = FeatureCache(args.cache_dir)
    logger.info(f"Processing {len(pdb_files)} PDBs, cache has {len(cache)} entries")

    computed = 0
    skipped = 0

    start = time.time()
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()
        cache_key = {'pdb_id': pdb_id, 'method': 'structural_features'}

        if not args.force and cache.has(cache_key):
            skipped += 1
            continue

        try:
            bb = load_pdb(str(pdb_path))

            features = compute_structural_features(bb)

            # Save tensors that are useful for scoring models
            # Pack into a single dict for caching
            cache_data = {
                'ca_distances': features['ca_distances'].numpy().tolist(),
                'contact_map_8A': features['contact_map_8A'].numpy().tolist(),
                'local_geometry': features['local_geometry'].numpy().tolist(),
                'sasa_proxy': features['sasa_proxy'].numpy().tolist(),
                'ss_proxy': features['ss_proxy'].numpy().tolist(),
                'geometry_quality': features['geometry_quality'],
                'clash_score': features['clash_score'],
                'length': features['length'],
                'sequence': bb.sequence,
            }

            metadata = CacheMetadata(
                method='structural_features',
                params={'version': '1.0'},
                source=f"{pdb_id} (chain {bb.chain_id})",
            )
            cache.save(cache_key, cache_data, metadata)

            computed += 1
            logger.info(
                f"{pdb_id}: L={features['length']}, "
                f"clash={features['clash_score']:.4f}, "
                f"N-CA={features['geometry_quality']['n_ca_bond_mean']:.3f}Å"
            )

        except Exception as e:
            logger.error(f"Failed {pdb_id}: {e}")

    elapsed = time.time() - start
    logger.info(f"\nDone: {computed} computed, {skipped} cached, {elapsed:.1f}s")
    logger.info(f"Cache now has {len(cache)} entries")


if __name__ == '__main__':
    main()
