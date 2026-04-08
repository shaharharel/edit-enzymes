"""Compute Rosetta scores for PDB structures and cache results.

Takes a directory of PDB files, runs PyRosetta scoring, and caches
all results using FeatureCache. Outputs a CSV with score columns
suitable for training scoring surrogate models.

Usage:
    conda run -n quris python scripts/data_prep/compute_rosetta_scores.py \
        --pdb-dir data/pdb --output cache/rosetta_features/rosetta_scores.csv

If PyRosetta is not installed, generates placeholder features from
backbone geometry for development/testing.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import csv

import numpy as np

from src.utils.feature_cache import FeatureCache, CacheMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _try_import_pyrosetta():
    """Try to import pyrosetta, return None if not available."""
    try:
        import pyrosetta
        pyrosetta.init('-mute all', silent=True)
        return pyrosetta
    except ImportError:
        logger.warning(
            "PyRosetta not installed. Will compute geometric features only. "
            "Install PyRosetta for full Rosetta scoring."
        )
        return None


def compute_geometric_features(pdb_path: str) -> dict:
    """Compute backbone geometry features as fallback when PyRosetta unavailable.

    Extracts basic structural descriptors from PDB coordinates:
    - Per-residue distances, angles, contact counts
    - Global radius of gyration, surface area proxy
    - Active site cavity volume proxy

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dict with 'features' (array), 'scores' (dict), and metadata.
    """
    from src.data.protein_structure import ProteinBackbone

    try:
        bb = ProteinBackbone.from_pdb(pdb_path)
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_path}: {e}")
        return None

    coords = bb.coords  # (L, 4, 3)
    ca_coords = coords[:, 1]  # CA atoms
    L = len(ca_coords)

    if L < 5:
        logger.warning(f"Skipping {pdb_path}: too few residues ({L})")
        return None

    # Global features
    centroid = ca_coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1)))

    # Contact map (CA-CA < 8A)
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(ca_coords))
    contact_counts = (dist_matrix < 8.0).sum(axis=1).astype(float)
    avg_contacts = contact_counts.mean()

    # Per-residue features (averaged to fixed-length vector)
    # Local packing density
    local_density = contact_counts / max(L, 1)

    # Sequential CA-CA distances
    ca_dists = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
    avg_ca_dist = ca_dists.mean()

    # Dihedral-like features (phi/psi proxy from CA trace)
    if L >= 4:
        angles = []
        for i in range(L - 3):
            v1 = ca_coords[i + 1] - ca_coords[i]
            v2 = ca_coords[i + 2] - ca_coords[i + 1]
            v3 = ca_coords[i + 3] - ca_coords[i + 2]
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm > 1e-8 and n2_norm > 1e-8:
                cos_angle = np.clip(np.dot(n1, n2) / (n1_norm * n2_norm), -1, 1)
                angles.append(cos_angle)
        avg_dihedral = np.mean(angles) if angles else 0.0
    else:
        avg_dihedral = 0.0

    # Build feature vector (fixed size for MLP input)
    # Histogram of contact counts (8 bins)
    contact_hist, _ = np.histogram(contact_counts, bins=8, range=(0, 30))
    contact_hist = contact_hist.astype(float) / max(L, 1)

    # Histogram of CA distances (8 bins)
    dist_hist, _ = np.histogram(ca_dists, bins=8, range=(3.0, 5.0))
    dist_hist = dist_hist.astype(float) / max(len(ca_dists), 1)

    # Distance from centroid histogram (8 bins)
    centroid_dists = np.linalg.norm(ca_coords - centroid, axis=1)
    centroid_hist, _ = np.histogram(centroid_dists, bins=8, range=(0, rg * 3))
    centroid_hist = centroid_hist.astype(float) / max(L, 1)

    # Assemble feature vector
    feature_parts = [
        [L / 500.0],            # normalized length
        [rg / 30.0],            # normalized radius of gyration
        [avg_contacts / 20.0],  # normalized avg contacts
        [avg_ca_dist / 5.0],    # normalized avg CA distance
        [avg_dihedral],         # average dihedral cosine
        contact_hist.tolist(),  # 8 features
        dist_hist.tolist(),     # 8 features
        centroid_hist.tolist(), # 8 features
    ]
    features = []
    for part in feature_parts:
        if isinstance(part, list):
            features.extend(part)
        else:
            features.append(part)

    # Pad or truncate to target dim (will be padded to input_dim in training)
    features = np.array(features, dtype=np.float32)

    # Compute proxy scores from geometry
    scores = {
        'stability': float(-rg * 0.5 + avg_contacts * 0.1),  # proxy
        'packing': float(avg_contacts / 15.0),                # proxy
        'desolvation': float(rg * 0.3 - avg_contacts * 0.05), # proxy
        'activity': float(avg_dihedral * 2.0 + 0.5),          # proxy
    }

    return {
        'features': features,
        'scores': scores,
        'pdb_id': Path(pdb_path).stem,
        'n_residues': L,
    }


def compute_rosetta_scores(pdb_path: str, pyrosetta) -> dict:
    """Compute full Rosetta scores for a PDB file.

    Args:
        pdb_path: Path to PDB file
        pyrosetta: Imported pyrosetta module

    Returns:
        Dict with 'features', 'scores', and metadata.
    """
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreFunctionFactory

    try:
        pose = pose_from_pdb(str(pdb_path))
    except Exception as e:
        logger.warning(f"PyRosetta failed to load {pdb_path}: {e}")
        return None

    # Score with ref2015
    sfxn = ScoreFunctionFactory.create_score_function('ref2015')
    total_score = sfxn(pose)

    # Extract individual energy terms
    energies = pose.energies()
    n_res = pose.total_residue()

    # Per-residue energies (averaged)
    per_res_total = []
    per_res_fa_atr = []
    per_res_fa_rep = []
    per_res_fa_sol = []
    per_res_hbond = []

    for i in range(1, n_res + 1):
        from pyrosetta.rosetta.core.scoring import ScoreType
        per_res_total.append(energies.residue_total_energy(i))
        try:
            per_res_fa_atr.append(energies.residue_total_energies(i)[ScoreType.fa_atr])
            per_res_fa_rep.append(energies.residue_total_energies(i)[ScoreType.fa_rep])
            per_res_fa_sol.append(energies.residue_total_energies(i)[ScoreType.fa_sol])
            per_res_hbond.append(
                energies.residue_total_energies(i)[ScoreType.hbond_sr_bb] +
                energies.residue_total_energies(i)[ScoreType.hbond_lr_bb] +
                energies.residue_total_energies(i)[ScoreType.hbond_bb_sc] +
                energies.residue_total_energies(i)[ScoreType.hbond_sc]
            )
        except Exception:
            per_res_fa_atr.append(0.0)
            per_res_fa_rep.append(0.0)
            per_res_fa_sol.append(0.0)
            per_res_hbond.append(0.0)

    # Build feature vector from energy histograms
    def make_hist(values, n_bins=8, lo=-5, hi=5):
        h, _ = np.histogram(values, bins=n_bins, range=(lo, hi))
        return h.astype(float) / max(len(values), 1)

    features = np.concatenate([
        [n_res / 500.0, total_score / n_res],
        make_hist(per_res_total),
        make_hist(per_res_fa_atr, lo=-10, hi=0),
        make_hist(per_res_fa_rep, lo=0, hi=10),
        make_hist(per_res_fa_sol, lo=-5, hi=5),
        make_hist(per_res_hbond, lo=-5, hi=0),
    ]).astype(np.float32)

    scores = {
        'stability': float(total_score / n_res),
        'packing': float(-np.mean(per_res_fa_rep)),
        'desolvation': float(np.mean(per_res_fa_sol)),
        'activity': float(-np.mean(per_res_hbond)),
    }

    return {
        'features': features,
        'scores': scores,
        'pdb_id': Path(pdb_path).stem,
        'n_residues': n_res,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute Rosetta scores for PDB structures'
    )
    parser.add_argument('--pdb-dir', type=str, required=True,
                        help='Directory containing PDB files')
    parser.add_argument('--output', type=str,
                        default='cache/rosetta_features/rosetta_scores.csv',
                        help='Output CSV path')
    parser.add_argument('--cache-dir', type=str,
                        default='cache/rosetta_features',
                        help='Cache directory for intermediate results')
    parser.add_argument('--target-dim', type=int, default=256,
                        help='Target feature dimension (zero-pad if shorter)')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.exists():
        logger.error(f"PDB directory not found: {pdb_dir}")
        sys.exit(1)

    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    if not pdb_files:
        logger.error(f"No PDB files found in {pdb_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")

    # Initialize cache
    cache = FeatureCache(args.cache_dir)

    # Try PyRosetta
    pyrosetta = _try_import_pyrosetta()

    # Process each PDB
    all_results = []
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem
        cache_key = {'pdb_id': pdb_id, 'method': 'rosetta_scores'}

        # Check cache
        if cache.has(cache_key):
            result = cache.load(cache_key)
            logger.info(f"Cache hit: {pdb_id}")
        else:
            # Compute scores
            if pyrosetta is not None:
                result = compute_rosetta_scores(str(pdb_path), pyrosetta)
            else:
                result = compute_geometric_features(str(pdb_path))

            if result is None:
                continue

            # Cache result
            cache.save(
                cache_key, result,
                metadata=CacheMetadata(
                    method='pyrosetta' if pyrosetta else 'geometric',
                    source=pdb_id,
                ),
            )
            logger.info(f"Computed and cached: {pdb_id}")

        all_results.append(result)

    if not all_results:
        logger.error("No results computed. Check PDB files.")
        sys.exit(1)

    # Pad features to target_dim
    target_dim = args.target_dim
    padded_features = []
    for r in all_results:
        feat = r['features']
        if len(feat) < target_dim:
            feat = np.pad(feat, (0, target_dim - len(feat)))
        elif len(feat) > target_dim:
            feat = feat[:target_dim]
        padded_features.append(feat)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        feat_cols = [f'feat_{i}' for i in range(target_dim)]
        score_cols = ['stability', 'packing', 'desolvation', 'activity']
        header = ['pdb_id', 'n_residues'] + feat_cols + score_cols
        writer.writerow(header)

        # Data rows
        for i, r in enumerate(all_results):
            row = [r['pdb_id'], r['n_residues']]
            row.extend(padded_features[i].tolist())
            row.extend([r['scores'].get(s, 0.0) for s in score_cols])
            writer.writerow(row)

    logger.info(f"Wrote {len(all_results)} rows to {output_path}")

    # Also save as tensors for fast loading
    import torch
    features_tensor = torch.tensor(np.array(padded_features), dtype=torch.float32)
    targets_dict = {}
    for score_name in score_cols:
        vals = [r['scores'].get(score_name, 0.0) for r in all_results]
        targets_dict[score_name] = torch.tensor(vals, dtype=torch.float32)

    torch.save(features_tensor, output_path.parent / 'features.pt')
    torch.save(targets_dict, output_path.parent / 'targets.pt')
    logger.info(f"Saved tensor cache to {output_path.parent}")


if __name__ == '__main__':
    main()
