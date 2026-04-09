"""Score one protein's mutations with full Rosetta energy breakdown.

Saves per-mutation: total ΔΔG + all individual energy terms + per-residue decomposition.

Usage:
    python scan_one_detailed.py <pdb_path> <n_mutations> <output_dir>
"""

import sys, json, time, os, numpy as np

pdb_path = sys.argv[1]
n_muts = int(sys.argv[2])
output_dir = sys.argv[3]
pdb_id = os.path.basename(pdb_path).replace(".pdb", "").upper()
output_file = os.path.join(output_dir, f"{pdb_id}_mutations.json")

if os.path.exists(output_file):
    existing = json.load(open(output_file))
    if len(existing) >= n_muts * 0.8:
        print(f"{pdb_id}: already done ({len(existing)} mutations)")
        sys.exit(0)

import pyrosetta
pyrosetta.init("-mute all -ex1 -ex2")

from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    RestrictToRepacking, PreventRepackingRLT, OperateOnResidueSubset,
)
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector, ResidueIndexSelector,
)

sfxn = pyrosetta.get_score_function(True)

# All score types we want to extract
SCORE_TERMS = [
    'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4',
    'lk_ball_wtd', 'fa_elec', 'pro_close', 'hbond_sr_bb', 'hbond_lr_bb',
    'hbond_bb_sc', 'hbond_sc', 'dslf_fa13', 'omega', 'fa_dun', 'p_aa_pp',
    'yhh_planarity', 'ref', 'rama_prepro',
]

def get_score_term_value(pose, term_name):
    """Get a specific score term from a scored pose."""
    try:
        st = getattr(ScoreType, term_name)
        return float(pose.energies().total_energies()[st])
    except:
        return 0.0

def get_per_residue_energies(pose, pos_1, radius=8.0):
    """Get per-residue energy breakdown at and near a position."""
    energies = {}
    # Energy at mutation site
    energies['site_total'] = float(pose.energies().residue_total_energy(pos_1))

    # Per-term at mutation site
    for term in SCORE_TERMS:
        try:
            st = getattr(ScoreType, term)
            energies[f'site_{term}'] = float(pose.energies().residue_total_energies(pos_1)[st])
        except:
            energies[f'site_{term}'] = 0.0

    # Sum of neighbors within radius
    ca_mut = pose.residue(pos_1).xyz("CA")
    neighbor_total = 0.0
    n_neighbors = 0
    for i in range(1, pose.total_residue() + 1):
        if i == pos_1:
            continue
        try:
            ca_i = pose.residue(i).xyz("CA")
            dist = ca_mut.distance(ca_i)
            if dist <= radius:
                neighbor_total += pose.energies().residue_total_energy(i)
                n_neighbors += 1
        except:
            pass
    energies['neighbor_total'] = float(neighbor_total)
    energies['n_neighbors'] = n_neighbors

    return energies

try:
    pose = pyrosetta.pose_from_pdb(pdb_path)
except Exception as e:
    print(f"{pdb_id}: failed to load - {e}")
    sys.exit(1)

wt_score = sfxn(pose)
sequence = pose.sequence()
L = pose.total_residue()

# Get WT whole-structure energy terms
wt_terms = {}
for term in SCORE_TERMS:
    wt_terms[term] = get_score_term_value(pose, term)

all_aas = "ACDEFGHIKLMNPQRSTVWY"
aa3_map = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

np.random.seed(42 + hash(pdb_id) % 10000)
positions = list(range(L))
np.random.shuffle(positions)

results = []
t0 = time.time()

for pos_0 in positions:
    pos_1 = pos_0 + 1
    if pos_1 > L:
        continue
    wt_aa = pose.residue(pos_1).name1()
    if wt_aa not in all_aas:
        continue

    # Get WT per-residue energies at this position
    wt_residue = get_per_residue_energies(pose, pos_1)

    candidates = [aa for aa in all_aas if aa != wt_aa]
    np.random.shuffle(candidates)

    for mut_aa in candidates[:3]:
        try:
            mut_pose = pose.clone()
            pyrosetta.rosetta.protocols.simple_moves.MutateResidue(
                pos_1, aa3_map[mut_aa]
            ).apply(mut_pose)

            # Repack shell
            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            ms = ResidueIndexSelector(str(pos_1))
            ns = NeighborhoodResidueSelector(ms, 8.0, True)
            prevent = PreventRepackingRLT()
            tf.push_back(OperateOnResidueSubset(prevent, ns, True))
            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
            packer.task_factory(tf)
            packer.apply(mut_pose)

            mut_score = sfxn(mut_pose)

            # Whole-structure energy term breakdown
            mut_terms = {}
            delta_terms = {}
            for term in SCORE_TERMS:
                val = get_score_term_value(mut_pose, term)
                mut_terms[term] = val
                delta_terms[f'd_{term}'] = val - wt_terms[term]

            # Per-residue breakdown at mutation site
            mut_residue = get_per_residue_energies(mut_pose, pos_1)

            # Build result
            result = {
                "pdb_id": pdb_id,
                "position": pos_0,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "seq_length": L,
                # Total scores
                "wt_score": float(wt_score),
                "mut_score": float(mut_score),
                "ddg": float(mut_score - wt_score),
                # Per-term deltas (whole structure)
                **delta_terms,
                # Per-residue at mutation site (WT)
                "wt_site_total": wt_residue['site_total'],
                "wt_neighbor_total": wt_residue['neighbor_total'],
                # Per-residue at mutation site (mutant)
                "mut_site_total": mut_residue['site_total'],
                "mut_neighbor_total": mut_residue['neighbor_total'],
                "mut_n_neighbors": mut_residue['n_neighbors'],
                # Site-level deltas for key terms
                "site_ddg": mut_residue['site_total'] - wt_residue['site_total'],
            }

            # Add site-level per-term deltas
            for term in SCORE_TERMS:
                wt_val = wt_residue.get(f'site_{term}', 0.0)
                mut_val = mut_residue.get(f'site_{term}', 0.0)
                result[f'site_d_{term}'] = mut_val - wt_val

            results.append(result)

        except Exception:
            pass

        if len(results) >= n_muts:
            break
    if len(results) >= n_muts:
        break

elapsed = time.time() - t0

with open(output_file, "w") as f:
    json.dump(results, f)

if results:
    ddgs = [r["ddg"] for r in results]
    n_terms = len([k for k in results[0].keys() if k.startswith('d_') or k.startswith('site_d_')])
    print(f"{pdb_id}: {len(results)} mutations, {n_terms} energy terms each, "
          f"ddg=[{min(ddgs):.1f},{max(ddgs):.1f}], {elapsed:.0f}s")
else:
    print(f"{pdb_id}: 0 mutations in {elapsed:.0f}s")
