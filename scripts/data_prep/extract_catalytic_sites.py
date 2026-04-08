"""Extract catalytic site information and create YAML constraint files.

For each downloaded PDB, uses hardcoded catalytic residue definitions
(sourced from M-CSA database) to extract atom positions and generate
YAML constraint files compatible with load_constraint_from_yaml().

Usage:
    conda run -n quris python scripts/data_prep/extract_catalytic_sites.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Catalytic residues from M-CSA (https://www.ebi.ac.uk/thornton-srv/m-csa/)
#
# Format: PDB_ID -> {
#   "fold_family": str,
#   "ec_number": str,
#   "chain": str,
#   "residues": [
#       (resname_3letter, chain_id, residue_seq_number, role, [key_atoms])
#   ]
# }
#
# Key atoms are the functionally important sidechain atoms to extract.
# Backbone atoms (N, CA, C, O) are always extracted.
# ---------------------------------------------------------------------------
CATALYTIC_SITES: Dict[str, dict] = {
    # === GFP (htFuncLib) - chromophore-forming residues ===
    "2B3P": {
        "fold_family": "beta_barrel",
        "ec_number": "N/A",
        "chain": "A",
        "residues": [
            ("SER", "A", 65, "nucleophile", ["OG"]),
            ("TYR", "A", 66, "chromophore", ["OH"]),
            ("GLY", "A", 67, "chromophore", []),
            ("GLU", "A", 222, "general_base", ["OE1", "OE2"]),
        ],
    },
    # === TIM barrel enzymes ===
    "1TIM": {
        "fold_family": "TIM_barrel",
        "ec_number": "5.3.1.1",
        "chain": "A",
        "residues": [
            ("LYS", "A", 13, "substrate_binding", ["NZ"]),
            ("HIS", "A", 95, "general_base", ["ND1", "NE2"]),
            ("GLU", "A", 165, "general_base", ["OE1", "OE2"]),
        ],
    },
    "7TIM": {
        "fold_family": "TIM_barrel",
        "ec_number": "5.3.1.1",
        "chain": "A",
        "residues": [
            ("LYS", "A", 12, "substrate_binding", ["NZ"]),
            ("HIS", "A", 95, "general_base", ["ND1", "NE2"]),
            ("GLU", "A", 165, "general_base", ["OE1", "OE2"]),
        ],
    },
    "1ALD": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.1.2.13",
        "chain": "A",
        "residues": [
            ("ASP", "A", 33, "general_base", ["OD1", "OD2"]),
            ("LYS", "A", 229, "nucleophile", ["NZ"]),
            ("GLU", "A", 187, "general_acid", ["OE1", "OE2"]),
        ],
    },
    "1ENO": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.2.1.11",
        "chain": "A",
        "residues": [
            ("GLU", "A", 168, "general_base", ["OE1", "OE2"]),
            ("GLU", "A", 211, "general_acid", ["OE1", "OE2"]),
            ("LYS", "A", 345, "substrate_binding", ["NZ"]),
        ],
    },
    "1PII": {
        "fold_family": "TIM_barrel",
        "ec_number": "3.1.8.1",
        "chain": "A",
        "residues": [
            ("HIS", "A", 55, "metal_ligand", ["ND1", "NE2"]),
            ("HIS", "A", 57, "metal_ligand", ["ND1", "NE2"]),
            ("ASP", "A", 301, "general_base", ["OD1", "OD2"]),
        ],
    },
    "1BTL": {
        "fold_family": "alpha_beta",
        "ec_number": "3.5.2.6",
        "chain": "A",
        "residues": [
            ("SER", "A", 70, "nucleophile", ["OG"]),
            ("LYS", "A", 73, "general_base", ["NZ"]),
            ("GLU", "A", 166, "oxyanion_hole", ["OE1", "OE2"]),
        ],
    },
    "1XNB": {
        "fold_family": "TIM_barrel",
        "ec_number": "3.2.1.8",
        "chain": "A",
        "residues": [
            ("GLU", "A", 78, "nucleophile", ["OE1", "OE2"]),
            ("GLU", "A", 172, "general_acid", ["OE1", "OE2"]),
        ],
    },
    "1TPH": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.2.1.20",
        "chain": "A",
        "residues": [
            ("LYS", "A", 87, "substrate_binding", ["NZ"]),
            ("ASP", "A", 60, "general_acid", ["OD1", "OD2"]),
        ],
    },
    "2MNR": {
        "fold_family": "TIM_barrel",
        "ec_number": "5.1.2.2",
        "chain": "A",
        "residues": [
            ("LYS", "A", 166, "general_base", ["NZ"]),
            ("ASP", "A", 195, "general_base", ["OD1", "OD2"]),
            ("GLU", "A", 317, "general_acid", ["OE1", "OE2"]),
            ("HIS", "A", 297, "general_base", ["ND1", "NE2"]),
        ],
    },
    # === Serine proteases ===
    "4CHA": {
        "fold_family": "serine_protease",
        "ec_number": "3.4.21.1",
        "chain": "A",
        "residues": [
            ("HIS", "A", 57, "general_base", ["ND1", "NE2"]),
            ("ASP", "A", 102, "general_acid", ["OD1", "OD2"]),
            ("SER", "A", 195, "nucleophile", ["OG"]),
        ],
    },
    "1S0Q": {
        "fold_family": "serine_protease",
        "ec_number": "3.4.21.4",
        "chain": "A",
        "residues": [
            ("HIS", "A", 57, "general_base", ["ND1", "NE2"]),
            ("ASP", "A", 102, "general_acid", ["OD1", "OD2"]),
            ("SER", "A", 195, "nucleophile", ["OG"]),
        ],
    },
    "1SBN": {
        "fold_family": "subtilisin",
        "ec_number": "3.4.21.62",
        "chain": "A",
        "residues": [
            ("ASP", "A", 32, "general_acid", ["OD1", "OD2"]),
            ("HIS", "A", 64, "general_base", ["ND1", "NE2"]),
            ("SER", "A", 221, "nucleophile", ["OG"]),
        ],
    },
    "1EAI": {
        "fold_family": "serine_protease",
        "ec_number": "3.4.21.36",
        "chain": "E",
        "residues": [
            ("HIS", "E", 57, "general_base", ["ND1", "NE2"]),
            ("ASP", "E", 102, "general_acid", ["OD1", "OD2"]),
            ("SER", "E", 195, "nucleophile", ["OG"]),
        ],
    },
    "3TEC": {
        "fold_family": "subtilisin",
        "ec_number": "3.4.21.66",
        "chain": "A",
        "residues": [
            ("ASP", "A", 37, "general_acid", ["OD1", "OD2"]),
            ("HIS", "A", 70, "general_base", ["ND1", "NE2"]),
            ("SER", "A", 224, "nucleophile", ["OG"]),
        ],
    },
    # === Cysteine proteases ===
    "1POP": {
        "fold_family": "papain_like",
        "ec_number": "3.4.22.2",
        "chain": "A",
        "residues": [
            ("CYS", "A", 25, "nucleophile", ["SG"]),
            ("HIS", "A", 159, "general_base", ["ND1", "NE2"]),
            ("ASN", "A", 175, "oxyanion_hole", ["OD1", "ND2"]),
        ],
    },
    "1AIM": {
        "fold_family": "caspase",
        "ec_number": "3.4.22.36",
        "chain": "A",
        "residues": [
            ("CYS", "A", 285, "nucleophile", ["SG"]),
            ("HIS", "A", 237, "general_base", ["ND1", "NE2"]),
        ],
    },
    # === Lipases / esterases ===
    "1LPB": {
        "fold_family": "alpha_beta_hydrolase",
        "ec_number": "3.1.1.3",
        "chain": "A",
        "residues": [
            ("SER", "A", 209, "nucleophile", ["OG"]),
            ("HIS", "A", 449, "general_base", ["ND1", "NE2"]),
            ("GLU", "A", 341, "general_acid", ["OE1", "OE2"]),
        ],
    },
    "1TCA": {
        "fold_family": "alpha_beta_hydrolase",
        "ec_number": "3.1.1.3",
        "chain": "A",
        "residues": [
            ("SER", "A", 105, "nucleophile", ["OG"]),
            ("HIS", "A", 224, "general_base", ["ND1", "NE2"]),
            ("ASP", "A", 187, "general_acid", ["OD1", "OD2"]),
        ],
    },
    "1QJT": {
        "fold_family": "alpha_beta_hydrolase",
        "ec_number": "3.1.1.74",
        "chain": "A",
        "residues": [
            ("SER", "A", 120, "nucleophile", ["OG"]),
            ("HIS", "A", 188, "general_base", ["ND1", "NE2"]),
            ("ASP", "A", 175, "general_acid", ["OD1", "OD2"]),
        ],
    },
    "2LIP": {
        "fold_family": "alpha_beta_hydrolase",
        "ec_number": "3.1.1.3",
        "chain": "A",
        "residues": [
            ("SER", "A", 144, "nucleophile", ["OG"]),
            ("HIS", "A", 257, "general_base", ["ND1", "NE2"]),
            ("ASP", "A", 203, "general_acid", ["OD1", "OD2"]),
        ],
    },
    # === Glycosidases ===
    "1BVV": {
        "fold_family": "TIM_barrel",
        "ec_number": "3.2.1.21",
        "chain": "A",
        "residues": [
            ("GLU", "A", 166, "nucleophile", ["OE1", "OE2"]),
            ("GLU", "A", 354, "general_acid", ["OE1", "OE2"]),
        ],
    },
    "1CEL": {
        "fold_family": "TIM_barrel",
        "ec_number": "3.2.1.4",
        "chain": "A",
        "residues": [
            ("GLU", "A", 139, "nucleophile", ["OE1", "OE2"]),
            ("GLU", "A", 228, "general_acid", ["OE1", "OE2"]),
        ],
    },
    "1HEW": {
        "fold_family": "lysozyme",
        "ec_number": "3.2.1.17",
        "chain": "A",
        "residues": [
            ("GLU", "A", 35, "general_acid", ["OE1", "OE2"]),
            ("ASP", "A", 52, "nucleophile", ["OD1", "OD2"]),
        ],
    },
    "2LZM": {
        "fold_family": "lysozyme",
        "ec_number": "3.2.1.17",
        "chain": "A",
        "residues": [
            ("GLU", "A", 11, "general_acid", ["OE1", "OE2"]),
            ("ASP", "A", 20, "nucleophile", ["OD1", "OD2"]),
        ],
    },
    "3LZM": {
        "fold_family": "lysozyme",
        "ec_number": "3.2.1.17",
        "chain": "A",
        "residues": [
            ("GLU", "A", 11, "general_acid", ["OE1", "OE2"]),
            ("ASP", "A", 20, "nucleophile", ["OD1", "OD2"]),
        ],
    },
    "1GOX": {
        "fold_family": "TIM_barrel",
        "ec_number": "1.1.3.4",
        "chain": "A",
        "residues": [
            ("HIS", "A", 516, "general_base", ["ND1", "NE2"]),
            ("HIS", "A", 559, "substrate_binding", ["ND1", "NE2"]),
        ],
    },
    # === Oxidoreductases ===
    "1LDG": {
        "fold_family": "rossmann",
        "ec_number": "1.1.1.27",
        "chain": "A",
        "residues": [
            ("HIS", "A", 195, "general_acid", ["ND1", "NE2"]),
            ("ARG", "A", 171, "substrate_binding", ["NH1", "NH2"]),
        ],
    },
    "3ADH": {
        "fold_family": "rossmann",
        "ec_number": "1.1.1.1",
        "chain": "A",
        "residues": [
            ("CYS", "A", 46, "metal_ligand", ["SG"]),
            ("HIS", "A", 67, "metal_ligand", ["ND1", "NE2"]),
            ("CYS", "A", 174, "metal_ligand", ["SG"]),
        ],
    },
    "1GRB": {
        "fold_family": "rossmann",
        "ec_number": "1.8.1.7",
        "chain": "A",
        "residues": [
            ("CYS", "A", 58, "nucleophile", ["SG"]),
            ("CYS", "A", 63, "nucleophile", ["SG"]),
            ("HIS", "A", 467, "general_acid", ["ND1", "NE2"]),
        ],
    },
    # === Kinases ===
    "2PKA": {
        "fold_family": "protein_kinase",
        "ec_number": "2.7.11.11",
        "chain": "E",
        "residues": [
            ("LYS", "E", 72, "substrate_binding", ["NZ"]),
            ("ASP", "E", 166, "general_base", ["OD1", "OD2"]),
            ("ASP", "E", 184, "metal_ligand", ["OD1", "OD2"]),
        ],
    },
    "1HCK": {
        "fold_family": "protein_kinase",
        "ec_number": "2.7.11.22",
        "chain": "A",
        "residues": [
            ("LYS", "A", 33, "substrate_binding", ["NZ"]),
            ("ASP", "A", 127, "general_base", ["OD1", "OD2"]),
            ("ASP", "A", 145, "metal_ligand", ["OD1", "OD2"]),
        ],
    },
    # === Metalloenzymes ===
    "1CA2": {
        "fold_family": "carbonic_anhydrase",
        "ec_number": "4.2.1.1",
        "chain": "A",
        "residues": [
            ("HIS", "A", 94, "metal_ligand", ["ND1", "NE2"]),
            ("HIS", "A", 96, "metal_ligand", ["ND1", "NE2"]),
            ("HIS", "A", 119, "metal_ligand", ["ND1", "NE2"]),
            ("GLU", "A", 106, "general_base", ["OE1", "OE2"]),
        ],
    },
    "1THL": {
        "fold_family": "metalloprotease",
        "ec_number": "3.4.24.27",
        "chain": "A",
        "residues": [
            ("HIS", "A", 142, "metal_ligand", ["ND1", "NE2"]),
            ("HIS", "A", 146, "metal_ligand", ["ND1", "NE2"]),
            ("GLU", "A", 166, "metal_ligand", ["OE1", "OE2"]),
            ("GLU", "A", 143, "general_base", ["OE1", "OE2"]),
        ],
    },
    "2CPP": {
        "fold_family": "cytochrome_p450",
        "ec_number": "1.14.15.1",
        "chain": "A",
        "residues": [
            ("CYS", "A", 357, "metal_ligand", ["SG"]),
            ("ASP", "A", 251, "general_acid", ["OD1", "OD2"]),
            ("THR", "A", 252, "substrate_binding", ["OG1"]),
        ],
    },
    # === Designed enzymes ===
    "2RKX": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.2.1.-",
        "chain": "A",
        "residues": [
            ("GLU", "A", 101, "general_base", ["OE1", "OE2"]),
            ("TRP", "A", 50, "substrate_binding", ["NE1"]),
        ],
    },
    "3IIO": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.2.1.-",
        "chain": "A",
        "residues": [
            ("ASP", "A", 48, "general_base", ["OD1", "OD2"]),
        ],
    },
    "4A29": {
        "fold_family": "TIM_barrel",
        "ec_number": "4.1.2.-",
        "chain": "A",
        "residues": [
            ("LYS", "A", 210, "nucleophile", ["NZ"]),
            ("TYR", "A", 51, "general_acid", ["OH"]),
        ],
    },
    # === Classic textbook enzymes ===
    "1LZA": {
        "fold_family": "lysozyme",
        "ec_number": "3.2.1.17",
        "chain": "A",
        "residues": [
            ("GLU", "A", 35, "general_acid", ["OE1", "OE2"]),
            ("ASP", "A", 53, "nucleophile", ["OD1", "OD2"]),
        ],
    },
    "3RN3": {
        "fold_family": "rnase",
        "ec_number": "3.1.27.5",
        "chain": "A",
        "residues": [
            ("HIS", "A", 12, "general_base", ["ND1", "NE2"]),
            ("HIS", "A", 119, "general_acid", ["ND1", "NE2"]),
            ("LYS", "A", 41, "oxyanion_hole", ["NZ"]),
        ],
    },
    "1ACB": {
        "fold_family": "serine_protease",
        "ec_number": "3.4.21.1",
        "chain": "E",
        "residues": [
            ("HIS", "E", 57, "general_base", ["ND1", "NE2"]),
            ("ASP", "E", 102, "general_acid", ["OD1", "OD2"]),
            ("SER", "E", 195, "nucleophile", ["OG"]),
        ],
    },
}


# Backbone atoms always extracted
BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_residue_from_structure(
    structure, chain_id: str, res_seq: int, res_name: str
):
    """Find a residue in a BioPython structure.

    Tries standard residue first, then hetero flags.
    """
    model = structure[0]
    try:
        chain = model[chain_id]
    except KeyError:
        # Fall back to first chain
        chain = list(model.get_chains())[0]
        logger.warning(
            f"Chain {chain_id} not found, using chain {chain.id}"
        )

    # Try standard residue
    for hetflag in [" ", "H_"]:
        res_id = (hetflag.strip() or " ", res_seq, " ")
        if res_id in chain:
            residue = chain[res_id]
            if residue.get_resname().strip() == res_name:
                return residue

    # Brute-force search by sequence number
    for residue in chain.get_residues():
        if residue.id[1] == res_seq:
            actual_name = residue.get_resname().strip()
            if actual_name == res_name:
                return residue
            else:
                logger.warning(
                    f"Residue {res_seq} in chain {chain_id}: "
                    f"expected {res_name}, found {actual_name}"
                )
                return residue

    return None


def extract_atom_positions(
    residue, atom_names: List[str]
) -> Dict[str, List[float]]:
    """Extract atom coordinates from a BioPython residue.

    Always extracts backbone atoms. Additionally extracts specified
    sidechain atoms.
    """
    positions = {}

    all_atoms = BACKBONE_ATOMS + atom_names
    for atom_name in all_atoms:
        if atom_name in residue:
            coord = residue[atom_name].get_vector().get_array()
            positions[atom_name] = [round(float(x), 3) for x in coord]

    return positions


def compute_pairwise_distances(
    residue_atoms: List[Dict[str, List[float]]]
) -> List[dict]:
    """Compute CA-CA distances between all pairs of catalytic residues."""
    distances = []
    n = len(residue_atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if "CA" in residue_atoms[i] and "CA" in residue_atoms[j]:
                ca_i = np.array(residue_atoms[i]["CA"])
                ca_j = np.array(residue_atoms[j]["CA"])
                dist = float(np.linalg.norm(ca_i - ca_j))
                distances.append({
                    "pair": [i, j],
                    "distance": round(dist, 2),
                })
    return distances


def create_constraint_yaml(
    pdb_id: str,
    site_info: dict,
    pdb_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Create a YAML constraint file for a single enzyme.

    Follows the format expected by load_constraint_from_yaml():
        fold_family, residues (type, role, position_index, atoms),
        pairwise_distances.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_id}: {e}")
        return None

    constraint_data = {
        "pdb_id": pdb_id,
        "fold_family": site_info["fold_family"],
        "ec_number": site_info["ec_number"],
        "residues": [],
    }

    residue_atom_positions = []

    for res_name, chain_id, res_seq, role, key_atoms in site_info["residues"]:
        residue = get_residue_from_structure(
            structure, chain_id, res_seq, res_name
        )

        if residue is None:
            logger.warning(
                f"  {pdb_id}: residue {res_name} {chain_id}:{res_seq} "
                f"not found, skipping"
            )
            continue

        atoms = extract_atom_positions(residue, key_atoms)

        if not atoms:
            logger.warning(
                f"  {pdb_id}: no atoms extracted for {res_name} "
                f"{chain_id}:{res_seq}"
            )
            continue

        residue_atom_positions.append(atoms)

        constraint_data["residues"].append({
            "type": res_name,
            "role": role,
            "position_index": res_seq,
            "atoms": atoms,
        })

    if not constraint_data["residues"]:
        logger.warning(f"  {pdb_id}: no catalytic residues found, skipping")
        return None

    # Compute pairwise distances
    pw_distances = compute_pairwise_distances(residue_atom_positions)
    if pw_distances:
        constraint_data["pairwise_distances"] = pw_distances

    # Write YAML
    output_path = output_dir / f"{pdb_id.upper()}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            constraint_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return output_path


def main():
    pdb_dir = project_root / "data" / "pdb"
    output_dir = project_root / "data" / "catalytic_sites"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_dir.exists():
        logger.error(
            f"PDB directory not found: {pdb_dir}. "
            f"Run download_pdbs.py first."
        )
        sys.exit(1)

    logger.info(
        f"Extracting catalytic sites for {len(CATALYTIC_SITES)} enzymes"
    )

    created = []
    skipped = []

    for pdb_id, site_info in CATALYTIC_SITES.items():
        pdb_path = pdb_dir / f"{pdb_id.upper()}.pdb"

        if not pdb_path.exists():
            logger.warning(f"  {pdb_id}: PDB file not found, skipping")
            skipped.append(pdb_id)
            continue

        try:
            output_path = create_constraint_yaml(
                pdb_id, site_info, pdb_path, output_dir
            )
            if output_path:
                created.append(pdb_id)
                logger.info(f"  OK: {pdb_id} -> {output_path.name}")
            else:
                skipped.append(pdb_id)
        except Exception as e:
            logger.warning(f"  FAILED: {pdb_id}: {e}")
            skipped.append(pdb_id)

    logger.info(
        f"\nDone. Created: {len(created)}, Skipped: {len(skipped)}"
    )
    if skipped:
        logger.info(f"Skipped PDBs: {', '.join(skipped)}")

    # Verify a sample YAML is loadable
    if created:
        sample = output_dir / f"{created[0].upper()}.yaml"
        logger.info(f"\nVerifying sample YAML: {sample}")
        try:
            from src.data.catalytic_constraints import load_constraint_from_yaml
            constraint = load_constraint_from_yaml(str(sample))
            logger.info(
                f"  Loaded OK: {constraint.num_residues} residues, "
                f"fold={constraint.fold_family}"
            )
        except Exception as e:
            logger.warning(f"  Verification failed: {e}")


if __name__ == "__main__":
    main()
