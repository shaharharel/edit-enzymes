"""Load PDB files into ProteinBackbone objects.

Uses BioPython for parsing. Extracts backbone atom coordinates (N, CA, C, O)
and optional sequence information.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

from src.data.protein_structure import ProteinBackbone
from src.utils.protein_constants import BACKBONE_ATOMS, AA_3TO1
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    model_id: int = 0,
) -> ProteinBackbone:
    """Load a PDB file into a ProteinBackbone.

    Args:
        pdb_path: Path to PDB file
        chain_id: Which chain to extract (None = first chain)
        model_id: Which model to use (for NMR structures)

    Returns:
        ProteinBackbone with coordinates and sequence
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[model_id]

    # Select chain
    chains = list(model.get_chains())
    if chain_id is not None:
        chain = model[chain_id]
    else:
        chain = chains[0]
        chain_id = chain.id

    # Extract backbone atoms
    coords_list = []
    sequence = []
    residue_mask = []

    for residue in chain.get_residues():
        # Skip hetero residues (water, ligands)
        if residue.id[0] != ' ':
            continue

        resname = residue.get_resname()
        if resname not in AA_3TO1:
            continue

        # Get backbone atom coordinates
        backbone_coords = np.zeros((4, 3), dtype=np.float32)
        has_all = True

        for i, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in residue:
                backbone_coords[i] = residue[atom_name].get_vector().get_array()
            else:
                has_all = False

        coords_list.append(backbone_coords)
        sequence.append(AA_3TO1.get(resname, 'X'))
        residue_mask.append(has_all)

    if not coords_list:
        raise ValueError(f"No valid residues found in {pdb_path} chain {chain_id}")

    coords = np.stack(coords_list)  # (L, 4, 3)
    seq_str = ''.join(sequence)
    mask = np.array(residue_mask, dtype=bool)

    pdb_id = Path(pdb_path).stem.upper()

    logger.info(
        f"Loaded {pdb_id} chain {chain_id}: {len(sequence)} residues, "
        f"{mask.sum()} with complete backbone"
    )

    return ProteinBackbone(
        coords=coords,
        sequence=seq_str,
        residue_mask=mask,
        chain_id=chain_id,
        pdb_id=pdb_id,
    )


def load_pdb_all_chains(
    pdb_path: str, model_id: int = 0,
) -> List[ProteinBackbone]:
    """Load all protein chains from a PDB file."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[model_id]

    backbones = []
    for chain in model.get_chains():
        try:
            bb = load_pdb(pdb_path, chain_id=chain.id, model_id=model_id)
            if bb.length >= 10:  # skip very short chains
                backbones.append(bb)
        except ValueError:
            continue

    return backbones
