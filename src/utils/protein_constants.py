"""Protein constants: amino acid vocabulary, atom types, ideal geometry."""

import numpy as np

# Standard amino acids
AA_LIST = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
]

AA_1TO3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}

AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}
AA_1_INDEX = {AA_3TO1[aa]: i for i, aa in enumerate(AA_LIST)}

NUM_AA = 20

# Backbone atom names (in order)
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
NUM_BACKBONE_ATOMS = 4

# Ideal backbone bond lengths (Angstroms)
BOND_LENGTHS = {
    ('N', 'CA'): 1.458,
    ('CA', 'C'): 1.523,
    ('C', 'N'): 1.329,  # peptide bond
    ('C', 'O'): 1.231,
}

# Ideal backbone bond angles (degrees)
BOND_ANGLES = {
    ('N', 'CA', 'C'): 111.2,
    ('CA', 'C', 'N'): 116.2,
    ('C', 'N', 'CA'): 121.7,
    ('CA', 'C', 'O'): 120.8,
}

# Van der Waals radii (Angstroms) for clash detection
VDW_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20,
}

# Typical residue-residue CA distance range
CA_CA_DISTANCE = 3.8  # Angstroms, for consecutive residues

# Catalytic residue roles
CATALYTIC_ROLES = [
    'nucleophile',
    'general_base',
    'general_acid',
    'oxyanion_hole',
    'metal_ligand',
    'substrate_binding',
]

# Common catalytic residue types
CATALYTIC_RESIDUE_TYPES = {
    'nucleophile': ['SER', 'CYS', 'THR', 'ASP', 'GLU'],
    'general_base': ['HIS', 'ASP', 'GLU', 'LYS'],
    'general_acid': ['HIS', 'ASP', 'GLU', 'TYR'],
    'oxyanion_hole': ['ASN', 'GLY', 'SER'],
    'metal_ligand': ['HIS', 'ASP', 'GLU', 'CYS'],
    'substrate_binding': AA_LIST,  # any
}

# Key sidechain atoms for catalytic residues (functional atoms)
FUNCTIONAL_ATOMS = {
    'SER': ['OG'],
    'CYS': ['SG'],
    'HIS': ['ND1', 'NE2'],
    'ASP': ['OD1', 'OD2'],
    'GLU': ['OE1', 'OE2'],
    'LYS': ['NZ'],
    'TYR': ['OH'],
    'THR': ['OG1'],
    'ASN': ['OD1', 'ND2'],
    'ARG': ['NH1', 'NH2', 'NE'],
    'TRP': ['NE1'],
}
