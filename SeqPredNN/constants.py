AMINO_ACID_INDICES = {
    "HIS": 0,
    "ARG": 1,
    "LYS": 2,
    "GLN": 3,
    "GLU": 4,
    "ASP": 5,
    "ASN": 6,
    "GLY": 7,
    "ALA": 8,
    "SER": 9,
    "THR": 10,
    "PRO": 11,
    "CYS": 12,
    "VAL": 13,
    "ILE": 14,
    "MET": 15,
    "LEU": 16,
    "PHE": 17,
    "TYR": 18,
    "TRP": 19,
    "X": 20}

STANDARD_AMINO_ACIDS = list(AMINO_ACID_INDICES.keys())[:-1]

AMINO_ACID_LETTERS = {
    "HIS": "H",
    "ARG": "R",
    "LYS": "K",
    "GLN": "Q",
    "GLU": "E",
    "ASP": "D",
    "ASN": "N",
    "GLY": "G",
    "ALA": "A",
    "SER": "S",
    "THR": "T",
    "PRO": "P",
    "CYS": "C",
    "VAL": "V",
    "ILE": "I",
    "MET": "M",
    "LEU": "L",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "X": "X"}

FEATURE_LIST = ["residue_labels", "translations", "rotations", "torsional_angles"]
