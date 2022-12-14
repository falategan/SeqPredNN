import argparse
import gzip
import logging
import pathlib
import time
from collections import defaultdict
import numpy as np
from Bio.PDB import PDBParser, Polypeptide
from scipy.spatial.transform import Rotation

feature_list = ["residue_labels", "translations", "rotations", "torsional_angles"]
amino_acid_list = ["GLY", "ALA", "CYS", "PRO", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "SER", "THR", "ASN", "GLN",
                   "TYR", "ASP", "GLU", "HIS", "LYS", "ARG", "X"]
aa_dict = {amino_acid: float(i) for i, amino_acid in enumerate(amino_acid_list)}
N_NEIGHBOURS = 16
parser = PDBParser(QUIET=True)
polypeptide_builder = Polypeptide.PPBuilder()


def get_arguments():
    """Fetch command-line arguments"""
    argument_parser = argparse.ArgumentParser(
        description="derive structural features from PDB files for training a model or predicting a sequence with a "
                    "pretrained model.")
    argument_parser.add_argument(
        "chain_list",
        type=str,
        help="path for the list of protein chains to be processed")
    argument_parser.add_argument(
        "pdb_directory",
        type=str,
        help="directory of pdb files. Subdirectories and gzipped pdb files must be named according to the PDB archive "
             "convention e.g. pdb_dir/xy/pdb1xyz.ent.gz")
    argument_parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="./features",
        help="output directory. Will create a new directory if OUT_DIR does not exist.")
    argument_parser.add_argument(
        "-g",
        "--gzip",
        action="store_true",
        help="unzip gzipped pdb files"
    )
    argument_parser.add_argument(
        "-m",
        "--mod",
        action="store_true",
        help="Parse modified residues as the unmodified amino acid. If this option is not selected SeqPredNN will treat "
             "these as the unknown residue X."
    )
    argument_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true")

    arguments = argument_parser.parse_args()
    return arguments.chain_list, pathlib.Path(arguments.output_directory), pathlib.Path(arguments.pdb_directory), \
           arguments.verbose, arguments.gzip, arguments.mod


def skip_completed_chains(protein_id, selected_chains):
    """Remove chains that are already present in the output folder from the list of chains to be featurised"""
    skipped_chains = 0
    new_selected_chains = []
    for chain in selected_chains:
        feature_path = output_directory / f"{protein_id}{chain}.npz"
        if feature_path.exists():
            skipped_chains += 1
            logging.info(f"Chain {chain} in protein {protein_id} is already processed")
        else:
            new_selected_chains.append(chain)
    return new_selected_chains, skipped_chains


def get_modified_residues(pdb_file):
    pdb_file.seek(0)
    modified_dictionary = {}
    for line in pdb_file:
        if line.startswith("MODRES"):
            modified_residue = line[12:15].strip(" ")
            standard_residue = line[24:27].strip(" ")
            modified_dictionary[modified_residue] = standard_residue
    return modified_dictionary


class Protein:
    def __init__(self, protein_id, file_handle):
        self.name = protein_id
        # parse pdb file
        self.structure = parser.get_structure(self.name, file_handle)
        if include_modified:
            self.modified_residues = get_modified_residues(file_handle)
        # process residues for each chain in protein structure
        self.chain_dictionary = {}
        for structure_chain in self.structure[0].get_chains():
            chain_object = Chain(self, structure_chain)
            self.chain_dictionary[chain_object.id] = chain_object

        self.ca_vectors = np.concatenate([chain.ca_vectors
                                          for chain in self],
                                         dtype=float)
        self.residues = np.array([residue
                                  for chain in self
                                  for residue in chain],
                                 dtype=object)
        self.phi_angles = np.concatenate([chain.phi_angles
                                          for chain in self],
                                         dtype=float)
        self.psi_angles = np.concatenate([chain.psi_angles
                                          for chain in self],
                                         dtype=float)

    def __len__(self):
        return len(self.chain_dictionary)

    def __contains__(self, item):
        return item in self.chain_dictionary

    def __iter__(self):
        return iter(self.chain_dictionary.values())

    def __getitem__(self, item):
        return self.chain_dictionary[item]

    def __str__(self):
        return self.name


def get_neighbours(protein, chain_ca_vectors):
    """
    Calculate the distance between the alpha carbons of all the residues in the chain, then determine the N_NEIGHBOURS
    residues with the smallest distance ordered by ascending distance for each residue in the chain.
    """
    start_distances = time.perf_counter()
    # get distances from all residues in chain to all residues in structure
    xyz_translations = protein.ca_vectors[None, :] - chain_ca_vectors[:, None]
    distances = np.linalg.norm(xyz_translations, axis=-1, keepdims=False)
    logging.debug(f"Distance time: {time.perf_counter() - start_distances}")

    # sort neighbours by distance to each cur_res
    start_sort = time.perf_counter()
    neighbour_indices = np.argsort(distances, axis=1)
    neighbouring_residues = np.take_along_axis(protein.residues[None, :], neighbour_indices, axis=1)
    phi_angles = np.take_along_axis(protein.phi_angles[None, :], neighbour_indices, axis=1)
    psi_angles = np.take_along_axis(protein.psi_angles[None, :], neighbour_indices, axis=1)
    xyz_translations = np.take_along_axis(xyz_translations, neighbour_indices[:, :, None], axis=1).squeeze()
    neighbour_ca_vectors = np.take_along_axis(protein.ca_vectors[None, :], neighbour_indices[:, :, None], axis=1)
    logging.debug(f"Sort time: {time.perf_counter() - start_sort}")

    # truncate neighbours to keep only cur_res and the N_NEIGHBOURS nearest neighbours
    if neighbouring_residues.shape[1] > N_NEIGHBOURS + 1:
        neighbouring_residues = np.copy(neighbouring_residues[:, :N_NEIGHBOURS + 1])
        phi_angles = np.copy(phi_angles[:, :N_NEIGHBOURS + 1])
        psi_angles = np.copy(psi_angles[:, :N_NEIGHBOURS + 1])
        xyz_translations = np.copy(xyz_translations[:, :N_NEIGHBOURS + 1])
        neighbour_ca_vectors = np.copy(neighbour_ca_vectors[:, :N_NEIGHBOURS + 1])
    elif neighbouring_residues.shape[1] < N_NEIGHBOURS + 1:
        # pad torsional angles with 0 degree angles if there are fewer residues than N_NEIGHBOURS
        pad_length = N_NEIGHBOURS + 1 - neighbouring_residues.shape[1]
        phi_angles = np.pad(phi_angles,
                            pad_width=((0, 0), (0, pad_length),),
                            mode="constant",
                            constant_values=0.)
        psi_angles = np.pad(psi_angles,
                            pad_width=((0, 0), (0, pad_length),),
                            mode="constant",
                            constant_values=0.)
    # encode torsional angles as (sin(angle), cos(angle))
    phi_angles = np.stack([np.sin(phi_angles), np.cos(phi_angles)], axis=2)
    psi_angles = np.stack([np.sin(psi_angles), np.cos(psi_angles)], axis=2)
    torsional_angles = np.stack([phi_angles, psi_angles], axis=3)
    return neighbouring_residues, neighbour_ca_vectors, xyz_translations, torsional_angles


def normalise_vectors(vector_array):
    return vector_array / np.linalg.norm(vector_array, axis=-1, keepdims=True)


def dot_product(array_a, array_b, keepdims=True):
    return np.sum(array_a * array_b, axis=-1, keepdims=keepdims)


def project_vectors(array_a, array_b, keepdims=True):
    a_dot_b = dot_product(array_a, array_b, keepdims=keepdims)
    b_dot_b = dot_product(array_b, array_b, keepdims=keepdims)
    return (a_dot_b / b_dot_b) * array_b


def get_basis_vectors(neighbours, ca_vectors):
    n_vectors = np.array([[neighbour["N"].get_coord() for neighbour in position]
                          for position in neighbours],
                         dtype=float).squeeze()
    c_vectors = np.array([[neighbour["C"].get_coord() for neighbour in position]
                          for position in neighbours],
                         dtype=float).squeeze()
    u = normalise_vectors(c_vectors - n_vectors)
    n_to_ca = normalise_vectors(ca_vectors - n_vectors)
    v = normalise_vectors(n_to_ca - project_vectors(n_to_ca, u))
    w = np.cross(u, v)
    basis_vectors = np.stack([u, v, w], axis=2)
    return basis_vectors


def get_translations(xyz_translation, basis_vectors):
    """
    Find the positions of the alpha-carbon atoms of the N_NEIGHBOURS nearest residues in the coordinate system of
    each residue in the chain
    """
    # get x component of global displacement by dot product with the unit X vector
    local_basis = basis_vectors[:, :1, :]
    uvw_translations = project_vectors(xyz_translation[:, :, None, :], local_basis)

    # pad translations with the origin if there are fewer residues than N_NEIGHBOURS
    if uvw_translations.shape[1] < N_NEIGHBOURS:
        pad_length = N_NEIGHBOURS - uvw_translations.shape[1]
        uvw_translations = np.pad(uvw_translations,
                                  pad_width=((0, 0), (0, pad_length), (0, 0)),
                                  mode="constant",
                                  constant_values=0.)
    return uvw_translations


def get_rotations(basis_vectors):
    """
    Find the rotations required to transform the orientation of the coordinate system of each residue in
    the chain to the orientation of the coordinate systems of the N_NEIGHBOURS nearest residues. Represent each rotation
    as a unit quaternion.
    """
    local_basis = basis_vectors[:, :1, :, :]
    neighbour_bases = basis_vectors[:, 1:, :, :]
    neighbour_bases_transpose = neighbour_bases.transpose((0, 1, 3, 2))
    rotation_matrices = neighbour_bases_transpose @ local_basis
    rotation_matrices = [Rotation.from_matrix(neighbour_matrices)
                         for neighbour_matrices in rotation_matrices]
    quaternions = np.array([matrix.as_quat()
                            for matrix in rotation_matrices]).squeeze()
    # pad quaternions with the identity rotation if there are fewer residues than N_NEIGHBOURS
    if quaternions.shape[1] < N_NEIGHBOURS:
        padding_array = np.tile([0., 0., 0., 1.], (quaternions.shape[0], N_NEIGHBOURS, 1))
        padding_array[:, :quaternions.shape[1], :] = quaternions
        quaternions = padding_array
    return quaternions


class Chain:
    def __init__(self, parent, chain):
        self.id = chain.get_id()
        self.protein = parent
        self.peptides = polypeptide_builder.build_peptides(chain, aa_only=False)
        self.residues = [residue
                         for peptide in self.peptides for
                         residue in peptide]

        logging.info(f"Parsing chain {self} in protein {self.protein}")

        # remove residues with missing atoms
        self.incomplete_residues = self.check_atoms()
        self.residues = [self.replace_nonstandard(residue)
                         for i, residue in enumerate(self)
                         if i not in self.incomplete_residues]
        self.ca_vectors = np.array([residue["CA"].get_coord()
                                    for residue in self]).squeeze()
        # get torsional angles
        self.torsional_angles = [angles
                                 for peptide in self.peptides
                                 for angles in peptide.get_phi_psi_list()]
        self.phi_angles = [angles[0] if angles[0] is not None else 0.
                           for angles in self.torsional_angles]
        self.psi_angles = [angles[1] if angles[1] is not None else 0.
                           for angles in self.torsional_angles]
        self.phi_angles = np.array([self.phi_angles[i]
                                    for i, angle in enumerate(self.phi_angles)
                                    if i not in self.incomplete_residues]).squeeze()
        self.psi_angles = np.array([self.psi_angles[i]
                                    for i, angle in enumerate(self.psi_angles)
                                    if i not in self.incomplete_residues]).squeeze()

    def __iter__(self):
        return iter(self.residues)

    def __getitem__(self, item):
        return self.residues[item]

    def __str__(self):
        return self.id

    def check_atoms(self):
        """Identify residues with missing backbone atoms and non-standard amino acids"""
        # find missing backbone atoms
        incomplete_residues = {i: residue.get_resname()
                               for i, residue in enumerate(self.residues)
                               if any(atom not in residue
                                      for atom in ("N", "C", "CA"))}
        return incomplete_residues

    def replace_nonstandard(self, residue):
        nonstandard_residues = {}
        # check if the residue is not a standard residue
        if not Polypeptide.is_aa(residue, standard=True):
            residue_name = residue.get_resname()
            try:
                # convert modified residue names into the corresponding standard residues
                residue.resname = self.protein.modified_residues[residue_name]
                logging.info(
                    f"Modified amino acid {residue_name} set to {residue.get_resname()}")
            except KeyError:
                # set the residue to the unknown amino acid X
                residue.resname = "X"
                logging.info(
                    f"Non-standard amino acid {residue_name} at position {residue.id[1]}.")
        return residue



    def featurise_chain(self):
        """
        Generate and save tensors encoding features for the N_NEIGBOURS nearest residues of each residue in the
        protein chain.
        The features are:
            uvw_translations - the positions of the alpha carbon of each nearby residue in the local coordinate system
                               of each residue in the chain
            rotations - a quaternion describing the rotation from the local coordinate system of each residue to the
                        local coordinate system of each nearby residue
            residue_labels - the amino acid type of the nearby residues
            torsional angles - the phi and psi angles of each residue and the nearby residues
        """
        start_feat = time.perf_counter()
        start_neighbours = time.perf_counter()

        neighbouring_residues, neighbour_ca_vectors, xyz_translations, torsional_angles \
            = get_neighbours(self.protein, self.ca_vectors)
        residue_labels = np.array([aa_dict[residue.get_resname()]
                                   for residue in neighbouring_residues[:, 0]]).squeeze()
        logging.debug(f"Neighbour time: {time.perf_counter() - start_neighbours}")
        # derive local coordinate system for each residue
        start_basis = time.perf_counter()
        basis_vectors = get_basis_vectors(neighbouring_residues, neighbour_ca_vectors)
        logging.debug(f"Basis time: {time.perf_counter() - start_basis}")
        # calculate relative positions of neighbours in the local coordinate system of each residue
        start_translations = time.perf_counter()
        uvw_translations = get_translations(xyz_translations, basis_vectors)
        logging.debug(f"Translation time: {time.perf_counter() - start_translations}")
        # calculate quaternion rotations to the local coordinate systems of neighbouring residues
        start_rotations = time.perf_counter()
        rotations = get_rotations(basis_vectors)
        logging.debug(f"Rotation time: {time.perf_counter() - start_rotations}")
        logging.debug(f"Featurisation time: {time.perf_counter() - start_feat}")
        if self.incomplete_residues:
            with open(output_directory / f"incomplete_residues_{self.protein}{self}.csv", "w") as file:
                for key in self.incomplete_residues:
                    file.write(f"{key}, {self.incomplete_residues[key]}\n")
        with open(output_directory / "chain_list.csv", "a") as file:
            file.write(f"{self.protein}{self},{len(residue_labels)}\n")
        return residue_labels, uvw_translations, rotations, torsional_angles


def read_chain_file(chain_file_path, pdb_directory):
    with open(chain_file_path, "r") as file:
        lines = [line.strip("\n").split(",") for line in file][1:]

        protein_dictionary = {}
        for line in lines:
            protein_name, pdb_file_path, chain_id = line
            pdb_file_path = pdb_directory / pdb_file_path
            try:
                _, chain_id_list = protein_dictionary[protein_name]
                if chain_id not in chain_id_list:
                    protein_dictionary[protein_name] = (pdb_file_path, chain_id_list + [chain_id])
            except KeyError:
                protein_dictionary[protein_name] = (pdb_file_path, [chain_id])
    return protein_dictionary


def write_exclusion(protein, chains):
    with open(output_directory / "excluded_chains.txt", "a") as file:
        if len(chains) > 1:
            chain_string = ", ".join(chains)
        else:
            chain_string = chains[0]
        logging.warning(f"Excluding chain {chain_string} in protein {protein} from the dataset.")
        for chain in chains:
            file.write(f"{protein}{chain}\n")


def read_pdb_file(protein_id, pdb_file_path):
    if not pdb_file_path.exists():  # pdb file is missing
        return None
    with gzip.open(pdb_file_path, "rt") if zipped else open(pdb_file_path, "r") as file:
        protein = Protein(protein_id, file)
    return protein


def save_features(protein, chain_id, output_directory, features):
    # save features to numpy array files
    start_save = time.perf_counter()
    residue_labels, translations, rotations, torsional_angles = features
    logging.debug(f"residue_labels: {residue_labels.shape}")
    logging.debug(f"translations: {translations.shape}")
    logging.debug(f"rotations: {rotations.shape}")
    logging.debug(f"torsional_angles: {torsional_angles.shape}")
    np.savez(output_directory / f"{protein}{chain_id}",
             residue_labels=residue_labels,
             translations=translations,
             rotations=rotations,
             torsional_angles=torsional_angles)
    logging.debug(f"Save time: {time.perf_counter() - start_save}")


if __name__ == "__main__":
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

    # get command-line arguments
    chain_file, output_directory, pdb_directory, verbose, zipped, include_modified = get_arguments()

    if verbose:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format="%(levelname)s - %(message)s")
    logging.captureWarnings(False)

    # create output directory
    if not output_directory.exists():
        output_directory.mkdir()

    # read chain file
    logging.info("Reading chain file...")
    protein_dictionary = read_chain_file(chain_file, pdb_directory)
    remaining_chains = sum(len(chain_ids) for _, chain_ids in protein_dictionary.values())
    logging.info("Done.")

    # featurise proteins
    chain_idx = 1
    protein_time = 0
    for protein_id, (pdb_file_path, chain_ids) in protein_dictionary.items():
        # remove chains that have already been featurised
        selected_chains, skipped_chains = skip_completed_chains(protein_id, chain_ids)
        remaining_chains -= skipped_chains
        if not selected_chains:
            # no chains selected
            continue
        protein = read_pdb_file(protein_id, pdb_file_path)
        if not protein:
            # pdb file not found
            logging.warning(f"Could not find the file {pdb_file_path}.")
            logging.debug(f"selected_chains: {selected_chains}")
            write_exclusion(protein_id, selected_chains)  # RECONSIDER THIS FUNCTION
            continue

        # generate structural features for each chain in chain list file
        for chain_id in selected_chains:
            logging.info(f"Chain {chain_idx}/{remaining_chains}")
            try:
                features = protein[chain_id].featurise_chain()
                save_features(protein, chain_id, output_directory, features)
                chain_idx += 1
            except KeyError:
                logging.error(f"No chain {chain_id} in protein {protein}.")
                write_exclusion(protein, [chain_id, ])
                remaining_chains -= 1
    logging.info(f"Done.")
