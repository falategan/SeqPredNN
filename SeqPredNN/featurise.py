import argparse
import gzip
import pathlib
import logging
import numpy as np
from Bio.PDB import PDBParser, Polypeptide
from constants import AMINO_ACID_INDICES
from scipy.spatial.transform import Rotation


def get_arguments():
    """Fetch command-line arguments"""
    argument_parser = argparse.ArgumentParser(
        description="derive structural features from PDB files for training a model or predicting a sequence with a "
                    "pretrained model.")
    argument_parser.add_argument(
        "chain_list",
        type=str,
        help="path for the csv file that lists the protein chains to be processed. This file should contain a "
             "long-format table of the names, PDB file paths and chain codes for each protein chain in 3 "
             "comma-seperated columns: Protein, Filename, and Chain. "
             "For example Chain A of protein 1I6W could have a row '1I6W,pdb1i6w.ent.gz,A'")
    argument_parser.add_argument(
        "pdb_directory",
        type=str,
        help="root directory containing all the pdb files, paths specified in the chain_list will be appended to this "
             "directory path.")
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
        help="Parse modified residues as the unmodified amino acid. If this option is not selected SeqPredNN will treat"
             " these as the unknown residue X."
    )
    argument_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true")

    argument_parser.add_argument(
        "-n",
        "--neighbours",
        type=int,
        default=16,
        help="Specify the number of neighbouring residues that should be included in the structural context.")

    arguments = argument_parser.parse_args()
    return arguments.chain_list, \
           pathlib.Path(arguments.output_directory), \
           pathlib.Path(arguments.pdb_directory), \
           arguments.verbose, \
           arguments.gzip, \
           arguments.mod, \
           arguments.neighbours


def skip_completed_chains(protein_id, selected_chains, output_directory):
    """Remove chains that are already present in the output folder from the list of chains to be featurised"""
    skipped_chains = 0
    new_selected_chains = []
    for chain in selected_chains:
        feature_path = output_directory / f"{protein_id}{chain}.npz"
        if feature_path.exists():
            skipped_chains += 1
            print(f"Chain {chain} in protein {protein_id} is already processed")
        else:
            new_selected_chains.append(chain)
    return new_selected_chains, skipped_chains


def get_modified_residues(pdb_file):
    """Read all modified residues from the pdb file"""
    pdb_file.seek(0)
    modified_dictionary = {}
    for line in pdb_file:
        if line.startswith("MODRES"):
            modified_residue = line[12:15].strip(" ")
            standard_residue = line[24:27].strip(" ")
            modified_dictionary[modified_residue] = standard_residue
    return modified_dictionary


class Protein:
    """Contains all protein chain objects parsed from the pdb file"""

    def __init__(self, protein_id, file_handle, include_modified):
        parser = PDBParser(QUIET=True)

        self.name = protein_id
        # parse pdb file
        self.structure = parser.get_structure(self.name, file_handle)
        if include_modified:
            self.modified_residues = get_modified_residues(file_handle)
        # process residues for each chain in protein structure
        self.chain_dictionary = {}
        for structure_chain in self.structure[0].get_chains():
            chain_object = Chain(self, structure_chain, include_modified)
            self.chain_dictionary[chain_object.id] = chain_object

        self.ca_vectors = np.array([vector
                                    for chain in self
                                    for vector in chain.ca_vectors],
                                   dtype=float)
        self.residues = np.array([residue
                                  for chain in self
                                  for residue in chain],
                                 dtype=object)
        self.phi_angles = np.array([angle
                                    for chain in self
                                    for angle in chain.phi_angles],
                                   dtype=float)
        self.psi_angles = np.array([angle
                                    for chain in self
                                    for angle in chain.psi_angles],
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


def get_neighbours(protein, chain_ca_vectors, neighbour_count):
    """
    Calculate the distance between the alpha carbons of all the residues in the chain, then determine the
    neighbour_count residues with the smallest distance ordered by ascending distance for each residue in the chain.
    """
    # get distances from all residues in chain to all residues in structure
    xyz_translations = protein.ca_vectors[None, :] - chain_ca_vectors[:, None]
    distances = np.linalg.norm(xyz_translations, axis=-1, keepdims=False)

    # sort neighbours by distance to each cur_res
    neighbour_indices = np.argsort(distances, axis=1)
    neighbouring_residues = np.take_along_axis(protein.residues[None, :], neighbour_indices, axis=1)
    phi_angles = np.take_along_axis(protein.phi_angles[None, :], neighbour_indices, axis=1)
    psi_angles = np.take_along_axis(protein.psi_angles[None, :], neighbour_indices, axis=1)
    xyz_translations = np.take_along_axis(xyz_translations, neighbour_indices[:, :, None], axis=1).squeeze()
    neighbour_ca_vectors = np.take_along_axis(protein.ca_vectors[None, :], neighbour_indices[:, :, None], axis=1)

    # truncate neighbours to keep only cur_res and the neighbour_count nearest neighbours
    if neighbouring_residues.shape[1] > neighbour_count + 1:
        neighbouring_residues = np.copy(neighbouring_residues[:, :neighbour_count + 1])
        phi_angles = np.copy(phi_angles[:, :neighbour_count + 1])
        psi_angles = np.copy(psi_angles[:, :neighbour_count + 1])
        xyz_translations = np.copy(xyz_translations[:, :neighbour_count + 1])
        neighbour_ca_vectors = np.copy(neighbour_ca_vectors[:, :neighbour_count + 1])
    elif neighbouring_residues.shape[1] < neighbour_count + 1:
        # pad torsional angles with 0 degree angles if there are fewer residues than neighbour_count
        pad_length = neighbour_count + 1 - neighbouring_residues.shape[1]
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
    return a_dot_b / b_dot_b


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


def get_translations(xyz_translation, basis_vectors, neighbour_count):
    """
    Find the positions of the alpha-carbon atoms of the neighbour_count nearest residues in the coordinate system of
    each residue in the chain
    """
    # get x component of global displacement by dot product with the unit X vector
    local_basis = basis_vectors[:, :1, :, :]
    uvw_translations = project_vectors(xyz_translation[:, 1:, None, :], local_basis,
                                       keepdims=False)
    # pad translations with the origin if there are fewer residues than neighbour_count
    if uvw_translations.shape[1] < neighbour_count:
        pad_length = neighbour_count - uvw_translations.shape[1]
        uvw_translations = np.pad(uvw_translations,
                                  pad_width=((0, 0), (0, pad_length), (0, 0)),
                                  mode="constant",
                                  constant_values=0.)
    return uvw_translations


def get_rotations(basis_vectors, neighbour_count):
    """
    Find the rotations required to transform the orientation of the coordinate system of each residue in
    the chain to the orientation of the coordinate systems of the neighbour_count nearest residues. Represent each rotation
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
    # pad quaternions with the identity rotation if there are fewer residues than neighbour_count
    if quaternions.shape[1] < neighbour_count:
        padding_array = np.tile([0., 0., 0., 1.], (quaternions.shape[0], neighbour_count, 1))
        padding_array[:, :quaternions.shape[1], :] = quaternions
        quaternions = padding_array
    return quaternions


class Chain:
    """Contains residue objects, torsional angles, translation vectors and rotational quaternions of a protein chain"""

    def __init__(self, parent, chain, include_modified):
        polypeptide_builder = Polypeptide.PPBuilder()
        self.id = chain.get_id()
        self.protein = parent
        self.peptides = polypeptide_builder.build_peptides(chain, aa_only=False)
        self.residues = [residue
                         for peptide in self.peptides for
                         residue in peptide]

        print(f"Parsing chain {self} in protein {self.protein}")

        # remove residues with missing atoms
        self.incomplete_residues = self.check_atoms()
        self.residues = [self.replace_nonstandard(residue, include_modified)
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

    def replace_nonstandard(self, residue, include_modified):
        # check if the residue is not a standard residue
        if not Polypeptide.is_aa(residue, standard=True):
            residue_name = residue.get_resname()
            if not include_modified:
                residue.resname = "X"
                print(
                    f"Non-standard amino acid {residue_name} at position {residue.id[1]}.")
                return residue
            try:
                # convert modified residue names into the corresponding standard residues
                residue.resname = self.protein.modified_residues[residue_name]
                print(
                    f"Modified amino acid {residue_name} set to {residue.get_resname()}")
            except KeyError:
                # set the residue to the unknown amino acid X
                residue.resname = "X"
                print(
                    f"Non-standard amino acid {residue_name} at position {residue.id[1]}.")
        return residue

    def featurise_chain(self,
                        output_directory: pathlib.Path,
                        neighbour_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate and save tensors encoding features for the neighbour_count nearest residues of each residue in the
        protein chain

        Args:
            output_directory: Path to directiory where features are saved
            neighbour_count: Number of nearby residues that are used to describe the structural context of each
                residue in the chain
        Returns:
            (tuple):
            residue_labels: array of integers representing the amino acid type of each residue

            uvw_translations: array of the positions of the alpha carbon of each nearby residue in the local coordinate
            system of each residue in the chain

            rotations: array of quaternions describing the rotation from the local coordinate system of each residue to
            the local coordinate system of each nearby residue

            torsional angles: the phi and psi angles of each residue and the nearby residues
        """
        neighbouring_residues, neighbour_ca_vectors, xyz_translations, torsional_angles \
            = get_neighbours(self.protein, self.ca_vectors, neighbour_count)
        residue_labels = np.array([AMINO_ACID_INDICES[residue.get_resname()]
                                   for residue in neighbouring_residues[:, 0]]).squeeze()
        # derive local coordinate system for each residue
        basis_vectors = get_basis_vectors(neighbouring_residues, neighbour_ca_vectors)
        # calculate relative positions of neighbours in the local coordinate system of each residue
        uvw_translations = get_translations(xyz_translations, basis_vectors, neighbour_count)
        # calculate quaternion rotations to the local coordinate systems of neighbouring residues
        rotations = get_rotations(basis_vectors, neighbour_count)

        if self.incomplete_residues:
            with open(output_directory / f"incomplete_residues_{self.protein}{self}.csv", "w") as file:
                for key in self.incomplete_residues:
                    file.write(f"{key}, {self.incomplete_residues[key]}\n")
        with open(output_directory / "chain_list.txt", "a") as file:
            file.write(f"{self.protein}{self}\n")
        return residue_labels, uvw_translations, rotations, torsional_angles


def read_chain_file(chain_file_path, pdb_directory):
    with open(chain_file_path, "r") as file:
        lines = [line.rstrip("\n").split(",") for line in file][1:]

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


def write_exclusion(protein, chains, output_directory):
    with open(output_directory / "excluded_chains.txt", "a") as file:
        if len(chains) > 1:
            chain_string = ", ".join(chains)
        else:
            chain_string = chains[0]
        logging.warning(f"Excluding chain {chain_string} in protein {protein} from the dataset.")
        for chain in chains:
            file.write(f"{protein}{chain}\n")


def read_pdb_file(protein_id, pdb_file_path, zipped, include_modified):
    if not pdb_file_path.exists():  # pdb file is missing
        return None
    try:
        with gzip.open(pdb_file_path, "rt") if zipped else open(pdb_file_path, "r") as file:
            protein = Protein(protein_id, file, include_modified)
    except EOFError as error:
        logging.warning(f"{pdb_file_path} - {error}")
        return None
    return protein


def save_features(protein, chain_id, output_directory, features):
    # save features to numpy array files
    residue_labels, translations, rotations, torsional_angles = features
    np.savez(output_directory / f"{protein}{chain_id}",
             residue_labels=residue_labels,
             translations=translations,
             rotations=rotations,
             torsional_angles=torsional_angles)


def featurise(chain_file: str,
              output_directory: str,
              pdb_directory: str,
              verbose: bool,
              zipped: bool,
              include_modified: bool,
              neighbour_count: int) -> None:
    """
    Derive structural features from PDB files for training models or predicting a sequence with a
    pretrained model. Structural features for each protein chain are saved to a zipped numpy file (.npz).

    Args:
        chain_file: Path to a file with a comma-delimited table of protein names, the name or path to the pdb file for
        that protein, and the chain ID
        output_directory: Path of the directory where the structural features should be saved
        verbose: Print more information to stdout
        zipped: Unzip the PDB files using gzip
        include_modified: Convert modified amino acids to their respective unmodified versions. Modified residues are
        converted to the unknown residue X, and excluded from the training set if include_modified == False
    """

    # get command-line arguments
    chain_file = pathlib.Path(chain_file)
    output_directory = pathlib.Path(output_directory)
    pdb_directory = pathlib.Path(pdb_directory)

    if verbose:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    logging.basicConfig(level=logging_level, format="%(message)s")
    logging.captureWarnings(False)

    # create output directory
    if not output_directory.exists():
        output_directory.mkdir()

    # read chain file
    print("Reading chain file...")
    protein_dictionary = read_chain_file(chain_file, pdb_directory)
    remaining_chains = sum(len(chain_ids) for _, chain_ids in protein_dictionary.values())
    print("Done.")

    # featurise proteins
    chain_idx = 1
    for protein_id, (pdb_file_path, chain_ids) in protein_dictionary.items():
        # remove chains that have already been featurised
        selected_chains, skipped_chains = skip_completed_chains(protein_id, chain_ids, output_directory)
        remaining_chains -= skipped_chains
        if not selected_chains:
            # no chains selected
            continue
        protein = read_pdb_file(protein_id, pdb_file_path, zipped, include_modified)
        if not protein:
            # pdb file not found
            logging.warning(f"Could not find the file {pdb_file_path}.")
            write_exclusion(protein_id, selected_chains, output_directory)
            remaining_chains -= 1
            continue

        # generate structural features for each chain in chain list file
        for chain_id in selected_chains:
            print(f"Chain {chain_idx}/{remaining_chains}")
            try:
                features = protein[chain_id].featurise_chain(output_directory, neighbour_count)
                save_features(protein, chain_id, output_directory, features)
                chain_idx += 1
            except KeyError:
                logging.error(f"No chain {chain_id} in protein {protein}.")
                write_exclusion(protein, [chain_id, ], output_directory)
                remaining_chains -= 1
            except ValueError as value_error:
                logging.error(f"Value Error: {value_error}")
                write_exclusion(protein, [chain_id, ], output_directory)
                remaining_chains -= 1
    print(f"Done.")


if __name__ == "__main__":
    arguments = get_arguments()
    featurise(*arguments)
