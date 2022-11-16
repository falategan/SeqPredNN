from Bio.PDB import *
import numpy as np
from scipy.spatial.transform import Rotation
import pathlib
import argparse
import gzip
from collections import defaultdict
import warnings
import torch
# import matplotlib.pyplot as plt
import time

aa_dict = {'GLY': 0., 'ALA': 1., 'CYS': 2., 'PRO': 3., 'VAL': 4., 'ILE': 5., 'LEU': 6., 'MET': 7.,
           'PHE': 8., 'TRP': 9., 'SER': 10., 'THR': 11., 'ASN': 12., 'GLN': 13., 'TYR': 14.,
           'ASP': 15., 'GLU': 16., 'HIS': 17., 'LYS': 18., 'ARG': 19., 'X': 20.}
N_NEIGHBOURS = 16


def get_args():
    arg_parser = argparse.ArgumentParser(description="derive structural features from PDB files for training a model "
                                                     "or predicting a sequence with a pretrained model.")
    arg_parser.add_argument('chain_list', type=str, help="path for the list of protein chains to be processed")
    arg_parser.add_argument('pdb_dir', type=str, help="directory of pdb files. Subdirectories and gzipped pdb files "
                                                      "must be named according to the PDB archive convention "
                                                      "e.g. pdb_dir/xy/pdb1xyz.ent.gz")
    arg_parser.add_argument('pdb_layout', type=str, choices=['all', 'divided'],
                            help="pdb directory structure. 'divided' requires gzipped pdb files in subdirectories named"
                                 " according to the PDB archive divided convention e.g. pdb_dir/xy/pdb1xyz.ent.gz for "
                                 "protein 1XYZ. 'all' requires all gzipped pdb files in a single directory e.g."
                                 " pdb_dir/pdb1xyz.ent.gz")
    arg_parser.add_argument('-o', '--out_dir', type=str, default='./features',
                            help="output directory. Will create a new directory if OUT_DIR does not exist.")
    arg_parser.add_argument('-v', '--verbose', action='store_true')
    args = arg_parser.parse_args()
    return args.chain_list, pathlib.Path(args.out_dir), pathlib.Path(args.pdb_dir), args.verbose, args.pdb_layout


def read_chain_file(path):
    with open(path, 'r') as file:
        codes = [line.split(' ')[0].strip('\n') for line in file]
        if codes[0] == 'PDBchain':
            codes = codes[1:]
    return codes


def write_exclusion(protein_id, chains):
    with open(out_dir / 'excluded_chains.txt', 'a') as file:
        if verbose:
            print('Excluding', protein_id, chains, 'from dataset.')
        if len(chains) > 1:
            excluded_chains = [protein_id + chain + '\n' for chain in chains]
        else:
            excluded_chains = protein_id + chains[0] + '\n'
        file.writelines(excluded_chains)


def featurise_protein(protein_id, selected_chains):
    skipped_chains = 0
    parsed_chains = 0
    completed_chains = []
    for chain in selected_chains:
        chain_paths = [out_dir / (feature + protein_id + chain + '.pt') for feature in
                       ['displacements_', 'residue_labels_', 'rotations_',
                        'torsional_angles_']]
        # ignore chains that are already present in the output folder
        if all(file.exists() for file in chain_paths):
            print('Chain', protein_id, chain, 'already processed')
            completed_chains.append(chain)
            skipped_chains += 1
    selected_chains = [chain for chain in selected_chains if chain not in completed_chains]
    if selected_chains:
        # fetch pdb file
        file_name = 'pdb' + protein_id.lower() + '.ent.gz'
        if pdb_layout == 'divided':
            protein_path = pdb_dir / protein_id.lower()[1:3] / file_name
        elif pdb_layout == 'all':
            protein_path = pdb_dir / file_name
        if protein_path.exists():
            # unzip pdb file
            with gzip.open(protein_path, 'rt') as gz_file:
                try:
                    # featurise protein
                    protein = Protein(protein_id, gz_file, selected_chains)
                    if protein.excluded_chains:
                        write_exclusion(protein_id, protein.excluded_chains)
                    parsed_chains += len(protein.selected_chains) - len(protein.excluded_chains)
                    skipped_chains += len(protein.excluded_chains)
                except Exception as error:
                    # Exception encountered during featurisation -- protein code is written in excluded chain file
                    warning = str(str(type(error)) + ': ' + str(error) + ' in protein ' + protein_id)
                    warnings.warn(warning, UserWarning)
                    write_exclusion(protein_id, selected_chains)
                    skipped_chains += len(selected_chains)
        else:
            # missing pdb file
            print(protein_path, 'does not exist.')
            write_exclusion(protein_id, selected_chains)
            skipped_chains += len(selected_chains)
            return parsed_chains, skipped_chains
    return parsed_chains, skipped_chains


class Protein:
    def __init__(self, protein_id, file_handle, selected_chain_ids):
        parser = PDBParser(QUIET=True)
        self.name = protein_id
        # parse pdb file
        self.structure = parser.get_structure(self.name, file_handle)
        # process residues for each chain in protein structure
        self.chains = [Chain(self, chain) for chain in self.structure[0].get_chains()]
        self.selected_chains = [chain for chain in self.chains if chain.id in selected_chain_ids]
        self.excluded_chains = []
        # generate structural features for each chain in chain list file
        for chain in self.selected_chains:
            try:
                neighbours, torsional_angles = chain.get_neighbours(self.chains)
                residue_labels = np.array([aa_dict[res.get_resname()] for res in neighbours[:, 0]])
                # derive local coordinate system for each residue
                ca_vectors, u, v, w = get_basis_vectors(neighbours)
                # calculate relative positions of neighbours in the local coordinate system of each residue
                translations = get_translations(ca_vectors, u, v, w)
                # calculate quaternion rotations to the local coordinate systems of neighbouring residues
                rotations = get_rotations(u, v, w)
                # save features to numpy array files
                torch.save(translations, out_dir / ('translations_' + self.name + chain.id + '.pt'))
                torch.save(residue_labels, out_dir / ('residue_labels_' + self.name + chain.id + '.pt'))
                torch.save(rotations, out_dir / ('rotations_' + self.name + chain.id + '.pt'))
                torch.save(torsional_angles, out_dir / ('torsional_angles_' + self.name + chain.id + '.pt'))
                if chain.ignored_residues:
                    with open(out_dir / ('excluded_residues_' + self.name + chain.id + '.csv'), 'w') as file:
                        for key in chain.ignored_residues:
                            file.write(str(key) + ',' + str(chain.ignored_residues[key]) + '\n')
                with open(out_dir / 'chain_list.csv', 'a') as file:
                    file.write(self.name + chain.id + ',' + str(len(residue_labels)) + '\n')
            except Exception as error:
                warning = str(str(type(error)) + ': ' + str(error) + ' in chain ' + self.name + chain.id)
                warnings.warn(warning, UserWarning)
                self.excluded_chains.append(chain.id)


def normalise_vectors(vector_array):
    return vector_array / np.linalg.norm(vector_array, axis=-1, keepdims=True)


def dot_product(array_a, array_b, keepdims=True):
    return np.sum(array_a * array_b, axis=-1, keepdims=keepdims)


def project_vectors(array_a, array_b, keepdims=True):
    a_dot_b = dot_product(array_a, array_b, keepdims=keepdims)
    b_dot_b = dot_product(array_b, array_b, keepdims=keepdims)
    return (a_dot_b / b_dot_b) * array_b


def get_basis_vectors(neighbours):
    ca_coords = np.array([[neighbour['CA'].get_coord() for neighbour in position]
                          for position in neighbours], dtype=float)
    n_coords = np.array([[neighbour['N'].get_coord() for neighbour in position]
                         for position in neighbours], dtype=float)
    c_coords = np.array([[neighbour['C'].get_coord() for neighbour in position]
                         for position in neighbours], dtype=float)

    u = normalise_vectors(c_coords - n_coords)
    n_to_ca = normalise_vectors(ca_coords - n_coords)
    v = normalise_vectors(n_to_ca - project_vectors(n_to_ca, u))
    w = np.cross(u, v)

    return ca_coords, u, v, w


def get_translations(ca_vectors, u, v, w):
    xyz_translation = ca_vectors[:, 1:] - ca_vectors[:, :1]

    # get x component of global displacement by dot product with the unit X vector
    u_translation = project_vectors(xyz_translation, u[:, :1])
    v_translation = project_vectors(xyz_translation, v[:, :1])
    w_translation = project_vectors(xyz_translation, w[:, :1])
    uvw_translations = np.stack([u_translation, v_translation, w_translation], axis=-1)

    '''
    # VISUALISE DISTANCES AS NORM OF DISPLACEMENT
    norm = np.vectorize(lambda disp: disp.norm())(global_displacement)
    fig, ax = plt.subplots(1, 2, figsize=(12, 9))
    im = ax[0].imshow(norm)
    norm = np.linalg.norm(displacements, axis=-1)
    im = ax[1].imshow(norm)
    plt.show()
    '''

    # pad displacements
    if uvw_translations.shape[1] < N_NEIGHBOURS:
        uvw_translations = np.pad(uvw_translations,
                                  pad_width=((0, 0), (0, N_NEIGHBOURS - uvw_translations.shape[1]), (0, 0)),
                                  mode='constant', constant_values=0.)
    return uvw_translations


def get_rotations(u, v, w):
    local_basis = np.stack([u[:, :1], v[:, :1], w[:, :1]], axis=2)
    neighbour_bases = np.stack([u[:, 1:], v[:, 1:], w[:, 1:]], axis=2)
    neighbour_bases_transpose = neighbour_bases.transpose((0, 1, 3, 2))
    rotation_matrices = neighbour_bases_transpose @ local_basis
    rotation_matrices = [Rotation.from_matrix(neighbour_matrices) for neighbour_matrices in rotation_matrices]
    quaternions = np.array([matrix.as_quat() for matrix in rotation_matrices])
    # pad quaternions
    if quaternions.shape[1] < N_NEIGHBOURS:
        padding_array = np.tile([0., 0., 0., 1.], (quaternions.shape[0], N_NEIGHBOURS, 1))
        padding_array[:, :quaternions.shape[1], :] = quaternions
        quaternions = padding_array
    return quaternions


class Chain:
    def __init__(self, parent, chain):
        pp_builder = Polypeptide.PPBuilder()
        self.id = chain.get_id()
        self.protein = parent
        self.peptides = pp_builder.build_peptides(chain, aa_only=False)
        self.residues = [res for pp in self.peptides for res in pp]
        if verbose:
            print('\tParsing', self.protein.name, self.id)
        # find missing backbone atoms
        self.ignored_residues = {}
        for i, res in enumerate(self.residues):
            try:
                self.N = res['N']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
                if verbose:
                    print('\t\tN missing in', res.get_resname(), 'at position', i)
            try:
                self.C = res['C']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
                if verbose:
                    print('\t\tC missing in', res.get_resname(), 'at position', i)
            try:
                self.CA = res['CA']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
                if verbose:
                    print('\t\tCA missing in', res.get_resname(), 'at position', i)
            if not is_aa(res, standard=True):
                self.ignored_residues[i] = res.get_resname()
                if verbose:
                    print('\t\tNon-standard amino acid', res.get_resname(), 'at position', i)
        # remove residues with missing atoms and nonstandard residues
        self.residues = [self.residues[i] for i in range(len(self.residues)) if i not in self.ignored_residues]
        # get torsional angles
        self.torsional_angles = [angles for pep in self.peptides for angles in pep.get_phi_psi_list()]
        self.phi_angles = [angles[0] if angles[0] is not None else 0. for angles in self.torsional_angles]
        self.psi_angles = [angles[1] if angles[1] is not None else 0. for angles in self.torsional_angles]
        self.phi_angles = [self.phi_angles[i] for i in range(len(self.phi_angles)) if i not in self.ignored_residues]
        self.psi_angles = [self.psi_angles[i] for i in range(len(self.psi_angles)) if i not in self.ignored_residues]

    def get_neighbours(self, chains):
        distances = []
        neighbours = []
        phi_angles = []
        psi_angles = []
        # get distances from all residues in chain to all residues in structure
        for cur_res in self.residues:
            distances.append([res2['CA'] - cur_res['CA'] for chain2 in chains for res2 in chain2.residues])
            neighbours.append([res2 for chain2 in chains for res2 in chain2.residues])
            phi_angles.append([phi for chain2 in chains for phi in chain2.phi_angles])
            psi_angles.append([psi for chain2 in chains for psi in chain2.psi_angles])
        distances = np.array(distances)  # dim 0: cur_res from selected chain, dim 1: res2 from any chain
        neighbours = np.array(neighbours, dtype=object)
        phi_angles = np.array(phi_angles)
        psi_angles = np.array(psi_angles)

        '''
        # VISUALISE DISTANCES
        if len(chains) > 1:
            fig, ax = plt.subplots(2, figsize=(12, 9))
            im = ax[0].imshow(distances)
        '''

        # sort neighbours by distance to each cur_res
        neighbour_indices = (np.argsort(distances, axis=1))
        neighbours = np.array([cur_neighbour[indices] for cur_neighbour, indices in zip(neighbours, neighbour_indices)])
        phi_angles = np.array([phi[idx] for phi, idx in zip(phi_angles, neighbour_indices)])
        psi_angles = np.array([psi[idx] for psi, idx in zip(psi_angles, neighbour_indices)])

        '''
        # VISUALISE SORTED DISTANCES
        distances = np.array([dist[indices] for dist, indices in zip(distances, neighbour_indices)])
        if len(chains) > 1:
            im = ax[1].imshow(distances)
            plt.show()
        '''

        # truncate neighbours to keep only cur_res and the N_NEIGHBOURS nearest neighbours
        if neighbours.shape[1] > N_NEIGHBOURS + 1:
            neighbours = np.copy(neighbours[:, :N_NEIGHBOURS + 1])
            phi_angles = np.copy(phi_angles[:, :N_NEIGHBOURS + 1])
            psi_angles = np.copy(psi_angles[:, :N_NEIGHBOURS + 1])
        # pad torsional angles
        if neighbours.shape[1] < N_NEIGHBOURS + 1:
            phi_angles = np.pad(phi_angles, pad_width=((0, 0), (0, N_NEIGHBOURS + 1 - neighbours.shape[1]),),
                                mode='constant', constant_values=0.)
            psi_angles = np.pad(psi_angles, pad_width=((0, 0), (0, N_NEIGHBOURS + 1 - neighbours.shape[1]),),
                                mode='constant', constant_values=0.)
        # encode torsional angles as (sin(angle), cos(angle))
        phi_angles = np.stack([np.sin(phi_angles), np.cos(phi_angles)], axis=2)
        psi_angles = np.stack([np.sin(psi_angles), np.cos(psi_angles)], axis=2)
        torsional_angles = np.stack([phi_angles, psi_angles], axis=3)
        return neighbours, torsional_angles


if __name__ == "__main__":
    # get arguments
    chain_file, out_dir, pdb_dir, verbose, pdb_layout = get_args()

    # create output directory
    if not out_dir.exists():
        out_dir.mkdir()

    # read chain file
    print('Reading chain file...')
    chain_codes = read_chain_file(chain_file)
    chain_total = len(chain_codes)
    print('\tDone.')

    # construct chain id lookup table
    print('Constructing chain lookup...')
    chain_lookup = defaultdict(list)
    if isinstance(chain_codes, list):
        for code in chain_codes:
            protein_code = code[:4].upper()
            chain_id = code[4:].upper()
            chain_lookup[protein_code].append(chain_id)
    else:
        # there is only one chain code in chain codes
        protein_code = chain_codes[:4].upper()
        chain_id = chain_codes[4:].upper()
        chain_lookup[protein_code].append(chain_id)
    print('\tDone.')

    # parse and save protein features
    chain_num = 1
    for protein_code in chain_lookup:
        if verbose:
            print('Chain', str(chain_num) + '/' + str(chain_total))
        parsed, skipped = featurise_protein(protein_code, chain_lookup[protein_code])
        chain_num += parsed
        chain_total -= skipped
    print('Done.')
