from Bio.PDB import *
import numpy as np
from scipy.spatial.transform import Rotation
import pathlib
import argparse
import gzip
from collections import defaultdict
import warnings
import torch
import matplotlib.pyplot as plt

aa_dict = {'GLY': 0., 'ALA': 1., 'CYS': 2., 'PRO': 3., 'VAL': 4., 'ILE': 5., 'LEU': 6., 'MET': 7.,
           'PHE': 8., 'TRP': 9., 'SER': 10., 'THR': 11., 'ASN': 12., 'GLN': 13., 'TYR': 14.,
           'ASP': 15., 'GLU': 16., 'HIS': 17., 'LYS': 18., 'ARG': 19., 'X': 20.}
N_NEIGHBOURS = 16

def get_args():
    arg_parser = argparse.ArgumentParser(description="Draw 3D plot of features.")
    arg_parser.add_argument('chain_name', type=str, help="name chains to be processed")
    arg_parser.add_argument('protein_id', type=str, help="name of protein. Structure must be in a gzipped pdb file "
                                                      "must be named according to the convention "
                                                      "1xyz.pdb")
    args = arg_parser.parse_args()
    return args.chain_name, args.protein_id


class Protein:
    def __init__(self, protein_id, file_handle, selected_chain_id):
        parser = PDBParser(QUIET=True)
        self.name = protein_id
        # parse pdb file
        self.structure = parser.get_structure(self.name, file_handle)
        # process residues for each chain in protein structure
        self.chains = [Chain(self, chain) for chain in self.structure[0].get_chains()]
        self.selected_chain = False
        for chain in self.chains:
            if chain.id == selected_chain_id:
                self.selected_chain = chain
                break
        if not self.selected_chain:
            print("chain not found")
        self.excluded_chains = []
        # generate structural features for each chain in chain list file
        neighbours = self.selected_chain.get_neighbours(self.chains)
        residue_labels = np.array([aa_dict[res.get_resname()] for res in neighbours[:, 0]])
        # derive local coordinate system for each residue
        CA_vectors, N_vectors, C_vectors, backbone_vectors, sidechain_vectors, third_bases = get_basis_vectors(neighbours)
        # calculate relative positions of neighbours in the local coordinate system of each residue
        displacements = get_displacements(CA_vectors, backbone_vectors, sidechain_vectors, third_bases)
        plot_atoms(CA_vectors, N_vectors, C_vectors, backbone_vectors, sidechain_vectors, third_bases, displacements)


def pseudo_beta(residue, CA_vector):
    N_vector = residue['N'].get_vector()
    C_vector = residue['C'].get_vector()

    N_CA = CA_vector - N_vector
    CA_C = C_vector - CA_vector
    N_C = C_vector - N_vector

    rotation_matrix_1 = vectors.rotaxis(131.5 * np.pi / 180, CA_C)
    rotation_matrix_2 = vectors.rotaxis(-7 * np.pi / 180, N_C)
    pseudo = (-N_CA).left_multiply(rotation_matrix_1)
    pseudo = pseudo.left_multiply(rotation_matrix_2)

    return pseudo


def get_basis_vectors(neighbours):
    CA_vectors = np.vectorize(lambda residue: residue['CA'].get_vector())(neighbours)
    N_vectors = np.vectorize(lambda residue: residue['N'].get_vector())(neighbours)
    C_vectors = np.vectorize(lambda residue: residue['C'].get_vector())(neighbours)

    backbone_vectors = np.vectorize(lambda vector: vector.normalized())(C_vectors - N_vectors)
    sidechain_vectors = np.vectorize(lambda residue, CA: residue['CB'].get_vector() - CA if 'CB' in residue
                                     else pseudo_beta(residue, CA))(neighbours, CA_vectors)

    # third basis is the cross product of the backbone vector and the sidechain vector
    third_bases = np.vectorize(lambda vec1, vec2: pow(vec1, vec2).normalized())(backbone_vectors, sidechain_vectors)
    # orthogonalise third basis and sidechain vector
    sidechain_vectors = np.vectorize(lambda vec1, vec2: pow(vec1, vec2).normalized())(backbone_vectors, third_bases)

    return CA_vectors, N_vectors, C_vectors, backbone_vectors, sidechain_vectors, third_bases


def get_displacements(CA_vectors, x_basis, y_basis, z_basis):
    global_displacement = CA_vectors[:, 1:] - CA_vectors[:, :1]
    # get x component of global displacement by dot product with the unit X vector
    x_displacement = np.vectorize(lambda disp, basis: disp * basis)(global_displacement, x_basis[:, :1])
    y_displacement = np.vectorize(lambda disp, basis: disp * basis)(global_displacement, y_basis[:, :1])
    z_displacement = np.vectorize(lambda disp, basis: disp * basis)(global_displacement, z_basis[:, :1])
    displacements = np.stack([x_displacement, y_displacement, z_displacement], axis=-1)

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
    if displacements.shape[1] < N_NEIGHBOURS:
        displacements = np.pad(displacements, pad_width=((0, 0), (0, N_NEIGHBOURS - displacements.shape[1]), (0, 0)),
                               mode='constant', constant_values=0.)
    return displacements


class Chain:
    def __init__(self, parent, chain):
        pp_builder = Polypeptide.PPBuilder()
        self.id = chain.get_id()
        self.protein = parent
        self.peptides = pp_builder.build_peptides(chain, aa_only=False)
        self.residues = [res for pp in self.peptides for res in pp]
        # find missing backbone atoms
        self.ignored_residues = {}
        for i, res in enumerate(self.residues):
            try:
                self.N = res['N']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
            try:
                self.C = res['C']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
            try:
                self.CA = res['CA']
            except KeyError:
                self.ignored_residues[i] = res.get_resname()
            try:
                self.CB = res['CB']
            except KeyError:
                pass
            if not is_aa(res, standard=True):
                self.ignored_residues[i] = res.get_resname()
        # remove residues with missing atoms and nonstandard residues
        self.residues = [self.residues[i] for i in range(len(self.residues)) if i not in self.ignored_residues]

    def get_neighbours(self, chains):
        distances = []
        neighbours = []
        # get distances from all residues in chain to all residues in structure
        for cur_res in self.residues:
            distances.append([res2['CA'] - cur_res['CA'] for chain2 in chains for res2 in chain2.residues])
            neighbours.append([res2 for chain2 in chains for res2 in chain2.residues])
        distances = np.array(distances)  # dim 0: cur_res from selected chain, dim 1: res2 from any chain
        neighbours = np.array(neighbours, dtype=object)
        '''
        # VISUALISE DISTANCES
        if len(chains) > 1:
            fig, ax = plt.subplots(2, figsize=(12, 9))
            im = ax[0].imshow(distances)
        '''
        # sort neighbours by distance to each cur_res
        neighbour_indices = (np.argsort(distances, axis=1))
        neighbours = np.array([cur_neighbour[indices] for cur_neighbour, indices in zip(neighbours, neighbour_indices)])

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

        return neighbours


def plot_atoms(CA, N, C, backbone_vec, sidechain_vec, third_bases, displacements):
    sel = 40
    range = 15
    #sel-range:sel+range
    CA_arr = np.array([vector.get_array() for vector in CA[:, 0]])
    N_arr = np.array([vector.get_array() for vector in N[:, 0]])
    C_arr = np.array([vector.get_array() for vector in C[:, 0]])
    print(CA.shape)
    disp = np.array([vector.get_array() for vector in CA[sel, 1:] - CA[sel, 0]])
    print(disp.shape)
    back_arr = np.array([vector.get_array() for vector in backbone_vec[:, 0]])
    side_arr = -np.array([vector.get_array() for vector in sidechain_vec[:, 0]])
    third_arr = np.array([vector.get_array() for vector in third_bases[:, 0]])

    C_CA = CA_arr - C_arr
    CA_N = N_arr - CA_arr
    N_C = C_arr[:-1] - N_arr[1:]
    CA_CA = CA_arr[:-1] - CA_arr[1:]
    C_N = N_arr[:] - C_arr[:]

    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(CA_arr[:, 0], CA_arr[:, 1], CA_arr[:, 2], back_arr[:, 0], back_arr[:, 1], back_arr[:, 2], color='red', length=1, lw=3)
    ax.quiver(CA_arr[:, 0], CA_arr[:, 1], CA_arr[:, 2], side_arr[:, 0], side_arr[:, 1], side_arr[:, 2], color='blue', length=1, lw=3)
    ax.quiver(CA_arr[:, 0], CA_arr[:, 1], CA_arr[:, 2], third_arr[:, 0], third_arr[:, 1], third_arr[:, 2], color='lime', length=1, lw=3)

    #ax.quiver(CA_arr[sel, 0], CA_arr[sel, 1], CA_arr[sel, 2], disp[:, 0], disp[:, 1], disp[:, 2], color='black', lw=2, arrow_length_ratio=0.1)

    ax.quiver(CA_arr[:, 0], CA_arr[:, 1], CA_arr[:, 2], CA_N[:, 0], CA_N[:, 1], CA_N[:, 2], arrow_length_ratio=0, color='black', lw=5)
    ax.quiver(C_arr[:, 0], C_arr[:, 1], C_arr[:, 2], C_CA[:, 0], C_CA[:, 1], C_CA[:, 2], arrow_length_ratio=0, color='black', lw=5)
    ax.quiver(N_arr[1:, 0], N_arr[1:, 1], N_arr[1:, 2], N_C[:, 0], N_C[:, 1], N_C[:, 2], arrow_length_ratio=0, color='black', lw=5)
    #ax.quiver(CA_arr[1:, 0], CA_arr[1:, 1], CA_arr[1:, 2], CA_CA[:, 0], CA_CA[:, 1], CA_CA[:, 2], arrow_length_ratio=0, color='grey', lw=5)
    #ax.quiver(C_arr[:, 0], C_arr[:, 1], C_arr[:, 2], C_N[:, 0], C_N[:, 1], C_N[:, 2], arrow_length_ratio=0,color='grey', lw=5)

    ax.scatter(CA_arr[:, 0], CA_arr[:, 1], CA_arr[:, 2], color='black', s=500, depthshade=True)
    ax.scatter(N_arr[:, 0], N_arr[:, 1], N_arr[:, 2], color='blue', s=200, depthshade=True)
    ax.scatter(C_arr[:, 0], C_arr[:, 1], C_arr[:, 2], color='black', s=200, depthshade=True)

    neighbours = CA_arr[sel, :] + disp
    #ax.scatter(CA_arr[sel, 0], CA_arr[sel, 1], CA_arr[sel, 2], color='cyan', s=100000, depthshade=True)
    #ax.scatter(neighbours[:, 0], neighbours[:, 1], neighbours[:, 2], color='yellow', s=1000, depthshade=False)


    xlim = ax.get_xlim()
    xlim = xlim[1]-xlim[0]
    ylim = ax.get_ylim()
    ylim = ylim[1] - ylim[0]
    zlim = ax.get_zlim()
    zlim = zlim[1] - zlim[0]

    ax.set_box_aspect(aspect=(xlim, ylim, zlim), zoom=1)
    ax.set_frame_on(False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    chain_name, protein_id = get_args()
    pdb_file = pathlib.Path(protein_id.lower() + '.pdb')
    selected_chain = chain_name
    if pdb_file.exists():
        with open(pdb_file, 'r') as file:
            # featurise protein
            protein = Protein(protein_id, file, selected_chain)
    else:
        # missing pdb file
        print(pdb_file, 'does not exist.')