import numpy as np
import torch
import argparse
import pathlib
from collections import defaultdict
# import matplotlib.pyplot as plt


class Sampling:
    def __init__(self, input_dir, test_list_path, train_ratio):
        self.train_ratio = train_ratio
        self.input_dir = input_dir
        # read list of test chains
        self.test_chain_codes = []
        if test_list_path is not None:
            print('Reading test list...\n')
            with open(test_list_path, 'r') as file:
                self.test_chain_codes = [line.split(' ')[0].strip('\n') for line in file]
                if self.test_chain_codes[0] == 'PDBchain':
                    self.test_chain_codes = self.test_chain_codes[1:]

        # generate a unique index for each residue in the processed chains
        print('Reading chain list...\n')
        self.chains = {}
        with open(input_dir / 'chain_list.csv', 'r') as file:
            start_idx = 0
            for line in file:
                chain, length = line.strip('\n').split(',')
                length = int(length)
                # only include chains that are not in the test set
                if chain not in self.test_chain_codes:
                    self.chains[chain] = (start_idx, length)
                    start_idx += length

        # store residue indices as {index: chain_code} key-value pairs
        print('Compiling index list...\n')
        self.idx_dict = {idx: chain for chain in self.chains for idx in
                         range(self.chains[chain][0], self.chains[chain][0] + self.chains[chain][1])}

    @staticmethod
    def undersample(indices, sample_size):
        # draw random sample of indices from each amino acid class for the training set
        remaining_idx = {float(i): [] for i in range(20)}
        sample_idx = []
        for amino_acid in indices:
            sample_idx.append(np.random.choice(indices[amino_acid], sample_size, replace=False))
            remaining_idx[amino_acid] = [idx for idx in indices[amino_acid] if not np.isin(idx, sample_idx[-1])]
        sample_idx = np.concatenate(sample_idx)
        return sample_idx, remaining_idx

    def get_sample_chains(self, indices):
        # store chain code for each sampled residue as {chain_code: residue_index} key-value pairs
        chain_idx = defaultdict(list)
        for idx in indices:
            chain_idx[self.idx_dict[idx]].append(idx)
        return chain_idx

    def load_feature_files(self, chain, chain_start, feature, subset):
        subset_feature = []
        feature_array = torch.load(self.input_dir / (feature + '_' + chain + '.pt'))
        if subset[chain] is not None:
            subset_feature.extend([feature_array[idx - chain_start] for idx in subset[chain]])
        return subset_feature

    def get_features(self, train_idx, validation_idx):
        # store chain code for each sampled residue as {chain_code: residue_index} key-value pairs
        train_chains = self.get_sample_chains(train_idx)
        validation_chains = self.get_sample_chains(validation_idx)

        train_feature_dict = {}
        validation_feature_dict = {}

        print('Fetching features for sampled indices...\n')
        feature_strings = ['displacements', 'residue_labels', 'rotations', 'torsional_angles']
        for feature in feature_strings:
            train_features, validation_features = [], []
            for chain in self.chains:
                if any([chain in subset for subset in [train_chains, validation_chains]]):
                    # load features for sampled residues from input folder
                    chain_train_features = self.load_feature_files(chain, self.chains[chain][0], feature, train_chains)
                    train_features.extend(chain_train_features)
                    chain_validation_features = self.load_feature_files(chain, self.chains[chain][0], feature, validation_chains)
                    validation_features.extend(chain_validation_features)
            train_feature_dict[feature] = torch.tensor(train_features, dtype=torch.float)
            validation_feature_dict[feature] = torch.tensor(validation_features, dtype=torch.float)
        return train_feature_dict, validation_feature_dict

    def unbalanced(self):
        print('\nDrawing unbalanced samples...\n')
        # draw a random sample of indices from the residue index dictionary for the training set
        indices = list(self.idx_dict)
        sample_size = int(self.train_ratio * len(indices))
        print('Training set:', sample_size, 'residues')
        train_idx = np.random.choice(indices, sample_size, replace=False)
        validation_idx = list(set(indices) - set(train_idx))
        print('Validation set:', len(validation_idx), 'residues\n')
        # get features for sampled residues
        train_feature_dict, validation_features_dict = self.get_features(train_idx, validation_idx)
        return train_feature_dict, validation_features_dict

    def balanced(self):
        amino_acids = ['GLY', 'ALA', 'CYS', 'PRO', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP',
                       'SER', 'THR', 'ASN', 'GLN', 'TYR', 'ASP', 'GLU', 'HIS', 'LYS', 'ARG']

        # get residue labels for each chain
        print([(chain, self.chains[chain], torch.load(self.input_dir / ('residue_labels_' + chain + '.pt')).shape)
                                       for chain in self.chains])
        all_residues = np.concatenate([torch.load(self.input_dir / ('residue_labels_' + chain + '.pt'))
                                       for chain in self.chains], axis=0)
        print(all_residues.shape)
        residue_idx = {float(i): [] for i in range(20)}
        # store indices for each amino acid class as {residue_class: [indices]} key-value pairs
        for idx, residue in enumerate(all_residues):
            residue_idx[residue].append(idx)

        frequencies = {residue: len(indices) for residue, indices in residue_idx.items()}
        print('Residue frequencies:')
        for residue, count in frequencies.items():
            print(' ', amino_acids[int(residue)] + ':', count)

        # sample size for each amino acid class is the frequency of the least common amino acid class.
        aa_sample_size = min(frequencies.values())
        print('\nDrawing', aa_sample_size, 'residues for each amino acid class\n')
        # draw random sample of indices from each amino acid class for the training set
        train_size = int(aa_sample_size * self.train_ratio)
        train_idx, remaining_idx = self.undersample(residue_idx, train_size)
        print('Training set:', train_size, 'residues per amino acid class')
        validation_size = aa_sample_size - train_size
        validation_idx, _ = self.undersample(remaining_idx, validation_size)
        print('Validation set:', validation_size, 'residues per amino acid class\n')

        # get features for sampled residues
        train_feature_dict, validation_features_dict = self.get_features(train_idx, validation_idx)
        return train_feature_dict, validation_features_dict
