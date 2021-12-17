# SeqPredNN

Deep feed-forward neural network for predicting amino acid sequences from protein conformations.

## Requirements

* Python >= 3.7
* Pytorch
* Numpy
* SciKit-Learn
* Matplotlib
* Scipy
* Biopython

## Usage

### Predicting protein sequences using the pretrained model:

1.  Featurise your protein strucutures using `featurise.py`

        featurise.py -v examples/chain_list.txt examples/divided_pdb divided -o example_features
        
-- examples of a chain list and PDB directory are given in examples/
-- examples/chain_list.txt is a text file specifying the protein chains that must be featurised. It contains a newline-seperated list of protein chain IDs in the format 1XYZA for chain A of protein 1XYZ
-- examples/divided_pdb is a folder containing protein structures in PDB format. The PDB files must be gzipped and named according to the wwpdb archive convention e.g. pdb1xyz.ent.gz
-- the divided/all keyword specifies the structure of the PDB directory, according to the convention used by the wwpdb archive. In a "divided" directory the PDB files are  stored in subdirectorys named according to the middle 2 letters of the PDB code e.g. protein 1XYZ would be found in pdb_dir/XY/. In an "all" directory, all the PDB files are in a single directory.
