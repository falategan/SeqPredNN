# SeqPredNN

Deep feed-forward neural network for predicting amino acid sequences from protein conformations

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

1.  ** Featurise your protein strucutures using `featurise.py` **
-   featurise.py [-h] [-o OUT_DIR] [-v] chain_list pdb_dir {all,divided}
