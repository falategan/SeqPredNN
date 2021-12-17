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

        featurise.py -v -o example_features examples/chain_list.txt examples/divided_pdb divided 

    - examples of a chain list and PDB directory are given in examples/
    - examples/chain_list.txt is a text file specifying the protein chains that must be featurised. It contains a newline-seperated list of protein chain IDs in the format 1XYZA for chain A of protein 1XYZ
    - examples/divided_pdb is a folder containing protein structures in PDB format. The PDB files must be gzipped and named according to the wwpdb archive convention e.g. pdb1xyz.ent.gz
    - the divided/all keyword specifies the structure of the PDB directory, according to the convention used by the wwpdb archive. In a "divided" directory the PDB files are stored in subdirectorys named according to the middle 2 letters of the PDB code e.g. protein 1XYZ would be found in pdb_dir/XY/. In an "all" directory, all the PDB files are in a single directory.

2. Predict sequence using `prediction.py`

                prediction.py -p example_features examples/chain_list.txt pretrained_parameters.pth
 
    - prediction-only mode (-p) does not evaluate the model by comparing predictions with the original sequence 
 
### Training your own model:

1. Download the PDB files of the structures in your training dataset - https://www.wwpdb.org/ftp/pdb-ftp-sites
2. Featurise your protein strucutures using `featurise.py`
    - see **Predicting protein sequences using the pretrained model** for more details
3. Train the model using `train_model.py`

                train_model.py -r 0.9 -t test_chains.txt -o my_model -e 200 feature_directory balanced

    - The train ratio (-r) is the fraction of residues assigned to the training dataset. The remaining residues are assigned to a validation set used to evaluate the model during training
    - the test chain file (-t) specifies which chains should be excluded from the training and validation datasets so that they can be used for independent evaluation of the model. The test chain file must be in the same format as examples/chain_list.txt.
    - (-e) is used to specify the number of epochs for training
    - the balanced/unbalanced keyword specifies the sampling mode. "unbalanced" sampling partitions all the residues in the features into the training and validation datasets. "balanced" sampling undersamples the residues so that each of the 20 amino acid classes occur the same number of times in the dataset.
4. Test your model using `prediction.py`
                
                prediction.py - o predictions feature_directory test_chains.txt my_model/model_parameters.pth
          
    - Evaluation output:
        - classification report with precision, recall and f1-score for each amino acid class
        - top K accuracy of the predictions for each amino acid class
        - 3 confusion matrices (unnormalised, normalised by prediciton and normalised by true residue
        - For each chain in the test set:
                - the predicted sequence
                - the probabilities for each amino acid class produced by the model for each preducted residue
                - a classification report
                - cross-entropy loss for each predicted residue

## Pretrained model 

The pretrained model was trained using the chains in 40_res3_nobrks.pisces
                
