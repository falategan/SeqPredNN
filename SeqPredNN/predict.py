import argparse
import logging
import pathlib
import warnings
from typing import TextIO

import numpy as np
import sklearn.metrics as metrics
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import AMINO_ACID_INDICES, STANDARD_AMINO_ACIDS, AMINO_ACID_LETTERS
from neural_net import StructureDataset, NeuralNetwork
from plots import Plots
from train_model import Dataset, read_test_file

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    """
    Fetch command-line arguments
    """
    arg_parser = argparse.ArgumentParser(description="predict sequences from protein conformational features and test "
                                                     "model performance")
    arg_parser.add_argument('feature_dir',
                            type=str,
                            help="directory of processed chain features")
    arg_parser.add_argument('test_list',
                            type=str,
                            help="path to a newline-delimited list of protein chain codes for sequence prediction")
    arg_parser.add_argument('model_params',
                            type=str,
                            help="file containing trained model parameters")
    arg_parser.add_argument('-p',
                            '--pred_only',
                            action='store_true',
                            help="only predict sequences, without testing the model and comparing predictions with "
                                 "true sequences.")
    arg_parser.add_argument('-o',
                            '--out_dir',
                            type=str,
                            default='./pred',
                            help="output directory. Will create a new directory if OUT_DIR does not exist.")
    args = arg_parser.parse_args()
    return pathlib.Path(args.feature_dir), pathlib.Path(args.test_list), pathlib.Path(args.model_params), \
        pathlib.Path(args.out_dir), args.pred_only


class Predictor:
    def __init__(self, parameter_path, prediction_only_mode, test_chains, feature_directory):
        parameters = torch.load(parameter_path)
        network_shape = [len(layer) for layer in parameters.values()][:-2:2]
        input_nodes = parameters['linear_relu_stack.0.weight'].shape[1]
        neighbour_count = int((input_nodes - 4) / 11)
        self.dataset = Dataset(test_chains, feature_directory, neighbour_count)
        self.model = NeuralNetwork(input_nodes=input_nodes, network_shape=network_shape, dropout=1, ).to(device)
        self.model.load_state_dict(parameters)
        self.prediction_only_mode = prediction_only_mode

    def sequence(self, chain_id: str) -> tuple[list[int], list[np.ndarray], list[torch.tensor], list[np.ndarray]]:
        """
        Input the structural features of each residue in a chain into the neural network to predict the amino acid for
        each residue in the chain

        Args:
            chain_id: the chain id of the polypeptide chain for which the sequence should be predicted

        Returns:
            (tuple):
            predicted_residues: a list of integers encoding the amino acids predicted as the sequence of the
                polypeptide chain
            chain_softmax: a list of 20 by 1 numpy arrays that represent the predicted probability for each amino
                acid at each residue position in the polypeptide chain
            chain_loss: a list of the cross-entropy loss values between the predicted amino acid probability tensor
                and the true amino acid tensor for each residue in the polypeptide chain
            true_residues: a list of integers encoding the true sequence of the amino acid chain as it was parsed
                by the featurise module
        """

        chain_features = self.dataset.chains[chain_id].features
        feature_tensors = {feature_name: torch.tensor(feature, dtype=torch.float64)
                           for feature_name, feature in chain_features.items()}
        chain_dataset = StructureDataset(feature_tensors)
        dataloader = DataLoader(chain_dataset, shuffle=False)
        self.model.eval()
        chain_softmax = []
        predicted_residues = []
        true_residues = []
        chain_loss = []
        # load residue features one at a time
        for inputs, label in dataloader:
            with torch.no_grad():
                # predict residue
                output = self.model(inputs.to(device))
                if not self.prediction_only_mode:
                    loss_fn = nn.CrossEntropyLoss(ignore_index=20)
                    loss = loss_fn(output, label.to(device)).item()
                    chain_loss.append(loss)
                    true_residues.extend(label.cpu().numpy())
            softmax = nn.functional.softmax(output, dim=1)
            chain_softmax.extend(softmax.cpu().numpy())
            # the output node with the highest value is the predicted residue
            top_residue = output.argmax(1)
            predicted_residues.append(int(top_residue))
        return predicted_residues, chain_softmax, chain_loss, true_residues

    def complete_seq(self, pred_residues, true_residues, chain, feat_dir):
        """
        Generate a string representing the true and predicted sequence of a polypeptide chain, adding unknown residues
        represented by the character 'X'

        Args:
            pred_residues: a list of integers representing the predicted amino acid at each residue position in a
            polypeptide chain
            true_residues: a list of integers representing the true amino acid at each residue position in a
            polypeptide chain
            chain: ID string of a polypeptide chain
            feat_dir: pathlib Path to the directory where structural features are saved

        Returns:
            (tuple):
                pred_seq: string representing the predicted amino acid sequence of the polypeptide chain

                true_seq: string representing the true amino acid sequence of the polypeptide chain
        """
        excluded_res_path = feat_dir / ('excluded_residues_' + chain + '.csv')
        excluded_residues = {}
        pred_seq = ''
        true_seq = ''
        # read list of excluded residues, if any residues were excluded
        if excluded_res_path.exists():
            with open(excluded_res_path, 'r') as file:
                # excluded residues written in file as index,residue_name
                for line in file:
                    line = line.split(',')
                    excluded_residues[int(line[0])] = line[1].strip('\n')
        j = 0
        for i in range(len(list(excluded_residues)) + len(pred_residues)):
            # predict 'X' for all excluded residues
            if i in excluded_residues:
                pred_seq += 'X'
                if not self.prediction_only_mode:
                    # true residues are only 'X' if they are non-standard residues
                    if excluded_residues[i] in AMINO_ACID_INDICES:
                        true_seq += AMINO_ACID_LETTERS[excluded_residues[i]]
                    else:
                        true_seq += 'X'
            else:
                pred_seq += list(AMINO_ACID_LETTERS.values())[int(pred_residues[j])]
                if not self.prediction_only_mode:
                    true_seq += list(AMINO_ACID_LETTERS.values())[int(true_residues[j])]
                j += 1
        return pred_seq, true_seq


def check_labels(true_residues):
    """
    List all residues that do not occur in the true sequence.

    Args:
        true_residues: list of true amino acids in the polypeptide chain
    Returns:
        (tuple):
            unused_residues: list of standard amino acids that do not occur in the sequence

            labels: list of standard amino acids that are present in the sequence
    """
    unused_residues = []
    labels = []
    for key in range(0, 20):
        if key not in true_residues:
            unused_residues.append(key)
        else:
            labels.append(key)
    return unused_residues, labels


def write_report(file: TextIO, chain_report: str, avg_loss: np.ndarray, unused_residues: list[int]) -> None:
    """
    Write a text file reporting precision, recall and F-score statistics for each amino acid class

    Args:
        file: file handle for the file where the report should be written.
        chain_report: text reporting classification statistics
        avg_loss: average cross-entropy loss between the true and predicted labels
        unused_residues: amino acid classes that do not occur in the true residues
    """
    file.writelines(chain_report)
    file.write(f'\nAverage loss: {avg_loss}')
    file.write('\nNote: precision and F-scores are set to 0.0 for classes that have no predictions')
    if unused_residues:
        file.write(
            f'\nAmino acids with 0 support: {", ".join([STANDARD_AMINO_ACIDS[idx] for idx in unused_residues])}.')


class Evaluator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.model_predictions = []
        self.model_true = []
        self.model_softmax = []
        self.model_loss = []

    def chain(self,
              chain: str,
              true_seq: str,
              true_residues: list[np.ndarray],
              pred_residues: list[int],
              loss: list[int],
              chain_softmax: list[np.ndarray]) -> None:
        """
        Generate and save text files with classification metric reports, prediction probabilities and per-residue loss.

        Args:
            chain: PDB Chain ID
            true_seq: True amino acid sequence of the polypeptide chain
            true_residues: Integers representing the true amino acid at each residue position
            pred_residues: Integers representing the predicted amino acid at each residue position
            loss: Cross-entropy lost for each predicted residue
            chain_softmax: Soft-max probability for each predicted residue
        """
        self.model_predictions.extend(pred_residues)
        self.model_true.extend(true_residues)
        self.model_softmax.extend(chain_softmax)
        self.model_loss.extend(loss)
        avg_loss = np.mean(loss)

        # list residues that do and do not occur in the true sequence
        unused_residues, labels = check_labels(true_residues)
        # only plot confusion matrices if all 20 residues occur in the label set
        report_amino_acids = [STANDARD_AMINO_ACIDS[i] for i in labels]
        with warnings.catch_warnings():
            # suppress sklearn warnings
            warnings.simplefilter("ignore")
            # generate report of per-class precision, recall and F1-scores
            chain_report = metrics.classification_report(true_residues, pred_residues,
                                                         target_names=report_amino_acids,
                                                         labels=labels, digits=3, zero_division=0)
            metric_dict = metrics.classification_report(self.model_true,
                                                        self.model_predictions,
                                                        target_names=report_amino_acids,
                                                        digits=5,
                                                        zero_division=0,
                                                        labels=labels,
                                                        output_dict=True)
        # write files
        report_dir = self.out_dir / 'chain_reports'
        if not report_dir.exists():
            report_dir.mkdir()
        with open(report_dir / f'{chain}_report.txt', 'w') as file:
            write_report(file, chain_report, avg_loss, unused_residues)
        with open(report_dir / f'{chain}_report.csv', 'w') as file:
            metric_list = ['precision', 'recall', 'f1-score', 'support']
            file.write(' ,Precision, Recall, F1-Score, Support\n')
            for label in STANDARD_AMINO_ACIDS:
                if label in report_amino_acids:
                    file.write(
                        f'{label},{",".join([str(metric_dict[label][metric]) for metric in metric_list])}\n')
                    continue
                file.write(
                    f'{label},N/A,N/A,N/A,0\n')
        original_dir = self.out_dir / 'original_sequences'
        if not original_dir.exists():
            original_dir.mkdir()
        with open(original_dir / f'{chain}_original.fasta', 'w') as file:
            file.write(f'>{chain[:4].upper()}|Chain {chain[4:].upper()}|SeqPredNN original PDB sequence\n{true_seq}')
        probability_dir = self.out_dir / 'probabilities'
        if not probability_dir.exists():
            probability_dir.mkdir()
        np.savetxt(probability_dir / f'{chain}_probabilities.csv', chain_softmax, delimiter=',',
                   header=','.join(STANDARD_AMINO_ACIDS), comments='')
        loss_dir = self.out_dir / 'residue_loss'
        if not loss_dir.exists():
            loss_dir.mkdir()
        np.savetxt(loss_dir / f'{chain}_residue_losses.csv', loss, delimiter=',')

    def model(self):
        """ Generate a classification report, confusion matrices and a top-K accuracy curve for the test set."""
        model_plot = Plots(self.out_dir)
        avg_loss = np.mean(self.model_loss)

        # list residue labels that do and do not occur in the test set
        unused_residues, labels = check_labels(self.model_true)
        # only plot confusion matrices if all 20 residues occur in the label set
        if not unused_residues:
            model_plot.confusion_matrix(self.model_true, self.model_predictions, None, 'unnormalised_conf_matrix')
            model_plot.confusion_matrix(self.model_true, self.model_predictions, 'pred', 'pred_norm_conf_matrix')
            model_plot.confusion_matrix(self.model_true, self.model_predictions, 'true', 'true_norm_conf_matrix')
            with warnings.catch_warnings():
                # suppress sklearn warnings
                warnings.simplefilter("ignore")
                top_k_accuracy = {k: metrics.top_k_accuracy_score(self.model_true, self.model_softmax, k=k) for k in
                                  range(1, 21)}
            with open(self.out_dir / 'top_K.csv', 'w') as file:
                file.write(f'k,Top-k accuracy')
                for k in top_k_accuracy:
                    file.write(f'{k},{top_k_accuracy[k]}\n')
        else:
            print("Could not plot confusion matrices or top-K accuracy curve because some residues do not have any "
                  "labels in the test set")
        with warnings.catch_warnings():
            # suppress sklearn warnings
            warnings.simplefilter("ignore")
            # generate report of per-class precision, recall and F1-scores
            report_amino_acids = [STANDARD_AMINO_ACIDS[i] for i in labels]

            model_report = metrics.classification_report(self.model_true,
                                                         self.model_predictions,
                                                         target_names=report_amino_acids,
                                                         digits=3,
                                                         zero_division=0,
                                                         labels=labels)
            metric_dict = metrics.classification_report(self.model_true,
                                                        self.model_predictions,
                                                        target_names=report_amino_acids,
                                                        digits=5,
                                                        zero_division=0,
                                                        labels=labels,
                                                        output_dict=True)

        with open(self.out_dir / 'report.txt', 'w') as file:
            write_report(file, model_report, avg_loss, unused_residues)

        with open(self.out_dir / 'report.csv', 'w') as file:
            metric_list = ['precision', 'recall', 'f1-score', 'support']
            file.write(' ,Precision, Recall, F1-Score, Support\n')
            for label in STANDARD_AMINO_ACIDS:
                if label in report_amino_acids:
                    file.write(
                        f'{label},{",".join([str(metric_dict[label][metric]) for metric in metric_list])}\n')
                    continue
                file.write(
                    f'{label},N/A,N/A,N/A,0\n')


def prediction():
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="%(message)s")
    logging.captureWarnings(False)

    feature_directory, test_list, parameter_path, output_directory, prediction_only_mode = get_args()

    if not feature_directory.exists():
        raise FileNotFoundError(feature_directory)
    if not output_directory.exists():
        output_directory.mkdir()

    test_chains = read_test_file(test_list)
    predict = Predictor(parameter_path, prediction_only_mode, test_chains, feature_directory)
    test_chains = predict.dataset.chains
    if predict.dataset.excluded_chains:
        with open(output_directory / 'excluded_chains.txt', 'w') as file:
            file.write('\n'.join(predict.dataset.excluded_chains))
    evaluate = Evaluator(output_directory)
    n_chains = len(test_chains)
    i = 1
    prediction_dir = output_directory / 'predicted_sequences'
    if not prediction_dir.exists():
        prediction_dir.mkdir()
    for chain in test_chains:
        print(f'Chain {i}/{n_chains}')
        pred_residues, chain_softmax, loss, true_residues = predict.sequence(chain)
        pred_seq, true_seq = predict.complete_seq(pred_residues, true_residues, chain, feature_directory)
        print(chain, '- Predicted sequence:\n' + pred_seq)
        if not prediction_only_mode:
            print(chain, '- Original sequence:\n' + true_seq + '\n')
        with open(prediction_dir / f'{chain}_predicted.fasta', 'w') as file:
            file.write(f'>{chain[:4].upper()}|Chain {chain[4:].upper()}|SeqPredNN Prediction\n{pred_seq}')
        true_known_residues = []
        pred_known_residues = []
        softmax_known_residues = []

        for true_residue, pred_residue, softmax_residue in zip(true_residues, pred_residues, chain_softmax):
            if true_residue != AMINO_ACID_INDICES['X']:
                true_known_residues.append(true_residue)
                pred_known_residues.append(pred_residue)
                softmax_known_residues.append(softmax_residue)
        if not prediction_only_mode:
            evaluate.chain(chain, true_seq, true_known_residues, pred_known_residues, loss, softmax_known_residues)
        i += 1

    if not prediction_only_mode:
        evaluate.model()


if __name__ == '__main__':
    prediction()
