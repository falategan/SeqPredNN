import argparse
import logging
import pathlib
import numpy as np
import torch
from neural_net import StructureDataset, NeuralNetwork
from plots import Plots
from sklearn.metrics import classification_report, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader
from constants import AMINO_ACID_INDICES, FEATURE_LIST, STANDARD_AMINO_ACIDS


def comma_separated_int_list(string):
    """Define a command-line argument written as a comma-separated list"""
    return [int(number) for number in string.split(',')]


def get_arguments():
    """Fetch command-line arguments"""
    argument_parser = argparse.ArgumentParser(
        description="Train a neural network model to predict amino acid residues.",
        formatter_class=argparse.RawTextHelpFormatter)
    argument_parser.add_argument(
        "input_directory",
        type=str,
        help="Input directory containing structural feature files.")
    argument_parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default="./model",
        help="output directory. Creates a new directory if OUT_DIR does not exist. Default: ./model")
    argument_parser.add_argument(
        "sampling_mode",
        type=str,
        choices=["balanced", "unbalanced"],
        help="balanced\tUndersample the data so each amino acid class in the training data has the same number of "
             "residues as the least abundant amino acid class\n"
             "unbalanced\tRandomly sample all residues irrespective of the amino acid distribution\n")

    validation_group = argument_parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "-v",
        "--validation_file",
        type=str,
        help="Path to a .npz file that specifies the residues in the validation set")
    validation_group.add_argument(
        "-r",
        "--train_ratio",
        type=float,
        default=0.8,
        help="The fraction of residues assigned to the training set. "
             "The rest is assigned to the validation set. Default: 0.8")

    argument_parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        help="path for the list of chains to be excluded from the training and validation sets. Default: None")
    argument_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="number of times the dataset is passed to through the model during training. Default: 100")
    argument_parser.add_argument(
        "--neighbours",
        type=int,
        default=16,
        help="Specify the number of neighbouring residues that should be included in the structural context.")
    argument_parser.add_argument(
        "--layers",
        type=comma_separated_int_list,
        default=[64, 64, 64],
        help="Specify the width of each layer in the neural network, excluding the input and output layers")
    argument_parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Specify the dropout value for the neural network layers.")
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The number of residues used for backpropagating gradients during a training step.")
    argument_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed for the random number generator")

    arguments = argument_parser.parse_args()
    logging.debug(f'DEBUG - arguments: {arguments}')

    input_directory = pathlib.Path(arguments.input_directory)
    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory {input_directory} does not exist")
    output_directory = pathlib.Path(arguments.out_dir)
    validation_path = None
    if arguments.validation_file is not None:
        validation_path = pathlib.Path(arguments.validation_file)
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation set file {validation_path} does not exist")
    test_set = None
    if arguments.test_set is not None:
        test_set = pathlib.Path(arguments.test_set)
        if not test_set.exists():
            raise FileNotFoundError(f"Test set file {test_set} does not exist")
    train_ratio = arguments.train_ratio
    if train_ratio <= 0 or train_ratio > 1:
        raise ValueError(f"Invalid train ratio {train_ratio}. The train ratio must be a number between "
                         f"0 and 1")
    epochs = arguments.epochs
    if epochs < 1:
        raise ValueError(f"Invalid number of epochs {epochs}. The number of epochs must be a positive integer value")
    neighbours = arguments.neighbours
    if neighbours < 1:
        raise ValueError(f"Invalid number of neighbours {neighbours}. The number of neighbours must be a positive "
                         f"integer value")
    dropout = arguments.dropout
    if dropout < 0 or dropout > 1:
        raise ValueError(f"Invalid dropout value {dropout}. The dropout value must be a number between "
                         f"0 and 1")
    batch_size = arguments.batch_size
    if batch_size < 1:
        raise ValueError(f"Invalid batch size value {batch_size}. The batch size value must be a a positive "
                         f"integer")

    argument_output = (input_directory,
                       output_directory,
                       arguments.sampling_mode,
                       validation_path,
                       train_ratio,
                       test_set,
                       epochs,
                       neighbours,
                       arguments.layers,
                       dropout,
                       batch_size,
                       arguments.seed
                       )

    return argument_output


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (batch_inputs, batch_labels) in enumerate(dataloader):
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # compute prediction error
        pred = model(batch_inputs)
        loss = loss_fn(pred, batch_labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(batch_inputs)
        total_loss += loss

    mean_loss = total_loss / size
    return mean_loss


def validate_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    validation_loss = 0
    correct = 0
    class_true_positive = torch.tensor([0] * 20, device=device, dtype=torch.float64)
    class_positive = torch.tensor([0] * 20, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            pred = model(batch_inputs)
            validation_loss += loss_fn(pred, batch_labels).item()
            correct += (pred.argmax(1) == batch_labels).type(torch.float).sum().item()
            class_positive_batch = torch.stack([(pred.argmax(1) == i) for i in range(20)])
            class_positive += class_positive_batch.type(torch.float).sum(dim=1)
            class_true_batch = torch.stack([(batch_labels == i) for i in range(20)])
            class_true_positive_batch = class_positive_batch * class_true_batch
            class_true_positive += class_true_positive_batch.type(torch.float).sum(dim=1)
    validation_loss /= size
    correct /= size
    class_recall = (class_true_positive / class_positive).tolist()
    recall_dictionary = {amino_acid: recall for amino_acid, recall in zip(STANDARD_AMINO_ACIDS, class_recall)}
    print(f"Accuracy: {(100 * correct):>.3f}%, Average loss: {validation_loss:>8f} \n")
    return validation_loss, 100 * correct, recall_dictionary


def validate_final(dataloader, model, device):
    pred = []
    true = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            output = model(inputs.to(device))  # Feed Network
            output = (torch.max(output, 1)[1]).data.cpu().numpy()  # take output node with the largest value
            pred.extend(output)
            labels = labels.data.int().cpu().numpy()
            true.extend(labels)
    return true, pred


class Features:
    def __init__(self, feature_directory):
        self.feature_directory = feature_directory


class Sampler:
    def __init__(self, dataset, train_ratio, validation_path, seed):
        self.dataset = dataset
        self.random_number_generator = np.random.default_rng(seed=seed)
        self.dataset_size = len(self.dataset)
        self.sample_space = np.arange(self.dataset_size)
        if validation_path:
            self.train_indices, self.validation_indices = self.read_preset_validation_set(validation_path)
        else:
            self.train_indices, self.validation_indices = self.randomize_validation_set(train_ratio)
        self.validation_features = {feature: self.dataset[feature][self.validation_indices]
                                    for feature in FEATURE_LIST}

    def read_preset_validation_set(self, validation_path):
        validation_indices = dict(np.load(validation_path))
        new_validation_indices = [residue_position + self.dataset.chains[chain_id].start
                                  for chain_id in validation_indices
                                  for residue_position in validation_indices[chain_id]]
        train_indices = np.setdiff1d(self.sample_space, new_validation_indices, assume_unique=True)
        return train_indices, new_validation_indices

    def randomize_validation_set(self, train_ratio):
        sample_size = int(self.dataset_size * train_ratio)
        train_indices = self.random_number_generator.choice(self.sample_space, size=sample_size,
                                                            replace=False)
        validation_indices = np.setdiff1d(self.sample_space, train_indices, assume_unique=True)
        return train_indices, validation_indices

    def unbalanced(self):
        print('\nDrawing unbalanced samples...\n')
        train_features = {feature: self.dataset[feature][self.train_indices]
                          for feature in FEATURE_LIST}
        return train_features.copy(), self.validation_features, self.validation_indices

    def balanced(self):
        non_validation_labels = self.dataset['residue_labels'][self.train_indices]
        class_indices = [np.extract(non_validation_labels == amino_acid, self.train_indices)
                         for amino_acid in list(AMINO_ACID_INDICES.values())[:-1]]
        residue_frequencies = [len(amino_acid_partition)
                               for amino_acid_partition in class_indices]
        class_sample_size = min(residue_frequencies)
        class_samples = [self.random_number_generator.choice(indices, size=class_sample_size, replace=False)
                         for indices in class_indices]
        balanced_train_indices = np.concatenate(class_samples, axis=0)
        train_features = {feature: self.dataset[feature][balanced_train_indices]
                          for feature in FEATURE_LIST}
        amino_acid_counts = {amino_acid: 0 for amino_acid in AMINO_ACID_INDICES}
        for residue in train_features['residue_labels']:
            amino_acid_counts[STANDARD_AMINO_ACIDS[residue]] += 1
        return train_features.copy(), self.validation_features, self.validation_indices


def get_chain_indices(dataset, validation_indices):
    validation_indices_by_chain = {}
    i = 0
    j = 0
    chains = list(dataset.chains.values())
    while i < len(chains) and j < len(validation_indices):
        chain = chains[i]
        validation_index = validation_indices[j]
        if validation_index > chain.end:
            i += 1
            continue
        if validation_index < chain.start:
            j += 1
            continue
        chain_index = validation_index - chain.start
        try:
            validation_indices_by_chain[chain.id] += [chain_index]
        except KeyError:
            validation_indices_by_chain[chain.id] = [chain_index]
        j += 1
    return validation_indices_by_chain


def sample_residues(sampling_mode, train_ratio, dataset, validation_path, seed):
    feature_sampler = Sampler(dataset, train_ratio, validation_path, seed)
    print('Sampling residues:\n-------------------------------')
    if sampling_mode == 'balanced':
        training_features, validation_features, validation_indices = feature_sampler.balanced()
    elif sampling_mode == 'unbalanced':
        training_features, validation_features, validation_indices = feature_sampler.unbalanced()
    else:
        raise ValueError(f"{sampling_mode} is not a valid sampling mode. Enter 'balanced' or 'unbalanced' instead.")
    print('Done sampling\n')
    validation_indices_by_chain = get_chain_indices(dataset, validation_indices)
    return training_features, validation_features, validation_indices_by_chain


def read_test_file(test_set_path):
    """reads single column of test chain ids"""
    if test_set_path is None:
        return None
    with open(test_set_path, "r") as file:
        test_chains = [line.strip('\n') for line in file]
    return test_chains


def exclude_test_chains(chain_ids, test_set_path):
    if not test_set_path:
        return chain_ids
    test_chains = read_test_file(test_set_path)
    chain_ids = [chain_id for chain_id in chain_ids if chain_id not in test_chains]
    return chain_ids


def read_chain_list(input_directory):
    with open(input_directory / "chain_list.txt") as file:
        chain_ids = [line.strip("\n") for line in file]
    return chain_ids


class Chain:
    __slots__ = ['id', 'features', 'start', 'end']

    def __init__(self, chain_id, feature_arrays, chain_start, neighbour_count):
        self.id = chain_id
        self.features = feature_arrays
        self.features = self.select_neighbours(neighbour_count)
        self.start = chain_start
        self.end = self.start + len(self)

    def __getitem__(self, feature):
        return self.features[feature]

    def __setitem__(self, feature, value):
        self.features[feature] = value

    def __len__(self):
        return len(self.features[FEATURE_LIST[0]])

    def __str__(self):
        return self.id

    def remove_residues(self, amino_acid):
        known_residues = self.features["residue_labels"] != AMINO_ACID_INDICES[amino_acid]
        new_feature_dictionary = {feature_name: array.compress(known_residues, axis=0) for feature_name, array in
                                  self.features.items()}
        return new_feature_dictionary

    def select_neighbours(self, neighbours):
        feature_lengths = {"residue_labels": None, "translations": neighbours, "rotations": neighbours,
                           "torsional_angles": neighbours + 1}
        if neighbours > (self.features['translations'].shape[1]):
            raise IndexError(
                f"The specified number of neighbours, {neighbours}, is greater than "
                f"{self.features['translations'].shape[1]}, "
                f"the number of neighbouring residues in the structural features.")
        new_feature_dictionary = {feature_name: array[:, :feature_lengths[feature_name]].copy()
        if not feature_name == 'residue_labels'
        else array
                                  for feature_name, array in self.features.items()}
        return new_feature_dictionary


class Dataset:
    def __init__(self, chain_ids, feature_directory, neighbour_count):
        self.chain_start = 0
        self.feature_directory = feature_directory
        self.array_paths = {}
        self.excluded_chains = []
        for chain_id in chain_ids:
            array_path = self.feature_directory / f'{chain_id}.npz'
            if not array_path.exists():
                print(f'{array_path} does not exist. Excluding chain {chain_id} from the dataset.')
                self.excluded_chains.append(chain_id)
                continue
            self.array_paths[chain_id] = array_path
        self.chains = {chain_id: self.load_chain(chain_id, array_path, self.chain_start, neighbour_count)
                       for chain_id, array_path in self.array_paths.items()}

    def __getitem__(self, feature):
        return np.concatenate([chain[feature] for chain in self.chains.values()])

    def __len__(self):
        return sum([len(chain) for chain in self.chains.values()])

    def load_chain(self, chain_id, array_path, chain_start, neighbour_count):
        with np.load(array_path) as arrays:
            chain = Chain(chain_id, arrays, chain_start, neighbour_count)
        self.chain_start = chain.end + 1
        return chain

    def remove_residues(self, amino_acid):
        for chain in self.chains.values():
            chain.features = chain.remove_residues(amino_acid)


def load_features(feature_directory, test_set_path, neighbours):
    chain_ids = read_chain_list(feature_directory)
    chain_ids = exclude_test_chains(chain_ids, test_set_path)
    dataset = Dataset(chain_ids, feature_directory, neighbours)
    dataset.remove_residues('X')
    return dataset


def train_model():
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format="%(message)s")
    logging.captureWarnings(False)

    input_directory, output_directory, sampling_mode, validation_path, train_ratio, test_set_path, epochs, neighbours, \
        layers, dropout, batch_size, seed = get_arguments()

    # torch tensors default on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create output directory
    if not output_directory.exists():
        output_directory.mkdir()

    dataset = load_features(input_directory, test_set_path, neighbours)
    if dataset.excluded_chains:
        with open(output_directory / 'excluded_chains.txt', 'w') as file:
            file.write('\n'.join(dataset.excluded_chains))

    training_features, validation_features, validation_indices_by_chain = sample_residues(sampling_mode,
                                                                                          train_ratio,
                                                                                          dataset,
                                                                                          validation_path,
                                                                                          seed)
    np.savez(output_directory / "validation_set.npz", **validation_indices_by_chain)

    training_features = {feature_name: torch.tensor(feature, dtype=torch.float64) for feature_name, feature in
                         training_features.items()}
    validation_features = {feature_name: torch.tensor(feature, dtype=torch.float64) for feature_name, feature in
                           validation_features.items()}

    # dataloader setup
    training_data = StructureDataset(training_features)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

    for data in train_dataloader:
        logging.debug(f'DEBUG - train_dataloader: {data}')
    if list(validation_features['residue_labels']):  # test if there are any residues in the validation set
        validation_data = StructureDataset(validation_features)
        validation_dataloader = DataLoader(validation_data, batch_size, shuffle=True)
    else:
        print('No validation set')

    # set model
    feature_size = training_data.examples.shape[1]
    model = NeuralNetwork(feature_size, layers, dropout).to(device)
    print(f'Model layout:\n{model.linear_relu_stack}\n')

    # setup optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train model
    train_loss_list = []
    validation_loss_list = []
    class_recall_dict = {amino_acid: [] for amino_acid in STANDARD_AMINO_ACIDS}
    accuracies = []
    print('Training model\n-------------------------------')
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        # train one epoch and get loss
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        train_loss_list.append(train_loss)
        # validate epoch if there is a validation set
        if list(validation_features['residue_labels']):
            validation_loss, accuracy, class_recall = validate_epoch(validation_dataloader, model, loss_fn, device)
            validation_loss_list.append(validation_loss)
            accuracies.append(accuracy)
            for amino_acid in class_recall:
                class_recall_dict[amino_acid] += [class_recall[amino_acid]]
            logging.debug(f'DEBUG - class_recall_dict: {class_recall_dict}')
    print("Done training\n")

    # evaluate model
    if list(validation_features['residue_labels']):
        print('Final model validation\n-------------------------------')
        # get list of predicted and true residues for the entire validation set
        true, pred = validate_final(validation_dataloader, model, device)
        # get per residue precision, recall and f1-scores
        report = classification_report(true, pred, target_names=STANDARD_AMINO_ACIDS, zero_division=0)
        print('Classification report:\n')
        print(report)
        print('Note: precision and F-scores are set to 0.0 for classes that have no predictions')
        kappa = cohen_kappa_score(true, pred)
        print(f'Cohen kappa score:{kappa}\n')
        with open(output_directory / 'report.txt', 'w') as file:
            file.write(report)
            file.write(f'Cohen kappa score: {kappa}\n')
            file.write('Note: precision and F-scores are set to 0.0 for classes that have no predictions')
        print('Generating graphs...\n')
        plot = Plots(output_directory)
        plot.learning_curve(epochs, accuracies, train_loss_list, validation_loss_list)
        plot.confusion_matrix(true, pred, None, 'unnormalised_conf_matrix')
        plot.confusion_matrix(true, pred, 'pred', 'pred_norm_conf_matrix')
        plot.confusion_matrix(true, pred, 'true', 'true_norm_conf_matrix')
        plot.class_learning_curve(epochs, class_recall_dict)

    torch.save(model.state_dict(), output_directory / "model_parameters.pth")
    print("Saved PyTorch Model State to model_parameters.pth")


if __name__ == "__main__":
    train_model()
