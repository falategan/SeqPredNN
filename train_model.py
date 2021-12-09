import torch
import pathlib
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, cohen_kappa_score
from sampling import Sampling
from plots import Plots

BATCH_SIZE = 4096


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_dir', type=str, help='input directory containing structural feature files.')
    arg_parser.add_argument('sampling_mode', type=str, choices=['balanced', 'unbalanced'],
                            help='select sampling mode. "balanced" undersamples the data so that each amino acid class '
                                 'occurs the same number of times. "unbalanced" samples the entire dataset')
    arg_parser.add_argument('-r', '--train_ratio', type=float, default=0.8,
                            help='the fraction of residues assigned to the training set. '
                                 'The rest is assigned to the validation set. Default: 0.8')
    arg_parser.add_argument('-t', '--test_set', type=str,
                            help='path for the list of chains to be excluded from the training and validation sets. '
                                 'Default: None')
    arg_parser.add_argument('-o', '--out_dir', type=str, default='./model',
                            help='output directory. Creates a new directory if OUT_DIR does not exist. '
                                 'Default: ./model')
    arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                            help='number of times the dataset is passed to through the model during training. '
                                 'Default: 100')
    args = arg_parser.parse_args()
    if args.train_ratio <= 0 or args.train_ratio > 1:
        raise ValueError('The train ratio must be a number between 0 and 1')
    return pathlib.Path(args.input_dir), args.sampling_mode, args.train_ratio, args.test_set, \
           pathlib.Path(args.out_dir), args.epochs


class StructureDataset(Dataset):
    def __init__(self, features):
        self.labels = features['residue_labels'].long()
        self.example_features = [torch.flatten(features[key], start_dim=1) for key in ['displacements', 'rotations',
                                                                                       'torsional_angles']]
        self.examples = torch.cat(self.example_features, dim=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_nodes, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 20)
        )
        print('Model layout:\n', self.linear_relu_stack, '\n')

    def forward(self, x):
        x = self.flatten(x)
        output_values = self.linear_relu_stack(x)
        return output_values


def train(dataloader, model, loss_fn, optimizer):
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


def validate_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            pred = model(batch_inputs)
            validation_loss += loss_fn(pred, batch_labels).item()
            correct += (pred.argmax(1) == batch_labels).type(torch.float).sum().item()
    validation_loss /= size
    correct /= size
    print(f"Accuracy: {(100 * correct):>.2f}%, Average loss: {validation_loss:>8f} \n")
    return validation_loss, 100*correct


def validate_final(dataloader, model):
    pred = []
    true = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            output = model(inputs.to(device))  # Feed Network
            output = (torch.max(output, 1)[1]).data.cpu().numpy()  # take output node with largest value
            pred.extend(output)
            labels = labels.data.int().cpu().numpy()
            true.extend(labels)
    return true, pred


if __name__ == "__main__":
    input_dir, sampling_mode, train_ratio, test_set, out_dir, epochs, = get_args()

    # create output directory
    if not out_dir.exists():
        out_dir.mkdir()

    # torch tensors default on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sample data
    feature_sampler = Sampling(input_dir, test_set, train_ratio)
    print('Sampling residues:\n-------------------------------')
    if sampling_mode == 'balanced':
        training_features, validation_features = feature_sampler.balanced()
    elif sampling_mode == 'unbalanced':
        training_features, validation_features = feature_sampler.unbalanced()
    print('Done sampling\n')

    # setup dataloader
    training_data = StructureDataset(training_features)
    train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
    if list(validation_features['residue_labels']):  # test if there are any residues in the validation set
        validation_data = StructureDataset(validation_features)
        validation_dataloader = DataLoader(validation_data, BATCH_SIZE, shuffle=True)
    else:
        print('No validation set')

    # set model
    feature_size = training_data.examples.shape[1]
    model = NeuralNetwork(feature_size).to(device)

    # setup optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train model
    train_loss_list = []
    validation_loss_list = []
    accuracies = []
    print('Training model\n-------------------------------')
    for t in range(epochs):
        print(f"Epoch {t+1}")
        # train one epoch and get loss
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        train_loss_list.append(train_loss)
        # validate epoch if there is a validation set
        if list(validation_features['residue_labels']):
            validation_loss, accuracy = validate_epoch(validation_dataloader, model, loss_fn)
            validation_loss_list.append(validation_loss)
            accuracies.append(accuracy)
    print("Done training\n")

    # evaluate model
    if list(validation_features['residue_labels']):
        print('Final model validation\n-------------------------------')
        # get list of predicted and true residues for the entire validation set
        pred, true = validate_final(validation_dataloader, model)
        classes = ('GLY', 'ALA', 'CYS', 'PRO', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'SER', 'THR', 'ASN', 'GLN',
                   'TYR', 'ASP', 'GLU', 'HIS', 'LYS', 'ARG')
        # get per residue precision, recall and f1-scores
        report = classification_report(true, pred, target_names=classes)
        print('Classification report:\n')
        print(report)
        kappa = cohen_kappa_score(true, pred)
        print('Cohen kappa score:', kappa, '\n')
        with open(out_dir / 'report.txt', 'w') as file:
            file.write(report)
            file.write('Cohen kappa score: ' + str(kappa))

        print('Generating graphs...\n')
        plot = Plots(out_dir)
        plot.learning_curve(epochs, accuracies, train_loss_list, validation_loss_list)
        plot.confusion_matrix(true, pred, None, 'unnormalised_conf_matrix.png')
        plot.confusion_matrix(true, pred, 'pred', 'pred_norm_conf_matrix.png')
        plot.confusion_matrix(true, pred, 'true', 'true_norm_conf_matrix.png')

    torch.save(model.state_dict(), out_dir / "model_parameters.pth")
    print("Saved PyTorch Model State to model_parameters.pth")
