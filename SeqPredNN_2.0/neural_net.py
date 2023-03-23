import torch
from torch import nn
from torch.utils.data import Dataset


class StructureDataset(Dataset):
    """
    Stores example feature data and the associated labels for training a neural network
    """
    def __init__(self, features):
        self.labels = features['residue_labels'].long()
        self.example_features = [torch.flatten(features[key], start_dim=1)
                                 for key in ["translations", "rotations", "torsional_angles"]]
        self.examples = torch.cat(self.example_features, dim=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    """
    Defines the neural network architecture
    """
    def __init__(self, input_nodes, network_shape, dropout):
        dtype = torch.float64
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(input_nodes, network_shape[0], dtype=dtype),
                  nn.ReLU(),
                  nn.Dropout(p=dropout)]
        for i in range(len(network_shape) - 1):
            layers += [nn.Linear(network_shape[i], network_shape[i + 1], dtype=dtype),
                       nn.ReLU(),
                       nn.Dropout(p=dropout)]
        layers += [nn.Linear(network_shape[-1], 20, dtype=dtype)]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        Feed an input through the neural network and return the network output

        Returns:
            output_values: a torch one dimensional torch tensor of length 20, containing the 20 float values output by
            the neural network for input x
        """
        x = self.flatten(x)
        output_values = self.linear_relu_stack(x)
        return output_values
