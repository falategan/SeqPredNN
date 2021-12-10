import torch
from torch import nn
from torch.utils.data import Dataset


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
