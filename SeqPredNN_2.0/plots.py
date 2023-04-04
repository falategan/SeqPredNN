import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from constants import STANDARD_AMINO_ACIDS


class Plots:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def learning_curve(self, epochs, accuracies, train_loss, validation_loss):
        """
        Plot and save a learning curve, a line graph of the model loss on the training set and the validation set, and the model
        accuracy on the valaidation set against the number of epochs during training.
        """
        epoch_list = [*range(1, epochs + 1)]
        fig, ax = plt.subplots(figsize=(42, 28))
        ax2 = ax.twinx()
        ax.tick_params(labelsize=56, pad=14, length=14, width=3)
        ax2.tick_params(labelsize=56, pad=14, length=14, width=3)
        accuracy_curve = ax.plot(epoch_list, accuracies, color='g', label='Validation accuracy', linewidth=5)
        train_curve = ax2.plot(epoch_list, train_loss, color='b', label='Train loss')
        validation_curve = ax2.plot(epoch_list, validation_loss, color='r', label='Validation loss', linewidth=5)
        curves = accuracy_curve + train_curve + validation_curve
        labels = [curve.get_label() for curve in curves]
        plt.legend(curves, labels, loc='center right', fontsize=56)
        plt.xlabel('Epoch', fontsize=70, labelpad=24)
        ax.set_ylabel('Accuracy', fontsize=70, labelpad=24)
        ax2.set_ylabel('Loss', fontsize=70, labelpad=24)
        plt.tight_layout()
        plt.savefig(self.out_dir / 'learn_curve.png')
        with open(self.out_dir / 'learn_curve.csv', 'w') as file:
            file.write("Epoch, Accuracy, Loss\n")
            for epoch, accuracy, loss in zip(epoch_list, accuracies, validation_loss):
                file.write(f'{epoch},{accuracy},{loss}\n')

    def confusion_matrix(self, true_residues, predicted_residues, normalize, file_name):
        """
        Plot and save confusion matrices that are unnormalised, normalised by the true residue and normalised by the
        predicted residue
        """
        matrix = metrics.confusion_matrix(true_residues, predicted_residues, normalize=normalize)
        fig, ax = plt.subplots(figsize=(42, 35))
        plt.rcParams['font.size'] = 32
        img = ax.imshow(matrix)
        ax.set_xticks(np.arange(20))
        ax.set_yticks(np.arange(20))
        ax.set_xticklabels(STANDARD_AMINO_ACIDS)
        ax.set_yticklabels(STANDARD_AMINO_ACIDS)
        ax.tick_params(labelsize=63, pad=14, length=14, width=3)
        ax.set_xlabel('Predicted residue', fontsize=84, labelpad=24)
        ax.set_ylabel('True residue', fontsize=84, labelpad=24)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        with open(self.out_dir / f'{file_name}.csv', 'w') as file:
            file.write("Predicted residue, True residue, Value\n")
            for i in range(20):
                for j in range(20):
                    ax.text(j, i, round(matrix[i, j], 3), ha="center", va="center", color="w")
                    file.write(f'{STANDARD_AMINO_ACIDS[i]},{STANDARD_AMINO_ACIDS[j]},{matrix[i, j]}\n')
        cbar = fig.colorbar(img, ax=ax, pad=0.01)
        cbar.ax.tick_params(labelsize=63, pad=14, length=14, width=3)
        fig.tight_layout()
        plt.savefig(self.out_dir / f'{file_name}.png')
        plt.close()

    def class_learning_curve(self, epochs, class_recall):
        """
        Plot and save a learning curve that shows the recall for each amino acid class in the validation set for each
        epoch during training
        """
        epoch_list = [*range(1, epochs + 1)]
        fig, ax = plt.subplots(figsize=(42, 28))
        ax.tick_params(labelsize=56, pad=14, length=14, width=3)
        for amino_acid, recall in class_recall.items():
            ax.plot(epoch_list, recall, label=f'{amino_acid} recall', linewidth=5)
        for line, amino_acid in zip(ax.lines, STANDARD_AMINO_ACIDS):
            ax.annotate(amino_acid, xy=(epochs+1, class_recall[amino_acid][-1]), color=line.get_color(),
                        size=32, va="center")
        plt.xlabel('Epoch', fontsize=70, labelpad=24)
        ax.set_ylabel('Recall', fontsize=70, labelpad=24)
        plt.tight_layout()
        plt.savefig(self.out_dir / 'class_learn_curve.png')
        with open(self.out_dir / 'class_learn_curve.csv', 'w') as file:
            file.write(f'Epoch, {",".join(STANDARD_AMINO_ACIDS)}\n')
            for epoch in epoch_list:
                file.write(f'{epoch},{",".join([str(class_recall[amino_acid][epoch-1]) for amino_acid in STANDARD_AMINO_ACIDS])}\n')
