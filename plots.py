import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np

classes = ('GLY', 'ALA', 'CYS', 'PRO', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'SER', 'THR', 'ASN', 'GLN',
           'TYR', 'ASP', 'GLU', 'HIS', 'LYS', 'ARG')


class Plots:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def learning_curve(self, epochs, accuracies, train_loss, validation_loss):
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
        plt.xlabel('Epoch',  fontsize=70, labelpad=24)
        ax.set_ylabel('Accuracy',  fontsize=70, labelpad=24)
        ax2.set_ylabel('Loss', fontsize=70, labelpad=24)
        plt.tight_layout()
        plt.savefig(self.out_dir / 'learn_curve.png')

    def confusion_matrix(self, true_residues, predicted_residues, normalize, file_name):
        matrix = metrics.confusion_matrix(true_residues, predicted_residues, normalize=normalize)
        fig, ax = plt.subplots(figsize=(42, 35))
        plt.rcParams['font.size'] = 32
        img = ax.imshow(matrix)
        ax.set_xticks(np.arange(20))
        ax.set_yticks(np.arange(20))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.tick_params(labelsize=63, pad=14, length=14, width=3)
        ax.set_xlabel('Predicted residue', fontsize=84, labelpad=24)
        ax.set_ylabel('True residue', fontsize=84, labelpad=24)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(20):
            for j in range(20):
                ax.text(j, i, round(matrix[i, j], 3), ha="center", va="center", color="w")
        cbar = fig.colorbar(img, ax=ax, pad=0.01)
        cbar.ax.tick_params(labelsize=63, pad=14, length=14, width=3)
        fig.tight_layout()
        plt.savefig(self.out_dir / file_name)
        plt.close()