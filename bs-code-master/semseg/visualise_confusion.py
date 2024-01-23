import argparse
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def row_wise_normalisation(confmat):
    """
    Normalise the confusion matrix row-wise.
    """
    # add 0.01 to every cell to avoid division by zero
    # confmat = confmat + 0.00001
    row_sums = confmat.sum(axis=1)
    confmat = confmat / row_sums[:, np.newaxis]
    return confmat

def column_wise_normalisation(confmat):
    """
    Normalise the confusion matrix column-wise.
    """
    # add 1 to every cell to avoid division by zero
    # confmat = confmat + 0.00001
    col_sums = confmat.sum(axis=0)
    confmat = confmat / col_sums[np.newaxis, :]
    return confmat

def row_and_column_wise_normalisation(confmat):
    """
    Divide each cell by the total sum of the matrix.
    """
    confmat = confmat.astype(np.float32)
    confmat = confmat / confmat.sum()
    return confmat

def get_class_totals(confmat):
    """
    Gets the sum of each row of the confusion matrix divided by the total sum of the matrix.
    """
    zz = confmat.sum(axis=1) / confmat.sum()
    return zz


def weigh_by_class_totals(confmat, class_totals):
    """
    Weigh each row of the confusion matrix by the class totals.
    """
    confmat = confmat.astype(np.float32)
    confmat = confmat / class_totals[:, np.newaxis]
    return confmat

def load_labels(file):
    """
    Load the labels from a file, where the label is the first column.
    """
    labels = []
    with open(file, 'r') as f:
        for line in f:
            labels.append(' '.join(line.split(' ')[:-1]))
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualises a confusion matrix given the .npy file, and optionally saves it to a file.")
    parser.add_argument("conf_mat_path", type=str, help="Path to the .npy file containing the confusion matrix.")
    parser.add_argument("model_labels", type=str, help="Path to the .txt file containing the model labels.")
    parser.add_argument("dataset_labels", type=str, help="Path to the .txt file containing the dataset labels.")
    parser.add_argument("--store_dir", type=str, default="", help="Path to the directory where the image should be stored.")
    args = parser.parse_args()

    conf_mat = np.load(args.conf_mat_path)
    class_totals = get_class_totals(conf_mat)
    # conf_mat = row_wise_normalisation(conf_mat)
    conf_mat = column_wise_normalisation(conf_mat)
    # conf_mat = weigh_by_class_totals(conf_mat, class_totals)
    conf_mat = np.around(conf_mat, decimals=7)
    print(conf_mat.shape, conf_mat[0])

    model_labels = load_labels(args.model_labels)
    dataset_labels = load_labels(args.dataset_labels)

    fig, ax = plt.subplots(figsize=(60,30))
    figure = sns.heatmap(conf_mat, xticklabels=model_labels, yticklabels=dataset_labels, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if args.store_dir != "":
        # create dir if it doesn't exist:
        Path(args.store_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(args.store_dir + '/confusion_matrix.png')

    # plt.show()
