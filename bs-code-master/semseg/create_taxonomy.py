import argparse
from pathlib import Path
import numpy as np
from itertools import product
from random import shuffle
import random


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

def get_class_sums(confmat):
    """
    Get the sum of each row of the confusion matrix.
    """
    return confmat.sum(axis=1)

def get_class_totals(confmat):
    """
    Gets the sum of each row of the confusion matrix divided by the total sum of the matrix.
    """
    zz = get_class_sums(confmat) / confmat.sum()
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

def create_taxonomy(first, second, first_labels, second_labels, unnormalized_first, unnormalized_second) -> list[str]:
    """
    Create a universal taxonomy from two confusion matrices.

    WARNING: very ugly... sorry, no time...
    """
    EPS = 0.33
    DELTA = 0.5
    minimum_row = 0.1
    minimum_n = 5000

    first_totals = get_class_sums(unnormalized_first)
    second_totals = get_class_sums(unnormalized_second)

    n_first = len(first_labels)
    n_second = len(second_labels)
    print(first.shape, second.shape)
    scores = [(first[y, x] + second[x, y], (x, y)) for x, y in product(range(n_first), range(n_second))]


    best_score = 0
    best_str = ""

    for iteration in range(1000):
        points = []
        # scores = sorted(scores, key=lambda x: x[0], reverse=True)
        shuffle(scores)
        # scores = [(k, v) for k, v in sorted(scores, key=lambda x: x[0] + random.gauss(0, 0), reverse=True)]

        super_x = set()
        sub_x = set()
        super_y = set()
        sub_y = set()
        total_score = 0
        full_str = ""

        for score, (x, y) in scores:
            first_on_second = first[y, x]
            second_on_first = second[x, y]
            order = determine_order(first_on_second, second_on_first)
            # p = max([unnormalized_first[y, x]/first_totals[y], unnormalized_second[x, y]/second_totals[x]])
            n = min([unnormalized_first[y, x], unnormalized_second[x, y]])
            row_score = unnormalized_first[y, x]/first_totals[y] + unnormalized_second[x, y]/second_totals[x]
            if score > EPS and n > minimum_n and row_score > minimum_row and order_is_legal(order, x, y, super_x, super_y, sub_x, sub_y):
                full_str += f'{order} {round(score, 2)}: {first_labels[x]} ({unnormalized_first[y, x]}, {unnormalized_first[y, x]/first_totals[y]}) {second_labels[y]} ({unnormalized_second[x, y]}, {unnormalized_second[x, y]/second_totals[x]})'
                full_str += '\n'
                if order == 'super':
                    super_x.add(x)
                    sub_y.add(y)
                elif order == 'sub':
                    sub_x.add(x)
                    super_y.add(y)
                else:
                    super_x.add(x)
                    super_y.add(y)
                    sub_x.add(x)
                    sub_y.add(y)
                total_score += score
                points.append(score)
        # points = [x*(0.95)**i for i, x in enumerate(sorted(points))]
        # total_score = sum(points)
        if total_score > best_score:
            best_score = total_score
            best_str = full_str
        # results.append((total_score, full_str))
    # results = sorted(results, key=lambda x: x[0], reverse=True)
    # total_score, full_str = results[0]
    print(best_str)
    print(f'Total score: {best_score}')
    return []

def determine_order(first_on_second, second_on_first):
    """
    Determine the order of the labels.
    """
    DELTA = 0.5
    if first_on_second - second_on_first > DELTA:
        return 'sub'
    elif second_on_first - first_on_second > DELTA:
        return 'super'
    else:
        return 'equal'

def order_is_legal(order, x, y, super_x, super_y, sub_x, sub_y):
    """
    Check if the order is legal.
    'super' => x > y
    'sub' => x < y
    'equal' => x == y
    """
    if order == 'super':
        return x not in sub_x and y not in super_y
    elif order == 'sub':
        return x not in super_x and y not in sub_y
    else:
        return x not in super_x and y not in super_y and x not in sub_x and y not in sub_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates an universal taxonomy given two .npy files, and optionally saves it to a file.")
    parser.add_argument("first_confmat", type=str, help="Path to the .npy file containing the first confusion matrix.")
    parser.add_argument("second_confmat", type=str, help="Path to the .npy file containing the second confusion matrix.")

    parser.add_argument("first_labels", type=str, help="Path to the .txt file containing the first origin labels.")
    parser.add_argument("second_labels", type=str, help="Path to the .txt file containing the second origin labels.")
    parser.add_argument("--store_dir", type=str, default="", help="Path to the directory where the taxonomy should be stored.")
    args = parser.parse_args()

    first = np.load(args.first_confmat)
    second = np.load(args.second_confmat)
    first_labels = load_labels(args.first_labels)
    second_labels = load_labels(args.second_labels)

    first_norm = column_wise_normalisation(first)
    second_norm = column_wise_normalisation(second)

    first = np.around(first, decimals=7)
    second = np.around(second, decimals=7)

    taxonomy = create_taxonomy(first_norm, second_norm, first_labels, second_labels, first, second)

    if args.store_dir != "":
        # create dir if it doesn't exist:
        Path(args.store_dir).mkdir(parents=True, exist_ok=True)
        # save taxonomy to txt
        with open(args.store_dir + "/taxonomy.txt", 'w') as f:
            for line in taxonomy:
                f.write(line + "\n")
