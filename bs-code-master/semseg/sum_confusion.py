import argparse
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sums two confusion matrices and saves them to a file.")
    parser.add_argument("conf_mat_path_first", type=str, help="Path to the first .npy file containing the confusion matrix.")
    parser.add_argument("conf_mat_path_second", type=str, help="Path to the second .npy file containing the confusion matrix.")
    parser.add_argument("--store_dir", type=str, help="Path to the directory where the image should be stored.", required=True)
    args = parser.parse_args()

    save_location = Path(args.store_dir) / 'sum_confmat.npy'


    conf_mat = np.load(args.conf_mat_path_first)
    conf_mat_second = np.load(args.conf_mat_path_second)
    conf_mat = conf_mat + conf_mat_second

    np.save(save_location, conf_mat)

