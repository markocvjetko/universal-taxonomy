import argparse
import importlib.util
from semseg.evaluation import evaluate_semseg
from pathlib import Path
import numpy as np

def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_taxonomy(file, class_info, split_on=19):
    """
    Loads the taxonomy from a file. The format is:
    super|equal|sub, first_class, second_class
    """
    taxonomy = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                order, first, second = line.split(',')
                first = split_on + class_info[split_on:].index(first.strip())
                second = class_info[:split_on].index(second.strip())
                taxonomy[(first, second)] = order
    return taxonomy

def _taxonomy_transform(taxonomy, logits, target, split_on=19):
    """
    Transforms the logits using the taxonomy.
    The logits are a 3d tensor.
    A super B -> we add B's logits to A's logits.
    A equal B -> we add both to eachother
    A sub B -> we add A's logits to B's logits.
    A is 19:
    B is :19
    """
    # copy logits
    transformed = np.copy(logits)
    max_in_target = np.max(target)
    min_first_key = min(taxonomy.keys())[0]
    # iterate over the taxonomy
    for (first, second), order in taxonomy.items():
        if order in ['super'] and split_on > max_in_target:
            transformed[:, second] += transformed[:, first]
            transformed[:, first] = 0
        elif order in ['sub'] and split_on < max_in_target:
            transformed[:, first] += transformed[:, second]
            transformed[:, second] = 0
    return transformed


parser = argparse.ArgumentParser()
parser.add_argument('config_model', type=str, help='Path to the model configuration .py file')
parser.add_argument('config_dataset', type=str, help='Path to the dataset configuration .py file')
parser.add_argument('--store_dir', type=str, help='Path to the directory where the confusion matrix should be stored', required=False)
parser.add_argument('--use_taxonomy', type=str, help='Path to the taxonomy txt file', required=False)
parser.add_argument('--concat', type=bool, help='Run in concat model mode', required=False, default=False)
parser.add_argument('--miou', type=bool, help='Run in concat model mode', required=False, default=False)
parser.add_argument('--concat_split', type=int, help='Split number for concat model mode', required=False, default=0)

if __name__ == '__main__':
    args = parser.parse_args()

    conf_model = import_module(args.config_model)
    conf_dataset = import_module(args.config_dataset)

    class_info = conf_dataset.dataset_val.class_info

    if args.use_taxonomy:
        taxonomy = load_taxonomy(args.use_taxonomy, class_info)
        taxonomy_transform = lambda logits, target: _taxonomy_transform(taxonomy, logits, target)
    else:
        taxonomy_transform = lambda _ : _

    model = conf_model.model.cuda()
    for loader, name in conf_dataset.eval_loaders:
        if args.concat:
            if args.miou:
                iou_acc, per_class_iou = evaluate_semseg(model, loader, class_info, observers=conf_dataset.eval_observers, return_conf_mat=False, split_on=args.concat_split, taxonomy_transform=taxonomy_transform)
            else:
                confusion = evaluate_semseg(model, loader, class_info, observers=conf_dataset.eval_observers, return_conf_mat=True, split_on=args.concat_split, taxonomy_transform=taxonomy_transform)
        else:
            if args.miou:
                iou_acc, per_class_iou = evaluate_semseg(model, loader, class_info, observers=conf_dataset.eval_observers, return_conf_mat=False, taxonomy_transform=taxonomy_transform)
            else:
                confusion = evaluate_semseg(model, loader, class_info, observers=conf_dataset.eval_observers, return_conf_mat=(not args.miou), taxonomy_transform=taxonomy_transform)
        if args.store_dir and not args.miou:
            store_path = Path(args.store_dir) / f'{name}_confmat.npy'
            np.save(store_path, confusion)
            print(f'Confusion matrix saved to {store_path}')
        print()
