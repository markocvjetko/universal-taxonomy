from typing import Any, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.datasets import VOCSegmentation
from swiftnet.data.pascal.labels import labels
from swiftnet.data.pascal.img_transforms import *
import torchvision.transforms.v2 as tt
import torch
import numpy
import torchvision

class_info = [label.name for label in labels if label.ignoreInEval is False]
print(class_info)
print(len(class_info))
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1

id_to_map = {id: i for i, id in map_to_id.items()}


class PascalVocSegmentation(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 21

    map_to_id = map_to_id
    id_to_map = id_to_map

    inst_map_to_id = inst_map_to_id

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, year='2012', image_set='train', download=False, epoch=None, scale=False, store_original_labels=False):
        self.root = root
        self.transforms = transforms
        self.epoch = epoch
        self.scale = scale
        self.store_original_labels = store_original_labels
        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        print(f'Num images: {len(self.dataset)}')
        # self.img_transforms = tt.Compose([
        #     tt.ToImage(),
        #     tt.ToDtype(torch.float32, scale=scale),
        # ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, labels = self.dataset[item]
        ret_dict = {}
        
        if self.store_original_labels:
            original_labels = labels
            original_labels = tt.PILToTensor()(original_labels)
            original_labels = original_labels.squeeze(0)
            ret_dict['original_labels'] = original_labels
        
        image, labels = self.transforms(image, labels)
        ret_dict['image'] = image
        ret_dict['labels'] = labels
        

        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        #print shapes
        # print('image', image.shape)
        # print('labels', labels.shape)


        return ret_dict
    
