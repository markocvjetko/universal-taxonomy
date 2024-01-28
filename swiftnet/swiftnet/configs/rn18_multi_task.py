import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from pathlib import Path
import swiftnet.data.pascal.img_transforms as pascal_transforms
from swiftnet.models.resnet.resnet_single_scale import resnet18
from swiftnet.models.semseg_multi_task import SemsegModelMultiTask
from swiftnet.models.loss import SemsegCrossEntropy
from swiftnet.models.util import get_n_params
from swiftnet.data.transform import custom_collate
from swiftnet.data.transform.base import custom_collate_new
from swiftnet.data.pascal.pascal import PascalVocSegmentation
import os
from torchvision.transforms import Compose
from swiftnet.data.transform import *
from swiftnet.data.cityscapes import Cityscapes
from swiftnet.data.pascal import pascal
import torch
from torch.utils.data.dataloader import default_collate

from swiftnet.data.multi_task.multi_task import VOCSegmentationWrapper, CityscapesWrapper
# from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap

#use cuda 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

crop_size = 448
scale = 1.0
evaluating = False
eval_each = 4

pascal_num_classes = PascalVocSegmentation.num_classes
pascal_ignore_id = 255
pascal_class_info = PascalVocSegmentation.class_info
pascal_color_info = PascalVocSegmentation.color_info

#root = Path('/home/mc/dipl-rad/data/voc/').resolve()
root = Path('/scratch/markoc-haeslerlab/msc-thesis/pascal/')

trans_train = pascal_transforms.Compose([
    pascal_transforms.RandomScale(0.7, 1.3),
    pascal_transforms.RandomHorizontalFlip(0.5),
    pascal_transforms.SquarePad(crop_size),
    pascal_transforms.RandomCrop(crop_size),
    pascal_transforms.ToTensors(),
    #pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    #pascal_transforms.NormalizeImage(mean, std)
])
# ])

trans_val = pascal_transforms.Compose([
    pascal_transforms.ToTensors(),
    #pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    #pascal_transforms.NormalizeImage(mean, std)
])
# ])

pascal_dataset_train = VOCSegmentationWrapper(root=root, transforms=trans_train, image_set='train')
pascal_dataset_val = VOCSegmentationWrapper(root=root, transforms=trans_val, image_set='val', store_original_labels=True)


################################################################################


# Cityscapes

root = Path('/scratch/markoc-haeslerlab/msc-thesis/cityscapes/')
            
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

random_crop_size = 448

scale = 1
mean = [73.15, 82.90, 72.3]             
std = [47.67, 48.49, 47.73]  

mean_rgb = tuple(np.uint8(scale * np.array(mean)))

cityscapes_num_classes = Cityscapes.num_classes
cityscapes_ignore_id = Cityscapes.num_classes
cityscapes_class_info = Cityscapes.class_info
cityscapes_color_info = Cityscapes.color_info

target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)
target_size = (1024, 512)
target_size_feats = (1024 // 4, 512 // 4)

eval_each = 4


trans_val = Compose(
    [Open(),
     RemapLabels(Cityscapes.map_to_id, ignore_id=cityscapes_num_classes),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     #Normalize(scale, mean, std),
     Tensor(),
     ]
)


trans_train = Compose(
    [Open(copy_labels=False),
        RemapLabels(Cityscapes.map_to_id, ignore_id=cityscapes_num_classes),
        RandomFlip(),
        RandomSquareCropAndScale(random_crop_size, ignore_id=cityscapes_num_classes, mean=mean_rgb),
        SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
        #Normalize(scale, mean, std),
        Tensor(),
        ])

city_dataset_train = CityscapesWrapper(root, transforms=trans_train, subset='train', )
city_dataset_val = CityscapesWrapper(root, transforms=trans_val, subset='val')


################################################################################

mean = [0, 0, 0]
std = [1, 1, 1]

# Concat datasets

dataset_train = ConcatDataset([pascal_dataset_train, city_dataset_train])
dataset_val = ConcatDataset([pascal_dataset_val, city_dataset_val])
num_classes = 21 + Cityscapes.num_classes # 21 + [255]

dataset_train.class_info = city_dataset_train.class_info + pascal_dataset_train.class_info
dataset_val.class_info = city_dataset_val.class_info + pascal_dataset_val.class_info

#random 100 samples of train as subset
# random_indices = np.random.choice(len(dataset_train), 100, replace=False)
# dataset_train = torch.utils.data.Subset(dataset_train, random_indices)

# RGB vs BGR?

################################################################################


batch_size = 20
print(f'Batch size: {batch_size}')


loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=custom_collate_new)
loader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, collate_fn=custom_collate_new)

# labels = set()

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
model = SemsegModelMultiTask(resnet, pascal_num_classes, cityscapes_num_classes)

model.criterion1 = SemsegCrossEntropy(num_classes=pascal_num_classes, ignore_id=pascal_ignore_id)
model.criterion2 = SemsegCrossEntropy(num_classes=cityscapes_num_classes, ignore_id=cityscapes_ignore_id)
lr = 4e-4
lr_min = 1e-6
fine_tune_factor = 4
weight_decay = 1e-4
epochs = 250

optim_params = [
    {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
    {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
        'weight_decay': weight_decay / fine_tune_factor},
]

optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)


total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

if evaluating:
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
