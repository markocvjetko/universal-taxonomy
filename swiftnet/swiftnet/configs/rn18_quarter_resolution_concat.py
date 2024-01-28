import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torchvision
from torchvision.datasets import VOCSegmentation
from pathlib import Path
import semseg.data.pascal.img_transforms as pascal_transforms
from semseg.models.resnet.resnet_single_scale import resnet18
from semseg.models.semseg import SemsegModel
from semseg.models.loss import SemsegCrossEntropy
from semseg.models.util import get_n_params

import os
from torchvision.transforms import Compose
from semseg.data.transform import *
from semseg.data.cityscapes import Cityscapes

# normalized cityscapes
# Mean: [0.014904000148405585, 0.060534896942559535, 0.002467117778614234]
# Std: [0.8018479304353021, 0.8262910055664351, 0.8408671940132132]

# normalized pascal
# Mean: [-0.0048991938450714, 0.07353992167358789, 0.005024070849266047]
# Std: [0.9790862216972388, 0.9762667105174981, 1.0027963444590569]

# raw pascal
# Mean: [0.3005062982869836, 0.2945264388019076, 0.29664307101987875]
# Std: [0.2834332982221475, 0.27957888355908483, 0.28044281599040216]

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]




# Pascal VOC 2012

mean = [0.3141681690198871, 0.32625401858240366, 0.31242337085020083]
std = [0.2764082704312526, 0.27861887202239954, 0.2688446443241376]

crop_size = 448
scale = 1.0
evaluating = False
eval_each = 4

root = Path('./datasets/VOC2012').resolve()

trans_train = pascal_transforms.Compose([
    pascal_transforms.RandomScale(0.7, 1.3),
    pascal_transforms.RandomHorizontalFlip(0.5),
    pascal_transforms.SquarePad(crop_size),
    pascal_transforms.RandomCrop(crop_size),
    pascal_transforms.ToTensors(),
    pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    pascal_transforms.NormalizeImage(mean, std)
])
# ])

trans_val = pascal_transforms.Compose([
    pascal_transforms.SquarePad(crop_size),
    pascal_transforms.ScaleToFit(crop_size, crop_size),
    pascal_transforms.ToTensors(),
    pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    pascal_transforms.NormalizeImage(mean, std)
])
# ])

pascal_class_info = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]#, "ignore"]
pascal_dataset_train = VOCSegmentation(root, transforms=trans_train, image_set="train")
pascal_dataset_train.class_info = pascal_class_info

pascal_dataset_val = VOCSegmentation(root, transforms=trans_val, image_set="val")
pascal_dataset_val.class_info = pascal_class_info

################################################################################


# Cityscapes

img_format = "ppm"
root = Path('./datasets/Cityscapes')
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

random_crop_size = 448

scale = 1
mean = [79.02770252947538, 75.50128215213992, 77.71374061872375]
std = [39.72938032870023, 37.959327189427505, 40.54397609098902]
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

num_classes = Cityscapes.num_classes
ignore_id = Cityscapes.num_classes
class_info = Cityscapes.class_info
color_info = Cityscapes.color_info

target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)
target_size = (1024, 512)
target_size_feats = (1024 // 4, 512 // 4)

eval_each = 4


trans_val = Compose(
    [Open(),
     RemapLabels(Cityscapes.map_to_id, ignore_id=num_classes),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Normalize(scale, mean, std),
     Tensor(),
     ]
)


trans_train = Compose(
    [Open(),
        RemapLabels(Cityscapes.map_to_id, ignore_id=num_classes),
        RandomFlip(),
        RandomSquareCropAndScale(random_crop_size, ignore_id=num_classes, mean=mean_rgb),
        SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
        Normalize(scale, mean, std),
        Tensor(),
        ])

city_dataset_train = Cityscapes(root, transforms=trans_train, subset='train', format=img_format, map_style=False)
city_dataset_val = Cityscapes(root, transforms=trans_val, subset='val', format=img_format, map_style=False)


################################################################################

mean = [0, 0, 0]
std = [1, 1, 1]

# Concat datasets

dataset_train = ConcatDataset([pascal_dataset_train, city_dataset_train])
dataset_val = ConcatDataset([pascal_dataset_val, city_dataset_val])
num_classes = 21 + Cityscapes.num_classes # 21 + [255]

dataset_train.class_info = city_dataset_train.class_info + pascal_dataset_train.class_info
dataset_val.class_info = city_dataset_val.class_info + pascal_dataset_val.class_info

# RGB vs BGR?

################################################################################

# def printing_collate_fn(batch):
#     """
#     Collate function to display each image of batch with PIL
#     """
#     # # show torch tensor in  format 'image' as image:
#     import cv2
#     for image, label in batch:
#         cv2.imshow('image', image[0].cpu().numpy())
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         print(image.shape)


batch_size = 14
print(f'Batch size: {batch_size}')

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True,
                            drop_last=True)
loader_val = DataLoader(dataset_val, batch_size=1)

# labels = set()

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
model = SemsegModel(resnet, num_classes)

model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=255)
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

# PRE:
# Mean: [0.032302498699743186, 0.0709163110013919, 0.08100182990314386]
# Std: [0.8554194034180055, 0.8916231187924225, 0.8969771822544302]

# NEW:
# Mean: [0.054588794156013656, 0.04587130108902153, 0.011648617039231896]
# Std: [1.0151454552491004, 0.994430094863338, 0.9957478660326274]

# new_mean = [0, 0, 0]
# new_std = [0, 0, 0]

# for x, y in loader_train:
#     for i in range(3):
#         new_mean[i] += np.mean(x[i].numpy())
#         new_std[i] += np.std(x[i].numpy())

# for i in range(3):
#     new_mean[i] /= len(loader_train)
#     new_std[i] /= len(loader_train)

# print(f'Mean: {new_mean}')
# print(f'Std: {new_std}')

# exit(1)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

if evaluating:
    # from data.transform import *
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    # store_dir = f'{dir_path}/out/'
    # for d in ['', 'val', 'train', 'training']:
    #     os.makedirs(store_dir + d, exist_ok=True)
    # to_color = ColorizeLabels(color_info)
    # to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
    # eval_observers = [StorePreds(store_dir, to_image, to_color)]
