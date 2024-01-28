import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from pathlib import Path
import swiftnet.data.pascal.img_transforms as pascal_transforms
from swiftnet.models.resnet.resnet_single_scale import resnet18
from swiftnet.models.semseg import SemsegModel
from swiftnet.models.loss import SemsegCrossEntropy
from swiftnet.models.util import get_n_params
from swiftnet.data.transform import custom_collate
from swiftnet.data.transform.base import custom_collate_new
from swiftnet.data.pascal.pascal import PascalVocSegmentation
import os
from torchvision.transforms import Compose
from swiftnet.data.transform import *
from swiftnet.data.transform.labels import RemapLabels2
from swiftnet.data.cityscapes import Cityscapes
from torch import nn
import torch
from torch.utils.data.dataloader import default_collate

#use cuda 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pascal_mean = [0.3141681690198871, 0.32625401858240366, 0.31242337085020083]
pascal_std = [0.2764082704312526, 0.27861887202239954, 0.2688446443241376]

#imagenet mean std
# pascal_mean = [0.485, 0.456, 0.406]
# pascal_std = [0.229, 0.224, 0.225]

pascal_crop_size = 448
pascal_scale = 1.0
evaluating = False
eval_each = 4

#root = Path('/home/mc/dipl-rad/data/voc/').resolve()
root = Path('/scratch/markoc-haeslerlab/msc-thesis/pascal/')

trans_train = pascal_transforms.Compose([
    pascal_transforms.RandomScale(0.7, 1.3),
    pascal_transforms.RandomHorizontalFlip(0.5),
    pascal_transforms.SquarePad(pascal_crop_size),
    pascal_transforms.RandomCrop(pascal_crop_size),
    pascal_transforms.ToTensors(),
    pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    pascal_transforms.NormalizeImage(pascal_mean, pascal_std)
])
# ])

trans_val = pascal_transforms.Compose([
    pascal_transforms.SquarePad(pascal_crop_size),
    pascal_transforms.ScaleToFit(pascal_crop_size, pascal_crop_size),
    pascal_transforms.ToTensors(),
    pascal_transforms.OffsetLabels(Cityscapes.num_classes),
    pascal_transforms.NormalizeImage(pascal_mean, pascal_std)
])
# ])

pascal_dataset_train = PascalVocSegmentation(root=root, transforms=trans_train, image_set='train', store_original_labels=False)

pascal_dataset_val = PascalVocSegmentation(root=root, transforms=trans_val, image_set='val', store_original_labels=True)

################################################################################


# Cityscapes

root = Path('/scratch/markoc-haeslerlab/msc-thesis/cityscapes/')
            
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

city_crop_size = pascal_crop_size

city_scale = 1
city_mean = [79.02770252947538, 75.50128215213992, 77.71374061872375]
city_std = [39.72938032870023, 37.959327189427505, 40.54397609098902]

#imagenet mean std 255
# city_scale = 1
# city_mean = [73.15, 82.90, 72.3]             # Imagenet parameters, adjust for different datasets and initialization
# city_std = [47.67, 48.49, 47.73]  

mean_rgb = tuple(np.uint8(city_scale * np.array(city_mean)))

city_num_classes = Cityscapes.num_classes
city_ignore_id = 255
city_class_info = Cityscapes.class_info
city_color_info = Cityscapes.color_info

city_target_size_crops = (city_crop_size, city_crop_size)
city_target_size_crops_feats = (city_crop_size // 4, city_crop_size // 4)
city_target_size = (1024, 512)
city_target_size_feats = (1024 // 4, 512 // 4)

eval_each = 4


trans_val = Compose(
    [Open(),
     #RemapLabels(Cityscapes.map_to_id, ignore_id=city_ignore_id),
     RemapLabels2(Cityscapes.map_to_id2),
     SetTargetSize(target_size=city_target_size, target_size_feats=city_target_size_feats),
     Normalize(city_scale, city_mean, city_std),
     Tensor(),
     ]
)


trans_train = Compose(
    [Open(copy_labels=False),
        RemapLabels2(Cityscapes.map_to_id2),
        RandomFlip(),
        RandomSquareCropAndScale(city_crop_size, ignore_id=255, mean=mean_rgb),
        SetTargetSize(target_size=city_target_size_crops, target_size_feats=city_target_size_crops_feats),
        Normalize(city_scale, city_mean, city_std),
        Tensor(),
        ])

city_dataset_train = Cityscapes(root, transforms=trans_train, subset='train')
city_dataset_val = Cityscapes(root, transforms=trans_val, subset='val')


################################################################################
scale = 1
mean = [0, 0, 0]
std = [1, 1, 1]

# Concat datasets

dataset_train = ConcatDataset([pascal_dataset_train, city_dataset_train])
dataset_val = ConcatDataset([pascal_dataset_val, city_dataset_val])
num_classes = PascalVocSegmentation.num_classes + Cityscapes.num_classes

#take random 100 samples from train
# sample_ids = np.random.choice(len(dataset_train), 100, replace=False)
# dataset_train = torch.utils.data.Subset(dataset_train, sample_ids)

dataset_train.class_info = pascal_dataset_train.class_info + city_dataset_train.class_info
dataset_val.class_info = pascal_dataset_val.class_info + city_dataset_val.class_info 

# RGB vs BGR?

################################################################################


batch_size = 14
print(f'Batch size: {batch_size}')

# def custom_collate(batch):
#     """
#     Collate function to display each image of batch with PIL
#     """
#     #take image, labels and original labels from batch
#     image = [item['image'] for item in batch]
#     labels = [item['labels'] for item in batch]
#     orig_labels = [item['original_labels'] for item in batch]
#     target_size = [item['target_size'] if 'target_size' in item else None for item in batch]

#     #use torch default collate
#     images = default_collate(images)
#     labels = default_collate(labels)    
#     orig_labels = default_collate(orig_labels)
#     target_size = default_collate(target_size)

#     #return batch with images, labels and original labels
#     return {'image': image, 'labels': labels, 'original_labels': orig_labels, 'target_size': target_size}


loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=custom_collate_new)
loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=custom_collate_new)

# loader_city_train = DataLoader(city_dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
# loader_pascal_train = DataLoader(pascal_dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
# loader_city_val = DataLoader(city_dataset_val, batch_size=1, shuffle=True)
# loader_pascal_val = DataLoader(pascal_dataset_val, batch_size=1, shuffle=True)

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
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

#model = MyDataParallel(model, device_ids=[0, 1, 2, 3])


if evaluating:
    # from data.transform import *
    eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    # store_dir = f'{dir_path}/out/'
    # for d in ['', 'val', 'train', 'training']:
    #     os.makedirs(store_dir + d, exist_ok=True)
    # to_color = ColorizeLabels(color_info)
    # to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
    # eval_observers = [StorePreds(store_dir, to_image, to_color)]

# unique_ids_pascal = set()
# j = 0
# for i in loader_pascal_train:
#     unique_ids_pascal.update(np.unique(i['labels'].numpy()))
#     j+=1
#     if j % 5 == 0:
#         print(f'Pascal: {j}')
# print(f'Unique ids pascal: {unique_ids_pascal}')

# unique_ids_city = set()

# j = 0
# for i in loader_city_train:
#     unique_ids_city.update(np.unique(i['labels'].numpy()))
#     j+=1
#     if j % 5 == 0:
#         print(f'City: {j}')

# print(f'Unique ids city: {unique_ids_city}')


# unique_ids_pascal = set()
# j = 0
# for i in loader_pascal_val:
#     unique_ids_pascal.update(np.unique(i['labels'].numpy()))
#     j+=1
#     # llllllll#     print(f'Pascal: {j}')
# print(f'Unique ids pascal: {unique_ids_pascal}')

# unique_ids_city = set()

# j = 0
# for i in loader_city_val:
#     unique_ids_city.update(np.unique(i['labels'].numpy()))
#     j+=1
#     # if j % 5 == 0:
#     #     print(f'City: {j}')

# print(f'Unique ids city: {unique_ids_city}')

#print min and max of the first sample image of every dataset
print(f'Pascal min: {torch.min(pascal_dataset_train[0]["image"])}')
print(f'Pascal max: {torch.max(pascal_dataset_train[0]["image"])}')
print(f'City min: {torch.min(city_dataset_train[0]["image"])}')
print(f'City max: {torch.max(city_dataset_train[0]["image"])}')

print(f'Pascal min: {torch.min(pascal_dataset_val[0]["image"])}')
print(f'Pascal max: {torch.max(pascal_dataset_val[0]["image"])}')
print(f'City min: {torch.min(city_dataset_val[0]["image"])}')
print(f'City max: {torch.max(city_dataset_val[0]["image"])}')