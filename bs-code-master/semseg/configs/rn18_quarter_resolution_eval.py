import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim
from pathlib import Path
import numpy as np
import os

from semseg.models.semseg import SemsegModel
from semseg.models.resnet.resnet_single_scale import *
from semseg.models.loss import SemsegCrossEntropy
from semseg.data.transform import *
from semseg.data.cityscapes import Cityscapes
from semseg.evaluation import StorePreds

from semseg.models.util import get_n_params

img_format = "ppm"
root = Path('./datasets/Cityscapes')
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

evaluating = True
random_crop_size = 448

scale = 1
mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
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
     Tensor(),
     ]
)

trans_train = trans_val

dataset_train = Cityscapes(root, transforms=trans_train, subset='train', format=img_format)
dataset_val = Cityscapes(root, transforms=trans_val, subset='val', format=img_format)

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
model = SemsegModel(resnet, num_classes)
model.load_state_dict(torch.load('weights/cityscapes/model_best.pt'))

batch_size = 14
print(f'Batch size: {batch_size}')

loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate)
loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
# store_dir = f'{dir_path}/out/'
# for d in ['', 'val', 'train', 'training']:
#     os.makedirs(store_dir + d, exist_ok=True)
# to_color = ColorizeLabels(color_info)
# to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
eval_observers = []
