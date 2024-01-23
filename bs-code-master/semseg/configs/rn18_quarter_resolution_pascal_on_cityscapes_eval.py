from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from pathlib import Path
from semseg.data.pascal.img_transforms import *
from semseg.models.resnet.resnet_single_scale import resnet18
from semseg.models.semseg import SemsegModel
from semseg.models.loss import SemsegCrossEntropy
from semseg.models.util import get_n_params

root = Path('./datasets/VOC2012').resolve()

mean = [73.15, 82.90, 72.3]
std = [47.67, 48.49, 47.73]
crop_size = 500
scale = 1.0
evaluating = True


eval_each = 4

trans_val = Compose([
    SquarePad(crop_size),
    ScaleToFit(crop_size, crop_size),
    ToTensors(),
])
#     NormalizeImage(mean, std)
# ])

trans_train = trans_val

class_info = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"] #"ignore"]
dataset_train = torchvision.datasets.VOCSegmentation(root, transforms=trans_train, image_set="train")
dataset_train.class_info = class_info

dataset_val = torchvision.datasets.VOCSegmentation(root, transforms=trans_val, image_set="val")
dataset_val.class_info = class_info

labels = set()
num_classes = 21 # 21 + [255]

resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
model = SemsegModel(resnet, num_classes)

model.load_state_dict(torch.load('weights/pascal/model_best.pt'))

batch_size = 14
print(f'Batch size: {batch_size}')


loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=4)
loader_val = DataLoader(dataset_val, batch_size=1)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
spp_params = get_n_params(model.backbone.spp.parameters())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')
print(f'SPP params: {spp_params:,}')

# from data.transform import *
eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
# store_dir = f'{dir_path}/out/'
# for d in ['', 'val', 'train', 'training']:
#     os.makedirs(store_dir + d, exist_ok=True)
# to_color = ColorizeLabels(color_info)
# to_image = Compose([DenormalizeTh(scale, mean, std), Numpy(), to_color])
eval_observers = []
