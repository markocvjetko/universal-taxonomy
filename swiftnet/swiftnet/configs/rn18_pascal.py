from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from swiftnet.data.pascal.pascal import PascalVocSegmentation
from swiftnet.data.pascal.img_transforms import *
from swiftnet.models.resnet.resnet_single_scale import resnet18
from swiftnet.models.semseg import SemsegModel
from swiftnet.models.loss import SemsegCrossEntropy
from swiftnet.models.util import get_n_params
from swiftnet.data.transform import custom_collate

#root = Path('/home/mc/dipl-rad/data/voc/').resolve()
root = Path('/scratch/markoc-haeslerlab/msc-thesis/pascal/')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# this was used in the concat
# mean = [0.3141681690198871, 0.32625401858240366, 0.31242337085020083]
# std = [0.2764082704312526, 0.27861887202239954, 0.2688446443241376]

crop_size = 500
scale = 1.0
evaluating = False


eval_each = 4

trans_train = Compose([
    RandomScale(0.7, 1.3),
    RandomHorizontalFlip(0.5),
    SquarePad(crop_size),
    RandomCrop(crop_size),
    ToTensors(),
])

trans_val = Compose([
    #SquarePad(crop_size),
    #ScaleToFit(crop_size, crop_size),
    ToTensors(),
])

dataset_train = PascalVocSegmentation(root=root, transforms=trans_train, image_set='train')
dataset_val = PascalVocSegmentation(root=root, transforms=trans_val, image_set='val', store_original_labels=True)

#dataset_train = torch.utils.data.Subset(dataset_train, range(100))

print(dataset_train[0]['image'].max())
print(dataset_train[0]['image'].min())

labels = set()
num_classes = 21 # 21 + [255]

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

batch_size = 14
print(f'Batch size: {batch_size}')

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True,
                            drop_last=True,)
                            #collate_fn=custom_collate)
loader_val = DataLoader(dataset_val, batch_size=1,) #collate_fn=custom_collate)

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
