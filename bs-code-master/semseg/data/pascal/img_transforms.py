import math
import PIL
from random import randint
import torch
import torchvision.transforms
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomScale:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, image, targets):
        scale = randint(self.lower*100, self.upper*100) / 100.0

        width = round(scale * image.width)
        height = round(scale * image.height)

        image = image.resize((width, height), PIL.Image.BILINEAR)
        targets = targets.resize((width, height), PIL.Image.NEAREST)

        return image, targets


class NormalizeImage(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def __call__(self, image, target):
        image = super().__call__(image)
        return image, target


class SquarePad:
    def __init__(self, crop_size):
        self.crop = crop_size

    def pad(self, x, value):
        w, h = x.size
        max_wh = np.max([w, h, self.crop])
        hp = int(math.ceil((max_wh - w) / 2))
        vp = int(math.ceil((max_wh - h) / 2))
        padding = (hp, vp, hp, vp)
        return torchvision.transforms.functional.pad(x, padding, value, 'constant')

    def __call__(self, image, target):
        image = self.pad(image, 0)
        target = self.pad(target, 255)
        return image, target


class OffsetLabels:
    # Offsets labels by self.offset for use in a concat dataset context.
    def __init__(self, offset):
        self.offset = offset

    def __call__(self, image, target):
        # offset only elements != 255
        target[target != 255] += self.offset
        return image, target


class RandomCrop:
    def __init__(self, max_crop_size: int):
        self.max_crop_size = max_crop_size

    def __call__(self, image, targets):
        crop_size = min(image.height, image.width, self.max_crop_size)

        left = randint(0, image.width - crop_size)
        top = randint(0, image.height - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        image = image.crop((left, top, right, bottom))
        targets = targets.crop((left, top, right, bottom))
        return image, targets


class ScaleToFit:
    def __init__(self, max_w, max_h):
        self.max_w = max_w
        self.max_h = max_h

    def __call__(self, image, targets):
        image = image.resize((self.max_w, self.max_h), PIL.Image.BILINEAR)
        targets = targets.resize((self.max_w, self.max_h), PIL.Image.NEAREST)
        return image, targets


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, targets):
        if randint(0, 1e6)/1e6 <= self.flip_prob:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            targets = targets.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return image, targets


class ToTensors:
    def __init__(self):
        self.image_transform = torchvision.transforms.ToTensor()

    def __call__(self, image, targets):
        image = self.image_transform(image)
        targets = self.targets_transform(targets)
        return image, targets

    def targets_transform(self, targets):
        array = np.asarray(targets)
        tensor = torch.from_numpy(array)
        return tensor.long()
