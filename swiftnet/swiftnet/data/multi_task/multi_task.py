#a factory function creating a merged pascal VOC and Cityscapes dataset

import torchvision
import torch
from swiftnet.data.cityscapes.cityscapes import Cityscapes
from swiftnet.data.pascal.pascal import PascalVocSegmentation

class VOCSegmentationWrapper(PascalVocSegmentation): 
    '''
    A wrapper class for the Pascal VOC dataset. Instead of returning image + segmentation mask, it
    return an image and two segmentation masks. The first segmentation mask is the original one, while
    the other is a tensor of 255s of the same size as the original segmentation mask.
    '''
    def __getitem__(self, index):
        ret_dict = super().__getitem__(index)
        wrapped_red_dict = {'image':ret_dict['image'],
            'labels1':ret_dict['labels'],
            'labels2':torch.ones_like(ret_dict['labels']) * 19}
        if 'original_labels' in ret_dict:
            wrapped_red_dict['original_labels1'] = ret_dict['original_labels']
            wrapped_red_dict['original_labels2'] = torch.ones_like(ret_dict['original_labels']) * 19
        return wrapped_red_dict


class CityscapesWrapper(Cityscapes):
    '''
    A wrapper class for the Cityscapes dataset. Instead of returning image + segmentation mask, it
    return an image and two segmentation masks. The first segmentation mask is the original one, while
    the other is a tensor of 255s of the same size as the original segmentation mask.
    '''
    def __getitem__(self, index):
        ret_dict = super().__getitem__(index)
        wrapped_red_dict = {'image':ret_dict['image'], 
                'labels1':torch.ones_like(ret_dict['labels']) * 255,
                'labels2':ret_dict['labels']}
        if 'original_labels' in ret_dict:
            wrapped_red_dict['original_labels1'] = torch.ones_like(ret_dict['original_labels']) * 255
            wrapped_red_dict['original_labels2'] = ret_dict['original_labels']
        return wrapped_red_dict