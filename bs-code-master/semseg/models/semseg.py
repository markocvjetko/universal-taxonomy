import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings

from semseg.models.util import _BNReluConv, upsample


class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, num_inst_classes=None, use_bn=True, k=1, bias=True,
                 loss_ret_additional=False, upsample_logits=True, logit_class=_BNReluConv,
                 multiscale_factors=(.5, .75, 1.5, 2.)):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = logit_class(self.backbone.num_features, self.num_classes, batch_norm=use_bn, k=k, bias=bias)
        if num_inst_classes is not None:
            self.border_logits = _BNReluConv(self.backbone.num_features, num_inst_classes, batch_norm=use_bn,
                                             k=k, bias=bias)
        self.criterion = None
        self.loss_ret_additional = loss_ret_additional
        self.img_req_grad = loss_ret_additional
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

    def forward(self, image, target_size, image_size):
        features, additional = self.backbone(image)
        logits = self.logits.forward(features)
        if (not self.training) or self.upsample_logits:
            logits = upsample(logits, image_size)
        if hasattr(self, 'border_logits'):
            additional['border_logits'] = self.border_logits(features).sigmoid()
        additional['logits'] = logits
        # print mean of torch tensor:
        # print(torch.mean(image))
        # print(image)
        # # show torch tensor in BGR format 'image' as image:
        # import cv2
        # cv2.imshow('image', image[0].permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(image.shape)
        # exit(1)
        return logits, additional

    def forward_down(self, image, target_size, image_size):
        return self.backbone.forward_down(image), target_size, image_size

    def forward_up(self, feats, target_size, image_size):
        feats, additional = self.backbone.forward_up(feats)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        logits = upsample(logits, image_size)
        return logits, additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda'), img_key='image'):
        if image_size is None:
            image_size = batch['target_size']
        warnings.warn(f'Image requires grad: {self.img_req_grad}', UserWarning)
        if type(batch) is dict:
            image = batch[img_key].detach().requires_grad_(self.img_req_grad).to(device)
        else:
            image = batch[0].detach().requires_grad_(self.img_req_grad).to(device)
        target_size = batch['target_size'] if type(batch) is dict and 'target_size' in batch else None
        return {
            'image': image,
            'image_size': image_size,
            'target_size': target_size
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        logits, additional = self.forward(**data)
        additional['model'] = self
        additional = {**additional, **data}
        return logits, additional

    def loss(self, batch):
        assert self.criterion is not None
        if type(batch) is dict:
            labels = batch['labels'].cuda()
        else:
            labels = batch[1].cuda()
        print('loss@LOC85: ', labels.shape)
        logits, additional = self.do_forward(batch, image_size=labels.shape[-2:])
        if self.loss_ret_additional:
            return self.criterion(logits, labels, batch=batch, additional=additional), additional
        return self.criterion(logits, labels, batch=batch, additional=additional)

    def random_init_params(self):
        params = [self.logits.parameters(), self.backbone.random_init_params()]
        if hasattr(self, 'border_logits'):
            params += [self.border_logits.parameters()]
        return chain(*(params))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def ms_forward(self, batch, image_size=None):
        if type(batch) is dict:
            image_size = batch.get('target_size', image_size if image_size is not None else batch['image'].shape[-2:])
        else:
            image_size = batch[0].shape[-2:]
        ms_logits = None
        if type(batch) is dict:
            pyramid = [batch['image'].cuda()]
        else:
            pyramid = [batch[0].cuda()]
        pyramid += [
            F.interpolate(pyramid[0], scale_factor=sf, mode=self.backbone.pyramid_subsample,
                          align_corners=self.backbone.align_corners) for sf in self.multiscale_factors
        ]
        for image in pyramid:
            if type(batch) is dict:
                batch['image'] = image
            else:
                batch[0] = image
            logits, additional = self.do_forward(batch, image_size=image_size)
            if ms_logits is None:
                ms_logits = torch.zeros(logits.size()).to(logits.device)
            ms_logits += F.softmax(logits, dim=1)
        if type(batch) is dict:
            batch['image'] = pyramid[0].cpu()
        else:
            batch[0] = pyramid[0].cpu()
        return ms_logits / len(pyramid), {}