import contextlib

import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
from torchmetrics import ConfusionMatrix

__all__ = ['compute_errors', 'evaluate_semseg']


def compute_errors(conf_mat, class_info, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(1)
    TPFN = conf_mat.sum(0)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def mt(sync=False):
    if sync:
        torch.cuda.synchronize()
    return 1000 * perf_counter()


def evaluate_semseg(model, data_loader, class_info, observers=()):
    model.eval()
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=model.num_classes)
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            #print all shapes
            # print('image', batch['image'].shape)
            # print('labels', batch['labels'].shape)
            # print('original_labels', batch['original_labels'].shape)

            batch['original_labels'] = batch['original_labels'].int().cpu()
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            pred = torch.argmax(logits.data, dim=1).int().cpu()
            for o in observers:
                o(pred, batch, additional)
            if model.criterion.ignore_id != -100:
                valid_idx = batch["original_labels"] != model.criterion.ignore_id
                pred = pred[valid_idx]
                gt = batch["original_labels"][valid_idx]
            conf_mat += conf_matrix(pred, gt).numpy().astype(np.uint64)
        print('')
        pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=True)
    model.train()
    return iou_acc, per_class_iou



def evaluate_semseg_multi_task(model, data_loader, class_info1, class_info2, observers=()):
    model.eval()
    #conf_matrix = ConfusionMatrix(task="multiclass", num_classes=model.num_classes)
    conf_matrix1 = ConfusionMatrix(task="multiclass", num_classes=model.num_classes1)
    conf_matrix2 = ConfusionMatrix(task="multiclass", num_classes=model.num_classes2)
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        #conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        conf_mat1 = np.zeros((model.num_classes1, model.num_classes1), dtype=np.uint64)
        conf_mat2 = np.zeros((model.num_classes2, model.num_classes2), dtype=np.uint64)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            
            #batch['original_labels'] = batch['original_labels'].int().cpu()
            batch['original_labels1'] = batch['original_labels1'].int().cpu()
            batch['original_labels2'] = batch['original_labels2'].int().cpu()

            logits1, logits2, additional = model.do_forward(batch, batch['original_labels1'].shape[1:3])
            #pred = torch.argmax(logits.data, dim=1).int().cpu()
            pred1 = torch.argmax(logits1.data, dim=1).int().cpu()
            pred2 = torch.argmax(logits2.data, dim=1).int().cpu()

            for o in observers:
                o(pred1, batch, additional)
                o(pred2, batch, additional)

            # if model.criterion.ignore_id != -100:
            #     valid_idx = batch["original_labels"] != model.criterion.ignore_id
            #     pred = pred[valid_idx]
            #     gt = batch["original_labels"][valid_idx]
            # conf_mat += conf_matrix(pred, gt).numpy().astype(np.uint64)
            if model.criterion1.ignore_id != -100:
                valid_idx1 = batch["original_labels1"] != model.criterion1.ignore_id
                pred1 = pred1[valid_idx1]
                gt1 = batch["original_labels1"][valid_idx1]
            conf_mat1 += conf_matrix1(pred1, gt1).numpy().astype(np.uint64)
            if model.criterion2.ignore_id != -100:
                valid_idx2 = batch["original_labels2"] != model.criterion2.ignore_id
                pred2 = pred2[valid_idx2]
                gt2 = batch["original_labels2"][valid_idx2]
            conf_mat2 += conf_matrix2(pred2, gt2).numpy().astype(np.uint64)
        print('')
        pixel_acc, iou_acc1, recall, precision, _, per_class_iou1 = compute_errors(conf_mat1, class_info1, verbose=True)
        pixel_acc, iou_acc2, recall, precision, _, per_class_iou2 = compute_errors(conf_mat2, class_info2, verbose=True)
    model.train()
    return iou_acc1, per_class_iou2, iou_acc2, per_class_iou2
