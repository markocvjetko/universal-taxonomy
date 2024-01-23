import contextlib

import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter

#import lib.cylib as cylib

__all__ = ['compute_errors', 'get_pred', 'evaluate_semseg']


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


def get_pred(logits, labels, conf_mat):
    _, pred = torch.max(logits.data, dim=1)
    pred = pred.byte().cpu()
    pred = pred.numpy().astype(np.int32)
    true = labels.numpy().astype(np.int32)
    cylib.collect_confusion_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)


def mt(sync=False):
    if sync:
        torch.cuda.synchronize()
    return 1000 * perf_counter()


def collect_confusion_matrix(y, yt, conf_mat):
    for l, lt in zip(y, yt):
        if lt < len(conf_mat):
            conf_mat[lt, l] += 1



def evaluate_semseg(model, data_loader, class_info, observers=(), return_conf_mat=False, verbose=True, split_on=None, taxonomy_transform=lambda _, __: (_, __)):
    model.eval()
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        if split_on:
            conf_mat = np.zeros((split_on, len(class_info)-split_on), dtype=np.uint64)
        else:
            conf_mat = np.zeros((len(class_info), model.num_classes), dtype=np.uint64)

        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if type(batch) == dict:
                batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
                logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            else:
                batch[1] = batch[1].numpy().astype(np.uint32)
                logits, additional = model.do_forward(batch, batch[1].shape[1:3])
            if type(batch) == dict:
                logits = torch.Tensor(taxonomy_transform(logits.cpu().numpy(), batch['original_labels'].flatten()))
            else:
                logits = torch.Tensor(taxonomy_transform(logits.cpu().numpy(), batch[1].flatten()))
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            for o in observers:
                o(pred, batch, additional)
            if type(batch) == dict:
                cylib.collect_confusion_matrix(pred.flatten(), batch['original_labels'].flatten(), conf_mat)
            else:
                # replace all 255 with 0 in batch[1]
                # batch[1][batch[1] != 255123] = 0
                # print('batch', np.unique(batch[1].flatten()))
                # print(pred.flatten())
                # print('pred', np.unique(pred))
                # print(batch[1].shape, pred.shape)

                # cylib.collect_confusion_matrix(batch[1].flatten(), pred.flatten(), conf_mat)
                if split_on: # masks first N preds and last len - N preds to get cross-dataset evaluation
                    first_preds = logits.data[:, :split_on, :, :]
                    first_preds = torch.argmax(first_preds, dim=1).byte().cpu().numpy().astype(np.uint32)
                    second_preds = logits.data[:, split_on:, :, :]
                    second_preds = torch.argmax(second_preds, dim=1).byte().cpu().numpy().astype(np.uint32)

                    cylib.collect_confusion_matrix(first_preds.flatten(), second_preds.flatten(), conf_mat)
                    # first_preds = pred.flatten()[:split_on]
                    # first_argmax = np.argmax(first_preds)
                    # first_onehot = np.zeros(first_preds.shape, dtype=np.uint32)
                    # first_onehot[first_argmax] = 1

                    # last_preds = pred.flatten()[split_on:]
                    # # print(first_preds)
                    # last_argmax = np.argmax(last_preds)
                    # last_onehot = np.zeros(last_preds.shape, dtype=np.uint32)
                    # last_onehot[last_argmax] = 1

                    # print(first_onehot, last_onehot)
                    # cylib.collect_confusion_matrix(first_onehot, last_onehot, conf_mat)
                else:
                    cylib.collect_confusion_matrix(pred.flatten(), batch[1].flatten(), conf_mat)
            # print(np.unique(batch[1].flatten()))
        print('')
        if not return_conf_mat:
            pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=verbose)
    model.train()
    if return_conf_mat:
        return conf_mat

    return iou_acc, per_class_iou
