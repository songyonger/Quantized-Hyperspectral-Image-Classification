import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def count_correct_perclass(output, target, correct_per_class, sample_per_classs):
    """Computes the correct_classification number per class in every batch"""
    batch_size = target.size(0)
    _, pred = output.topk(1,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    correct = correct[:1].cpu().float().numpy().squeeze(axis=0)
    pred = pred[:,1].cpu().float().numpy()
    target = target.cpu().numpy()
    mask = np.unique(target)
    for v in mask:
        sample_per_classs[v] +=np.sum(target == v)
    for v in np.nonzero(correct)[0]:
        correct_class_index = target[v]
        correct_per_class[correct_class_index] +=1
