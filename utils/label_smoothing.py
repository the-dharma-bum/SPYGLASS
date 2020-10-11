""" See https://arxiv.org/abs/1906.02629. Adapted from fastai implementation. """

import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):

    """
    Args:
        smoothing (int): The one true label will be 1-smoothing instead of 1,
                         and others false labels will be smoothing instead of 0.
        reduction (str): 'mean' or 'sum'.
                          The smoothed cross entropy (sce) is in general:
                                * sce(i) = (1-eps)ce(i) + eps*reduced_loss
                          where eps is the smoothing param
                          If reduction='mean', reduced_loss = sum(ce(j))/N
                          If reduction='sum',  reduced_loss = sum(ce(j))
    """

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing, self.reduction = smoothing, reduction

    def reduce_loss(self, loss, reduction):
        """ takes the loss tensor and returns its mean or sum """
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

    def lin_comb(self, value1, value2, beta):
        """ linear combination = exponentially weighted moving average """
        return beta*value1 + (1-beta)*value2

    def forward(self, output, target):
        nb_classes = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.lin_comb(loss/nb_classes, nll, self.smoothing)