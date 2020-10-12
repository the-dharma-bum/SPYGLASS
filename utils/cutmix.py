""" See https://arxiv.org/abs/1905.04899 """

import numpy as np
import torch

def rand_bbox(size, lam):
    """ Returns 4 random coordinates corresponding to a box vertices.
    The lambda param defines the cropped area ratio: 1 - lam
    """
    width, height = size[2], size[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_w = np.int(width  * cut_ratio)
    cut_h = np.int(height * cut_ratio)
    # uniform
    cut_x = np.random.randint(width)
    cut_y = np.random.randint(height)
    bbx1 = np.clip(cut_x - cut_w // 2, 0, width)
    bby1 = np.clip(cut_y - cut_h // 2, 0, height)
    bbx2 = np.clip(cut_x + cut_w // 2, 0, width)
    bby2 = np.clip(cut_y + cut_h // 2, 0, height)
    return bbx1, bby1, bbx2, bby2


def Cutmix(inputs, labels, beta):
    """ generate mixed sample """
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inputs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    return lam, inputs, target_a, target_b