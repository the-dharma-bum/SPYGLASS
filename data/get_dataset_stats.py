from tqdm import tqdm
import torch
from data import SpyGlassDataModule
from typing import Tuple


def get_mean_std_dataset(input_root: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Adapted from:
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

    Args:
        input_root (str): dir containing all the 2d images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors of size 3 containing the mean and std 
                                           per channels.
    """
    datamodule = SpyGlassDataModule(input_root)
    datamodule.setup(stage='test')
    dataloader = datamodule.test_dataloader()
    mean, std = 0., 0.
    for images in tqdm(dataloader):
        batch_samples = images.size(0) # batch size (the last batch can have smaller size)
        # from channel last to channel first: (N, W, H, C) -> (N, C, W, H)
        images = images.permute(0,3,1,2)
        # flatten 2d images into 1d vector: (N, C, W, H) -> (N, C, W*H)
        images = images.view(batch_samples, images.size(1), -1)
        # cast uint to double to avoid overflow
        mean += images.double().mean(2).sum(0) # mean over the 2nd dimension (ie flatten image)
        std  += images.double().std(2).sum(0)  #  std over the 2nd dimension (ie flatten image)
    mean /= len(dataloader.dataset)
    std  /= len(dataloader.dataset)
    return mean, std        