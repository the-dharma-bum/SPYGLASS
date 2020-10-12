import os
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from data import SpyGlass2dDataset
from typing import Optional, Callable, NewType


# Type hint
Transform =  NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])


class SpyGlassDataModule(LightningDataModule):
    
    """ Generates three dataloaders (train, eval, test) to be used by a Lightning Model. """

    def __init__(self, input_root: str, medical_data_csv_path: str=None,
                 train_batch_size: int=64, val_batch_size: int=64, num_workers: int=4) -> None:
        """ Instanciate a Pytorch Lightning DataModule.

        Args:
            input_root (str): The folder containing all the npz files.

            medical_data_csv_path (str): The csv file containing the label for the 98 patients.

            train_batch_size (int): Batch size of the training dataloader.

            val_batch_size (int): Batch size of the validation dataloader.

            num_workers (int): Num of threads for each of the 3 dataloaders (train, val, test).

        """
        super().__init__()
        self.input_root = input_root
        if medical_data_csv_path is not None:
            self.medical_data_csv_path = medical_data_csv_path
        self.train_batch_size = train_batch_size
        self.val_batch_size   = val_batch_size
        self.num_workers      = num_workers
        # value obtained by calling data.get_dataset_stats.get_mean_std_dataset()
        mean, std = [78.5606, 111.8194, 135.2136], [64.6343,  72.6750,  69.9263]
        self.train_transform: Transform = transforms.Compose([
            transforms.ToPILImage(), # most pytorch transforms work on PIL image
            transforms.RandomCrop(200),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Normalize works on Tensor though
            transforms.Normalize(mean=mean,std=std)
        ])
        self.test_transform: Transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])

    def setup(self, stage: str=None) -> None:
        """ Basically nothing more than train/val split.

        Args:
            stage (str, optional): fit or test.
                                   Acts on the train param of the SpyGlass2dDataset constructor. 
                                   Defaults to None.
        """
        total_length = len(os.listdir(self.input_root))
        train_length = int(0.8*total_length)
        val_length   = int(0.2*total_length)
        if train_length + val_length != total_length: # round error
            val_length += 1
        if stage == 'fit' or stage is None:
            # if stage is not 'test', medical_data_csv_path should have been set during datamodule instanciation.
            assert self.medical_data_csv_path is not None, 'did you forget to specify a stage ?'
            spyglass_full = SpyGlass2dDataset(self.input_root, self.medical_data_csv_path, 
                                              train=True, transform=self.train_transform)
            self.spyglass_train, self.spyglass_val = random_split(spyglass_full, [train_length, val_length])
        if stage == 'test' or stage is None:
            self.spyglass_test = SpyGlass2dDataset(self.input_root, train=False, transform=self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.spyglass_train, num_workers=self.num_workers,
                          batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.spyglass_val, num_workers=self.num_workers, 
                          batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.spyglass_test, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, shuffle=False)