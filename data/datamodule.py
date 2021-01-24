import os
import numpy as np
from typing import Optional, Callable, NewType
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from data import VideoDataset

# Type hint
Transform =  NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])


class DataModule(LightningDataModule):
    
    """ Generates three dataloaders (train, eval, test) to be used by a Lightning Model. """

    def __init__(self, input_root: str, channels: int, x_size: int, y_size: int,
                 sampling: int, medical_data_csv_path: str=None, batch_size: int=64,
                 num_workers: int=4, reshape_method: str='resize') -> None:
        """ Instanciate a Pytorch Lightning DataModule.

        Args:
            input_root (str): The folder containing all the npz files.

            channels (int): Number of channels of each video frame.

            x_size, y_size (int, int): Frame sizes. Will apply center cropping if needed.

            mean (List[float]): Mean of the whole dataset over each channels.

            std (List[float]): Standard deviation of the whole dataset over each channels.

            samplint(int): Sampling rate of video reading. Takes one frame every sampling frames.

            medical_data_csv_path (str): The csv file containing the label for the 98 patients.

            batch_size (int): Batch size of the training dataloader.

            num_workers (int): Num of threads for each of the 3 dataloaders (train, val, test).

            reshape_method (str): How to reshape frames: resize (interpolation) or center crop.
                                  One of 'resize', 'crop'. Default to 'resize'.
        """
        super().__init__()
        self.input_root = input_root
        self.channels   = channels
        self.x_size     = x_size
        self.y_size     = y_size
        self.sampling   = sampling
        if medical_data_csv_path is not None:
            self.medical_data_csv_path = medical_data_csv_path
        self.batch_size  = batch_size
        self.num_workers = num_workers
        # value obtained by calling data.get_dataset_stats.get_mean_std_dataset()
        self.mean, self.std = [78.5606, 111.8194, 135.2136], [64.6343,  72.6750,  69.9263]
        self.reshape_method = reshape_method
        self.train_transform, self.test_transform = self.init_transforms()

    def init_transforms(self):
        """ To be implemented. """
        #TODO: make transforms that perfom on video.
        return None, None

    def get_splitted_lengths(self):
        total_length = len(os.listdir(self.input_root))
        train_length = int(0.8 * total_length)
        val_length   = int(0.2 * total_length)
        if train_length + val_length != total_length: # round error
            val_length += 1
        return train_length, val_length

    def check_csv(self):
        csv_error = "did you forget to specify stage='test' ?"
        assert self.medical_data_csv_path is not None, csv_error

    def setup(self, stage: str=None) -> None:
        """ Basically nothing more than train/val split.

        Args:
            stage (str, optional): fit or test.
                                   Acts on the train param of the SpyGlassDataset constructor.
                                   Defaults to None.
        """
        train_length, val_length = self.get_splitted_lengths()
        if stage == 'fit' or stage is None:
            # if stage is not 'test', medical_data_csv_path should have been set during
            # datamodule instanciation.
            self.check_csv()
            dataset_full = VideoDataset(self.input_root, self.channels,
                                        self.x_size, self.y_size,
                                        self.mean, self.std, self.sampling,
                                        self.medical_data_csv_path,
                                        transform=self.train_transform,
                                        reshape_method=self.reshape_method)
            self.dataset_train, self.dataset_val = random_split(dataset_full,
                                                                [train_length, val_length])
        if stage == 'test' or stage is None:
            self.dataset_test = VideoDataset(self.input_root, self.channels,
                                             self.x_size, self.y_size,
                                             self.mean, self.std, self.sampling,
                                             self.medical_data_csv_path,
                                             transform=self.test_transform,
                                             reshape_method=self.reshape_method)

    @staticmethod
    def collate_fn(batch):
        # return [item[0] for item in batch], torch.LongTensor([item[1] for item in batch]) 
        return [item[0] for item in batch], torch.Tensor([item[1] for item in batch])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, num_workers=self.num_workers,
                          batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, num_workers=self.num_workers, 
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, num_workers=self.num_workers,
                          batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    @classmethod    
    def from_config(cls, cfg):
        return cls(cfg.video_root, cfg.channels, cfg.x_size, cfg.y_size, cfg.sampling,
                   cfg.medical_data_csv_path, cfg.batch_size, cfg.num_workers)