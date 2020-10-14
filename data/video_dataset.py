""" Adapted from:
https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset/blob/master/GeneralVideoDataset.py.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from typing import List, Tuple, Dict, Optional, Union, Callable, NewType


# Type hint
Transform = NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])

class SpyGlassVideoDataset(Dataset):

    """ Pytorch Video Dataset. 
    Loads and stacks video frames and overwrite __getitem__ and __len__ methods.
    Notation:
        (N,C,T,W,H) = (batch_size, num_channels, time_depth, x_size, y_size).
    """

    def __init__(self, input_root: str, channels: int, time_depth: int,
                 x_size: int, y_size: int, mean: List[float], std: List[float],
                 medical_data_csv_path: str=None, transform: Transform=None) -> None:
        """ Instanciate video SpyGlass Dataset.

        Args:
            input_root (str): The folder containing all the npz files.

            channels (int): Number of channels of each video frame.

            time_depth (int): Number of frames to take per video. 
                              For now, we only take the first time_depth frames of each video.

            x_size, y_size (int, int): Frame sizes. Will apply center cropping if needed.

            mean (List[float]): Mean of the whole dataset over each channels.

            std (List[float]): Standard deviation of the whole dataset over each channels.

            medical_data_csv_path (str, optional): The csv file containing the label for the 98 patients.
                                                   It is not required while testing.
            transform (Transform, optional): Takes a numpy array and apply a tranformation to it 
                                             (ie data augmentation).
                                             Returns a transformed numpy array. Defaults to None.
        """
        super().__init__()
        self.input_root   = input_root
        self.input_list   = sorted(os.listdir(input_root))
        self.channels     = channels
        self.time_depth   = time_depth
        self.x_size       = x_size
        self.y_size       = y_size
        self.mean         = mean
        self.std          = std
        if medical_data_csv_path is not None:
            self.medical_data = pd.read_csv(medical_data_csv_path)
        self.transform    = transform

    def get_target(self, patient_index: int) -> int:
        """ Gets a binary label (0: benign, 1: malign).

        Args:
            patient_index (int): The patient_index ([1,98]) obtained from 
                                 the dataset index.

        Returns:
            int: A binary label (ie 0 or 1).
                 If the corresponding label line in the medical_data csv 
                 is >= 5 (ie 5 or 6), retunrs 0, else returns 1. 
                 See presentation_data.pdf.
        """
        target = 0 if self.medical_data.loc[patient_index].label >= 5 else 1
        return target

    def center_crop(self, frame: np.ndarray) -> np.ndarray:
        """ Apply center cropping  on one given frame.

        Args:
            frame (np.ndarray): Shape (W,H,C).

        Returns:
            np.ndarray: Shape (x_size,y_size,C).
        """
        center_x, center_y = frame.shape[0]//2, frame.shape[1]//2
        offset_x, offset_y = self.x_size//2, self.y_size//2
        return frame[center_x-offset_x:center_x+offset_x, center_y-offset_y:center_y+offset_y]

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """ Normalize one given frame.

        Args:
            frame (np.ndarray): Shape (W,H,C).

        Returns:
            np.ndarray: Shape (W,H,C).
        """
        return (frame-self.mean)/self.std

    def read_video(self, video_file: str) -> Tuple[torch.FloatTensor, bool]:
        """ Stack as many frames as self.time_depth in a tensor.
        Apply cropping and normalization.  

        Args:
            video_file (str): A video path.

        Returns:
            Tuple[torch.FloatTensor, bool]: Tensor of shape (C,T,W,H).
                                            Boolean as a flag for video reading.
        """
        capture = cv2.VideoCapture(video_file)
        frames = torch.FloatTensor(self.channels, self.time_depth, self.x_size, self.y_size)
        failed_clip = False
        for t in range(self.time_depth):
            ret, frame = capture.read()
            if ret:
                frame = self.center_crop(self.normalize(frame))
                frame = torch.from_numpy(frame)
                # from channel last to channel first: (W,H,C) -> (C,W,H)
                frame = frame.permute(2,0,1)
                frames[:, t, :, :] = frame
            else:
                print('Skipped')
                failed_clip = True
                break
        return frames, failed_clip


    def __getitem__(self, index: int) -> Dict:
        """ Generates a sample from a dataset index.

        Args:
            index (int): a dataset index (in [1,98])

        Returns:
            sample (Dict): A dict containing:
                            - the 4d-tensor (C,T,W,H)
                            - the target (int in [0,1])
                            - a boolean falg for video reading 
        """
        video_file = os.path.join(self.input_root, self.input_list[index])
        clip, failed_clip = self.read_video(video_file)
        if self.transform is not None:
            clip = self.transform(clip)
        sample = {
            'clip': clip,
            'target': self.get_target(index),
            'failed_clip': failed_clip 
        }
        return sample

    def __len__(self) ->  int:
        return len(self.input_list)