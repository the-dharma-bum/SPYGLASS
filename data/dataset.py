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

    def __init__(self, input_root: str, channels: int, x_size: int, y_size: int,
                 mean: List[float], std: List[float], sampling: int=25, 
                 medical_data_csv_path: str=None, transform: Transform=None, criteria: str=None) -> None:
        """ Instanciate video SpyGlass Dataset.

        Args:
            input_root (str): The folder containing all the npz files.

            channels (int): Number of channels of each video frame.

            x_size, y_size (int, int): Frame sizes. Will apply center cropping if needed.

            mean (List[float]): Mean of the whole dataset over each channels.

            std (List[float]): Standard deviation of the whole dataset over each channels.
            
            sampling (int): Sampling rate: takes one frame every sampling frames.

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
        self.x_size       = x_size
        self.y_size       = y_size
        self.mean         = mean
        self.std          = std
        self.sampling     = sampling
        if medical_data_csv_path is not None:
            self.medical_data = pd.read_csv(medical_data_csv_path)
        self.transform    = transform
        self.criteria     = criteria
    
    def get_patient_index(self, dataset_index) -> int:
        indexed_file = self.input_list[dataset_index]
        return int(indexed_file.split('_')[0]) - 1


    def get_target(self, patient_index: int) -> int:
        """ Gets a binary label (0: benign, 1: malign).

        Args:
            patient_index (int): The patient_index ([1,98]) obtained from the dataset index.

        Returns:
            int: A binary label (ie 0 or 1). If the corresponding label line in the medical_data csv 
                 is >= 5 (ie 5 or 6), retunrs 0, else returns 1. See presentation_data.pdf.
        """

        target = 0 if self.medical_data.loc[patient_index].label >= 5 else 1
        return target

    @staticmethod
    def is0possible(criteria):
        list_of_criteria_without_0 = ['vaisseaux_irrreguliers', 'bile', 'stenose_multi', 'infiltration_stenose', 'reduction_lumiere', 'ulceration_multi', 'ulceration_creusante', 'ulceration_irreg', 'type_relief', 'dilatation_vaisseaux', 'couleur', 'diagnostique', 'video']
        return criteria not in list_of_criteria_without_0

    def get_criteria_target(self, patient_index: int):
        sum_label = 0
        nb_doctor   = 0
        for doctor in range(7):
            csv_line = doctor*98 + patient_index
            label = self.medical_data.loc[csv_line][self.criteria]
            if  label != '0':
                try:
                    sum_label += int(label)
                    nb_doctor += 1
                except:
                    continue
            else:
                if self.is0possible(self.criteria):
                    nb_doctor += 1
        return round(sum_label/nb_doctor)

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

    def read_video(self, video_file: str) -> torch.FloatTensor:
        """ Stack sampled video frames in a tensor.
        Apply cropping and normalization.  

        Args:
            video_file (str): A video path.

        Returns:
            torch.FloatTensor: Tensor of shape (C,T,W,H).
        """
        capture = cv2.VideoCapture(video_file)
        time_depth = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_depth = int(time_depth / self.sampling) + 1
        frames = torch.FloatTensor(self.channels, output_depth, self.x_size, self.y_size)
        frames_count = 0
        for t in range(time_depth):
            _, frame = capture.read()
            if frames_count % self.sampling == 0:
                frame = self.center_crop(self.normalize(frame))
                frame = torch.from_numpy(frame)
                # from channel last to channel first: (W,H,C) -> (C,W,H)
                frame = frame.permute(2,0,1)
                frames[:,t,:,:] = frame
            frames_count += 1
        return frames

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, int]:
        """ Generates a sample from a dataset index.

        Args:
            index (int): A dataset index (in [1,98])

        Returns:
            sample (Tuple[torch.FloatTensor, int]): Torch tensor of shape (C,T,W,H).
                                                    int in [0,1].
        """
        video_file = os.path.join(self.input_root, self.input_list[index])
        clip = self.read_video(video_file)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip, self.get_criteria_target(self.get_patient_index(index))

    def __len__(self) ->  int:
        return len(self.input_list)