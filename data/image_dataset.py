import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from typing import Tuple, Optional, Union, Callable, NewType


# Type hint
Transform = NewType('Transform', Optional[Callable[[np.ndarray], torch.Tensor]])
Batch     = NewType('Batch',     Union[Tuple[np.ndarray,int], np.ndarray])


class SpyGlassImageDataset(Dataset):

    """ Pytorch image Dataset. It just overwrite __getitem__ and __len__ methods. """

    def __init__(self, input_root: str, medical_data_csv_path: str=None,
                 train: Optional[bool]=True, transform: Transform=None) -> None:
        """ Instanciate image SpyGlass Dataset.

        Args:
            input_root (str): The folder containing all the npz files.
            medical_data_csv_path (str, optional): The csv file containing the label for the 98 patients.
                                                   It is not required while testing.
            train (bool, optional): If True,  __get_item__ returns a tuple (image, target). 
                                    If False, __get_item__ returns the image only. 
                                    Defaults to True.
            transform (Transform, optional): Takes a numpy array and apply a tranformation to it 
                                             (ie data augmentation).
                                             Returns a transformed numpy array. Defaults to None.
        """
        super().__init__()
        self.input_root   = input_root
        self.input_list   = sorted(os.listdir(input_root))
        if medical_data_csv_path is not None:
            self.medical_data = pd.read_csv(medical_data_csv_path)
        self.transform    = transform
        self.train        = train

    def patient_index_from_dataset_index(self, dataset_index: int) -> int:
        """ Map the dataset index into the patient index.
        
        Since there are multiples 2d files for 1 patient, we need to map 
        the dataset index with the patient index.
        E.g: any file named i_*.npz will have a unique dataset_index
        while refering to the patient_index i. 

        Args:
            dataset_index (int): The index corresponding to every files, ie 
                                 every element in self.input_root

        Returns:
            int: A patient index, obtained by taking the first element of the string
                 in self.input_list at index dataset_index.
        """
        return int(self.input_list[dataset_index][0])

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

    def __getitem__(self, index: int) -> Batch:
        """ Generates a batch from a dataset index.

        Args:
            index (int): a dataset index

        Returns:
            Batch: If training, returns a tuple (image, target), else returns 
                   an image (optionnaly transformed) only. 
        """
        # by default, the npz format stores arrays in a dict with keys 'arr_0', 'arr_1', ... 
        image = np.load(os.path.join(self.input_root, self.input_list[index]))['arr_0']
        if self.transform is not None:
            image = self.transform(image)
        if self.train:
            patient_index = self.patient_index_from_dataset_index(index)
            target = self.get_target(patient_index)
            return image, target
        return image

    def __len__(self) ->  int:
        return len(self.input_list)