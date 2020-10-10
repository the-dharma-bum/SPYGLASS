import os
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict

class Dataset2DGenerator:

    """ Generates a 2d dataset from a directory containing video.
    For each video file, a certain amout of frames will be saved as npz files.
    """

    def __init__(self, video_root: str, output_dir: str, sampling_factor: int, crop: int) -> None:
        """ Instanciate a 2d dataset generator.

        Args:
            video_root (str): Path of the directory containing all the videos.
            output_dir (str): Path to save the 2d images.
            sampling_factor (int): For each video, one out of sampling_factor frames will be saved.
            crop (int): The frame are cropped before being saved so that each resulting file has the same 
                        size regardless of the original video resolution (final size will be (crop, crop)).
        """
        self.video_root = video_root
        self.video_list = sorted(os.listdir(video_root))
        self.output_dir = output_dir
        self.sampling_factor = sampling_factor
        self.crop = crop

    def prepare_output_folder(self) -> None:
        """ Create output dir if needed. """
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def center_crop(self, frame: np.ndarray) -> np.ndarray:
        """ Crop a frame using self.crop.

        Args:
            frame (np.ndarray): A 2d matrix representing a frame of a given video.

        Returns:
            np.ndarray: A 2d matrix of shape (self.crop, self.crop) obtained by center cropping
                        the given 2d-array.
        """
        assert frame.shape[0] == frame.shape[1]
        center = frame.shape[0]//2
        offset = self.crop//2
        return frame[center-offset:center+offset, center-offset:center+offset]

    def process_one_video(self, video_path: str, patient_index: int) -> None:
        """ Save a frame every self.subsampling frame.
        
        A saved frame will be named i_j.npz where i is the original video index
        (from 1 to 98) and j is the frame index within this video.  

        Args:
            video_path (str): Path of one given video (out of 98).
            patient_index (int): The patient index (in [1,98]). Will be used to name the saved frame.
                                 This name will be used when loading file to get the corresponding label.
        """
        capture = cv2.VideoCapture(video_path)
        not_last_frame = True
        frame_index = 0
        while not_last_frame:
            not_last_frame, frame = capture.read()
            frame_index += 1
            if not_last_frame and frame_index%self.sampling_factor == 0:
                cropped_frame = self.center_crop(frame)
                output_name = f"{patient_index}_{frame_index}" 
                np.savez(os.path.join(self.output_dir, output_name), cropped_frame)
        capture.release()

    def run(self) -> None:
        """ Iterate the process_one_video() method on the self.video_root folder. """
        self.prepare_output_folder()
        for i in tqdm(range(len(self.video_list))):
            video_path = os.path.join(self.video_root, self.video_list[i])
            self.process_one_video(video_path, i)