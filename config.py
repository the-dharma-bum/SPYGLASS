from dataclasses import dataclass


@dataclass
class Config:

    """ Preprocessing & Training config to be used in main.py routines.

    Args:
        video_root (str): Path of the directory containing the 98 videos.

        data_2d_root (str): Path of the directory containing 2d images, ie frames
                            from the originals 98 videos.
                            Note that this can be used as the output params for creating
                            the 2d dataset aswell as the input param for neural network training.

        medical_data_csv_path (str): Path of the csv containing the label used for training.
                                     Labels are integers in [1,6]. 
                                     This is used to create ground_truth label for neural network 
                                     training in data/dataset.py.
                                     Can be None for testing.

        mode (str): 'image' or 'video'. Pipeline mode.

        sampling_factor (int): When creating a 2d dataset from the videos, a certain amout of frames
                               is selected. This param controls how many frames to keep in the 2d 
                               dataset (one out of sampling_factor.)

        crop (int): When creating a 2d dataset from the videos, the frames need to be center cropped to obtain
                    2d images of same size since the orginal videos have differents resolutions.
                    This will be used in data/make_2d_dataset.center_crop()/
                    Note that this procedure could be avoided by doing the following:
                        1. Save 2d images of different size in make_2d_dataset.py
                        2. Resize each batch just before training by applying the RandomCrop transform. 

        train_batch_size (int): Batch size of the training dataloader.

        val_batch_size (int): Batch size of the validation dataloader.

        num_workers (int): Num of threads for the 3 dataloaders (train, val, test).

        use_label_smoothing (bool):  Controls the loss used: Vanilla CrossEntropy if False, 
                                     smoothed CrossEntropy if True.
                                     See utils/label_smoothing.py.

        smoothing (float): This has no effect if use_label_smoothing is False:
                           Else, the one true label will be 1-smoothing instead of 1,
                           and others false labels will be smoothing instead of 0.
    """

    video_root:            str = "/homes/l17vedre/Bureau/Sanssauvegarde/cropped"
    data_2d_root:          str = "/homes/l17vedre/Bureau/Sanssauvegarde/2D_sampling1"
    medical_data_csv_path: str = "/homes/l17vedre/SPYGLASS/medical_data.csv"
    mode:                  str = 'image' 
    channels:              int = 3
    time_depth:            int = 25*1 # first 1 seconds
    x_size:                int = 224
    y_size:                int = 224   
    sampling_factor:       int = 10
    crop:                  int = 400
    train_batch_size:      int = 2
    val_batch_size:        int = 2
    num_workers:           int = 4
    use_label_smoothing:  bool = True
    smoothing:           float = 0.1                 