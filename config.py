import data
from dataclasses import dataclass


# +-------------------------------------------------------------------------------------------+ #
# |                                                                                           | #
# |                                         DATAMODULE                                        | #
# |                                                                                           | #
# +-------------------------------------------------------------------------------------------+ #

@dataclass
class DataModule:
    
    """ Preprocessing and data loading config used to instanciate a DataModule object.

    Args:

        video_root (str): Path of the directory containing the 98 videos.

        medical_data_csv_path (str): Path of the csv containing the label used for training. 
                                     This is used to create ground_truth label for neural network
                                     training in data/dataset.py. Labels are integers in [0,1].
                                     Can be None for testing.

        channels (int): Number of input channels.

        x_size, y_size (int): Output size of video. The DataModule will apply center cropping 
                              if needed.

        sampling (int): Sampling rate. Reads one video frame every sampling frames.
        
        batch_size (int): Batch size of the training dataloader.

        num_workers (int): Num of threads for the 3 dataloaders (train, val, test).

        reshape_method (str): How to reshape frames: resize (interpolation) or center crop.
                              One of 'resize', 'crop'. Default to 'resize'.
                              Note that ofc, the 'resize' method is way slowlier.
    """

    video_root:            str = "/homes/l17vedre/Bureau/Sanssauvegarde/SPYGLASS/cropped"
    medical_data_csv_path: str = "/homes/l17vedre/SPYGLASS/medical_data.csv" 
    channels:              int = 3
    x_size:                int = 224
    y_size:                int = 224   
    sampling:              int = 12
    batch_size:            int = 2
    num_workers:           int = 4
    reshape_method:        str = 'crop'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         TRAIN                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Model:
    
    """ Training config used to instanciate a LightningModel and a Trainer.

    Args:
        use_label_smoothing (bool):  Controls the use of label smoothing during training.

        smoothing (float): This has no effect if use_label_smoothing is False:
                           Else, the one true label will be 1-smoothing instead of 1,
                           and others false labels will be smoothing instead of 0.
    """
    window:                 int = 10
    stride:                 int = 9
    early_fusion_dim:       int = 3
    fusion_mode:            str = 'early'
    encoder_base_net:       str = 'resnet50'
    pretrained:            bool = False
    encoder_hidden_dim_1:   int = 512
    encoder_hidden_dim_2:   int = 512
    encoder_dropout_rate: float = 0.3
    embed_dim:              int = 300
    decoder:                str = 'rnn'
    decoder_dropout_rate: float = 0.3
    fc_hidden_dim:          int = 128
    h_RNN_layers:           int = 3
    h_RNN:                  int = 256
    num_classes:            int = 2
    aggregation_mode:       str = 'last'
    use_label_smoothing:   bool = False
    smoothing:            float = 0.1
    lr:                   float = 1e-6
    momentum:             float = 0.9
    nesterov:              bool = True
    weight_decay:         float = 5e-4
    rop_mode:               str = 'min'
    rop_factor:           float = 0.2
    rop_patience:           int = 5
    verbose:               bool = True