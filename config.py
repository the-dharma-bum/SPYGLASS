import data
from dataclasses import dataclass


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      DATAMODULE                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

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

        x_size, y_size (int): Output size of video. The DataModule will apply center cropping if needed.

        sampling (int): Sampling rate. Reads one video frame every sampling frames.
        
        train_batch_size (int): Batch size of the training dataloader.

        val_batch_size (int): Batch size of the validation dataloader.

        num_workers (int): Num of threads for the 3 dataloaders (train, val, test).
    """

    video_root:            str = "/homes/l17vedre/Bureau/Sanssauvegarde/SPYGLASS/cutted/"
    medical_data_csv_path: str = "/homes/l17vedre/SPYGLASS/medical_data.csv" 
    channels:              int = 3
    x_size:                int = 224
    y_size:                int = 224   
    sampling:              int = 1
    train_batch_size:      int = 2
    val_batch_size:        int = 2
    num_workers:           int = 4
    



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
    fusion_mode:            str = 'single_frame'
    encoder_base_net:       str = 'resnet18'
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
    aggregation_mode:       str = 'mean'
    use_label_smoothing:   bool = False
    smoothing:            float = 0.1