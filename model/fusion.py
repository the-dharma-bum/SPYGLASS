import torch
import torch.nn as nn
from typing import NewType, Union
import pytorch_lightning as pl


# type hint
Network = NewType('Network', Union[nn.Sequential, nn.Module])


class Fusion(pl.LightningModule):

    """ Pytorch Implementation of:
        Large-scale Video Classification with Convolutional Neural Networks.
    
        Implements:
            - single frame
            - early fusion
    """

    def __init__(self, encoder: Network, window: int=10, stride: int=1,
                 early_fusion_hidden_dim: int=3, mode: str='early') -> None:
        """ Initalise a fusion environnement.

        Takes an encoder and feeds it with fused frames.

        A fusion takes as input a video, that is a 5d tensor:
        (batch_size, num_channels, time_depth, x_size, y_size)
        It outputs a 4d tensor:
        (batch_size, fused_depth, encoder_output_dim)

        Args:
            encoder (Network): Takes a 4d input (fused frames)
                               (batch_size, encoder_input_dim, x_size, y_size).
            window (int): How many consecutive frames do we fuse. Defaults to 10.
            stride (int, optional): Stride between each window. Defaults to 1.
            mode (str): Which fusion to run: 'single_frame' or 'early'. Defaults to 'early'.
            early_fusion_hidden_dim (int): The intermediate fusion state's hidden dim.
                                           Defaults to 3.
        """
        super().__init__()
        self.encoder = encoder
        self.mode    = mode
        self.window  = window
        self.stride  = stride
        self.early_fusion_hidden_dim = early_fusion_hidden_dim
        if mode == 'early':
            self.init_early_fusion()

    def init_early_fusion(self) -> None:
        """ Prepare the encoder to perfom an early fusion.
        As we take a 3d inputs (of depth called window),we replace the first Conv2d
        of the base encoder by the following:
            1. a Conv3d taking with 3 inputs channels and a output dim defined by
               early_fusion_hidden_dim.
               shape after conv_3d: (batch_size, early_fusion_hidden_dim, window, x_size, y_size)
            2. a flattening
               shape after flattening: (batch_size, early_fusion_hidden_dim * windows, x_size, y_size)
            2. a Conv2d with an input dim equals to early_fusion_hidden_dim and an output 
               dim of 64 so that it matches the output dim of the initial Conv2d.
        Args:
            early_fusion_hidden_dim (int):  the intermediate state's hidden dim. 
            window (int): how many consecutive frames do we fuse.
        """
        # early fusion 3 layers
        conv3d  = nn.Conv3d(3,self.early_fusion_hidden_dim, kernel_size=3, padding=1)
        flatten = torch.nn.Flatten(start_dim=1, end_dim=2)
        conv2d  = nn.Conv2d(self.early_fusion_hidden_dim * self.window, 64, kernel_size=(7, 7),
                            stride=(2, 2), padding=(3, 3), bias=False)
        early_fusion = [conv3d, flatten, conv2d]
        # append all conv modules minus the first conv2d(3,64) to our early_fusion stem.
        early_fusion.extend(list(self.encoder.net.children())[1:])
        # redefine our encoder as a sequential
        self.encoder.net = nn.Sequential(*early_fusion)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()

    def init_features_tensor(self, inputs_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, time_depth = inputs_tensor.size(0), inputs_tensor.size(2)
        if self.mode == 'early':
            output_depth = time_depth // self.stride
        else:
            output_depth = time_depth
        features = torch.FloatTensor(batch_size, output_depth, self.encoder.CNN_embed_dim)
        if torch.cuda.is_available():
            features = features.cuda()
        return features, output_depth

    def single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """ Single Frame Fusion.

        Predicts all frames independantly and returns as many outputs as frames.

        Args:
            x (torch.Tensor): A 5d tensor: (batch_size, channels, time_depth, x_size, y_size)
                              
        Returns:
            torch.Tensor: The final output, of shape (time_depth, batch_size, num_classes).
        """
        features, output_depth = self.init_features_tensor(x)
        for t in range(output_depth):
            features[:,t,:] = self.encoder(x[:,:,t,:,:])
        return features

    def early(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse a window of consecutive frames at the beginning of the network
        by doing a Conv3d followed by a flattening followed by a Conv2d.

        Args:
            x (torch.Tensor): A 5d tensor: (batch_size, channels, time_depth, x_size, y_size)

        Returns:
            torch.Tensor: Encoder output. 
                          Shape: (batch_size, time_depth // stride, num_classes). 
        """
        features, output_depth = self.init_features_tensor(x)
        for t in range(0,output_depth-self.window,self.stride): 
            # shape before narrow: (batch_size, num_channels, time_depth, x_size, y_size) 
            frame_window = x.narrow(2, t, self.window)
            # shape after narrow: (batch_size, num_channels, window, x_size, y_size)
            features[:,t,:] = self.encoder(frame_window)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'single_frame':
            return self.single_frame(x)
        elif self.mode == 'early':
            return self.early(x)


