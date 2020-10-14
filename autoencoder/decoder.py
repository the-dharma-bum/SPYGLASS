""" Adapted from:
https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/functions.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):

    """ RNN decoder: LSTM followed by linear. """

    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128,
                 drop_p=0.3, num_classes=2) -> None:
        """ Initialize  a basic LSTM based decoder.

        Args:
            CNN_embed_dim (int, optional): Output size of the Resnet encoder. Defaults to 300.
            h_RNN_layers (int, optional): RNN hidden layers. Defaults to 3.
            h_RNN (int, optional): RNN hidden nodes. Defaults to 256.
            h_FC_dim (int, optional): Fully connected layers size. Defaults to 128.
            drop_p (float, optional): Dropout rate. Defaults to 0.3.
            num_classes (int, optional): Final fully connected layers output size. Defaults to 2.
        """
        super(DecoderRNN, self).__init__()
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers   = h_RNN_layers  
        self.h_RNN          = h_RNN                 
        self.h_FC_dim       = h_FC_dim
        self.drop_p         = drop_p
        self.num_classes    = num_classes
        self.LSTM = nn.LSTM(input_size=self.RNN_input_size, hidden_size=self.h_RNN,        
                            num_layers=h_RNN_layers, batch_first=True)
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN: torch.Tensor) -> torch.Tensor:
        """ Decode a tensor already passed through a CNN.

        h_n shape: (n_layers, batch, hidden_size)
        h_c shape: (n_layers, batch, hidden_size)
        None represents zero initial hidden state.
        RNN_out has shape (batch, time_step, output_size).

        Args:
            x_RNN (torch.Tensor): Shape (batch_size, num_channels, CNN_embed_dims).

        Returns:
            torch.Tensor: Shape (batch_size, num_classes).
        """
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        # FC layers
        x = self.fc1(RNN_out[:, -1, :]) # choose RNN_out at the last time step.
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x