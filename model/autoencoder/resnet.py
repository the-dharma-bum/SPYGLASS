""" Adapted from:
https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/functions.py.
"""

import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import pytorch_lightning as pl

class ResNetEncoder(pl.LightningModule):

    """ 2D CNN encoder using ResNet pretrained. 
    Notation:
        (N,C,W,H) = (batch_size, num_channels, x_size, y_size).
    """

    def __init__(self, network_name: str='resnet50', pretrained: bool=False,
                 fc_hidden1: int=512, fc_hidden2: int=512, drop_p: float=0.3, CNN_embed_dim: int=300):
        """Load the pretrained network and replace top fc layer."""
        super(ResNetEncoder, self).__init__()
        network  = torch.hub.load('pytorch/vision:v0.7.0', network_name, pretrained=pretrained)
        self.net = nn.Sequential(*list(network.children())[:-1])# delete the last fc layer.
        self.fc1 = nn.Linear(network.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        self.drop_p = drop_p
        self.CNN_embed_dim = CNN_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ A usual resnet forward pass minus the last layer.
        Hence acts as an encoder.

        Args:
            x (torch.Tensor): Shape (N,C,W,H).
                                 
        Returns:
            torch.Tensor: Shape (batch_size, CNN_embed_dims).
        """
        x = self.net(x)  # ResNet
        x = torch.flatten(x, 1)# flatten output of conv
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        return x