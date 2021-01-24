
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import NewType, Union
from ..autoencoder import ResNetEncoder, DecoderRNN, FullyConnectedDecoder
from ..fusion import Fusion
from ..aggregate import Aggregate

# type hint
Network = NewType('Network', Union[nn.Sequential, nn.Module])


class Builder:

    def __init__(self, cfg):
        self.cfg = cfg

    def optimizer(self, parameters):
        return SGD(parameters,
                   lr=self.cfg.lr,
                   momentum=self.cfg.momentum,
                   nesterov=self.cfg.nesterov, 
                   weight_decay=self.cfg.weight_decay)

    def scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer,
                                 mode=self.cfg.rop_mode,
                                 factor=self.cfg.rop_factor,
                                 patience=self.cfg.rop_patience,
                                 verbose=self.cfg.verbose)

    def encoder(self):
        return ResNetEncoder(self.cfg.encoder_base_net,
                             self.cfg.pretrained,
                             self.cfg.encoder_hidden_dim_1,
                             self.cfg.encoder_hidden_dim_2,
                             self.cfg.encoder_dropout_rate,
                             self.cfg.embed_dim)

    def fusion(self, encoder: Network):
        return Fusion(encoder,
                      self.cfg.window,
                      self.cfg.stride,
                      self.cfg.early_fusion_dim,
                      self.cfg.fusion_mode)

    def decoder(self):
        if self.cfg.decoder == 'fc':
            return FullyConnectedDecoder(self.cfg.embed_dim,
                                         self.cfg.fc_hidden_dim,
                                         self.cfg.decoder_dropout_rate,
                                         self.cfg.num_classes)
        elif self.cfg.decoder == 'rnn':
            return DecoderRNN(self.cfg.embed_dim, 
                              self.cfg.h_RNN_layers,
                              self.cfg.h_RNN,
                              self.cfg.fc_hidden_dim,
                              self.cfg.decoder_dropout_rate,
                              self.cfg.num_classes)

    def aggregation(self):
        return Aggregate(self.cfg.aggregation_mode)

