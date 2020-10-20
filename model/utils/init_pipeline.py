from typing import NewType, Union
import torch.nn as nn
from ..autoencoder import ResNetEncoder, DecoderRNN, FullyConnectedDecoder
from ..fusion import Fusion
from ..aggregate import Aggregate

# type hint
Network = NewType('Network', Union[nn.Sequential, nn.Module])


def init_encoder(cfg):
    return ResNetEncoder(cfg.encoder_base_net,
                         cfg.pretrained,
                         cfg.encoder_hidden_dim_1,
                         cfg.encoder_hidden_dim_2,
                         cfg.encoder_dropout_rate,
                         cfg.embed_dim)


def init_fusion(encoder: Network, cfg):
    return Fusion(encoder,
                  cfg.window,
                  cfg.stride,
                  cfg.early_fusion_dim,
                  cfg.fusion_mode)

def init_decoder(cfg):
    if cfg.decoder == 'fc':
        return FullyConnectedDecoder(cfg.embed_dim,
                                     cfg.fc_hidden_dim,
                                     cfg.decoder_dropout_rate,
                                     cfg.num_classes)
    elif cfg.decoder == 'rnn':
        return DecoderRNN(cfg.embed_dim, 
                          cfg.h_RNN_layers,
                          cfg.h_RNN,
                          cfg.fc_hidden_dim,
                          cfg.decoder_dropout_rate,
                          cfg.num_classes)

def init_aggregation(cfg):
    return Aggregate(cfg.aggregation_mode)

