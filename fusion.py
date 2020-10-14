import torch
import torch.nn as nn

class Fusion(nn.Module):

    def __init__(self, net):
        self.net = net
    
    def single_frame(self, x):
        batch_size, channels, time_depth, x_size, y_size = x.size()