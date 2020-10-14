import numpy as np
import torch
import torch.nn as nn
from fusion import Fusion

X_SIZE = 200
Y_SIZE = 200
NUM_CHANNELS = 3
BATCH_SIZE = 2

NUM_CLASSES = 2
HIDDEN_DIM1 = 2048
HIDDEN_DIM2 = 512

DISTANCE    = 5  # late fusion 
WINDOW      = 10 # early fusion
STRIDE      = 1
EARLY_FUSION_HIDDEN_DIM = 3


# Init dummy network and input tensor
net = torch.hub.load('pytorch/vision:v0.7.0', 'densenet121', 
                     pretrained=False, num_classes=NUM_CLASSES).cuda()
# Shape : (batch_size, channels, time_depth, x_size, y_size)
dummy_tensor = torch.FloatTensor(BATCH_SIZE,NUM_CHANNELS,10,X_SIZE,Y_SIZE).cuda()



print(f'Expected Shape: ({BATCH_SIZE}, {NUM_CLASSES})')

fusion = Fusion(net).cuda()
single_frame_outputs = fusion.single_frame(dummy_tensor)
print(single_frame_outputs.size())

fusion = Fusion(net).cuda()
late_fusion_outputs  = fusion.late_fusion(dummy_tensor, DISTANCE)
print(late_fusion_outputs.size())

fusion = Fusion(net).cuda()
early_fusion_outputs = fusion.early_fusion(dummy_tensor, EARLY_FUSION_HIDDEN_DIM, WINDOW, STRIDE)
print(early_fusion_outputs.size())

