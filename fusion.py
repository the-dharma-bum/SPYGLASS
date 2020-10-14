import torch
import torch.nn as nn

class Fusion(nn.Module):

    def __init__(self, net: nn.Module,
                 hidden_dim1: int=2048, hidden_dim2: int=512, num_classes: int=2) -> None:
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.hidden_dim1 = hidden_dim1
        # final classifier 
        self.bn1 = nn.BatchNorm1d(hidden_dim1, momentum=0.01)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, momentum=0.01)
        self.fc3 = nn.Linear(hidden_dim2, self.num_classes)

    def init_early_fusion(self, early_fusion_hidden_dim : int, window: int):
        self.conv3d = nn.Conv3d(3,early_fusion_hidden_dim, kernel_size=3, padding=1)
        self.net.features[0] = nn.Conv2d(early_fusion_hidden_dim * window, 64,
                                         kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if torch.cuda.is_available():
            self.conv3d = self.conv3d.cuda()
            self.net.features[0] = self.net.features[0].cuda()
    
    def init_first_fully_connected(self, num_output_filters: int) -> None:
        self.fc1 = nn.Linear(num_output_filters, self.hidden_dim1)
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()

    def remove_last_fully_connected(self):
        modules  = list(self.net.children())[:-1]
        self.net = nn.Sequential(*modules)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        num_output_filters = x.size(1)
        self.init_first_fully_connected(num_output_filters)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        outputs = self.fc3(x)
        return outputs
    
    def single_frame(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_depth = x.size(0), x.size(2)
        outputs = torch.FloatTensor(time_depth, batch_size, self.num_classes)
        for t in range(time_depth):
            outputs[t,:,:] = self.net(x[:,:,t,:,:])
        return outputs.mean(axis=0)

    def late_fusion(self, x: torch.Tensor, distance: int=5) -> torch.Tensor:
        self.remove_last_fully_connected()
        batch_size, time_depth = x.size(0), x.size(2)
        outputs = torch.FloatTensor(time_depth, batch_size, self.num_classes)
        for t in range(time_depth-distance):
            output_early = self.net(x[:,:,t,:,:])
            output_late  = self.net(x[:,:,t+distance-1,:,:])
            concatenated_outputs = torch.cat((output_early, output_late), 1)
            concatenated_outputs = torch.flatten(concatenated_outputs, 1)
            outputs[t,:,:] = self.classifier(concatenated_outputs)
        return outputs.mean(axis=0)

    def early_fusion(self, x: torch.Tensor, early_fusion_hidden_dim: int=3,
                     window: int=4, stride: int=1) -> torch.Tensor:
        self.init_early_fusion(early_fusion_hidden_dim, window)
        batch_size, time_depth = x.size(0), x.size(2)
        outputs = torch.FloatTensor(time_depth, batch_size, self.num_classes)
        for t in range(0,time_depth-window,stride): 
            # early fusion
            frame_window = x.narrow(2,t,window)
            embedded_vector = self.conv3d(frame_window)
            # shape after conv_3d: (batch_size,early_fusion_hidden_dim,windows,x_size,y_size)
            embedded_vector = torch.flatten(embedded_vector, start_dim=1, end_dim=2) # IS THIS OK ??? 
            # shape after flattening: (batch_size, early_fusion_hidden_dim*windows, x_size, y_size)
            outputs[t,:,] = self.net(embedded_vector)
        return outputs.mean(axis=0)

