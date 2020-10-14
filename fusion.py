import torch
import torch.nn as nn

class Fusion(nn.Module):

    """ Pytorch Implementation of:
        Large-scale Video Classification with Convolutional Neural Networks.
    
        Implements:
            - single frame
            - early fusion
            - late fusion
    """

    def __init__(self, net: nn.Module,
                 hidden_dim1: int=2048, hidden_dim2: int=512, num_classes: int=2) -> None:
        """ Initalise a fusion environnement.
        
        We need to remove the last fully connected layer of the base network and replace it
        by our own clasifier.
        Hence the two hidden dimensions and the num classes.

        Args:
            net (nn.Module): a base network on which to perform fusion.
            hidden_dim1 (int, optional): Classifier's first hidden dimension. Defaults to 2048.
            hidden_dim2 (int, optional): Classifier's second hidden dimension. Defaults to 512.
            num_classes (int, optional): Last fully connected layer output dimension. Defaults to 2.
        """
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.hidden_dim1 = hidden_dim1
        # final classifier 
        self.bn1 = nn.BatchNorm1d(hidden_dim1, momentum=0.01)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, momentum=0.01)
        self.fc3 = nn.Linear(hidden_dim2, self.num_classes)

    def init_early_fusion(self, early_fusion_hidden_dim : int, window: int) -> None:
        """ Prepare the network to perfom an early fusion.
        As we take a 3d inputs (of depth called window),we replace the first Conv2d
        of the base network by the following:
            1. a Conv3d taking with 3 inputs channels and a output dim defined by
               early_fusion_hidden_dim.
            2. a Conv2d with an input dim equals to early_fusion_hidden_dim and an output 
               dim of 64 so that it matches the output dim of the initial Conv2d.

        Args:
            early_fusion_hidden_dim (int):  the intermediate state's hidden dim. 
            window (int): how many consecutive frames do we fuse.
        """
        self.conv3d = nn.Conv3d(3,early_fusion_hidden_dim, kernel_size=3, padding=1)
        self.net.features[0] = nn.Conv2d(early_fusion_hidden_dim * window, 64,
                                         kernel_size=(7, 7), stride=(2, 2),
                                         padding=(3, 3), bias=False)
        if torch.cuda.is_available():
            self.conv3d = self.conv3d.cuda()
            self.net.features[0] = self.net.features[0].cuda()
    
    def init_first_fully_connected(self, num_output_filters: int) -> None:
        """ Initialise the first fully connected of self.classifier.

        When doing late fusion, we concatenate to frames outputs at the end
        of the network so we need to adust the first fully connected layer
        input dim accordingly.

        Args:
            num_output_filters (int): How many filters where outputed by the 
                                      base network before its first fc.
        """
        self.fc1 = nn.Linear(num_output_filters, self.hidden_dim1)
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()

    def remove_last_fully_connected(self):
        """ Removes the last layer of the base network. """
        modules  = list(self.net.children())[:-1]
        self.net = nn.Sequential(*modules)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        """ Three stages of (BatchNorm + FC) to classify a single frame.

        Args:
            x (torch.Tensor): The output of a network without its
                              last fc layer.

        Returns:
            torch.Tensor: The output for one given frame, of shape (batc_size, num_classes).
        """
        num_output_filters = x.size(1)
        self.init_first_fully_connected(num_output_filters)
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        outputs = self.fc3(x)
        return outputs
    
    def single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """ Single Frame Fusion.

        Predicts all frames independantly and outputs the mean output
        of all frame.

        Args:
            x (torch.Tensor): A 5d tensor: (batch_size, channesl, time_depth, x_size, y_size)
                              
        Returns:
            torch.Tensor: The final output, of shape (batc_size, num_classes).
        """
        batch_size, time_depth = x.size(0), x.size(2)
        outputs = torch.FloatTensor(time_depth, batch_size, self.num_classes)
        for t in range(time_depth):
            outputs[t,:,:] = self.net(x[:,:,t,:,:])
        return outputs.mean(axis=0)

    def late_fusion(self, x: torch.Tensor, distance: int=5) -> torch.Tensor:
        """ Remove the last fully connected layer. 
        Encode two frames separated by a defined distance (pass through the network
        minus the last fc layer). 
        Concatenate the encoded frames.
        Use self.classier on that concatenated tensor.
        Args:
            x (torch.Tensor): A 5d tensor: (batch_size, channesl, time_depth, x_size, y_size)
            distance (int, optional): How far apart are the two frames to fuse. Defaults to 5.

        Returns:
            torch.Tensor: The final output, of shape (batc_size, num_classes).
        """
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
        """
        Fuse a window of consecutive frames at the beginning of the network
        by doing a Conv3d followed by a flattening followed by a Conv2d.

        Args:
            x (torch.Tensor): A 5d tensor: (batch_size, channesl, time_depth, x_size, y_size)
            early_fusion_hidden_dim (int, optional): Output dim of the Conv3d. Defaults to 3.
            window (int, optional): How many consecutive frames to fuse. Defaults to 4.
            stride (int, optional): Stride between each window. Defaults to 1.

        Returns:
            torch.Tensor: The final output, of shape (batc_size, num_classes). 
        """
        self.init_early_fusion(early_fusion_hidden_dim, window)
        batch_size, time_depth = x.size(0), x.size(2)
        outputs = torch.FloatTensor(time_depth, batch_size, self.num_classes)
        for t in range(0,time_depth-window,stride): 
            # shape before get window: (batch_size, num_channels, time_depth, x_size, y_size) 
            frame_window = x.narrow(2,t,window)
            # shape after get window: (batch_size, num_channels, window, x_size, y_size)
            embedded_vector = self.conv3d(frame_window)
            # shape after conv_3d: (batch_size, early_fusion_hidden_dim, window, x_size, y_size)
            embedded_vector = torch.flatten(embedded_vector, start_dim=1, end_dim=2) # IS THIS OK ??? 
            # shape after flattening: (batch_size, early_fusion_hidden_dim * windows, x_size, y_size)
            outputs[t,:,] = self.net(embedded_vector)
        return outputs.mean(axis=0)

