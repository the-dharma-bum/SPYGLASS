import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class FullyConnectedDecoder(pl.LightningModule):

    """ A basic fully connected classifier. """

    def __init__(self, CNN_embed_dim: int=300, hidden_dim=128, drop_p=0.3, num_classes=2) -> None:
        """ Initialize  a basic fully connected decoder.

        Args:
            CNN_embed_dim (int, optional): Output dim of the Encoder. Defaults to 300.
            hidden_dim (int, optional): Fully connected layers size. Defaults to 128.
            drop_p (float, optional): Dropout rate. Defaults to 0.3.
            num_classes (int, optional): Final fully connected layers output size. Defaults to 2.
        """
        super(FullyConnectedDecoder, self).__init__()
        self.CNN_embed_dim = CNN_embed_dim
        self.hidden_dim    = hidden_dim 
        self.num_classes   = num_classes
        self.fc1 = nn.Linear(CNN_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.drop_p = drop_p

    def init_outputs_tensor(self, encoder_out_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, time_depth = encoder_out_tensor.size(0), encoder_out_tensor.size(1)
        outputs = torch.FloatTensor(batch_size, time_depth, self.num_classes)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        return outputs, time_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Decode a tensor already passed through a CNN.

        Args:
            x (torch.Tensor): Shape (batch_size, time_depth, CNN_embed_dim).

        Returns:
            torch.Tensor: Shape (batch_size, time_depth, num_classes).
        """
        outputs, time_depth = self.init_outputs_tensor(x)
        for t in range(time_depth):
            out = self.fc1(x[:,t,:])
            out = F.relu(out)
            out = F.dropout(out, p=self.drop_p, training=self.training)
            out = self.fc2(out)
            outputs[:,t,:] = out
        return outputs