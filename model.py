""" Base Model Class: A Lighning Module

This class implements all the logic code.
This model class will be the one to be fit by a Trainer
 """

from typing import Tuple, Dict
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from utils import LabelSmoothingCrossEntropy


class LightningModel(pl.LightningModule):
    
    """ LightningModule handling everything training related.
    
    Some attributes and methods used aren't explicitely defined here but comes from the
    LightningModule class. Please refer to the Lightning documentation for further details.

    Note that Lighning Callbacks handles tensorboard logging, early stopping, and auto checkpoints
    for this class. Those are gave to a Trainer object. See init_trainer() in main.py.
    """

    def __init__(self, **kwargs) -> None:
        """ Instanciate a Lightning Model. 
        The call to the Lightning method save_hyperparameters() make every hp
        accessible through self.hparams. e.g: self.hparams.use_label_smoothing.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.hub.load('pytorch/vision:v0.7.0', 'densenet121', pretrained=False)
        self.criterion = self._init_criterion()
        self.accuracy  = pl.metrics.Accuracy()
        
    def _init_criterion(self) -> torch.nn:
        """ returns the loss to be used by a LightningModel object,
            possibly using label smoothing.
        """
        if self.hparams.use_label_smoothing:
            return LabelSmoothingCrossEntropy(smoothing=self.hparams.smoothing,
                                              reduction=self.hparams.reduction)
        return CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Calls the forward method of self.net. 

        Args:
            x (torch.Tensor): Input batch of shape (N,C,W,H).

        Returns:
            torch.Tensor: Predicted class for each element of input batch. Shape: (N,).
        """
        return self.net(x)

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.

        Returns:
            (Dict): Dict containing the optimizer(s) and learning rate scheduler(s) to be used by a Trainer
                    object using this model. 
                    The monitor key is used by the ReduceLROnPlateau scheduler.                        
        """
        optimizer = SGD(self.net.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor=0.2, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Note that the backward pass is handled under the hood by Pytorch Lightning.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be classified, of shape (N,C,W,H).
                                  The other being the target classes, of shape (N,).
            batch_idx ([type]): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks' methods
                  from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss    = self.criterion(outputs, targets)
        acc     = self.accuracy(outputs, targets)
        self.log('Loss/Train', loss)
        self.log('Accuracy/Train', acc, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """ Perform the classic training step (infere + compute loss) on a batch.

        Args:
            batch (torch.Tensor): Tuple of two tensor. 
                                  One given to the network to be classified, of shape (N,C,W,H).
                                  The other being the target classes, of shape (N,).
            batch_idx (int): Dataset index of the batch. In range (dataset length)/(batch size).

        Returns:
            Dict: Scalars computed in this function. Note that this dict is accesible from 'hooks' methods
                  from Lightning, e.g on_epoch_start, on_epoch_end, etc...
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss    = self.criterion(outputs, targets)
        acc     = self.accuracy(outputs, targets)
        self.log('Loss/Validation', loss)
        self.log('Accuracy/Validation', acc, logger=True)
        return {'val_loss': loss, 'acc': acc}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ (Almost) Not implemented. 
        
        Note that this relies on the test_dataloader from the DataModule and 
        this dataloder returns inputs only and no targets.
        """
        inputs  = batch
        outputs = self(inputs)
        return outputs