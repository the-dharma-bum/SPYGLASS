""" Base Model Class: A Lighning Module

This class implements all the logic code.
This model class will be the one to be fit by a Trainer.
 """

from typing import Tuple, Dict
import torch
from torch.optim import SGD
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from autoencoder import ResCNNEncoder, DecoderRNN


class LightningModel(pl.LightningModule):
    
    """ LightningModule handling everything training related.

    This model accepts to "mode", ie two (dataset,architecture) config:
        - 'image': 2d dataset and a densenet classifier.
        - 'video': 3d dataset and a ResNet+RNN autoencoder.
    
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
        if self.hparams.mode == 'image':
            self.net = torch.hub.load('pytorch/vision:v0.7.0', 'densenet121', pretrained=False, num_classes=2)
        elif self.hparams.mode == 'video':
            self.encoder, self.decoder = ResCNNEncoder(), DecoderRNN(num_classes=2)
        self.criterion = BCEWithLogitsLoss()
        self.accuracy  = pl.metrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ If in image mode, returns the output of a classic densenet classifier.
        If in video mode, returns the output of an autoencoder (Resnet, RNN).

        Notation:
            (N,C,T,W,H) = (batch_size, num_channels, time_depth, x_size, y_size).

        Args:
            x (torch.Tensor): Input batch of shape:
                                - (N,C,W,H) in image mode.
                                - (N,T,W,H) in video mode.

        Returns:
            torch.Tensor: Predicted class for each element of input batch. Shape: (N,num_classes).
        """
        if self.hparams.mode=='image':
            return self.net(x)
        if self.hparams.mode=='video':
            return self.decoder(self.encoder(x))

    def label_smoothing(self, one_hot_targets: torch.Tensor) -> torch.Tensor:
        """ The one true label will be 1-epsilon instead of 1,
            and others false labels will be epsilon instead of 0.

        Args:
            one_hot_targets ([type]): Shape (N,num_classes).

        Returns:
            [type]: Shape (N, num_classes).
        """
        for target in one_hot_targets:
            for x in target:
                if x.item() == 0.:
                    x += self.hparams.smoothing
                else:
                    x -= self.hparams.smoothing
        return one_hot_targets

    def encode_targets(self, targets: torch.Tensor, train: bool=True) -> torch.Tensor:
        """ One hot encoding and label smoothing. 

        Args:
            targets (torch.Tensor): Shape (N,). Contains one int in [0,1] for each batch element.
            train (bool, optional): Prevents the use of label smoothing if False. Defaults to True.

        Returns:
            torch.Tensor: Shape (N, num_classes). 
                          Contains one list in [[0.,1.],[1.,0.]] for each batch element.
        """
        one_hot_targets = one_hot(targets, num_classes=2).float()
        if train and self.hparams.use_label_smoothing:
            one_hot_targets = self.label_smoothing(one_hot_targets)
        return one_hot_targets

    def get_arch_parameters(self) -> torch.nn.parameter.Parameter:
        """ Get current architecture parameters (to give them to an optimizer).

        If in video mode, concatenate encoder and decoder params.
        Handles multi GPUs training by using DataParallel if needed.

        Returns:
            torch.nn.parameter.Parameter: cnn or crnn parameters.
        """
        if self.hparams.mode == 'image':
            return self.net.parameters()
        # Bellow code is for video mode.
        # Parallelize model to multiple GPUs if needed.
        if torch.cuda.device_count() > 1:
            cnn_encoder = DataParallel(self.encoder)
            rnn_decoder = DataParallel(self.decoder)
            # Combine all EncoderCNN + DecoderRNN parameters
            crnn_params = list(cnn_encoder.module.fc1.parameters()) + \
                          list(cnn_encoder.module.bn1.parameters()) + \
                          list(cnn_encoder.module.fc2.parameters()) + \
                          list(cnn_encoder.module.bn2.parameters()) + \
                          list(cnn_encoder.module.fc3.parameters()) + \
                          list(rnn_decoder.parameters())
        elif torch.cuda.device_count() == 1:
            # Combine all EncoderCNN + DecoderRNN parameters
            crnn_params = list(self.encoder.fc1.parameters()) + \
                          list(self.encoder.bn1.parameters()) + \
                          list(self.encoder.fc2.parameters()) + \
                          list(self.encoder.bn2.parameters()) + \
                          list(self.encoder.fc3.parameters()) + \
                          list(self.decoder.parameters())
        return crnn_params

    def configure_optimizers(self) -> Dict:
        """ Instanciate an optimizer and a learning rate scheduler to be used during training.

        Returns:
            (Dict): Dict containing the optimizer(s) and learning rate scheduler(s) to be used by a Trainer
                    object using this model. 
                    The monitor key is used by the ReduceLROnPlateau scheduler.                        
        """
        optimizer = SGD(self.get_arch_parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)
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
        if self.hparams.mode == 'image':
            inputs, targets = batch
        elif self.hparams.mode == 'video':
            inputs, targets  = batch['clip'], batch['target']
        encoded_targets = self.encode_targets(targets)
        outputs = self(inputs)
        loss    = self.criterion(outputs, encoded_targets)
        acc     = self.accuracy(outputs,  encoded_targets)
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
        if self.hparams.mode == 'image':
            inputs, targets = batch
        elif self.hparams.mode == 'video':
            inputs, targets  = batch['clip'], batch['target']
        encoded_targets = self.encode_targets(targets, train=False)
        outputs = self(inputs)
        loss    = self.criterion(outputs, encoded_targets)
        acc     = self.accuracy(outputs,  encoded_targets)
        self.log('Loss/Validation', loss)
        self.log('Accuracy/Validation', acc, logger=True)
        return {'val_loss': loss, 'acc': acc}

    def test_step(self, batch: torch.Tensor, batch_idx) ->  torch.Tensor:
        """ Not implemented. """