
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from model import LightningModel
from data import SpyGlassDataModule
import config as cfg


def init_trainer():
  early_stopping = EarlyStopping(monitor   = 'val_loss',
                                 mode      = 'min', 
                                 min_delta = 0.001,
                                 patience  = 100,
                                 verbose   = True)
  return Trainer(gpus=1, callbacks = [early_stopping])


def download_outputs(file_module):
  os.system('zip -r /content/output.zip /content/SPYGLASS/lightning_logs/version_0/')
  file_module.download("/content/output.zip")