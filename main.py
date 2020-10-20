""" Main Python file to start training using config.py """

from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateLogger, EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from model.utils.init_model import init_model
from data.utils import init_datamodule
import config as cfg


def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
    Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger      = LearningRateLogger()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=10, verbose=True)
    return Trainer.from_argparse_args(args, callbacks = [lr_logger, early_stopping])


def run_training(datamodule_config, model_config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data) """
    data    = init_datamodule(datamodule_config)
    model   = init_model(model_config)
    trainer = init_trainer()
    trainer.fit(model, data)


if __name__ == '__main__':
    configs = cfg.DataModule, cfg.Model 
    run_training(*configs) 

