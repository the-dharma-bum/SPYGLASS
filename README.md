<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                               MAIN TITLE                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SPYGLASS     

Pytorch implementation for hepatic bile duct pathologies detection. 

Based on [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                          TABLE OF CONTENTS                                         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# SUMMARY

- [To Do](#to-do)
     - [New Features](#new-features)
     - [Bugfixes](#bugfixes)
- [Last Commit Changes Log](#last-commit-changes-log)
- [Installation](#installation)
- [Usage](#usage)
- [Fastai integration](#fastai-integration)


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                  TO DO                                             |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# To Do
[(Back to top)](#summary)

## New features:

| Features                                                 |      Status      |     Owner    |
|----------------------------------------------------------|:----------------:|:------------:|
| Adaptative sampling factor to balance class              |      TO DO       |              |
| Add optimizer and scheduler choice                       |      TO DO       |              |
| Move optimizer and scheduler params to config.py         |      TO DO       |              |


## Bugfixes:

| Bugfixes                                                 |      Status      |     Owner    |
|----------------------------------------------------------|:----------------:|:------------:|
| Nothing for now (!)                                      |                  |              |

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              CHANGES LOG                                           |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->


# Last Commit Changes Log

- Add fusion implementation from [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)


<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                              INSTALLATION                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Installation
[(Back to top)](#summary)

Clone repo:

```git clone https://github.com/the-dharma-bum/spyglass```

Install dependancies by running: 

``` pip install requirements.txt ```

(Be careful this can be pretty long.)

<!--
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                                 USAGE                                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
 -->

# Usage
[(Back to top)](#summary)

Data related hyperparameters (batch sizes, transforms, ...) can be configured in data/datamodule.py
Paths and some preprocessing parameters can be configured in config.py
Some basic routine are implemented in main.py.
One can choose which one to call and then run:

```python main.py ```

This command accepts a huge number of parameters. Run 

```python main.py -h ```

to see them all, or refer to [documentation de Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Some usefull parameters:

- ```--gpus n```: launch training on n gpus
- ```--distributed_backend ddp``` : use DistributedDataParallel as multi gpus training backend.
- ```--fast_dev_run True``` : launch a training loop (train, eval, test) on a single batch. Use it to debug.

If fast_dev_run doesn't suit your debugging need (for instance if you wanna see what's happening between two epochs), 
you can use:

- ```--limit_train_batches i --limit_val_batches j --max_epochs k```
     
     i,j,k being of course three integers of your choice.




# Fastai Integration
[(Back to top)](#summary)

It's very easy to integrate a Lighning code into the Fastai training environnement.

One must define

- a model (see model.py):
```python
from model import LightningModel

model = LightningModel()
```

- a datamodule (see data/datamodule.py):
```python
from datamodule import SpyGlassDataModule

dm = SpyGlassDataModule(args, kwargs)
```

Using this datamodule, two fastai DataLoaders can be defined like this:
```python
from fastai.vision.all import DataLoaders

data = Dataloaders(dm.train_dataloader(), dm.val_dataloader()).cuda()
```

Then a Learner can be defined and used like a standart Fastai code, for instance:
```python
learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam, metrics=accuracy)
learn.fit_one_cycle(1, 0.001)
```

This makes every fastai training fonctionalites availables (callbacks, transforms, visualizations ...).