from typing import List
import torch
import torch.nn as nn
import autoencoder
from data import SpyGlassDataModule
from autoencoder import ResCNNEncoder, DecoderRNN


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                   DATAMODULE CONSTANTS                              | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

video_root:            str = "/homes/l17vedre/Bureau/Sanssauvegarde/cropped/"
data_2d_root:          str = "/homes/l17vedre/Bureau/Sanssauvegarde/2D_sampling1/"
medical_data_csv_path: str = "/homes/l17vedre/SPYGLASS/medical_data.csv"
channels:   int = 3
time_depth: int = 25*1 # first 1 seconds
x_size:     int = 224
y_size:     int = 224       
mean: List[float] = [78.5606, 111.8194, 135.2136]
std:  List[float] = [64.6343,  72.6750,  69.9263]
train_batch_size: int = 2
val_batch_size:   int = 2
num_workers:      int = 1




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                               INIT DATAMODULES & GET BATCHS                         | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

image_datamodule = SpyGlassDataModule(data_2d_root, channels, time_depth, x_size, y_size,
                                      medical_data_csv_path, 'image',
                                      train_batch_size, val_batch_size, num_workers)

video_datamodule = SpyGlassDataModule(video_root, channels, time_depth, x_size, y_size,
                                      medical_data_csv_path, 'video',
                                      train_batch_size, val_batch_size, num_workers)

image_datamodule.setup()
video_datamodule.setup()

image_batch = next(iter(image_datamodule.train_dataloader()))
video_batch = next(iter(video_datamodule.train_dataloader()))


image_inputs, image_targets = image_batch
video_inputs, video_targets = video_batch
image_inputs, image_targets = image_inputs.cuda(), image_targets.cuda()
video_inputs, video_targets  = video_inputs.cuda(), video_targets.cuda()
one_hot_image_targets = nn.functional.one_hot(image_targets).float()
one_hot_video_targets = nn.functional.one_hot(video_targets).float()

print(one_hot_video_targets.size())


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                              INIT MODELS & FORWARD PASS                             | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

cnn = torch.hub.load('pytorch/vision:v0.7.0', 'densenet121',
                     pretrained=False, num_classes=2).cuda()
encoder = ResCNNEncoder().cuda()
decoder = DecoderRNN(num_classes=2).cuda()


# predictions shape: (batch_size, num_classes)
cnn_predictions         = cnn(image_inputs)
autoencoder_predictions = decoder(encoder(video_inputs))




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          LOSS                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

# bce_logits is a sigmoid followed by a BCE.
loss = nn.BCEWithLogitsLoss()

image_loss = loss(cnn_predictions, one_hot_image_targets)
video_loss = loss(autoencoder_predictions, one_hot_video_targets)



# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         PRINTS                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

print(39*'-' + ' IMAGE MODE ' + 39*'-')
print('image inputs size...........: ', image_inputs.size())
print('cnn outputs size............: ', cnn_predictions.size())
print('image targets size..........: ', image_targets.size())
print('one hot image targets size..: ', one_hot_image_targets.size())
print(39*'-' + ' VIDEO MODE ' + 39*'-')
print('video inputs size...........: ', video_inputs.size())
print('autoencoder outputs size....: ', autoencoder_predictions.size())
print('video targets tensor........: ', video_targets)
print('one hot video targets size..: ', one_hot_video_targets.size())
print(42*'-' + ' LOSS ' + 42*'-' )
print('image bce with logits loss..: ', image_loss)
print('video bce with logits loss..: ', video_loss)
print(80*'-')