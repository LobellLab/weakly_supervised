import argparse
import numpy as np
import time
import os
import copy
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

from datasets import masked_tile_dataloader
from unet import UNet
from train_masked import train_model



parser = argparse.ArgumentParser(description="Masked UNet for Remote Sensing Image Segmentation")

# directories
parser.add_argument('--model_dir', type=str)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# model and dataset hyperparameters
param_file = os.path.join(args.model_dir, 'params.json')
with open(param_file) as f:
    params = json.load(f)

# dataloaders
train_files = []
val_files = []
with open(params['train_file']) as f:
    for l in f:
        if '.npy' in l:
            train_files.append(l[:-1])
with open(params['val_file']) as f:
    for l in f:
        if '.npy' in l:
            val_files.append(l[:-1])

dataloaders = {}
dataloaders['train'] = masked_tile_dataloader(params['tile_dir'], 
                                              train_files,
                                              params['mask_dir'],
                                              augment=True, 
                                              batch_size=params['batch_size'], 
                                              shuffle=params['shuffle'], 
                                              num_workers=params['num_workers'], 
                                              n_samples=params['n_train'])
dataloaders['val'] = masked_tile_dataloader(params['tile_dir'],
                                            val_files,
                                            params['mask_dir'],
                                            augment=False,
                                            batch_size=params['batch_size'],
                                            shuffle=params['shuffle'],
                                            num_workers=params['num_workers'], 
                                            n_samples=params['n_val'])
dataset_sizes = {}
dataset_sizes['train'] = len(train_files)
dataset_sizes['val'] = len(val_files)

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=7, out_channels=1, 
             starting_filters=params['starting_filters'], 
             bn_momentum=params['bn_momentum'])

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                       betas=(params['beta1'], params['beta2']), weight_decay=params['weight_decay'])

# Decay LR by a factor of gamma every X epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                       step_size=params['decay_steps'],
                                       gamma=params['gamma'])

model, metrics = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler,
                             params, num_epochs=params['epochs'], gpu=args.gpu)

if params['save_model']:
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))

with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)
