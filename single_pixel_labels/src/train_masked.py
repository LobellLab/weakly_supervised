import numpy as np
import time
import os
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models

from tqdm import tqdm

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, params, num_epochs=20, gpu=0):
    """
    Trains a model for all epochs using the provided dataloader.
    """
    t0 = time.time()
    
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {'train_loss': [], 'train_acc': [], 'train_segacc': [],
               'val_loss': [], 'val_acc': [], 'val_segacc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_full = 0
            i = 0

            # Iterate over data.
            for inputs, labels, masks in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                
                labels = labels.to(device)
                label_size = labels.shape[-1]
                target_size = params["label_size"]
                offset = (label_size - target_size)//2
                labels = labels[:,offset:offset+target_size,offset:offset+target_size]

                masks = masks.to(device)
                masks = masks[:,offset:offset+target_size,offset:offset+target_size]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                s = nn.Sigmoid()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    preds = s(outputs) >= 0.5
                    preds = preds.float()

                    outputs_masked = torch.masked_select(outputs, masks)
                    labels_masked = torch.masked_select(labels, masks)
                    preds_masked = torch.masked_select(preds, masks)

                    loss = criterion(outputs_masked, labels_masked)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds_masked == labels_masked.data)
                running_corrects_full += torch.sum(preds == labels.data)

                i += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_segacc = running_corrects_full.double() / (dataset_sizes[phase] * params["label_size"]**2)

            metrics[phase+'_loss'].append(epoch_loss)
            metrics[phase+'_acc'].append(float(epoch_acc.cpu().numpy()))
            metrics[phase+'_segacc'].append(float(epoch_segacc.cpu().numpy()))

            print('{} loss: {:.4f}, single pixel accuracy: {:.4f}, full segmentation accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_segacc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_segacc = epoch_segacc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val single pixel accuracy: {:4f}'.format(best_acc))
    print('Corresponding val full segmentation accuracy: {:.4f}'.format(best_segacc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


