import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


class conv_conv(nn.Module):
    ''' (conv => ReLU) * 2 => maxpool '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        """
        Args:
            in_channels (int): input channel
            out_channels (int): output channel
            bn_momentum (float): batch norm momentum
        """
        super(conv_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class downconv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(downconv, self).__init__()
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X
        

class upconv_concat(nn.Module):
    ''' (conv => ReLU) * 2 => maxpool '''

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        """
        Args:
            in_channels (int): input channel
            out_channels (int): output channel
        """
        super(upconv_concat, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        X1_dim = X1.size()[2]
        X2 = extract_img(X1_dim, X2)
        X1 = torch.cat((X1, X2), dim=1)
        X1 = self.conv(X1)
        return X1


def extract_img(size, in_tensor):
    """
    Args:
        size (int): size of crop
        in_tensor (tensor): tensor to be cropped
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2), int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UNet, self).__init__()
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(starting_filters * 2, starting_filters * 4, bn_momentum)
        self.conv4 = conv_conv(starting_filters * 4, starting_filters * 8, bn_momentum)
        self.upconv1 = upconv_concat(starting_filters * 8, starting_filters * 4, bn_momentum)
        self.upconv2 = upconv_concat(starting_filters * 4, starting_filters * 2, bn_momentum)
        self.upconv3 = upconv_concat(starting_filters * 2, starting_filters, bn_momentum)
        self.conv_out = nn.Conv2d(starting_filters, out_channels, 1, padding=0, stride=1)

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        X = self.conv_out(X)
        return X


class UCAM(nn.Module):
    
    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UCAM, self).__init__()
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(starting_filters * 2, starting_filters * 4, bn_momentum)
        self.conv4 = conv_conv(starting_filters * 4, starting_filters * 8, bn_momentum)
        self.upconv1 = upconv_concat(starting_filters * 8, starting_filters * 4, bn_momentum)
        self.upconv2 = upconv_concat(starting_filters * 4, starting_filters * 2, bn_momentum)
        self.upconv3 = upconv_concat(starting_filters * 2, starting_filters, bn_momentum)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(starting_filters, 1)

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        out = self.gap(X)
        out = self.dense(out)
        return X, out



