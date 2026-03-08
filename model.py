# This is the main 2D U-Net File

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement two consecutive 3x3 Convolutions, 
        # each followed by Batch Normalization and a ReLU activation.
        pass

    def forward(self, x):
        # TODO: Pass x through the layers defined in __init__
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_classes=4):
        super(UNet, self).__init__()
        
        # ENCODER (Downsampling)
        # TODO: Define encoder blocks using the DoubleConv class and MaxPool2d
        
        # BOTTLENECK
        # TODO: Define the bottom-most DoubleConv
        
        # DECODER (Upsampling)
        # TODO: Define upsampling layers (e.g., nn.ConvTranspose2d)
        # TODO: Define decoder blocks (DoubleConv)
        
        # OUTPUT
        # TODO: Define the final 1x1 convolution to map to `out_classes`
        pass

    def forward(self, x):
        # TODO: 1. Pass x through encoder, saving outputs for skip connections
        # TODO: 2. Pass through bottleneck
        # TODO: 3. Pass through decoder, using torch.cat() to concatenate skip connections
        # TODO: 4. Return the final logits
        pass