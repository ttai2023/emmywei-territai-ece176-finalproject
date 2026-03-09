# 2D U-Net Model

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement two consecutive 3x3 Convolutions with Batch Normalization and ReLU activation.

        # block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # block 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x      

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_classes=4):
        super(UNet, self).__init__()
        
        # ENCODER (Downsampling)
        
        # block 1
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)
        
        # DECODER (Upsampling)

        # block 1
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64) 
       
        # block 2
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128) 

        # block 3
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256) 

        # block 4
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512) 
        
        # OUTPUT
        self.finalconv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # TODO: 1. Pass x through encoder, saving outputs for skip connections
        # TODO: 2. Pass through bottleneck
        # TODO: 3. Pass through decoder, using torch.cat() to concatenate skip connections
        # TODO: 4. Return the final logits

        # ENCODER
        
        # block 1
        x1 = self.enc1(x)          
        p1 = self.pool1(x1)

        # block 2
        x2 = self.enc2(p1)          
        p2 = self.pool2(x2)

        # block 3
        x3 = self.enc3(p2)          
        p3 = self.pool3(x3)

        # block 4
        x4 = self.enc4(p3)          
        p4 = self.pool4(x4)

        # BOTTLENECK
        b = self.bottleneck(p4)

        # DECODER

        # block 4 
        up4 = self.upconv4(b)
        merge4 = torch.cat([up4, x4], dim=1) 
        d4 = self.dec4(merge4)

        # block 3
        up3 = self.upconv3(d4)
        merge3 = torch.cat([up3, x3], dim=1) 
        d3 = self.dec3(merge3)

        # block 2
        up2 = self.upconv2(d3)
        merge2 = torch.cat([up2, x2], dim=1) 
        d2 = self.dec2(merge2)

        # block 1 (Top)
        up1 = self.upconv1(d2)
        merge1 = torch.cat([up1, x1], dim=1) 
        d1 = self.dec1(merge1)

        out = self.finalconv(d1)
        return out