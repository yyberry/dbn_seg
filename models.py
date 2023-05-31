import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchinfo import summary

#Double Convolution in 3D U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        if in_ch <= out_ch:
            channels = out_ch//2
        else:
            channels = in_ch//2
        self.doubleconv = nn.Sequential(
            nn.Conv3d(in_ch, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x

#Downsampling in 3D U-Net
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

#Upsampling in 3D U-Net
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + in_ch//2, out_ch)

    def forward(self, x1, x2):
        # x1 is upsampling data
        # x2 is concat data
        x1 = self.up(x1)

        output = torch.cat([x2, x1], dim=1)
        return self.conv(output)

# The last Conv in 3D U-Net
class LastConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LastConv, self).__init__()
        # convolution + sigmoid
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigm(x)
        return x

# 3D U-Net
class UNet3D(nn.Module):
    def __init__(self, in_ch, num_classes = 1):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = LastConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x = self.outc(x7)
        return x
    
## reference: Beliveau, et al. "HFP‐QSMGAN: QSM from homodyne‐filtered phase images." Magnetic Resonance in Medicine (2022). https://doi.org/10.1002/mrm.29260
class Discriminator(nn.Module):

    def __init__(
                self,
                output_kernel,
                depth=4,
                n_base_filters=8,
                batch_norm=False
            ):

        super().__init__()

        blocks = []
        # binarize the input by using nn.Threshold
        blocks += [nn.Threshold(0.5, 0.0)]
        for d in range(depth):
            n_filters = n_base_filters*(2**d)
            if d == 0:
                in_filters = 1
                batch_norm_ = False
            else:
                in_filters = n_filters//2
                batch_norm_ = batch_norm

            blocks += [nn.Conv3d(
                            in_filters, n_filters,
                            kernel_size=4,
                            stride=2,
                            padding=1
                        )]
            if batch_norm_:
                blocks += [nn.BatchNorm3d(num_features=n_filters)]
            blocks += [nn.LeakyReLU(0.2)]

        blocks += [nn.Conv3d(
                        n_filters, 1,
                        kernel_size=output_kernel,
                        padding=0
                    )]

        self.model = nn.Sequential(*blocks)


    def forward(self, input):
        return self.model(input)
