import numpy as np

from module.constant import device

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_FACTOR = 12
dropout_rate = 0.0

use_norm = True
if not use_norm: nn.BatchNorm2d,  nn.BatchNorm3d = nn.Identity, nn.Identity


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_norm)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_norm)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_norm),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.downsample(x)
        out = F.gelu(out)

        return out


class UpsampleResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UpsampleResnetBlock, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=not use_norm)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_norm)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=stride-1, bias=not use_norm),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv_transpose1(x)))
        out = self.bn2(self.conv_transpose2(out))

        out += self.upsample(x)
        out = F.gelu(out)

        return out

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1, bias=not use_norm)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv_transpose2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=not use_norm)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=stride - 1, bias=not use_norm),
            nn.BatchNorm3d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv_transpose1(x)))
        out = self.bn2(self.conv_transpose2(out))

        out += self.upsample(x)
        out = F.gelu(out)

        return out

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, CONV_FACTOR, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            ResNetBlock(CONV_FACTOR, CONV_FACTOR * 4, stride=2),
            ResNetBlock(CONV_FACTOR * 4, CONV_FACTOR * 4, stride=1),

            ResNetBlock(CONV_FACTOR * 4, CONV_FACTOR * 8, stride=2),
            ResNetBlock(CONV_FACTOR * 8, CONV_FACTOR * 8, stride=1),

            ResNetBlock(CONV_FACTOR * 8, CONV_FACTOR * 12, stride=2),
            ResNetBlock(CONV_FACTOR * 12, CONV_FACTOR * 12, stride=1),

            ResNetBlock(CONV_FACTOR * 12, CONV_FACTOR * 16, stride=1),
            ResNetBlock(CONV_FACTOR * 16, CONV_FACTOR * 16, stride=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Restore1Net(nn.Module):
    def __init__(self):
        super(Restore1Net, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(CONV_FACTOR * 16, CONV_FACTOR * 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.Conv2d(CONV_FACTOR * 16, CONV_FACTOR * 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(CONV_FACTOR * 8, CONV_FACTOR * 8, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.Conv2d(CONV_FACTOR * 8, CONV_FACTOR * 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(CONV_FACTOR * 4, CONV_FACTOR * 4, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.Conv2d(CONV_FACTOR * 4, CONV_FACTOR * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(CONV_FACTOR*2, CONV_FACTOR*1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(CONV_FACTOR * 1, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.decoder(x)

class Restore3Net(nn.Module):
    def __init__(self):
        super(Restore3Net, self).__init__()

        self.decoder = nn.Sequential(
            UpsampleResnetBlock(CONV_FACTOR * 64, CONV_FACTOR * 32,  stride=2),
            UpsampleResnetBlock(CONV_FACTOR * 32, CONV_FACTOR * 16, stride=2),
            UpsampleResnetBlock(CONV_FACTOR * 16, CONV_FACTOR * 16, stride=1),

            UpsampleResnetBlock(CONV_FACTOR * 16, CONV_FACTOR * 8, stride=2),
            UpsampleResnetBlock(CONV_FACTOR * 8, CONV_FACTOR * 8, stride=1),

            UpsampleResnetBlock(CONV_FACTOR * 8, CONV_FACTOR * 4, stride=2),
            UpsampleResnetBlock(CONV_FACTOR * 4, CONV_FACTOR * 4, stride=1),

            UpsampleResnetBlock(CONV_FACTOR * 4, CONV_FACTOR * 2, stride=2),

            nn.ConvTranspose2d(CONV_FACTOR * 2, CONV_FACTOR * 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(CONV_FACTOR * 1, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.decoder(x)

class CNNBottleNeck(nn.Module):
    def __init__(self):
        super(CNNBottleNeck, self).__init__()
        self.bottleneck = nn.Sequential(
            ResNetBlock(CONV_FACTOR * 16*2, CONV_FACTOR * 16, stride=1),

            ResNetBlock(CONV_FACTOR * 16, CONV_FACTOR * 32, stride=2),
            ResNetBlock(CONV_FACTOR * 32, CONV_FACTOR * 32, stride=1),

            ResNetBlock(CONV_FACTOR * 32, CONV_FACTOR * 64, stride=2),
            ResNetBlock(CONV_FACTOR * 64, CONV_FACTOR * 64, stride=1),

            nn.Conv2d(CONV_FACTOR * 64, CONV_FACTOR * 64, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.bottleneck(x)
        return x



class Net3D(nn.Module):
    def __init__(self):
        super(Net3D, self).__init__()

        self.net = nn.Sequential(
            BasicBlock3D(CONV_FACTOR * 12, CONV_FACTOR * 12, stride=1),
            BasicBlock3D(CONV_FACTOR * 12, CONV_FACTOR * 10, stride=2),
            BasicBlock3D(CONV_FACTOR * 10, CONV_FACTOR * 8, stride=2),
            BasicBlock3D(CONV_FACTOR * 8, CONV_FACTOR * 6, stride=2),
            BasicBlock3D(CONV_FACTOR * 6, CONV_FACTOR * 4, stride=2),
            BasicBlock3D(CONV_FACTOR * 4, CONV_FACTOR * 2, stride=np.int32([2, 1, 1])),
            nn.ConvTranspose3d(CONV_FACTOR*2, CONV_FACTOR*1, kernel_size=3, stride=1, padding=1), nn.GELU(),
            nn.Conv3d(CONV_FACTOR, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], CONV_FACTOR * 12, 4, 4, 4)
        x = self.net(x)
        return x


class Model3dMultiTask(nn.Module):
    def __init__(self):
        super(Model3dMultiTask, self).__init__()
        self.encoder1 = EncoderNet()
        self.encoder2 = EncoderNet()

        self.cnn_bottleneck = CNNBottleNeck()

        self.cnn3d = Net3D()

        self.restore11 = Restore1Net()
        self.restore12 = Restore1Net()

        self.restore21 = Restore3Net()
        self.restore22 = Restore3Net()

    def forward(self, x1, x2):
        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)

        general_information = torch.cat([features1, features2], dim=1)

        cnn_output = self.cnn_bottleneck(general_information)

        restored_img11 = self.restore11(features1)
        restored_img12 = self.restore12(features2)

        restored_img21 = self.restore21(cnn_output)
        restored_img22 = self.restore22(cnn_output)

        distribution = self.cnn3d(cnn_output)

        return (
            distribution,
            restored_img11,
            restored_img12,
            restored_img21,
            restored_img22,
        )

if __name__ == "__main__":
    model = Model3dMultiTask()
    model.to(device)

    from torchview import draw_graph

    draw_graph(
        model, depth=1,
        input_data=[torch.rand((1, 1, 64, 192)), torch.rand((1, 1,  64, 192))],
        expand_nested=True, save_graph=True, filename='model'
    )
