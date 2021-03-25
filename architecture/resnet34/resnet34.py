import torch.nn as nn
from architecture.resnet34.resnet_block import ResBlock


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        # 3 input image channels, 64 output channels, 7x7 convolution, stride 2, padding 3
        # -- Blue --
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # -- Green --
        # maxpooling stride default is kernel size
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Residual block 1
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1)
        )

        # Residual block 2
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1)
        )

        # Residual block 3
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1)
        )

        # Residual block 4
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512, 1),
            ResBlock(512, 512, 1)
        )

    def forward(self, x):
        # Traverse down the architecture (ResNet34)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.maxpool(y)
        y = self.layer1(y)

        y = self.layer2(y)

        y = self.layer3(y)

        return self.layer4(y)
