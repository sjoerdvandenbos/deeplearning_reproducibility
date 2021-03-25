import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block (Green)
    ---
    A special case of highway network without any gates
    in their skip connection. Thus allowing the flow of
    memory (or info) from initial layers to last layers.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.stride_one = (stride == 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if not self.stride_one:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if not self.stride_one:
            x = self.downsample(x)
        y += x

        return self.relu(y)
