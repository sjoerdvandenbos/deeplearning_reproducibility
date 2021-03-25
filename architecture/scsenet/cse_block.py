import torch.nn as nn


class CSEBlock(nn.Module):
    """
    Spatial Squeeze and Channel Excitation block
    ---
    Channel-wise focus

    Recalibrates the channels by incorporating global
    spatial information. It provides a receptive field
    of whole spatial extent at the fc's.

    Assign each channel (feature) of a convolutional
    block (feature map) a different weightage
    (excitation) based on how important each channel
    is (squeeze) instead of equally weighing each
    feature. This improves channel interdependencies.
    """

    def __init__(self, in_channels, reduction=2):
        super(CSEBlock, self).__init__()
        # Global pooling == AdaptiveAvgPool2d
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

        # Do He. K. et al. init for Upsampling decoder
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()

        avg_pool_x = self.avg_pool(x).view(batch_size, num_channels)

        y = self.fc1(avg_pool_x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        return x * y.view(batch_size, num_channels, 1, 1)
