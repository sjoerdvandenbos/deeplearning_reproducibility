import torch.nn as nn


class SSEBlock(nn.Module):
    """
    Channel Squeeze and Spatial Excitation block
    ---
    Spatial-wise focus

    It behaves like a spatial attention map indicating
    where the network should focus more to aid the
    segmentation.

    Assign importance to spatial locations sort of
    telling where features are better to focus
    instead of reweighing which features are more
    important.
    """

    def __init__(self, in_channels):
        super(SSEBlock, self).__init__()
        # Output channel = 1, 1x1 convolution
        self.conv = nn.Conv2d(in_channels, 1, 1)
        # Do He, K. et al. init as described in the paper.
        nn.init.kaiming_uniform_(self.conv, mode='fan_in', nonlinearity='relu')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()

        y = self.conv(x)
        y = self.sigmoid(y)

        return x * y.view(batch_size, 1, H, W)
