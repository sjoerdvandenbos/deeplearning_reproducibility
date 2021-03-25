import torch.nn as nn
from architecture.scsenet.scse_block import SCSEBlock


class UpsampBlock(nn.Module):
    """
    Upsampling block
    ---
    Includes SCSEBlock
    """

    def __init__(self, in_channels, out_channels):
        super(UpsampBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scse = SCSEBlock(in_channels)

    def forward(self, x):
        y = self.relu(x)
        y = self.bn(y)

        return self.scse(y)
