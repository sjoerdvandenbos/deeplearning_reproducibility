import torch
import torch.nn as nn
from scsenet.cse_block import CSEBlock
from scsenet.sse_block import SSEBlock


class SCSEBlock(nn.Module):
    """
    Spatial and Channel Squeeze and Excitation block
    ---
    Return the block with the most promising values.
    """

    def __init__(self, in_channels, reduction=2):
        super(SCSEBlock, self).__init__()
        self.CSE = CSEBlock(in_channels, reduction)
        self.SSE = SSEBlock(in_channels)

    def forward(self, x):
        return torch.max(self.CSE(x), self.SSE(x))
