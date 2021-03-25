import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from resnet34.resnet34 import Resnet34


class Resnet34_Pretrained(nn.Module):
    def __init__(self):
        super(Resnet34_Pretrained, self).__init__()
        self.resnet34 = Resnet34()

        # Loading pre-trained model
        pretrained_model = models.resnet34(pretrained=True)
        # Create new dict to remove last two layers.
        new_pretrain_state_dict = OrderedDict()

        for k, v in pretrained_model.state_dict().items():
            if k != "fc.weight" and k != "fc.bias":
                new_pretrain_state_dict[k] = v

        self.resnet34.load_state_dict(new_pretrain_state_dict)

    def forward(self, x):
        return self.resnet34(x)
