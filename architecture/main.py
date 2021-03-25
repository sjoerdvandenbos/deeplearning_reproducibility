import torch
import torch.nn as nn
from resnet34.pretrained_resnet34 import Resnet34_Pretrained
from upsampling_block import UpsampBlock


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pretrained_model = Resnet34_Pretrained()
        # 3 input image channels, 64 output channels, 7x7 convolution, stride 2, padding 3
        # -- Blue --
        self.blueBlock = nn.Sequential(*(list(list(self.pretrained_model.children())[0].children())[:3]))

        # -- Green --
        # maxpooling stride default is kernel size
        self.resBlock1 = nn.Sequential(*(list(list(self.pretrained_model.children())[0].children())[3:5]))
        self.resBlock2 = nn.Sequential(*(list(list(self.pretrained_model.children())[0].children())[5:6]))
        self.resBlock3 = nn.Sequential(*(list(list(self.pretrained_model.children())[0].children())[6:7]))
        self.resBlock4 = nn.Sequential(*(list(list(self.pretrained_model.children())[0].children())[7:]))

        # -- Yellow --
        # 64/128/256 input image channels, 128 output channels, 1x1 convolution, stride 1
        # Green (residual) block to Yellow block (Up to down)
        self.conv_blue_to_yel = nn.Conv2d(64, 128, 1, stride=1)
        self.conv_gr_to_yel_1 = nn.Conv2d(64, 128, 1, stride=1)
        self.conv_gr_to_yel_2 = nn.Conv2d(128, 128, 1, stride=1)
        self.conv_gr_to_yel_3 = nn.Conv2d(256, 128, 1, stride=1)

        # -- Purple --
        # 512 input image channels, 512 output channels, 1x1 convolution, stride 1
        # Green (residual) block to Purple block
        self.conv_gr_to_purp = nn.Conv2d(512, 512, 1, stride=1)
        # Magenta block to Purple block
        self.conv_mag_to_purp_1 = UpsampBlock(256, 256)
        self.conv_mag_to_purp_2 = UpsampBlock(256, 256)
        self.conv_mag_to_purp_3 = UpsampBlock(256, 256)
        self.conv_mag_to_purp_4 = UpsampBlock(256, 256)

        # -- Magenta --
        # 512/256 input image channels, 128 output channels, 2x2 convolution, stride 2
        # Purple (residual) block to Magenta block (Down to up)
        self.conv_purp_to_mag_1 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.conv_purp_to_mag_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_purp_to_mag_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_purp_to_mag_4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_purp_to_mag_5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        # Traverse down the architecture (ResNet34)
        y = self.blueBlock(x)
        y_y1 = self.conv_blue_to_yel(y)

        y = self.resBlock1(y)
        y_y2 = self.conv_gr_to_yel_1(y)

        y = self.resBlock2(y)
        y_y3 = self.conv_gr_to_yel_2(y)

        y = self.resBlock3(y)
        y_y4 = self.conv_gr_to_yel_3(y)

        y = self.resBlock4(y)
        y_y5 = self.conv_gr_to_purp(y)

        # Traverse up the U-based architecture
        y_y5 = self.conv_purp_to_mag_1(y_y5)

        y_y4 = torch.cat((y_y4, y_y5), dim=1)
        y_y4 = self.conv_mag_to_purp_1(y_y4)
        y_y4 = self.conv_purp_to_mag_2(y_y4)

        y_y3 = torch.cat((y_y3, y_y4), dim=1)
        y_y3 = self.conv_mag_to_purp_2(y_y3)
        y_y3 = self.conv_purp_to_mag_3(y_y3)

        y_y2 = torch.cat((y_y2, y_y3), dim=1)
        y_y2 = self.conv_mag_to_purp_3(y_y2)
        y_y2 = self.conv_purp_to_mag_4(y_y2)

        y_y1 = torch.cat((y_y1, y_y2), dim=1)
        y_y1 = self.conv_mag_to_purp_4(y_y1)

        return self.conv_purp_to_mag_5(y_y1)

#
# from torchsummary import summary
#
# net = Net()
# # print(net)
# summary(net, input_size=(3, 320, 480))
