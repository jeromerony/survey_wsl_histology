from collections import OrderedDict
from sacred import Ingredient

import torch
import torch.nn as nn
import torch.nn.functional as F

unet_ingredient = Ingredient('UNet')


@unet_ingredient.config
def config():
    in_channels = 3
    n_classes = 2
    base_width = 16


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConv, self).__init__()

        layers = OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(True)),
        ])

        for name, module in layers.items():
            self.add_module(name, module)


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_width=16):
        super(UNet, self).__init__()

        self.base_width = base_width

        self.init_conv = DoubleConv(in_channels, self.base_width, stride=1)
        self.down_convs = nn.ModuleList([
            DoubleConv(self.base_width, self.base_width * 2, stride=2),
            DoubleConv(self.base_width * 2, self.base_width * 4, stride=2),
            DoubleConv(self.base_width * 4, self.base_width * 8, stride=2),
            DoubleConv(self.base_width * 8, self.base_width * 16, stride=2),
            DoubleConv(self.base_width * 16, self.base_width * 32, stride=2),
        ])

        self.upsample_convs = nn.ModuleList([
            DoubleConv(self.base_width * 32, self.base_width * 32),
            DoubleConv(self.base_width * 16, self.base_width * 16),
            DoubleConv(self.base_width * 8, self.base_width * 8),
            DoubleConv(self.base_width * 4, self.base_width * 4),
            DoubleConv(self.base_width * 2, self.base_width * 2),
        ])

        self.up_convs = nn.ModuleList([
            DoubleConv(self.base_width * 48, self.base_width * 16),
            DoubleConv(self.base_width * 24, self.base_width * 8),
            DoubleConv(self.base_width * 12, self.base_width * 4),
            DoubleConv(self.base_width * 6, self.base_width * 2),
            DoubleConv(self.base_width * 3, self.base_width),
        ])

        self.end_conv = nn.Conv2d(self.base_width, n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = [(x.shape[2:], self.init_conv(x))]
        for i, module in enumerate(self.down_convs):
            x = module(down[i][1])
            down.append((x.shape[2:], x))

        for i, (upsample, module) in enumerate(zip(self.upsample_convs, self.up_convs)):
            upsampled = F.interpolate(upsample(x), size=down[-(i + 2)][0])
            x = module(torch.cat((upsampled, down[-(i + 2)][1]), 1))

        return self.end_conv(x)


@unet_ingredient.capture
def load_unet(in_channels, n_classes, base_width):
    return UNet(in_channels=in_channels, n_classes=n_classes, base_width=base_width)


if __name__ == '__main__':
    unet = UNet()
    print(unet)

    x = torch.rand(2, 3, 733, 427)
    with torch.no_grad():
        print('Input:', x.shape, '-> UNet(input):', unet(x).shape)
