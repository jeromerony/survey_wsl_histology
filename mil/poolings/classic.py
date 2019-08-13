import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, in_channels, classes):
        super(Classification, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels=classes, kernel_size=1)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)


class Average(Classification):
    def __init__(self, in_channels, classes):
        super(Average, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        return self.pool(out).flatten(1)


class Max(Classification):
    def __init__(self, in_channels, classes):
        super(Max, self).__init__(in_channels, classes)

        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        return self.pool(out).flatten(1)


class LogSumExp(Classification):
    def __init__(self, in_channels, classes, r=10):
        super(LogSumExp, self).__init__(in_channels, classes)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        self.cam = out.detach()
        m = self.maxpool(out)
        out = self.avgpool((self.r * (out - m)).exp()).log().mul(1 / self.r) + m

        return out.flatten(1)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4)

    modules = [
        Average(in_channels=3, classes=2),
        Max(in_channels=3, classes=2),
        LogSumExp(in_channels=3, classes=2),
    ]

    for m in modules:
        print(m, '\n', x.shape, ' -> ', m(x).shape, '\n', sep='')
