import torch
import torch.nn as nn
import torch.nn.functional as F


# original pooling from https://github.com/durandtibo/wildcat.pytorch: complicated and thus not efficient
# New pooling is around twice as fast on GPU
class ClassWisePooling(nn.Module):
    def __init__(self, in_channels, classes, modalities=4):
        super(ClassWisePooling, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=classes * modalities, kernel_size=1)
        self.C = classes
        self.M = modalities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape

        out = self.conv(x) # N x (C x M) x H x W

        return out.view(N, self.C, self.M, H, W).mean(dim=2)

    def __repr__(self):
        return self.__class__.__name__ + '(in_channels={}, classes={}, modalities={})'.format(self.conv.in_channels,
                                                                                              self.C, self.M)


# Simpler implementation of the WildCatPool from https://github.com/durandtibo/wildcat.pytorch:
# Added a custom dropout not setting the activations to 0 but rather removing them
class WildCatPooling(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=0.6):
        super(WildCatPooling, self).__init__()
        assert isinstance(kmax, (int, float)) and kmax > 0, 'kmax must be an integer or a 0 < float < 1'
        assert kmin is None or (isinstance(kmin, (int, float)) and kmin >= 0), 'kmin must be None or same type as kmax'

        self.kmax = kmax
        self.kmin = kmax if kmin is None else kmin
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        activs = x.view(N, C, H * W)

        activs = torch.sort(activs, dim=-1, descending=True)[0]
        kmax = round(H * W * self.kmax) if self.kmax < 1 else int(self.kmax)
        zkmax = activs.narrow(-1, 0, kmax)
        scmax = zkmax.sum(-1) / kmax
        scmin = 0
        if self.kmin:
            kmin = round(H * W * self.kmin) if self.kmin < 1 else int(self.kmin)
            zkmin = activs.narrow(-1, activs.size(-1) - kmin, kmin)
            scmin = zkmin.sum(-1) / kmin

        return scmax + self.alpha * scmin

    def __repr__(self):
        return self.__class__.__name__ + '(kmax={}, kmin={}, alpha={})'.format(self.kmax, self.kmin, self.alpha)


class Wildcat(nn.Module):
    def __init__(self, in_channels, classes, modalities=4, kmax=0.1, kmin=None, alpha=0.6):
        super(Wildcat, self).__init__()

        self.classification = ClassWisePooling(in_channels, classes, modalities=modalities)
        self.pooling = WildCatPooling(kmax=kmax, kmin=kmin, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classification(x)
        self.cam = out.detach()
        out = self.pooling(out)
        return out

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4)

    modules = [
        Wildcat(in_channels=3, classes=2, kmax=0.1)
    ]

    for m in modules:
        print(m, '\n', x.shape, ' -> ', m(x).shape, '\n', sep='')