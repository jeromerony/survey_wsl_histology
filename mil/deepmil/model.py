from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, mid_channels=128, num_classes=1, gated=False):
        super(Attention, self).__init__()
        self.gated = gated
        self.in_c = in_channels
        self.mid_c = mid_channels

        self.non_lin_1 = nn.Sequential(OrderedDict([
            ('fc', nn.Conv2d(self.in_c, self.mid_c, kernel_size=1)),
            ('tanh', nn.Tanh()),
        ]))

        if self.gated:
            self.non_lin_2 = nn.Sequential(OrderedDict([
                ('fc', nn.Conv2d(self.in_c, self.mid_c, kernel_size=1)),
                ('sigmoid', nn.Sigmoid()),
            ]))

        self.to_score = nn.Conv2d(self.mid_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.non_lin_1(x)
        if self.gated:
            out = out * self.non_lin_2(x)
        out = self.to_score(out)

        return out


class DeepMIL(nn.Module):
    def __init__(self, in_channels, mid_channels=128, gated=False):
        super(DeepMIL, self).__init__()

        self.attention = Attention(in_channels, mid_channels=mid_channels, gated=gated)
        self.softmax = nn.Softmax(dim=1)
        self.classification = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(x)  # N x 1 x H x W
        shape = attention_scores.shape
        attention = self.softmax(attention_scores.flatten(1))  # N x (H x W)
        self.cam = attention.detach().view(shape)

        out = torch.einsum('np,ncp->nc', attention, x.flatten(2))
        out = self.classification(out)

        return out.squeeze(1)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return (logits >= 0).long()

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class DeepMILMulti(DeepMIL):
    def __init__(self, in_channels, mid_channels=128, num_classes=2, gated=False):
        super(DeepMILMulti, self).__init__(in_channels)

        self.attention = Attention(in_channels, mid_channels=mid_channels, num_classes=num_classes, gated=gated)
        self.softmax = nn.Softmax(dim=2)
        self.classification = nn.ModuleList([nn.Linear(in_channels, 1) for _ in range(num_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(x)  # N x C x H x W
        shape = attention_scores.shape
        attention = self.softmax(attention_scores.flatten(2))  # N x C x (H x W)
        self.cam = attention.detach().view(shape)

        out = torch.einsum('nkp,ncp->nkc', attention, x.flatten(2))
        scores = []
        for i, m in enumerate(self.classification):
            scores.append(m(out[:, i]))
        out = torch.cat(scores, 1)

        return out

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 6)

    deepmil = DeepMIL(3)

    print(x.shape, '->', deepmil(x).shape)
