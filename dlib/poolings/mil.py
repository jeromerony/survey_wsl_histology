import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))

sys.path.append(root_dir)

from dlib.configure import constants
from dlib.poolings.core import _BasicPooler


class _Attention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 classes: int,
                 gated: bool):
        super(_Attention, self).__init__()

        for v in [in_channels, mid_channels, classes]:
            assert isinstance(v, int)
            assert v > 0

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = classes
        self.gated = gated

        self.non_lin_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1),
            nn.Tanh()
        )

        if self.gated:
            self.non_lin_2 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1),
                nn.Sigmoid()
            )

        self.to_score = nn.Conv2d(self.mid_channels, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.non_lin_1(x)
        if self.gated:
            out = out * self.non_lin_2(x)
        pixel_logits = self.to_score(out)

        return pixel_logits

    def __str__(self):
        return "{}(): MIL attention. gated={}. in_c={}, mid_c={}, " \
               "classes={}".format(self.__class__.__name__, self.gated,
                                   self.in_channels, self.mid_channels,
                                   self.num_classes)


class DeepMil(_BasicPooler):
    def __init__(self, **kwargs):
        super(DeepMil, self).__init__(**kwargs)

        self.name = 'DeepMIL'
        assert not self.support_background

        self.attention = _Attention(in_channels=kwargs['in_channels'],
                                    mid_channels=kwargs['mid_channels'],
                                    classes=kwargs['classes'],
                                    gated=kwargs['gated'])
        self.softmax = nn.Softmax(dim=2)
        self.classification = nn.ModuleList(
            [nn.Linear(self.in_channels, 1) for _ in range(self.classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        attention_scores = self.attention(x)  # N x C x H x W
        shape = attention_scores.shape
        attention = self.softmax(attention_scores.flatten(2))  # N x C x (H x W)
        self.cams = attention.detach().view(shape)

        out = torch.einsum('nkp,ncp->nkc', attention, x.flatten(2))
        scores = []
        for i, m in enumerate(self.classification):
            scores.append(m(out[:, i]))
        logits = torch.cat(scores, 1)

        return logits

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, ' \
               f'mid_channels={self.mid_channels}, classes={self.classes}, ' \
               f'gated={self.gated})'


def test_Attention():
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "1"
    device = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    in_channels = 512
    mid_channels = 128
    num_classes = 1
    bsz = 32
    h, w = 28, 28
    x = torch.rand((bsz, in_channels, h, w), device=device)

    for gated in [False, True]:
        model = _Attention(in_channels=in_channels, mid_channels=mid_channels,
                           classes=num_classes, gated=gated).to(device)
        announce_msg(f'testing {model}')
        out = model(x)
        print(f'x: {x.shape}, out: {out.shape}')


def test_DeepMil():
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "1"
    device = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    in_channels = 512
    mid_channels = 128
    num_classes = 4
    bsz = 32
    h, w = 28, 28
    x = torch.rand((bsz, in_channels, h, w), device=device)

    for gated in [False, True]:
        model = DeepMil(in_channels=in_channels, mid_channels=mid_channels,
                        classes=num_classes, gated=gated).to(device)
        announce_msg(f'testing {model}')
        logits = model(x)
        print(f'x: {x.shape}, logits: {logits.shape}, cams: {model.cams.shape}')


if __name__ == '__main__':
    # test_Attention()
    test_DeepMil()
