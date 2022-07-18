import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['GAP', 'WGAP', 'MaxPool', 'LogSumExpPool']


class _BasicPooler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 classes: int,
                 support_background: bool = False,
                 r: float = 10.,
                 modalities: int = 5,
                 kmax: float = 0.5,
                 kmin: float = None,
                 alpha: float = 0.6,
                 dropout: float = 0.0,
                 mid_channels: int = 128,
                 gated: bool = False
                 ):
        super(_BasicPooler, self).__init__()

        self.cams = None
        self.in_channels = in_channels
        self.classes = classes
        self.support_background = support_background

        # logsumexp
        self.r = r
        # wildcat
        self.modalities = modalities
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha
        self.dropout = dropout

        # mil
        self.mid_channels = mid_channels
        self.gated = gated

        self.name = 'null-name'

    @property
    def builtin_cam(self):
        return True

    def assert_x(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

    def correct_cl_logits(self, logits):
        if self.support_background:
            return logits[:, 1:]
        else:
            return logits

    def get_nbr_params(self):
        return sum([p.numel() for p in self.parameters()])

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={})'.format(
            self.__class__.__name__, self.in_channels, self.classes,
            self.support_background)


class GAP(_BasicPooler):
    """ https://arxiv.org/pdf/1312.4400.pdf """
    def __init__(self, **kwargs):
        super(GAP, self).__init__(**kwargs)
        self.name = 'GAP'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits


class WGAP(_BasicPooler):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, **kwargs):
        super(WGAP, self).__init__(**kwargs)
        self.name = 'WGAP'

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, self.classes)

    @property
    def builtin_cam(self):
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        return logits


class MaxPool(_BasicPooler):
    def __init__(self, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.name = 'MaxPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)

        logits = self.correct_cl_logits(logits)
        return logits


class LogSumExpPool(_BasicPooler):
    """ https://arxiv.org/pdf/1411.6228.pdf """
    def __init__(self, **kwargs):
        super(LogSumExpPool, self).__init__(**kwargs)
        self.name = 'LogSumExpPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        out = self.avgpool((self.r * out).exp()).log() * (1/self.r)

        logits = out.flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={}, ' \
               'r={})'.format(self.__class__.__name__, self.in_channels,
                              self.classes, self.support_background, self.r)


if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    b, c, h, w = 3, 1024, 8, 8
    classes = 5
    x = torch.rand(b, c, h, w).to(DEVICE)

    for support_background in [True, False]:
        for cl in [GAP, WGAP, MaxPool, LogSumExpPool]:
            instance = cl(in_channels=c, classes=classes,
                          support_background=support_background)
            instance.to(DEVICE)
            announce_msg('TEsting {}'.format(instance))
            out = instance(x)
            if instance.builtin_cam:
                print('x: {}, cam: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, instance.cams.shape, out.shape, out))
            else:
                print('x: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, out.shape, out))
