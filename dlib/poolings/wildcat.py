import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))

sys.path.append(root_dir)

from dlib.configure import constants
from dlib.poolings.core import _BasicPooler


__all__ = ['WildCatCLHead']


class _WildCatPoolDecision(nn.Module):
    def __init__(self, kmax=0.5, kmin=None, alpha=1., dropout=0.0):
        """
        Input:
            kmax: int or float scalar. The number of maximum features to
            consider.
            kmin: int or float scalar. If None, it takes the same value as
            kmax. The number of minimal features to consider.
            alpha: float scalar. A weight , used to compute the final score.
            dropout: float scalar. If not zero, a dropout is performed over the
            min and max selected features.
        """
        super(_WildCatPoolDecision, self).__init__()

        assert isinstance(kmax, (int, float))
        assert kmax > 0.
        assert kmin is None or isinstance(kmin, (int, float))
        if isinstance(kmin, (int, float)):
            assert kmin >= 0.

        self.kmax = kmax
        self.kmin = kmax if kmin is None else kmin
        self.alpha = alpha
        self.dropout = dropout

        self.dropout_md = nn.Dropout(p=dropout, inplace=False)

    def get_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k == 1 and isinstance(k, float):
            return int(n)
        elif k == 1 and isinstance(k, int):
            return 1
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, x):
        """
        Input:
            In the case of K classes:
                x: torch tensor of size (n, c, h, w), where n is the batch
                size, c is the number of classes,
                h is the height of the feature map, w is its width.
            seed: int, seed for the thread to guarantee reproducibility over a
            fixed number of gpus.
        Output:
            scores: torch vector of size (k). Contains the wildcat score of
            each class. A score is a linear combination
            of different features. The class with the highest features is the
            winner.
        """
        b, c, h, w = x.shape
        activations = x.view(b, c, h * w)

        n = h * w

        sorted_features = torch.sort(activations, dim=-1, descending=True)[0]
        kmax = self.get_k(self.kmax, n)
        kmin = self.get_k(self.kmin, n)

        # assert kmin != 0, "kmin=0"
        assert kmax != 0, "kmax=0"

        # dropout
        if self.dropout != 0.:
            sorted_features = self.dropout_md(sorted_features)

        scores = sorted_features.narrow(-1, 0, kmax).sum(-1).div_(kmax)

        if kmin > 0 and self.alpha != 0.:
            scores.add(
                sorted_features.narrow(
                    -1, n - kmin, kmin).sum(-1).mul_(
                    self.alpha / kmin)).div_(2.)

        return scores

    def __str__(self):
        return self.__class__.__name__ + "(kmax={}, kmin={}, alpha={}, " \
                                         "dropout={}".format(
            self.kmax, self.kmin, self.alpha,
            self.dropout)

    def __repr__(self):
        return super(_WildCatPoolDecision, self).__repr__()


class _ClassWisePooling(nn.Module):
    def __init__(self, classes, modalities):
        """
        Init. function.
        :param classes: int, number of classes.
        :param modalities: int, number of modalities per class.
        """
        super(_ClassWisePooling, self).__init__()

        self.C = classes
        self.M = modalities

    def forward(self, inputs):
        N, C, H, W = inputs.size()
        msg = 'Wrong number of channels, expected {} ' \
              'channels but got {}'.format(self.C * self.M, C)
        assert C == self.C * self.M, msg
        return torch.mean(
            inputs.view(N, self.C, self.M, -1), dim=2).view(N, self.C, H, W)

    def __str__(self):
        return self.__class__.__name__ + \
               '(classes={}, modalities={})'.format(self.C, self.M)

    def __repr__(self):
        return super(_ClassWisePooling, self).__repr__()


class WildCatCLHead(_BasicPooler):
    """ https://openaccess.thecvf.com/content_cvpr_2017/papers
    /Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.pdf """
    def __init__(self, **kwargs):
        super(WildCatCLHead, self).__init__(**kwargs)
        self.name = 'WILDCAT'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.to_modalities = nn.Conv2d(
            self.in_channels, classes * self.modalities, kernel_size=1,
            bias=True)
        self.to_maps = _ClassWisePooling(classes, self.modalities)
        self.wildcat = _WildCatPoolDecision(
            kmax=self.kmax, kmin=self.kmin, alpha=self.alpha,
            dropout=self.dropout)

        self.cams_attached = None

    def forward(self, x):
        self.assert_x(x)

        modalities = self.to_modalities(x)
        out = self.to_maps(modalities)
        self.cams_attached = out
        self.cams = out.detach()
        logits = self.wildcat(x=out)
        logits = self.correct_cl_logits(logits)

        return logits

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={},' \
               'modalities={}, kmax={}, kmin={}, alpha={}, ' \
               'dropout={})'.format(
                self.__class__.__name__, self.in_channels, self.classes,
                self.support_background, self.modalities, self.kmax, self.kmin,
                self.alpha, self.dropout
                )


if __name__ == "__main__":
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
        for cl in [WildCatCLHead]:
            instance = cl(in_channels=c, classes=classes,
                          support_background=support_background,
                          modalities=5, kmin=0.1, kmax=0.6, dropout=0.1)
            instance.to(DEVICE)
            announce_msg('TEsting {}'.format(instance))
            out = instance(x)
            print('x: {}, cam: {}, logitcl shape: {}, logits: {}'.format(
                x.shape, instance.cams.shape, out.shape, out))

