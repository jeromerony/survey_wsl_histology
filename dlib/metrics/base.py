import sys
from os.path import dirname, abspath

import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses import base
from dlib.functional import core as F
from dlib.base.modules import Activation


__all__ = [
    'IoU',
    'Fscore',
    'Accuracy',
    'Recall',
    'Precision'
]


class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None,
                 ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        with torch.no_grad():
            y_pr = self.activation(y_pr)
            return F.iou(
                y_pr, y_gt,
                eps=self.eps,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None,
                 ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        with torch.no_grad():
            y_pr = self.activation(y_pr)
            return F.f_score(
                y_pr, y_gt,
                eps=self.eps,
                beta=self.beta,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        with torch.no_grad():
            y_pr = self.activation(y_pr)
            return F.accuracy(
                y_pr, y_gt,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None,
                 ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        with torch.no_grad():
            y_pr = self.activation(y_pr)
            return F.recall(
                y_pr, y_gt,
                eps=self.eps,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None,
                 ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        with torch.no_grad():
            y_pr = self.activation(y_pr)
            return F.precision(
                y_pr, y_gt,
                eps=self.eps,
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )


if __name__ == "__main__":
    for name in __all__:
        module = sys.modules["__main__"].__dict__[name]
        print(module.__name__, module().__name__)
