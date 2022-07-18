import sys
from os.path import dirname, abspath

import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics import base
from dlib.losses.base import Metric

from dlib.utils.reproducibility import set_seed


class MeanMetric(Metric):
    def __init__(self, ignore_c=None, eps=1e-7, **kwargs):
        super(MeanMetric, self).__init__(**kwargs)

        self.op = None
        self.eps = eps
        self.ignore_c = ignore_c

    def forward(self, y_pre, y_gt):

        with torch.no_grad():
            uniqc = torch.unique(y_gt)
            score = 0.0
            t = 0.
            for c in uniqc:
                c = c.item()
                if c == self.ignore_c:
                    continue
                score += self.op((y_pre == c) * 1., (y_gt == c) * 1.)
                t += 1.

            return score / t


class MeanIoU(MeanMetric):
    __name__ = 'mean_iou_score'

    def __init__(self, **kwargs):
        super(MeanIoU, self).__init__(**kwargs)

        self.op = base.IoU(eps=self.eps, threshold=None, activation=None,
                           ignore_channels=None)


class MeanFscore(MeanMetric):

    def __init__(self, **kwargs):
        super(MeanFscore, self).__init__(**kwargs)

        self.op = base.Fscore(beta=1., eps=self.eps, threshold=None,
                              activation=None,  ignore_channels=None)


class MeanAccuracy(MeanMetric):

    def __init__(self, **kwargs):
        super(MeanAccuracy, self).__init__(**kwargs)

        self.op = base.Accuracy(threshold=None, activation=None,
                                ignore_channels=None)


class MeanRecall(MeanMetric):

    def __init__(self, **kwargs):
        super(MeanRecall, self).__init__(**kwargs)

        self.op = base.Recall(eps=self.eps, threshold=None, activation=None,
                              ignore_channels=None)


class MeanPrecision(MeanMetric):

    def __init__(self, **kwargs):
        super(MeanPrecision, self).__init__(**kwargs)

        self.op = base.Precision(eps=self.eps, threshold=None, activation=None,
                                 ignore_channels=None)


if __name__ == "__main__":
    set_seed(0)

    _y_pre = torch.randint(low=0, high=5, size=(30, 50), dtype=torch.float32)
    _y_gt = torch.randint(low=0, high=5, size=(30, 50), dtype=torch.float32)

    mdl = [MeanIoU, MeanFscore, MeanAccuracy, MeanRecall, MeanPrecision]
    _ignore_c = None
    for m in mdl:
        inst = m(ignore_c=_ignore_c, eps=1e-7)
        print(inst.__name__, inst.eps, inst.ignore_c, 'score:',
              inst(_y_pre, _y_gt))


