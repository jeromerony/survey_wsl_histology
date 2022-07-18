import sys
from os.path import dirname, abspath


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics import base


__all__ = [
    'IoUBackG',
    'FscoreBackG',
    'AccuracyBackG',
    'RecallBackG',
    'PrecisionBackG'
]


class IoUBackG(base.IoU):
    __name__ = 'iou_score_back_g'

    def forward(self, y_pr, y_gt):
        return super(IoUBackG, self).forward(1. - y_pr, 1. - y_gt)


class FscoreBackG(base.Fscore):

    def forward(self, y_pr, y_gt):
        return super(FscoreBackG, self).forward(1. - y_pr, 1. - y_gt)


class AccuracyBackG(base.Accuracy):

    def forward(self, y_pr, y_gt):
        return super(AccuracyBackG, self).forward(1. - y_pr, 1. - y_gt)


class RecallBackG(base.Recall):

    def forward(self, y_pr, y_gt):
        return super(RecallBackG, self).forward(1. - y_pr, 1. - y_gt)


class PrecisionBackG(base.Precision):

    def forward(self, y_pr, y_gt):
        return super(PrecisionBackG, self).forward(1. - y_pr, 1. - y_gt)
