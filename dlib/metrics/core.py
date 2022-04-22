import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.base import Metric
from dlib.metrics import average
from dlib.metrics import base
from dlib.metrics import background


from dlib.configure import constants


__all__ = [
    'ClMetrics',
    'SegMetric'
]


class _FullMetric(Metric):
    def __init__(self, device, **kwargs):
        super(_FullMetric, self).__init__(**kwargs)

        self.metrics = []
        self.device = device
        self.counter = 0

    @property
    def max_counter(self):
        return len(self.metrics)

    def _to_device(self):
        for metric in self.metrics:
            metric.to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.max_counter:
            raise StopIteration
        metric = self.metrics[self.counter]
        self.counter += 1
        return metric


class ClMetrics(_FullMetric):
    def __init__(self, **kwargs):
        super(ClMetrics, self).__init__(**kwargs)
        self.metrics = [average.MeanAccuracy(name="mean_accu_out")]


class SegMeanMetrics(_FullMetric):
    def __init__(self, ignore_c=None, **kwargs):
        super(SegMeanMetrics, self).__init__(**kwargs)

        self.metrics = [average.MeanIoU(ignore_c=ignore_c),
                        average.MeanFscore(ignore_c=ignore_c),
                        average.MeanPrecision(ignore_c=ignore_c),
                        average.MeanRecall(ignore_c=ignore_c),
                        average.MeanAccuracy(ignore_c=ignore_c)
                        ]


class SegBinMetrics(_FullMetric):
    def __init__(self, **kwargs):
        super(SegBinMetrics, self).__init__(**kwargs)

        self.metrics = [
            base.IoU(eps=1e-7, threshold=None,
                     activation=None, ignore_channels=None),
            base.Fscore(beta=1., eps=1e-7, threshold=None,
                        activation=None, ignore_channels=None),
            base.Precision(eps=1e-7, threshold=None,
                           activation=None, ignore_channels=None),
            base.Recall(eps=1e-7, threshold=None,
                        activation=None, ignore_channels=None),
            base.Accuracy(threshold=None, activation=None,
                          ignore_channels=None),
            # background.
            background.IoUBackG(eps=1e-7, threshold=None,
                                activation=None, ignore_channels=None),
            background.FscoreBackG(beta=1., eps=1e-7, threshold=None,
                                   activation=None, ignore_channels=None),
            background.PrecisionBackG(eps=1e-7, threshold=None,
                                      activation=None, ignore_channels=None),
            background.RecallBackG(eps=1e-7, threshold=None,
                                   activation=None, ignore_channels=None),
            background.AccuracyBackG(threshold=None, activation=None,
                                     ignore_channels=None)
                        ]


class SegMetric(_FullMetric):
    def __init__(self, seg_mode=constants.MULTICLASS_MODE, ignore_c=None,
                 **kwargs):
        super(SegMetric, self).__init__(**kwargs)

        self.seg_mode = seg_mode

        self.seg_mean_mtr = SegMeanMetrics(ignore_c=ignore_c, **kwargs)
        self.seg_bin_mtr = SegBinMetrics(**kwargs)

        self.metrics = []
        if self.seg_mode == constants.BINARY_MODE:
            self.metrics = self.seg_bin_mtr.metrics + self.seg_mean_mtr.metrics
        elif self.seg_mode == constants.MULTICLASS_MODE:
            self.metrics = self.seg_mean_mtr.metrics
        else:
            raise NotImplementedError
