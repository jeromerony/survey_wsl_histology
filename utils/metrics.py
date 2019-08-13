import numpy as np
import torch

from sklearn import metrics


def metric_report(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict:
    results = metrics.classification_report(labels, predictions, output_dict=True)
    results['confusion_matrix'] = metrics.confusion_matrix(labels, predictions)
    results['accuracy'] = metrics.accuracy_score(labels, predictions)

    # mean Average Precision
    if len(probabilities.shape) == 1:
        results['AP'] = metrics.average_precision_score(labels, probabilities)
    else:
        one_hot_labels = np.zeros(probabilities.shape)
        one_hot_labels[range(len(labels)), labels] = 1
        results['AP'] = metrics.average_precision_score(one_hot_labels, probabilities, average=None)

    return results


def nanmean(x: torch.Tensor):
    mask = ~torch.isnan(x)
    return x[mask].mean()


@torch.no_grad()
class Evaluator(object):
    def __init__(self, num_class: int):
        self.num_class = num_class
        self.cm = None

    def pixel_accuracy(self) -> float:
        acc = self.cm.trace() / self.cm.sum()
        return acc.item()

    def pixel_accuracy_class(self) -> float:
        acc = self.cm.diag() / self.cm.sum(1)
        return nanmean(acc).item()

    def intersection_over_union(self) -> torch.Tensor:
        iou = (self.cm.diag() + 1e-8) / (self.cm.sum(1) + self.cm.sum(0) - self.cm.diag() + 1e-8)
        return iou

    def mean_intersection_over_union(self) -> float:
        iou = self.intersection_over_union()
        miou = nanmean(iou)
        return miou.item()

    def dice(self) -> torch.Tensor:
        dice = (2 * self.cm.diag() + 1e-8) / (self.cm.sum(1) + self.cm.sum(0) + 1e-8)
        return dice

    def mean_dice(self) -> float:
        dice = self.dice()
        mdice = nanmean(dice)
        return mdice.item()

    def frequency_weighted_intersection_over_union(self) -> float:
        freq = self.cm.sum(1) / self.cm.sum()
        iu = self.cm.diag() / (self.cm.sum(1) + self.cm.sum(0) - self.cm.diag())
        fiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fiou.item()

    def _generate_matrix(self, gt_image: torch.Tensor, pre_image: torch.Tensor) -> float:
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].long() + pre_image[mask].long()
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape((self.num_class, self.num_class)).float()
        return confusion_matrix

    def add_batch(self, gt_image: torch.Tensor, pre_image: torch.Tensor) -> None:
        assert gt_image.shape == pre_image.shape
        if self.cm is None:
            self.cm = self._generate_matrix(gt_image, pre_image)
        else:
            self.cm += self._generate_matrix(gt_image, pre_image)

    def reset(self) -> None:
        self.cm = None
