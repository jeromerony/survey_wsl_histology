import os
import time
from copy import deepcopy
import sys
from os.path import dirname, abspath, join
import threading
from copy import deepcopy
from typing import Optional, Union, Tuple

import cv2
import numpy as np

import torch.utils.data as torchdata
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.datasets.wsol_data_core import get_image_ids
from dlib.datasets.wsol_data_core import get_bounding_boxes
from dlib.datasets.wsol_data_core import get_image_sizes
from dlib.datasets.wsol_data_core import get_mask_paths
from dlib.datasets.wsol_data_core import get_mask
from dlib.datasets.wsol_data_core import RESIZE_LENGTH

from dlib.utils.tools import check_scoremap_validity
from dlib.utils.tools import check_box_convention

from dlib.configure import constants
from dlib.parallel import sync_tensor_across_gpus


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


__all__ = ['BoxEvaluator', 'MaskEvaluator', 'calculate_multiple_iou',
           'compute_bboxes_from_scoremaps']


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)

        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self,
                 metadata,
                 dataset_name,
                 split,
                 cam_threshold_list,
                 iou_threshold_list,
                 mask_root,
                 multi_contour_eval,
                 args,
                 best_valid_tau: float = None
                 ):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split  # todo: delete. not used.
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

        self.best_tau_list = []
        self.curve_s = None

        self.foreground = None
        self.background = None

        self.top1 = None
        self.top5 = None
        self.curve_top_1_5 = None

        self.perf_gist = None

        self.args = args
        self.best_valid_tau = best_valid_tau

    def accumulate(self, scoremap, image_id, target, preds_ordered):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def _synch_across_gpus(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

        self.num_correct_top1 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.num_correct_top5 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id, target, preds_ordered):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
            target: int. longint. the true class label. for bbox evaluation (
            top1/5).
            preds_ordered: numpy.ndarray. vector of predicted labels ordered
            from from the most probable the least probable. for evaluation of
            bbox using top1/5.
        """

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))  # (nbr_boxes_in_img, 1)

        idx = 0
        sliced_multiple_iou = []  # length == number tau thresh
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            # pick the maximum score iou among all the boxes.
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]

            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1

            if target == preds_ordered[0]:
                self.num_correct_top1[_THRESHOLD][
                    correct_threshold_indices] += 1

            if target in preds_ordered[:5]:
                self.num_correct_top5[_THRESHOLD][
                    correct_threshold_indices] += 1

        self.cnt += 1

    def _synch_across_gpus(self):

        for tracker in [self.num_correct, self.num_correct_top1,
                        self.num_correct_top5]:
            for k in tracker.keys():
                _k_val = torch.from_numpy(tracker[k]).cuda(
                    self.args.c_cudaid).view(1, -1)

                tracker[k] = sync_tensor_across_gpus(
                    _k_val).sum(dim=0).cpu().view(-1).numpy()

        cnt = torch.tensor([self.cnt], dtype=torch.float,
                           requires_grad=False, device=torch.device(
                self.args.c_cudaid))

        cnt = sync_tensor_across_gpus(cnt)
        self.cnt = cnt.sum().cpu().item()

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        self.best_tau_list = []
        self.curve_s = {
            'x': self.cam_threshold_list
        }

        self.top1 = []
        self.top5 = []
        self.curve_top_1_5 = {
            'x': self.cam_threshold_list,
            'top1': dict(),
            'top5': dict()
        }

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

            self.curve_s[_THRESHOLD] = localization_accuracies

            self.best_tau_list.append(
                self.cam_threshold_list[np.argmax(localization_accuracies)])

            loc_acc = self.num_correct_top1[_THRESHOLD] * 100. / float(self.cnt)
            self.top1.append(loc_acc.max())

            self.curve_top_1_5['top1'][_THRESHOLD] = deepcopy(loc_acc)

            loc_acc = self.num_correct_top5[_THRESHOLD] * 100. / float(self.cnt)
            self.top5.append(loc_acc.max())

            self.curve_top_1_5['top5'][_THRESHOLD] = deepcopy(loc_acc)

        return max_box_acc


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name not in [constants.OpenImages, constants.GLAS,
                                     constants.CAMELYON512]:
            raise ValueError(f"Cant evalaute masks on {self.dataset_name}.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])

        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, scoremap, image_id, target=None, preds_ordered=None):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
            target and preds_ordered are not used in this case.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def get_best_operating_point(self, miou, tau: float = None):
        if tau is None:
            idx = np.argmax(miou)
        else:
            idx = np.argmin(
                np.abs(np.array(self.threshold_list_right_edge) - tau))

        return self.threshold_list_right_edge[idx]

    def _synch_across_gpus(self):

        for tracker in [self.gt_true_score_hist,
                        self.gt_false_score_hist]:
            _k_val = torch.from_numpy(tracker).cuda(self.args.c_cudaid)
            tracker = sync_tensor_across_gpus(
                _k_val).sum(dim=0).cpu().numpy()

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """

        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        dice_fg = 2. * tp / (2. * tp + fp + fn)
        dice_bg = 2. * tn / (2. * tn + fp + fn)

        iou_fg = tp / (tp + fp + fn)
        iou_bg = tn / (tn + fp + fn)
        miou = 0.5 * (iou_fg + iou_bg)

        if self.best_valid_tau is None:
            self.best_tau_list = [self.get_best_operating_point(
                miou=miou, tau=None)]
        else:
            self.best_tau_list = [self.best_valid_tau]

        idx = np.argmin(np.abs(
            self.threshold_list_right_edge - self.best_tau_list[0]))

        total_fg = float(tp[idx] + fn[idx])
        total_bg = float(tn[idx] + fp[idx])

        self.perf_gist = {
            constants.MTR_PXAP: auc,
            constants.MTR_TP: 100 * tp[idx] / total_fg,
            constants.MTR_FN: 100 * fn[idx] / total_fg,
            constants.MTR_TN: 100 * tn[idx] / total_bg,
            constants.MTR_FP: 100 * fp[idx] / total_bg,
            constants.MTR_DICEFG: 100 * dice_fg[idx],
            constants.MTR_DICEBG: 100 * dice_bg[idx],
            constants.MTR_MIOU: 100 * miou[idx],
            constants.MTR_BESTTAU: self.best_tau_list
        }

        self.curve_s = {
            'x': recall,
            'y': precision,
            constants.MTR_MIOU: 100. * miou,
            constants.MTR_TP: 100. * tp / total_fg,
            constants.MTR_TN: 100. * tn / total_bg,
            constants.MTR_FP: 100. * fp / total_bg,
            constants.MTR_FN: 100. * fn / total_fg,
            constants.MTR_DICEFG: dice_fg,
            constants.MTR_DICEBG: dice_bg,
            constants.MTR_BESTTAU: self.best_tau_list,
            'idx': idx,
            'num_bins': self.num_bins,
            'threshold_list_right_edge': self.threshold_list_right_edge,
            'cam_threshold_list': self.cam_threshold_list,
            'perf_gist': self.perf_gist
        }

        return auc
