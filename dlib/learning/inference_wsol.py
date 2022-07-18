import copy
import random
import time
from pathlib import Path
import subprocess
from os.path import normpath

import kornia.morphology
import numpy as np
import os
import sys
from os.path import dirname, abspath, join
import datetime as dt
import pickle as pkl
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.cuda.amp import autocast
import torch.distributed as dist

from tqdm import tqdm as tqdm

from skimage.filters import threshold_otsu
from skimage import filters

from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics.wsol_metrics import BoxEvaluator
from dlib.metrics.wsol_metrics import MaskEvaluator
from dlib.metrics.wsol_metrics import compute_bboxes_from_scoremaps
from dlib.metrics.wsol_metrics import calculate_multiple_iou
from dlib.datasets.wsol_data_core import get_mask
from dlib.datasets.wsol_data_core import load_mask_image
from dlib.datasets.wsol_data_core import RESIZE_LENGTH


from dlib.datasets.wsol_loader import configure_metadata
from dlib.visualization.vision_wsol import Viz_WSOL

from dlib.utils.tools import t2n
from dlib.utils.tools import check_scoremap_validity
from dlib.configure import constants
from dlib.cams import build_std_cam_extractor
from dlib.cams import build_fcam_extractor
from dlib.cams import build_negev_extractor
from dlib.cams import build_seg_extractor
from dlib.utils.shared import reformat_id

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float).
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()

    return cam


def max_normalize(cam):
    max_val = cam.max()
    if max_val == 0.:
        return cam

    return cam / max_val


def entropy_cam(cam: torch.Tensor) -> torch.Tensor:
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == 2

    ops = 1. - cam
    entrop = - cam * torch.log2(cam) - ops * torch.log2(ops)
    assert ((entrop > 1.) + (entrop < 0.)).sum() == 0.

    return entrop


class CAMComputer(object):
    def __init__(self,
                 args,
                 model,
                 loader: DataLoader,
                 metadata_root,
                 mask_root,
                 iou_threshold_list,
                 dataset_name,
                 split,
                 multi_contour_eval,
                 cam_curve_interval: float = .001,
                 out_folder=None,
                 fcam_argmax: bool = False,
                 best_valid_tau: float = None
                 ):
        self.args = args
        self.model = model
        self.model.eval()
        self.loader = loader
        self.dataset_name = dataset_name
        self.split = split
        self.out_folder = out_folder
        self.fcam_argmax = fcam_argmax

        if args.task in [constants.F_CL, constants.NEGEV]:
            self.req_grad = False
        elif args.task == constants.STD_CL:
            self.req_grad = constants.METHOD_REQU_GRAD[args.method]
        elif args.task == constants.SEG:
            self.req_grad = False
        else:
            raise NotImplementedError

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {constants.OpenImages: MaskEvaluator,
                          constants.CUB: BoxEvaluator,
                          constants.ILSVRC: BoxEvaluator,
                          constants.GLAS: MaskEvaluator,
                          constants.CAMELYON512: MaskEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval,
                                          args=args,
                                          best_valid_tau=best_valid_tau
                                          )

        self.bbox = args.dataset in [constants.CUB, constants.ILSVRC]

        self.viz = Viz_WSOL()
        self.default_seed = int(os.environ["MYSEED"])

        self.std_cam_extractor = None
        self.fcam_extractor = None
        self.seg_extractor = None

        self.special1 = self.args.method in [constants.METHOD_SPG,
                                             constants.METHOD_ACOL,
                                             constants.METHOD_ADL]

        if args.task == constants.STD_CL:
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=self.model, args=self.args)
        elif args.task == constants.F_CL:
            self.fcam_extractor = self._build_fcam_extractor(
                model=self.model, args=self.args)
            # useful for drawing side-by-side.
            # todo: build classifier from scratch and create its cam extractor.
        elif args.task == constants.NEGEV:
            self.fcam_extractor = self._build_negev_cam_extractor(
                model=self.model, args=self.args)
        elif args.task == constants.SEG:
            self.seg_extractor = self._build_seg_extractor(model=self.model,
                                                           args=self.args)
        else:
            raise NotImplementedError

    def _build_std_cam_extractor(self, classifier, args):
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _build_fcam_extractor(self, model, args):
        return build_fcam_extractor(model=model, args=args)

    def _build_negev_cam_extractor(self, model,args):
        return build_negev_extractor(model=model, args=args)

    def _build_seg_extractor(self, model, args):
        return build_seg_extractor(model=model, args=args)

    def get_cam_one_sample(self, image: torch.Tensor, target: int,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_shape = image.shape[2:]

        with autocast(enabled=self.args.amp_eval):
            if self.special1:
                output = self.model(image, labels=target)
            else:
                output = self.model(image)

        if self.args.task == constants.STD_CL:

            if self.args.amp_eval:
                output = output.float()

            cl_logits = output
            cam = self.std_cam_extractor(class_idx=target,
                                         scores=cl_logits,
                                         normalized=True,
                                         reshape=img_shape if self.special1
                                         else None)

            # (h`, w`)

        elif self.args.task in [constants.F_CL, constants.NEGEV]:

            if self.args.amp_eval:
                tmp = []
                for term in output:
                    tmp.append(term.float() if term is not None else None)
                output = tmp

            cl_logits, fcams, im_recon = output
            cam = self.fcam_extractor(argmax=self.fcam_argmax)
            # (h`, w`)

        elif self.args.task == constants.SEG:
            cam = self.seg_extractor(argmax=self.fcam_argmax)
            cl_logits = None

        else:
            raise NotImplementedError

        if self.args.amp_eval:
            cam = cam.float()

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        # cl_logits: 1, nc.
        return cam, cl_logits

    def minibatch_accum(self, images, targets, image_ids, image_size) -> None:

        for image, target, image_id in zip(images, targets, image_ids):

            with torch.set_grad_enabled(self.req_grad):
                cam, cl_logits = self.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target.item())

            with torch.no_grad():
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)
                cam = cam.detach()
                # todo:
                # cam = torch.clamp(cam, min=0.0, max=1.)

                # cam: (h, w)
                cam = t2n(cam)
                _preds_ordered = None
                if cl_logits is not None:
                    assert cl_logits.ndim == 2
                    _, preds_ordered = torch.sort(
                        input=cl_logits.cpu().squeeze(0), descending=True,
                        stable=True)
                    _preds_ordered = preds_ordered.numpy()

                self.evaluator.accumulate(
                    cam, image_id, target.item(), _preds_ordered)

    def normalizecam(self, cam):
        if self.args.task == constants.STD_CL:
            cam_normalized = normalize_scoremap(cam)
        elif self.args.task in [constants.F_CL, constants.NEGEV]:
            cam_normalized = cam
        elif self.args.task == constants.SEG:
            cam_normalized = cam
        else:
            raise NotImplementedError
        return cam_normalized

    def fix_random(self):
        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = True

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for batch_idx, (images, targets, image_ids, _, _, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            self.minibatch_accum(images=images, targets=targets,
                                 image_ids=image_ids, image_size=image_size)

            # # cams shape (batchsize, h, w)..
            # for cam, image_id in zip(cams, image_ids):
            #     # cams shape (h, w).
            #     assert cam.shape == image_size
            #
            #     # cam_resized = cv2.resize(cam, image_size,
            #     #                          interpolation=cv2.INTER_CUBIC)
            #
            #     cam_resized = cam
            #     cam_normalized = self.normalizecam(cam_resized)
            #     self.evaluator.accumulate(cam_normalized, image_id)

        if self.args.distributed:
            self.evaluator._synch_across_gpus()
            dist.barrier()

        return self.evaluator.compute()

    def build_bbox(self, scoremap, image_id, tau: float):
        cam_threshold_list = [tau]

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=cam_threshold_list,
            multi_contour_eval=self.evaluator.multi_contour_eval)

        assert len(boxes_at_thresholds) == 1
        assert len(number_of_box_list) == 1

        # nbrbox, 4
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.evaluator.gt_bboxes[image_id]))  # (nbrbox, 1)

        multiple_iou = multiple_iou.flatten()
        idx = np.argmax(multiple_iou)
        bbox_iou = multiple_iou[idx]
        best_bbox = boxes_at_thresholds[idx]  # shape: (4,)

        return best_bbox, bbox_iou

    def build_mask(self):
        pass

    def assert_datatset_bbx(self):
        assert self.dataset_name in [constants.CUB, constants.ILSVRC]

    def assert_dataset_mask(self):
        assert self.dataset_name in [constants.OpenImages, constants.GLAS,
                                     constants.CAMELYON512]

    def assert_tau_list(self):
        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list

        if isinstance(self.evaluator, BoxEvaluator):
            assert len(best_tau_list) == len(iou_threshold_list)
        elif isinstance(self.evaluator, MaskEvaluator):
            assert len(best_tau_list) == 1
        else:
            raise NotImplementedError

    def create_folder(self, fd):
        os.makedirs(fd, exist_ok=True)

    def reformat_id(self, img_id):
        tmp = str(Path(img_id).with_suffix(''))
        return tmp.replace('/', '_')

    def _get_ids_with_zero_ignore_mask(self):
        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            ignore_file = os.path.join(self.evaluator.mask_root,
                                       self.evaluator.ignore_paths[id])
            ignore_box_mask = load_mask_image(ignore_file,
                                              (RESIZE_LENGTH, RESIZE_LENGTH))
            if ignore_box_mask.sum() == 0:
                out.append(id)

        return out

    def _get_ids_bin_datasets(self, nbr: int):
        ids = self.loader.dataset.image_ids

        zeros = [id for id in ids if self.loader.dataset.image_labels[id] == 0]
        ones = [id for id in ids if self.loader.dataset.image_labels[id] == 1]

        self.fix_random()
        for i in range(100):
            random.shuffle(zeros)
            random.shuffle(ones)

        z = int(nbr / 2.)
        self.fix_random()
        return ones[:z] + zeros[:(nbr - z)]

    def select_random_ids_to_draw(self, nbr: int) -> list:
        self.fix_random()
        if isinstance(self.evaluator, BoxEvaluator):
            ids = self.loader.dataset.image_ids
            total_s = len(ids)
            n = min(nbr, total_s)
            idx = np.random.choice(a=total_s, size=n, replace=False).flatten()

        elif isinstance(self.evaluator, MaskEvaluator):
            if self.args.dataset == constants.OpenImages:
                ids = self._get_ids_with_zero_ignore_mask()
            elif self.args.dataset in [constants.GLAS, constants.CAMELYON512]:
                ids = self._get_ids_bin_datasets(nbr)
            else:
                raise NotImplementedError

            total_s = len(ids)
            n = min(nbr, total_s)
            idx = np.random.choice(a=total_s, size=n, replace=False).flatten()
        else:
            raise NotImplementedError

        selected_ids = [ids[z] for z in idx]
        self.fix_random()

        return selected_ids

    def _get_fd(self, fd: str, glabel: int, separate_by_class: bool):
        if not separate_by_class:
            return fd

        assert isinstance(glabel, int)
        assert os.path.isdir(fd)
        new_fd = join(normpath(fd), f'{glabel}')
        self.create_folder(new_fd)
        return new_fd

    def draw_some_best_pred(self, nbr=200, separate=False,
                            separate_orient=constants.PLOT_HOR, compress=True,
                            store_imgs=False, store_cams_alone=False,
                            plot_cam_on_img=False, tag_cam_on_img=False,
                            plot_gt_on_img=False, separate_by_class=False):
        print('Drawing some pictures')
        assert self.evaluator.best_tau_list != []
        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list
        self.assert_tau_list()

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        # todo: optimize. unnecessary loading of useless samples.
        # for idxb, (images, targets, image_ids, raw_imgs, _, _) in tqdm(
        #         enumerate(self.loader), ncols=constants.NCOLS,
        #         total=len(self.loader)):
        img_fd = join(self.out_folder, 'vizu/imgs')
        raw_fd = join(self.out_folder, 'vizu/cams_alone/low_res')
        cam_fd_h = join(self.out_folder, 'vizu/cams_alone/high_res')
        cam_on_img_fd = join(self.out_folder, 'vizu/cam_on_img')
        gt_on_img_fd = join(self.out_folder, 'vizu/gt_on_img')

        for fdx in [img_fd, raw_fd, cam_fd_h, cam_on_img_fd, gt_on_img_fd]:
            self.create_folder(fdx)

        for _image_id in tqdm(ids_to_draw, ncols=constants.NCOLS, total=len(
                ids_to_draw)):
            img_idx = self.loader.dataset.index_id[_image_id]
            image, target, image_id, raw_img, _, _ = self.loader.dataset[
                img_idx]
            assert image_id == _image_id

            self.fix_random()

            image_size = image.shape[1:]
            image = image.cuda(self.args.c_cudaid)

            # raw_img: 3, h, w
            raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
            raw_img = raw_img.astype(np.uint8)

            if store_imgs:
                _img_fd = self._get_fd(img_fd, glabel=target,
                                  separate_by_class=separate_by_class)
                Image.fromarray(raw_img).save(
                    join(_img_fd, f'{self.reformat_id(image_id)}.png'))

            with torch.set_grad_enabled(self.req_grad):
                low_cam, _ = self.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target)

            with torch.no_grad():
                cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False
                                    ).squeeze(0).squeeze(0)

                cam = torch.clamp(cam, min=0.0, max=1.)

            if store_cams_alone:
                _raw_fd = self._get_fd(raw_fd, glabel=target,
                                       separate_by_class=separate_by_class)
                self.viz.plot_cam_raw(
                    t2n(low_cam),
                    outf=join(_raw_fd, f'{self.reformat_id(image_id)}.png'),
                    interpolation='none')

                _cam_fd_h = self._get_fd(cam_fd_h, glabel=target,
                                         separate_by_class=separate_by_class)
                self.viz.plot_cam_raw(
                    t2n(cam),
                    outf=join(_cam_fd_h, f'{self.reformat_id(image_id)}.png'),
                    interpolation='bilinear')

            cam = torch.clamp(cam, min=0.0, max=1.)
            cam = t2n(cam)

            # cams shape (h, w).
            assert cam.shape == image_size

            cam_resized = cam
            cam_normalized = cam_resized
            check_scoremap_validity(cam_normalized)

            if isinstance(self.evaluator, BoxEvaluator):
                self.assert_datatset_bbx()
                l_datum = []
                ploted_cam_on_img = False if plot_cam_on_img else True
                ploted_gt_on_img = False if plot_gt_on_img else True

                for k, _THRESHOLD in enumerate(iou_threshold_list):
                    th_fd = join(self.out_folder, 'vizu', str(_THRESHOLD))
                    self.create_folder(th_fd)

                    tau = best_tau_list[k]
                    best_bbox, bbox_iou = self.build_bbox(
                        scoremap=cam_normalized, image_id=image_id,
                        tau=tau
                    )
                    gt_bbx = self.evaluator.gt_bboxes[image_id]
                    gt_bbx = np.array(gt_bbx)
                    datum = {'img': raw_img, 'img_id': image_id,
                             'gt_bbox': gt_bbx,
                             'pred_bbox': best_bbox.reshape((1, 4)),
                             'iou': bbox_iou, 'tau': tau,
                             'sigma': _THRESHOLD, 'cam': cam_normalized}

                    if separate:
                        _th_fd = self._get_fd(
                            th_fd, glabel=target,
                            separate_by_class=separate_by_class)
                        outf = join(_th_fd, '{}.png'.format(self.reformat_id(
                            image_id)))
                        self.viz.plot_single(datum=datum, outf=outf,
                                             orient=separate_orient)

                    if plot_cam_on_img and not ploted_cam_on_img:
                        _cam_on_img_fd = self._get_fd(
                            cam_on_img_fd, glabel=target,
                            separate_by_class=separate_by_class)

                        outf = join(_cam_on_img_fd,
                                    '{}.png'.format(self.reformat_id(image_id)))
                        self.viz.plot_single_cam_on_img(datum=datum, outf=outf,
                                                        tagit=tag_cam_on_img)

                        ploted_cam_on_img = True

                    if plot_gt_on_img and not ploted_gt_on_img:
                        _gt_on_img_fd = self._get_fd(
                            gt_on_img_fd, glabel=target,
                            separate_by_class=separate_by_class)

                        outf = join(_gt_on_img_fd,
                                    '{}.png'.format(self.reformat_id(image_id)))
                        self.viz.plot_single_gt_on_img(datum=datum, outf=outf)

                        ploted_gt_on_img = True

                    l_datum.append(datum)

                th_fd = join(self.out_folder, 'vizu', 'all_taux')
                self.create_folder(th_fd)
                _th_fd = self._get_fd(
                    th_fd, glabel=target,
                    separate_by_class=separate_by_class)

                outf = join(_th_fd, '{}.png'.format(self.reformat_id(
                    image_id)))
                self.viz.plot_multiple(data=l_datum, outf=outf)

            elif isinstance(self.evaluator, MaskEvaluator):
                self.assert_dataset_mask()
                taux = sorted(list({0.5, 0.6, 0.7, 0.8, 0.9,
                                    best_tau_list[0]}))
                gt_mask = get_mask(self.evaluator.mask_root,
                                   self.evaluator.mask_paths[image_id],
                                   self.evaluator.ignore_paths[image_id])
                # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)

                l_datum = []
                ploted_cam_on_img = False if plot_cam_on_img else True
                ploted_gt_on_img = False if plot_gt_on_img else True

                for tau in taux:
                    th_fd = join(self.out_folder, 'vizu', str(tau))
                    self.create_folder(th_fd)
                    l_datum.append(
                        {'img': raw_img, 'img_id': image_id,
                         'gt_mask': gt_mask, 'tau': tau,
                         'best_tau': tau == best_tau_list[0],
                         'cam': cam_normalized}
                    )
                    # todo: plotting singles is not necessary for now.
                    # todo: control it latter for standalone inference.
                    if separate:
                        _th_fd = self._get_fd(
                            th_fd, glabel=target,
                            separate_by_class=separate_by_class)
                        outf = join(_th_fd, '{}.png'.format(self.reformat_id(
                            image_id)))
                        self.viz.plot_single(datum=l_datum[-1], outf=outf,
                                             orient=separate_orient)

                    if plot_cam_on_img and not ploted_cam_on_img:
                        _cam_on_img_fd = self._get_fd(
                            cam_on_img_fd, glabel=target,
                            separate_by_class=separate_by_class)
                        outf = join(_cam_on_img_fd,
                                    '{}.png'.format(self.reformat_id(image_id)))
                        self.viz.plot_single_cam_on_img(datum=l_datum[-1],
                                                        outf=outf,
                                                        tagit=tag_cam_on_img)

                        ploted_cam_on_img = True

                    if plot_gt_on_img and not ploted_gt_on_img:
                        _gt_on_img_fd = self._get_fd(
                            gt_on_img_fd, glabel=target,
                            separate_by_class=separate_by_class)

                        outf = join(_gt_on_img_fd,
                                    '{}.png'.format(self.reformat_id(image_id)))
                        self.viz.plot_single_gt_on_img(datum=l_datum[-1],
                                                       outf=outf)

                        ploted_gt_on_img = True

                th_fd = join(self.out_folder, 'vizu', 'some_taux')
                self.create_folder(th_fd)
                _th_fd = self._get_fd(th_fd, glabel=target,
                                      separate_by_class=separate_by_class)

                outf = join(_th_fd, '{}.png'.format(self.reformat_id(
                    image_id)))
                self.viz.plot_multiple(data=l_datum, outf=outf)
            else:
                raise NotImplementedError

        if compress:
            self.compress_fdout(self.out_folder, 'vizu')

    def compress_fdout(self, parent_fd, fd_trg):
        assert os.path.isdir(join(parent_fd, fd_trg))

        cmdx = [
            "cd {} ".format(parent_fd),
            "tar -cf {}.tar.gz {} ".format(fd_trg, fd_trg),
            "rm -r {} ".format(fd_trg)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))

    def _watch_plot_perfs_meter(self, split: str, meters: dict, perfs: list,
                                fout: str):
        out = self.viz._watch_plot_perfs_meter(meters=meters, split=split,
                                               perfs=perfs, fout=fout)
        pklout = join(dirname(fout), '{}.pkl'.format(os.path.basename(
            fout).split('.')[0]))
        with open(pklout, 'wb') as fx:
            pkl.dump(out, file=fx, protocol=pkl.HIGHEST_PROTOCOL)

    def _watch_build_histogram_scores_cams(self, split):
        print('Building histogram of cams scores. ')
        threshs = list(np.arange(0, 1, 0.001)) + [1.]
        density = 0
        cnt = 0.
        for budx, (images, targets, image_ids, raw_imgs, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                with torch.set_grad_enabled(self.req_grad):
                    low_cam, _ = self.get_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                with torch.no_grad():
                    cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                    cam = torch.clamp(cam, min=0.0, max=1.)
                cam = t2n(cam)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = cam
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)
                _density, bins = np.histogram(
                    cam_normalized,
                    bins=threshs,
                    density=False)

                _density = _density / _density.sum()
                density += _density

                cnt += 1.
        density /= cnt
        density *= 100.

        basename = 'histogram_normalized_cams-{}'.format(split)
        outf = join(self.out_folder, 'vizu/{}.png'.format(basename))
        self.viz._watch_plot_histogram_activations(density=density,
                                                   bins=bins,
                                                   outf=outf,
                                                   split=split)
        outf = join(self.out_folder, 'vizu/{}.pkl'.format(basename))
        with open(outf, 'wb') as fout:
            pkl.dump({'density': density, 'bins': bins}, file=fout,
                     protocol=pkl.HIGHEST_PROTOCOL)

    def _watch_build_store_std_cam_low(self, fdout):
        print('Building low res. cam and storing them.')
        for idx, (images, targets, image_ids, raw_imgs, _, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    if self.special1:
                        output = self.model(image.unsqueeze(0),
                                            labels=target.item())
                    else:
                        output = self.model(image.unsqueeze(0))

                    assert self.args.task == constants.STD_CL
                    cl_logits = output
                    cam = self.std_cam_extractor(class_idx=target.item(),
                                                 scores=cl_logits,
                                                 normalized=True,
                                                 reshape=image_size if
                                                 self.special1 else None)
                    # (h`, w`)

                    # Quick fix: todo...
                    cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)

                    cam = cam.detach().cpu()
                torch.save(cam, join(fdout, '{}.pt'.format(reformat_id(
                    image_id))))

    def _watch_get_std_cam_one_sample(self, image: torch.Tensor, target: int,
                                      ) -> torch.Tensor:

        img_sz = image.shape[2:]
        if self.special1:
            output = self.model(image, labels=target)
        else:
            output = self.model(image)

        assert self.args.task == constants.STD_CL
        cl_logits = output
        cam = self.std_cam_extractor(class_idx=target,
                                     scores=cl_logits,
                                     normalized=False,
                                     reshape=img_sz if self.special1 else None)
        # (h`, w`)

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        return cam

    def _watch_analyze_entropy_std(self, nbr=200):
        assert self.args.task == constants.STD_CL

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        vizu_cam_fd = join(self.out_folder, 'vizualization', 'cams')
        self.create_folder(vizu_cam_fd)

        for images, targets, image_ids, raw_imgs, _ in self.loader:
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    low_cam = self._watch_get_std_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)

                cam = cam.detach()

                cam_max_normed = max_normalize(cam)
                cam_entropy = entropy_cam(cam_max_normed)

                cam = t2n(cam)

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = normalize_scoremap(cam.copy())
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)

                if image_id in ids_to_draw:

                    if isinstance(self.evaluator, BoxEvaluator):
                        self.assert_datatset_bbx()
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)
                        datum = {
                            'visu': {
                                'cam': cam,
                                'cam_max_normed': t2n(cam_max_normed),
                                'cam_entropy': t2n(cam_entropy),
                                'cam_normalized': cam_normalized
                            },
                            'tags': {
                                'cam': 'Raw cam',
                                'cam_max_normed': 'Max-normed-cam',
                                'cam_entropy': 'Entropy-Cam',
                                'cam_normalized': 'Normed-cam'
                            },
                            'raw_img': raw_img,
                            'img_id': image_id,
                            'gt_bbox': gt_bbx
                        }
                        outf = join(vizu_cam_fd, '{}.png'.format(
                            self.reformat_id(image_id)))

                        self.viz._watch_plot_entropy(data=datum, outf=outf)

                    elif isinstance(self.evaluator, MaskEvaluator):
                       pass
                    else:
                        raise NotImplementedError

        # self.compress_fdout(self.out_folder, 'vizu')

    def _watch_analyze_thresh_std(self, nbr=200):
        assert self.args.task == constants.STD_CL

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        vizu_cam_fd = join(self.out_folder, 'vizualization', 'cams-thresh')
        self.create_folder(vizu_cam_fd)

        threshs = list(np.arange(0, 1, 0.001))

        for images, targets, image_ids, raw_imgs, _ in self.loader:
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda()

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    low_cam = self._watch_get_std_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                with torch.no_grad():
                    cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                cam = cam.detach()
                cam = t2n(cam)

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = normalize_scoremap(cam.copy())
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)
                grey_cam = (cam_normalized * 255).astype(np.uint8)
                t0 = dt.datetime.now()
                otsu_thresh = threshold_otsu(grey_cam)
                print('otsu thresholding took: {}'.format(dt.datetime.now() -
                                                          t0))
                t0 = dt.datetime.now()
                li_thres = filters.threshold_li(grey_cam,
                                                initial_guess=otsu_thresh)
                print('li thresholding took: {}'.format(dt.datetime.now() -
                                                        t0))
                fg_otsu = (grey_cam > otsu_thresh).astype(np.uint8) * 255
                fg_li = (grey_cam > li_thres).astype(np.uint8) * 255

                fg_auto = (grey_cam > .2 * grey_cam.max()).astype(
                    np.uint8) * 255
                # erosion
                kernel_erose = torch.ones(11, 11).to(low_cam.device)
                fg_li_torch = torch.from_numpy(fg_li).to(
                    low_cam.device).unsqueeze(0).unsqueeze(0)
                li_eroded = kornia.morphology.erosion(fg_li_torch * 1.,
                                                      kernel_erose)
                li_eroded = li_eroded.cpu().squeeze().numpy().astype(
                    np.uint8) * 255
                fg_otsu_torch = torch.from_numpy(fg_otsu).to(
                    low_cam.device).unsqueeze(0).unsqueeze(0)
                ostu_eroded = fg_otsu_torch * 1.
                for kkk in range(2):
                    ostu_eroded = kornia.morphology.erosion(ostu_eroded,
                                                            kernel_erose)
                otsu_eroded = ostu_eroded.cpu().squeeze().numpy().astype(
                    np.uint8) * 255

                density, bins = np.histogram(
                    cam_normalized,
                    bins=threshs,
                    density=True)

                datum = {
                    'visu': {
                        'cam': cam,
                        'cam_normalized': cam_normalized,
                        'density': (density / density.sum(), bins),
                        'discrete_cam': (cam_normalized *
                                         255).astype(np.uint8),
                        'bin_otsu': fg_otsu,
                        'bin_li': fg_li,
                        'otsu_bin_eroded': otsu_eroded,
                        'li_bin_eroded':  li_eroded,
                        'fg_auto': fg_auto
                    },
                    'tags': {
                        'cam': 'Raw cam',
                        'cam_normalized': 'Cam normed',
                        'density': 'Cam-normed histo',
                        'discrete_cam': 'Discrete normed cam',
                        'bin_otsu': 'FG Otsu',
                        'bin_li': 'FG Li',
                        'otsu_bin_eroded': 'UTSU ERODED',
                        'li_bin_eroded': 'LI ERODED',
                        'fg_auto': 'FG AUTO'
                    },
                    'raw_img': raw_img,
                    'img_id': image_id,
                    'nbins': len(threshs),
                    'otsu_thresh': otsu_thresh / 255.,
                    'li_thres': li_thres / 255.
                }

                if image_id in ids_to_draw:

                    if isinstance(self.evaluator, BoxEvaluator):
                        self.assert_datatset_bbx()
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)
                        datum['gt_bbox'] = gt_bbx

                    elif isinstance(self.evaluator, MaskEvaluator):
                        self.assert_dataset_mask()
                        gt_mask = get_mask(self.evaluator.mask_root,
                                           self.evaluator.mask_paths[image_id],
                                           self.evaluator.ignore_paths[image_id])
                        # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)
                        datum['gt_mask'] = gt_mask
                    else:
                        raise NotImplementedError

                    outf = join(vizu_cam_fd, '{}.png'.format(
                        self.reformat_id(image_id)))
                    self.viz._watch_plot_thresh(data=datum, outf=outf)
