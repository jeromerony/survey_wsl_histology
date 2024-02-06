import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.cams import selflearning
from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor

from dlib.parallel import sync_tensor_across_gpus
from dlib.parallel import MyDDP as DDP

from dlib.div_classifiers.parts.has import has as wsol_has
from dlib.div_classifiers.parts.cutmix import cutmix as wsol_cutmix

from dlib import losses


__all__ = ['Basic', 'Trainer']


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        val = constants.VALIDSET
        # self.value_per_epoch = [] \
        #     if split == val else [-np.inf if higher_is_better else np.inf]
        self.value_per_epoch = []

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        if len(self.value_per_epoch) > 1:
            idx = [i for i, x in enumerate(
                self.value_per_epoch) if x == self.best_value]
            if len(idx) > 0:
                self.best_epoch = idx[-1]
            else:
                self.best_epoch = 0  # issue: nan, inf.
        else:
            self.best_epoch = 0  # issue: nan. inf.


class PerfGistTracker(object):
    def __init__(self, split):
        self.value_per_epoch = dict()
        self.split = split

    def update(self, epoch: int, new_value: dict):
        assert epoch not in self.value_per_epoch
        self.value_per_epoch[epoch] = new_value

    def __getitem__(self, item):
        return self.value_per_epoch[item]


class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.PXVALIDSET, constants.CLVALIDSET,
               constants.TESTSET)
    _EVAL_METRICS = ['loss', constants.CLASSIFICATION_MTR,
                     constants.LOCALIZATION_MTR]

    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
        constants.GLAS: constants.NUMBER_CLASSES[constants.GLAS],
        constants.CAMELYON512: constants.NUMBER_CLASSES[constants.CAMELYON512],
        constants.ICIAR: constants.NUMBER_CLASSES[constants.ICIAR],
        constants.BREAKHIS: constants.NUMBER_CLASSES[constants.BREAKHIS]
    }

    @property
    def _BEST_CRITERION_METRIC(self):
        assert self.inited
        assert self.args is not None

        if self.args.localization_avail:
            return constants.LOCALIZATION_MTR
        else:
            return constants.CLASSIFICATION_MTR

    def __init__(self, args):
        self.args = args

    def _set_perf_gist_tracker(self) -> dict:
        _dict = {
            split: PerfGistTracker(split) for split in self._SPLITS
        }
        return _dict

    def _set_performance_meters(self) -> dict:

        if self.bbox:
            self._EVAL_METRICS += ['localization_IOU_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

            self._EVAL_METRICS += ['top1_loc_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

            self._EVAL_METRICS += ['top5_loc_{}'.format(
                threshold) for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict


class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss: losses.MasterLoss,
                 classifier=None):
        super(Trainer, self).__init__(args=args)

        self.device = torch.device(args.c_cudaid)
        self.args = args

        self.bbox = args.dataset in [constants.CUB, constants.ILSVRC]

        self.performance_meters = self._set_performance_meters()
        self.perf_gist_tracker = self._set_perf_gist_tracker()
        self.model = model

        if isinstance(model, DDP):
            self._pytorch_model = self.model.module
        else:
            self._pytorch_model = self.model

        self.loss: losses.MasterLoss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.load_tr_masks = args.task == constants.SEG
        self.load_tr_masks &= args.localization_avail
        mask_root = args.mask_root if self.load_tr_masks else ''

        self.loaders, self.train_sampler = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            load_tr_masks=self.load_tr_masks,
            mask_root=mask_root,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder
        )

        self.sl_mask_builder = None
        if args.task in [constants.F_CL, constants.NEGEV]:
            self.sl_mask_builder = self._get_sl(args)

        self.epoch = 0
        self.counter = 0
        self.seed = int(args.MYSEED)
        self.default_seed = int(args.MYSEED)

        self.best_loc_model = deepcopy(self._pytorch_model).to(
            self.cpu_device).eval()
        self.best_cl_model = deepcopy(self._pytorch_model).to(
            self.cpu_device).eval()

        self.perf_meters_backup = None
        self.perf_gist_backup = None
        self.inited = True

        self.classifier = classifier
        self.std_cam_extractor = None
        if args.task in [constants.F_CL, constants.NEGEV]:
            assert classifier is not None
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

        self.best_valid_tau_loc = None
        self.best_valid_tau_cl = None

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _get_sl(self, args):
        if args.task == constants.F_CL:
            return selflearning.MBSeederSLFCAMS(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

        elif args.task == constants.NEGEV:

            if args.sl_ng_seeder == constants.SEED_TH:
                return selflearning.MBSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

            elif args.sl_ng_seeder == constants.SEED_PROB:
                return selflearning.MBProbSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx
                )
            elif args.sl_ng_seeder == constants.SEED_PROB_N_AREA:
                return selflearning.MBProbNegAreaSeederSLNEGEV(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    min_p=args.sl_ng_min_p,
                    ksz=args.sl_ksz,
                    seg_ignore_idx=args.seg_ignore_idx
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


    def prepare_std_cams_disq(self, std_cams: torch.Tensor,
                              image_size: Tuple) -> torch.Tensor:

        assert std_cams.ndim == 4
        cams = std_cams.detach()

        # cams: (bsz, 1, h, w) == image_size.
        assert cams.ndim == 4
        # Quick fix: todo...
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        # todo: unnecessary. cams and images have same size.
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        return cams

    def get_std_cams_minibatch(self, images, targets) -> torch.Tensor:
        # used only for task f_cl/negev
        assert self.args.task in [constants.F_CL, constants.NEGEV]
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            cl_logits = self.classifier(image.unsqueeze(0))
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def is_seed_required(self, _epoch):
        cmd = (self.args.task in [constants.F_CL, constants.NEGEV])
        cmd &= (('self_learning_fcams' in self.loss.n_holder) or (
            'self_learning_negev' in self.loss.n_holder
        ))
        cmd2 = False
        for _l in self.loss.losses:
            if isinstance(_l, losses.SelfLearningFcams):
                cmd2 = _l.is_on(_epoch=_epoch)

        return cmd and cmd2

    def _do_cutmix(self):
        return (self.args.method == constants.METHOD_CUTMIX and
                self.args.cutmix_prob > np.random.rand(1).item() and
                self.args.cutmix_beta > 0)

    def _one_step_train(self, images, raw_imgs, targets, std_cams, masks):
        y_global = targets

        if self.args.method == constants.METHOD_HAS:
            images = wsol_has(image=images, grid_size=self.args.has_grid_size,
                              drop_rate=self.args.has_drop_rate)

        cutmix_holder = None
        if self.args.method == constants.METHOD_CUTMIX:
            if self._do_cutmix():
                images, target_a, target_b, lam = wsol_cutmix(
                    x=images, target=targets, beta=self.args.cutmix_beta)
                cutmix_holder = [target_a, target_b, lam]
                # todo: this holder is never used.

        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            loss = self.loss(epoch=self.epoch,
                             model=self.model,
                             cl_logits=cl_logits,
                             glabel=y_global,
                             cutmix_holder=cutmix_holder
                             )
            logits = cl_logits

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits

        elif self.args.task == constants.NEGEV:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits

        elif self.args.task == constants.SEG:
            assert masks is not None
            assert isinstance(masks, torch.Tensor)
            assert masks.ndim == 4
            assert masks.shape[1] == 1

            seg_logits = output
            loss = self.loss(seg_logits=seg_logits, masks=masks.squeeze(1))
            logits = None
        else:
            raise NotImplementedError

        return logits, loss

    def _fill_minibatch(self, _x: torch.Tensor, mbatchsz: int) -> torch.Tensor:
        assert isinstance(_x, torch.Tensor)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0

        if _x.shape[0] == mbatchsz:
            return _x

        s = _x.shape[0]
        t = math.ceil(float(mbatchsz) / s)
        v = torch.cat(t * [_x])
        assert v.shape[1:] == _x.shape[1:]

        out = v[:mbatchsz]
        assert out.shape[0] == mbatchsz
        return out

    def on_epoch_start(self):
        torch.cuda.empty_cache()

        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.model.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))

        torch.cuda.empty_cache()

    def random(self):
        self.counter = self.counter + 1
        self.seed = self.seed + self.counter
        set_seed(seed=self.seed, verbose=False)

    def train(self, split: str, epoch: int) -> dict:
        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        assert split == constants.TRAINSET
        loader = self.loaders[split]

        total_loss = None
        num_correct = 0
        num_images = 0

        scaler = GradScaler(enabled=self.args.amp)

        mbatchsz = 0

        for batch_idx, (images, targets, _, raw_imgs, std_cams, masks) in tqdm(
                enumerate(loader), ncols=constants.NCOLS, total=len(loader)):
            
            self.random()
            if batch_idx == 0:
                mbatchsz = images.shape[0]

            # fill
            images = self._fill_minibatch(images, mbatchsz)
            targets = self._fill_minibatch(targets, mbatchsz)
            raw_imgs = self._fill_minibatch(raw_imgs, mbatchsz)

            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)

            if masks.ndim == 1:
                masks = None
            else:
                masks = self._fill_minibatch(masks, mbatchsz)
                masks = masks.cuda(self.args.c_cudaid)

            if std_cams.ndim == 1:
                std_cams = None
            else:
                assert std_cams.ndim == 4
                std_cams = self._fill_minibatch(std_cams, mbatchsz)
                std_cams = std_cams.cuda(self.args.c_cudaid)

                with autocast(enabled=self.args.amp):
                    with torch.no_grad():
                        std_cams = self.prepare_std_cams_disq(
                            std_cams=std_cams, image_size=images.shape[2:])

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.amp):
                logits, loss = self._one_step_train(
                    images, raw_imgs, targets, std_cams, masks)

            with torch.no_grad():
                if self.args.task != constants.SEG:
                    pred = logits.argmax(dim=1)
                    num_correct += (pred == targets).sum().detach()

            if total_loss is None:
                total_loss = loss.detach().squeeze() * images.size(0)
            else:
                total_loss += loss.detach().squeeze() * images.size(0)
            num_images += images.size(0)

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        if self.args.distributed:
            nxx = torch.tensor([num_images], dtype=torch.float,
                               requires_grad=False, device=torch.device(
                    self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nxx).sum().item()
            total_loss = sync_tensor_across_gpus(total_loss.view(1, )).sum()

        loss_average = total_loss.item() / float(num_images)

        classification_acc = 0.0
        if self.args.task != constants.SEG:

            if self.args.distributed:
                num_correct = sync_tensor_across_gpus(
                    num_correct.view(1, )).sum()

            classification_acc = num_correct.item() / float(num_images) * 100

        dist.barrier()

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self, checkpoint_type=None):
        # todo: adapt.
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def serialize_perf_gist_tracker(self) -> dict:
        return {
            split: vars(self.perf_gist_tracker[split]) for split in self._SPLITS
        }

    def save_performances(self, epoch: int, checkpoint_type: str):
        assert isinstance(epoch, int)
        assert isinstance(checkpoint_type, str)
        assert checkpoint_type != ''

        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        tag = '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))

        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        log_path_g = join(self.args.outd,
                          'performance_gist_tracker_log{}{}.pickle'.format(
                            tag, tagargmax))
        with open(log_path_g, 'wb') as f:
            pkl.dump(self.serialize_perf_gist_tracker(), f,
                     protocol=pkl.HIGHEST_PROTOCOL)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            _splits = [constants.TESTSET]
            for split in _splits:
                for metric in self._EVAL_METRICS:
                    if self._skip_print_metric(metric=metric, _split=split,
                                               split=split):
                        continue

                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, split, metric,
                                      self.performance_meters[split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, split, metric,
                                    self.performance_meters[split][
                                        metric].best_value))

                if self.args.localization_avail:
                    f.write(f'REPORT EPOCH/{epoch} split: {split}: \n'
                            f'{self.perf_gist_to_str(split, epoch)} \n')

    def perf_gist_to_str(self, split: str, epoch: int) -> str:
        out = [
            f'{k}: '
            f'{self.perf_gist_tracker[split].value_per_epoch[epoch][k]}'
            for k in self.perf_gist_tracker[split].value_per_epoch[epoch]
        ]
        return '\n'.join(out)

    def cl_forward(self, images):
        output = self._pytorch_model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task in [constants.F_CL, constants.NEGEV]:
            cl_logits, fcams, im_recon = output

        else:
            raise NotImplementedError

        return cl_logits

    def _compute_accuracy(self, loader):
        torch.cuda.empty_cache()

        num_correct = 0
        num_images = 0

        for i, (images, targets, _, _, _, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            with torch.no_grad():
                cl_logits = self.cl_forward(images)
                pred = cl_logits.argmax(dim=1)

            num_correct += (pred == targets).sum().detach()
            num_images += images.size(0)

        # sync
        if self.args.distributed:
            num_correct = sync_tensor_across_gpus(
                num_correct.view(1, )).sum()
            nx = torch.tensor(
                [num_images], dtype=torch.float, requires_grad=False,
                device=torch.device(self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nx).sum().item()
            dist.barrier()

        classification_acc = num_correct / float(num_images) * 100
        dist.barrier()

        torch.cuda.empty_cache()
        return classification_acc.item()

    def evaluate(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        torch.cuda.empty_cache()
        assert split in [constants.TESTSET, constants.VALIDSET]

        if split == constants.TESTSET:
            splitpx = split
            splitcl = split
        elif split == constants.VALIDSET:
            splitpx = constants.PXVALIDSET
            splitcl = constants.CLVALIDSET
        else:
            raise NotImplementedError

        if fcam_argmax:
            assert self.args.task in [constants.F_CL, constants.NEGEV,
                                      constants.SEG]

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task in [constants.F_CL, constants.NEGEV]:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
        elif split == constants.VALIDSET:
            _chpt = 'training' if checkpoint_type is None else checkpoint_type
            if fcam_argmax:
                outd = join(self.args.outd, _chpt, 'argmax-true', split)
            else:
                outd = join(self.args.outd, _chpt, split)
        else:
            raise NotImplementedError

        os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()
        self._pytorch_model.eval()

        # cl.
        accuracy = 0.0
        if self.args.task != constants.SEG:
            accuracy = self._compute_accuracy(loader=self.loaders[splitcl])

        self.performance_meters[
            splitcl][constants.CLASSIFICATION_MTR].update(accuracy)

        # loc.
        if not self.args.localization_avail:
            return None

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.CUB, constants.ILSVRC]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        if split == constants.VALIDSET:
            best_valid_tau = None
        elif split == constants.TESTSET:
            if checkpoint_type == constants.BEST_LOC:
                best_valid_tau = self.best_valid_tau_loc
            elif checkpoint_type == constants.BEST_CL:
                best_valid_tau = self.best_valid_tau_cl
            else:
                raise NotImplementedError
            assert best_valid_tau is not None
        else:
            raise NotImplementedError
        # todo: remove code above. to measure segmentation performance,
        #  we estimate the bes tthreshold over the current set. e.g. in the
        #  case of testset, we estimate the best segmentation metrics with
        #  the best threshold obtained with miou.
        best_valid_tau = None

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self._pytorch_model,
            loader=self.loaders[splitpx],
            metadata_root=os.path.join(self.args.metadata_root, splitpx),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=splitpx,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax,
            best_valid_tau=best_valid_tau
        )
        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg("CAM EVALUATE TIME of {} split: {}".format(
            split, dt.datetime.now() - t0)))

        if split == constants.TESTSET and self.args.is_master:
            cam_computer.draw_some_best_pred()

        avg = self.args.multi_iou_eval
        avg |= self.args.dataset in [constants.OpenImages, constants.GLAS,
                                     constants.CAMELYON512]
        if avg:
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[splitpx][constants.LOCALIZATION_MTR].update(
            loc_score)

        self.perf_gist_tracker[splitpx].update(
            epoch=epoch, new_value=cam_computer.evaluator.perf_gist)

        if split in [constants.TESTSET,
                     constants.VALIDSET] and self.args.is_master:

            curves = cam_computer.evaluator.curve_s
            if split == constants.TESTSET:
                with open(join(outd, 'curves.pkl'), 'wb') as fc:
                    pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)
            elif split == constants.VALIDSET:
                if checkpoint_type is None:
                    _outd = None
                    if self._is_best_model_cl(epoch):
                        _outd = join(outd, constants.BEST_CL)
                        os.makedirs(_outd, exist_ok=True)
                        with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                            pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)
                    if self._is_best_model_loc(epoch):
                        _outd = join(outd, constants.BEST_LOC)
                        os.makedirs(_outd, exist_ok=True)
                        with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                            pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

                else:
                    _outd = outd
                    os.makedirs(_outd, exist_ok=True)
                    with open(join(_outd, 'curves.pkl'), 'wb') as fc:
                        pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            if split == constants.TESTSET:
                title = get_tag(self.args, checkpoint_type=checkpoint_type)

                if fcam_argmax:
                    title += '_argmax_true.'
                else:
                    title += '_argmax_false.'
                title += r' Best $\tau$: {}'.format(
                    curves[constants.MTR_BESTTAU][0])
                self.plot_loc_perf_curves(
                    curves=curves, fdout=outd, title=title,
                    checkpoint_type=checkpoint_type)

        torch.cuda.empty_cache()

    def plot_loc_perf_curves(self, curves: dict, fdout: str, title: str,
                             checkpoint_type: str):

        ncols = 4
        ks = ['y', constants.MTR_MIOU, constants.MTR_TP, constants.MTR_TN,
              constants.MTR_FP, constants.MTR_FN, constants.MTR_DICEFG,
              constants.MTR_DICEBG]

        best_tau = curves[constants.MTR_BESTTAU][0]
        x_tau = curves['threshold_list_right_edge'][:-1]
        x_pxap = curves['x']
        nbr_th = len(x_tau)
        idx = curves['idx']
        hand = checkpoint_type
        assert idx < len(x_tau)
        assert idx < x_pxap.size

        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)

        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                xx_t = None
                if t == 0:
                    val = curves['y']
                    x = x_pxap
                    subtitle = 'Recall/precision'
                    x_label = 'Recall'
                    y_label = 'Precision'
                    axes[i, j].plot(x, val, color='tab:orange')
                else:
                    val = curves[ks[t]][:nbr_th]
                    x = x_tau
                    n = len(x)
                    subtitle = ks[t]
                    x_label = r'$\tau$'
                    y_label = None
                    xx_t = list(range(n))
                    axes[i, j].plot(xx_t, val, color='tab:orange')

                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel(x_label, fontsize=4)
                if y_label is not None:
                    axes[i, j].set_ylabel(y_label, fontsize=4)

                axes[i, j].grid(True)

                if ks[t] != 'y':
                    # todo: not perfect but ok.
                    a = axes[i, j].get_xticks().tolist()
                    aa = [z / 1000. for z in a]
                    axes[i, j].set_xticklabels(aa)

                if t == 0:
                    axes[i, j].plot([x[idx]], [val[idx]],
                                    marker='o',
                                    markersize=5,
                                    color=constants.COLOUR_BEST_CP[hand],
                                    label=hand
                                    )
                    axes[i, j].legend(loc='best', prop={'size': 5})
                else:
                    axes[i, j].plot([xx_t[idx]], [val[idx]],
                                    marker='o',
                                    markersize=5,
                                    color=constants.COLOUR_BEST_CP[hand]
                                    )

                t += 1

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
        fig.savefig(join(fdout, 'curves_loc_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)
        self.perf_gist_backup = deepcopy(self.perf_gist_tracker)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.perf_gist_tracker = deepcopy(self.perf_gist_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_best_epoch(self):
        if self.args.localization_avail:
            split = constants.PXVALIDSET

            best_loc_epoch = self.performance_meters[split][
                constants.LOCALIZATION_MTR].best_epoch
            self.args.best_loc_epoch = best_loc_epoch

        if self.args.task != constants.SEG:
            split = constants.CLVALIDSET

            best_cl_epoch = self.performance_meters[split][
                constants.CLASSIFICATION_MTR].best_epoch
            self.args.best_cl_epoch = best_cl_epoch

    def save_checkpoints(self):
        if self.args.localization_avail:
            split = constants.PXVALIDSET

            best_epoch = self.performance_meters[split][
                constants.LOCALIZATION_MTR].best_epoch

            self._save_model(checkpoint_type=constants.BEST_LOC,
                             epoch=best_epoch)

        if self.args.task != constants.SEG:
            split = constants.CLVALIDSET

            best_epoch = self.performance_meters[split][
                constants.CLASSIFICATION_MTR].best_epoch
            self._save_model(checkpoint_type=constants.BEST_CL,
                             epoch=best_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        if checkpoint_type == constants.BEST_CL:
            _model = deepcopy(self.best_cl_model).to(self.cpu_device).eval()
        elif checkpoint_type == constants.BEST_LOC:
            _model = deepcopy(self.best_loc_model).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)
        if not os.path.isdir(path):
            os.makedirs(path)

        if self.args.task == constants.STD_CL:
            if self.args.method in [constants.METHOD_ACOL,
                                    constants.METHOD_ADL,
                                    constants.METHOD_SPG,
                                    constants.METHOD_TSCAM]:
                torch.save(_model.state_dict(),
                           join(path, 'model.pt'))

            elif self.args.method == constants.METHOD_MAXMIN:
                torch.save(_model.encoder.state_dict(),
                           join(path, 'encoder.pt'))
                torch.save(_model.classification_head1.state_dict(),
                           join(path, 'classification_head1.pt'))
                torch.save(_model.classification_head2.state_dict(),
                           join(path, 'classification_head2.pt'))
                if _model.mask_head is not None:
                    torch.save(_model.mask_head.state_dict(),
                               join(path, 'mask_head.pt'))

            else:
                torch.save(_model.encoder.state_dict(),
                           join(path, 'encoder.pt'))
                torch.save(_model.classification_head.state_dict(),
                           join(path, 'classification_head.pt'))

        elif self.args.task in [constants.F_CL, constants.NEGEV]:

            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
            torch.save(_model.classification_head.state_dict(),
                       join(path, 'classification_head.pt'))

            if _model.reconstruction_head is not None:
                torch.save(_model.reconstruction_head.state_dict(),
                           join(path, 'reconstruction_head.pt'))

        elif self.args.task == constants.SEG:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
        else:
            raise NotImplementedError

        self._save_args(path=join(path, 'config_model.yaml'))
        DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                             " {}".format(checkpoint_type, epoch, tag, path))

    def _is_best_model_loc(self, epoch: int) -> bool:
        cnd = self.args.localization_avail
        cnd &= (self.performance_meters[constants.PXVALIDSET][
                    constants.LOCALIZATION_MTR].best_epoch) == epoch

        return cnd

    def _is_best_model_cl(self, epoch: int) -> bool:
        cnd = self.args.task != constants.SEG
        cnd &= (self.performance_meters[constants.CLVALIDSET][
                    constants.CLASSIFICATION_MTR].best_epoch) == epoch

        return cnd

    def model_selection(self, epoch):

        if self._is_best_model_loc(epoch):
            self.best_loc_model = deepcopy(self._pytorch_model).to(
                self.cpu_device).eval()
            self.best_valid_tau_loc = self.perf_gist_tracker[
                constants.PXVALIDSET][epoch][constants.MTR_BESTTAU][0]
            self.args.best_valid_tau_loc = self.best_valid_tau_loc

        if self._is_best_model_cl(epoch):
            self.best_cl_model = deepcopy(self._pytorch_model).to(
                self.cpu_device).eval()

            if self.args.localization_avail:
                self.best_valid_tau_cl = self.perf_gist_tracker[
                    constants.PXVALIDSET][epoch][constants.MTR_BESTTAU][0]
                self.args.best_valid_tau_cl = self.best_valid_tau_cl

    def load_checkpoint(self, checkpoint_type):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)

        if self.args.task == constants.STD_CL:
            if self.args.method in [constants.METHOD_ACOL,
                                    constants.METHOD_ADL,
                                    constants.METHOD_SPG,
                                    constants.METHOD_TSCAM]:
                weights = torch.load(join(path, 'model.pt'),
                                     map_location=self.device)
                self._pytorch_model.load_state_dict(weights, strict=True)

            elif self.args.method == constants.METHOD_MAXMIN:
                weights = torch.load(join(path, 'encoder.pt'),
                                     map_location=self.device)
                self._pytorch_model.encoder.super_load_state_dict(
                    weights, strict=True)

                weights = torch.load(join(path, 'classification_head1.pt'),
                                     map_location=self.device)
                self._pytorch_model.classification_head1.load_state_dict(
                    weights, strict=True)

                weights = torch.load(join(path, 'classification_head2.pt'),
                                     map_location=self.device)
                self._pytorch_model.classification_head2.load_state_dict(
                    weights, strict=True)

                if self._pytorch_model.mask_head is not None:
                    weights = torch.load(join(path, 'mask_head.pt'),
                                         map_location=self.device)
                    self._pytorch_model.mask_head.load_state_dict(
                        weights, strict=True)

            else:
                weights = torch.load(join(path, 'encoder.pt'),
                                     map_location=self.device)
                self._pytorch_model.encoder.super_load_state_dict(
                    weights, strict=True)

                weights = torch.load(join(path, 'classification_head.pt'),
                                     map_location=self.device)
                self._pytorch_model.classification_head.load_state_dict(
                    weights, strict=True)

        elif self.args.task in [constants.F_CL, constants.NEGEV]:

            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self._pytorch_model.encoder.super_load_state_dict(weights,
                                                              strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self._pytorch_model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self._pytorch_model.segmentation_head.load_state_dict(weights,
                                                                  strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=self.device)
            self._pytorch_model.classification_head.load_state_dict(
                weights, strict=True)

            if self._pytorch_model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=self.device)
                self._pytorch_model.reconstruction_head.load_state_dict(
                    weights, strict=True)

        elif self.args.task == constants.SEG:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self._pytorch_model.encoder.super_load_state_dict(weights,
                                                              strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self._pytorch_model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self._pytorch_model.segmentation_head.load_state_dict(weights,
                                                                  strict=True)
        else:
            raise NotImplementedError

        DLLogger.log("Checkpoint {} loaded.".format(path))

    def report_train(self, train_performance: dict, epoch: int):
        split = constants.TRAINSET

        if self.args.task != constants.SEG:
            DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
                epoch, split, train_performance['classification_acc']))

        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None):
        # todo: adapt.
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        _splits = []
        if split in [constants.TRAINSET, constants.TESTSET]:
            _splits = [split]
        elif split == constants.VALIDSET:
            _splits = []
            if self.args.task != constants.SEG:
                _splits = [constants.CLVALIDSET]
            if self.args.localization_avail:
                _splits += [constants.PXVALIDSET]
            assert _splits != []
        else:
            raise NotImplementedError

        for _split in _splits:
            for metric in self._EVAL_METRICS:
                if self._skip_print_metric(metric=metric, _split=_split,
                                           split=split):
                    continue

                DLLogger.log(
                    "REPORT EPOCH/{}: split: {}/metric {}: {} ".format(
                        epoch, _split, metric,
                        self.performance_meters[_split][metric].current_value))
                DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: "
                             "{}_best ".format(
                              epoch, _split, metric,
                              self.performance_meters[_split][
                                  metric].best_value))

        if self.args.localization_avail and split in [
                constants.TESTSET, constants.VALIDSET]:
            _split = constants.TESTSET if split == constants.TESTSET else \
                constants.PXVALIDSET

            if _split == constants.TESTSET:
                _epoch = epoch
            else:
                _epoch = self.performance_meters[_split][
                    constants.LOCALIZATION_MTR].best_epoch
            DLLogger.log(f'REPORT EPOCH/{_epoch} split: {_split}: [BEST]\n'
                         f'{self.perf_gist_to_str(_split, _epoch)} \n')

    def _skip_print_metric(self, metric, _split, split):
        if (metric == constants.CLASSIFICATION_MTR) and (
                self.args.task == constants.SEG):
            return True
        if (metric == constants.LOCALIZATION_MTR) and (
                not self.args.localization_avail):
            return True

        if (metric == 'loss') and split in [
                    constants.TESTSET, constants.VALIDSET]:
            return True

        if (metric == constants.LOCALIZATION_MTR) and (
                _split == constants.CLVALIDSET):
            return True

        if (metric == constants.CLASSIFICATION_MTR) and (
                _split == constants.PXVALIDSET):
            return True

        return False

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_meter(self, metrics: dict, filename: str, title: str = '',
                   xlabel: str = '', best_iter_cl: int = None,
                   best_iter_loc: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                subtitle = ks[t]
                if subtitle == constants.LOCALIZATION_MTR:
                    subtitle = 'PXAP localization'
                if subtitle == constants.CLASSIFICATION_MTR:
                    subtitle = 'Classification accuracy'

                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                for hand, best_iter in zip([constants.BEST_CL,
                                            constants.BEST_LOC],
                                           [best_iter_cl, best_iter_loc]):

                    if best_iter is not None:
                        _best_iter = best_iter if best_iter < len(x) else \
                            len(x) - 1
                        if i == j == 0:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand],
                                            label=hand
                                            )
                            axes[i, j].legend(loc='best', prop={'size': 5})
                        else:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand]
                                            )

                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def clean_metrics(self, metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_gist_tracker(self, gist_tracker: PerfGistTracker,
                          filename: str, title: str = '', xlabel: str = '',
                          best_iter_cl: int = None, best_iter_loc: int = None):

        ncols = 4
        ks = list(gist_tracker.value_per_epoch[0].keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = [gist_tracker.value_per_epoch[elem][ks[t]] for elem in
                       gist_tracker.value_per_epoch]
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                subtitle = ks[t]
                axes[i, j].set_title(subtitle, fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                for hand, best_iter in zip([constants.BEST_CL,
                                            constants.BEST_LOC],
                                           [best_iter_cl, best_iter_loc]):

                    if best_iter is not None:
                        _best_iter = best_iter if best_iter < len(x) else \
                            len(x) - 1
                        if i == j == 0:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand],
                                            label=hand
                                            )
                            axes[i, j].legend(loc='best', prop={'size': 5})
                        else:
                            axes[i, j].plot([x[_best_iter]], [val[_best_iter]],
                                            marker='o',
                                            markersize=5,
                                            color=constants.COLOUR_BEST_CP[
                                                hand]
                                            )

                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        if self.args.localization_avail:
            best_loc_epoch = self.performance_meters[constants.PXVALIDSET][
                constants.LOCALIZATION_MTR].best_epoch
            if self.args.task != constants.SEG:
                best_cl_epoch = self.performance_meters[constants.CLVALIDSET][
                    constants.CLASSIFICATION_MTR].best_epoch
            else:
                best_cl_epoch = None
            splits = [constants.TRAINSET, constants.PXVALIDSET,
                      constants.CLVALIDSET]
        else:
            best_loc_epoch = None
            best_cl_epoch = self.performance_meters[constants.CLVALIDSET][
                constants.CLASSIFICATION_MTR].best_epoch
            splits = [constants.TRAINSET,  constants.CLVALIDSET]

        for split in splits:
            title = f'DS: {self.args.dataset}, Split: {split}. ' \
                    f'Best iter. CL:{best_cl_epoch} ' \
                    f'Best iter. LOC: {best_loc_epoch} ({xlabel})'

            filename = '{}-{}'.format(self.args.dataset, split)
            self.plot_meter(
                self.clean_metrics(meters[split]),
                filename=filename,
                title=title,
                xlabel=xlabel,
                best_iter_cl=best_cl_epoch,
                best_iter_loc=best_loc_epoch
            )

            if split == constants.PXVALIDSET:
                filename = '{}-{}-gist'.format(self.args.dataset, split)
                self.plot_gist_tracker(
                    self.perf_gist_tracker[split],
                    filename=filename,
                    title=title,
                    xlabel=xlabel,
                    best_iter_cl=best_cl_epoch,
                    best_iter_loc=best_loc_epoch
                )



