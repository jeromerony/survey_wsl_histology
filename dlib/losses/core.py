import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from itertools import cycle

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.entropy import Entropy
from dlib import crf

from dlib.div_classifiers.parts.spg import get_loss as get_spg_loss
from dlib.div_classifiers.parts.acol import get_loss as get_acol_loss

from dlib.configure import constants

__all__ = [
    'MasterLoss',
    'ClLoss',
    'SpgLoss',
    'AcolLoss',
    'CutMixLoss',
    'MaxMinLoss',
    'SegLoss',
    'ImgReconstruction',
    'SelfLearningFcams',
    'ConRanFieldFcams',
    'EntropyFcams',
    'MaxSizePositiveFcams',
    #
    'SelfLearningNegev',
    'ConRanFieldNegev',
    'JointConRanFieldNegev',
    'MaxSizePositiveNegev',
    'NegativeSamplesNegev'
]


class _ElementaryLoss(nn.Module):
    def __init__(self,
                 cuda_id,
                 name=None,
                 lambda_=1.,
                 elb=nn.Identity(),
                 logit=False,
                 support_background=False,
                 multi_label_flag=False,
                 sigma_rgb=15.,
                 sigma_xy=100.,
                 scale_factor=0.5,
                 start_epoch=None,
                 end_epoch=None,
                 seg_ignore_idx=-255
                 ):
        super(_ElementaryLoss, self).__init__()
        self._name = name
        self.lambda_ = lambda_
        self.elb = elb
        self.logit = logit
        self.support_background = support_background

        assert not multi_label_flag
        self.multi_label_flag = multi_label_flag

        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

        if end_epoch == -1:
            end_epoch = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.c_epoch = 0

        if self.logit:
            assert isinstance(self.elb, ELB)

        self.loss = None
        self._device = torch.device(cuda_id)

        self._zero = torch.tensor([0.0], device=self._device,
                                  requires_grad=False, dtype=torch.float)

        self.seg_ignore_idx = seg_ignore_idx

    def is_on(self, _epoch=None):
        if _epoch is None:
            c_epoch = self.c_epoch
        else:
            assert isinstance(_epoch, int)
            c_epoch = _epoch

        if (self.start_epoch is None) and (self.end_epoch is None):
            return True

        l = [c_epoch, self.start_epoch, self.end_epoch]
        if all([isinstance(z, int) for z in l]):
            return self.start_epoch <= c_epoch <= self.end_epoch

        if self.start_epoch is None and isinstance(self.end_epoch, int):
            return c_epoch <= self.end_epoch

        if isinstance(self.start_epoch, int) and self.end_epoch is None:
            return c_epoch >= self.start_epoch

        return False

    def unpacke_low_cams(self, cams_low, glabel):
        n = cams_low.shape[0]
        select_lcams = [None for _ in range(n)]

        for i in range(n):
            llabels = [glabel[i]]

            if self.support_background:
                llabels = [xx + 1 for xx in llabels]
                llabels = [0] + llabels

            for l in llabels:
                tmp = cams_low[i, l, :, :].unsqueeze(
                        0).unsqueeze(0)
                if select_lcams[i] is None:
                    select_lcams[i] = tmp
                else:
                    select_lcams[i] = torch.cat((select_lcams[i], tmp), dim=1)

        return select_lcams

    def update_t(self):
        if isinstance(self.elb, ELB):
            self.elb.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            out = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            if isinstance(self.elb, ELB):
                out = out + '_elb'
            if self.logit:
                out = out + '_logit'
            return out
        else:
            return self._name

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        self.c_epoch = epoch


class ClLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(ClLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        return self.loss(input=cl_logits, target=glabel) * self.lambda_


class SpgLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(SpgLoss, self).__init__(**kwargs)

        self.spg_threshold_1h = None
        self.spg_threshold_1l = None
        self.spg_threshold_2h = None
        self.spg_threshold_2l = None
        self.spg_threshold_3h = None
        self.spg_threshold_3l = None

        self.hyper_p_set = False

    @property
    def spg_thresholds(self):
        assert self.hyper_p_set

        h1 = self.spg_threshold_1h
        l1 = self.spg_threshold_1l

        h2 = self.spg_threshold_2h
        l2 = self.spg_threshold_2l

        h3 = self.spg_threshold_3h
        l3 = self.spg_threshold_3l

        return (h1, l1), (h2, l2), (h3, l3)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(SpgLoss, self).forward(epoch=epoch)

        assert self.hyper_p_set

        if not self.is_on():
            return self._zero

        return get_spg_loss(output_dict=model.logits_dict, target=glabel,
                            spg_thresholds=self.spg_thresholds)


class AcolLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(AcolLoss, self).__init__(**kwargs)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(AcolLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        return get_acol_loss(output_dict=model.logits_dict, gt_labels=glabel)


class CutMixLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(CutMixLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(CutMixLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        if cutmix_holder is None:
            return self.loss(input=cl_logits, target=glabel) * self.lambda_

        assert isinstance(cutmix_holder, list)
        assert len(cutmix_holder) == 3
        target_a, target_b, lam = cutmix_holder
        loss = (self.loss(cl_logits, target_a) * lam + self.loss(
            cl_logits, target_b) * (1. - lam))

        return loss


class MaxMinLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxMinLoss, self).__init__(**kwargs)

        self.dataset_name: str = ''
        assert isinstance(self.elb, ELB)
        self.lambda_size = 0.
        self.lambda_neg = 0.

        self._lambda_size_set = False
        self._lambda_neg_set = False

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)
        self.BCE = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

        self.softmax = nn.Softmax(dim=1)

    def set_lambda_neg(self, lambda_neg: float):
        assert isinstance(lambda_neg, float)
        assert lambda_neg >= 0.
        self.lambda_neg = lambda_neg

        self._lambda_neg_set = True

    def set_lambda_size(self, lambda_size: float):
        assert isinstance(lambda_size, float)
        assert lambda_size >= 0.
        self.lambda_size = lambda_size

        self._lambda_size_set = True

    def set_dataset_name(self, dataset_name: str):
        self._assert_dataset_name(dataset_name=dataset_name)
        self.dataset_name = dataset_name

    def _is_ready(self):
        assert self._lambda_size_set
        assert self._lambda_neg_set
        self._assert_dataset_name(dataset_name=self.dataset_name)

    def _assert_dataset_name(self, dataset_name: str):
        assert isinstance(dataset_name, str)
        assert dataset_name in [constants.GLAS, constants.CAMELYON512]

    def kl_uniform_loss(self, logits):
        assert logits.ndim == 2
        logsoftmax = torch.log2(self.softmax(logits))
        return (-logsoftmax).mean(dim=1).mean()

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(MaxMinLoss, self).forward(epoch=epoch)

        self._is_ready()

        if not self.is_on():
            return self._zero

        logits = model.logits_dict['logits']
        logits_pos = model.logits_dict['logits_pos']
        logits_neg = model.logits_dict['logits_neg']

        cam = model.logits_dict['cam']
        cam_logits = model.logits_dict['cam_logits']

        assert cam.ndim == 4
        assert cam.shape[1] == 1
        assert cam.shape == cam_logits.shape
        bs, _, _, _ = cam.shape

        cl_losss = self.loss(input=logits, target=glabel)
        total_l = cl_losss
        size = cam.contiguous().view(bs, -1).sum(dim=-1).view(-1, )

        if self.dataset_name == constants.GLAS:
            size_loss = self.elb(-size) + self.elb(-1. + size)
            total_l = total_l + self.lambda_size * size_loss * 0.0

            total_l = total_l + self.loss(input=logits_pos, target=glabel) * 0.
            total_l = total_l + self.lambda_neg * self.kl_uniform_loss(
                logits=logits_neg) * 0.0

        if self.dataset_name == constants.CAMELYON512:
            # pos
            ind_metas = (glabel == 1).nonzero().view(-1)
            if ind_metas.numel() > 0:
                tmps = size[ind_metas]
                size_loss = self.elb(-tmps) + self.elb(-1. + tmps)
                total_l = total_l + self.lambda_size * size_loss

            # neg
            ind_normal = (glabel == 0).nonzero().view(-1)
            if ind_normal.numel() > 0:
                trg_cams = torch.zeros(
                    (ind_normal.numel(), 1, cam.shape[2], cam.shape[3]),
                    dtype=torch.float, device=cam.device)

                total_l = total_l + self.BCE(input=cam_logits[ind_normal],
                                             target=trg_cams)

        return total_l


class SegLoss(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(SegLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(SegLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=seg_logits, target=masks) * self.lambda_


class ImgReconstruction(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction="none").to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(ImgReconstruction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean(
            dim=1).view(-1, ))
        return self.lambda_ * loss.mean()


class SelfLearningFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningFcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldFcams, self).__init__(**kwargs)

        self.loss = crf.DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(ConRanFieldFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyFcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(EntropyFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class MaxSizePositiveFcams(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveFcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class SelfLearningNegev(SelfLearningFcams):
    def __init__(self, **kwargs):
        super(SelfLearningNegev, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        if self.apply_negative_samples:
            return self.loss(input=fcams, target=seeds) * self.lambda_

        ind_non_neg = (glabel != self.negative_c).nonzero().view(-1)

        nbr = ind_non_neg.numel()

        if nbr == 0:
            return self._zero

        fcams_n_neg = fcams[ind_non_neg]
        seeds_n_neg = seeds[ind_non_neg]
        return self.loss(input=fcams_n_neg, target=seeds_n_neg) * self.lambda_


class ConRanFieldNegev(ConRanFieldFcams):
    pass


class MaxSizePositiveNegev(MaxSizePositiveFcams):
    def __init__(self, **kwargs):
        super(MaxSizePositiveNegev, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

        self.apply_negative_samples: bool = False
        self.negative_c: int = 0

        self._is_already_set = False

    def set_it(self, apply_negative_samples: bool, negative_c: int):
        assert isinstance(apply_negative_samples, bool)
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self.apply_negative_samples = apply_negative_samples

        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        fcams_n_input = fcams_n

        if not self.apply_negative_samples:
            ind_non_neg = (glabel != self.negative_c).nonzero().view(-1)

            nbr = ind_non_neg.numel()

            if nbr == 0:
                return self._zero

            fcams_n_input = fcams_n[ind_non_neg]

        n = fcams_n_input.shape[0]
        loss = None

        for c in [0, 1]:
            bl = fcams_n_input[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class JointConRanFieldNegev(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(JointConRanFieldNegev, self).__init__(**kwargs)

        self.loss = crf.ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor).to(self._device)

        self.pair_mode: str = ''
        self.n: int = 0

        self.dataset_name: str = ''

        self._already_set = False

    def _assert_dataset_name(self, dataset_name: str):
        assert dataset_name in [constants.GLAS, constants.CAMELYON512]

    def _assert_pair_mode(self, pair_mode: str):
        assert pair_mode in [constants.PAIR_SAME_C, constants.PAIR_MIXED_C,
                             constants.PAIR_DIFF_C]

    def _assert_n(self, n: int):
        assert isinstance(n, int)
        assert n > 0

    def set_it(self, pair_mode: str, n: int, dataset_name: str):
        self._assert_pair_mode(pair_mode)
        self._assert_n(n)
        self._assert_dataset_name(dataset_name)

        self.pair_mode = pair_mode
        self.n = n
        self.dataset_name = dataset_name
        self._already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(JointConRanFieldNegev, self).forward(epoch=epoch)
        assert self._already_set

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        raw_img_grey = rgb_to_grayscale(img=raw_img, num_output_channels=1)

        p_imgs, p_cams = self.pair_samples(imgs=raw_img_grey, glabel=glabel,
                                           prob_cams=fcams_n)

        return self.loss(images=raw_img, segmentations=fcams_n)

    def pair_samples(self,
                     imgs: torch.Tensor,
                     glabel: torch.Tensor,
                     prob_cams: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:

        assert imgs.ndim == 4
        assert prob_cams.ndim == 4

        b, c, h, w = imgs.shape
        out_img = None
        out_prob_cams = None
        all_idx = torch.arange(b)
        for i in range(b):
            _c = glabel[i]

            if self.pair_mode == constants.PAIR_SAME_C:
                idx = torch.nonzero(glabel == _c, as_tuple=False).squeeze()
            elif self.pair_mode == constants.PAIR_DIFF_C:
                idx = torch.nonzero(glabel != _c, as_tuple=False).squeeze()
            elif self.pair_mode == constants.PAIR_MIXED_C:
                idx = all_idx
            else:
                raise NotImplementedError

            idx = idx[idx != i]

            nbr = idx.numel()
            if (nbr == 0) or (nbr == 1):
                continue

            tmp_img = imgs[i].unsqueeze(0)
            tmp_prob_cams = prob_cams[i].unsqueeze(0)

            didx = torch.randperm(nbr)
            n_max = min(nbr, self.n)
            pool = cycle(list(range(n_max)))

            for _ in range(self.n):
                z = next(pool)
                # cat width.
                tmp_img = torch.cat(
                    (tmp_img, imgs[idx[didx[z]]].unsqueeze(0)), dim=3)
                tmp_prob_cams = torch.cat(
                    (tmp_prob_cams, prob_cams[idx[didx[z]]].unsqueeze(0)),
                    dim=3)

            if out_img is None:
                out_img = tmp_img
                out_prob_cams = tmp_prob_cams
            else:
                out_img = torch.cat((out_img, tmp_img), dim=0)
                out_prob_cams = torch.cat((out_prob_cams, tmp_prob_cams), dim=0)

        return out_img, out_prob_cams


class NegativeSamplesNegev(_ElementaryLoss):
    def __init__(self, **kwargs):
        super(NegativeSamplesNegev, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

        self.negative_c: int = 0
        self._is_already_set = False

    def set_it(self, negative_c: int):
        assert isinstance(negative_c, int)
        assert negative_c >= 0

        self.negative_c = negative_c
        self._is_already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None
                ):
        super(NegativeSamplesNegev, self).forward(epoch=epoch)
        assert self._is_already_set

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        ind_neg = (glabel == self.negative_c).nonzero().view(-1)
        nbr = ind_neg.numel()

        if nbr == 0:
            return self._zero

        b, c, h, w = fcams.shape

        trg = torch.zeros(
            (ind_neg.numel(), h, w), dtype=torch.long, device=fcams.device)
        logits = fcams[ind_neg]

        return self.loss(input=logits, target=trg) * self.lambda_


class MasterLoss(nn.Module):
    def __init__(self, cuda_id: int, name=None):
        super().__init__()
        self._name = name

        self.losses = []
        self.l_holder = []
        self.n_holder = [self.__name__]
        self._device = torch.device(cuda_id)

    def add(self, loss_: _ElementaryLoss):
        self.losses.append(loss_)
        self.n_holder.append(loss_.__name__)

    def update_t(self):
        for loss in self.losses:
            loss.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, **kwargs):
        assert self.losses != []

        self.l_holder = []
        for loss in self.losses:

            self.l_holder.append(loss(**kwargs).to(self._device))

        loss = sum(self.l_holder)
        self.l_holder = [loss] + self.l_holder
        return loss

    def to_device(self):
        for loss in self.losses:
            loss.to(self._device)

    def check_losses_status(self):
        print('-' * 60)
        print('Losses status:')

        for i, loss in enumerate(self.losses):
            if hasattr(loss, 'is_on'):
                print(self.n_holder[i+1], ': ... ',
                      loss.is_on(),
                      "({}, {})".format(loss.start_epoch, loss.end_epoch))
        print('-' * 60)

    def __str__(self):
        return "{}():".format(
            self.__class__.__name__, ", ".join(self.n_holder))


if __name__ == "__main__":
    from dlib.utils.reproducibility import set_seed
    set_seed(seed=0)
    b, c = 10, 4
    cudaid = 0
    torch.cuda.set_device(cudaid)

    loss = MasterLoss(cuda_id=cudaid)
    print(loss.__name__, loss, loss.l_holder, loss.n_holder)
    loss.add(SelfLearningFcams(cuda_id=cudaid))
    loss.add(SelfLearningNegev(cuda_id=cudaid))

    for l in loss.losses:
        print(l, isinstance(l, SelfLearningNegev))

    for e in loss.n_holder:
        print(e)

