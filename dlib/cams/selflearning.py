import operator
import sys
import os
from os.path import dirname, abspath
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage.util.dtype import dtype_range

from kornia.morphology import dilation
from kornia.morphology import erosion
from skimage.filters import threshold_otsu
from skimage import filters

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

_NBINS = 256

_FG = 'foreground'
_BG = 'background'

_TRGS = [_FG, _BG]

__all__ = ['GetFastSeederSLFCAMS',
           'MBSeederSLFCAMS',
           'MBProbSeederSLFCAMS',
           'MBProbNegAreaSeederSLFCAMS',
           'MBSeederSLNEGEV',
           'MBProbSeederSLNEGEV',
           'MBProbNegAreaSeederSLNEGEV']


def rv1d(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 1
    return torch.flip(t, dims=(0, ))


class _STOtsu(nn.Module):
    def __init__(self):
        super(_STOtsu, self).__init__()

        self.bad_egg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bad_egg = False

        min_x = x.min()
        max_x = x.max()

        if min_x == max_x:
            self.bad_egg = True
            return torch.tensor(min_x)

        bins = int(max_x - min_x + 1)
        bin_centers = torch.arange(min_x, max_x + 1, 1, dtype=torch.float32,
                                   device=x.device)

        hist = torch.histc(x, bins=bins)
        weight1 = torch.cumsum(hist, dim=0)
        _weight2 = torch.cumsum(rv1d(hist), dim=0)
        weight2_r = _weight2
        weight2 = rv1d(_weight2)
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1
        mean2 = rv1d(torch.cumsum(rv1d(hist * bin_centers), dim=0) / weight2_r)
        diff_avg_sq = torch.pow(mean1[:-1] - mean2[1:], 2)
        variance12 = weight1[:-1] * weight2[1:] * diff_avg_sq

        idx = torch.argmax(variance12)
        threshold = bin_centers[:-1][idx]

        return threshold


class _STFG(nn.Module):
    def __init__(self, max_):
        super(_STFG, self).__init__()
        self.max_ = max_

    def forward(self, roi: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        # roi: h,w
        idx_fg = torch.nonzero(roi, as_tuple=True)  # (idx, idy)
        n_fg = idx_fg[0].numel()
        if (n_fg > 0) and (self.max_ > 0):
            probs = torch.ones(n_fg, dtype=torch.float)
            selected = probs.multinomial(
                num_samples=min(self.max_, n_fg), replacement=False)
            fg[idx_fg[0][selected], idx_fg[1][selected]] = 1

        return fg


class _STBG(nn.Module):
    def __init__(self, nbr_bg, min_):
        super(_STBG, self).__init__()

        self.nbr_bg = nbr_bg
        self.min_ = min_

    def forward(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2
        # cam: h, w
        h, w = cam.shape
        val, idx_bg_ = torch.sort(cam.view(h * w), dim=0, descending=False)

        tmp = torch.zeros_like(bg)
        if self.nbr_bg > 0:
            tmp = tmp.view(h * w)
            tmp[idx_bg_[:self.nbr_bg]] = 1
            tmp = tmp.view(h, w)

            idx_bg = torch.nonzero(tmp, as_tuple=True)  #
            # (idx, idy)
            n_bg = idx_bg[0].numel()
            if (n_bg > 0) and (self.min_ > 0):
                probs = torch.ones(n_bg, dtype=torch.float)
                selected = probs.multinomial(
                    num_samples=min(self.min_, n_bg),
                    replacement=False)
                bg[idx_bg[0][selected], idx_bg[1][selected]] = 1

        return bg


class _STOneSample(nn.Module):
    def __init__(self, min_, max_, nbr_bg):
        super(_STOneSample, self).__init__()

        self.min_ = min_
        self.max_ = max_

        self.otsu = _STOtsu()
        self.fg_capture = _STFG(max_=max_)
        self.bg_capture = _STBG(nbr_bg=nbr_bg, min_=min_)

    def forward(self, cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        # otsu
        cam_ = torch.floor(cam * 255)
        th = self.otsu(x=cam_)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        if self.otsu.bad_egg:
            return fg, bg

        # ROI
        roi = (cam_ > th).long()
        roi = erode(roi.unsqueeze(0).unsqueeze(0)).squeeze()

        fg = self.fg_capture(roi=roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class MBSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(MBSeederSLFCAMS, self).__init__()

        assert not multi_label_flag

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def mb_erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = x

        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        nbr_bg = int(self.min_p * h * w)

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _STOneSample(min_=self.min_, max_=self.max_, nbr_bg=nbr_bg)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode)

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.support_background,
                self.multi_label_flag, self.ignore_idx)


class _ProbaSampler(nn.Module):
    def __init__(self, nbr: int, trg: str):
        super(_ProbaSampler, self).__init__()
        assert trg in _TRGS

        assert nbr >= 0

        self.trg = trg
        self.nbr = nbr

        self.eps = 1e-6

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2

        _cam: torch.Tensor = cam + self.eps
        _cam = _cam / _cam.sum()

        if self.trg == _BG:
            _cam = 1. - _cam

        # cam: h, w
        h, w = _cam.shape
        probs = _cam.contiguous().view(h * w)

        tmp = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                          requires_grad=False).contiguous().view(h * w)

        if self.nbr == 0:
            return tmp.contiguous().view(h, w)

        selected = probs.multinomial(num_samples=min(self.nbr, h * w),
                                     replacement=False)
        tmp[selected] = 1
        return tmp.contiguous().view(h, w)


class _ProbaAreaSampler(nn.Module):
    def __init__(self, nbr: int, p: float, trg: str):
        super(_ProbaAreaSampler, self).__init__()
        assert trg in _TRGS

        assert nbr >= 0

        self.trg = trg
        self.nbr = nbr
        self.p = p

        self.eps = 1e-6

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2

        _cam: torch.Tensor = cam + self.eps
        _cam = _cam / _cam.sum()

        if self.trg == _BG:
            _cam = 1. - _cam

        # cam: h, w
        h, w = _cam.shape
        probs = _cam.contiguous().view(h * w)

        tmp = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                          requires_grad=False)

        val, idx_ = torch.sort(probs, dim=0, descending=True)
        nbr_roi = int(self.p * h * w)

        if (self.nbr == 0) or (nbr_roi == 0):
            return tmp

        marker = torch.zeros_like(tmp).view(h * w)
        marker[idx_[:nbr_roi]] = 1
        marker = marker.view(h, w)
        idx_area = torch.nonzero(marker, as_tuple=True)  # (idx, idy)

        selected = val[:nbr_roi].multinomial(num_samples=min(self.nbr, nbr_roi),
                                             replacement=False)
        tmp[idx_area[0][selected], idx_area[1][selected]] = 1
        return tmp


class _ProbaOneSample(nn.Module):
    def __init__(self, min_: int, max_: int):
        super(_ProbaOneSample, self).__init__()

        self.min_ = min_
        self.max_ = max_

        self.fg_capture = _ProbaSampler(nbr=max_, trg=_FG)
        self.bg_capture = _ProbaSampler(nbr=min_, trg=_BG)

    def forward(self, cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w

        fg = self.fg_capture(cam=cam)
        bg = self.bg_capture(cam=cam)
        return fg, bg


class _ProbaAreaOneSample(nn.Module):
    def __init__(self, min_: int, max_: int, min_p: float):
        super(_ProbaAreaOneSample, self).__init__()

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        if min_ > 0:
            assert min_p > 0.

        self.min_ = min_
        self.max_ = max_
        self.min_p = min_p

        self.fg_capture = _ProbaSampler(nbr=max_, trg=_FG)
        self.bg_capture = _ProbaAreaSampler(nbr=min_, p=min_p, trg=_BG)

    def forward(self, cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w

        fg = self.fg_capture(cam=cam)
        bg = self.bg_capture(cam=cam)
        return fg, bg


class MBProbSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int = 1,
                 max_: int = 1,
                 ksz: int = 3,
                 seg_ignore_idx: int = -255
                 ):
        """
        Sample seeds from CAM.
        Does the same job as 'MBSeederSLFCAMS' but, it does not require to
        define foreground and background sampling regions.
        We use a probabilistic approach to sample foreground and background.

        Supports batch.

        :param min_: int. number of pixels to sample from background.
        :param max_: int. number of pixels to sample from foreground.
        :param ksz: int. kernel size to dilate the seeds.
        :param seg_ignore_idx: int. index for unknown seeds. 0: background.
        1: foreground. seg_ignore_idx: unknown.
        """
        super(MBProbSeederSLFCAMS, self).__init__()

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        self.ignore_idx = seg_ignore_idx

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor. average cam: batchs, 1, h, w. Detach it first.
        :return: pseudo labels. batchsize, h, w
        """
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _ProbaOneSample(min_=self.min_, max_=self.max_)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze())

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ksz={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.ignore_idx)


class MBProbNegAreaSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int = 1,
                 max_: int = 1,
                 min_p: float = .2,
                 ksz: int = 3,
                 seg_ignore_idx: int = -255
                 ):
        """
        Sample seeds from CAM.
        Does the same job as 'MBProbSeederSLFCAMS' but, instead of sampling
        negative pixels using the entire inverse of the cam, we limit it to a
        size region min_p. when min_p = 1., it is identical to
        'MBProbSeederSLFCAMS'.
        We use a probabilistic approach to sample foreground and background.

        Supports batch.

        :param min_: int. number of pixels to sample from background.
        :param max_: int. number of pixels to sample from foreground.
        :param min_p: float. area to be considered as background.
        :param ksz: int. kernel size to dilate the seeds.
        :param seg_ignore_idx: int. index for unknown seeds. 0: background.
        1: foreground. seg_ignore_idx: unknown.
        """
        super(MBProbNegAreaSeederSLFCAMS, self).__init__()

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        if min_ > 0:
            assert min_p > 0.

        self.min_p = min_p

        self.ignore_idx = seg_ignore_idx

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor. average cam: batchs, 1, h, w. Detach it first.
        :return: pseudo labels. batchsize, h, w
        """
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _ProbaAreaOneSample(min_=self.min_, max_=self.max_,
                                  min_p=self.min_p)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze())

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, min_p={} ksz={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.min_p, self.ksz, self.ignore_idx)


class GetFastSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(GetFastSeederSLFCAMS, self).__init__()
        assert not multi_label_flag

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = x
        tmp = x
        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)
            if out.sum() == 0:
                out = tmp
                break
            else:
                tmp = out

        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)
        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.
        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device)

        nbr_bg = int(self.min_p * h * w)

        for i in range(b):
            cam = x[i].squeeze()  # h, w
            t0 = time.perf_counter()
            # cant send to cpu. too expensive.*******************************
            cam_img = (cam.cpu().detach().numpy() * 255).astype(np.uint8)
            # print('time to cpu {}'.format(time.perf_counter() - t0))
            _bad_egg = False

            fg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            bg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            if cam_img.min() == cam_img.max():
                _bad_egg = True

            if not _bad_egg:
                import datetime as dt
                t0 = dt.datetime.now()
                # convert to gpu + batch. ******************************
                otsu_thresh = threshold_otsu(cam_img)
                # print('otsu {}'.format(dt.datetime.now() - t0))

                if otsu_thresh == 0:
                    otsu_thresh = 1
                if otsu_thresh == 255:
                    otsu_thresh = 254

                # GPU + BATCH *************************************************
                ROI = torch.from_numpy(cam_img > otsu_thresh).to(self._device)
                # GPU + BATCH *************************************************
                ROI = erode(ROI.unsqueeze(0).unsqueeze(0) * 1).squeeze()

                # fg
                idx_fg = torch.nonzero(ROI, as_tuple=True)  # (idx, idy)
                n_fg = idx_fg[0].numel()
                if n_fg > 0:
                    if self.max_ > 0:
                        probs = torch.ones(n_fg, dtype=torch.float)
                        selected = probs.multinomial(
                            num_samples=min(self.max_, n_fg), replacement=False)
                        fg[idx_fg[0][selected], idx_fg[1][selected]] = 1
                        # xxxxxxxxxxxxxxxxxxxx
                        fg = dilate(fg.view(1, 1, h, w)).squeeze()

                # bg
                val, idx_bg_ = torch.sort(cam.view(h * w), dim=0,
                                          descending=False)
                tmp = bg * 1.
                if nbr_bg > 0:
                    tmp = tmp.view(h * w)
                    tmp[idx_bg_[:nbr_bg]] = 1
                    tmp = tmp.view(h, w)

                    idx_bg = torch.nonzero(tmp, as_tuple=True)  #
                    # (idx, idy)
                    n_bg = idx_bg[0].numel()
                    if n_bg >= 0:
                        if self.min_ > 0:
                            probs = torch.ones(n_bg, dtype=torch.float)
                            selected = probs.multinomial(
                                num_samples=min(self.min_, n_bg),
                                replacement=False)
                            bg[idx_bg[0][selected], idx_bg[1][selected]] = 1
                            # xxxxxxxxxxxxxxxx
                            bg = dilate(bg.view(1, 1, h, w)).squeeze()

            # all this is gpu batchable.
            # sanity
            outer = fg + bg
            fg[outer == 2] = 0
            bg[outer == 2] = 0

            seeds = torch.zeros((h, w), dtype=torch.long, device=self._device,
                                requires_grad=False) + self.ignore_idx

            seeds[fg == 1] = 1
            seeds[bg == 1] = 0

            out[i] = seeds.detach().clone()

        assert out.dtype == torch.long
        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.support_background,
                self.multi_label_flag, self.ignore_idx)


class MBSeederSLNEGEV(MBSeederSLFCAMS):
    pass


class MBProbSeederSLNEGEV(MBProbSeederSLFCAMS):
    pass


class MBProbNegAreaSeederSLNEGEV(MBProbNegAreaSeederSLFCAMS):
    pass


def test_Linear_vs_Conc_SeederSLFCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 10
    min_p = .2
    fg_erode_k = 11
    fg_erode_iter = 1

    batchs = 64

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_lin = [(cam_2Img(cam), 'CAM')]
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_linear = GetFastSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module_linear))

        # cProfile.runctx('module_linear(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_linear(cam)
        print('time LINEAR: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_linear.ignore_idx] = 0

        if batchs == 1:
            limgs_lin.append((out.squeeze().cpu().numpy().astype(np.uint8),
                              'pseudo_ksz_{}_linear'.format(ksz)))

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_lin, 'LINEAR')
        plot_limgs(limgs_conc, 'CONCURRENT')


def test_MBProbSeederSLFCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 10

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBProbSeederSLFCAMS(
            min_=min_,
            max_=max_,
            ksz=ksz,
            seg_ignore_idx=-255)
        print('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'CONCURRENT-PROB')


def test_MBProbNegAreaSeederSLFCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 1
    min_p = .3

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBProbNegAreaSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            ksz=ksz,
            seg_ignore_idx=-255)
        print('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'CONCURRENT-PROB')


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd)


def test_stotsu_vs_skiamgeotsu():
    from os.path import join
    import time
    import cProfile
    from dlib.utils.reproducibility import set_seed
    import matplotlib.pyplot as plt
    from torch.cuda.amp import autocast
    from tqdm import tqdm
    from torch.profiler import profile, record_function, ProfilerActivity

    amp = False
    print('amp: {}'.format(amp))
    fdout = join(root_dir, 'data/debug/otsu')
    mkdir(fdout)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=False, squeeze=False)

    cuda = 1
    torch.cuda.set_device(cuda)

    times = []
    ths = []

    def atom(seed):
        set_seed(seed)
        h, w = 224, 224
        img = np.random.rand(h, w) * 100 + np.random.rand(h, w) * 10

        img = img.astype(np.uint8)

        img_torch = torch.from_numpy(img).float().cuda()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        st_otsu = _STOtsu().cuda()

        st_th = st_otsu(img_torch)
        start_event.record()
        with autocast(enabled=amp):
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                with record_function("compute_th"):
                    st_otsu(img_torch)
            # cProfile.runctx('st_otsu(img_torch)', globals(), locals())
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms_ours = start_event.elapsed_time(end_event)
        print('')
        print(prof.key_averages().table(sort_by="cuda_time_total",
                                        row_limit=10))

        start_event.record()
        kimh_th = threshold_otsu(img)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms_skim = start_event.elapsed_time(end_event)

        return (st_th, kimh_th), (elapsed_time_ms_ours, elapsed_time_ms_skim)

    n = 2
    for seed_ in tqdm(range(n), ncols=150, total=n):
        th, t = atom(seed_)
        times.append(t)
        ths.append(th)

    axes[0, 0].plot([z[0].item() for z in ths], color='tab:blue',
                    label='Our threshold')
    axes[0, 0].plot([z[1] for z in ths], color='tab:orange',
                    label='SKimage threshold')
    axes[0, 0].set_title('Thresholds')

    axes[0, 1].plot([z[0] for z in times], color='tab:blue',
                    label='Our time')
    axes[0, 2].plot([z[1] for z in times], color='tab:orange',
                    label='SKimage time')
    axes[0, 1].set_title('Time (ms) [AMP: {}]'.format(amp))
    axes[0, 2].set_title('Time (ms')

    fig.suptitle('Otsu: ours vs. Skimage', fontsize=6)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

    fig.savefig(join(fdout, 'otsu-compare-amp-{}'.format(amp)),
                bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    import datetime as dt

    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    # test_Linear_vs_Conc_SeederSLFCAMS()

    # set_seed(0)
    # test_stotsu_vs_skiamgeotsu()

    # set_seed(0)
    # test_MBProbSeederSLFCAMS()

    set_seed(0)
    test_MBProbNegAreaSeederSLFCAMS()
