import sys
from os.path import dirname, abspath
import os
import random
from typing import Tuple, Optional
import numbers
from collections.abc import Sequence
import math

import numpy as np
import torch
import torch.nn as nn


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['IterativeGaussianKDE', 'GaussianKDE']


class IterativeGaussianKDE(nn.Module):
    """
    Iterative KDE. Processes kernels by blocks (minibatch) to allow a memory
    friendly usage. otherwise, computations wont fit in GPU memory as in
    GaussianKDE() class.
    """
    def __init__(self,
                 device,
                 kde_bw: float,
                 nbin: int = 128,
                 max_color: int = 255,
                 ndim: int = 3,
                 blocksz: int = 64):
        """
        :param blocksz: int. split the kernels into blocks of size blocksz.
        run blocks sequentially. this determines the size of the required
        memory.
        """
        super(IterativeGaussianKDE, self).__init__()

        assert isinstance(kde_bw, float)
        assert kde_bw > 0

        assert isinstance(ndim, int)
        assert ndim > 0

        assert isinstance(nbin, int)
        assert isinstance(max_color, int)
        assert 0 < nbin <= max_color + 1

        assert isinstance(blocksz, int)
        assert blocksz > 0

        self.nbin = nbin
        self.max_color = max_color

        self.kde_bw = kde_bw
        self.ndim = ndim

        self.blocksz = blocksz

        self._device = device
        self.color_space = self._get_color_space()
        self.nbr_kernels = self.nbin**self.ndim
        assert self.color_space.shape == (self.nbr_kernels, self.ndim)

        self.const1 = torch.tensor(
            np.sqrt(kde_bw) * (2. * math.pi)**(-ndim / 2.),
            dtype=torch.float32, device=device, requires_grad=False)
        self.const2 = torch.tensor(
            2. * kde_bw, dtype=torch.float32, device=device,
            requires_grad=False)

    def _get_color_space(self) -> torch.Tensor:
        x = torch.linspace(start=0., end=self.max_color, steps=self.nbin,
                           device=self._device, dtype=torch.float32,
                           requires_grad=False)
        tensors = [x for _ in range(self.ndim)]
        if self.ndim > 1:
            return torch.cartesian_prod(*tensors)  # nbin**ndim, ndim
        elif self.ndim == 1:
            return torch.cartesian_prod(*tensors).view(-1, 1)  # nbin**ndim,
            # ndim=1
        else:
            raise ValueError

    def _get_px_one_img(self,
                        img: torch.Tensor,
                        mask_fg: torch.Tensor,
                        mask_bg: torch.Tensor) -> Tuple[torch.Tensor,
                                                        torch.Tensor]:
        assert img.ndim == 3
        assert mask_fg.ndim == 2
        assert mask_fg.ndim == 2
        assert mask_fg.shape == mask_bg.shape
        assert mask_fg.shape == img.shape[1:]
        assert img.shape[0] == self.ndim

        dim, h, w = img.shape
        x: torch.Tensor = img.contiguous().view(h * w, dim)
        assert x.shape[-1] == self.color_space.shape[-1]

        ki = (x.unsqueeze(1) - self.color_space)**2  # h*w, nbin**ndim, ndim
        ki = ki.sum(dim=-1) / self.const2  # h*w, nbin**ndim
        ki = self.const1 * torch.exp(-ki)  # h*w, nbin**ndim

        # fg
        roi_fg: torch.Tensor = mask_fg.contiguous().view(-1, 1)  # h*w, 1
        assert roi_fg.shape[0] == x.shape[0]
        p_fg = (roi_fg * ki).sum(dim=0)  # 1, nbin**ndim
        if roi_fg.sum() != 0.:
            p_fg = p_fg / roi_fg.sum()  # 1, nbin**ndim

        # bg
        roi_bg: torch.Tensor = mask_bg.contiguous().view(-1, 1)  # h*w, 1
        assert roi_bg.shape[0] == x.shape[0]
        p_bg = (roi_bg * ki).sum(dim=0)  # 1, nbin**ndim
        if roi_bg.sum() != 0.:
            p_bg = p_bg / roi_bg.sum()  # 1, nbin**ndim

        return p_fg, p_bg

    def _eval_partial_pdf(self, subkernels: torch.Tensor,
                          x: torch.Tensor) -> torch.Tensor:

        assert subkernels.ndim == x.ndim == 2
        assert subkernels.shape[1] == x.shape[1] == self.ndim


        ki = (x.unsqueeze(1) - subkernels) ** 2  # ?x, ?k, ndim
        ki = ki.sum(dim=-1) / self.const2  # ?x, ?k
        ki = self.const1 * torch.exp(-ki)  # ?x, ?k
        ki = ki.mean(dim=0)  # 1, ?k

        return ki

    def _loop_one_img(self,
                      img: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        assert img.ndim == 3
        assert mask.ndim == 2
        assert mask.shape == img.shape[1:]
        assert img.shape[0] == self.ndim

        dim, h, w = img.shape
        assert dim == self.color_space.shape[-1] == self.ndim

        p = torch.zeros((1, self.nbin ** self.ndim), dtype=torch.float32,
                        device=self._device, requires_grad=True) * 0.0

        x_ = img[mask.repeat(dim, 1, 1) != 0]
        if x_.numel() > 0:
            x_ = x_.view(-1, dim)  # ?x, ndim
            p = self._pdf(x=x_, pdf=p)  # 1, nbin**ndim

        return p

    def _pdf(self, x: torch.Tensor, pdf: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert x.shape[1] == self.ndim

        assert pdf.ndim == 2
        assert pdf.shape == (1, self.nbin**self.ndim)

        for ker_i in range(0, self.nbr_kernels, self.blocksz):
            right = min(ker_i + self.blocksz, self.nbr_kernels)
            subkernel = self.color_space[ker_i: right]  # ?k, ndim

            pdf[0, ker_i: right] = self._eval_partial_pdf(
                subkernels=subkernel, x=x)

        return pdf

    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:

        assert images.ndim == 4
        assert masks.ndim == 4

        assert masks.shape[1] == 1
        assert masks.shape[2:] == images.shape[2:]
        assert images.shape[1] == self.ndim
        assert masks.shape[0] == images.shape[0]

        b, c, h, w = images.shape

        p = None
        for i in range(b):
            tmp = self._loop_one_img(
                img=images[i],
                mask=masks[i].squeeze(0)
            )  # 1, nbin**ndim

            if p is None:
                p = tmp
            else:
                p = torch.vstack((p, tmp))

        assert p.shape == (b, self.nbr_kernels)
        return p  # b, nbin**ndim

    def extra_repr(self) -> str:
        return f'kde_bw: {self.kde_bw}, nbin: {self.nbin}, ' \
               f'blocksz: {self.blocksz}, max_color:{self.max_color}, ' \
               f'ndim: {self.ndim}'


class GaussianKDE(nn.Module):
    """
    KDE for RGB images. Computes a KDE per plan.
    """
    def __init__(self,
                 device,
                 kde_bw: float,
                 nbin: int = 128,
                 max_color: int = 255,
                 ndim: int = 3,
                 blocksz: int = 64
                 ):
        super(GaussianKDE, self).__init__()

        assert isinstance(kde_bw, float)
        assert kde_bw > 0

        assert isinstance(ndim, int)
        assert ndim > 0

        assert isinstance(nbin, int)
        # assert isinstance(max_color, int)
        # assert 0 < nbin <= max_color + 1

        self.nbin = nbin
        self.max_color = max_color

        self.kde_bw = kde_bw
        self.ndim = ndim

        self._device = device
        self.color_space = self._get_color_space()

        self.const1 = torch.tensor((2. * math.pi * kde_bw)**(-1 / 2.),
                                   dtype=torch.float32, device=device,
                                   requires_grad=False)
        self.const2 = torch.tensor(2. * kde_bw, dtype=torch.float32,
                                   device=device, requires_grad=False)

    def _get_color_space(self) -> torch.Tensor:
        x = torch.linspace(start=0., end=self.max_color, steps=self.nbin,
                           device=self._device, dtype=torch.float32,
                           requires_grad=False)
        print(x, x.shape)
        return x.view(-1, 1)  # nbin, 1

    def _get_px_one_img(self,
                        img: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        assert img.ndim == 3
        assert mask.ndim == 2
        assert mask.shape == img.shape[1:]
        assert img.shape[0] == self.ndim

        dim, h, w = img.shape
        x: torch.Tensor = img.contiguous().view(dim, 1, h * w)

        ki = (x - self.color_space)**2  # ndim, nbin, h*w
        ki = ki / self.const2  # ndim, nbin, h*w
        ki = self.const1 * torch.exp(-ki)  # ndim, nbin, h*w

        roi: torch.Tensor = mask.contiguous().view(1, 1, -1)  # 1, 1, h*w
        assert roi.shape[-1] == x.shape[-1]
        p = (roi * ki)  # ndim, nbin, h*w
        p = p.sum(dim=-1)  # ndim, nbin

        if roi.sum() != 0.:
            p = p / roi.sum()  # ndim, nbin

        print(p.shape, (self.ndim, self.nbin))
        assert p.shape == (self.ndim, self.nbin)

        return p

    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:

        assert images.ndim == 4
        assert masks.ndim == 4

        assert masks.shape[1] == 1
        assert masks.shape[2:] == images.shape[2:]
        assert images.shape[1] == self.ndim

        b, c, h, w = images.shape

        _p_ = None
        for i in range(b):
            p_ = self._get_px_one_img(
                img=images[i],
                mask=masks[i].squeeze(0)
            )

            if _p_ is None:
                _p_ = p_.unsqueeze(0)
            else:
                _p_ = torch.vstack((_p_, p_.unsqueeze(0)))

        assert _p_.shape == (b, self.ndim, self.nbin)

        return _p_

    def extra_repr(self) -> str:
        return f'kde_bw: {self.kde_bw}, nbin: {self.nbin}, ' \
               f'max_color:{self.max_color}, ndim: {self.ndim}'


def test_IterativeGaussianKDE():
    def plot(fg, bg, outf, tag):
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                                 sharey=False, squeeze=False)
        axes[0, 0].plot(fg, label='fg', color="tab:orange",
                        linestyle='-', linewidth=3, alpha=0.3)
        axes[0, 0].plot(bg, label='bg', color="tab:blue",
                        linestyle='-', linewidth=1, alpha=.3)
        axes[0, 0].set_title(tag)
        axes[0, 0].grid(True)
        plt.legend(loc='best')
        plt.show()
        fig.savefig(outf, bbox_inches='tight', dpi=300)
        plt.close(fig)


    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        wim, him = _lims[0][0].size
        r = him / float(wim)
        fw = 20
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (obj, tag) in enumerate(_lims):
            if isinstance(obj, Image.Image):
                axes[0, i].imshow(obj)
                axes[0, i].text(3, 40, tag,
                                bbox={'facecolor': 'white', 'pad': 1,
                                      'alpha': 0.8})
            elif isinstance(obj, np.ndarray):
                if obj.shape[0] == 3:  # dist.
                    h = {0: 'R', 1: 'G', 2: 'B'}
                    for j in range(obj.shape[0]):
                        axes[0, i].plot(obj[j], label=h[j])
                    axes[0, i].set_title(tag)
                else:  # mask
                    axes[0, i].matshow(obj.astype(np.uint8), cmap='gray')
                    axes[0, i].set_title(tag)

        plt.suptitle(title)
        plt.show()

    seed = 0
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    cuda = "0"
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    kde_bw = 50.
    nbin = 128
    max_color = 255
    ndim = 2
    downscale = 1.
    h = int(224 / downscale)
    w = int(224 / downscale)
    b = 32

    d = 20

    outfd = join(root_dir, 'data/debug/kde')
    os.makedirs(outfd, exist_ok=True)
    path_imng = join(root_dir, 'data/debug/input',
                     'Black_Footed_Albatross_0002_55.jpg')
    img = Image.open(path_imng, 'r').convert('RGB')
    img = Image.open(path_imng, 'r').convert('RGB')
    w_, h_ = img.size
    rw = w_ / float(w)
    rh = h_ / float(h)
    img = img.resize((w, h), resample=Image.BICUBIC)
    image_raw = np.array(img, dtype=np.float32)  # h, w, 3
    totensor = transforms.ToTensor()
    image_normalized = totensor(img) * 255  # 3, h, w
    image_normalized = image_normalized[1:, :, :]
    if image_normalized.ndim == 2:
        image_normalized = image_normalized.unsqueeze(0)
    assert image_normalized.shape[0] == ndim


    images = image_normalized.repeat(b, 1, 1, 1).to(device)
    mask_fg = torch.zeros((h, w), dtype=torch.float32, device=device,
                          requires_grad=True) * 0.
    # mask_fg[int(h/2.) - d: int(h/2.) + d, int(w/2.) - d: int(w/2.) + d] = 1.
    x0, y0 = int(112 / rh), int(14 / rw)
    x1, y1 = int(298 / rh), int(402 / rw)
    print(x0, y0, y1, y1)

    mask_fg[x0: x1, y0: y1] = 1.

    # mask_bg = mask_fg * 0.
    mask_bg = 1. - mask_fg

    masks_fg = mask_fg.repeat(b, 1, 1, 1)
    masks_bg = mask_bg.repeat(b, 1, 1, 1)
    stats = []

    bls = [5, 6, 7, 8, 9, 10, 11, 12]
    for t in [12]:
        blocksz = 2**t
        kde = IterativeGaussianKDE(device=device,
                                   kde_bw=kde_bw,
                                   nbin=nbin,
                                   max_color=max_color,
                                   ndim=ndim,
                                   blocksz=blocksz
                                   )
        announce_msg("testing {}".format(kde))
        set_seed(seed=seed)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with autocast(enabled=True):
            p_fg = kde(images=images, masks=masks_fg)
            p_bg = kde(images=images, masks=masks_bg)

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f'time op (blocksz: {blocksz}): {elapsed_time_ms} '
              f'(batchsize: {b}, h*w: {h}*{w})')
        stats.append((blocksz, elapsed_time_ms))
        tag = f'bw: {kde_bw}, nbin: {nbin}, hxw: {h}x{w}, ' \
              f'downscale: {downscale}'
        plot(fg=p_fg[0].squeeze().detach().cpu().numpy(),
             bg=p_bg[0].squeeze().detach().cpu().numpy(),
             outf=join(outfd,
                       f'bw-{kde_bw}-nbin-{nbin}-h-{h}-w-{w}-ds-{downscale}.png'
                       ),
             tag=tag)
        plot_limgs([(img, 'image'),
                    (masks_fg[0].squeeze().detach().cpu().numpy() * 255,
                     'fg.')],
                   'Image/mask fg')

    for bs, t in stats:
        print(f'batchsize: {b} blocksz {bs} time {t} hxw: {h}x{w}')


def test_GaussianKDE():
    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        wim, him = _lims[0][0].size
        r = him / float(wim)
        fw = 20
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (obj, tag) in enumerate(_lims):
            if isinstance(obj, Image.Image):
                axes[0, i].imshow(obj)
                axes[0, i].text(3, 40, tag,
                                bbox={'facecolor': 'white', 'pad': 1,
                                      'alpha': 0.8})
            elif isinstance(obj, np.ndarray):
                if obj.shape[0] in [3, 1]:  # dist.
                    h = {0: 'R', 1: 'G', 2: 'B'}
                    for j in range(obj.shape[0]):
                        axes[0, i].plot(obj[j], label=h[j])
                    axes[0, i].set_title(tag)
                else:  # mask
                    axes[0, i].matshow(obj.astype(np.uint8), cmap='gray')
                    axes[0, i].set_title(tag)

        plt.suptitle(title)
        plt.show()

    seed = 0
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    cuda = "0"
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    kde_bw = 10./ (255**2)  # adjust variance to changed scale.
    nbin = 256
    max_color = 1.
    ndim = 1
    h = int(64 / 1)
    w = int(64 / 1)
    b = 32

    d = 20

    path_imng = join(root_dir, 'data/debug/input',
                     'Black_Footed_Albatross_0002_55.jpg')
    img = Image.open(path_imng, 'r').convert('RGB').resize(
        (w, h), resample=Image.BICUBIC)
    image = np.array(img, dtype=np.float32)  # h, w, 3

    # grey.------
    image = image[:, :, 0]
    image = np.expand_dims(image, axis=2)
    image = image / 255.  # change scale.
    # -----------

    image = image.transpose(2, 0, 1)  # 3, h, w
    image = torch.tensor(image, dtype=torch.float32)  # 3, h, w
    assert image.shape[0] == ndim

    images = image.repeat(b, 1, 1, 1).to(device)
    mask_fg = torch.zeros((h, w), dtype=torch.float32, device=device,
                          requires_grad=True) * 0.
    mask_fg[int(h/2.) - d: int(h/2.) + d, int(w/2.) - d: int(w/2.) + d] = 1.
    mask_bg = 1. - mask_fg

    masks_fg = mask_fg.repeat(b, 1, 1, 1)
    masks_bg = mask_bg.repeat(b, 1, 1, 1)

    kde = GaussianKDE(device=device,
                      kde_bw=kde_bw,
                      nbin=nbin,
                      max_color=max_color,
                      ndim=ndim)
    announce_msg("testing {}".format(kde))
    set_seed(seed=seed)

    l_imgs = [(img, 'Input')]

    t0 = dt.datetime.now()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()


    with autocast(enabled=False):
        t0 = dt.datetime.now()
        p_fg = kde(images=images, masks=masks_fg)
        p_bg = kde(images=images, masks=masks_bg)
        t1 = dt.datetime.now()
        print(p_fg.shape)


    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    t1 = dt.datetime.now()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'time op: {elapsed_time_ms} (batchsize: {b}, h*w: {h}*{w})')
    print(f'time: {t1 - t0}')
    l_imgs.append((p_fg[0].detach().cpu().numpy(), 'fg dist.'))
    l_imgs.append((p_bg[0].detach().cpu().numpy(), 'bg dist.'))
    l_imgs.append((mask_fg.cpu().detach().numpy() * 255, 'fg. mask'))

    plot_limgs(_lims=l_imgs, title='Distribution.')


if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    from dlib.functional import _functional as dlibf

    from os.path import join
    import datetime as dt

    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from torch.cuda.amp import autocast

    torch.backends.cudnn.benchmark = True
    # for i in range(1):
    #     print(f'run {i}')
    #     test_IterativeGaussianKDE()
    test_GaussianKDE()
