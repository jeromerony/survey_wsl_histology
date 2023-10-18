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


class NormedHistogram(nn.Module):
    def __init__(self,
                 nbins: int = 256,
                 r_min: float = 0.,
                 r_max: float = 255.):
        super(NormedHistogram, self).__init__()

        assert isinstance(nbins, int), type(nbins)
        assert nbins > 0, nbins

        self.nbins = nbins

        assert isinstance(r_min, float), type(r_min)
        assert isinstance(r_max, float), type(r_max)
        assert r_min < r_max, f'{r_min}, {r_max}'
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w

        b, c, h, w = x.shape

        out = torch.zeros((b, c, self.nbins), dtype=torch.float32,
                          requires_grad=x.requires_grad)

        bins = self.nbins
        r = (self.r_min, self.r_max)
        w = (1. / (x[0, 0].numel())) * torch.zeros(x[0, 0].numel(),
                                                   requires_grad=False,
                                                   device=x.device)
        for i in range(b):
            for j in range(c):
                out[i, j] = torch.histogram(
                    x[i, j], bins=bins, range=r, weight=w, density=False)

        return out


def test_NormedHistogram():
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
    # image = image[:, :, 0]
    # image = np.expand_dims(image, axis=2)
    # image = image / 255.  # change scale.
    # -----------

    image = image.transpose(2, 0, 1)  # 3, h, w
    image = torch.tensor(image, dtype=torch.float32)  # 3, h, w
    assert image.shape[0] == ndim

    images = image.repeat(b, 1, 1, 1).to(device)

    hist = NormedHistogram(nbins=256,
                           r_min=0.,
                           r_max=255.)
    announce_msg("testing {}".format(hist))
    set_seed(seed=seed)

    l_imgs = [(img, 'Input')]

    t0 = dt.datetime.now()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()


    with autocast(enabled=False):
        t0 = dt.datetime.now()
        com_hist = hist(x=images)
        t1 = dt.datetime.now()
        print(images.shape, com_hist.shape)


    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    t1 = dt.datetime.now()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'time op: {elapsed_time_ms} (batchsize: {b}, h*w: {h}*{w})')
    print(f'time: {t1 - t0}')
    l_imgs.append((com_hist[0, 0].detach().cpu().numpy(), 'hist R.'))
    l_imgs.append((com_hist[0, 1].detach().cpu().numpy(), 'hist G.'))
    l_imgs.append((com_hist[0, 2].detach().cpu().numpy(), 'hist B.'))

    plot_limgs(_lims=l_imgs, title='Normalized hsitogram.')


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
    test_NormedHistogram()