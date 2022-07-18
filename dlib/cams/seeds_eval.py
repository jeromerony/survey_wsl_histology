from copy import deepcopy

import sys
from os.path import dirname, abspath

import torch
import torch.nn.functional as F
import torch.nn as nn

from kornia.morphology import dilation

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['AccSeeds',
           'AccSeedsmeter',
           'BasicAccSeedsMeter'
           ]


class AccSeeds(nn.Module):
    def __init__(self, multi_label_flag, device, max_ns: int = 2000,
                 ksz: int = 1):
        super(AccSeeds, self).__init__()

        assert not multi_label_flag
        self .multi_label_flag = multi_label_flag
        self.max_sn = max_ns
        step = 10
        self.n_seeds = list(range(10, max_ns + step, step))

        assert ksz >= 1
        self.ksz = ksz

        self._device = device

    @property
    def n(self):
        return len(self.n_seeds)

    def set_ksz(self, ksz: int):
        self.ksz = ksz

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        kernel = torch.ones(self.ksz, self.ksz).to(self._device)
        out = dilation(x, kernel) - 1.
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, cam, true_mask):
        assert cam.shape == true_mask.shape
        n, c, h, w = cam.shape
        assert n == 1
        assert c == 1

        val, idx = torch.sort(cam.view(h * w), dim=0, descending=False)
        forg = true_mask.view(h * w)
        backg = 1. - true_mask.view(h * w)

        holder = torch.zeros((h * w), dtype=torch.float, device=self._device,
                             requires_grad=False)
        acc_forg = torch.zeros(self.n, dtype=torch.float, device=self._device,
                               requires_grad=False)
        acc_backg = torch.zeros(self.n, dtype=torch.float, device=self._device,
                                requires_grad=False)

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        for i in range(self.n):
            z = self.n_seeds[i]
            holder = holder * 0.

            # todo:in multilabel: pick the min acc over all the existing labels?
            holder[idx[-z:]] = 1.
            holder = dilate(holder.view(1, 1, h, w)).contiguous().view(h * w)

            holder_bg = holder.clone() * 0.
            holder_bg[idx[0:z]] = 1.
            holder_bg = dilate(
                holder_bg.view(1, 1, h, w)).contiguous().view(h * w)

            outer = holder + holder_bg
            holder[outer == 2.] = 0.
            holder_bg[outer == 2.] = 0.

            acc_forg[i] = 100. * (holder * forg).sum() / holder.sum()
            acc_backg[i] = 100. * (holder_bg * backg).sum() / holder_bg.sum()

        return acc_forg, acc_backg, self.n_seeds.copy()


class BasicAccSeedsMeter:
    def __init__(self, n_seeds: list, nbr_samples: int, device):

        self.n_seeds = n_seeds
        self.c_mean: torch.Tensor = torch.zeros(len(n_seeds), device=device)
        self.min: torch.Tensor = torch.zeros(len(n_seeds), device=device) + 1.
        self.max: torch.Tensor = torch.zeros(len(n_seeds), device=device)

        self.nbr_samples = nbr_samples
        self.alpha = 2. / float(nbr_samples + 1)
        self.ema: torch.Tensor = torch.zeros(len(n_seeds), device=device)
        self.emvar: torch.Tensor = torch.zeros((len(n_seeds)), device=device)
        self.em_init = False

        self.c = 0.
        self._device = device

    @property
    def mean(self):
        return self.c_mean / float(self.nbr_samples)

    def __call__(self, x: torch.Tensor):
        if self.em_init:
            delta = x - self.ema
            self.ema = self.ema + self.alpha * delta
            self.emvar = (1. - self.alpha) * (self.emvar + self.alpha * delta
                                              * delta)
        else:
            self.ema = x
            self.em_init = True

        self.c_mean = self.c_mean + x
        self.min = torch.minimum(self.min, x)
        self.max = torch.maximum(self.max, x)


class AccSeedsmeter:
    def __init__(self, n_seeds: list, nbr_samples: int, device):

        self.n_seeds = n_seeds
        self.nbr_samples = nbr_samples

        self.fg_meter = None
        self.bg_meter = None

        self._device = device

        self.flush()

    def flush(self):
        self.fg_meter = BasicAccSeedsMeter(
            n_seeds=self.n_seeds, nbr_samples=self.nbr_samples,
            device=self._device)
        self.bg_meter = BasicAccSeedsMeter(
            n_seeds=self.n_seeds, nbr_samples=self.nbr_samples,
            device=self._device)

    def meters_cp(self):
        return deepcopy({"Foreground": self.fg_meter,
                         "Background": self.bg_meter})

    def __call__(self, x_fg, x_bg):
        self.fg_meter(x_fg)
        self.bg_meter(x_bg)


def test_AccSeeds():
    from dlib.utils.reproducibility import set_seed
    import matplotlib.pyplot as plt
    import datetime as dt

    set_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h, w = 1000, 1000
    cam = torch.rand((1, 1, h, w), dtype=torch.float, device=device)
    true_mask = torch.rand((1, 1, h, w), dtype=torch.float, device=device)
    true_mask = (true_mask > 0.01).float()

    for ksz in [1, 3, 5, 7, 9]:
        print('KSZ: {}'.format(ksz))

        ins = AccSeeds(multi_label_flag=False, device=device, ksz=ksz)
        t0 = dt.datetime.now()
        acc_forg, acc_backg, n_seeds = ins(cam, true_mask)
        print("time: {} for {} cases. ksz={}.".format(
            dt.datetime.now() - t0, len(n_seeds), ksz))

        fig = plt.figure()
        plt.plot(n_seeds, acc_forg.cpu().numpy(), label='acc_forg')
        plt.plot(n_seeds, acc_backg.cpu().numpy(), label='acc_back')
        plt.title('KSZ: {}'.format(ksz))
        plt.legend()
        fig.savefig('acc_seeds-{}.png'.format(ksz))
        plt.show()


if __name__ == "__main__":
    test_AccSeeds()



