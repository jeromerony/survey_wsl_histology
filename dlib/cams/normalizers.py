import sys
from os.path import dirname, abspath

import torch
import torch.nn.functional as F
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['CamStandardizer']


class CamStandardizer(nn.Module):
    def __init__(self, w=5., a=-1., b=1.):
        super(CamStandardizer, self).__init__()

        assert isinstance(w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        self.w = w
        self.a = a
        self.b = b

        self.eps = 1e-8

    def forward_list(self, x):
        assert isinstance(x, list)

        c = (self.b - self.a)

        n = len(x)
        out = [None for _ in range(n)]
        dims = [2, 3]
        for i in range(n):
            z = x[i]
            assert z.ndim == 4

            min_ = x[i]
            max_ = x[i]
            for dim in dims:
                min_ = torch.min(min_, dim=dim, keepdim=True)[0]
                max_ = torch.max(max_, dim=dim, keepdim=True)[0]

            t = self.a + c * (z - min_) / (max_ - min_ + self.eps)
            out[i] = torch.sigmoid(self.w * t)

        return out

    def forward_tensor(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        c = (self.b - self.a)
        dims = [2, 3]
        min_ = x
        max_ = x
        for dim in dims:
            min_ = torch.min(min_, dim=dim, keepdim=True)[0]
            max_ = torch.max(max_, dim=dim, keepdim=True)[0]

        t = self.a + c * (x - min_) / (max_ - min_ + self.eps)
        out = torch.sigmoid(self.w * t)

        return out

    def forward(self, x):
        if isinstance(x, list):
            return self.forward_list(x)
        elif isinstance(x, torch.Tensor):
            return self.forward_tensor(x)
        else:
            raise NotImplementedError






