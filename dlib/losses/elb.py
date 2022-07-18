import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.reproducibility import set_seed

__all__ = ['ELB']


class ELB(nn.Module):
    """
    Extended log-barrier loss (ELB).
    Optimize inequality constraint : f(x) <= 0.

    Refs:
    1. Kervadec, H., Dolz, J., Yuan, J., Desrosiers, C., Granger, E., and Ben
     Ayed, I. (2019b). Constrained deep networks:Lagrangian optimization
     via log-barrier extensions.CoRR, abs/1904.04205
    2. S. Belharbi, I. Ben Ayed, L. McCaffrey and E. Granger,
    “Deep Ordinal Classification with Inequality Constraints”, CoRR,
    abs/1911.10720, 2019.
    """
    def __init__(self,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01
                 ):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(ELB, self).__init__()

        assert isinstance(mulcoef, float)
        assert mulcoef > 0.
        assert isinstance(init_t, float)
        assert init_t > 0.
        assert isinstance(max_t, float)
        assert max_t > init_t

        self.init_t = init_t

        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float())
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float())
        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float())

    def set_t(self, val):
        """
        Set the value of `t`, the hyper-parameter of the log-barrier method.
        :param val: float > 0. new value of `t`.
        :return:
        """
        msg = "`t` must be a float. You provided {} ....[NOT OK]".format(
            type(val))
        assert isinstance(val, float) or (isinstance(val, torch.Tensor) and
                                          val.ndim == 1 and val.dtype ==
                                          torch.float)
        assert val > 0.

        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float()).to(
                self.t_lb.device
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer("t_lb", val.float().requires_grad_(False))

    def get_t(self):
        """
        Returns the value of 't_lb'.
        """
        return self.t_lb

    def update_t(self):
        """
        Update the value of `t`.
        :return:
        """
        self.set_t(torch.min(self.t_lb * self.mulcoef, self.max_t))

    def forward(self, fx):
        """
        The forward function.
        :param fx: pytorch tensor. a vector.
        :return: real value extended-log-barrier-based loss.
        """
        assert fx.ndim == 1

        loss_fx = fx * 0.

        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb**2))

        idx_less = torch.nonzero((fx < ct) | (fx == ct),
                                 as_tuple=False).squeeze()
        if idx_less.numel() > 0:
            val_less = fx[idx_less]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_fx[idx_less] = loss_less

        # vals > -1/(t**2).
        idx_great = torch.nonzero(fx > ct, as_tuple=False).squeeze()
        if idx_great.numel() > 0:
            val_great = fx[idx_great]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                torch.log((1. / (self.t_lb**2))) + (1. / self.t_lb)
            loss_fx[idx_great] = loss_great

        loss = loss_fx.mean()

        return loss

    def __str__(self):
        return "{}(): ELB method.".format(self.__class__.__name__)


def test_ELB():
    from dlib.utils.shared import announce_msg

    set_seed(0)
    instance = ELB(init_t=1., max_t=10., mulcoef=1.01)
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)

    b = 16
    fx = (torch.rand(b)).to(DEVICE)

    out = instance(fx)
    for r in range(10):
        instance.update_t()
        print("epoch {}. t: {}.".format(r, instance.t_lb))
    print("Loss ELB.sum(): {}".format(out))


if __name__ == "__main__":
    test_ELB()
