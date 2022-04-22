import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.reproducibility import set_seed

__all__ = ['Entropy']


class _CrossEntropy(nn.Module):
    """
    Compute Entropy between two distributions p, q:

    H(p, q) = - sum_i pi * log q_i.

    This is different from torch.nn.CrossEntropy() (and _CE) in the sens that
    _CrossEntropy() operates on probabilities and computes the entire sum
    because the target is not discrete but continuous.

    This loss can be used to compute the entropy of a distribution H(p):

    H(p) = - sum_i pi * log p_i.

    For this purpose use `forward(p, p)`.
    """
    def __init__(self, sumit=True):
        """
        Init. function.
        :param sumit: bool. If True, we sum across the dim=1 to obtain the true
        cross-entropy. In this case, the output is a vector of shape (n) where
        each component is the corresponding cross-entropy.

        If False, we do not sum across dim=1, and return a matrix of shape (
        n, m) where n is the number of samples and m is the number of
        elements in the probability distribution.
        """
        super(_CrossEntropy, self).__init__()

        self.sumit = sumit

    def forward(self, p, q):
        """
        The forward function. It operate on batches.
        :param p: tensor of size (n, m). Each row is a probability
        distribution.
        :param q: tensor of size (n, m). Each row is a probability
        distribution.
        :return: a vector of size (n) or (n, m) dependent on self.sumit.
        """
        if self.sumit:
            return (-p * torch.log2(q)).sum(dim=1)
        else:
            return -p * torch.log2(q)

    def __str__(self):
        return "{}(): Cross-entropy over continuous target. sumit={}.".format(
            self.__class__.__name__, self.sumit)


class Entropy(_CrossEntropy):
    """
    Class that computes the entropy of a distribution p:
    entropy = - sum_i p_i * log(p_i).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(Entropy, self).__init__(sumit=True)

    def forward(self, p, q=None):
        """
        Forward function.
        :param p: tensor of probability distribution (each row).
        :param q: None.
        """
        return super(Entropy, self).forward(p, p)

    def __str__(self):
        return "{}(): Entropy.".format(self.__class__.__name__)


def test_Entropy():
    from dlib.utils.shared import announce_msg

    set_seed(0)
    instance = Entropy()
    announce_msg("Testing {}".format(instance))

    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)

    b, c = 32, 2
    prob = (torch.rand(b, c)).to(DEVICE)

    out = instance(torch.softmax(prob, dim=1))
    print("Loss Entropy: {}".format(out))

    out = instance(torch.softmax(prob * 0., dim=1))
    print("Loss Entropy (Uniform): {}".format(out))


if __name__ == "__main__":
    test_Entropy()
