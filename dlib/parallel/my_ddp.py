import sys
from os.path import dirname, abspath

from torch.nn.parallel import DistributedDataParallel as DDP

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['MyDDP']


class MyDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

