import sys
from os.path import dirname, abspath
from typing import Union

import torch
import torch.distributed as dist
root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.parallel.my_ddp import MyDDP


def sync_tensor_across_gpus(t: Union[torch.Tensor, None]
                            ) -> Union[torch.Tensor, None]:
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor, dim=0)
