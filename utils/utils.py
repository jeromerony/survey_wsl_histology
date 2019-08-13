import csv

from collections import OrderedDict


def txt_loader(fname):
    with open(fname, 'r') as f:
        out = f.read().splitlines()
    return out


def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values = []
        self.counter = 0

    def append(self, val: float) -> None:
        self.values.append(val)
        self.counter += 1

    @property
    def val(self) -> float:
        return self.values[-1]

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self) -> float:
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


def state_dict_to_cpu(state: OrderedDict):
    new_state = OrderedDict()
    for k in state.keys():
        newk = k.replace('module.', '')  # remove "module." if model was trained using DataParallel
        new_state[newk] = state[k].cpu()
    return new_state
