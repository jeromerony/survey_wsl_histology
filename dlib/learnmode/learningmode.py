import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants


class LmTracker:
    def __init__(self,
                 task,
                 sr,
                 sr_start_ep,
                 sr_end_ep,
                 sl_fc,
                 sl_start_ep,
                 sl_end_ep,
                 c_epoch
                 ):
        raise NotImplementedError

        self.task = task

        # assert sr
        self.sr = sr
        self.sr_start_ep = sr_start_ep

        if sr_end_ep == -1:
            sr_end_ep = None
        self.sr_end_ep = sr_end_ep

        self.sl_fc = sl_fc
        self.sl_start_ep = sl_start_ep
        if sl_end_ep == -1:
            sl_end_ep = None
        self.sl_end_ep = sl_end_ep

        self.c_epoch = c_epoch

    def set_c_epoch(self, epoch):
        self.c_epoch = epoch

    def get_status(self, epoch, start_, end_):
        if (start_ is None) and (end_ is None):
            return True

        l = [epoch, start_, end_]
        if all([isinstance(z, int) for z in l]):
            return start_ <= epoch <= end_

        if start_ is None and isinstance(end_, int):
            return epoch <= end_

        if isinstance(start_, int) and end_ is None:
            return epoch >= start_

        return False

    def time_to_reset(self):
        if self.task != constants.F_CL:
            return False

        return self.sl_fc and (self.c_epoch == self.sl_start_ep)

    def mode(self):
        if self.task != constants.F_CL:
            return constants.LM_VOID

        if self.get_status(self.c_epoch, self.sr_start_ep, self.sr_end_ep):
            return constants.LM_EMERGENCE
        elif self.get_status(self.c_epoch, self.sl_start_ep, self.sl_end_ep):
            return constants.LM_SELFLEARNING
        else:
            raise ValueError
