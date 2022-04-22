from .learningmode import (
    LmTracker
)

__version__ = "0.1.0"


class LMTRACKERNotInitialized(Exception):
    pass


class LMTRACKERAlreadyInitialized(Exception):
    pass


class NotInitializedObject(object):
    def __getattribute__(self, name):
        raise LMTRACKERNotInitialized(
            "LMTRACKER not initialized. "
            "Initialize LMTRACKER with init() function"
        )


GLOBAL_LMTRACKER = NotInitializedObject()


def mode():
    return GLOBAL_LMTRACKER.mode()


def set_c_epoch(epoch):
    GLOBAL_LMTRACKER.set_c_epoch(epoch)


def time_to_reset():
    return GLOBAL_LMTRACKER.time_to_reset()


def init(task,
         sr,
         sr_start_ep,
         sr_end_ep,
         sl_fc,
         sl_start_ep,
         sl_end_ep,
         c_epoch
         ):
    global GLOBAL_LMTRACKER
    try:
        if isinstance(GLOBAL_LMTRACKER, LmTracker):
            raise LMTRACKERAlreadyInitialized()
    except LMTRACKERNotInitialized:
        GLOBAL_LMTRACKER = LmTracker(task,
                                     sr,
                                     sr_start_ep,
                                     sr_end_ep,
                                     sl_fc,
                                     sl_start_ep,
                                     sl_end_ep,
                                     c_epoch
                                     )
