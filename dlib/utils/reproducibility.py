# self-contained-as-possible module.
# handles reproducibility procedures.

import random
import os
import warnings


import numpy as np
import torch


DEFAULT_SEED = 0   # the default seed.


__all__ = [
    'check_if_allow_multgpu_mode',
    'set_seed',
    'set_default_seed',
    'reset_default_seed',
    'set_to_deterministic'
]


def _announce_msg(msg, upper=True):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into
    uppercase. Else, the message is displayed
    as it is.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    print(output_msg)

    return output_msg


def _get_current_seed(verbose=False):
    """
    Get the default seed from the environment variable.
    If not set, we use our default seed.
    :return: int, a seed.
    """
    try:
        if verbose:
            msg = "ACQUIRING SEED FROM OS.ENVIRON['MYSEED']: {}  " \
                  "".format(os.environ["MYSEED"])
            _announce_msg(msg)

        return int(os.environ["MYSEED"])
    except KeyError:
        if verbose:
            print("`os.environ` does not have a key named `MYSEED`."
                  "This key is supposed to hold the current seed. "
                  "Please set it, and try again, if you want.")

            warnings.warn("MEANWHILE, .... WE ARE GOING TO USE OUR DEFAULT "
                          "SEED: {}".format(DEFAULT_SEED))
        os.environ["MYSEED"] = str(DEFAULT_SEED)
        if verbose:
            msg = "DEFAULT SEED: {}  ".format(os.environ["MYSEED"])
            _announce_msg(msg)
        return DEFAULT_SEED


def _make_cudnn_deterministic(verbose=False):
    if verbose:
        _announce_msg("SETTING CUDNN TO DETERMINISTIC")

    torch.backends.cudnn.benchmark = False
    # Deterministic mode can have a performance impact, depending on your
    torch.backends.cudnn.deterministic = True
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn


def set_seed(seed=None, verbose=False):
    """
    Note:

    While this attempts to ensure reproducibility, it does not offer an
    absolute guarantee. The results may be similar to some precision.
    Also, they may be different due to an amplification to extremely
    small differences.

    See:

    https://pytorch.org/docs/stable/notes/randomness.html
    https://stackoverflow.com/questions/50744565/
    how-to-handle-non-determinism-when-training-on-a-gpu

    :param seed: int, a seed. Default is None: use the default seed (0).
    :param verbose: bool.
    :return:
    """
    if seed is None:
        seed = _get_current_seed(verbose=verbose)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # If multigpu is on, deactivate cudnn since it has many random things
    # that we can not control.
    # if check_cudnn:
    #     cond = torch.cuda.device_count() > 1
    #     cond = cond and (os.environ["ALLOW_MULTIGPUS"] == 'True')
    #     if cond:
    #         torch.backends.cudnn.enabled = False


def reset_default_seed(seed, verbose=False):
    assert seed is not None
    if verbose:
        _announce_msg("Reset default seed from [{}] to [{}]".format(
            os.environ['MYSEED'], seed
        ))

    os.environ["MYSEED"] = str(seed)
    set_seed(seed=seed, verbose=False)


def set_default_seed():
    set_seed(seed=None)


def set_to_deterministic(seed=None, verbose=False):
    set_seed(seed=seed, verbose=verbose)
    _make_cudnn_deterministic(verbose=verbose)


