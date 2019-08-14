import numpy as np
import random
import os
import warnings


DEFAULT_SEED = 0
MAX_SEED = 2**32 - 1  # maximum allowed seed (for now, it is the minimum value accepted by all the concerned modules
# as a maximum seed: numpy.random.seed()).


def get_seed():
    """
    Get the default seed from the environment variable.
    If not set, we use our default seed.
    :return: int, a seed.
    """
    try:
        print("===========================================================================")
        print("                          SEED: {}  ".format(os.environ["MYSEED"]))
        print("===========================================================================")
        return int(os.environ["MYSEED"])
    except KeyError:
        print(
            "In Bash, you need to create an environment variable of the seed named `MYSEED`, then set its value to an "
            "integer.\n"
            "For example, to create an environment named `MYSEED` and set it to the value 0, in your Bash terminal, "
            "before running this script, type: `export MYSEED=0`.")
        print(" .... [NOT OK]")

        warnings.warn("WE ARE GOING TO USE OUR DEFAULT SEED: {}  .... [NOT OK]".format(DEFAULT_SEED))
        os.environ["MYSEED"] = str(DEFAULT_SEED)
        print("===========================================================================")
        print("                          DEFAULT SEED: {}  ".format(os.environ["MYSEED"]))
        print("===========================================================================")
        return DEFAULT_SEED


def set_seed():
    """
    Set a seed to some modules for reproducibility.

    :return:
    """
    seed = get_seed()
    set_seed_to_modules(seed)


def set_seed_to_modules(seed):
    """
    Set manually a seed to the concerned modules for reproducibility.

    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)


def force_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The current seend.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Deterministic mode can have a performance impact, depending on your
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    # If multigpu is on, deactivate cudnn since it has many randmon things that we can not control.
    if torch.cuda.device_count() > 1 and os.environ["ALLOW_MULTIGPUS"] == 'True':
        torch.backends.cudnn.enabled = False