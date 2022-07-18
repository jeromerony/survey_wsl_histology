"""
Contains high level functions.
modules in this folder can not import from this module (recursive importing
may happen).
"""

import sys
from os.path import dirname
from os.path import abspath
from copy import deepcopy

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.cr_csv_cityscapes import encode_meta_cityscapes

from dlib.configure import constants

__all__ = [
    "re_encode_global_label_to_meta_label"
]


def re_encode_global_label_to_meta_label(l_samples, dataset_name):
    """
    Re-encode global labels using meta-labels.
    Applicable ONLY in the case of multi-label.

    :param l_samples: list of samples loaded from csv (samples with absolute
    paths).
    :param dataset_name: str. name of the dataset.
    :return: return a copy of the list with global labels re-encoded
    using metal-labels.
    """
    multi_label = (dataset_name in constants.MULTI_LABEL_DATASETS)

    if not multi_label:
        raise ValueError("ERROR. Function called on not multi-label "
                         "dataset. Dataset {} not multi-label.".format(
            dataset_name
        ))


    out = deepcopy(l_samples)

    if dataset_name == constants.CSCAPES:

        for i in range(len(out)):
            msg = "ERROR: Encoding of CSV files has changed. " \
                  "Expected 5. Found {}".format(len(out[i]))
            assert len(out[i]) == 5, msg

            # get the label and change it to a metal-label.
            out[i][3] = encode_meta_cityscapes(out[i][3])
    else:

        raise NotImplementedError

    return out