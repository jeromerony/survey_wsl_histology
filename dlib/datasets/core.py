import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath


from PIL import Image


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.configure import constants
from dlib.configure.config import get_root_wsol_dataset
from dlib.utils.tools import chunk_it


__all__ = ['dump_subset', 'create_folds_of_one_class', 'get_k']


def get_k(p: str, base: str) -> str:
    k_: str = p.replace(base, '')
    k_ = k_[1:] if k_.startswith('/') else k_

    return k_


def dump_subset(fold_folder: str,
                fold: int,
                subset: str,
                samples: dict,
                encoding: dict,
                dataset: str):
    assert subset in constants.SPLITS
    root = join(get_root_wsol_dataset(), dataset)

    # ids
    out_dir = join(fold_folder, f'fold-{fold}', subset)
    os.makedirs(out_dir, exist_ok=True)

    with open(join(out_dir, 'image_ids.txt'), 'w') as fout:
        for k in samples.keys():
            fout.write(f'{k}\n')

    # cl labels
    with open(join(out_dir, 'class_labels.txt'), 'w') as fout:
        for k in samples.keys():
            fout.write(f'{k},{encoding[samples[k][1]]}\n')

    # image_sizes
    with open(join(out_dir, 'image_sizes.txt'), 'w') as fout:
        for k in samples.keys():
            path = join(root, k)
            assert os.path.isfile(path), path
            w, h = Image.open(path).convert('RGB').size
            fout.write(f'{k},{w},{h}\n')

    # localization
    with open(join(out_dir, 'localization.txt'), 'w') as fout:
        for k in samples.keys():
            fout.write(f'{k},{samples[k][0]},\n')


def create_folds_of_one_class(lsamps, s_tr, s_vl):
    """
    Create k folds from a list of samples of the same class, each fold
    contains a train, and valid set with a
    predefined size.

    Note: Samples are expected to be shuffled beforehand.

    :param lsamps: list of ids of samples from the same class.
    :param s_tr: int, number of samples in the train set.
    :param s_vl: int, number of samples in the valid set.
    :return: list_folds: list of k tuples (tr_set, vl_set): w
    here each element is the list (str paths)
             of the samples of each set: train, valid, and test,
             respectively.
    """
    msg = "Something wrong with the provided sizes."
    assert len(lsamps) == s_tr + s_vl, msg

    # chunk the data into chunks of size ts
    # (the size of the test set), so we can rotate the test set.
    list_chunks = list(chunk_it(lsamps, s_vl))
    list_folds = []

    for i in range(len(list_chunks)):
        vl_set = list_chunks[i]

        right, left = [], []
        if i < len(list_chunks) - 1:
            right = list_chunks[i + 1:]
        if i > 0:
            left = list_chunks[:i]

        leftoverchunks = right + left

        leftoversamples = []
        for e in leftoverchunks:
            leftoversamples += e

        tr_set = leftoversamples
        list_folds.append((tr_set, vl_set))

    return list_folds
