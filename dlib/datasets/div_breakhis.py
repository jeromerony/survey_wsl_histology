import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath, basename

import yaml


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.utils.tools import Dict2Obj

from dlib.utils.shared import announce_msg
from dlib.utils.shared import find_files_pattern

from dlib.configure import constants
from dlib.configure.config import get_root_wsol_dataset

from dlib.utils.reproducibility import set_seed
from dlib.datasets.core import dump_subset
from dlib.datasets.core import create_folds_of_one_class
from dlib.datasets.core import get_k


__all__ = ['do_breakhis']


def split_breakhis(args):
    os.makedirs(args.fold_folder, exist_ok=True)
    classes = ["benign", "malignant"]
    datasetname = args.dataset
    dict_classes_names = {'benign': 0, 'malignant': 1}

    baseurl = args.baseurl
    pre = 'mkfold'

    ext = args.img_extension
    nbr_folds = args.nbr_folds
    mag = args.mag

    os.makedirs(args.fold_folder, exist_ok=True)

    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    for fold in range(args.nbr_folds):
        fold += 1
        print(f'Processing fold {fold}')

        trainsamples = dict()
        testsamples = dict()
        holder = dict()
        fx = f'fold{fold}'
        files = find_files_pattern(
            join(args.baseurl, pre, fx, 'test', mag), f'*.{ext}')

        for i, f in enumerate(files):
            k = get_k(f, baseurl)
            assert k not in holder
            holder[k] = None
            b = basename(f)
            assert b.startswith("SOB_M") or b.startswith("SOB_B")

            if b.startswith('SOB_M'):
                z = 'malignant'
            else:
                z = 'benign'

            mask = ''
            testsamples[k] = (mask, z)

        files = find_files_pattern(
            join(args.baseurl, pre, fx, 'train', mag), f'*.{ext}')

        for i, f in enumerate(files):
            k = get_k(f, baseurl)
            assert k not in holder
            holder[k] = None
            b = basename(f)
            assert b.startswith("SOB_M") or b.startswith("SOB_B")

            if b.startswith('SOB_M'):
                z = 'malignant'
            else:
                z = 'benign'

            mask = ''
            trainsamples[k] = (mask, z)

        benign = [s for s in trainsamples.keys() if
                  trainsamples[s][1] == "benign"]
        malignant = [s for s in trainsamples.keys(
        ) if trainsamples[s][1] == "malignant"]

        for i in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)

        sz_vl = math.ceil(len(benign) * args.folding["vl"] / 100.)
        validset_b = benign[:sz_vl]
        validset_m = malignant[:sz_vl]

        train = benign[sz_vl:] + malignant[sz_vl:]
        for _ in range(1000):
            random.shuffle(train)

        dump_subset(fold_folder=args.fold_folder, fold=fold,
                    subset=constants.TRAINSET,
                    samples={kk: trainsamples[kk] for kk in train},
                    encoding=dict_classes_names, dataset=datasetname)

        dump_subset(fold_folder=args.fold_folder, fold=fold,
                    subset=constants.TESTSET,
                    samples=testsamples,
                    encoding=dict_classes_names, dataset=datasetname)

        validcl = validset_b + validset_m
        dump_subset(fold_folder=args.fold_folder, fold=fold,
                    subset=constants.CLVALIDSET,
                    samples={kk: trainsamples[kk] for kk in validcl},
                    encoding=dict_classes_names, dataset=datasetname)

        n = args.vl_sup_per_cl
        validpx = random.sample(validset_b, n)
        validpx += random.sample(validset_m, n)
        dump_subset(fold_folder=args.fold_folder, fold=fold,
                    subset=constants.PXVALIDSET,
                    samples={kk: trainsamples[kk] for kk in validpx},
                    encoding=dict_classes_names, dataset=datasetname)

    print(f"All {datasetname} splitting ({args.nbr_folds}) "
          f"ended with success [OK].")


def do_breakhis(root_main: str, seed: int, mag: str):
    assert mag in ["40X", "100X", "200X", "400X"]

    assert seed == 0
    set_seed(seed=seed, verbose=False)
    ds = constants.BREAKHIS

    announce_msg(f"Processing dataset: {ds} Magnefication: {mag}")

    args = {"baseurl": join(get_root_wsol_dataset(), ds),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": ds,
            "fold_folder": join(root_main,
                                f"folds/wsol-done-right-splits/{ds}/{mag}"),
            "img_extension": "png",
            'vl_sup_per_cl': 3,  # number of fully sup. samples per class for
            # validation. irrelevant.
            'mag': mag,  # magnification.
            'nbr_folds': 5
            }

    set_seed(seed=seed, verbose=False)
    split_breakhis(Dict2Obj(args))


if __name__ == '__main__':
    do_breakhis(root_main=root_dir, seed=0, mag='40X')
    do_breakhis(root_main=root_dir, seed=0, mag='100X')
    do_breakhis(root_main=root_dir, seed=0, mag='200X')
    do_breakhis(root_main=root_dir, seed=0, mag='400X')

