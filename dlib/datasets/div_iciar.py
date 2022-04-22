import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath

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


__all__ = ["do_iciar"]


def split_iciar(args):
    os.makedirs(args.fold_folder, exist_ok=True)
    classes = ["Normal", "Benign", 'InSitu', 'Invasive']
    datasetname = args.dataset
    dict_classes_names = {'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3}

    baseurl = args.baseurl
    pre = 'ICIAR2018_BACH_Challenge/Photos'
    trainsamples = dict()
    testsamples = dict()
    ext = args.img_extension
    test_portion = args.test_portion

    testsamples = dict()
    trainsamples = dict()
    holder = dict()

    for z in classes:
        files = find_files_pattern(join(args.baseurl, pre, z), f'*.{ext}')
        for i in range(1000):
            random.shuffle(files)

        t = int(test_portion * len(files))
        for i, f in enumerate(files):
            k = get_k(f, baseurl)
            assert k not in holder
            holder[k] = None

            mask = ''
            if i <= t:
                testsamples[k] = (mask, z)
            else:
                trainsamples[k] = (mask, z)

    holder = {
        z: [s for s in trainsamples.keys() if trainsamples[s][1] == z]
        for z in classes
    }

    os.makedirs(args.fold_folder, exist_ok=True)

    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    for z in classes:
        for _ in range(1000):
            random.shuffle(holder[z])

    def create_kfolds():
        sz_vl = {
            z: math.ceil(len(holder[z]) * args.folding["vl"] / 100.) for z
            in classes
        }

        folds = {
            z: create_folds_of_one_class(
                holder[z], len(holder[z]) - sz_vl[z], sz_vl[z]) for z in classes
        }

        outd = args.fold_folder
        for i in range(args.nbr_folds):
            print(f'Creating fold {i}')

            train = []
            for z in classes:
                train += folds[z][i][0]

            for t in range(1000):
                random.shuffle(train)

            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.TRAINSET,
                        samples={kk: trainsamples[kk] for kk in train},
                        encoding=dict_classes_names, dataset=datasetname)

            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.TESTSET,
                        samples=testsamples,
                        encoding=dict_classes_names, dataset=datasetname)

            validcl = []
            for z in classes:
                validcl += folds[z][i][1]

            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.CLVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validcl},
                        encoding=dict_classes_names, dataset=datasetname)

            n = args.vl_sup_per_cl
            validpx = []
            for z in classes:
                validpx += random.sample(folds[z][i][1], n)

            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.PXVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validpx},
                        encoding=dict_classes_names, dataset=datasetname)

    create_kfolds()
    print(f"All {datasetname} splitting ({args.nbr_folds}) "
          f"ended with success [OK].")


def do_iciar(root_main: str, seed: int):

    assert seed == 0
    set_seed(seed=seed, verbose=False)
    ds = constants.ICIAR

    announce_msg("Processing dataset: {}".format(ds))

    args = {"baseurl": join(get_root_wsol_dataset(), ds),
            'test_portion': 0.5,  # task 50% of samples for test. the rest
            # split it between train and valid.
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": ds,
            "fold_folder": join(root_main,
                                f"folds/wsol-done-right-splits/{ds}"),
            "img_extension": "tif",
            'vl_sup_per_cl': 3  # number of fully sup. samples per class for
            # validation.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    set_seed(seed=seed, verbose=False)
    split_iciar(Dict2Obj(args))


if __name__ == '__main__':
    do_iciar(root_main=root_dir, seed=0)
