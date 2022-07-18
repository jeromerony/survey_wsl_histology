"""
Create csv file of camelyon16 dataset.
"""
import random
import csv
import sys
import os
from os.path import join, dirname, abspath, basename
import math

import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.utils.tools import Dict2Obj

from dlib.utils.shared import announce_msg

from dlib.configure import constants
from dlib.configure.config import get_root_wsol_dataset

from dlib.utils.shared import find_files_pattern

from dlib.utils.reproducibility import set_seed
from dlib.datasets.core import dump_subset
from dlib.datasets.core import create_folds_of_one_class
from dlib.datasets.core import get_k


__all__ = ["do_camelyon16"]


def split_camelyon16(args):
    os.makedirs(args.fold_folder, exist_ok=True)
    classes = ["normal", "tumor"]
    datasetname = args.dataset
    dict_classes_names = {'normal': 0, 'tumor': 1}
    baseurl = args.baseurl
    psize = args.patch_size
    pre = f'camelyon16/w-{psize}xh-{psize}'
    ext = args.img_extension

    # test set.
    testsamples = dict()
    files = find_files_pattern(join(args.baseurl, pre,
                                    'metastatic-patches/testing'), f'*.{ext}')
    for f in files:
        b = basename(f)
        if b.startswith('mask'):
            continue

        k = get_k(f, baseurl)
        assert k not in testsamples
        mask = join(dirname(f), b.replace('Patch', 'mask'))
        assert os.path.isfile(mask)
        testsamples[k] = (get_k(mask, baseurl), 'tumor')

    files = find_files_pattern(
        join(args.baseurl, pre, 'normal-patches/split_1/fold_0/test'),
        f'*.{ext}')
    for f in files:
        k = get_k(f, baseurl)
        assert k not in testsamples
        mask = 'None'
        testsamples[k] = (get_k(mask, baseurl), 'normal')

    # train set.
    trainsamples = dict()
    files = find_files_pattern(join(args.baseurl, pre,
                                    'metastatic-patches/training'),
                               f'*.{ext}')
    for f in files:
        b = basename(f)
        if b.startswith('mask'):
            continue

        k = get_k(f, baseurl)
        assert k not in trainsamples
        mask = join(dirname(f), b.replace('Patch', 'mask'))
        assert os.path.isfile(mask)
        trainsamples[k] = (get_k(mask, baseurl), 'tumor')

    files = find_files_pattern(
        join(args.baseurl, pre, 'normal-patches/split_0/fold_0'),
        f'*.{ext}')
    for f in files:
        k = get_k(f, baseurl)
        assert k not in trainsamples
        mask = ''
        trainsamples[k] = (get_k(mask, baseurl), 'normal')

    tumor = [s for s in trainsamples.keys() if trainsamples[s][1] == "tumor"]
    normal = [s for s in trainsamples.keys() if trainsamples[s][1] == "normal"]

    os.makedirs(args.fold_folder, exist_ok=True)

    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    for _ in range(1000):
        random.shuffle(tumor)
        random.shuffle(normal)

    def create_kfolds():
        vl_size_normal = math.ceil(len(normal) * args.folding["vl"] / 100.)
        vl_size_tumor = math.ceil(
            len(tumor) * args.folding["vl"] / 100.)

        list_folds_normal = create_folds_of_one_class(
            normal, len(normal) - vl_size_normal, vl_size_normal)
        list_folds_tumor = create_folds_of_one_class(
            tumor, len(tumor) - vl_size_tumor, vl_size_tumor)

        assert len(list_folds_normal) == len(list_folds_tumor)

        print("We found {} folds .... [OK]".format(len(list_folds_tumor)))

        outd = args.fold_folder
        for i in range(args.nbr_folds):
            print(f'Creating fold {i}')

            train = list_folds_tumor[i][0] + list_folds_normal[i][0]
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

            validcl = list_folds_tumor[i][1] + list_folds_normal[i][1]
            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.CLVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validcl},
                        encoding=dict_classes_names, dataset=datasetname)

            n = args.vl_sup_per_cl
            validpx = random.sample(list_folds_tumor[i][1], n)
            validpx += random.sample(list_folds_normal[i][1], n)
            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.PXVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validpx},
                        encoding=dict_classes_names, dataset=datasetname)

    create_kfolds()
    print(f"All {datasetname} splitting ({args.nbr_folds}) "
          f"ended with success [OK].")


def do_camelyon16(root_main: str, seed: int, psize: int = 512):
    assert seed == 0
    set_seed(seed=seed, verbose=False)
    ds = constants.CAMELYON512
    assert str(psize) in ds

    announce_msg("Processing dataset: {}".format(ds))
    args = {"baseurl": join(get_root_wsol_dataset(), ds),
            "dataset": ds,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "fold_folder": join(root_main,
                                f"folds/wsol-done-right-splits/{ds}"),
            "img_extension": "png",
            'vl_sup_per_cl': 5,  # number of fully sup. samples per class for
            # validation.
            'patch_size': psize
            }

    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    set_seed(seed=seed, verbose=False)
    split_camelyon16(Dict2Obj(args))


if __name__ == '__main__':
    do_camelyon16(root_main=root_dir, seed=0)


