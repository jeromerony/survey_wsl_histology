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

from dlib.configure import constants
from dlib.configure.config import get_root_wsol_dataset

from dlib.utils.reproducibility import set_seed
from dlib.datasets.core import dump_subset
from dlib.datasets.core import create_folds_of_one_class


__all__ = ["do_glas"]


def split_glas(args):
    os.makedirs(args.fold_folder, exist_ok=True)
    classes = ["benign", "malignant"]
    datasetname = args.dataset
    dict_classes_names = {'benign': 0, 'malignant': 1}

    baseurl = args.baseurl
    pre = 'Warwick_QU_Dataset_(Released_2016_07_08)'
    trainsamples = dict()
    testsamples = dict()
    ext = args.img_extension

    with open(join(baseurl, pre, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space
            # before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            msg = "The class `{}` is not within the predefined " \
                  "classes `{}`".format(row[2], classes)
            assert row[2] in classes, msg
            # name file, patient id, label

            key = f'{pre}/{row[0]}.{ext}'
            c_s = (f'{pre}/{row[0]}_anno.{ext}', row[2])
            assert key not in trainsamples
            assert key not in testsamples

            if row[0].startswith('train'):
                trainsamples[key] = c_s
            elif row[0].startswith('test'):
                testsamples[key] = c_s

    assert len(trainsamples.keys()) + len(testsamples.keys()) == 165
    assert len(testsamples.keys()) == 80, len(testsamples.keys())

    # the number of samples per patient are highly unbalanced. so, we do not
    # split patients, but classes. --> we allow that samples from same
    # patient end up in train and valid. it is not that bad. it is just the
    # validation. plus, they are histology images. only the stain is more
    # likely to be relatively similar.

    benign = [s for s in trainsamples.keys() if trainsamples[s][1] == "benign"]
    malignant = [s for s in trainsamples.keys(
    ) if trainsamples[s][1] == "malignant"]

    os.makedirs(args.fold_folder, exist_ok=True)

    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    for _ in range(1000):
        random.shuffle(benign)
        random.shuffle(malignant)


    def create_kfolds():
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(
            len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(
            benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(
            malignant, len(malignant) - vl_size_malignant, vl_size_malignant)

        assert len(list_folds_benign) == len(list_folds_malignant)

        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(args.nbr_folds):
            print(f'Creating fold {i}')

            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
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

            validcl = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.CLVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validcl},
                        encoding=dict_classes_names, dataset=datasetname)

            n = args.vl_sup_per_cl
            validpx = random.sample(list_folds_malignant[i][1], n)
            validpx += random.sample(list_folds_benign[i][1], n)
            dump_subset(fold_folder=args.fold_folder, fold=i,
                        subset=constants.PXVALIDSET,
                        samples={kk: trainsamples[kk] for kk in validpx},
                        encoding=dict_classes_names, dataset=datasetname)

    create_kfolds()
    print(f"All {datasetname} splitting ({args.nbr_folds}) "
          f"ended with success [OK].")


def do_glas(root_main: str, seed: int):

    assert seed == 0
    set_seed(seed=seed, verbose=False)
    ds = constants.GLAS

    announce_msg("Processing dataset: {}".format(ds))

    args = {"baseurl": join(get_root_wsol_dataset(), ds),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": ds,
            "fold_folder": join(root_main,
                                f"folds/wsol-done-right-splits/{ds}"),
            "img_extension": "bmp",
            'vl_sup_per_cl': 3  # number of fully sup. samples per class for
            # validation.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    set_seed(seed=seed, verbose=False)
    split_glas(Dict2Obj(args))


if __name__ == '__main__':
    do_glas(root_main=root_dir, seed=0)
