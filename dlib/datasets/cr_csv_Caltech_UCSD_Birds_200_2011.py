"""
Create csv file of Caltech_UCSD_Birds_200_2011 dataset.
"""
import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath

import numpy as np
import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.csv_tools import get_stats

from dlib.utils.tools import chunk_it
from dlib.utils.tools import Dict2Obj
from dlib.datasets.tools import get_rootpath_2_dataset

from dlib.utils.shared import announce_msg

from dlib.configure import constants

from dlib.utils.reproducibility import set_default_seed


__all__ = ["do_Caltech_UCSD_Birds_200_2011"]


def dump_fold_into_csv_CUB(lsamples, outpath, tag):
    """
    for Caltech_UCSD_Birds_200_2011 dataset.
    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    assert tag == constants.L

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_path, mask_path, img_label, idcnt in lsamples:
            filewriter.writerow(
                [str(int(idcnt)),
                 img_path,
                 mask_path,
                 img_label,
                 tag]
            )


def split_Caltech_UCSD_Birds_200_2011(args):
    """
    Create a validation/train sets in Caltech_UCSD_Birds_200_2011 dataset.
    Test set is provided.

    :param args:
    :return:
    """
    baseurl = args.baseurl
    classes_names, classes_id = [], []
    # Load the classes: id class
    with open(join(baseurl, "CUB_200_2011", "classes.txt"), "r") as fcl:
        content = fcl.readlines()
        for el in content:
            el = el.rstrip("\n\r")
            idcl, cl = el.split(" ")
            classes_id.append(idcl)
            classes_names.append(cl)
    # Load the images and their id.
    images_path, images_id = [], []
    with open(join(baseurl, "CUB_200_2011", "images.txt"), "r") as fim:
        content = fim.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, imgpath = el.split(" ")
            images_id.append(idim)
            images_path.append(imgpath)

    # Load the image labels.
    images_label = (np.zeros(len(images_path)) - 1).tolist()
    with open(join(baseurl, "CUB_200_2011", "image_class_labels.txt"),
              "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, clid = el.split(" ")
            # find the image index correspd. to the image id
            images_label[images_id.index(idim)] = classes_names[
                classes_id.index(clid)]

    # All what we need is in images_label, images_path.
    # classes_names will be used later to convert class name into integers.
    msg = "We expect Caltech_UCSD_Birds_200_2011 dataset to have " \
          "11788 samples. We found {} ... [NOT OK]".format(len(images_id))
    assert len(images_id) == 11788, msg
    all_samples = list(zip(images_path, images_label))  # Not used.

    # Split into train and test.
    all_train_samples = []
    test_samples = []
    idcnt = 0.  # count the unique id for each sample

    with open(join(baseurl, "CUB_200_2011", "train_test_split.txt"), "r") as flb:
        content = flb.readlines()
        for el in content:
            el = el.strip("\n\r")
            idim, st = el.split(" ")
            img_idx = images_id.index(idim)
            img_path = images_path[img_idx]
            img_label = images_label[img_idx]
            filename, file_ext = os.path.splitext(img_path)
            mask_path = join("segmentations", filename + ".png")
            img_path = join("CUB_200_2011", "images", img_path)

            msg = "Image {} does not exist!".format(join(args.baseurl, img_path))
            assert os.path.isfile(join(args.baseurl, img_path)), msg

            msg = "Mask {} does not exist!".format(join(args.baseurl,
                                                        mask_path)
                                                   )
            assert os.path.isfile(join(args.baseurl, mask_path)), msg

            samplex = (img_path, mask_path, img_label, idcnt)
            if st == "1":  # train
                all_train_samples.append(samplex)
            elif st == "0":  # test
                test_samples.append(samplex)
            else:
                raise ValueError("Expected 0 or 1. "
                                 "Found {} .... [NOT OK]".format(st))

            idcnt += 1.

    print("Nbr. ALL train samples: {}".format(len(all_train_samples)))
    print("Nbr. test samples: {}".format(len(test_samples)))

    msg = "Something is wrong. We expected 11788. " \
          "Found: {}... [NOT OK]".format(
        len(all_train_samples) + len(test_samples))
    assert len(all_train_samples) + len(test_samples) == 11788, msg

    # Keep only the required classes:
    if args.nbr_classes is not None:
        fyaml = open(args.path_encoding, 'r')
        contyaml = yaml.load(fyaml)
        keys_l = list(contyaml.keys())
        indexer = np.array(list(range(len(keys_l)))).squeeze()
        select_idx = np.random.choice(indexer, args.nbr_classes, replace=False)
        selected_keys = []
        for idx in select_idx:
            selected_keys.append(keys_l[idx])

        # Drop samples outside the selected classes.
        tmp_all_train = []
        for el in all_train_samples:
            if el[2] in selected_keys:
                tmp_all_train.append(el)
        all_train_samples = tmp_all_train

        tmp_test = []
        for el in test_samples:
            if el[2] in selected_keys:
                tmp_test.append(el)

        test_samples = tmp_test

        classes_names = selected_keys

    # Train: Create dict where a key is the class name,
    # and the value is all the samples that have the same class.

    samples_per_class = dict()
    for cl in classes_names:
        samples_per_class[cl] = [el for el in all_train_samples if el[2] == cl]

    # Split
    splits = []
    print("Shuffling to create splits. May take some time...")
    for i in range(args.nbr_splits):
        for key in samples_per_class.keys():
            for _ in range(1000):
                random.shuffle(samples_per_class[key])
                random.shuffle(samples_per_class[key])
        splits.append(copy.deepcopy(samples_per_class))

    # encode class name into int.
    dict_classes_names = dict()
    for i in range(len(classes_names)):
        dict_classes_names[classes_names[i]] = i

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold
         contains a train, and valid set with a     predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set): where each
                 element is the list (str paths)
                 of the samples of each set: train, and valid, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the" \
                                           " provided sizes."

        # chunk the data into chunks of size ts (the size of the test set),
        # so we can rotate the test set.
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

    def create_one_split(split_i, test_samples, c_split, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param c_split: dict, contains the current split.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        l_folds_per_class = []
        for key in c_split.keys():
            # count the number of tr, vl for this current class.
            vl_size = math.ceil(len(c_split[key]) * args.folding["vl"] / 100.)
            tr_size = len(c_split[key]) - vl_size
            # Create the folds.
            list_folds = create_folds_of_one_class(c_split[key], tr_size, vl_size)

            msg = "We did not get exactly {} folds, " \
                  "but `{}` .... [ NOT OK]".format(nbr_folds,  len(list_folds))
            assert len(list_folds) == nbr_folds, msg

            l_folds_per_class.append(list_folds)

        outd = args.fold_folder
        # Re-arrange the folds.
        for i in range(nbr_folds):
            print("\t Fold: {}".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv_CUB(
                test_samples,
                join(out_fold, "test_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the train set
            train = []
            for el in l_folds_per_class:
                train += el[i][0]  # 0: tr
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv_CUB(
                train,
                join(out_fold, "train_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the valid set
            valid = []
            for el in l_folds_per_class:
                valid += el[i][1]  # 1: vl

            dump_fold_into_csv_CUB(
                valid,
                join(out_fold, "valid_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)
    # Creates the splits
    print("Starting splitting...")
    for i in range(args.nbr_splits):
        print("Split: {}".format(i))
        create_one_split(i, test_samples, splits[i], args.nbr_folds)

    print(
        "All Caltech_UCSD_Birds_200_2011 splitting (`{}`) ended with "
        "success .... [OK]".format(args.nbr_splits))


def do_Caltech_UCSD_Birds_200_2011(root_main):
    """
    Caltech-UCSD-Birds-200-2011.

    :param root_main: str. absolute path to folder containing main.py.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    set_default_seed()

    # ===========================

    announce_msg("Processing dataset: {}".format(constants.CUB))

    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.CUB})),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "Caltech-UCSD-Birds-200-2011",
            "fold_folder": join(root_main, "folds/Caltech-UCSD-Birds-200-2011"),
            "img_extension": "bmp",
            "nbr_splits": 1,  # how many times to perform the k-folds over
            # the available train samples.
            "path_encoding": join(
                root_main,
                "folds/Caltech-UCSD-Birds-200-2011/encoding-origine.yaml"),
            "nbr_classes": None  # Keep only 5 random classes. If you want
            # to use the entire dataset, set this to None.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    set_default_seed()
    split_Caltech_UCSD_Birds_200_2011(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')