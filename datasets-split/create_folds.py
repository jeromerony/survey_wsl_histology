"""
Splits the following dataset into k-folds:
1. BreakHis.
2. BACH (part A) 2018.
3. GlaS.
4. CAMELYON16.
"""

__author__ = "Soufiane Belharbi, https://sbelharbi.github.io/"
__copyright__ = "Copyright 2018, ÉTS-Montréal"
__license__ = "GPL"
__version__ = "3"
__maintainer__ = "Soufiane Belharbi"
__email__ = "soufiane.belharbi.1@etsmtl.net"


import glob
from os.path import join
import os
import traceback
import random
import sys
import math
import csv
import copy
import getpass


from tools import chunk_it, Dict2Obj, announce_msg

import reproducibility


def create_k_folds_csv_bach_part_a(args):
    """
    Create k folds of the dataset BACH (part A) 2018 and store the image path of each fold in a *.csv file.

    1. Test set if fixed for all the splits/folds.
    2. We do a k-fold over the remaining data to create train, and validation sets.

    :param args: object, contain the arguments of splitting.
    :return:
    """
    announce_msg("Going to create the  splits, and the k-folds fro BACH (PART A) 2018 .... [OK]")

    rootpath = args.baseurl
    if args.dataset == "bc18bch":
        rootpath = join(rootpath, "ICIAR2018_BACH_Challenge/Photos")
    else:
        raise ValueError("Dataset name {} is unknown.".format(str(args.dataset)))

    samples = glob.glob(join(rootpath, "*", "*." + args.img_extension))
    # Originally, the function was written where 'samples' contains the absolute paths to the files.
    # Then, we realise that using absolute path on different platforms leads to a non-deterministic folds even
    # with the seed fixed. This is a result of glob.glob() that returns a list of paths more likely depending on the
    # how the files are saved within the OS. Therefore, to get rid of this, we use only a short path that is constant
    # across all the hosts where our dataset is saved. Then, we sort the list of paths BEFORE we go further. This
    # will guarantee that whatever the OS the code is running in, the sorted list is the same.
    samples = sorted([join(*sx.split(os.sep)[-4:]) for sx in samples])

    classes = {key: [s for s in samples if s.split(os.sep)[-2] == key] for key in args.name_classes.keys()}

    all_train = {}
    test_fix = []
    # Shuffle to avoid any bias.
    for key in classes.keys():
        for i in range(1000):
            random.shuffle(classes[key])

        nbr_test = int(len(classes[key]) * args.test_portion)
        test_fix += classes[key][:nbr_test]
        all_train[key] = classes[key][nbr_test:]

    # Test set is ready. Now, we need to do k-fold over the train.

    # Create the splits over the train
    splits = []
    for i in range(args.nbr_splits):
        for t in range(1000):
            for k in all_train.keys():
                random.shuffle(all_train[k])

        splits.append(copy.deepcopy(all_train))

    readme = "csv format:\n" \
             "relative path to the image file.\n" \
             "Example:\n" \
             "ICIAR2018_BACH_Challenge/Photos/Normal/n047.tif\n" \
             "There are four classes: normal, benign, in situ, and invasive.\n" \
             "The class of the sample may be infered from the parent folder of the image (in the example " \
             "above:\n " \
             "Normal:\n" \
             "Normal: class 'normal'\n" \
             "Benign: class 'benign'\n" \
             "InSitu: class 'in situ'\n" \
             "Invasive: class 'invasive'"

    # Create k-folds for each split.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Samples need to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes .... [NOT OK]"

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
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

    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname in lsamples:
                filewriter.writerow([fname])

    def create_one_split(split_i, test_samples, train_samples_all, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: dict of list, each key represents a class (test set, fixed).
        :param train_samples_all: dict of list, each key represent a class (all train set).
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        # Create the k-folds
        list_folds_of_class = {}

        for key in train_samples_all.keys():
            vl_size = math.ceil(len(train_samples_all[key]) * args.folding["vl"] / 100.)
            tr_size = len(train_samples_all[key]) - vl_size
            list_folds_of_class[key] = create_folds_of_one_class(train_samples_all[key], tr_size, vl_size)

            assert len(list_folds_of_class[key]) == nbr_folds, "We didn't get `{}` folds, but `{}` .... " \
                                                               "[NOT OK]".format(
                nbr_folds, len(list_folds_of_class[key]))

            print("We obtained `{}` folds for the class {}.... [OK]".format(args.nbr_folds, key))

        outd = args.fold_folder
        for i in range(nbr_folds):
            print("Fold {}:\n\t".format(i))
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            train = []
            valid = []
            for key in list_folds_of_class.keys():
                # Train
                train += list_folds_of_class[key][i][0]

                # Valid.
                valid += list_folds_of_class[key][i][1]

            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)
        print("BACH (PART A) 2018 splitting N° `{}` ends with success .... [OK]".format(split_i))

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    # Creates the splits
    for i in range(args.nbr_splits):
        print("Split {}:\n\t".format(i))
        create_one_split(i, test_fix, splits[i], args.nbr_folds)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    print("All BACH (PART A) 2018 splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_valid_breakhis(args):
    """
    Create a validation/train sets in BreakHis dataset.

    NOTE: samples that start with "SOB_B" are benign. Samples that strat with "SOB_M" are malignant.

    :param args:
    :return:
    """

    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line in lsamples:
                filewriter.writerow([line])

    readme = ("csv format:\n"
              "relative path to the image file.\n"
              "Example:\n"
              "mkfold/fold5/test/40X/SOB_B_A-14-22549AB-40-015.png\n"
              "There are two classes: benign, and malignant.\n"
              "The class of the sample depends on the beginning of the base name:\n"
              "SOB_B: Benign\n"
              "SOB_M: Malignant.")
    for fold in range(1, args.nbr_folds + 1, 1):
        print("Fold {} .... [OK]".format(fold))
        for mag in args.magnification:
            print("Magnification factor {} .... [OK]".format(mag))
            ind = join(args.baseurl, "fold" + str(fold))
            # Do test
            test_samples = glob.glob(join(ind, "test", mag, "*." + args.img_extension))
            test_samples = [join(*sx.split(os.sep)[-5:]) for sx in test_samples]  # keep "mkfold/fold1/test/40X/xx.png

            # Do train
            all_train_samples = glob.glob(join(ind, "train", mag, "*." + args.img_extension))
            # Sort to remove the dependency to the baseurl.
            all_train_samples = sorted([join(*sx.split(os.sep)[-5:]) for sx in all_train_samples])

            # take each class separably: benign, malignant.
            benign = [sx for sx in all_train_samples if sx.split(os.sep)[-1].startswith("SOB_B")]
            maligant = [sx for sx in all_train_samples if sx.split(os.sep)[-1].startswith("SOB_M")]

            # shuffle very well to mix all the sub-classes.
            for i in range(1000):
                random.shuffle(benign)
                random.shuffle(maligant)

            # DO the splits:
            for split_i in range(args.nbr_splits):
                # Output folder
                outd = join(args.fold_folder, "split_" + str(split_i), "fold_" + str(fold), mag)
                # This will allows the splits to be different.
                for i in range(1000):
                    random.shuffle(benign)
                    random.shuffle(maligant)

                # take x% for validation from each class
                ben_vl = int(len(benign) * args.folding["vl"] / 100.)
                malig_vl = int(len(maligant) * args.folding["vl"] / 100.)
                valid_samples = benign[:ben_vl] + maligant[:malig_vl]
                train_samples = benign[ben_vl:] + maligant[malig_vl:]

                # shuffle very well the train set
                for i in range(1000):
                    random.shuffle(train_samples)

                print("Number of test samples {} .... [OK]".format(len(test_samples)))
                print("Number of ALL train samples {} .... [OK]".format(len(all_train_samples)))
                print("Number of ACTUAL train samples {} .... [OK]".format(len(train_samples)))
                print("Number of valid samples {} .... [OK]".format(len(valid_samples)))

                # Write into *.csv
                if not os.path.exists(outd):
                    os.makedirs(outd)

                # write test, train, valid
                dump_fold_into_csv(test_samples, join(outd, "test_s_{}_f_{}_mag_{}.csv".format(
                    split_i, fold, mag)))
                dump_fold_into_csv(train_samples, join(outd, "train_s_{}_f_{}_mag_{}.csv".format(
                    split_i, fold, mag)))
                dump_fold_into_csv(valid_samples, join(outd, "valid_s_{}_f_{}_mag_{}.csv".format(
                    split_i, fold, mag)))

                with open(join(outd, "seed.txt"), 'w') as fx:
                    fx.write("MYSEED: " + os.environ["MYSEED"])

                with open(join(outd, "readme.md"), 'w') as fx:
                    fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)

    print("BreakHis splitting ends with success .... [OK]")


def split_valid_glas(args):
    """
    Create a validation/train sets in GlaS dataset.
    csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

    :param args:
    :return:
    """
    classes = ["benign", "malignant"]
    all_samples = []
    # Read the file Grade.csv
    baseurl = args.baseurl
    with open(join(baseurl, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            assert row[2] in classes, "The class `{}` is not within the predefined classes `{}`".format(row[2], classes)
            all_samples.append([row[0], row[2]])

    assert len(all_samples) == 165, "The number of samples {} do not match what they said (165) .... [NOT " \
                                    "OK]".format(len(all_samples))

    # Take test samples aside. They are fix.
    test_samples = [s for s in all_samples if s[0].startswith("test")]
    assert len(test_samples) == 80, "The number of test samples {} is not 80 as they said .... [NOT OK]".format(len(
        test_samples))

    all_train_samples = [s for s in all_samples if s[0].startswith("train")]
    assert len(all_train_samples) == 85, "The number of train samples {} is not 85 as they said .... [NOT OK]".format(
        len(all_train_samples))

    benign = [s for s in all_train_samples if s[1] == "benign"]
    malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # Split
    splits = []
    for i in range(args.nbr_splits):
        for _ in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)
        splits.append({"benign": copy.deepcopy(benign),
                       "malignant": copy.deepcopy(malignant)}
                      )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes."

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
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

    # Save the folds.
    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for name, clas in lsamples:
                filewriter.writerow([name + ".bmp", name + "_anno.bmp", clas])

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(malignant, len(malignant) - vl_size_malignant,
                                                         vl_size_malignant)

        assert len(list_folds_benign) == len(list_folds_malignant), "We didn't obtain the same number of fold" \
                                                                    " .... [NOT OK]"
        assert len(list_folds_benign) == 5, "We did not get exactly 5 folds, but `{}` .... [ NOT OK]".format(
            len(list_folds_benign))
        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

        with open(join(outd, "readme.md"), 'w') as fx:
            fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                     "(str: benign, malignant).")

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write("csv format:\nrelative path to the image, relative path to the mask, class "
                 "(str: benign, malignant).")
    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(i, test_samples, splits[i]["benign"], splits[i]["malignant"], args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


def split_valid_camelyon16_WSI_level(args):
    """
    Create a validation/train sets in CAMELYON16 dataset at WSI level.
    csv file format:
        relative path to the image, class (str: normal, metastases), path to .xml file (if tumor, else "").

    NOTE: WSI with tumor and without xml file annotation are discarded such as: test_114.tif (the only file that we
    found without annotation).

    :param args:
    :return:
    """
    classes = ["normal", "tumor"]
    baseurl = args.baseurl

    # Load all provided train samples
    all_tr_normal = glob.glob(join(baseurl, "training/normal", "*." + args.img_extension))
    all_tr_tumor = glob.glob(join(baseurl, "training/tumor", "*." + args.img_extension))

    # Sort in order to be platform independent.
    all_tr_normal = sorted([join(*sx.split(os.sep)[-3:]) for sx in all_tr_normal])
    all_tr_tumor = sorted([join(*sx.split(os.sep)[-3:]) for sx in all_tr_tumor])

    # Add: class, path to .xml to train samples.
    all_tr_normal = [[im, "normal", ""] for im in all_tr_normal]
    all_tr_tumor = [[im, "tumor", "training/lesion_annotations/" + im.split(os.sep)[-1].split(".")[0] + ".xml"]
                    for im in all_tr_tumor if os.path.isfile(
            join(baseurl, "training/lesion_annotations/" + im.split(os.sep)[-1].split(".")[0] + ".xml"))]

    # Test set
    all_test = []
    with open(join(baseurl, "testing/reference.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fname = row[0]
            clas = row[1].lower()
            assert clas in classes, "The class `{}` is not within the predefined classes `{}`".format(clas, classes)
            if clas == "tumor" and os.path.isfile(join(baseurl, join("testing/lesion_annotations", fname + ".xml"))):
                all_test.append(
                    [join("testing/images", fname + ".tif"), clas, join("testing/lesion_annotations", fname + ".xml")]
                )
            elif clas == "normal":
                all_test.append(
                    [join("testing/images", fname + ".tif"), clas, ""])

    # Splits:
    splits = []
    for i in range(args.nbr_splits):
        for t in range(1000):
            random.shuffle(all_tr_normal)
            random.shuffle(all_tr_tumor)
        splits.append({"normal": copy.deepcopy(all_tr_normal),
                       "tumor": copy.deepcopy(all_tr_tumor)}
                      )

    # Create k-folds for each split.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold contains a train, and valid set with a
        predefined size.

        Note: samples need to be shufled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): where each element is the list (str paths)
                 of the samples of each set: train, valid, and test, respectively.
        """
        assert len(lsamps) == s_tr + s_vl, "Something wrong with the provided sizes .... [NOT OK]"

        # chunk the data into chunks of size ts (the size of the test set), so we can rotate the test set.
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

    # Save the folds into *.csv files.
    def dump_fold_into_csv(lsamples, outpath):
        """
        Write a list of RELATIVE paths into a csv file.
        Relative paths allow running the code an any device.
        The absolute path within the device will be determined at the running time.

        csv file format: relative path to the image, relative path to the mask, class (str: benign, malignant).

        :param lsamples: list of str of relative paths.
        :param outpath: str, output file name.
        :return:
        """
        with open(outpath, 'w') as fcsv:
            filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for fname, clas, fxml in lsamples:
                filewriter.writerow([fname, clas, fxml])

    def create_one_split(split_i, test_samples, tr_normal, tr_tumor, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param tr_normal: list, list of train normal samples.
        :param tr_tumor: list, list of train tumor samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_normal = math.ceil(len(tr_normal) * args.folding["vl"] / 100.)
        vl_size_tumor = math.ceil(len(tr_tumor) * args.folding["vl"] / 100.)

        # Create the k-folds
        list_folds_normal = create_folds_of_one_class(tr_normal, len(tr_normal) - vl_size_normal, vl_size_normal)
        list_folds_tumor = create_folds_of_one_class(tr_tumor, len(tr_tumor) - vl_size_tumor, vl_size_tumor)

        assert len(list_folds_normal) == len(list_folds_tumor), "We didn't obtain the same number of fold" \
                                                                " .... [NOT OK]"
        assert len(list_folds_normal) == nbr_folds, "We didn't get `{}` folds, but `{}` .... [NOT OK]".format(
            nbr_folds, len(list_folds_normal))

        print("We obtained `{}` folds .... [OK]".format(args.nbr_folds))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold)

            # dump the test set
            dump_fold_into_csv(test_samples, join(out_fold, "test_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the train set
            train = list_folds_normal[i][0] + list_folds_tumor[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)
            dump_fold_into_csv(train, join(out_fold, "train_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the valid set
            valid = list_folds_normal[i][1] + list_folds_tumor[i][1]
            dump_fold_into_csv(valid, join(out_fold, "valid_s_" + str(split_i) + "_f_" + str(i) + ".csv"))

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])

            with open(join(outd, "split_" + str(split_i) + "/readme.md"), 'w') as fx:
                fx.write(
                    "csv format:\nrelative path to the image (str), class (str: normal, tumor), relative path to "
                    "the xml file if class is tumor, else ''.")

        print("CAMELYON16 WSI-LEVEL splitting N° `{}` ends with success .... [OK]".format(split_i))

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)
    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(i, all_test, splits[i]["normal"], splits[i]["tumor"], args.nbr_folds)

    print("All CAMELYON16 WSI-LEVEL splitting (`{}`) ended with success .... [OK]".format(args.nbr_splits))


# ========================================================================================
#                               RUN
# ========================================================================================


def do_bach_parta_2018():
    """
    BACH (PART A) 2018.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/ICIAR-2018-BACH-Challenge"
    elif username == "sbelharb":
        baseurl = "/project/6004986/sbelharb/workspace/datasets/ICIAR-2018-BACH-Challenge"
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "test_portion": 0.5,  # percentage of samples to take from test. The left over if for train; and it will
            # be divided into actual train, and validation sets.
            "folding": {"vl": 20},  # vl/100 % of train set will be used for validation, while the leftover (
            # 100-vl)/100% will be used for actual training.
            "name_classes": {'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3},
            "dataset": "bc18bch",
            "fold_folder": "folds/bach-part-a-2018",
            "img_extension": "tif",
            "nbr_folds": 5,
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    create_k_folds_csv_bach_part_a(Dict2Obj(args))


def do_breakhis():
    """
    BreakHis.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/" \
                  "Breast-Cancer-Histopathological-Database-BreakHis/mkfold"
    elif username == "sbelharb":
        baseurl = "/project/6004986/sbelharb/workspace/datasets/" \
                  "Breast-Cancer-Histopathological-Database-BreakHis/mkfold"
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80% for train, 20% for validation.
            "dataset": "breakhis",
            "fold_folder": "folds/breakhis",
            "img_extension": "png",
            "nbr_folds": 5,
            "magnification": ["40X", "100X", "200X", "400X"],
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    split_valid_breakhis(Dict2Obj(args))


def do_glas():
    """
    GlaS.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/GlaS-2015/Warwick QU Dataset (Released 2016_07_08)"
    elif username == "sbelharb":
        baseurl = "/project/6004986/sbelharb/workspace/datasets/GlaS-2015/Warwick QU Dataset (Released 2016_07_08)"
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "glas",
            "fold_folder": "folds/glas",
            "img_extension": "bmp",
            "nbr_folds": 5,
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    split_valid_glas(Dict2Obj(args))


def do_camelyon16_WSI_level():
    """
    GlaS.

    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/camelyon16"
    elif username == "sbelharb":
        baseurl = "/project/6004986/sbelharb/workspace/datasets/camelyon16"
    else:
        raise ValueError("username `{}` unknown .... [NOT OK]".format(username))

    args = {"baseurl": baseurl,
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "camelyon16",
            "fold_folder": "folds/camelyon16/WSI-level",
            "img_extension": "tif",
            "nbr_folds": 5,
            "nbr_splits": 2  # how many times to perform the k-folds over the available train samples.
            }
    split_valid_camelyon16_WSI_level(Dict2Obj(args))


if __name__ == "__main__":
    # ============== CREATE FOLDS OF BACH (PART A) 2018 DATASET
    do_bach_parta_2018()

    # ============== CREATE FOLDS OF GlaS DATASET
    do_glas()

    # ============== CREATE FOLDS OF CAMELYON16 DATASET (WSI-LEVEL)
    do_camelyon16_WSI_level()

    # ============== CREATE FOLDS OF BreakHis DATASET
    do_breakhis()
