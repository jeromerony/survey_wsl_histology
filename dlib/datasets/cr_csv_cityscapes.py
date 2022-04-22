"""
Create csv file of cityscapes dataset.
"""
import math
import random
import copy
import csv
import sys
import os
from os.path import join
from os.path import dirname
from os.path import abspath
from collections import Counter

import numpy as np
import yaml
from PIL import Image
import tqdm

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.csv_tools import find_files_pattern
from dlib.datasets.csv_tools import show_msg
from dlib.datasets.csv_tools import get_stats
from dlib.datasets import cityscapes_labels as cs_labels


from dlib.datasets.csv_tools import get_stats

from dlib.utils.tools import chunk_it
from dlib.utils.tools import Dict2Obj
from dlib.datasets.tools import get_rootpath_2_dataset

from dlib.utils.shared import announce_msg

from dlib.configure import constants

from dlib.utils.reproducibility import set_default_seed

__all__ = [
    "do_cityscapes",
    "encode_meta_cityscapes"
]


def dump_fold_into_csv_CSCAPES(lsamples, outpath, tag):
    """
    for cityscapes dataset.
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


def create_folds(lsamps, s_tr, s_vl):
    """
    Create k folds from a list of samples, each fold
     contains a train, and valid set with a predefined size.

    Note: Samples are expected to be shuffled beforehand.

    :param lsamps: list of samples.
    :param s_tr: int, number of samples in the train set.
    :param s_vl: int, number of samples in the valid set.
    :return: list_folds: list of k tuples (tr_set, vl_set): where each
             element is the list of the samples of each set: train, and
             valid, respectively.
    """
    assert len(lsamps) == s_tr + s_vl, "Something wrong with the" \
                                       " provided sizes."

    # chunk the data into chunks of size vl (the size of the vl set),
    # so we can rotate the vl set.
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


def get_labels_from_mask(path_mask):
    """
    Return list of the labels in the mask coded using our mapping.
    :param path_mask: str. absolute path to the mask.
    :return: list of labels.
    """
    unique_original_codes = np.unique(Image.open(path_mask)).tolist()

    # our encoding
    labels = [cs_labels.mapping_20[i] for i in unique_original_codes]
    # compress to unique labels.
    labels = sorted(np.unique(np.array(labels)).tolist())

    msg = "ERROR: mask {} has {} labels".format(path_mask, len(labels))
    assert len(labels) > 0, msg

    return labels


def filter_out_empty_samples(l_cities, baseurl):
    """
    Removes samples that are left only with Ignore class.
    this could happen if we reduce the encoding to 2 classes for instance.
    if only Ignore class is present, the sample is useless and should not
    count in the dataset (must be removed).

    :param l_cities: list of str. relative path to the cities.
    :return: 2 lists:
    - list of valid cities. a valid city is a city that contains at lean one
    valid sample.
    - list of str, list of banned cities (wities where all samples are banned)
    - list of str, list of banned samples.
    """
    l_banned_cities = []
    l_valid_cities = []
    l_banned_imgs = []

    for city in l_cities:
        spl, bn = create_samples_for_city(city_img=city,
                                           baseurl=baseurl,
                                           allids=None
                                          )

        l_banned_imgs.extend(bn)

        if not spl:
            l_banned_cities.append(city)
        else:
            l_valid_cities.append(city)

    return l_valid_cities, l_banned_cities, l_banned_imgs


def create_samples_for_city(city_img, baseurl, allids=None):
    """
    Create list of samples for a list of cities of images.
    for each sample:
    - get the path of the image
    - get the path of the mask (gtFine_labelIds)
    - get the list of global labels.

    :param city_img: partial path to a city (images folder).
    :param baseurl: str. baseurl of the dataset.
    :param allids: dict of the unique ids. each key if the absolute path to
    the image. the value is the id (float). or None.
    :return: - list of valid samples of tuple
            (path_img_relative,
             path_mask_relative,
             labels,
             id or None
             )
             - list of banned samples (they contain only the Ignore class).
             Samples are absolute path to the image.
    """
    samples = []
    banned_samples = []
    announce_msg("Processing city {}".format(city_img))

    cctiy_img = join(baseurl, city_img)
    ccpaths_img = find_files_pattern(cctiy_img, "*.png")

    if not ccpaths_img:
        return samples

    for path_img in tqdm.tqdm(ccpaths_img, ncols=150, total=len(ccpaths_img)):
        # mask
        path_mask = path_img.replace("leftImg8bit", "gtFine")
        path_mask = path_mask.replace(".png", "_labelIds.png")
        msg = "{} does not exist".format(path_mask)
        assert os.path.isfile(path_mask), msg

        # labels: find all global labels in the image through the  mask.
        labels = get_labels_from_mask(path_mask)
        labels = ";".join([str(kk) for kk in labels])

        # check if sample contains labels more than Ignore class.
        if labels == "0":  # ban sample because it contains only the Ignore cl.
            banned_samples.append(path_img)
            continue

        # check of this sample

        # id
        cid = allids[path_img] if allids is not None else None

        # append the sample
        path_img_relative = join(*path_img.split(os.sep)[-4:])
        path_mask_relative = join(*path_mask.split(os.sep)[-4:])
        samples.append(
            (path_img_relative,
             path_mask_relative,
             labels,
             cid
             )
        )

    return samples, banned_samples


def iou_sets(a, b):
    """
    Computes Intersection over Union between 2 sets a, b.

    :param a: set.
    :param b: set.
    :return: float. IOU metric.
    """
    return len(a.intersection(b))  / float(len(a.union(b)))


def inter_sets(a, b):
    """
    Computes cardinal Intersection between 2 sets a, b.

    :param a: set.
    :param b: set.
    :return: float. cardinal of intersection between a, b.
    """
    return float(len(a.intersection(b)))


def encode_meta_cityscapes(code):
    """
    Re-encode the label using meta-label.
    code example:
    "0;3;4;7".
    meta-label:
    "0"
    The meta-label is the label of the `closest` set among all the met-labels
    below using IOU metric:
    argmax_j (|code inter meta-label-set_j| / |code union meta-label-set_j|)
    to pick the meta-label that has a code with maximum IOU with the
    our code.

    Due to the division by |code union meta-label-set_j|, this function will
    tend to generate the meta-label with the smallest typical multi-label set
    cardinal.

    All meta-labels:
    meta-label : typical multi-label set
    0: 0
    1: 0;1
    2: 0;1;2
    3: 0;1;2;3
    4: 0;1;2;3;4
    5: 0;1;2;3;4;5
    6: 0;1;2;3;4;5;6
    7: 0;1;2;3;4;5;6;7
    8: 0;1;2;3;4;5;6;7;8
    ...


    :param code: str. code.
    :return: str. meta-label.
    """
    meta_labels = cs_labels.meta_labels

    a = set(code.split(";"))

    mtr = {
        k: iou_sets(a, set(meta_labels[k].split(";"))
                      ) for k in meta_labels.keys()
    }

    # return the FIRST meta-label with  the maximum mtr.
    return max(mtr, key=lambda key: mtr[key])


def meta(l_samples):
    """
    Do some stats on the labels for multi-label case.

    Intended for trainset.

    :param l_samples: list of samples.
    :return:
    """
    l_meta_labels = [encode_meta_cityscapes(e[2]) for e in l_samples]
    stats = Counter(l_meta_labels)

    return stats


def split_cityscapes(args):
    """
    Splits cityscapes dataset for active learning.

    Since the annotation of the true test set of the competition is not
    public, we consider the folowing scenario:
    - original test set is not used at all.
    - our train, valid, test sets:
        - the test set is fixed and it is the validation set provided in the
        competition (3 cities).
        - the original train set is divided into train and validation sets
        based on cities following the competition rules. cities are exclusive
        (i.e. the ones used for train can not be used for validation). the
        same rule is applied between all the sets (train, valid, test). from
        the original train set, we take 20% of the cities for validation (4
        cities),  and the rest 80% are for training (14 cities).
        - 1 split only. the number of folds is determined using the
        percentage of cities for train/valid.

    :param args:
    :return:
    """
    l_all_banned_cities = []
    l_all_banned_samples = []

    baseurl = args.baseurl
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    log = open(join(args.fold_folder, "full-log.txt"), "w")

    # test cities: are the original validation cities: 3 cities.
    cities_ts = [
        join("leftImg8bit/val", name) for name in os.listdir(
            join(baseurl, "leftImg8bit/val"))
        if os.path.isdir(join(baseurl, "leftImg8bit/val", name))
    ]
    #  BAN CHECK ===============================================================
    cities_ts, bn_c_tst, bn_s_tst = filter_out_empty_samples(
        l_cities=cities_ts,
        baseurl=baseurl
    )
    l_all_banned_cities.extend(bn_c_tst)
    l_all_banned_samples.extend(bn_s_tst)

    msg = "Nbr cities for test set: {}\n" \
          "Nbr banned cities: {}\n" \
          "Nbr banned samples: {}\n".format(len(cities_ts),
                                            len(bn_c_tst),
                                            len(bn_s_tst)
                                            )
    sum_log = msg
    show_msg(ms=msg, lg=log)

    # train + valid cities
    tv_cities = [
        join("leftImg8bit/train", name) for name in os.listdir(
            join(baseurl, "leftImg8bit/train"))
        if os.path.isdir(join(baseurl, "leftImg8bit/train", name))
    ]
    # sort to avoid platform dependency.
    tv_cities = sorted(tv_cities)
    #  BAN CHECK ===============================================================
    tv_cities, bn_c_tv, bn_s_tv = filter_out_empty_samples(
        l_cities=tv_cities,
        baseurl=baseurl
    )
    l_all_banned_cities.extend(bn_c_tv)
    l_all_banned_samples.extend(bn_s_tv)

    # shuffle the cities well.
    for i in range(1000):
        random.shuffle(tv_cities)

    vl_size = math.ceil(len(tv_cities) * args.folding["vl"] / 100.)
    tr_size = len(tv_cities) - vl_size

    msg = "nbr train cities {}. vl cities {}\n" \
          "Nbr cities banned: {}\n" \
          "Nbr samples banned: {}\n".format(
        tr_size,
        vl_size,
        len(bn_c_tv),
        len(bn_s_tv)
    )
    sum_log = "{}\n{}".format(sum_log, msg)
    show_msg(ms=msg, lg=log)

    list_folds_cities = create_folds(lsamps=tv_cities,
                                     s_tr=tr_size,
                                     s_vl=vl_size
                                     )

    # store the folds.
    nbr_folds = len(list_folds_cities)
    outd = args.fold_folder

    # create unique ids for ALL the samples
    all_ids = dict()
    cities = cities_ts + tv_cities
    idcnt = 0.
    for city in cities:

        if city in l_all_banned_cities:  # not possible.
            continue

        cctiy = join(baseurl, city)
        ccpaths = find_files_pattern(cctiy, "*.png")
        if ccpaths:
            for p in ccpaths:
                if p in l_all_banned_samples:  # possible.
                    continue

                assert p not in all_ids, "ERROR DUPLICATED SAMPLE"
                all_ids[p] = idcnt
                idcnt += 1.

    msg = "TOTAL samples: {}".format(len(list(all_ids.keys())))
    sum_log = "{}\n{}".format(sum_log, msg)
    show_msg(ms=msg, lg=log)

    # gather samples for all cities
    allsamples = dict()  # holds samples of each city. city is key.
    for city in cities:
        # banned samples are automatically ignored in create_samples_for_city()
        # cities are already filter out.
        csamples, _ = create_samples_for_city(city_img=city,
                                              baseurl=baseurl,
                                              allids=all_ids
                                              )
        allsamples[city] = copy.deepcopy(csamples)

    test_samples = []
    for city in cities_ts:
        test_samples.extend(allsamples[city])

    readme = "Format: float `id`: 0, str `img`: 1, str `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    id2name = cs_labels.name_classes_rever_mapping_20
    dict_classes_names = {v: k for k, v in id2name.items()}

    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    meta_labels = cs_labels.meta_labels

    for i in range(nbr_folds):
        print("\t Fold: {}".format(i))
        out_fold = join(outd, "split_0/fold_{}".format(i))
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)

        # dump the test set
        dump_fold_into_csv_CSCAPES(
            test_samples,
            join(out_fold, "test_s_0_f_{}.csv".format(i)),
            constants.L
        )

        # dump the train set
        traincities = list_folds_cities[i][0]

        train_samples = []
        for city in traincities:
            train_samples.extend(allsamples[city])

        # shuffle
        for t in range(10000):
            random.shuffle(train_samples)

        dump_fold_into_csv_CSCAPES(
            train_samples,
            join(out_fold, "train_s_0_f_{}.csv".format(i)),
            constants.L
        )

        # dump the valid set
        validcities = list_folds_cities[i][1]

        valid_samples = []
        for city in validcities:
            valid_samples.extend(allsamples[city])

        dump_fold_into_csv_CSCAPES(
            valid_samples,
            join(out_fold, "valid_s_0_f_{}.csv".format(i)),
            constants.L
        )

        # dump the seed
        with open(join(out_fold, "seed.txt"), 'w') as fx:
            fx.write("MYSEED: " + os.environ["MYSEED"])
        # dump the coding.
        with open(join(out_fold, "encoding.yaml"), 'w') as f:
            yaml.dump(dict_classes_names, f)

        # readme
        with open(join(out_fold, "readme.md"), 'w') as fx:
            fx.write(readme)

        # stats
        with open(join(out_fold, "stats-sets.txt"), 'w') as fx:
            # top
            msg = "{}\nSplit: 0, fold: {} \n{}\n".format(80 * "=", i, 80 * "=")
            fx.write(msg)
            sum_log = "{}\n{}".format(sum_log, msg)
            show_msg(ms=msg, lg=log)
            meta_train = meta(train_samples)

            # train
            msg = "Train: " \
                  "\nnbr cities: {} " \
                  "\nnbr samples: {} " \
                  "\ncities: {} " \
                  "\nMeta: {} meta-labels\n {} \n".format(
                len(traincities),
                len(train_samples),
                traincities,
                len(list(meta_train.keys())),
                "\n".join(["{}: {} \t {} ( {}%)".format(
                    k, meta_labels[k], meta_train[k],
                    100 * meta_train[k] / float(len(train_samples))) for k in
                           meta_train.keys()]
                          )
            )
            fx.write(msg)
            sum_log = "{}\n{}".format(sum_log, msg)
            show_msg(ms=msg, lg=log)
            # valid
            msg = "Valid: " \
                  "\nnbr cities: {} " \
                  "\nnbr samples: {} " \
                  "\ncities: {} \n".format(
                len(validcities),
                len(valid_samples),
                validcities
            )
            fx.write(msg)
            sum_log = "{}\n{}".format(sum_log, msg)
            show_msg(ms=msg, lg=log)
            # test
            msg = "Test: " \
                  "\nnbr cities: {} " \
                  "\nnbr samples: {} " \
                  "\ncities: {} \n".format(
                len(cities_ts),
                len(test_samples),
                cities_ts
            )
            fx.write(msg)
            sum_log = "{}\n{}".format(sum_log, msg)
            show_msg(ms=msg, lg=log)


    log.close()
    print(sum_log)
    print("All cityscapes splitting ended with success .... [OK]")


def do_cityscapes(root_main):
    """
    GlaS.

    :param root_main: str. absolute path to folder containing main.py.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============
    set_default_seed()

    announce_msg("Processing dataset: {}".format(constants.CSCAPES))

    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.CSCAPES})),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": constants.CSCAPES,
            "fold_folder": join(root_main,
                                "folds/{}".format(constants.CSCAPES)
                                ),
            "img_extension": "bmp"
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    set_default_seed()
    split_cityscapes(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')