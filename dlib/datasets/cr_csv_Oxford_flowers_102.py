"""
Create csv file of Oxford-flowers-102 dataset.
"""
import random
import csv
import sys
import os
from os import path
from os.path import join, relpath, basename, splitext, isfile

import numpy as np
import yaml
from scipy.io import loadmat
import tqdm
from PIL import Image
from PIL import ImageChops

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from create_folds.csv_tools import get_stats
from create_folds.csv_tools import find_files_pattern

from tools import Dict2Obj
from tools import get_rootpath_2_dataset

from shared import announce_msg

import constants

from dlib.utils import reproducibility

__all__ = ["do_Oxford_flowers_102"]


def dump_fold_into_csv_OXF(lsamples, outpath, tag):
    """
    for Oxford_flowers_102 dataset.
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

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

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


def al_split_Oxford_flowers_102(args):
    """
    Use the provided split:
    http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
    for active learning.

    :param args:
    :return:
    """
    baseurl = args.baseurl

    # splits
    fin = loadmat(join(baseurl, 'setid.mat'))
    trnid = fin['trnid'].reshape((-1)).astype(np.uint16)
    valid = fin['valid'].reshape((-1)).astype(np.uint16)
    tstid = fin['tstid'].reshape((-1)).astype(np.uint16)

    # labels
    flabels = loadmat(join(baseurl, 'imagelabels.mat'))['labels'].flatten()
    flabels -= 1  # labels are encoded from 1 to 102. We change that to
    # be from 0 to 101.

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []  # (img, mask, label (int))
    filesin = find_files_pattern(fdimg, '*.jpg')
    lid = []
    idcnt = 0.  # count the unique id for each sample
    for f in filesin:
        rpath = relpath(f, baseurl)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        mask = join(baseurl, 'segmim_bin', 'segmim_{}.jpg'.format(id))
        msg = 'File {} does not exist. Inconsistent logic. ' \
              '.... [NOT OK]'.format(mask)
        assert isfile(mask), msg
        rpath_mask = relpath(mask, baseurl)
        id = int(id)  # ids start from 1. Array indexing starts from 0.
        label = int(flabels[id - 1])
        sample = (rpath, rpath_mask, label, idcnt)
        lid.append(id)
        if id in trnid:
            tr_set.append(sample)
        elif id in valid:
            vl_set.append(sample)
        elif id in tstid:
            ts_set.append(sample)
        else:
            raise ValueError('ID:{} not found in train, valid, test. '
                             'Inconsistent logic. ....[NOT OK]'.format(id))

        idcnt += 1.

    dict_classes_names = dict()
    uniquel = np.unique(flabels)
    for i in range(uniquel.size):
        dict_classes_names[str(uniquel[i])] = int(uniquel[i])

    outd = args.fold_folder
    out_fold = join(outd, "split_{}/fold_{}".format(0, 0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # shuffle train
    for t in range(1000):
        random.shuffle(tr_set)

    dump_fold_into_csv_OXF(tr_set,
                           join(out_fold, "train_s_{}_f_{}.csv".format(0, 0)),
                           constants.U
                           )
    dump_fold_into_csv_OXF(vl_set,
                           join(out_fold, "valid_s_{}_f_{}.csv".format(0, 0)),
                           constants.L
                           )
    dump_fold_into_csv_OXF(ts_set,
                           join(out_fold, "test_s_{}_f_{}.csv".format(0, 0)),
                           constants.L
                           )

    # current fold.

    # dump the coding.
    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # dump the seed
    with open(join(out_fold, "seed.txt"), 'w') as fx:
        fx.write("MYSEED: " + os.environ["MYSEED"])

    with open(join(out_fold, "readme.md"), 'w') as fx:
        fx.write(readme)

    # folder of folds

    # readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print(
        "Oxford_flowers_102 splitting (`{}`) ended with "
        "success .... [OK]".format(0))


def create_bin_mask_Oxford_flowers_102(args):
    """
    Create binary masks.
    :param args:
    :return:
    """
    def get_id(pathx, basex):
        """
        Get the id of a sample.
        :param pathx:
        :return:
        """
        rpath = relpath(pathx, basex)
        basen = basename(rpath)
        id = splitext(basen)[0].split('_')[1]
        return id

    baseurl = args.baseurl
    imgs = find_files_pattern(join(baseurl, 'jpg'), '*.jpg')
    bin_fd = join(baseurl, 'segmim_bin')
    if not os.path.exists(bin_fd):
        os.makedirs(bin_fd)
    else:  # End.
        print('Conversion to binary mask has already been done. [OK]')
        return 0

    # Background color [  0   0 254]. (blue)
    print('Start converting the provided masks into binary masks ....')
    for im in tqdm.tqdm(imgs, ncols=80, total=len(imgs)):
        id_im = get_id(im, baseurl)
        mask = join(baseurl, 'segmim', 'segmim_{}.jpg'.format(id_im))
        assert isfile(mask), 'File {} does not exist. Inconsistent logic. .... [NOT OK]'.format(mask)
        msk_in = Image.open(mask, 'r').convert('RGB')
        arr_ = np.array(msk_in)
        arr_[:, :, 0] = 0
        arr_[:, :, 1] = 0
        arr_[:, :, 2] = 254
        blue = Image.fromarray(arr_.astype(np.uint8), mode='RGB')
        dif = ImageChops.subtract(msk_in, blue)
        x_arr = np.array(dif)
        x_arr = np.mean(x_arr, axis=2)
        x_arr = (x_arr != 0).astype(np.uint8)
        img_bin = Image.fromarray(x_arr * 255, mode='L')
        img_bin.save(join(bin_fd, 'segmim_{}.jpg'.format(id_im)), 'JPEG')


def do_Oxford_flowers_102(root_main):
    """
    Oxford-flowers-102.
    The train/valid/test sets are already provided.

    :param root_main: str. absolute path to folder containing main.py.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_default_seed()

    # ===========================

    announce_msg("Processing dataset: {}".format(constants.OXF))
    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.OXF})),
            "dataset": "Oxford-flowers-102",
            "fold_folder": join(root_main, "folds/Oxford-flowers-102"),
            "img_extension": "jpg",
            "path_encoding": join(
                root_main, "folds/Oxford-flowers-102/encoding-origine.yaml")
            }
    # Convert masks into binary masks: already done.
    # create_bin_mask_Oxford_flowers_102(Dict2Obj(args))
    reproducibility.set_default_seed()
    al_split_Oxford_flowers_102(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')