import os
from os.path import join
import sys
from os.path import dirname
from os.path import abspath
import fnmatch

import matplotlib.pyplot as plt
from PIL import Image
import tqdm

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import get_rootpath_2_dataset

from dlib.datasets.tools import csv_loader
from dlib.utils.shared import announce_msg


__all__ = [
    "show_msg",
    "get_stats"
]


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist " \
                                   ".... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def show_msg(ms, lg):
    announce_msg(ms)
    lg.write(ms + "\n")


def get_stats(args, split, fold, subset):
    """
    Get some stats on the image sizes of specific dataset, split, fold.
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder)

    tag = "ds-{}-s-{}-f-{}-subset-{}".format(args.dataset,
                                             split,
                                             fold,
                                             subset
                                             )
    log = open(join(
        args.fold_folder, "log-stats-ds-{}.txt".format(tag)), 'w')
    announce_msg("Going to check {}".format(args.dataset.upper()))

    relative_fold_path = join(args.fold_folder,
                              "split_{}".format(split),
                              "fold_{}".format(fold)
                              )

    subset_csv = join(relative_fold_path,
                      "{}_s_{}_f_{}.csv".format(subset, split, fold)
                      )
    rootpath = get_rootpath_2_dataset(args)
    samples = csv_loader(subset_csv, rootpath)

    lh, lw = [], []
    for el in tqdm.tqdm(samples, ncols=150, total=len(samples)):
        img = Image.open(el[1], 'r').convert('RGB')
        w, h = img.size
        lh.append(h)
        lw.append(w)

    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(join(args.fold_folder, "size-stats-{}.png".format(tag)))

    log.close()
