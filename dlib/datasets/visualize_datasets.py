"""
Visualize some samples of the datasets (trainset by default).
sample 1 | sample 2 | sample 3 ... in the same image.

Datsets:
- glas
- Caltech_UCSD_Birds_200_2011
- cityscapes
"""

import os
import sys
from os.path import join
from os.path import dirname
from os.path import abspath

sys.path.append(dirname(dirname(abspath(__file__))))

from visu_sets.visu_tools import see_multi_class_ds
from visu_sets.visu_tools import see_multi_label_ds
from visu_sets.visu_tools import create_thumbnail


from shared import announce_msg

from dlib.utils.reproducibility import set_seed

import constants

SEED = 0


if __name__ == "__main__":
    set_seed(seed=SEED)
    root_main = dirname(dirname(abspath(__file__)))

    print("CWD: {}".format(os.getcwd()))
    print("root: {}".format(root_main))

    outdir = join(root_main, "visu_sets/output")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)


    todosets = [
        {
            "dataset": constants.GLAS,  # name of the dataset
            "nbr_samples": 5,  # how many sample to draw per class.
            "nbr_classes": 2  # how many classes to consider.
        },
        {
            "dataset": constants.CUB,  # name of the dataset
            "nbr_samples": 1,  # how many sample to draw per class.
            "nbr_classes": 10  # how many classes to consider.
        },
        {
            "dataset": constants.CSCAPES,  # name of the dataset
            "nbr_samples": 5,  # total number of samples to draw.
        }

    ]

    # generate the samples.
    outpaths = []
    for atom in todosets:
        reset_seed(SEED, check_cudnn=False)

        announce_msg("Processing dataset {}".format(atom['dataset']))

        if atom['dataset'] in constants.MULTI_LABEL_DATASETS:
            op = see_multi_label_ds(atom, outdir, root_main)
        else:
            op = see_multi_class_ds(atom, outdir, root_main)

        outpaths.append(op)

    # create panorama of all the samples.
    thumbnailpath = join(outdir, "samples-datasets.png")
    announce_msg("Create thumbnail {}".format(thumbnailpath))

    create_thumbnail(l_imgs=outpaths,
                     file_out=thumbnailpath,
                     scale=5
                     )


