import os
import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple
import numbers
from collections.abc import Sequence
from tqdm import tqdm

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.configure import config
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets import wsol_loader


def check_files_exist():
    assert not constants.DEBUG
    args = config.get_config(constants.ILSVRC)

    # os.env['TMP'] hold a tmp path to ilsvrc.
    # tod: remove this line for publication.
    args['data_root'] = os.environ['TMP']

    for split in wsol_loader._SPLITS:
        print(f'Inspecting split: {split}')
        path = os.path.normpath(join(root_dir, args['metadata_root'], split))
        meta = configure_metadata(path)
        ids = wsol_loader.get_image_ids(metadata=meta, proxy=False)

        missed = []
        for id in tqdm(ids, ncols=80, total=len(ids)):
            pathimg = join(args['data_root'], constants.ILSVRC, id)

            if not os.path.isfile(pathimg):
                missed.append(pathimg)

        print(f'Split: {split}.  Found {len(missed)} missed images')


if __name__ == '__main__':
    check_files_exist()
