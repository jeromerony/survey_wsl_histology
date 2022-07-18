import sys
from os.path import join, dirname, abspath, basename
import random
import os
from copy import deepcopy


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
assert not constants.DEBUG


from dlib.datasets.wsol_loader import get_class_labels

from dlib.configure.config import get_config
from dlib.datasets.wsol_loader import configure_metadata
from dlib.utils.tools import chunks_into_n

from dlib.utils.reproducibility import set_seed

_SPLITS = [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]
_PROXY = False

_SPLIT_TRAIN_INTO_CHUNCKS_ILSVRC = True

_NBR_CHUNKS_ILSVRC_TRAIN = constants.NBR_CHUNKS_TR_ILSVRC[
    constants.FORMAT_DEBUG.format(constants.ILSVRC)]

_NBR_S_PER_CLASS = {
    constants.CUB: 1,
    constants.OpenImages: 1,
    constants.ILSVRC: 130
}

_KEEP_FULL_VAL_TST = {
    constants.CUB: False,
    constants.OpenImages: False,
    constants.ILSVRC: True
}


def dump_image_ids(s_ids: dict, old_meta, new_meta):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    suffix = '_proxy' if _PROXY else ''
    infile = open(old_meta['image_ids' + suffix], 'r')
    outfile = open(new_meta['image_ids' + suffix], 'w')

    for line in infile.readlines():
        line_id = line.strip('\n')
        if line_id in s_ids:
            outfile.write(line)

    infile.close()
    outfile.close()


def dump_chunk_i_image_ids(s_ids: dict, old_meta, new_meta, chunk_i: int):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    suffix = '_proxy' if _PROXY else ''
    infile = open(old_meta['image_ids' + suffix], 'r')

    path_out = new_meta['image_ids' + suffix]
    bname = basename(path_out)
    assert bname == 'image_ids.txt'
    bname = f'image_ids_{chunk_i}.txt'
    path_out = join(dirname(path_out), bname)
    outfile = open(path_out, 'w')

    for line in infile.readlines():
        line_id = line.strip('\n')
        if line_id in s_ids:
            outfile.write(line)

    infile.close()
    outfile.close()


def dump_class_labels(s_ids: dict, old_meta, new_meta):
    """
        image_ids.txt has the structure

        <path>,<integer_class_label>
        path/to/image1.jpg,0
        path/to/image2.jpg,1
        path/to/image3.jpg,1
        ...
    """
    infile = open(old_meta.class_labels, 'r')
    outfile = open(new_meta.class_labels, 'w')

    for line in infile.readlines():
        image_id, class_label_string = line.strip('\n').split(',')
        if image_id in s_ids.keys():
            outfile.write(line)

    infile.close()
    outfile.close()


def dump_image_sizes(s_ids: dict, old_meta, new_meta):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    infile = open(old_meta.image_sizes, 'r')
    outfile = open(new_meta.image_sizes, 'w')

    for line in infile.readlines():
        image_id, ws, hs = line.strip('\n').split(',')
        if image_id in s_ids.keys():
            outfile.write(line)

    infile.close()
    outfile.close()


def dump_bounding_boxes(s_ids: dict, old_meta, new_meta):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    infile = open(old_meta.localization, 'r')
    outfile = open(new_meta.localization, 'w')

    for line in infile.readlines():
        image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
        if image_id in s_ids.keys():
            outfile.write(line)

    infile.close()
    outfile.close()


def dump_mask_paths(s_ids: dict, old_meta, new_meta):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    infile = open(old_meta.localization, 'r')
    outfile = open(new_meta.localization, 'w')

    for line in infile.readlines():
        image_id, mask_path, ignore_path = line.strip('\n').split(',')
        if image_id in s_ids.keys():
            outfile.write(line)

    infile.close()
    outfile.close()


def make_set(ds):
    nbr = _NBR_S_PER_CLASS[ds]
    print('Process {} dataset. with {} per class.'.format(ds, nbr))

    config = get_config(ds)
    outd_ds = join(root_dir, constants.RELATIVE_META_ROOT,
                   constants.FORMAT_DEBUG.format(ds))
    outd_ds = os.path.normpath(outd_ds)

    for split in _SPLITS:
        metadata = configure_metadata(
            join(root_dir, config['metadata_root'], split))
        new_metadata = configure_metadata(join(outd_ds, split))
        parent_d = join(outd_ds, split)
        if not os.path.isdir(parent_d):
            os.makedirs(parent_d)

        img_labels = get_class_labels(metadata)
        unique_labels = set([img_labels[k] for k in img_labels.keys()])

        skip_split = False

        if _KEEP_FULL_VAL_TST[ds] and split in [constants.VALIDSET,
                                                constants.TESTSET]:
            skip_split = True

        if skip_split:
            input('Going to take all samples in split {} from dataset '
                  '{}: [hit enter]'.format(split, ds))

        if skip_split:
            l_ids = [k for k in img_labels.keys()]
        else:
            l_ids = []
            for i, cl in enumerate(unique_labels):
                ids_cl = [k for k in img_labels.keys() if img_labels[k] == cl]
                set_seed(i, verbose=False)
                for t in range(100):
                    random.shuffle(ids_cl)
                set_seed(i, verbose=False)
                l_ids.extend(ids_cl[:nbr])

        ratio = 100. * len(l_ids) / float(len(list(img_labels.keys())))
        print('Dataset {} split {} will have {} / {} [{:.3f}%] samples.'.format(
            ds, split, len(l_ids), len(list(img_labels.keys())), ratio))

        assert len(set(l_ids)) == len(l_ids)
        s_ids_dict = {k: None for k in l_ids}

        dump_image_ids(s_ids_dict, old_meta=metadata, new_meta=new_metadata)
        dump_class_labels(s_ids_dict, old_meta=metadata, new_meta=new_metadata)
        dump_image_sizes(s_ids_dict, old_meta=metadata, new_meta=new_metadata)
        if ds == constants.OpenImages:
            dump_mask_paths(s_ids_dict, old_meta=metadata,
                            new_meta=new_metadata)
        elif ds in [constants.ILSVRC, constants.CUB]:
            dump_bounding_boxes(s_ids_dict, old_meta=metadata,
                                new_meta=new_metadata)
        else:
            raise NotImplementedError

        cnd = split == constants.TRAINSET
        cnd &= ds == constants.ILSVRC
        cnd &= _SPLIT_TRAIN_INTO_CHUNCKS_ILSVRC

        if cnd:
            print(f'Chunking {ds} split {split} into'
                  f' {_NBR_CHUNKS_ILSVRC_TRAIN}')
            _lids = deepcopy(l_ids)
            for t in range(10000):
                random.shuffle(_lids)

            chunks_ids = chunks_into_n(_lids, _NBR_CHUNKS_ILSVRC_TRAIN)

            for zz, c_ids in enumerate(chunks_ids):
                dump_chunk_i_image_ids(
                    s_ids={k: None for k in c_ids}, old_meta=metadata,
                    new_meta=new_metadata, chunk_i=zz)

    print('Done process {} dataset. with {} per class.'.format(ds, nbr))


def make_debug_sets():
    print('**BUILDING DEBUG DATASETS**')

    # make_set(constants.CUB)
    # make_set(constants.OpenImages)
    make_set(constants.ILSVRC)


if __name__ == '__main__':
    make_debug_sets()
