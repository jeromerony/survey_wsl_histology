from os.path import join
from utils import csv_reader


def get_files(folds_dir, size, split, fold):
    splits = ['train', 'valid', 'test']
    csv_dir = join(folds_dir, 'w-{}xh-{}'.format(size, size), 'split_{}'.format(split), 'fold_{}'.format(fold))
    csv_files = [join(csv_dir, '{}_s_{}_f_{}.csv'.format(s, split, fold)) for s in splits]
    split_files = [csv_reader(csv) for csv in csv_files]
    return split_files


def decode_classes(files: list) -> list:
    classes = {'normal': 0, 'tumor': 1}
    files_decoded_classes = []
    for f in files:
        class_name = f[2]
        files_decoded_classes.append((f[0], f[1], classes[class_name]))

    return files_decoded_classes