from os.path import join
from utils import csv_reader


def get_files(folds_dir, split, fold, zoom):
    splits = ['train', 'valid', 'test']
    csv_dir = join(folds_dir, 'split_{}'.format(split), 'fold_{}'.format(fold), '{}X'.format(zoom))
    csv_files = [join(csv_dir, '{}_s_{}_f_{}_mag_{}X.csv'.format(s, split, fold, zoom)) for s in splits]
    split_files = [csv_reader(csv) for csv in csv_files]
    return split_files


def decode_classes(files: list) -> list:
    classes = {'B': 0, 'M': 1}
    files_decoded_classes = []
    for f in files:
        class_name = f[0].split('_')[1]
        files_decoded_classes.append((f[0], classes[class_name]))

    return files_decoded_classes
