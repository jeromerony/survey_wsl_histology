import os
import warnings
from functools import partial
from multiprocessing import Pool


def check_file(f, path):
    file_path, mask_path, label = f
    file_full_path = os.path.join(path, file_path)

    full_mask_path = ''
    if mask_path != '':
        full_mask_path = os.path.join(path, mask_path)
        mask_found = os.path.isfile(full_mask_path)

    if os.path.isfile(file_full_path) and (mask_path == '' or mask_found):
        return file_full_path, full_mask_path, label
    else:
        return None


def check_files(path: str, files: list) -> list:
    if not os.path.isdir(path):
        raise NotADirectoryError('{} is not present.'.format(path))

    check_file_partial = partial(check_file, path=path)
    with Pool(4) as p:
        found_files = p.map(check_file_partial, files)
    found_files = list(filter(lambda x: x is not None, found_files))

    if len(found_files) != len(files):
        warnings.warn('Only {} image files found out of the {} provided.'.format(len(found_files), len(files)))

    return found_files
