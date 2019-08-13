import os
import warnings


def check_files(path: str, files: list) -> list:
    if not os.path.isdir(path):
        raise NotADirectoryError('{} is not present.'.format(path))

    found_files = []
    for file_path, label in files:
        full_path = os.path.join(path, file_path)
        if os.path.isfile(full_path):
            found_files.append((full_path, label))

    if len(found_files) != len(files):
        warnings.warn('Only {} image files found out of the {} provided.'.format(len(found_files), len(files)))

    return found_files
