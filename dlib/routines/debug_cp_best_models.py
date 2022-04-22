import sys
from os.path import dirname, abspath, join
import os
import subprocess

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import pickle as pkl

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import find_files_pattern


def move_folders():
    fd = join(root_dir, 'exps')
    l_files = find_files_pattern(fd, 'performance_log_best.pickle')
    distin = join(root_dir, 'pretrained')
    for f in l_files:
        for r, d, file in os.walk(dirname(f)):
            for fd in d:
                if fd.startswith('DEBUG_'):
                    cmd = 'cp -r {} {}'.format(join(dirname(f), fd), distin)
                    print(cmd)
                    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    raise ValueError('YOU CANT RUN THIS unless you are SUP and know what you '
                     'are doing.')
    move_folders()
