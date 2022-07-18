"""
Create cvs folds for all the datasets.
- glas
- Caltech_UCSD_Birds_200_2011
- Cityscapes
"""
import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.div_glas import do_glas
from dlib.datasets.div_camelyon16 import do_camelyon16
from dlib.datasets.div_iciar import do_iciar
from dlib.datasets.div_breakhis import do_breakhis


from dlib.utils.reproducibility import set_seed

if __name__ == "__main__":
    seed = 0
    set_seed(seed=seed, verbose=False)

    # ==========================================================================
    #                             START
    # ==========================================================================
    do_glas(root_main=root_dir, seed=seed)
    do_camelyon16(root_main=root_dir, seed=seed, psize=512)
    do_iciar(root_main=root_dir, seed=seed)
    do_breakhis(root_main=root_dir, seed=0, mag='40X')
    do_breakhis(root_main=root_dir, seed=0, mag='100X')
    do_breakhis(root_main=root_dir, seed=0, mag='200X')
    do_breakhis(root_main=root_dir, seed=0, mag='400X')

    # ==========================================================================
    #                             END
    # ==========================================================================
