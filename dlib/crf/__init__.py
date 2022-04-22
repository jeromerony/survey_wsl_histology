import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from dlib.crf.dense_crf_loss import DenseCRFLoss
from dlib.crf.color_dense_crf_loss import ColorDenseCRFLoss
