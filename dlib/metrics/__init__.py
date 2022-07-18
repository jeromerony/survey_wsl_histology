import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics.core import ClMetrics
from dlib.metrics.core import SegMetric
