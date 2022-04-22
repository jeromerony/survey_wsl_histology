import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.deeplabv3.model import DeepLabV3, DeepLabV3Plus
