import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from dlib.stdcl.classifier import STDClassifier
from dlib.stdcl.maxmin_classifier import MaxMinClassifier
