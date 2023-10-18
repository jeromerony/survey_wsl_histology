# divers classifiers.
import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from dlib.div_classifiers.resnet import ResNet50Spg
from dlib.div_classifiers.resnet import ResNet50Adl
from dlib.div_classifiers.resnet import ResNet50Acol

from dlib.div_classifiers.vgg import Vgg16Spg
from dlib.div_classifiers.vgg import Vgg16Adl
from dlib.div_classifiers.vgg import Vgg16Acol

from dlib.div_classifiers.inception import InceptionV3Spg
from dlib.div_classifiers.inception import InceptionV3Adl
from dlib.div_classifiers.inception import InceptionV3Acol

from dlib.div_classifiers.deit import deit_tscam_base_patch16_224
from dlib.div_classifiers.deit import deit_tscam_tiny_patch16_224
from dlib.div_classifiers.deit import deit_tscam_small_patch16_224

from dlib.configure import constants

models = dict()
for method in [constants.METHOD_SPG, constants.METHOD_ADL,
               constants.METHOD_ACOL]:
    models[method] = dict()

models[constants.METHOD_SPG][constants.RESNET50] = ResNet50Spg
models[constants.METHOD_SPG][constants.INCEPTIONV3] = InceptionV3Spg
models[constants.METHOD_SPG][constants.VGG16] = Vgg16Spg

models[constants.METHOD_ADL][constants.RESNET50] = ResNet50Adl
models[constants.METHOD_ADL][constants.INCEPTIONV3] = InceptionV3Adl
models[constants.METHOD_ADL][constants.VGG16] = Vgg16Adl

models[constants.METHOD_ACOL][constants.RESNET50] = ResNet50Acol
models[constants.METHOD_ACOL][constants.INCEPTIONV3] = InceptionV3Acol
models[constants.METHOD_ACOL][constants.VGG16] = Vgg16Acol

models[constants.METHOD_TSCAM] = dict()
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_TINY_P16_224] = deit_tscam_tiny_patch16_224
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_SMALL_P16_224] = deit_tscam_small_patch16_224
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_BASE_P16_224] = deit_tscam_base_patch16_224

