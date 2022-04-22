# possible tasks
STD_CL = "STD_CL"  # standard classification using only the encoder features.
# ouputs: logits, cams.
F_CL = 'F_CL'  # fcam.
NEGEV = 'NEGEV'  # negative evidence. classification and detailed roi.
SEG = "SEGMENTATION"  # standard supervised segmentation. outputs:
# segmentation masks.

TASKS = [STD_CL, F_CL, SEG, NEGEV]

# meta-task
CLASSIFICATION = 'classification'
SEGMENTATION = 'segmentation'

# name of the classifier head (pooling operation)
WILDCATHEAD = "WildCatCLHead"

GAP = 'GAP'
WGAP = 'WGAP'
MAXPOOL = 'MaxPool'
LSEPOOL = 'LogSumExpPool'
NONEPOOL = 'NONE'
DEEPMIL = 'DeepMil'

SPATIAL_POOLINGS = [WILDCATHEAD, GAP, WGAP, MAXPOOL, LSEPOOL, NONEPOOL, DEEPMIL]

# methods
METHOD_WILDCAT = 'WILDCAT'  # pooling: WILDCATHEAD
METHOD_GAP = 'GAP'  # pooling: GAP

METHOD_MAXPOOL = 'MaxPOL'  # pooling: MAXPOOL
METHOD_LSE = 'LogSumExp'  # pooling: logsumexp.

# -- all methods below use WGAP.

METHOD_CAM = 'CAM'
METHOD_SCORECAM = 'ScoreCAM'
METHOD_SSCAM = 'SSCAM'
METHOD_ISCAM = 'ISCAM'

METHOD_GRADCAM = 'GradCam'
METHOD_GRADCAMPP = 'GradCAMpp'
METHOD_SMOOTHGRADCAMPP = 'SmoothGradCAMpp'
METHOD_XGRADCAM = 'XGradCAM'
METHOD_LAYERCAM = 'LayerCAM'

METHOD_HAS = 'HaS'
METHOD_CUTMIX = 'CutMIX'

METHOD_ADL = 'ADL'
METHOD_SPG = 'SPG'
METHOD_ACOL = 'ACoL'


METHOD_DEEPMIL = 'DEEPMIL'

METHOD_MAXMIN = 'MaxMin'

# SEG method
METHOD_SEG = 'SEG'

METHODS = [METHOD_WILDCAT,
           METHOD_GAP,
           METHOD_MAXPOOL,
           METHOD_LSE,
           METHOD_CAM,
           METHOD_SCORECAM,
           METHOD_SSCAM,
           METHOD_ISCAM,
           METHOD_GRADCAM,
           METHOD_GRADCAMPP,
           METHOD_SMOOTHGRADCAMPP,
           METHOD_XGRADCAM,
           METHOD_LAYERCAM,
           METHOD_ADL,
           METHOD_SPG,
           METHOD_ACOL,
           METHOD_HAS,
           METHOD_CUTMIX,
           METHOD_DEEPMIL,
           METHOD_MAXMIN,
           METHOD_SEG]

METHOD_2_POOLINGHEAD = {
        METHOD_WILDCAT: WILDCATHEAD,
        METHOD_GAP: GAP,
        METHOD_MAXPOOL: MAXPOOL,
        METHOD_LSE: LSEPOOL,
        METHOD_CAM: WGAP,
        METHOD_SCORECAM: WGAP,
        METHOD_SSCAM: WGAP,
        METHOD_ISCAM: WGAP,
        METHOD_GRADCAM: WGAP,
        METHOD_GRADCAMPP: WGAP,
        METHOD_SMOOTHGRADCAMPP: WGAP,
        METHOD_XGRADCAM: WGAP,
        METHOD_LAYERCAM: WGAP,
        METHOD_SEG: NONEPOOL,
        METHOD_ACOL: NONEPOOL,
        METHOD_ADL: NONEPOOL,
        METHOD_SPG: NONEPOOL,
        METHOD_HAS: WGAP,
        METHOD_CUTMIX: WGAP,
        METHOD_DEEPMIL: DEEPMIL,
        METHOD_MAXMIN: WILDCATHEAD
    }

METHOD_REQU_GRAD = {
        METHOD_WILDCAT: False,
        METHOD_GAP: False,
        METHOD_MAXPOOL: False,
        METHOD_LSE: False,
        METHOD_CAM: False,
        METHOD_SCORECAM: False,
        METHOD_SSCAM: False,
        METHOD_ISCAM: False,
        METHOD_GRADCAM: True,
        METHOD_GRADCAMPP: True,
        METHOD_SMOOTHGRADCAMPP: True,
        METHOD_XGRADCAM: True,
        METHOD_LAYERCAM: True,
        METHOD_SEG: False,
        METHOD_ACOL: False,
        METHOD_ADL: False,
        METHOD_SPG: False,
        METHOD_HAS: False,
        METHOD_CUTMIX: False,
        METHOD_DEEPMIL: False,
        METHOD_MAXMIN: False
}

METHOD_LITERAL_NAMES = {
        METHOD_WILDCAT: 'WILDCAT',
        METHOD_GAP: 'GAP',
        METHOD_MAXPOOL: 'MaxPool',
        METHOD_LSE: 'LSEPool',
        METHOD_CAM: 'CAM*',
        METHOD_SCORECAM: 'ScoreCAM',
        METHOD_SSCAM: 'SSCAM',
        METHOD_ISCAM: 'ISCAM',
        METHOD_GRADCAM: 'GradCAM',
        METHOD_GRADCAMPP: 'GradCam++',
        METHOD_SMOOTHGRADCAMPP: 'Smooth-GradCAM++',
        METHOD_XGRADCAM: 'XGradCAM',
        METHOD_LAYERCAM: 'LayerCAM',
        METHOD_SEG: 'Segmentation',
        METHOD_ACOL: 'ACoL',
        METHOD_ADL: 'ADL',
        METHOD_SPG: 'SPG',
        METHOD_HAS: 'HaS',
        METHOD_CUTMIX: 'CutMix',
        METHOD_DEEPMIL: 'DeepMIL',
        METHOD_MAXMIN: 'MaxMin'
}
# datasets mode
DS_TRAIN = "TRAIN"
DS_EVAL = "EVAL"

dataset_modes = [DS_TRAIN, DS_EVAL]

# Tags for samples
L = 0  # Labeled samples

samples_tags = [L]  # list of possible sample tags.

# pixel-wise supervision:
ORACLE = "ORACLE"  # provided by an oracle.
SELF_LEARNED = "SELF-LEARNED"  # self-learned.
VOID = "VOID"  # None

# segmentation modes.
#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


# pretraining
IMAGENET = "imagenet"

# archs
STDCLASSIFIER = "STDClassifier"
MaxMinClassifier = 'MaxMinClassifier'

UNETFCAM = 'UnetFCAM'  # USED
UNETNEGEV = 'UnetNEGEV'

ACOLARCH = 'ACOL'
SPGARCH = 'SPG'
ADLARCH = 'ADL'

UNET = "Unet"
UNETPLUPLUS = "UnetPlusPlus"
MANET = "MAnet"
LINKNET = "Linknet"
FPN = "FPN"
PSPNET = "PSPNet"
DEEPLABV3 = "DeepLabV3"
DEEPLABV3PLUS = "DeepLabV3Plus"
PAN = "PAN"

ARCHS = [STDCLASSIFIER, MaxMinClassifier, ACOLARCH, ADLARCH, SPGARCH,
         UNETFCAM, UNETNEGEV, UNET]

# std cld method to arch.
STD_CL_METHOD_2_ARCH = {
    METHOD_WILDCAT: STDCLASSIFIER,
    METHOD_GAP: STDCLASSIFIER,
    METHOD_MAXPOOL: STDCLASSIFIER,
    METHOD_LSE: STDCLASSIFIER,
    METHOD_CAM: STDCLASSIFIER,
    METHOD_SCORECAM: STDCLASSIFIER,
    METHOD_SSCAM: STDCLASSIFIER,
    METHOD_ISCAM: STDCLASSIFIER,
    METHOD_GRADCAM: STDCLASSIFIER,
    METHOD_GRADCAMPP: STDCLASSIFIER,
    METHOD_SMOOTHGRADCAMPP: STDCLASSIFIER,
    METHOD_XGRADCAM: STDCLASSIFIER,
    METHOD_LAYERCAM: STDCLASSIFIER,
    METHOD_ACOL: ACOLARCH,
    METHOD_ADL: ADLARCH,
    METHOD_SPG: SPGARCH,
    METHOD_HAS: STDCLASSIFIER,
    METHOD_CUTMIX: STDCLASSIFIER,
    METHOD_DEEPMIL: STDCLASSIFIER,
    METHOD_MAXMIN: MaxMinClassifier
}
# ecnoders

#  resnet
RESNET50 = 'resnet50'

# vgg
VGG16 = 'vgg16'

# inceptionv3
INCEPTIONV3 = 'inceptionv3'

BACKBONES = [RESNET50,
             VGG16,
             INCEPTIONV3
             ]

# ------------------------------------------------------------------------------

# datasets
DEBUG = False
assert not DEBUG

ILSVRC = "ILSVRC"
CUB = "CUB"
OpenImages = 'OpenImages'

GLAS = 'GLAS'  # GLAS 15
ICIAR = 'ICIAR'  # ICIAR-2018-BACH-Challenge
CAMELYON512 = 'CAMELYON512'  # Camelyon16 512 patch.
BREAKHIS = 'BREAKHIS'

FORMAT_DEBUG = 'DEBUG_{}'
if DEBUG:
    CUB = FORMAT_DEBUG.format(CUB)
    ILSVRC = FORMAT_DEBUG.format(ILSVRC)
    OpenImages = FORMAT_DEBUG.format(OpenImages)


datasets = [CUB, ILSVRC, OpenImages, GLAS, BREAKHIS, CAMELYON512, ICIAR]
SUPPORTED_DS = [GLAS, BREAKHIS, CAMELYON512, ICIAR]

# Magnification for breakhis dataset
MAG40X = '40X'
MAG100X = '100X'
MAG200X = '200X'
MAG400X = '400X'

MAGNIFICATIONSBHIS = [MAG40X, MAG100X, MAG200X, MAG400X]

NBR_CHUNKS_TR_ILSVRC = {
    'ILSVRC': 30,
    'DEBUG_ILSVRC': 2
}

LOCALIZATION_AVAIL = {
    GLAS: True,
    CAMELYON512: True,
    BREAKHIS: False,
    ICIAR: False
}

RELATIVE_META_ROOT = './folds/wsol-done-right-splits'
SCRATCH_FOLDER = 'benchmark_wsol_histo'

NUMBER_CLASSES = {
    ILSVRC: 1000,
    CUB: 200,
    OpenImages: 100,
    GLAS: 2,
    CAMELYON512: 2,
    ICIAR: 4,
    BREAKHIS: 2
}

CROP_SIZE = 224
RESIZE_SIZE = 256

# ================= check points
BEST_CL = 'best_classification'
BEST_LOC = 'best_localization'

COLOUR_BEST_CP = {
    BEST_CL: 'blue',
    BEST_LOC: 'lawngreen'
}

EVAL_CHECKPOINT = {
    GLAS: BEST_LOC,
    CAMELYON512: BEST_LOC,
    BREAKHIS: BEST_CL,
    ICIAR: BEST_CL
}

# ==============================================================================

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

# =================================================
NCOLS = 80  # tqdm ncols.

# stages:
STGS_TR = "TRAIN"
STGS_EV = "EVAL"


# datasets:
TRAINSET = 'train'
CLVALIDSET = 'valcl'
PXVALIDSET = 'valpx'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, PXVALIDSET, CLVALIDSET, VALIDSET, TESTSET]

# image range: [0, 1] --> Sigmoid. [-1, 1]: TANH
RANGE_TANH = "tanh"
RANGE_SIGMOID = 'sigmoid'

# ==============================================================================
# cams extractor
TRG_LAYERS = {
            RESNET50: 'encoder.layer4.2.relu3',
            VGG16: 'encoder.relu',
            INCEPTIONV3: 'encoder.SPG_A3_2b.2'
        }
FC_LAYERS = {
    RESNET50: 'classification_head.fc',
    VGG16: 'classification_head.fc',
    INCEPTIONV3: 'classification_head.fc'
}

# EXPs
OVERRUN = False

# cam_curve_interval: for bbox. use high interval for validation (not test).
# high number of threshold slows down the validation because of
# `cv2.findContours`. this gets when cams are bad leading to >1k contours per
# threshold. default evaluation: .001.
VALID_FAST_CAM_CURVE_INTERVAL = .004

# data: name of the folder where cams will be stored.
DATA_CAMS = 'data_cams'

FULL_BEST_EXPS = 'full_best_exps'

# DDP
NCCL = 'nccl'
GLOO = 'gloo'
MPI = 'mpi'


# metrics names
LOCALIZATION_MTR = 'localization'
CLASSIFICATION_MTR = 'classification'

# partial names of metrics
MTR_PXAP = 'PXAP'
MTR_TP = 'True positive'
MTR_FN = 'False negative'
MTR_TN = 'True negative'
MTR_FP = 'False positive'
MTR_DICEFG = 'Dice foreground'
MTR_DICEBG = 'Dice background'
MTR_MIOU = 'MIOU'
MTR_BESTTAU = 'Best tau'

MTR_CL = 'Classification accuracy'

# experiment mode: hyper-parameters search or final mode.
RMODE_SEARCH = 'search-mode'
RMODE_FINAL = 'final-mode'

# nbr folds: all datasets have 5 folds.
FOLDS_NBR = 5
FOLD_SEARCH = 0  # fold used for hyper-prama search.

# folder
FOLDER_PRETRAINED_IMAGENET = 'pretrained-imgnet'
FOLDER_EXP_SEARCH = 'exps'
FOLDER_EXP_FINAL = 'exp_final'

FOLDER_EXP = {
    RMODE_SEARCH: FOLDER_EXP_SEARCH,
    RMODE_FINAL: FOLDER_EXP_FINAL
}

# see sampler type: f_Cam, NEGEV
SEED_TH = 'threshold_seeder'
SEED_PROB = 'probability_seeder'
SEED_PROB_N_AREA = 'probability_negative_area_seeder'

# pairing samples: negev
PAIR_SAME_C = 'same_class'
PAIR_DIFF_C = 'different_class'
PAIR_MIXED_C = 'mixed_class'

# datasets with negative samples:
DS_HAS_NEG_SAM = {
    CAMELYON512: True,
    GLAS: False
}


DS_NEG_CL = {
    CAMELYON512: 0,
    GLAS: 0
}

# plot orientation.
PLOT_HOR = 'horizontal'
PLOT_VER = 'vertical'
PLOT_ORIENTATIONS = [PLOT_VER, PLOT_HOR]

