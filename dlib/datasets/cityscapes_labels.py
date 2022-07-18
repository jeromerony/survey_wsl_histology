"""
Cityscapes original labels.
Reference: https://github.com/mcordts/cityscapesScripts/blob/46fb3dfdc0caa45c9eb1b0e2dc86c4ceb87cd6b4/cityscapesscripts/helpers/labels.py
"""
#
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple
import sys
from os.path import dirname
from os.path import abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants


# ------------------------------------------------------------------------------
# Definitions
# ------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


# ------------------------------------------------------------------------------
# A list of all labels
# ------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


#-------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#-------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#-------------------------------------------------------------------------------
# Assure single instance name
#-------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None

def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name


# ==============================================================================
#                         OUR ENCODING OF THE LABELS
#             WE CONSIDER 20 LABELS:
#             1 LABEL FOR THE IGNORED OBJECTS (ignoreInEval = TRUE)
#             19 CLASSES  (ignoreInEval = FALSE)
# ==============================================================================
# colors are commented on the right.
# IgnoredInEval: code : 0, color: (  0,  0,  0)
mapping_20 = {
        0: 0,  # (  0,  0,  0)
        1: 0,  # (  0,  0,  0)
        2: 0,  # (  0,  0,  0)
        3: 0,  # (  0,  0,  0)
        4: 0,  # (  0,  0,  0)
        5: 0,  # (  0,  0,  0)
        6: 0,  # (  0,  0,  0)
        7: 1,  # (128, 64,128)
        8: 2,  # (244, 35,232)
        9: 0,  # (  0,  0,  0)
        10: 0,  # (  0,  0,  0)
        11: 3,  # ( 70, 70, 70)
        12: 4,  # (102,102,156)
        13: 5,  # (190,153,153)
        14: 0,  # (  0,  0,  0)
        15: 0,  # (  0,  0,  0)
        16: 0,  # (  0,  0,  0)
        17: 6,  # (153,153,153)
        18: 0,  # (  0,  0,  0)
        19: 7,  # (250,170, 30)
        20: 8,  # (220,220,  0)
        21: 9,  # (107,142, 35)
        22: 10,  # (152,251,152)
        23: 11,  # ( 70,130,180)
        24: 12,  # (220, 20, 60)
        25: 13,  # (255,  0,  0)
        26: 14,  # (  0,  0,142)
        27: 15,  # (  0,  0, 70)
        28: 16,  # (  0, 60,100)
        29: 0,  # (  0,  0,  0)
        30: 0,  # (  0,  0,  0)
        31: 17,  # (  0, 80,100)
        32: 18,  # (  0,  0,230)
        33: 19,  # (119, 11, 32)
        -1: 0  # (  0,  0,  0)
    }

# sanity check: number of labels in our coding must match
# constants.NUMBER_CLASSES[constants.CSCAPES]

actual_nbr_c = len(set([mapping_20[k] for k in mapping_20.keys()]))
expected_nbr_c = constants.NUMBER_CLASSES[constants.CSCAPES]
msg = "Expected nbr classes {}, actual nbr classes {} do not match." \
      "Cityscapes dataset.".format(
    expected_nbr_c, actual_nbr_c
)

assert actual_nbr_c == expected_nbr_c, msg

# meta-labels
meta_labels = dict()
meta_labels_reversed = dict()

c = "0"
for i in range(constants.NUMBER_CLASSES[constants.CSCAPES]):
    if i > 0:
        c = "{};{}".format(c, i)
    meta_labels[str(i)] = c
    meta_labels_reversed[c] = str(i)


def get_colormap():
    """
    Returns the color map associated with our encoding of the 20 classes.
    IgnoreInEval class (background): color is (0, 0, 0).
    :return:
    """
    palette = [
        0, 0, 0,
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        70, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32
    ]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    return palette

reversed_mapping_20 = dict()
for k in mapping_20.keys():
    val = mapping_20[k]
    if val == 0:
        reversed_mapping_20[val] = None  # ignored.
    else:
        reversed_mapping_20[val] = k

name_classes_rever_mapping_20 = dict()

for k in reversed_mapping_20.keys():
    if k == 0:
        name_classes_rever_mapping_20[k] = "Ignored"
    else:
        name_classes_rever_mapping_20[k] = id2label[reversed_mapping_20[k]].name


#-------------------------------------------------------------------------------
# Main for testing
#-------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format(
        id=id, category=category))

    print("Name of label with ID '{id}': {name}".format(
        id=id, name=id2label[id].name))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(id=trainId,
                                                             name=name))

    # test our encoding
    codes = []
    for item in labels:
        our_code = mapping_20[item.id]
        if item.ignoreInEval:
            assert our_code == 0, "ERROR 1"
        if our_code not in [0, 3]:
            codes.append(our_code)

    # assert that the codes are unique. I
    assert len(codes) == len(set(codes)), "ERROR 2"
    print("Number of our codes: {}".format(len(codes) + 1))

    # meta-labels
    print("Meta-labels: ")
    for k in meta_labels.keys():
        print("{}: {}".format(k, meta_labels[k]))
    print("meta-labels reversed:")
    for k in meta_labels_reversed.keys():
        print("{}: {}".format(k, meta_labels_reversed[k]))


