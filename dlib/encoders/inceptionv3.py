""" Each encoder should have following attributes and methods and be inherited
from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder
    feature tensor
    _depth (int): specify number of stages in decoder (in other words number of
    downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer
    for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature
         is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to
        spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16),
                (1, 256, 8, 8), (1, 512, 4, 4), (1, 1024, 2, 2)]
                 (C - dim may differ)

        also should support number of features according to specified depth,
        e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5
        downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as
        input and 3 downsampled).
"""
import sys
from os.path import dirname, abspath

import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders._base import EncoderMixin
from dlib.encoders.wsol_backbones.inceptionv3 import InceptionV3
from dlib.encoders.wsol_backbones.inceptionv3 import pretrained_settings

from dlib.configure import constants


class InceptionV3Encoder(InceptionV3, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        self.name = 'null-name'

        self.task = None

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.last_linear

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("InceptionV4 encoder does not support dilated mode "
                         "due to pooling operation for downsampling!")

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[: self._stage_idxs[0]],
            self.features[self._stage_idxs[0]: self._stage_idxs[1]],
            self.features[self._stage_idxs[1]: self._stage_idxs[2]],
            self.features[self._stage_idxs[2]: self._stage_idxs[3]],
            self.features[self._stage_idxs[3]:],
        ]

    def forward(self, x):

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)

            if self.task != constants.STD_CL:
                features.append(x)

        if not features:
            features = [x]

        return features

    def load_state_dict(self, state_dict, **kwargs):
        remove_layer(state_dict, 'Mixed_7')
        remove_layer(state_dict, 'AuxLogits')
        remove_layer(state_dict, 'fc.')
        super().load_state_dict(state_dict, **kwargs)

    def super_load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


inceptionv3_encoders = {
    "inceptionv3": {
        "encoder": InceptionV3Encoder,
        "pretrained_settings": pretrained_settings["inceptionv3"],
        "params": {
            "stage_idxs": (3, 5, 9, 15),
            "out_channels": (3, 64, 80, 288, 768, 1024),
            "num_classes": 1001,  # not used.
        },
    }
}
