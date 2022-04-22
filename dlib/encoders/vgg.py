""" Each encoder should have following attributes and methods and be inherited
from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder
    feature tensor
    _depth (int): specify number of stages in decoder (in other words number
    of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer
    for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature
        is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to
        spatial resolution, starting with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16),
                (1, 256, 8, 8), (1, 512, 4, 4), (1, 1024, 2, 2)]
                 (C - dim may differ)

        also should support number of features according to specified depth,
        e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and
        5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as
        input and 3 downsampled).
"""

import sys
from os.path import dirname, abspath
from typing import Union, List, cast

import torch.nn as nn
from torchvision.models.vgg import VGG
from pretrainedmodels.models.torchvision_models import pretrained_settings

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders._base import EncoderMixin
from dlib.configure import constants

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'WSOL16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
               512, 512, 512]
}
# fmt: on


class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, depth=5,
                 **kwargs):
        super().__init__(make_layers(config), **kwargs)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

        self.full_features = nn.Sequential(*list(self.features.children()),
                                           self.conv6,
                                           self.relu)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        self.name = 'null-name'
        self.task = None

        del self.classifier

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError(
            "'VGG' models do not support dilated mode due to Max Pooling"
            " operations for downsampling!")

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.full_features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

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

    def load_state_dict_origin(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = remove_layer(state_dict, 'classifier.')
        state_dict = adjust_pretrained_model(state_dict, self)
        super().load_state_dict(state_dict, **kwargs)

    def super_load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)


def make_layers_old(cfg: List[Union[str, int]],
                batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v

    return nn.Sequential(*layers)


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 1024),
            "config": cfg["WSOL16"],  # D
            "depth": 3,  # del for D
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["WSOL16"],  # instead of D
            "depth": 3,  # del for D.
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}
