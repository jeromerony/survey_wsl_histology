"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url


import sys
from os.path import dirname, abspath, join

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.div_classifiers.parts import AcolBase
from dlib.div_classifiers.parts import ADL
from dlib.div_classifiers.parts import spg
from dlib.div_classifiers.parts.util import normalize_tensor
from dlib.div_classifiers.util import remove_layer
from dlib.div_classifiers.util import replace_layer
from dlib.div_classifiers.util import initialize_weights

from dlib.div_classifiers.core import CoreClassifier

from dlib.configure import constants


__all__ = ['Vgg16Adl', 'Vgg16Acol', 'Vgg16Spg']


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    constants.ACOLARCH: {
        '14x14': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M1', 512, 512, 512, 'M2'],
        '28x28': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M2', 512, 512, 512, 'M2'],
    },
    constants.SPGARCH: {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    },
    constants.ADLARCH: {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 'M', 512, 512, 512, 'A'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 512, 512, 512, 'A'],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        print(x.shape)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams
        return {'logits': logits}


class Vgg16Acol(AcolBase, CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 acol_drop_threshold=0.1,
                 scale_in=1.,
                 in_channels: int = 3):
        super(Vgg16Acol, self).__init__()

        self.encoder_name = constants.VGG16
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_ACOL
        self.arch = constants.ACOLARCH

        config_key = '28x28' if large_feature_map else '14x14'
        features = make_layers(configs_dict[self.arch][config_key])

        self.logits_dict = None

        self.features = features
        self.drop_threshold = acol_drop_threshold

        self.classifier_A = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        batch_size = x.shape[0]

        feature = self.features(x)
        feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        self.logits_dict = self._acol_logits(feature=feature, labels=labels,
                                             drop_threshold=self.drop_threshold)

        if labels is not None:
            normalized_a = normalize_tensor(
                self.logits_dict['feat_map_a'].detach().clone())
            normalized_b = normalize_tensor(
                self.logits_dict['feat_map_b'].detach().clone())
            feature_map = torch.max(normalized_a, normalized_b)
            self.cams = feature_map[range(batch_size), labels].detach()

        return self.logits_dict['logits']

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None)


class Vgg16Spg(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 scale_in=1.,
                 in_channels: int = 3):
        super(Vgg16Spg, self).__init__()

        self.encoder_name = constants.VGG16

        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_SPG
        self.arch = constants.SPGARCH

        config_key = '28x28' if large_feature_map else '14x14'
        features = make_layers(configs_dict[self.arch][config_key])

        self.logits_dict = None

        self.features = features
        self.lfs = large_feature_map

        self.SPG_A_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_A_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_A_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_A_4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        batch_size = x.shape[0]

        x = self.features(x)
        x = self.SPG_A_1(x)
        if not self.lfs:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.SPG_A_3(x)
        logits_c = self.SPG_C(x)

        feat_map = self.SPG_A_4(x)
        logits = self.avgpool(feat_map)
        logits = logits.flatten(1)

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if labels is not None:
            feature_map = feat_map.clone().detach()
            self.cams = feature_map[range(batch_size), labels].detach()

        self.logits_dict = {'attention': attention,
                            'fused_attention': fused_attention,
                            'logits': logits, 'logits_b1': logits_b1,
                            'logits_b2': logits_b2, 'logits_c': logits_c}
        return logits

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None)


class Vgg16Adl(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 adl_drop_rate=.4,
                 adl_drop_threshold=.1,
                 scale_in=1.,
                 in_channels: int = 3):
        super(Vgg16Adl, self).__init__()
        self.encoder_name = constants.VGG16

        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_ADL
        self.arch = constants.ADLARCH

        self.logits_dict = None

        config_key = '28x28' if large_feature_map else '14x14'
        features = make_layers(configs_dict[self.arch][config_key],
                               adl_drop_rate=adl_drop_rate,
                               adl_drop_threshold=adl_drop_threshold)

        self.features = features

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if labels is not None:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            self.cams = cams.detach()  # b, h`, w`

        self.logits_dict = {'logits': logits}

        return logits

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None)


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            print('key {} split {}'.format(key, split))

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


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'features.17', 'SPG_A_1.0')
    state_dict = replace_layer(state_dict, 'features.19', 'SPG_A_1.2')
    state_dict = replace_layer(state_dict, 'features.21', 'SPG_A_1.4')
    state_dict = replace_layer(state_dict, 'features.24', 'SPG_A_2.0')
    state_dict = replace_layer(state_dict, 'features.26', 'SPG_A_2.2')
    state_dict = replace_layer(state_dict, 'features.28', 'SPG_A_2.4')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True,
                              model_dir=IMG_NET_W_FD)

    if architecture_type == constants.SPGARCH:
        state_dict = batch_replace_layer(state_dict)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, adl_drop_rate=None, adl_drop_threshold=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            assert adl_drop_rate is not None
            assert adl_drop_threshold is not None
            layers += [ADL(adl_drop_rate, adl_drop_threshold)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):

    model = {constants.ACOLARCH: Vgg16Acol,
             constants.SPGARCH: Vgg16Spg,
             constants.ADLARCH: Vgg16Adl}[architecture_type](**kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
    return model


def findout_names(model, architecture):
    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],  # CLASSIFIER
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(architecture):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if architecture.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(architecture))

    for name, parameter in model.named_parameters():
        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in ('vgg16', 'inception_v3'):
                # param_features.append(parameter)
                print(name, '==>', 'feature')
            elif architecture == 'resnet50':
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
        else:
            if architecture in ('vgg16', 'inception_v3'):
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
            elif architecture == 'resnet50':
                # param_features.append(parameter)
                print(name, '==>', 'feature')


def test_vgg():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = False
    num_classes = 200
    for arch in [constants.ADLARCH, constants.SPGARCH, constants.ACOLARCH]:
        if arch == constants.ADLARCH:
            model = vgg16(arch, pretrained, num_classes=num_classes,
                          adl_drop_rate=.4, adl_drop_threshold=.1,
                          large_feature_map=True)

        elif arch == constants.ACOLARCH:
            model = vgg16(arch, pretrained, num_classes=num_classes,
                          acol_drop_threshold=.1,
                          large_feature_map=True)

        elif arch == constants.SPGARCH:
            model = vgg16(arch, pretrained, num_classes=num_classes,
                          large_feature_map=True)
        else:
            raise NotImplementedError

        model.to(device)
        print(model.get_info_nbr_params())
        bsize = 1
        h, w = 224, 224
        x = torch.rand(bsize, 3, 224, 224).to(device)
        labels = torch.zeros((bsize,), dtype=torch.long)
        logits = model(x)
        print(f'logits shape : {logits.shape} x : {x.shape} '
              f'classes : {num_classes}')

        t0 = dt.datetime.now()
        model(x, labels=labels)
        cams = model.cams
        print(cams.shape)
        if cams.shape != (1, h, w):
            tx = dt.datetime.now()
            full_cam = F.interpolate(
                input=cams.unsqueeze(0),
                size=[h, w],
                mode='bilinear',
                align_corners=True)
        print(x.shape, cams.shape)
        print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    test_vgg()
