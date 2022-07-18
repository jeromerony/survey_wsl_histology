"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
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
from dlib.div_classifiers.util import initialize_weights

from dlib.div_classifiers.core import CoreClassifier

from dlib.configure import constants

__all__ = ['InceptionV3Adl', 'InceptionV3Acol', 'InceptionV3Spg']

model_urls = {
    'inception_v3_google':
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size,
                                     stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3,
                                          stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride,
                                   padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7_2 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCam(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionCam, self).__init__()

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x)


        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        if return_cam:
            feature_map = feat_map.clone().detach()
            cams = feature_map[range(batch_size), labels]
            return cams

        return {'logits': logits}

    def get_loss(self, logits, target):
        loss_cls = nn.CrossEntropyLoss()(logits, target.long())
        return loss_cls


class InceptionV3Acol(AcolBase, CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 acol_drop_threshold=0.1,
                 scale_in=1.,
                 in_channels: int = 3):
        super(InceptionV3Acol, self).__init__()

        self.encoder_name = constants.INCEPTIONV3
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

        self.logits_dict = None

        self.large_feature_map = large_feature_map

        self.drop_threshold = acol_drop_threshold

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.classifier_A = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
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

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feature = self.Mixed_6e(x)

        self.logits_dict = self._acol_logits(feature=feature, labels=labels,
                                             drop_threshold=self.drop_threshold)

        if labels is not None:
            normalized_a = normalize_tensor(
                self.logits_dict['feat_map_a'].clone().detach())
            normalized_b = normalize_tensor(
                self.logits_dict['feat_map_b'].clone().detach())
            feature_maps = torch.max(normalized_a, normalized_b)
            self.cams = feature_maps[range(batch_size), labels].detach()

        return self.logits_dict['logits']

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, path=None)


class InceptionV3Spg(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 scale_in=1.,
                 in_channels: int = 3):
        super(InceptionV3Spg, self).__init__()

        self.encoder_name = constants.INCEPTIONV3

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

        self.logits_dict = None

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.SPG_A4 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(288, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
        )
        self.SPG_C = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
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

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

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
        load_pretrained_model(model=self, path=None)


class InceptionV3Adl(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 adl_drop_rate=.4,
                 adl_drop_threshold=.1,
                 scale_in=1.,
                 in_channels: int = 3
                 ):
        super(InceptionV3Adl, self).__init__()

        self.encoder_name = constants.INCEPTIONV3

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

        self.large_feature_map = large_feature_map

        self.adl_drop_rate = adl_drop_rate
        self.adl_threshold = adl_drop_threshold

        self.ADL_5d = ADL(self.adl_drop_rate, self.adl_threshold)
        self.ADL_6e = ADL(self.adl_drop_rate, self.adl_threshold)
        self.ADL_A3_2b = ADL(self.adl_drop_rate, self.adl_threshold)

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )

        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)

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

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.ADL_5d(x)
        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.ADL_6e(x)

        x = self.SPG_A3_1b(x)
        x = self.SPG_A3_2b(x)
        x = self.ADL_A3_2b(x)
        x = self.SPG_A4(x)

        logits = self.avgpool(x)
        logits = logits.view(x.shape[0:2])

        if labels is not None:
            feature_map = x.clone().detach()
            self.cams = feature_map[range(batch_size), labels].detach()

        self.logits_dict = {'logits': logits}
        return logits

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, path=None)


def load_pretrained_model(model, path=None):
    if path:
        state_dict = torch.load(
            os.path.join(path, 'inception_v3.pth'))
    else:
        state_dict = load_url(model_urls['inception_v3_google'],
                              progress=True,
                              model_dir=IMG_NET_W_FD)

    remove_layer(state_dict, 'Mixed_7')
    remove_layer(state_dict, 'AuxLogits')
    remove_layer(state_dict, 'fc.')

    model.load_state_dict(state_dict, strict=False)
    return model


def inception_v3(architecture_type, pretrained=False, pretrained_path=None,
                 **kwargs):
    model = {constants.ACOLARCH: InceptionV3Acol,
             constants.SPGARCH: InceptionV3Spg,
             constants.ADLARCH: InceptionV3Adl}[architecture_type](**kwargs)
    if pretrained:
        model = load_pretrained_model(model, pretrained_path)
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


def test_inceptionv3():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = False
    num_classes = 200
    for arch in [constants.ADLARCH, constants.SPGARCH, constants.ACOLARCH]:
        if arch == constants.ADLARCH:
            model = inception_v3(arch, pretrained, num_classes=num_classes,
                                 adl_drop_rate=.4, adl_drop_threshold=.1,
                                 large_feature_map=True)

        elif arch == constants.ACOLARCH:
            model = inception_v3(arch, pretrained, num_classes=num_classes,
                                 acol_drop_threshold=.1, large_feature_map=True)

        elif arch == constants.SPGARCH:
            model = inception_v3(arch, pretrained, num_classes=num_classes,
                                 large_feature_map=True)
        else:
            raise NotImplementedError

        model.to(device)
        print(model.get_info_nbr_params())
        bsize = 1
        h, w = 224, 224
        x = torch.rand(bsize, 3, 224, 224).to(device)
        labels = torch.zeros((bsize,), dtype=torch.long)
        out = model(x, labels=labels)

        if arch == constants.ADLARCH:
            print(out.shape)

        t0 = dt.datetime.now()
        model(x, labels=labels)
        cams = model.cams
        print(cams.shape)
        if cams.shape != (bsize, h, w):
            tx = dt.datetime.now()
            full_cam = F.interpolate(
                input=cams.unsqueeze(0),
                size=[h, w],
                mode='bilinear',
                align_corners=True)
        print(x.shape, cams.shape)
        print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    test_inceptionv3()
