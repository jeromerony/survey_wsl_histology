import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel
from dlib.base import initialization as init

from dlib import poolings

from dlib.configure import constants
from dlib.utils.shared import count_params


class _PullMask(torch.nn.Module):
    def __init__(self, in_channels: int, classes: int, modalities: int):
        super(_PullMask, self).__init__()
        self.C = classes
        self.M = modalities
        self.in_channels = in_channels

        for v in [in_channels, classes, modalities]:
            assert isinstance(v, int)
            assert v > 0

        self.to_modalities = torch.nn.Conv2d(
            self.in_channels, self.C * self.M, kernel_size=1,  bias=True)


    def forward(self, x):
        modalities = self.to_modalities(x)
        N, C, H, W = modalities.size()
        assert C == self.C * self.M
        return torch.mean(modalities.view(N, self.C, self.M, -1), dim=2).view(
            N, self.C, H, W)

    def __str__(self):
        return self.__class__.__name__ + \
               f'(in_channels={self.in_channels}, ' \
               f'classes={self.C}, modalities={self.M})'

    def __repr__(self):
        return super(_PullMask, self).__repr__()


class MaxMinClassifier(STDClModel):
    """
    Standard classifier.
    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        aux_params: Optional[dict] = None,
        scale_in: float = 1.,
        w: float = 8.,
        dataset_name: str = ''
    ):
        super(MaxMinClassifier, self).__init__()

        self.encoder_name = encoder_name
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        self._assert_dataset_name(dataset_name)
        self.dataset_name = dataset_name

        assert w > 0
        assert isinstance(w, float)
        self.maxmin_w = w

        self.x_in = None

        self.encoder = get_encoder(
            task,
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        assert aux_params is not None
        pooling_head = aux_params['pooling_head']
        aux_params.pop('pooling_head')

        self.classification_head1 = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )
        self.classification_head2 = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        modalities = aux_params['modalities'] if 'modalities'\
                                                 in aux_params else 5
        # self.mask_head = _PullMask(
        #     in_channels=self.encoder.out_channels[-1], classes=1,
        #     modalities=modalities
        # )

        self.mask_head = None

        self.logits_dict: dict = dict()

        self.name = "u-{}".format(encoder_name)
        init.initialize_head(self.classification_head1)
        init.initialize_head(self.classification_head2)
        # init.initialize_head(self.mask_head)

        self.cams = None

    def get_info_nbr_params(self) -> str:
        info = self.__str__() + ' \n NBR-PARAMS: \n'
        if self.encoder:
            info += '\tEncoder [{}]: {}. \n'.format(self.encoder.name,
                                                    count_params(self.encoder))
        if self.classification_head1:
            info += '\tClassification head1 [{}]: {}. \n'.format(
                self.classification_head1.name,
                count_params(self.classification_head1))
        info += '\tTotal: {}. \n'.format(count_params(self))

        return info

    def _assert_dataset_name(self, dataset_name: str):
        assert isinstance(dataset_name, str)
        assert dataset_name in [constants.GLAS, constants.CAMELYON512]

    def forward(self, x):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self._assert_dataset_name(self.dataset_name)

        b, _, h, w = x.shape

        self.x_in = x
        features = self.encoder(x)
        cl_logits_loc = self.classification_head1(features[-1])

        if self.dataset_name == constants.CAMELYON512:
            cam = self.mask_head(features[-1])  # b, 1, h', w'.
        elif self.dataset_name == constants.GLAS:
            cams = self.classification_head1.cams_attached
            prob = F.softmax(cl_logits_loc, dim=1)
            mpositive = torch.zeros((b, 1, cams.size()[2], cams.size()[3]),
                                    dtype=cams.dtype,
                                    layout=cams.layout,
                                    device=cams.device
                                    )
            for i in range(b):
                for j in range(prob.size()[1]):
                    mpositive[i] = mpositive[i] + prob[i, j] * cams[i, j, :, :]

        cam = mpositive
        cam_f = F.interpolate(input=cam,
                              size=x.shape[2:],
                              mode='bilinear',
                              align_corners=True
                              )
        cam_f = (cam_f - cam_f.amin(dim=(1, 2, 3), keepdim=True)) / (cam_f.amax(
            dim=(1, 2, 3), keepdim=True) - cam_f.amin(
            dim=(1, 2, 3), keepdim=True))

        cam_logits = self.maxmin_w * (cam_f - 0.15)
        cam_f = torch.sigmoid(cam_logits)
        self.cams = cam_f.detach()

        cl_logits_pos = self.classification_head2(self.encoder(x * cam_f)[-1])
        cl_logits_neg = self.classification_head2(
            self.encoder(x * (1. - cam_f))[-1])

        self.logits_dict = {
            'logits': cl_logits_loc,
            'logits_pos': cl_logits_pos,
            'logits_neg': cl_logits_neg,
            'cam': cam_f,
            'cam_logits': cam_logits
        }

        return cl_logits_loc


def findout_names(model, architecture):
    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1', 'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
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
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_features.append(parameter)
                print(name, '==>', 'feature')
            elif architecture == constants.RESNET50:
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
            elif architecture == constants.RESNET50:
                # param_features.append(parameter)
                print(name, '==>', 'feature')


if __name__ == "__main__":
    import datetime as dt
    import dlib
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    vgg_encoders = dlib.encoders.vgg_encoders
    in_channels = 3
    SZ = 224
    sample = torch.rand((32, in_channels, SZ, SZ)).to(DEVICE)
    encoders = [constants.RESNET50, constants.INCEPTIONV3, constants.VGG16]

    amp = True

    for encoder_name in encoders:

        announce_msg("Testing backbone {}".format(encoder_name))
        if encoder_name == constants.VGG16:
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            encoder_depth = 5

        # task: STD_CL
        model = MaxMinClassifier(
            task=constants.STD_CL,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=constants.IMAGENET,
            in_channels=in_channels,
            w=8.,
            aux_params=dict(pooling_head="WildCatCLHead", classes=200)
        ).to(DEVICE)
        announce_msg("TESTING: {} -- amp={} \n {}".format(model, amp,
                     model.get_info_nbr_params()))
        t0 = dt.datetime.now()
        with torch.no_grad():
            with autocast(enabled=amp):
                cl_logits = model(sample).detach()

        torch.cuda.empty_cache()
        with torch.no_grad():
            with autocast(enabled=amp):
                cl_logitsx = model(sample[0].unsqueeze(0)).detach()
        print('forward time {}'.format(dt.datetime.now() - t0))
        print("x: {} \t cl_logits: {}".format(sample.shape, cl_logits.shape))
        for v in model.logits_dict:
            print(f'{v}: {model.logits_dict[v].shape}')

        findout_names(model, encoder_name)
