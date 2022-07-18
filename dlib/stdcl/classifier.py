import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel

from dlib import poolings

from dlib.configure import constants


class STDClassifier(STDClModel):
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
    ):
        super(STDClassifier, self).__init__()

        self.encoder_name = encoder_name
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

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
        self.classification_head = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()


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
        model = STDClassifier(
            task=constants.STD_CL,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=constants.IMAGENET,
            in_channels=in_channels,
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
        print('logits', cl_logits)
        val, ind = torch.sort(cl_logits.cpu(), dim=1, descending=True,
                              stable=True)
        print(val, ind)

        findout_names(model, encoder_name)
