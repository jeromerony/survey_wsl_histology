import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from functools import partial

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib
from dlib.stdcl.classifier import STDClassifier
from dlib.unet.model import UnetFCAM
from dlib.unet.model import UnetNEGEV
from dlib.unet.model import Unet

from dlib.parallel import MyDDP as DDP


__all__ = ['BuiltinCam', 'ReadyCam', 'DeepMILCam', 'MaxMinCam',
           'SegmentationCam']


class BuiltinCam:
    """Implements a class activation map extractor over models that construct
     cam tensor first, then use it to compute score classes.
     cams are already computed and stored in the score pooling module.
     Example of poolings: GAP, MAXPOOLING, LOGSUMEXP, WILDCATHEAD.

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: STDClassifier
    ) -> None:

        self.assert_model(model)

        self.model = model
        self.support_backgr = self.model.classification_head.support_background

    @staticmethod
    def assert_model(model: STDClassifier) -> None:
        # _model = model if not isinstance(model, DDP) else model.module
        _model = model

        assert isinstance(_model, STDClassifier)

        assert any([isinstance(_model.encoder,
                               dlib.encoders.resnet.ResNetEncoder),
                    isinstance(_model.encoder, dlib.encoders.vgg.VGGEncoder),
                    isinstance(_model.encoder,
                               dlib.encoders.inceptionv3.InceptionV3Encoder)])

        assert any([isinstance(_model.classification_head, dlib.poolings.GAP),
                    isinstance(_model.classification_head,
                               dlib.poolings.MaxPool),
                    isinstance(_model.classification_head,
                               dlib.poolings.WildCatCLHead),
                    isinstance(_model.classification_head,
                               dlib.poolings.LogSumExpPool),
                    ])

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _precheck(self, class_idx: int) -> None:
        """Check for invalid computation cases"""

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

    def __call__(self,
                 class_idx: int,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: Optional[bool] = False) -> Tensor:

        # Integrity check
        self._precheck(class_idx)

        # Compute CAM: (h, w)
        cam = self.compute_cams(class_idx, normalized)
        if reshape is not None:
            assert len(reshape) == 2
            interpolation_mode = 'bilinear'
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode=interpolation_mode,
                                align_corners=False).squeeze(0).squeeze(0)
        return cam

    def compute_cams(self, class_idx: int, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM
               will be computed
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        cams = self.model.classification_head.cams
        assert cams.shape[0] == 1

        if self.support_backgr:
            cam = cams[:, class_idx + 1, :, :].squeeze(0)
        else:
            cam = cams[:, class_idx, :, :].squeeze(0)  # (h, w)

        # Normalize the CAM
        if normalized:
            cam = self._normalize(cam)

        return cam

    def extra_repr(self) -> str:
        return f"support_background='{self.support_backgr}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class ReadyCam:
    """
    Implements a class activation map extractor over models that have the
    cam already ready. This concerns: SPG, ADL, ACOL methods.
     cams are already computed and stored in classifier under: model.cams of
     shape (1, h, w).
    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model
    ) -> None:

        self.assert_model(model)

        self.model = model
        # not supported.
        self.support_backgr = False

    @staticmethod
    def assert_model(model) -> None:
        pass

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _precheck(self, class_idx: int) -> None:
        """Check for invalid computation cases"""

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

    def __call__(self,
                 class_idx: int,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: Optional[bool] = False) -> Tensor:

        # Integrity check
        self._precheck(class_idx)

        # Compute CAM: (h, w)
        cam = self.compute_cams(class_idx, normalized)
        if reshape is not None:
            assert len(reshape) == 2
            interpolation_mode = 'bilinear'
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode=interpolation_mode,
                                align_corners=False).squeeze(0).squeeze(0)
        return cam

    def compute_cams(self, class_idx: int, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM
               will be computed
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        cams = self.model.cams
        assert cams.shape[0] == 1
        assert cams.ndim == 3  # bs, h, w

        cam = cams.squeeze(0)  # (h, w)

        # Normalize the CAM
        if normalized:
            cam = self._normalize(cam)

        return cam

    def extra_repr(self) -> str:
        return f"support_background='{self.support_backgr}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class DeepMILCam:
    """
    Implements a class activation map extractor over models that have the
    cam already ready. This concerns: DeepMil method.
    cams are already computed and stored in classifier head under:
    model.classifier_head.cams of
    shape (bsz, c, h, w) where c is the total number of classes.
    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model
    ) -> None:

        self.assert_model(model)

        self.model = model
        self.support_backgr = model.classification_head.support_background
        assert not self.support_backgr

    @staticmethod
    def assert_model(model) -> None:
        pass

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _precheck(self, class_idx: int) -> None:
        """Check for invalid computation cases"""

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

    def __call__(self,
                 class_idx: int,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: Optional[bool] = False) -> Tensor:

        # Integrity check
        self._precheck(class_idx)

        # Compute CAM: (h, w)
        cam = self.compute_cams(class_idx, normalized)
        if reshape is not None:
            assert len(reshape) == 2
            interpolation_mode = 'bilinear'
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode=interpolation_mode,
                                align_corners=False).squeeze(0).squeeze(0)
        return cam

    def compute_cams(self, class_idx: int, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM
               will be computed
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        cams = self.model.classification_head.cams

        assert cams.shape[0] == 1
        assert cams.ndim == 4  # bs, c, h, w
        cam = cams[:, class_idx, :, :].squeeze(0)  # (h, w)

        # Normalize the CAM
        if normalized:
            cam = self._normalize(cam)

        return cam

    def extra_repr(self) -> str:
        return f"support_background='{self.support_backgr}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class MaxMinCam:
    """
    Implements a class activation map extractor over models that have the
    cam already ready. This concerns: maxmin method.
     cams are already computed and stored in classifier under: model.cams of
     shape (1, h, w).
    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model
    ) -> None:

        self.assert_model(model)

        self.model = model
        self.support_backgr = False

    @staticmethod
    def assert_model(model) -> None:
        pass

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _precheck(self, class_idx: int) -> None:
        """Check for invalid computation cases"""

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

    def __call__(self,
                 class_idx: int,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: Optional[bool] = False) -> Tensor:

        # Integrity check
        self._precheck(class_idx)

        # Compute CAM: (h, w)
        cam = self.compute_cams(class_idx, normalized)
        if reshape is not None:
            assert len(reshape) == 2
            interpolation_mode = 'bilinear'
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode=interpolation_mode,
                                align_corners=False).squeeze(0).squeeze(0)
        return cam

    def compute_cams(self, class_idx: int, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM
               will be computed
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        cams = self.model.cams
        assert cams.shape[0] == 1
        assert cams.shape[1] == 1
        assert cams.ndim == 4  # bs, 1, h, w

        cam = cams.squeeze(0).squeeze(0)  # (h, w)

        # Normalize the CAM
        if normalized:
            cam = self._normalize(cam)

        return cam

    def extra_repr(self) -> str:
        return f"support_background='{self.support_backgr}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class SegmentationCam:
    """Extract CAM frpm segmentation model.

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: Union[UnetFCAM, UnetNEGEV, Unet]
    ) -> None:

        self.assert_model(model)

        self.model = model
        if hasattr(self.model, 'classification_head'):
            self.support_backgr = self.model.classification_head.support_background
        else:
            # todo: segmentation.
            self.support_backgr = False

    @staticmethod
    def assert_model(model: Union[UnetFCAM, Unet]) -> None:
        # _model = model if not isinstance(model, DDP) else model.module

        _model = model

        # assert isinstance(_model, UnetFCAM), type(model)

        assert any([isinstance(_model.encoder,
                               dlib.encoders.resnet.ResNetEncoder),
                    isinstance(_model.encoder, dlib.encoders.vgg.VGGEncoder),
                    isinstance(_model.encoder,
                               dlib.encoders.inceptionv3.InceptionV3Encoder)])

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def __call__(self,
                 class_idx: Optional[int] = None,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: bool = False) -> Tensor:

        # Compute CAM: (h, w)
        cam = self.compute_cams(argmax=argmax)
        if reshape is not None:
            assert len(reshape) == 2
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0)
        return cam

    def compute_cams(self, argmax: bool = False) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            argmax (bool, optional): if true, we compute the argmax over the
            segmentation cams to get a binary map where 1 is foreground and 0
            is background. if false, we return the normalize cam at index 1.
            the segmentation cams are normalized using softmax.

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        cams = self.model.cams
        cams = cams.float()

        assert cams.ndim == 4
        assert cams.shape[0] == 1
        assert cams.shape[1] == 2

        if argmax:
            cam = torch.argmax(cams, dim=1).squeeze(0).float()  # (h, w)
        else:
            cam = torch.softmax(cams, dim=1)[:, 1, :, :].squeeze(0)  # (h, w)

        return cam

    def extra_repr(self) -> str:
        return f""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


if __name__ == "__main__":
    import itertools
    import datetime as dt
    from os.path import join
    import os
    import subprocess

    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    from dlib.dllogger import ArbJSONStreamBackend
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend
    from dlib.dllogger import ArbTextStreamBackend
    import dlib.dllogger as DLLogger

    from dlib.configure import constants
    from dlib import create_model
    from dlib.utils.shared import fmsg

    outd = join(root_dir, 'data/debug/cams')
    if not os.path.isdir(outd):
        os.makedirs(outd)

    exp_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    log_backends = [ArbJSONStreamBackend(
        Verbosity.VERBOSE, join(outd, "log-bcam-{}.json".format(exp_id))),
        ArbTextStreamBackend(
            Verbosity.VERBOSE,
            join(outd, "log-bcam-{}.txt".format(exp_id))),
        ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends)

    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    txt = subprocess.run(
        ['nvidia-smi', '--list-gpus'],
        stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    try:
        tag = txt[int(cuda)]
    except IndexError:
        tag = 'GPU'
    DLLogger.log('Device: {}'.format(tag))

    def test_dlib_models(encoder_name, support_background, pooling_head):
        set_seed(0)

        b, c, h, w = 1, 3, 224, 224
        x = torch.rand(b, c, h, w).to(DEVICE)
        nbr_classes = 100
        class_idx = 88

        DLLogger.log('x input shape: {}'.format(x.shape))
        DLLogger.log('NBR CLASSES: {}'.format(nbr_classes))

        if encoder_name in [constants.VGG16]:
            vgg_encoders = dlib.encoders.vgg_encoders
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
            decoder_channels = (256, 128, 64)
        else:
            encoder_depth = 5
            decoder_channels = (256, 128, 64, 32, 16)

        aux_params = {
            "pooling_head": pooling_head,
            "classes": nbr_classes,
            "modalities": 5,
            "kmax": 0.6,
            "kmin": 0.1,
            "alpha": 0.6,
            "dropout": 0.,
            "r": 10.,
            "support_background": support_background
        }
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            encoder_name=encoder_name,
            encoder_weights=constants.IMAGENET,
            in_channels=3,
            encoder_depth=encoder_depth,
            scale_in=1.,
            aux_params=aux_params
        )
        model.to(DEVICE)
        model.eval()

        DLLogger.log(fmsg('TEST DLIB models: {}'.format(encoder_name)))

        cam = BuiltinCam(model)

        DLLogger.log(fmsg('Testing {}: ENCODER: {} POOLINGHEAD: {} SUPPBACK: {}'
                     ''.format(cam, encoder_name, pooling_head,
                               support_background)))

        with torch.no_grad():
            scores = model(x)
            DLLogger.log("cl scores: {}".format(scores.shape))
        t0 = dt.datetime.now()
        pooled_cam = cam(class_idx=class_idx)
        DLLogger.log('x: {}, cam: {}'.format(x.shape, pooled_cam.shape))
        DLLogger.log(fmsg('time: {}'.format(dt.datetime.now() - t0)))


    encoders = [constants.RESNET50, constants.VGG16, constants.INCEPTIONV3]
    bgs = [True, False]
    poolings = [constants.GAP, constants.MAXPOOL, constants.WILDCATHEAD,
                constants.LSEPOOL]

    ll = [encoders, bgs, poolings]

    for el in itertools.product(*ll):
        test_dlib_models(*el)
