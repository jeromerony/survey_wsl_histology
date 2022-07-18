import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.unet.decoder import UnetDecoder
from dlib.unet.decoder import UnetFCAMDecoder
from dlib.unet.decoder import UnetNEGEVDecoder
from dlib.encoders import get_encoder
from dlib.base import SegmentationModel
from dlib.base import FCAMModel
from dlib.base import NEGEVModel
from dlib.base import SegmentationHead
from dlib.base import ReconstructionHead


from dlib import poolings

from dlib.configure import constants


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic
    segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract
    features of different spatial
    resolution (skip connections) which are used by decoder to define accurate
    segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone) to extract features of different spatial
            resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and other pretrained
        weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels**
            parameter for convolutions used in decoder. Length of the list
            should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. If **"inplace"** InplaceABN will
            be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask). Useful ONLY for the task
            constants.SEG.
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        scale_in: float = 1.
    ):
        super().__init__()

        assert task == constants.SEG
        self.task = task
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

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.cams = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class UnetFCAM(FCAMModel):
    """
    FCAMs using U-Net like.
    Unet_ is a fully convolution neural network for image semantic
    segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract
    features of different spatial
    resolution (skip connections) which are used by decoder to define accurate
    segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone) to extract features of different spatial
            resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and other pretrained
        weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels**
            parameter for convolutions used in decoder. Length of the list
            should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. If **"inplace"** InplaceABN will
            be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask). Useful ONLY for the task
            constants.SEG.
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        seg_h_out_channels: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        scale_in: float = 1.,
        freeze_cl: bool = False,
        im_rec: bool = False,
        img_range: str = constants.RANGE_TANH
    ):
        super().__init__()

        self.freeze_cl = freeze_cl
        assert task == constants.F_CL
        self.task = task
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        self.im_rec = im_rec
        self.img_range = img_range

        self.x_in = None

        self.encoder = get_encoder(
            task,
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetFCAMDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        assert aux_params is not None, 'ERROR'
        pooling_head = aux_params['pooling_head']
        aux_params.pop('pooling_head')

        self.classification_head = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=seg_h_out_channels,
            activation=activation,
            kernel_size=3,
        )

        self.reconstruction_head = None
        if self.im_rec:
            self.reconstruction_head = ReconstructionHead(
                in_channels=decoder_channels[-1],
                out_channels=in_channels,
                activation=self.img_range,
                kernel_size=3,
            )

        self.cams = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class UnetNEGEV(NEGEVModel):
    """
    NEGEV using U-Net like.
    Unet_ is a fully convolution neural network for image semantic
    segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract
    features of different spatial
    resolution (skip connections) which are used by decoder to define accurate
    segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone) to extract features of different spatial
            resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and other pretrained
        weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels**
            parameter for convolutions used in decoder. Length of the list
            should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. If **"inplace"** InplaceABN will
            be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask). Useful ONLY for the task
            constants.SEG.
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        seg_h_out_channels: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        scale_in: float = 1.,
        freeze_cl: bool = False,
        im_rec: bool = False,
        img_range: str = constants.RANGE_TANH
    ):
        super().__init__()

        self.freeze_cl = freeze_cl
        assert task == constants.NEGEV
        self.task = task
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        self.im_rec = im_rec
        self.img_range = img_range

        self.x_in = None

        self.encoder = get_encoder(
            task,
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetNEGEVDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        assert aux_params is not None, 'ERROR'
        pooling_head = aux_params['pooling_head']
        aux_params.pop('pooling_head')

        self.classification_head = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=seg_h_out_channels,
            activation=activation,
            kernel_size=3,
        )

        self.reconstruction_head = None
        if self.im_rec:
            self.reconstruction_head = ReconstructionHead(
                in_channels=decoder_channels[-1],
                out_channels=in_channels,
                activation=self.img_range,
                kernel_size=3,
            )

        self.cams = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()



def test_Unet():
    import datetime as dt

    from dlib.cams import build_seg_extractor
    from dlib.utils.tools import Dict2Obj

    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    encoders = [constants.RESNET50, constants.VGG16, constants.INCEPTIONV3]
    in_channels = 3
    SZ = 224
    bsz = 1
    sample = torch.rand((bsz, in_channels, SZ, SZ)).to(DEVICE)
    classes = 2
    loss = torch.nn.MSELoss(reduction='mean').to(DEVICE)
    target = torch.rand((bsz, classes, SZ, SZ)).to(DEVICE)
    args = Dict2Obj({'task': constants.SEG})

    for encoder_name in encoders:
        vgg_encoders = dlib.encoders.vgg_encoders

        if encoder_name == constants.VGG16:
            decoder_channels = (256, 128, 64)
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5

        announce_msg("Testing backbone {}".format(encoder_name))
        model = Unet(
            task=constants.SEG,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=None,
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=classes
        ).to(DEVICE)
        announce_msg("TESTING: {}".format(model))
        t0 = dt.datetime.now()
        out = model(sample)
        print('{}: forward time {}'.format(model, dt.datetime.now() - t0))
        t0 = dt.datetime.now()
        l = loss(out, target)
        print('{}: loss eval time {}'.format(model, dt.datetime.now() - t0))
        t0 = dt.datetime.now()
        l.backward()
        print('{}: loss backward time {}'.format(model, dt.datetime.now() - t0))
        print("x: {} \t mask: {}".format(sample.shape, out.shape))
        print(model.get_info_nbr_params())

        cambuilder = build_seg_extractor(model=model, args=args)
        model(sample)
        pooled_cam = cambuilder()
        print(print("x: {} \t mask: {} \t cam: {}".format(
            sample.shape, out.shape, pooled_cam.shape)))


def run_time_UnetFCAM():
    import datetime as dt
    import os
    import subprocess
    from os.path import join

    import torch.nn.functional as F
    from torchvision.models import resnet50
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    from dlib.dllogger import ArbJSONStreamBackend
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend
    from dlib.dllogger import ArbTextStreamBackend
    import dlib.dllogger as DLLogger
    from dlib.cams import build_fcam_extractor
    from dlib.utils.tools import Dict2Obj

    import dlib
    from dlib.configure import constants
    from dlib import create_model
    from dlib.utils.shared import fmsg

    outd = join(root_dir, 'data/debug/cams')
    if not os.path.isdir(outd):
        os.makedirs(outd)

    exp_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    log_backends = [ArbJSONStreamBackend(
        Verbosity.VERBOSE, join(outd, "log-unetfcam-{}.json".format(exp_id))),
        ArbTextStreamBackend(
            Verbosity.VERBOSE,
            join(outd, "log-gradcam-{}.txt".format(exp_id))),
        ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends)

    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    encoders = [constants.RESNET50, constants.VGG16, constants.INCEPTIONV3]
    SZ = 224
    in_channels = 3
    bsz = 32
    x = torch.rand((bsz, in_channels, SZ, SZ)).to(DEVICE)
    classes = 200

    amp = True

    seg_h_out_channels = classes
    args = Dict2Obj({'task': constants.F_CL})
    cam = build_fcam_extractor

    for encoder_name in encoders:
        # vgg16
        # if encoder_name != constants.VGG16:
        #     continue

        vgg_encoders = dlib.encoders.vgg_encoders

        if encoder_name == constants.VGG16:
            decoder_channels = (256, 128, 64)
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5

        support_background = True
        im_rec = False

        model = UnetFCAM(
            task=constants.F_CL,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            encoder_weights=constants.IMAGENET,
            in_channels=in_channels,
            seg_h_out_channels=2,
            aux_params=dict(pooling_head=constants.WGAP,
                            classes=classes,
                            support_background=support_background),
            im_rec=im_rec,
            img_range=constants.RANGE_SIGMOID
        ).to(DEVICE)

        DLLogger.log(fmsg("TESTING: {} -- {} ]n {}".format(
            model, amp, model.get_info_nbr_params())))

        cambuilder = build_fcam_extractor(model=model, args=args)
        for __ in [1, 2]:
            with autocast(enabled=amp):
                model(x)

        t0 = dt.datetime.now()
        with autocast(enabled=amp):
            out = model(x)
        print('forward time {}'.format(dt.datetime.now() - t0))

        # to run cambuilder,
        # comment SegmentationCam.__init__: self.assert_model(model).
        # todo: solve later. __main__.UnetFCAM exception.
        pooled_cam = cambuilder()

        if pooled_cam.shape != (SZ, SZ):
            full_cam = F.interpolate(
                input=pooled_cam.unsqueeze(0).unsqueeze(0),
                size=[SZ, SZ],
                mode='bilinear',
                align_corners=True)
        DLLogger.log(fmsg('time (forward + build+ interpolation) [{}]: '
                          '{}'.format(encoder_name,
                                      dt.datetime.now() - t0)))

def test_UnetFCAM():
    import datetime as dt
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    encoders = [constants.INCEPTIONV3, constants.VGG16, constants.RESNET50]
    SZ = 224
    in_channels = 3
    bsz = 8
    sample = torch.rand((bsz, in_channels, SZ, SZ)).to(DEVICE)
    classes = 2
    loss = torch.nn.CrossEntropyLoss(reduction='mean').to(DEVICE)

    seg_h_out_channels = classes

    for encoder_name in encoders:
        # vgg16
        # if encoder_name != constants.VGG16:
        #     continue

        vgg_encoders = dlib.encoders.vgg_encoders

        if encoder_name == constants.VGG16:
            decoder_channels = (256, 128, 64)
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5

        announce_msg("Testing backbone {}".format(encoder_name))
        for support_background in [False, True]:
            for im_rec in [True, False]:
                target = torch.randint(low=0, high=classes,
                                       size=(bsz, SZ, SZ)).to(DEVICE)

                model = UnetFCAM(
                    task=constants.F_CL,
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    decoder_channels=decoder_channels,
                    encoder_weights=constants.IMAGENET,
                    in_channels=in_channels,
                    seg_h_out_channels=seg_h_out_channels,
                    aux_params=dict(pooling_head=constants.WILDCATHEAD,
                                    classes=classes,
                                    support_background=support_background),
                    im_rec=im_rec,
                    img_range=constants.RANGE_SIGMOID
                ).to(DEVICE)
                announce_msg("TESTING: {} -- ]n {}".format(
                    model, model.get_info_nbr_params()))
                glabel = torch.randint(low=0, high=classes, size=(bsz,),
                                       dtype=torch.long, device=DEVICE)
                t0 = dt.datetime.now()
                out = model(sample)
                cl_logits, fcams, im_recon = out
                cams_low = model.classification_head.cams
                print('FCAMs shape: {}'.format(fcams.shape))

                print('{}: forward time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))
                t0 = dt.datetime.now()
                l = loss(out[1], target)
                print('{}: loss eval time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))
                t0 = dt.datetime.now()
                l.backward()
                print('{}: loss backward time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))

                print(
                    "x: {} \t cams_low: {} \t  \t cl_logits: "
                    "{} \t fcam: {}".format(sample.shape, cams_low.shape,
                                            cl_logits.shape, fcams.shape))
                if im_rec:
                    print('IMAGE reconstruction: '
                          'x shape: {}, im recon.shape: {}'.format(
                           sample.shape, im_recon.shape))

        # return 0


def test_UnetNEGEV():
    import datetime as dt
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    encoders = [constants.INCEPTIONV3, constants.VGG16, constants.RESNET50]
    SZ = 224
    in_channels = 3
    bsz = 8
    sample = torch.rand((bsz, in_channels, SZ, SZ)).to(DEVICE)
    classes = 2
    loss = torch.nn.CrossEntropyLoss(reduction='mean').to(DEVICE)

    seg_h_out_channels = classes

    for encoder_name in encoders:
        # vgg16
        # if encoder_name != constants.VGG16:
        #     continue

        vgg_encoders = dlib.encoders.vgg_encoders

        if encoder_name == constants.VGG16:
            decoder_channels = (256, 128, 64)
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5

        announce_msg("Testing backbone {}".format(encoder_name))
        for support_background in [False, True]:
            for im_rec in [True, False]:
                target = torch.randint(low=0, high=classes,
                                       size=(bsz, SZ, SZ)).to(DEVICE)

                model = UnetNEGEV(
                    task=constants.NEGEV,
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    decoder_channels=decoder_channels,
                    encoder_weights=constants.IMAGENET,
                    in_channels=in_channels,
                    seg_h_out_channels=seg_h_out_channels,
                    aux_params=dict(pooling_head=constants.WILDCATHEAD,
                                    classes=classes,
                                    support_background=support_background),
                    im_rec=im_rec,
                    img_range=constants.RANGE_SIGMOID
                ).to(DEVICE)
                announce_msg("TESTING: {} -- ]n {}".format(
                    model, model.get_info_nbr_params()))
                glabel = torch.randint(low=0, high=classes, size=(bsz,),
                                       dtype=torch.long, device=DEVICE)
                t0 = dt.datetime.now()
                out = model(sample)
                cl_logits, fcams, im_recon = out
                cams_low = model.classification_head.cams
                print('CAMs shape: {}'.format(fcams.shape))

                print('{}: forward time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))
                t0 = dt.datetime.now()
                l = loss(out[1], target)
                print('{}: loss eval time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))
                t0 = dt.datetime.now()
                l.backward()
                print('{}: loss backward time {} [SUPBACK {}]'.format(
                    model, dt.datetime.now() - t0, support_background))

                print(
                    "x: {} \t cams_low: {} \t  \t cl_logits: "
                    "{} \t fcam: {}".format(sample.shape, cams_low.shape,
                                            cl_logits.shape, fcams.shape))
                if im_rec:
                    print('IMAGE reconstruction: '
                          'x shape: {}, im recon.shape: {}'.format(
                           sample.shape, im_recon.shape))

        # return 0


if __name__ == "__main__":
    import dlib
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    # test_Unet()

    set_seed(0)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = False

    # test_UnetFCAM()

    set_seed(0)
    # torch.backends.cudnn.benchmark = False
    # run_time_UnetFCAM()

    set_seed(0)
    torch.backends.cudnn.benchmark = False
    test_UnetNEGEV()



