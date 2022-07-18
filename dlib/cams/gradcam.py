# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for
# full license details.

import sys
from os.path import dirname, abspath, join

import torch
from torch import Tensor
from typing import Optional, Tuple

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger

from dlib.cams.core import _CAM

""" Source: https://github.com/ml-edu/torch-cam """


__all__ = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM']


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[str] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)
        # Init hook
        self.hook_g: Optional[Tensor] = None
        # Ensure ReLU is applied before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = True
        # cf. https://github.com/pytorch/pytorch/pull/46163
        if torch.__version__ >= '1.8.0':
            bw_hook = 'register_full_backward_hook'
        else:
            bw_hook = 'register_backward_hook'
        # Backward hook
        self.hook_handles.append(
            getattr(self.submodule_dict[self.target_layer],
                    bw_hook)(self._hook_g))

    def _hook_g(self, module: torch.nn.Module, input: Tensor,
                output: Tensor) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_g = output[0].data

    def _backprop(self, scores: Tensor, class_idx: int) -> None:
        """Backpropagate the loss for a specific output class"""

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for"
                            " the conv features to be hooked")

        # Backpropagate to get the gradients on the hooked layer
        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, class_idx, scores):

        raise NotImplementedError


class GradCAM(_GradCAM):
    """Implements a class activation map extractor as described in
    `"Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
        A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{H \\cdot W} \\sum\\limits_{i=1}^H
        \\sum\\limits_{j=1}^W \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before
    softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import GradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.squeeze(0).flatten(1).mean(-1)


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in
    `"Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks" <https://arxiv.org/pdf/1710.11063.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W
        \\alpha_k^{(c)}(i, j) \\cdot
         ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before
    softmax, and :math:`\\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \\alpha_k^{(c)}(i, j) = \\frac{1}{\\sum\\limits_{i, j} \\frac{\\partial
        Y^{(c)}}{\\partial A_k(i, j)}}
        = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot
        \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} +
        \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}}

    if :math:`\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1` else :math:`0`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import GradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.pow(2)
        grad_3 = grad_2 * self.hook_g
        # Watch out for NaNs produced by underflow
        spatial_dims = self.hook_a.ndim - 2  # type: ignore[union-attr]
        denom = 2 * grad_2 + (grad_3 * self.hook_a).flatten(2).sum(-1)[
            (...,) + (None,) * spatial_dims]
        nan_mask = grad_2 > 0
        alpha = grad_2
        alpha[nan_mask].div_(denom[nan_mask])

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(
            torch.relu(self.hook_g.squeeze(0))).flatten(1).sum(-1)


class SmoothGradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in
    `"Smooth Grad-CAM++: An Enhanced Inference Level
    Visualization Technique for Deep Convolutional Neural Network Models"
    <https://arxiv.org/pdf/1908.01224.pdf>`_
    with a personal correction to the paper (alpha coefficient numerator).

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Smooth Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W
        \\alpha_k^{(c)}(i, j) \\cdot
        ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before
    softmax, and :math:`\\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \\alpha_k^{(c)}(i, j)
        = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot
        \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} +
        \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}}
        = \\frac{\\frac{1}{n} \\sum\\limits_{m=1}^n D^{(c, 2)}_k(i, j)}{
        \\frac{2}{n} \\sum\\limits_{m=1}^n D^{(c, 2)}_k(i, j) +
        \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{1}{n} \\sum\\limits_{m=1}^n D^{(c, 3)}_k(i, j)}

    if :math:`\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1` else
    :math:`0`. Here :math:`D^{(c, p)}_k(i, j)`
    refers to the p-th partial derivative of the class score of class :math:`c`
    relatively to the activation in layer
    :math:`k` at position :math:`(i, j)`, and :math:`n` is the number of samples
    used to get the gradient estimate.

    Please note the difference in the numerator of
    :math:`\\alpha_k^{(c)}(i, j)`,
    which is actually :math:`\\frac{1}{n} \\sum\\limits_{k=1}^n
    D^{(c, 1)}_k(i,j)` in the paper.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import SmoothGradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SmoothGradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        num_samples: number of samples to use for smoothing
        std: standard deviation of the noise
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[str] = None,
        num_samples: int = 4,
        std: float = 0.3,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)
        # Model scores is not used by the extractor
        self._score_used = False

        # Input hook
        self.hook_handles.append(
            model.register_forward_pre_hook(self._store_input))
        # Noise distribution
        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        # Specific input hook updater
        self._ihook_enabled = True

    def _store_input(self, module: torch.nn.Module, input: Tensor) -> None:
        """Store model input tensor"""

        if self._ihook_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_a: Tensor
        self.hook_g: Tensor
        # Disable input update
        self._ihook_enabled = False
        # Keep initial activation
        init_fmap = self.hook_a.clone()
        # Initialize our gradient estimates
        grad_2, grad_3 = torch.zeros_like(self.hook_a), torch.zeros_like(
            self.hook_a)
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(
                self._input.size()).to(device=self._input.device)
            # Forward & Backward
            out = self.model(noisy_input)
            self.model.zero_grad()
            self._backprop(out, class_idx)

            # Sum partial derivatives
            grad_2.add_(self.hook_g.pow(2))
            grad_3.add_(self.hook_g.pow(3))

        # Reenable input update
        self._ihook_enabled = True

        # Average the gradient estimates
        grad_2.div_(self.num_samples)
        grad_3.div_(self.num_samples)

        # Alpha coefficient for each pixel
        spatial_dims = self.hook_a.ndim - 2
        alpha = grad_2 / (2 * grad_2 + (grad_3 * init_fmap).flatten(2).sum(-1)[
            (...,) + (None,) * spatial_dims])

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(
            torch.relu(self.hook_g.squeeze(0))).flatten(1).sum(-1)

    def extra_repr(self) -> str:
        return f"target_layer='{self.target_layer}', " \
               f"num_samples={self.num_samples}, std={self.std}"


class XGradCAM(_GradCAM):
    """Implements a class activation map extractor as described in
    `"Axiom-based Grad-CAM: Towards Accurate
    Visualization and Explanation of CNNs"
    <https://arxiv.org/pdf/2008.02312.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{XGrad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
         A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W
        \\Big( \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} \\cdot
        \\frac{A_k(i, j)}{\\sum\\limits_{m=1}^H \\sum\\limits_{n=1}^W
        A_k(m, n)} \\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c`
    before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import XGradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = XGradCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_a: Tensor
        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)

        return (self.hook_g * self.hook_a
                ).squeeze(0).flatten(1).sum(-1) / self.hook_a.squeeze(
            0).flatten(1).sum(-1)


class LayerCAM(_GradCAM):
    """Implements a class activation map extractor as described in
    `"LayerCAM: Exploring Hierarchical Class Activation
    Maps for Localization"
    <http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Layer-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}(x, y)
        \\cdot A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}(x, y)` being defined as:

    .. math::
        w_k^{(c)}(x, y) = ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial
         A_k(i, j)}(x, y)\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before
     softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import LayerCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = LayerCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
          dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)

        return torch.relu(self.hook_g).squeeze(0)


if __name__ == "__main__":
    import datetime as dt
    import os
    import subprocess

    import torch.nn.functional as F
    from torch.cuda.amp import autocast

    from torchvision.models import resnet50
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    from dlib.dllogger import ArbJSONStreamBackend
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend
    from dlib.dllogger import ArbTextStreamBackend

    import dlib
    from dlib.configure import constants
    from dlib import create_model
    from dlib.utils.shared import fmsg

    amp = False

    outd = join(root_dir, 'data/debug/cams')
    if not os.path.isdir(outd):
        os.makedirs(outd)

    exp_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    log_backends = [ArbJSONStreamBackend(
        Verbosity.VERBOSE, join(outd, "log-gradcam-{}.json".format(exp_id))),
                    ArbTextStreamBackend(
                        Verbosity.VERBOSE,
                        join(outd, "log-gradcam-{}.txt".format(exp_id))),
                    ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends)

    model_names = {resnet50: 'resnet50'}
    mthds = [GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM]

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
    DLLogger.log('AMP: {}'.format(amp))

    def test_torchvision_models(modelx=resnet50):
        assert modelx == resnet50

        set_seed(0)

        b, c, h, w = 1, 3, 224, 224
        x = torch.rand(b, c, h, w).to(DEVICE)
        DLLogger.log('x input shape: {}'.format(x.shape))
        DLLogger.log('NBR CLASSES: {}'.format(1000))

        # test original.
        model = modelx(pretrained=True).eval()
        mname = model_names[modelx]
        DLLogger.log(fmsg('TEST torchvision models: {}'.format(mname)))

        model.to(DEVICE)
        target_layer = None

        for method in mthds:
            cam = method(model, target_layer=target_layer)
            DLLogger.log(fmsg('Testing: {} [{}]'.format(cam, mname)))

            scores = model(x)
            DLLogger.log('cl scores shape: {}'.format(scores.shape))
            t0 = dt.datetime.now()
            pooled_cam = cam(class_idx=100, scores=scores)
            DLLogger.log('x: {}, cam: {}'.format(x.shape, pooled_cam.shape))
            DLLogger.log(fmsg('time: {}'.format(dt.datetime.now() - t0)))
            DLLogger.flush()


    def test_dlib_models(encoder_name):
        set_seed(0)

        b, c, h, w = 1, 3, 224, 224
        x = torch.rand(b, c, h, w).to(DEVICE)
        nbr_classes = 200
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

        support_background = True
        aux_params = {
            "pooling_head": constants.WGAP,
            "classes": nbr_classes,
            "modalities": 5,
            "kmax": 0.6,
            "kmin": 0.1,
            "alpha": 0.6,
            "dropout": 0.,
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
        model(x)

        # for name, layer in model.named_modules():
        #     print('name', name, type(layer))
        # sys.exit()

        DLLogger.log(fmsg('TEST DLIB models: {}'.format(encoder_name)))

        target_layer = constants.TRG_LAYERS[encoder_name]
        DLLogger.log(fmsg('Model log: {}'.format(model.get_info_nbr_params())))

        for method in mthds:
            cam = method(model, target_layer=target_layer)
            DLLogger.log(fmsg('Testing: {} [{}]'.format(cam, encoder_name)))

            scores = model(x)

            t0 = dt.datetime.now()
            with autocast(enabled=amp):
                pooled_cam = cam(class_idx=class_idx, scores=scores)

            if amp:
                pooled_cam = pooled_cam.float()

            if pooled_cam.shape != (h, w):
                tx = dt.datetime.now()
                full_cam = F.interpolate(
                        input=pooled_cam.unsqueeze(0).unsqueeze(0),
                        size=[h, w],
                        mode='bilinear',
                        align_corners=True)
                DLLogger.log('time interpolation {}'.format(dt.datetime.now()
                                                            - tx))
            DLLogger.log(fmsg('time (build+ interpolation) [{}, {}]: '
                              '{}'.format(cam, encoder_name,
                                          dt.datetime.now() - t0)))

            DLLogger.log('cl scores shape: {}'.format(scores.shape))
            features = model.encoder(x)
            DLLogger.log(fmsg('FNBR/size feature maps [{} {}]: {}'.format(
                cam, encoder_name, features[-1].shape)))
            DLLogger.log('x: {}, cam: {}'.format(x.shape, pooled_cam.shape))

            DLLogger.flush()


    # test_torchvision_models()

    test_dlib_models(constants.RESNET50)
    # test_dlib_models(constants.VGG16)
    # test_dlib_models(constants.INCEPTIONV3)
