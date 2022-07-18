import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from functools import partial

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib
from dlib.stdcl.classifier import STDClassifier
from dlib.poolings import WGAP

import dlib.dllogger as DLLogger

""" Source: https://github.com/ml-edu/torch-cam """


__all__ = ['_CAM', 'locate_linear_layer', 'locate_candidate_layer']


class _CAM:
    """Implements a class activation map extractor

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        self.assert_model(model)

        # Obtain a mapping from module name to module instance for each layer
        # in the model
        self.submodule_dict = dict(model.named_modules())

        # If the layer is not specified, try automatic resolution
        if target_layer is None:
            target_layer = locate_candidate_layer(model, input_shape)
            # Warn the user of the choice
            if isinstance(target_layer, str):
                DLLogger.log("no value was provided for `target_layer`, "
                             "thus set to `{}`.".format(target_layer))
            else:
                raise ValueError("unable to resolve `target_layer` "
                                 "automatically, please specify its value.")

        if target_layer not in self.submodule_dict.keys():
            raise ValueError(
                f"Unable to find submodule {target_layer} in the model")
        self.target_layer = target_layer
        self.model = model
        # Init hooks
        self.hook_a: Optional[Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Forward hook
        self.hook_handles.append(
            self.submodule_dict[target_layer].register_forward_hook(
                self._hook_a))
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    @staticmethod
    def assert_model(model: STDClassifier) -> None:

        if not isinstance(model, STDClassifier):
            return

        assert any([isinstance(model.encoder,
                               dlib.encoders.resnet.ResNetEncoder),
                    isinstance(model.encoder, dlib.encoders.vgg.VGGEncoder),
                    isinstance(model.encoder,
                               dlib.encoders.inceptionv3.InceptionV3Encoder)])
        assert isinstance(model.classification_head, WGAP)

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def clear_hooks(self) -> None:
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:

        raise NotImplementedError

    def _precheck(self, class_idx: int, scores: Optional[Tensor] = None) -> None:
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if not isinstance(self.hook_a, Tensor):
            raise AssertionError("Inputs need to be forwarded in the "
                                 "model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. "
                             f"Received: {self.hook_a.shape[0]}")

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be "
                             "passed to compute CAMs")

    def __call__(self,
                 class_idx: int,
                 scores: Optional[Tensor] = None,
                 normalized: bool = True,
                 reshape: Optional[Tuple] = None,
                 argmax: Optional[bool] = False) -> Tensor:

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM: (h, w)
        cam = self.compute_cams(class_idx, scores, normalized)
        if reshape is not None:
            assert len(reshape) == 2
            interpolation_mode = 'bilinear'
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                reshape,
                                mode=interpolation_mode,
                                align_corners=False).squeeze(0).squeeze(0)
        cam = cam.detach()

        return cam

    def compute_cams(self, class_idx: int, scores: Optional[Tensor] = None,
                     normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM
               will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the
               hooked model
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        # Get map weight & unsqueeze it
        weights = self._get_weights(class_idx, scores)
        missing_dims = self.hook_a.ndim - weights.ndim - 1
        weights = weights[(...,) + (None,) * missing_dims]

        # Perform the weighted combination to get the CAM
        batch_cams = torch.nansum(weights * self.hook_a.squeeze(0), dim=0)
        # type: ignore[union-attr]

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams

    def extra_repr(self) -> str:
        return f"target_layer='{self.target_layer}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


def locate_candidate_layer(mod: nn.Module,
                           input_shape: Tuple[int, ...] = (3, 224, 224)
                           ) -> Optional[str]:
    """Attempts to find a candidate layer to use for CAM extraction

    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch
          dimension

    Returns:
        str: the candidate layer for CAM
    """

    # Set module in eval mode
    module_mode = mod.training
    mod.eval()

    output_shapes: List[Tuple[Optional[str], Tuple[int, ...]]] = []

    def _record_output_shape(module: nn.Module, input: Tensor, output: Tensor,
                             name: Optional[str] = None) -> None:
        """Activation hook"""
        output_shapes.append((name, output.shape))

    hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    # forward hook on all layers
    for n, m in mod.named_modules():
        hook_handles.append(m.register_forward_hook(
            partial(_record_output_shape, name=n)))

    # forward empty
    with torch.no_grad():
        _ = mod(torch.rand(1, *input_shape,
                           device=next(mod.parameters()).data.device))

    # Remove all temporary hooks
    for handle in hook_handles:
        handle.remove()

    # Put back the model in the corresponding mode
    mod.training = module_mode

    # Check output shapes
    candidate_layer = None
    for layer_name, output_shape in output_shapes:
        # Stop before flattening or global pooling
        if len(output_shape) != (len(input_shape) + 1) or all(
                v == 1 for v in output_shape[2:]):
            break
        else:
            candidate_layer = layer_name

    return candidate_layer


def locate_linear_layer(mod: nn.Module) -> Optional[str]:
    """Attempts to find a fully connecter layer to use for CAM extraction

    Args:
        mod: the module to inspect

    Returns:
        str: the candidate layer
    """

    candidate_layer = None
    for layer_name, m in mod.named_modules():
        if isinstance(m, nn.Linear):
            candidate_layer = layer_name
            break

    return candidate_layer
