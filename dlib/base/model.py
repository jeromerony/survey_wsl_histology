import sys
from os.path import dirname, abspath
import datetime as dt

import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.base import initialization as init
from dlib.utils.shared import count_params


class STDClModel(torch.nn.Module):
    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

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

        self.x_in = x
        features = self.encoder(x)
        cl_logits = self.classification_head(features[-1])
        return cl_logits

    def __str__(self):
        return "{}. Task: {}.".format(self.name, self.task)

    def get_info_nbr_params(self) -> str:
        info = self.__str__() + ' \n NBR-PARAMS: \n'
        if self.encoder:
            info += '\tEncoder [{}]: {}. \n'.format(self.encoder.name,
                                                    count_params(self.encoder))
        if self.classification_head:
            info += '\tClassification head [{}]: {}. \n'.format(
                self.classification_head.name,
                count_params(self.classification_head))
        info += '\tTotal: {}. \n'.format(count_params(self))

        return info


class FCAMModel(torch.nn.Module):
    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

        if self.decoder is not None:
            init.initialize_decoder(self.decoder)

        if self.segmentation_head is not None:
            init.initialize_head(self.segmentation_head)

        if self.reconstruction_head is not None:
            init.initialize_head(self.reconstruction_head)

    def get_reconstructed_img(self, *features):
        assert self.reconstruction_head
        return self.reconstruction_head(self.decoder.forward_reconstruction(
            *features))

    def forward(self, x):
        x_shape = x.shape

        if self.scale_in != 1.:
            raise ValueError

            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        features = self.encoder(x)
        if self.freeze_cl:
            features = [f.detach() for f in features]

        cl_logits = self.classification_head(features[-1])
        decoder_output = self.decoder(*features)
        fcams = self.segmentation_head(decoder_output)

        if fcams.shape[2:] != x_shape[2:]:
            fcams = F.interpolate(
                input=fcams,
                size=x_shape[2:],
                mode='bilinear',
                align_corners=True
            )

        im_recon = None
        if self.im_rec:
            im_recon = self.get_reconstructed_img(*features)

        self.cams = fcams.detach()

        return cl_logits, fcams, im_recon

    def train(self, mode=True):
        super(FCAMModel, self).train(mode=mode)

        if self.freeze_cl:
            self.freeze_classifier()

        return self

    def freeze_classifier(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def assert_cl_is_frozen(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

        return True

    def __str__(self):
        return "{}. Task: {}. Supp.BACK: {}. Freeze CL: {}. " \
               "IMG-RECON: {}:".format(
                self.name, self.task,
                self.classification_head.support_background,
                self.freeze_cl, self.im_rec
                )

    def get_info_nbr_params(self) -> str:
        info = self.__str__() + ' \n NBR-PARAMS: \n'

        if self.encoder:
            info += '\tEncoder [{}]: {}. \n'.format(
                self.encoder.name,  count_params(self.encoder))

        if self.classification_head:
            info += '\tClassification head [{}]: {}. \n'.format(
                self.classification_head.name,
                count_params(self.classification_head))

        if self.decoder:
            info += '\tDecoder: {}. \n'.format(
                count_params(self.decoder))

        if self.segmentation_head:
            info += '\tSegmentation head: {}. \n'.format(
                count_params(self.classification_head))

        if self.reconstruction_head:
            info += '\tReconstruction head: {}. \n'.format(
                count_params(self.reconstruction_head))

        info += '\tTotal: {}. \n'.format(count_params(self))

        return info


class NEGEVModel(FCAMModel):
    pass


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        if self.decoder is not None:
            init.initialize_decoder(self.decoder)
        if self.segmentation_head is not None:
            init.initialize_head(self.segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
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

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks_logits = self.segmentation_head(decoder_output)

        if masks_logits.shape[2:] != x_shape[2:]:
            masks_logits = F.interpolate(
                input=masks_logits,
                size=x_shape[2:],
                mode='bilinear',
                align_corners=True
            )

        self.cams = masks_logits.detach()

        return masks_logits

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes,
            height, width) # todo: change the description.

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            out = self.forward(x)

        return out

    def __str__(self):
        return "{}. Task: {}.".format(self.name, self.task)

    def get_info_nbr_params(self) -> str:
        info = self.__str__() + ' \n NBR-PARAMS: \n'

        if self.encoder:
            info += '\tEncoder [{}]: {}. \n'.format(
                self.encoder.name,  count_params(self.encoder))

        if self.decoder:
            info += '\tDecoder: {}. \n'.format(
                count_params(self.decoder))

        if self.segmentation_head:
            info += '\tSegmentation head: {}. \n'.format(
                count_params(self.segmentation_head))

        info += '\tTotal: {}. \n'.format(count_params(self))

        return info
