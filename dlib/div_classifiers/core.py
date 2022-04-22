import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

import sys
from os.path import dirname, abspath, join

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import count_params

import torch


__all__ = ['CoreClassifier']


class CoreClassifier(torch.nn.Module):
    """
    Standard classifier.
    """

    def __init__(self):
        super().__init__()

        self.encoder_name: str = None
        self.task: str = constants.STD_CL
        self.scale_in: float = 1.

        self.x_in = None
        self._out_channels = []
        self._in_channels = None

        self.classification_head = None

        self.name = "u-{}".format(self.encoder_name)
        self.encoder_weights = None
        self.cams = None
        self.method = None
        self.arch = None

    def __str__(self):
        return "{}. Task: {}.".format(self.name, self.task)

    def get_info_nbr_params(self) -> str:
        totaln = count_params(self)
        cl_head_n = 0
        if self.classification_head:
            cl_head_n = count_params(self.classification_head)

        info = self.__str__() + ' \n NBR-PARAMS: \n'

        info += '\tEncoder [{}]: {}. \n'.format(self.name, totaln - cl_head_n)
        if self.classification_head:
            info += '\tClassification head [{}]: {}. \n'.format(
                self.classification_head.name, cl_head_n)
        info += '\tTotal: {}. \n'.format(totaln)

        return info

    @property
    def out_channels(self) -> list:
        """Return channels dimensions for each tensor of forward output of
        encoder"""
        return self._out_channels

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        raise NotImplementedError

    def set_model_name(self, name: str):
        self.name: str = name

    def set_task(self, task: str):
        self.task: str = task
