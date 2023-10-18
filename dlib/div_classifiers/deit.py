import torch
import torch.nn as nn
from functools import partial

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


from dlib.div_classifiers.vision_transformer import VisionTransformer, _cfg
from dlib.div_classifiers import vision_transformer

IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)

__all__ = [
    'deit_tscam_tiny_patch16_224',
    'deit_tscam_small_patch16_224',
    'deit_tscam_base_patch16_224',
]


class TSCAM(VisionTransformer):
    def __init__(self, encoder_name: str, encoder_weights: str, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3,
                              stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

        self.encoder_name: str = encoder_name
        self.task: str = constants.STD_CL
        self.scale_in: float = 1.

        self.x_in = None
        self._out_channels = []
        self._in_channels = None

        self.classification_head = None

        self.name = "u-{}".format(self.encoder_name)
        self.encoder_weights = encoder_weights
        self.cams = None

        self.method = constants.METHOD_TSCAM
        self.arch = constants.TSCAMCLASSIFIER

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

    def _init_load_pretrained_w(self):
        if self.encoder_name == constants.DEIT_TSCAM_TINY_P16_224:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )['model']
            model_dict = self.state_dict()

            for k in ['head.weight', 'head.bias', 'head_dist.weight',
                      'head_dist.bias']:
                if k in checkpoint and checkpoint[k].shape != model_dict[
                    k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]

            pretrained_dict = {k: v for k, v in checkpoint.items() if
                               k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        if self.encoder_name == constants.DEIT_TSCAM_SMALL_P16_224:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )['model']
            model_dict = self.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight',
                      'head_dist.bias']:
                if k in checkpoint and checkpoint[k].shape != model_dict[
                    k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            pretrained_dict = {k: v for k, v in checkpoint.items() if
                               k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        if self.encoder_name == constants.DEIT_TSCAM_BASE_P16_224:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )['model']
            model_dict = self.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight',
                      'head_dist.bias']:
                if k in checkpoint and checkpoint[k].shape != model_dict[
                    k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            pretrained_dict = {k: v for k, v in checkpoint.items() if
                               k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/
        # master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, labels=None):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        if labels is not None:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            #cams = attn_weights.mean(0).mean(1)[:, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights[:-1].sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights.sum(0)[:][:, 1:, 1:].sum(1).reshape([n, h, w]).unsqueeze(1)
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            cams = cams * feature_map                           # B * C * 14 * 14

            self.cams = cams.detach()[:, labels, :, :].squeeze(1)  # b, 14, 14

        return x_logits

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


def deit_tscam_tiny_patch16_224(encoder_weights=constants.IMAGENET,
                                pretrained=False,
                                **kwargs):
    model = TSCAM(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_TSCAM_TINY_P16_224,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def deit_tscam_small_patch16_224(encoder_weights=constants.IMAGENET,
                                 pretrained=False, **kwargs):
    model = TSCAM(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_TSCAM_SMALL_P16_224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def deit_tscam_base_patch16_224(encoder_weights=constants.IMAGENET,
                                pretrained=False, **kwargs):
    model = TSCAM(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_TSCAM_BASE_P16_224,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def test_TSCAM():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = False
    num_classes = 200

    for encoder in [constants.DEIT_TSCAM_BASE_P16_224,
                    constants.DEIT_TSCAM_SMALL_P16_224,
                    constants.DEIT_TSCAM_TINY_P16_224]:

        if encoder == constants.DEIT_TSCAM_TINY_P16_224:
            model = deit_tscam_tiny_patch16_224(pretrained=pretrained,
                                                num_classes=num_classes)

        if encoder == constants.DEIT_TSCAM_SMALL_P16_224:
            model = deit_tscam_small_patch16_224(pretrained=pretrained,
                                                 num_classes=num_classes)

        if encoder == constants.DEIT_TSCAM_BASE_P16_224:
            model = deit_tscam_base_patch16_224(pretrained=pretrained,
                                                num_classes=num_classes)

        model.to(device)
        print(model.get_info_nbr_params())
        bsize = 1
        h, w = 224, 224
        x = torch.rand(bsize, 3, 224, 224).to(device)
        labels = torch.zeros((bsize,), dtype=torch.long)
        model(x)
        # print(f'logits shape : {logits.shape} x : {x.shape} '
        #       f'classes : {num_classes}')

        t0 = dt.datetime.now()
        model(x, labels=labels)
        cams = model.cams
        print(cams.shape, x.shape)
        # if cams.shape != (1, h, w):
        #     tx = dt.datetime.now()
        #     full_cam = F.interpolate(
        #         input=cams.unsqueeze(0),
        #         size=[h, w],
        #         mode='bilinear',
        #         align_corners=True)
        # print(x.shape, cams.shape)
        print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    test_TSCAM()
