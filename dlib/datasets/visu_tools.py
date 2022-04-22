import sys
import os
from os import path
from os.path import join
from os.path import dirname
from os.path import abspath
from copy import deepcopy
from random import shuffle

import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib as mlp
import yaml



sys.path.append(dirname(dirname(abspath(__file__))))

from prologues import get_validationset
from prologues import get_csv_files

from parseit import Dict2Obj

from tools import get_rootpath_2_dataset
from tools import VisualiseMIL

from shared import csv_loader

import constants

from create_folds import cityscapes_labels as cslabels


__all__ = [
    "VisualizeImages",
    "VisualizeImagesMultiLabel",
    "see_multi_class_ds",
    "see_multi_label_ds",
    "create_thumbnail"
]


class VisualizeImages(VisualiseMIL):
    """
    Visualize images from dataset with multi-class.
    """
    def create_tag_paper(self, wim, msg, font=None):
        """
        Craeate a VISIBLE tag for the paper.

        :param wim: int, image width.
        :param msg: message (str) to display.
        :return:
        """
        if font is None:
            font = self.font_bold_paper

        img_tag = Image.new("RGB", (wim, self.height_tag_paper), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, 0, msg, self.white, font, self.dx)

        return img_tag

    def __call__(self,
                 name_classes,
                 list_images,
                 list_true_masks,
                 list_labels,
                 rows,
                 columns,
                 show_tags=False
                 ):
        """

        :param name_classes:
        :param list_images:
        :param list_true_masks:
        :return:
        """
        for i, msk in enumerate(list_true_masks):
            msg = "'mask' must be `{}` or None, but we found `{}` " \
                  ".... [NOT OK]".format(np.ndarray, type(msk))
            assert isinstance(msk, np.ndarray), msg
        for i, img in enumerate(list_images):
            msg = "'input_image' type must be `{}`, but we found `{}` " \
                  ".... [NOT OK]".format(Image.Image, type(img))
            assert isinstance(img, Image.Image), msg

        msg = "'name_classes' must be of type `{}`, but we found `{}` " \
              ".... [NOT OK]".format(dict, type(name_classes))
        assert isinstance(name_classes, dict), msg

        assert rows == 1, "We support only 1 row!!!! You asked for {}".format(
            rows)
        msg = "list_images and list_true_masks must have the same number " \
              "of elements. You provided: len(list_images) = {}, " \
              "len(list_true_masks) = {}".format(len(list_images),
                                                 len(list_true_masks))
        assert len(list_images) == len(list_true_masks), msg

        nbr_imgs = len(list_images)
        extra_w_space = self.space * (nbr_imgs - 1)
        w_out = 0
        max_h = 0
        for im in list_images:
            w_out += im.size[0]
            max_h = max(max_h, im.size[1])

        w_out += extra_w_space
        img_out = Image.new("RGB", (w_out, max_h))
        img_tags = Image.new("RGB", (w_out, self.height_tag_paper))
        i = 0
        p = 0
        for im, msk in zip(list_images, list_true_masks):
            wim = im.size[0]
            tmp = self.convert_mask_into_heatmap(im, msk, binarize=False)
            img_out.paste(tmp, (p + i * self.space, 0), None)
            img_tags.paste(
                self.create_tag_paper(wim, list_labels[i]),
                (p + i * self.space, 0),
                None
            )
            p += wim
            i += 1

        if show_tags:
            final_out = Image.new("RGB", (w_out, max_h + self.height_tag_paper))
        else:
            final_out = Image.new("RGB", (w_out, max_h))
        final_out.paste(img_out, (0, 0), None)
        if show_tags:
            final_out.paste(img_tags, (0, max_h), None)

        return final_out


class VisualizeImagesMultiLabel(VisualizeImages):
    """
    Visualize images from dataset with multi-label.
    """

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask.
        """
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.color_map)
        return new_mask

    def super_mask_over_image(self, input_img, mask):
        """
        Convert a mask into a heatmap.

        :param input_img: PIL.Image.Image of type uint8. The input image.
        :param mask: PIL.Image.Image mode P.
        :return:
        """
        forg = mask.convert("RGBA")

        return self.superpose_two_images_using_alpha(
            input_img.copy(),
            forg,
            self.alpha
        )

    def __call__(self,
                 name_classes,
                 list_images,
                 list_true_masks,
                 list_labels,
                 rows,
                 columns,
                 show_tags=False
                 ):
        """

        :param name_classes:
        :param list_images:
        :param list_true_masks:
        :return:
        """
        for i, msk in enumerate(list_true_masks):
            msg = "'mask' must be `{}` or None, but we found `{}` " \
                  ".... [NOT OK]".format(np.ndarray, type(msk))
            assert isinstance(msk, np.ndarray), msg
        for i, img in enumerate(list_images):
            msg = "'input_image' type must be `{}`, but we found `{}` " \
                  ".... [NOT OK]".format(Image.Image, type(img))
            assert isinstance(img, Image.Image), msg

        msg = "'name_classes' must be of type `{}`, but we found `{}` " \
              ".... [NOT OK]".format(dict, type(name_classes))
        assert isinstance(name_classes, dict), msg

        assert rows == 1, "We support only 1 row!!!! You asked for {}".format(
            rows)
        msg = "list_images and list_true_masks must have the same number " \
              "of elements. You provided: len(list_images) = {}, " \
              "len(list_true_masks) = {}".format(len(list_images),
                                                 len(list_true_masks))
        assert len(list_images) == len(list_true_masks), msg

        nbr_imgs = len(list_images)
        extra_w_space = self.space * (nbr_imgs - 1)
        w_out = 0
        max_h = 0
        for im in list_images:
            w_out += im.size[0]
            max_h = max(max_h, im.size[1])

        w_out += extra_w_space
        img_out = Image.new("RGB", (w_out, max_h))
        img_tags = Image.new("RGB", (w_out, self.height_tag_paper))
        i = 0
        p = 0
        for im, msk in zip(list_images, list_true_masks):
            wim = im.size[0]
            tmp = self.super_mask_over_image(im, self.colorize_mask(msk))
            img_out.paste(tmp, (p + i * self.space, 0), None)
            img_tags.paste(
                self.create_tag_paper(wim, list_labels[i]),
                (p + i * self.space, 0),
                None
            )
            p += wim
            i += 1

        if show_tags:
            final_out = Image.new("RGB", (w_out, max_h + self.height_tag_paper))
        else:
            final_out = Image.new("RGB", (w_out, max_h))
        final_out.paste(img_out, (0, 0), None)
        if show_tags:
            final_out.paste(img_tags, (0, max_h), None)

        return final_out


def see_multi_class_ds(atom,
                       outdir,
                       root_main
                       ):
    """
    Sample from a dataset with multi-class.
    :param atom: dict with keys/values example:
        {
            "dataset": constants.GLAS,  # name of the dataset
            "nbr_samples": 5,  # how many sample to draw per class.
            "nbr_classes": 2  # how many classes to consider.
        }
    :param outdir: str. absolute path to folder where to store the output image.
    :param root_main: str, absolute path to the folder where main.py lives.
    :return: absolute path to the stored image.
    """

    yaml_file = join(root_main, "config_yaml/{}.yaml".format(atom['dataset']))
    with open(yaml_file, 'r') as f:
        args = Dict2Obj(yaml.load(f))
    os.environ['MYSEED'] = str(args.MYSEED)

    if args.fold_folder.startswith("./"):
        args.fold_folder = args.fold_folder.replace("./", "")

    args.fold_folder = join(root_main, args.fold_folder)
    train_csv, valid_csv, test_csv = get_csv_files(args)
    rootpath = get_rootpath_2_dataset(args)

    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    train_samples = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)
    # valid_samples = csv_loader(valid_csv, rootpath, drop_normal=cnd_drop_n)
    # test_samples = csv_loader(test_csv, rootpath, drop_normal=cnd_drop_n)

    ll_all_class = list(args.name_classes.keys())
    for _ in range(100):
        shuffle(ll_all_class)

    l_classes = ll_all_class[:atom['nbr_classes']]
    l_samples = []
    for cl in l_classes:
        cl_s = [s for s in train_samples if s[3] == cl]
        for _ in range(100):
            shuffle(cl_s)

        l_samples.extend(cl_s[:atom['nbr_samples']])


    trainset, _ = get_validationset(args,
                                    train_samples,
                                    None,
                                    (None, None),
                                    batch_size=None
                                    )

    visu_dataset = VisualizeImages(
        alpha=128,
        floating=3,
        height_tag=60,
        bins=100,
        rangeh=(0, 1),
        color_map=mlp.cm.get_cmap("seismic"),
        height_tag_paper=130
    )
    imgs, msks, lbls = [], [], []

    for s in l_samples:
        id_ = s[0]
        i = trainset.get_index_of_id(id_)
        imgs.append(
            trainset.get_original_input_img(i)
        )
        msks.append(
            (np.array(trainset.get_original_input_mask(i)) != 0
             ).astype(float)
        )
        lbls.append(
            s[3]
        )


    fig = visu_dataset(name_classes=args.name_classes,
                       list_images=imgs,
                       list_true_masks=msks,
                       list_labels=lbls,
                       rows=1,
                       columns=len(imgs),
                       show_tags=False
                       )
    outdes = join(outdir, "Samples-{}.png".format(atom['dataset']))
    fig.save(outdes, "PNG")

    return outdes



def see_multi_label_ds(atom,
                       outdir,
                       root_main
                       ):
    """
    Sample from a dataset with multi-label.
    :param atom: dict with keys/values example:
        {
            "dataset": constants.CSCAPES,  # name of the dataset
            "nbr_samples": 10,  # total number of samples to draw.
        }
    :param outdir: str. absolute path to folder where to store the output image.
    :return: absolute path to the stored image.
    """

    yaml_file = join(root_main, "config_yaml/{}.yaml".format(atom['dataset']))
    with open(yaml_file, 'r') as f:
        args = Dict2Obj(yaml.load(f))
    os.environ['MYSEED'] = str(args.MYSEED)

    if args.fold_folder.startswith("./"):
        args.fold_folder = args.fold_folder.replace("./", "")

    args.fold_folder = join(root_main, args.fold_folder)
    train_csv, valid_csv, test_csv = get_csv_files(args)
    rootpath = get_rootpath_2_dataset(args)

    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.
    cnd_drop_n = (args.dataset == constants.CAM16)
    cnd_drop_n &= (args.al_type != constants.AL_WSL)

    train_samples = csv_loader(train_csv, rootpath, drop_normal=cnd_drop_n)

    # select randomly samples from the entire dataset. there are no labels.
    csmples = deepcopy(train_samples)
    for _ in range(2000):
        shuffle(csmples)

    l_samples = csmples[: atom['nbr_samples']]

    trainset, _ = get_validationset(args,
                                    train_samples,
                                    None,
                                    (None, None),
                                    batch_size=None
                                    )

    visu_dataset = VisualizeImagesMultiLabel(
        alpha=128,
        floating=3,
        height_tag=60,
        bins=100,
        rangeh=(0, 1),
        color_map=cslabels.get_colormap(),
        height_tag_paper=130
    )
    imgs, msks, lbls = [], [], []

    for s in l_samples:
        id_ = s[0]
        i = trainset.get_index_of_id(id_)
        imgs.append(
            trainset.get_original_input_img(i)
        )
        msks.append(
            np.array(trainset.get_original_input_mask(i))
        )
        lbls.append(
            s[3]
        )

    fig = visu_dataset(name_classes=args.name_classes,
                       list_images=imgs,
                       list_true_masks=msks,
                       list_labels=lbls,
                       rows=1,
                       columns=len(imgs),
                       show_tags=False
                       )

    outdes = join(outdir, "Samples-{}.png".format(atom['dataset']))
    fig.save(outdes, "PNG")

    return outdes


def create_thumbnail(l_imgs, file_out, scale=5):
    """
    Create a thumbnail.
    :param l_imgs: list of absolute paths to images.
    :param file_out: absolute path to output PNG.
    :param scale: int, float. how much to scale height and width of the
    finale panorama.
    :return:
    """
    l_imgs = [Image.open(f, "r").convert("RGB") for f in l_imgs]

    w = max([im.size[0] for im in l_imgs])
    # resize images: give them all the same w. and compute the
    # corresponding height.
    for i, im in enumerate(l_imgs):
        wx, hx = im.size
        rw = (wx / float(w))
        hx_new = int(hx / rw)
        im = im.resize((w, hx_new), Image.ANTIALIAS)
        l_imgs[i] = im

    h = sum([im.size[1] for im in l_imgs])

    w_ = int(w / scale)
    h_ = int(h / scale)
    print("w {} --> {}".format(w, w_))
    print("h {} --> {}".format(h, h_))

    # Create the big image
    img_big = Image.new("RGB", (w, h))
    hidx = 0
    for img in l_imgs:
        img_big.paste(img, (0, hidx), None)
        hidx += img.size[1]

    # Create the thumbnail
    img_big.thumbnail((w_, h_))

    # Save file
    img_big.save(file_out, format="PNG", quality=100, optimize=True)