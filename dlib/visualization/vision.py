import os
import sys
from os.path import dirname, abspath, join

from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import ListedColormap

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants


def get_bin_colormap():
    palette = [
        0, 0, 0,  # 0
        255, 255, 255  # 1
    ]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    return palette


def get_mpl_bin_seeds_colormap():
    col_dict = {0: (0., 0., 1.),  # negative (blue)
                1: (1., 0., 0.)  # positive (red). unknown is transparent.
                }

    colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    return colormap


class VisualizeCAMs(object):

    def __init__(self,
                 fdout,
                 mask_color_map,
                 task,
                 heatmap_colormap=plt.get_cmap("jet"),
                 alpha=128,
                 height_tag=60,
                 multi_label_flag=False,
                 dpi=100
                 ):
        super(VisualizeCAMs, self).__init__()

        if not os.path.isdir(fdout):
            os.makedirs(fdout)
        self.fdout = fdout
        self.fdout_tags = join(fdout, 'tags')
        if not os.path.isdir(self.fdout_tags):
            os.makedirs(self.fdout_tags)

        self.mask_color_map = mask_color_map
        self.heatmap_colormap = heatmap_colormap
        self.alpha = alpha

        self.dpi = dpi
        self.h = None
        self.w = None

        self.im_rec = False

        self.task = task

        # precision of writing the probabilities.
        floating = 3
        self.prec = "%." + str(floating) + "f"
        self.height_tag = height_tag

        # y position of the text inside the tag. (first line)
        self.y = int(self.height_tag / 4)
        # y position of the text inside the tag. (second line)
        self.y2 = self.y * 2
        # y position of the text inside the tag. (3rd line)
        self.y3 = self.y * 3
        # how much space to put between LABELS (not word) inside the tag.
        self.dx = 10
        # (pixels) how much space to leave between images.
        self.space = 5

        self.font_regular = None
        self.font_bold = None
        self.init_fonts(15)

        # Colors:
        self.white = "rgb(255, 255, 255)"
        self.black = "rgb(0, 0, 0)"
        self.green = "rgb(0, 255, 0)"
        self.red = "rgb(255, 0, 0)"
        self.orange = "rgb(255,165,0)"

        # Margin:
        self.left_margin = 10  # the left margin.

        self.multi_label_flag = multi_label_flag

        self.background_color = self.white
        self.text_color = self.black

    def init_fonts(self, sz):
        base = join(root_dir, "dlib/visualization/fonts/Inconsolata")
        self.font_regular = ImageFont.truetype(
            join(base, 'Inconsolata-Regular.ttf'), sz)
        self.font_bold = ImageFont.truetype(
            join(base, 'Inconsolata-Bold.ttf'), sz)

    def convert_cam_into_heatmap_(self, input_img, mask, binarize=False):
        """
        Convert a cam into a heatmap.

        :param input_img: PIL.Image.Image of type float32. The input image.
        :param mask: 2D numpy.ndarray (binary or continuous in [0, 1]).
        :param binarize: bool. If True, the mask is binarized by thresholding
         (values >=0.5 will be set to 1. ELse, 0).
        :return: PIL.Image.Image
        """
        if binarize:
            mask = ((mask >= 0.5) * 1.).astype(np.float32)

        img_arr = self.heatmap_colormap((mask * 255).astype(np.uint8))
        # --> in [0, 1.], (h, w, 4)

        return self.superpose_two_images_using_alpha(
            input_img.copy(), Image.fromarray(np.uint8(img_arr * 255)),
            self.alpha)

    def convert_cam_into_heatmap(self, input_img, mask):
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(input_img))
        ax.imshow(mask, interpolation='bilinear', cmap=self.heatmap_colormap,
                  alpha=self.alpha)

        ax.axis('off')
        ax.text(5, 5, 'your legend', bbox={'facecolor': 'white', 'pad': 10})
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data, mode="RGB").resize(
            size=(self.w, self.h), resample=Image.BICUBIC)
        plt.close(fig)
        del fig
        return img

    @staticmethod
    def superpose_two_images_using_alpha(back, forg, alpha):
        """
        Superpose two PIL.Image.Image uint8 images.
        images must have the same size.

        :param back: background image. (RGB)
        :param forg: foreground image (L).
        :param alpha: int, in [0, 255] the alpha value.
        :return:R PIL.Image.Image RGB uint8 image.
        """
        # adjust the alpha
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.
        # Image.Image.putalpha
        forg.putalpha(alpha)
        # back.putalpha(int(alpha/4))
        back.paste(forg, (0, 0), forg)

        return back

    def colorize_mask(self, image_array):
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.mask_color_map)
        return new_mask

    def tag(self, wim, msg):
        img_tag = Image.new("RGB", (wim, self.height_tag),
                            self.background_color)

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg, self.text_color,
                                self.font_bold, self.dx)

        return img_tag

    def create_tag_input(self, wim):
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, "Input", self.white,
                                self.font_regular, self.dx)

        return img_tag

    def create_tag_pred_mask(self, wim, msg1, msg2):
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg1, self.white,
                                self.font_regular, self.dx)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y2, msg2, self.white,
                                self.font_regular, self.dx)

        return img_tag

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

    def create_tag_true_mask(self, wim):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over
        it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong",
        None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "True mask",
                                self.white,
                                self.font_regular,
                                self.dx
                                )

        return img_tag

    @staticmethod
    def drawonit(draw, x, y, label, fill, font, dx):
        """
        Draw text on an ImageDraw.new() object.

        :param draw: object, ImageDraw.new()
        :param x: int, x position of top left corner.
        :param y: int, y position of top left corner.
        :param label: str, text message to draw.
        :param fill: color to use.
        :param font: font to use.
        :param dx: int, how much space to use between LABELS (not word).
        Useful to compute the position of the next
        LABEL. (future)
        :return:
            . ImageDraw object with the text drawn on it as requested.
            . The next position x.
        """
        draw.text((x, y), label, fill=fill, font=font)
        x += font.getsize(label)[0] + dx

        return draw, x

    def vstitch_tag_to_img(self, tag_img, img):
        wt, ht = tag_img.size
        wi, hi = img.size
        assert wi == wt
        assert ht == self.height_tag

        out = Image.new("RGB", (wi, hi + ht), self.background_color)
        out.paste(img, (0, 0), None)
        out.paste(tag_img, (0, hi))
        return out

    def tagax(self, ax, text):
        ax.text(3, 40,
                text, bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8}
                )

    def tagaxlow(self, ax, text):
        ax.text(1, 1,
                text, bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.5}
                )

    def main_pred(self, axes, img_pil, true_mask, cam, pred_mask):
        ax = axes[0, 0]
        ax.imshow(np.asarray(img_pil))
        self.tagax(ax, 'Input')

        if not self.multi_label_flag:
            ax = axes[0, 1]
            ax.imshow(np.asarray(img_pil))
            ax.imshow(cam, interpolation='bilinear', cmap=self.heatmap_colormap,
                      alpha=self.alpha)
            self.tagax(ax, 'CAM target')
        else:
            raise NotImplementedError

        ax = axes[1, 0]
        ax.imshow(self.colorize_mask(true_mask))
        self.tagax(ax, 'True mask')

        ax = axes[1, 1]
        ax.imshow(self.colorize_mask(pred_mask))
        self.tagax(ax, 'Pred. mask')

    def del_fcam_block(self, axes):
        im = Image.fromarray((np.ones((self.h, self.w, 3), dtype=np.uint8) *
                              255))
        axes[0, 2].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[0, 3].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[1, 2].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[1, 3].imshow(im, interpolation=None, cmap=self.heatmap_colormap)

    def imshow_heatmap(self, heatmap, inter):
        fig, ax = plt.subplots()
        ax.imshow(heatmap, interpolation=inter, cmap=self.heatmap_colormap)
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data, mode="RGB").resize(
            size=(self.w, self.h), resample=Image.BICUBIC)
        plt.close(fig)
        del fig
        return img

    def inter_cam_f_cl(self, axes, heatmap, heatmap_normalized, tag):
        ax = axes[0, 3]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[0, 4]
        ax.imshow(self.self_nromalize_cam(heatmap), interpolation=None,
                  cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (S-Norm/Raster)'.format(tag))

        ax = axes[0, 5]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm/Raster)'.format(tag))

        # --
        ax = axes[1, 3]
        ax.imshow(heatmap, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[1, 4]
        ax.imshow(self.self_nromalize_cam(heatmap), interpolation='bilinear',
                  cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (S-Norm/Int)'.format(tag))

        ax = axes[1, 5]
        ax.imshow(heatmap_normalized, interpolation='bilinear',
                  cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm/Int)'.format(tag))

    def low_res_cams(self, axes, heatmap, heatmap_normalized, tag):
        ax = axes[2, 0]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap,
                  aspect='auto')
        self.tagaxlow(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[2, 1]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap, aspect='auto')
        self.tagaxlow(ax, '{} (Norm/Raster)'.format(tag))

        # --

        ax = axes[3, 0]
        ax.imshow(heatmap, interpolation='bilinear',
                  cmap=self.heatmap_colormap, aspect='auto')
        self.tagaxlow(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[3, 1]
        ax.imshow(heatmap_normalized, interpolation='bilinear',
                  cmap=self.heatmap_colormap, aspect='auto')
        self.tagaxlow(ax, '{} (Norm/Int)'.format(tag))

    def del_low_res_cam(self, axes):
        im = Image.fromarray((np.ones((self.h, self.w, 3), dtype=np.uint8) *
                              255))
        axes[2, 0].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[2, 1].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[2, 2].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[3, 0].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[3, 1].imshow(im, interpolation=None, cmap=self.heatmap_colormap)
        axes[3, 2].imshow(im, interpolation=None, cmap=self.heatmap_colormap)

    def final_cams(self, axes, heatmap, heatmap_normalized, tag):
        ax = axes[2, 2]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[2, 3]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm/Raster)'.format(tag))

        # --
        ax = axes[3, 2]
        ax.imshow(heatmap, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[3, 3]
        ax.imshow(heatmap_normalized,
                  interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm/Int)'.format(tag))

    def call_segmentation(self, img_pil, true_mask, pred_mask, cam, cam_raw):

        assert self.task == constants.SEG

        nrows = 2
        ncols = 4

        wim, him = img_pil.size
        self.h = him
        self.w = wim
        r = him / float(wim)
        fw = 15
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        self.main_pred(axes, img_pil, true_mask, cam, pred_mask)

        # todo: indicate maximum, local maxima on cams.

        # todo: normalized raw cam internally:
        # https://www.pyimagesearch.com/2020/03/09/
        # grad-cam-visualize-class-activation-maps-with-keras
        # -tensorflow-and-deep-learning/

        tag = 'SEG'
        ax = axes[0, 2]
        ax.imshow(cam_raw, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[0, 3]
        ax.imshow(cam, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm./Raster)'.format(tag))

        ax = axes[1, 2]
        ax.imshow(cam_raw, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[1, 3]
        ax.imshow(cam, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Norm./Int)'.format(tag))

        return fig

    def call_wsl(self, img_pil, true_mask,
                 pred_mask,
                 cam_low, cam_low_raw,
                 cam_inter, cam_inter_raw,
                 cam, cam_raw):

        assert self.task == constants.STD_CL

        nrows = 4
        ncols = 4

        wim, him = img_pil.size
        self.h = him
        self.w = wim
        r = him / float(wim)
        fw = 15
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        self.main_pred(axes, img_pil, true_mask, cam, pred_mask)

        # todo: indicate maximum, local maxima on cams.

        # todo: normalized raw cam internally:
        # https://www.pyimagesearch.com/2020/03/09/
        # grad-cam-visualize-class-activation-maps-with-keras
        # -tensorflow-and-deep-learning/

        self.del_fcam_block(axes)
        self.low_res_cams(axes, cam_low_raw, cam_low, tag='LR CAM')
        tag = 'HR CAM Int'
        self.final_cams(axes, cam_raw, cam,  tag=tag)

        return fig

    def closing(self, fig, basefile):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
                            wspace=0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        fig.savefig(join(self.fdout, '{}.png'.format(basefile)),
                    pad_inches=0, bbox_inches='tight', dpi=self.dpi)
        plt.close(fig)

    def self_nromalize_cam(self, heatmap):
        assert isinstance(heatmap, np.ndarray)

        x = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        return self.cam_norm(x).squeeze().numpy()

    def __call__(self, img_pil, true_mask,
                 pred_mask, mask_pred_inter,
                 cam_low, cam_low_raw,
                 cam_inter, cam_inter_raw,
                 cam, cam_raw, im_recon,
                 basefile, seed=None):
        if self.task == constants.SEG:
            fig = self.call_segmentation(img_pil, true_mask, pred_mask, cam,
                                         cam_raw)
        elif self.task == constants.STD_CL:
            fig = self.call_wsl(img_pil, true_mask, pred_mask, cam_low,
                                cam_low_raw, cam_inter, cam_inter_raw, cam,
                                cam_raw)
        else:
            raise NotImplementedError

        self.closing(fig, basefile)


class VisualizeFCAMs(object):

    def __init__(self,
                 fdout,
                 mask_color_map,
                 task,
                 heatmap_colormap=plt.get_cmap("jet"),
                 alpha=128,
                 height_tag=60,
                 multi_label_flag=False,
                 dpi=100,
                 seg_ignore_idx=-255,
                 seed_cmap=get_mpl_bin_seeds_colormap()
                 ):
        super(VisualizeFCAMs, self).__init__()

        if not os.path.isdir(fdout):
            os.makedirs(fdout)
        self.fdout = fdout
        self.fdout_tags = join(fdout, 'tags')
        if not os.path.isdir(self.fdout_tags):
            os.makedirs(self.fdout_tags)

        self.mask_color_map = mask_color_map
        self.heatmap_colormap = heatmap_colormap
        self.alpha = alpha
        self.seg_ignore_idx = seg_ignore_idx

        self.seed_cmap = seed_cmap

        self.dpi = dpi
        self.h = None
        self.w = None

        self.im_rec = False

        self.task = task
        assert task == constants.F_CL

        # precision of writing the probabilities.
        floating = 3
        self.prec = "%." + str(floating) + "f"
        self.height_tag = height_tag

        # y position of the text inside the tag. (first line)
        self.y = int(self.height_tag / 4)
        # y position of the text inside the tag. (second line)
        self.y2 = self.y * 2
        # y position of the text inside the tag. (3rd line)
        self.y3 = self.y * 3
        # how much space to put between LABELS (not word) inside the tag.
        self.dx = 10
        # (pixels) how much space to leave between images.
        self.space = 5

        self.font_regular = None
        self.font_bold = None
        self.init_fonts(15)

        # Colors:
        self.white = "rgb(255, 255, 255)"
        self.black = "rgb(0, 0, 0)"
        self.green = "rgb(0, 255, 0)"
        self.red = "rgb(255, 0, 0)"
        self.orange = "rgb(255,165,0)"

        # Margin:
        self.left_margin = 10  # the left margin.

        self.multi_label_flag = multi_label_flag

        self.background_color = self.white
        self.text_color = self.black

    def init_fonts(self, sz):
        base = join(root_dir, "dlib/visualization/fonts/Inconsolata")
        self.font_regular = ImageFont.truetype(
            join(base, 'Inconsolata-Regular.ttf'), sz)
        self.font_bold = ImageFont.truetype(
            join(base, 'Inconsolata-Bold.ttf'), sz)

    def convert_cam_into_heatmap_(self, input_img, mask, binarize=False):
        """
        Convert a cam into a heatmap.

        :param input_img: PIL.Image.Image of type float32. The input image.
        :param mask: 2D numpy.ndarray (binary or continuous in [0, 1]).
        :param binarize: bool. If True, the mask is binarized by thresholding
         (values >=0.5 will be set to 1. ELse, 0).
        :return: PIL.Image.Image
        """
        if binarize:
            mask = ((mask >= 0.5) * 1.).astype(np.float32)

        img_arr = self.heatmap_colormap((mask * 255).astype(np.uint8))
        # --> in [0, 1.], (h, w, 4)

        return self.superpose_two_images_using_alpha(
            input_img.copy(), Image.fromarray(np.uint8(img_arr * 255)),
            self.alpha)

    def convert_cam_into_heatmap(self, input_img, mask):
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(input_img))
        ax.imshow(mask, interpolation='bilinear', cmap=self.heatmap_colormap,
                  alpha=self.alpha)

        ax.axis('off')
        ax.text(5, 5, 'your legend', bbox={'facecolor': 'white', 'pad': 10})
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data, mode="RGB").resize(
            size=(self.w, self.h), resample=Image.BICUBIC)
        plt.close(fig)
        del fig
        return img

    @staticmethod
    def superpose_two_images_using_alpha(back, forg, alpha):
        """
        Superpose two PIL.Image.Image uint8 images.
        images must have the same size.

        :param back: background image. (RGB)
        :param forg: foreground image (L).
        :param alpha: int, in [0, 255] the alpha value.
        :return:R PIL.Image.Image RGB uint8 image.
        """
        # adjust the alpha
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.
        # Image.Image.putalpha
        forg.putalpha(alpha)
        # back.putalpha(int(alpha/4))
        back.paste(forg, (0, 0), forg)

        return back

    def colorize_mask(self, image_array):
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.mask_color_map)
        return new_mask

    def tag(self, wim, msg):
        img_tag = Image.new("RGB", (wim, self.height_tag),
                            self.background_color)

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg, self.text_color,
                                self.font_bold, self.dx)

        return img_tag

    def create_tag_input(self, wim):
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, "Input", self.white,
                                self.font_regular, self.dx)

        return img_tag

    def create_tag_pred_mask(self, wim, msg1, msg2):
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg1, self.white,
                                self.font_regular, self.dx)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y2, msg2, self.white,
                                self.font_regular, self.dx)

        return img_tag

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

    def create_tag_true_mask(self, wim):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over
        it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong",
        None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "True mask",
                                self.white,
                                self.font_regular,
                                self.dx
                                )

        return img_tag

    @staticmethod
    def drawonit(draw, x, y, label, fill, font, dx):
        """
        Draw text on an ImageDraw.new() object.

        :param draw: object, ImageDraw.new()
        :param x: int, x position of top left corner.
        :param y: int, y position of top left corner.
        :param label: str, text message to draw.
        :param fill: color to use.
        :param font: font to use.
        :param dx: int, how much space to use between LABELS (not word).
        Useful to compute the position of the next
        LABEL. (future)
        :return:
            . ImageDraw object with the text drawn on it as requested.
            . The next position x.
        """
        draw.text((x, y), label, fill=fill, font=font)
        x += font.getsize(label)[0] + dx

        return draw, x

    def vstitch_tag_to_img(self, tag_img, img):
        wt, ht = tag_img.size
        wi, hi = img.size
        assert wi == wt
        assert ht == self.height_tag

        out = Image.new("RGB", (wi, hi + ht), self.background_color)
        out.paste(img, (0, 0), None)
        out.paste(tag_img, (0, hi))
        return out

    def tagax(self, ax, text):
        ax.text(3, 40,
                text, bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8}
                )

    def tagaxlow(self, ax, text):
        ax.text(1, 1,
                text, bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.5}
                )

    def convert_img_recon(self, im_recon):
        x = torch.from_numpy(im_recon)
        x = x.squeeze().permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        x = x.numpy()
        x = (x * 255).astype(np.uint8)
        return x

    def main_pred(self, axes, img_pil, true_mask, cam, pred_mask,
                  mask_pred_inter, cam_inter, im_recon):
        if im_recon is not None:
            ax = axes[0, 0]
            ax.imshow(self.convert_img_recon(im_recon))
            self.tagax(ax, 'IMG-Recon')
            z = 1
        else:
            z = 0

        ax = axes[0, z]
        ax.imshow(np.asarray(img_pil))
        self.tagax(ax, 'Input')

        if not self.multi_label_flag:
            ax = axes[0, z + 1]
            ax.imshow(np.asarray(img_pil))
            ax.imshow(cam, interpolation='bilinear',
                      cmap=self.heatmap_colormap, alpha=self.alpha,
                      vmin=0., vmax=1.)

            self.tagax(ax, 'FCAM')
        else:
            raise NotImplementedError

        ax = axes[0, z + 2]
        ax.imshow(np.asarray(img_pil))
        ax.imshow(cam_inter, interpolation='bilinear',
                  cmap=self.heatmap_colormap,  alpha=self.alpha,
                  vmin=0., vmax=1.)
        self.tagax(ax, 'HR CAM Int')

        ax = axes[1, z]
        ax.imshow(self.colorize_mask(true_mask))
        self.tagax(ax, 'True mask')

        ax = axes[1, z + 1]
        ax.imshow(self.colorize_mask(pred_mask))
        self.tagax(ax, 'Pred. FCAM mask')

        ax = axes[1, z + 2]
        ax.imshow(self.colorize_mask(mask_pred_inter))
        self.tagax(ax, 'Pred. HR Int mask')

    def inter_cam_f_cl(self, axes, heatmap, heatmap_normalized, tag):
        z = 0
        if self.im_rec:
            z = 1

        ax = axes[0, 3 + z]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[0, 4 + z]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap, vmin=0., vmax=1.)
        self.tagax(ax, '{} (Norm/Raster)'.format(tag))

        # --
        ax = axes[1, 3 + z]
        ax.imshow(heatmap, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[1, 4 + z]
        ax.imshow(heatmap_normalized, interpolation='bilinear',
                  cmap=self.heatmap_colormap, vmin=0., vmax=1.)
        self.tagax(ax, '{} (Norm/Int)'.format(tag))

    def low_res_cams(self, axes, heatmap, heatmap_normalized, tag):
        z = 0
        if self.im_rec:
            z = 1

        ax = axes[2, 1 + z]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap,
                  aspect='auto')
        self.tagaxlow(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[2, 2 + z]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap, aspect='auto', vmin=0., vmax=1.)
        self.tagaxlow(ax, '{} (Norm/Raster)'.format(tag))

        # --

        ax = axes[3, 1 + z]
        ax.imshow(heatmap, interpolation='bilinear',
                  cmap=self.heatmap_colormap, aspect='auto')
        self.tagaxlow(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[3, 2 + z]
        ax.imshow(heatmap_normalized, interpolation='bilinear',
                  cmap=self.heatmap_colormap, aspect='auto', vmin=0., vmax=1.)
        self.tagaxlow(ax, '{} (Norm/Int)'.format(tag))

    def final_cams(self, axes, heatmap, heatmap_normalized, tag):
        z = 0
        if self.im_rec:
            z = 1

        ax = axes[2, 3 + z]
        ax.imshow(heatmap, interpolation=None, cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Raster)'.format(tag))

        ax = axes[2, 4 + z]
        ax.imshow(heatmap_normalized, interpolation=None,
                  cmap=self.heatmap_colormap, vmin=0., vmax=1.)
        self.tagax(ax, '{} (Norm/Raster)'.format(tag))

        # --
        ax = axes[3, 3 + z]
        ax.imshow(heatmap, interpolation='bilinear', cmap=self.heatmap_colormap)
        self.tagax(ax, '{} (Raw/Int)'.format(tag))

        ax = axes[3, 4 + z]
        ax.imshow(heatmap_normalized,
                  interpolation='bilinear', cmap=self.heatmap_colormap,
                  vmin=0., vmax=1.)
        self.tagax(ax, '{} (Norm/Int)'.format(tag))

    def show_seed(self, axes, img_pil, true_mask, seed, tag):
        z = 0
        if self.im_rec:
            z = 1

        masked_seed = np.ma.masked_where(seed == self.seg_ignore_idx, seed)

        ax = axes[2, z]
        ax.imshow(np.asarray(img_pil))
        ax.imshow(masked_seed, interpolation=None, cmap=self.seed_cmap,
                  vmin=0., vmax=1.)
        self.tagax(ax, '{} (Input w/seed)'.format(tag))

        ax = axes[3, z]
        ax.imshow(self.colorize_mask(true_mask))
        ax.imshow(masked_seed, interpolation=None, cmap=self.seed_cmap,
                  vmin=0., vmax=1.)
        # self.tagax(ax, '{} (True mask w/seed)'.format(tag))

    def __call__(self, img_pil, true_mask,
                 pred_mask, mask_pred_inter,
                 cam_low, cam_low_raw,
                 cam_inter, cam_inter_raw,
                 cam, cam_raw, im_recon,
                 seed,
                 basefile):

        assert self.task == constants.F_CL
        self.im_rec = (im_recon is not None)

        nrows = 4
        ncols = 5
        if self.im_rec:
            ncols = ncols + 1

        wim, him = img_pil.size
        self.h = him
        self.w = wim
        r = him / float(wim)
        fw = 15
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        self.main_pred(axes, img_pil, true_mask, cam, pred_mask,
                       mask_pred_inter, cam_inter, im_recon)

        # todo: indicate maximum, local maxima on cams.

        self.inter_cam_f_cl(axes, cam_inter_raw, cam_inter, tag='HR CAM Int')
        self.low_res_cams(axes, cam_low_raw, cam_low, tag='LR CAM')
        self.final_cams(axes, cam_raw, cam,  tag='HR FCAM')
        if seed is not None:
            self.show_seed(axes, img_pil, true_mask, seed, tag='HR SL-FCAM')

        self.closing(fig, basefile)

    def closing(self, fig, basefile):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
                            wspace=0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        fig.savefig(join(self.fdout, '{}.png'.format(basefile)),
                    pad_inches=0, bbox_inches='tight', dpi=self.dpi)
        plt.close(fig)

    def self_nromalize_cam(self, heatmap):
        raise NotImplementedError
        assert isinstance(heatmap, np.ndarray)

        x = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        return self.cam_norm(x).squeeze().numpy()


def test_VisualizeFCAMs():
    import datetime as dt
    import torch
    import torch.nn.functional as F

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    debug_fd = join(root_dir, 'data/debug/input')
    img_pil = Image.open(
        join(debug_fd, 'Black_Footed_Albatross_0002_55.jpg'), 'r').convert(
        "RGB")
    w, h = img_pil.size
    cam_low = torch.rand(size=(int(h/32.), int(w/32.)), dtype=torch.float)
    cam_inter = F.interpolate(
        input=cam_low.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=True
    ).squeeze()
    cam = cam_inter
    mask_pred = (cam > 0.5).float()
    mask_pred_inter = torch.rand(size=(h, w), dtype=torch.float)
    mask_pred_inter = (mask_pred_inter > 0.3).float()
    true_mask = Image.open(
        join(debug_fd, 'Black_Footed_Albatross_0002_55.png'), 'r').convert(
        "L")
    true_mask = (np.array(true_mask) > (255 / 2.)).astype(np.uint8)
    debug_out = join(root_dir, 'data/debug/visualization')

    im_recon = torch.rand(size=(1, 3, h, w), dtype=torch.float)
    im_recon = (im_recon - 0.5) / 0.5
    im_recon = im_recon.squeeze().numpy()
    im_recon = None

    seg_ignore_idx = -255
    seed = torch.rand(size=(h, w), dtype=torch.float).numpy()
    seed[np.where(seed > 0.9)] = 1.
    seed[np.where(seed < 0.1)] = 0.
    seed[np.where(np.logical_and(0.1 <= seed, seed <= 0.9))] = seg_ignore_idx

    visu = VisualizeFCAMs(fdout=debug_out,
                          mask_color_map=get_bin_colormap(),
                          task=constants.F_CL,
                          alpha=100,
                          height_tag=60,
                          multi_label_flag=False,
                          dpi=50,
                          seg_ignore_idx=seg_ignore_idx
                          )
    t0 = dt.datetime.now()
    visu(img_pil=img_pil, true_mask=true_mask,
         pred_mask=mask_pred.numpy(),
         mask_pred_inter=mask_pred_inter.numpy(),
         cam_low=torch.sigmoid(cam_low).numpy(),
         cam_low_raw=cam_low.numpy(),
         cam_inter=torch.sigmoid(cam_inter).numpy(),
         cam_inter_raw=cam_inter.numpy(),
         cam=torch.sigmoid(cam).numpy(),
         cam_raw=cam.numpy(),
         im_recon=im_recon,
         seed=seed,
         basefile='Black_Footed_Albatross_0002_55' + visu.task
         )
    print('Work time: {}'.format(dt.datetime.now() - t0))


if __name__ == '__main__':
    test_VisualizeFCAMs()
