import math
from os.path import join

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import getpass
from PIL import Image, ImageDraw, ImageFont


def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks_into_n(l, n):
    """
    Split iterable l into n chunks (iterables) with the same size.

    :param l: iterable.
    :param n: number of chunks.
    :return: iterable of length n.
    """
    chunksize = int(math.ceil(len(l) / n))
    return (l[i * chunksize:i * chunksize + chunksize] for i in range(n))


def draw_hist_probability_of_list(list_v, bins, rangeh, title):
    """
    Draw the histogram of a list of values.

    :param list_v: list, list of numerical values.
    :param bins: int, number of bins.
    :param rangeh: tuple, (min, max) values.
    :param title: str, title of the plot.
    :return:
    """
    fig = plt.figure()
    l_array = np.array(list_v)

    plt.hist(l_array.ravel(), bins=bins, weights=np.ones_like(l_array.ravel()) / float(l_array.size),
             range=rangeh)
    plt.xlabel("p: percentage of metastatic pixels within a patch.", fontsize=8)
    plt.ylabel("Percentage of patches with a \n percentage of metastatic pixels equals p.", fontsize=8)
    plt.title(title, fontsize=8)
    plt.grid(True)

    return fig


def announce_msg(msg, upper=True):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into uppercase. Else, the message is displayed
    as it is.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    print(output_msg)

    return output_msg


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_rootpath_2_dataset(args):
    """
    Returns the root path to the dataset depending on the server.
    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: baseurl, a str. The root path to the dataset independently from the host.
    """
    datasetname = args.dataset
    username = getpass.getuser()
    baseurl = None
    if username == 'brian':
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets"
    elif username == "sbelharb":  # cedar
        baseurl = "/project/6004986/sbelharb/workspace/datasets"
    elif username == "sbelharbi":  # livia
        baseurl = "/export/livia/data/sbelharbi/datasets"

    if baseurl is None:
        raise ValueError("Sorry, it seems we are enable to recognize you. You seem to be new to this code. So, "
                         "we recommend you add your baseurl on your own.")

    if datasetname == "camelyon16":
        baseurl = join(baseurl, "")
    elif datasetname == "glas":
        baseurl = join(baseurl, "GlaS-2015/Warwick QU Dataset (Released 2016_07_08)")

    if baseurl is None:
        raise ValueError("Sorry, it seems we are enable to recognize you. You seem to be new to this code. So, "
                         "we recommend you add your baseurl on your own.")

    return baseurl


class VisualiseMIL(object):
    def __init__(self, alpha=128, floating=3, height_tag=60, bins=100, rangeh=(0, 1),
                 color_map=mlp.cm.get_cmap("seismic"), height_tag_paper=130):
        """
        A visualisation tool for MIL predictions.

        :param alpha: the transparency value for the overlapped image.
        :param floating: int, number of decimals to display.
        :param height_tag: int, the height of the tag banner.
        :param bins: int, number of bins. Used when one wants to plot the distribution of the scores.
        :param rangeh: tuple, default range of the x-axis for the histograms.
        :param color_map: type of the color map to use.
        """
        super(VisualiseMIL, self).__init__()

        self.color_map = color_map  # default color map.
        self.alpha = alpha

        self.bins = bins
        self.rangeh = rangeh

        # precision of writing the probabilities.
        self.prec = "%." + str(floating) + "f"
        self.height_tag = height_tag
        self.height_tag_paper = height_tag_paper  # for the paper.
        self.y = int(self.height_tag / 4)  # y position of the text inside the tag. (first line)
        self.y2 = self.y * 2  # y position of the text inside the tag. (second line)
        self.y3 = self.y * 3  # y position of the text inside the tag. (3rd line)
        self.dx = 10  # how much space to put between LABELS (not word) inside the tag.
        self.space = 10  # (pixels) how much space to leave between images.

        # Fonts:
        self.font_regular = ImageFont.truetype("./fonts/Inconsolata/Inconsolata-Regular.ttf", size=15)
        self.font_bold = ImageFont.truetype("./fonts/Inconsolata/Inconsolata-Bold.ttf", size=15)

        self.font_bold_paper = ImageFont.truetype("./fonts/Inconsolata/Inconsolata-Bold.ttf", size=120)
        self.font_bold_paper_small = ImageFont.truetype("./fonts/Inconsolata/Inconsolata-Bold.ttf", size=50)

        # Colors:
        self.white = "rgb(255, 255, 255)"
        self.green = "rgb(0, 255, 0)"
        self.red = "rgb(255, 0, 0)"
        self.orange = "rgb(255,165,0)"

        # Margin:
        self.left_margin = 10  # the left margin.

        # dim of the input image.
        self.h = None
        self.w = None

    def convert_mask_into_heatmap(self, input_img, mask, binarize=False):
        """
        Convert a mask into a heatmap.

        :param input_img: PIL.Image.Image of type float32. The input image.
        :param mask: 2D numpy.ndarray (binary or continuous in [0, 1]).
        :param binarize: bool. If True, the mask is binarized by thresholding (values >=0.5 will be set to 1. ELse, 0).
        :return:
        """
        if binarize:
            mask = ((mask >= 0.5) * 1.).astype(np.float32)

        img_arr = self.color_map((mask * 255).astype(np.uint8))  # --> in [0, 1.], (h, w, 4)

        return self.superpose_two_images_using_alpha(input_img.copy(), Image.fromarray(np.uint8(img_arr * 255)),
                                                     self.alpha)

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
        forg.putalpha(alpha)  # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.putalpha
        # back.putalpha(int(alpha/4))
        back.paste(forg, (0, 0), forg)

        return back

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
        :param dx: int, how much space to use between LABELS (not word). Useful to compute the position of the next
        LABEL. (future)
        :return:
            . ImageDraw object with the text drawn on it as requested.
            . The next position x.
        """
        draw.text((x, y), label, fill=fill, font=font)
        x += font.getsize(label)[0] + dx

        return draw, x

    def create_tag_input(self, him, wim, label, name):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the input image.

        Written message: "Input: label  (h) him pix. x (w) wim pix."

        :param him: int, height of the image.
        :param wim: int, the width of the image containing the tag.
        :param label: str, the textual tag.
        :param name: str, name of the input image file.
        :return:
        """
        if label is None:
            label = "unknown"
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, "In.cl.:", self.white, self.font_regular, self.dx)
        draw, x = self.drawonit(draw, x, self.y, label, self.white, self.font_bold, self.dx)

        x = self.left_margin
        msg = "(h){}pix.x(w){}pix.".format(him, wim)
        self.drawonit(draw, x, self.y2, msg, self.white, self.font_bold, self.dx)

        return img_tag

    def create_tag_pred_mask(self, wim, label, probability, status, f1pos, f1neg):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the predicted map.

        Written message:
        "Class: label  probability % [correct or wrong] (h) him pix. x (w) wim pix. #Patches = "
        :param wim: int, width of the image.
        :param label: str, the class name.
        :param probability: float, the probability of the prediction.
        :param status: str, the status of the prediction: "correct", "wrong", None. If None, no display of the status.
        :param dice: float or None, Dice index. (if possible)
        :return:
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw, x, self.y, "Pred.cl.:", self.white, self.font_regular, self.dx)
        draw, x = self.drawonit(draw, x, self.y, label, self.white, self.font_bold, self.dx)

        # Jump to the second line (helpful when the name of the class is long).
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y2, "Prob.: {}%".format(str(self.prec % (probability * 100))),
                                self.white, self.font_regular, self.dx)

        if status == "correct":
            color = self.green
        elif status == "wrong":
            color = self.red
        elif status is None:
            color = self.orange
            status = "predicted"
        else:
            raise ValueError("Unsupported status `{}` .... [NOT OK]".format(status))

        draw, x = self.drawonit(draw, x, self.y2, "Status: [", self.white, self.font_regular, 0)
        draw, x = self.drawonit(draw, x, self.y2, "{}".format(status), color, self.font_bold, 0)
        self.drawonit(draw, x, self.y2, "]", self.white, self.font_regular, self.dx)

        x = self.left_margin
        f1posstr = "None" if status is None else str(self.prec % (f1pos * 100)) + "%"
        f1negstr = "None" if status is None else str(self.prec % (f1neg * 100)) + "%"
        draw, x = self.drawonit(draw, x, self.y3, "F1+: {}".format(f1posstr), self.white, self.font_regular, self.dx)
        draw, x = self.drawonit(draw, x, self.y3, "F1-: {}".format(f1negstr), self.white, self.font_regular, self.dx)

        return img_tag

    def create_tag_true_mask(self, wim, status):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong", None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw, x, self.y, "True mask:", self.white, self.font_regular, self.dx)
        draw, x = self.drawonit(draw, x, self.y, "[", self.white, self.font_regular, 0)
        draw, x = self.drawonit(draw, x, self.y, status, self.green, self.font_bold, 0)
        draw, x = self.drawonit(draw, x, self.y, "]", self.white, self.font_regular, self.dx)

        return img_tag

    def create_tag_heatmap_pred_mask(self, wim, iter):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the predicted mask.

        Written message:
        "Heatmap pred. mask.       [iter.?/Final]"
        :param wim: int, width of the image.
        :param iter: str, the number of iteration when this mask was draw. "final" to indicate that this is final
        prediction.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw, x, self.y, "Heatmap ped.mask.", self.white, self.font_regular, self.dx)
        draw, x = self.drawonit(draw, x, self.y, "[", self.white, self.font_regular, 0)
        draw, x = self.drawonit(draw, x, self.y, "iter.{}".format(iter), self.green, self.font_bold, 0)
        self.drawonit(draw, x, self.y, "]", self.white, self.font_regular, 0)

        return img_tag

    def get_class_name(self, name_classes, i):
        """
        Get the str name of the class based on the integer.

        :param name_classes: dict, {"class_name": int}.
        :param i: int or None, the class ID. None if unknown label.
        :return: str, the class name.
        """
        assert isinstance(i, int) or i is None, "'i' must be an integer. Provided: {}, {}".format(i, type(i))
        error_msg = "class ID `{}` does not exist within possible IDs `{}` .... [NOT OK]".format(
            i, list(name_classes.values()))
        assert (i in list(name_classes.values())) or (i is None), error_msg

        if i is not None:
            return list(name_classes.keys())[list(name_classes.values()).index(i)]
        else:
            return "Unknown"

    def __call__(self, input_img, probab, pred_label, pred_mask, f1pos, f1neg, name_classes, iter,
                 use_tags=True, label=None, mask=None, show_hists=False, bins=None, rangeh=None, name_file=""):
        """
        Visualise MIL prediction.

        :param input_img: PIL.Image.Image RGB uint8 image. of size (h, w).
        :param probab: float, the probability of the predicted class.
        :param pred_label: int, the ID of the predicted class. We allow the user to provide the prediction.
        Generally, it can be inferred from the scores.
        :param pred_mask: numpy.ndarray, 2D float matrix of size (h, w). The predicted mask.
        :param f1pos: float [0, 1]. Dice index over the positive regions.
        :param f1neg: float [0, 1]. Dice index over the negative regions.
        :param name_classes: dict, {"class_name": int}.
        :param iter: str, indicates the iteration when this call happens. "Final" to indicate this is the final
        prediction.
        :param use_tags: True/False, if True, additional information will be allowed to be displayed.
        :param label: int or None, the the ID of the true class of the input_image. None: if the true label is unknown.
        :param mask: numpy.ndarray or None, 2D float matrix of size (h, w). The true mask. None if the true mask is
        unknown.
        :param show_hists: True/False. If True, a histogram of the scores in each map will be displayed.
        :param bins: int, number of bins in the histogram. If None, self.bins will be used.
        :param rangeh: tuple, range of the histogram. If None, self.rangeh will be used.
        :param name_file: str, name of the input image file.
        :return: PIL.Image.Image RGB uint8 image.
        """
        assert isinstance(input_img, Image.Image), "'input_image' type must be `{}`, but we found `{}` .... [NOT OK]" \
                                                   "".format(Image.Image, type(input_img))
        assert isinstance(probab, float), "'probab' must of type `{}` but we found `{}` .... [NOT OK]".format(
            float, type(probab))
        assert isinstance(pred_label, int), "'pred_label' must be of type `{}` but we found `{}` .... [NOT " \
                                            "OK]".format(int, type(pred_label))
        assert (isinstance(label, int) or label is None), "'label' must be `{}` or None. We found `{}` .... [NOT " \
                                                          "OK]".format(int, type(label))
        assert isinstance(pred_mask, np.ndarray), "'pred_mask' must be `{}`, but we found `{}` .... [NOT OK]".format(
            np.ndarray, type(mask))
        assert isinstance(mask, np.ndarray) or mask is None, "'mask' must be `{}` or None, but we found `{}` .... [" \
                                                             "NOT OK]".format(np.ndarray, type(mask))
        assert isinstance(name_classes, dict), "'name_classes' must be of type `{}`, but we found `{}` .... [NOT " \
                                               "OK]".format(dict, type(name_classes))

        assert isinstance(use_tags, bool), "'use_tags' must be of type `{}`, but we found `{}` .... [NOT OK]".format(
            bool, type(use_tags))

        wim, him = input_img.size
        assert wim == pred_mask.shape[1] and him == pred_mask.shape[0], "predicted mask {} and image shape ({}, " \
                                                                        "{}) do not " \
                                                                        "match .... [NOT OK]".format(
            pred_mask.shape, him, wim)
        # convert masks into images.
        if mask is None:
            true_mask = np.zero((him, wim), dtype=np.float32)
        else:
            true_mask = mask

        mask_img = self.convert_mask_into_heatmap(input_img, true_mask, binarize=False)

        pred_mask_img = self.convert_mask_into_heatmap(input_img, pred_mask, binarize=False)
        pred_mask_bin_img = self.convert_mask_into_heatmap(input_img, pred_mask, binarize=True)

        # create tags
        if use_tags:
            input_tag = self.create_tag_input(him, wim, self.get_class_name(name_classes, label), name_file)
            true_mask_tag = self.create_tag_true_mask(wim, "unknown" if mask is None else "known")
            class_name = self.get_class_name(name_classes, pred_label)
            if label is not None:
                status = "correct" if label == pred_label else "wrong"
            else:
                status = "unknown"
            pred_mask_tag = self.create_tag_pred_mask(wim, class_name, probab, status, f1pos, f1neg)
            heat_pred_mask_tag = self.create_tag_heatmap_pred_mask(wim, iter)

        # creates histograms
        nbr_imgs = 4

        img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him))
        if use_tags:
            img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him + self.height_tag))

        list_imgs = [input_img, mask_img, pred_mask_bin_img, pred_mask_img]
        list_tags = [input_tag, true_mask_tag, pred_mask_tag, heat_pred_mask_tag]
        for i, img in enumerate(list_imgs):
            img_out.paste(img, (i * (wim + self.space), 0), None)
            if use_tags:
                img_out.paste(list_tags[i], (i * (wim + self.space), him))

        img_final = img_out

        return img_final


class VisualizePaper(VisualiseMIL):
    """
    Visualize overlapped images for the paper.
    """

    def create_tag_input(self, him, wim, label, name_file):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the input image.

        Written message: "Input: label  (h) him pix. x (w) wim pix."

        :param him: int, height of the image.
        :param wim: int, the width of the image containing the tag.
        :param label: str, the textual tag.
        :param name: str, name of the input image file.
        :return:
        """
        if label is None:
            label = "unknown"
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, "Input: {} | ".format(name_file), self.white, self.font_regular,
                                self.dx)
        draw, x = self.drawonit(draw, x, self.y, label, self.white, self.font_bold, self.dx)

        # msg = "(h){}pix.x(w){}pix.".format(him, wim)
        # self.drawonit(draw, x, self.y, msg, self.white, self.font_bold, self.dx)

        return img_tag

    def create_tag_pred_mask(self, wim, msg1, msg2):
        """

        :param wim:
        :param msg1:
        :param msg2:
        :return:
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg1, self.white, self.font_regular, self.dx)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y2, msg2, self.white, self.font_regular, self.dx)

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
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong", None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw, x, self.y, "True mask", self.white, self.font_regular, self.dx)

        return img_tag

    def __call__(self, name_classes, img, label, name_file, true_mask, per_method, methods, order_methods,
                 show_heat_map=False, show_tags=False, show_tag_paper=False, use_small_font_paper=False):
        """

        :param img:
        :param name_file:
        :param true_mask:
        :param per_method:
        :param show_heat_map: Bool. If true, we show heat maps. Else, we show binary masks.
        :param show_tags: Bool. If True, we show tags below the images.
        :return:
        """

        assert isinstance(img, Image.Image), "'input_image' type must be `{}`, but we found `{}` .... [NOT OK]" \
                                             "".format(Image.Image, type(img))
        assert isinstance(true_mask, np.ndarray) or true_mask is None, "'mask' must be `{}` or None, but we found `{}` .... [" \
                                                                       "NOT OK]".format(np.ndarray, type(true_mask))
        assert isinstance(name_classes, dict), "'name_classes' must be of type `{}`, but we found `{}` .... [NOT " \
                                               "OK]".format(dict, type(name_classes))

        wim, him = img.size
        assert wim == true_mask.shape[1] and him == true_mask.shape[0], "predicted mask {} and image shape ({}, " \
                                                                        "{}) do not " \
                                                                        "match .... [NOT OK]".format(
            true_mask.shape, him, wim)

        mask_img = self.convert_mask_into_heatmap(img, true_mask, binarize=False)
        true_mask_tag = self.create_tag_true_mask(wim)

        list_imgs = [img, mask_img]
        input_tag = self.create_tag_input(him, wim, self.get_class_name(name_classes, label), name_file)
        list_tags = [input_tag, true_mask_tag]
        for k in order_methods:
            if per_method[k]["pred_label"] is not None:
                pred_label = self.get_class_name(name_classes, int(per_method[k]["pred_label"]))
            else:
                pred_label = "--"

            f1_foreg = per_method[k]["f1_score_forg_avg"]
            f1_back = per_method[k]["f1_score_back_avg"]
            msg1 = "F1+: {}%  F1-: {}% ".format(self.prec % f1_foreg, self.prec % f1_back)
            msg2 = "Prediction: {} (Method: {})".format(pred_label, methods[k])
            list_tags.append(self.create_tag_pred_mask(wim, msg1, msg2))

            if show_heat_map:
                list_imgs.append(self.convert_mask_into_heatmap(img, per_method[k]["pred_mask"], binarize=False))
            else:
                list_imgs.append(self.convert_mask_into_heatmap(img, per_method[k]["binary_mask"], binarize=False))

        nbr_imgs = len(methods.keys()) + 2
        font = self.font_bold_paper
        if use_small_font_paper:
            font = self.font_bold_paper_small

        tag_paper_img = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), self.height_tag_paper))
        list_tags_paper = [self.create_tag_paper(wim, "Input", font), self.create_tag_paper(wim, "True mask", font)]
        for k in order_methods:
            list_tags_paper.append(self.create_tag_paper(wim, methods[k], font))

        img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him))
        if show_tags:
            img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him + self.height_tag))
        for i, img in enumerate(list_imgs):
            img_out.paste(img, (i * (wim + self.space), 0), None)
            tag_paper_img.paste(list_tags_paper[i], (i * (wim + self.space), 0), None)
            if show_tags:
                img_out.paste(list_tags[i], (i * (wim + self.space), him))

        if show_tag_paper:
            img_out_final = Image.new("RGB", (img_out.size[0], img_out.size[1] + self.height_tag_paper))
            img_out_final.paste(img_out, (0, 0), None)
            img_out_final.paste(tag_paper_img, (0, img_out.size[1]), None)
            img_out = img_out_final

        return img_out, tag_paper_img


class VisualizeImages(VisualizePaper):
    """
    Visualize images from dataset.
    """
    def __call__(self, name_classes, list_images, list_true_masks, list_labels, rows, columns, show_tags=False):
        """

        :param name_classes:
        :param list_images:
        :param list_true_masks:
        :return:
        """
        for i, msk in enumerate(list_true_masks):
            assert isinstance(msk, np.ndarray), "'mask' must be `{}` or None, but we found `{}` .... [" \
                                                "NOT OK]".format(np.ndarray, type(msk))
        for i, img in enumerate(list_images):
            assert isinstance(img, Image.Image), "'input_image' type must be `{}`, but we found `{}` .... [NOT OK]" \
                                                 "".format(Image.Image, type(img))

        assert isinstance(name_classes, dict), "'name_classes' must be of type `{}`, but we found `{}` .... [NOT " \
                                               "OK]".format(dict, type(name_classes))

        assert rows == 1, "We support only 1 row!!!! You asked for {}".format(rows)
        assert len(list_images) == len(list_true_masks), "list_images and list_true_masks must have the same number " \
                                                         "of elements. You provided: len(list_images) = {}," \
                                                         "len(list_true_masks) = {}".format(len(list_images),
                                                                                            len(list_true_masks))

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
            img_tags.paste(self.create_tag_paper(wim, self.get_class_name(name_classes, list_labels[i])),
                           (p + i * self.space, 0), None)
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
