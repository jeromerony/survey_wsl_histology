"""
Info: https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/
"""

__author__ = "Soufiane Belharbi, https://sbelharbi.github.io/"
__copyright__ = "Copyright 2018, ÉTS-Montréal"
__license__ = "GPL"
__version__ = "3"
__maintainer__ = "Soufiane Belharbi"
__email__ = "soufiane.belharbi.1@etsmtl.net"

import sys
import copy
import glob
import os
from os.path import join
import xml.etree.ElementTree as ET
import json
import warnings
import datetime as dt
from multiprocessing import Process
import shutil

import openslide
import numpy as np
import getpass
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.measure import points_in_poly
import tqdm
import csv

from tools import Dict2Obj, chunks_into_n
from tools import announce_msg, draw_hist_probability_of_list

from openslide_python_fix import _load_image_lessthan_2_29, _load_image_morethan_2_29

import reproducibility


class WSI_Factory(object):
    """
    Class that deals Whole-Slide Images. It is made for CAMELYON16 dataset (
    https://camelyon16.grand-challenge.org/Home/) in mind (*.tif files). But, it is supposed to work on WSI image in
    general. It uses OpenSlide (https://openslide.org/) using its Python API (https://openslide.org/api/python/).

    """
    def __init__(self, args):
        """

        :param args: object, contains the attributes of the WSI module:
            level: int, level of resolution in the wsi. Levels are numbered from 0 (highest resolution) to
        slide.level_count - 1 (lowest resolution). [JUST FOR DEBUG]
            rgb_min: int, minimal value for rgb. Used for thresholding.
            delta_w: int, delta of width to move the patch.
            delta_h: int, delta of height to move the patch.
            h_: int, the width of the patch to be sampled.
            w_: int, the height of the patch to be sampled.
            tissue_mask_min: float in ]0, 1], the minimum percentage of tissue mask required to accept the patch.
            p0: float in [0, 1], the minimum percentage of cancerous pixels required within a patch.
            p1: float in ]0, 1], the maximum percentage of cancerous pixels required within a patch.
            delta_w_inter: int, intermediate delta of width to move the patch.
            dichotomy_step: int, step used in a pseudo-dichotomy when sampling metastatic patches.
        """
        self.level = args.level
        self.rgb_min = args.rgb_min
        self.delta_w = args.delta_w
        self.delta_h = args.delta_h
        self.h_ = args.h_
        self.w_ = args.w_
        self.tissue_mask_min = args.tissue_mask_min
        self.p0 = args.p0
        self.p1 = args.p1
        self.delta_w_inter = args.delta_w_inter
        self.dichotomy_step = args.dichotomy_step

        assert (args.delta_w % args.delta_w_inter) == 0, "Inconsistent deltas .... [NOT OK]"

    @staticmethod
    def indexer(him, wim, h_, w_, delta_h, delta_w):
        """
        Generator of positions of patches: i, j, h_, w_.

        :param him: int, Height of the image (WSI).
        :param wim: int, Width of the image (WSI).
        :param h_: height of the patch.
        :param w_: Width of the patch.
        :param delta_h: int, delta height.
        :param delta_w: int, delta width.
        :return: The next coordinates (i, j) of patch with size (h_, w_).
        """
        assert (him >= h_) and (wim >= w_), "Inconsistent sizes."

        for r in range(0, him, delta_h):
            if r + h_ > him:
                break

            for c in range(0, wim, delta_w):
                if c + w_ > wim:
                    break
                yield r, c

    @staticmethod
    def random_indexer_grid_wsi_patch_level(him, wim, h_, w_, delta_h, delta_w):
        """
        Generator of patches' coordinates: (i, j) [i: row, j: column]
        Patches have size (w_: width, h_: height).

        The coordinates are generated randomly over a grid determined by the sliding strides `delta_h` and `delta_w`.

        :param him: int, Height of the image (WSI).
        :param wim: int, Width of the image (WSI).
        :param h_: height of the patch.
        :param w_: Width of the patch.
        :param delta_h: int, delta height.
        :param delta_w: int, delta width.
        :return:
        """
        columns = np.arange(0, wim, delta_w)
        rows = np.arange(0, him, delta_h)

        # return a valid position
        while True:
            r = np.random.choice(rows, 1)[0]
            c = np.random.choice(columns, 1)[0]

            if ((r + h_) <= him) and ((c + w_) <= wim):
                return r, c

    @staticmethod
    def find_largest_rectangle_round_tumor(annotation):
        """
        Find the largest rectangle around each tumor.
        This will help speeding up sampling of cancerous regions.

        :param annotation: dict, result of self.get_annotation().
        :return: List of rectangles.
        """
        def get_rectangle(region):
            """

            :param region: [N, 2] numpy.ndarray matrix, contains (X, Y). (x is on the width axis).
            :return: upper_left, lower_right corners coordinates.
                upper_left: x, y.
                lower_right: x, y.
            """
            upper_left = [region[:, 0].min(), region[:, 1].min()]
            lower_right = [region[:, 0].max(), region[:, 1].max()]

            return upper_left, lower_right

        list_regions = []
        for tumor in annotation["positive"]:
            vertices = tumor["vertices"]
            list_regions.append(get_rectangle(vertices))

        return list_regions

    @staticmethod
    def indexer_metastatic_regions(list_regions, him, wim, h_, w_, delta_h, delta_w, allow_overlap_top=False):
        """
        Generate patch coordinates. For fast computation, this target directly the cancerous regions and do not spend
        time in normal regions. It finds the largest rectangle around the tumor. This rectangle is splitted in a
        fixed grid fashion.

        Cons:
        1. We still spend some time on normal regions before we get to the border of the metastatic region.
        2. Since the sampling starts far away from the border, the first patch that contains metastatic pixels may
        be entirely metastatic. This (worst case) may happen when the previous patch ends up one pixel before the
        border. Therefore, the next patch will likely to be full of cancerous pixels. This means, we missed many
        patches that has for instance 2% of cancerous pixels. Another `better` strategy can be used: at each row,
        find the first ACCEPTABLE patch. For instance, if we accept patches with at least 20% of metastatic pixels,
        then, at the current row, find the position of the first patch that meets this constraint. Then,
        start sampling from there. ---> This behavior can be simulated within the loop that gets the patches WITHOUT
        the need that this function knows the constraints on the patches. All we need is to sample with delta_w = 1,
        then, if a patch has 0% cancerous pixels, move (horizontally) to the next patch with a delta of the same
        width of the patch. In this scenario, only horizontal patches are sampled using delta_w = 1. The delta_h is
        set to the predefined value. This way of sampling requires to store the patches' indices in row fashion.

        :param list_regions: list of cancerous regions. It is the output of self.find_largest_rectange_round_tumor()
        :param him: int, WSI height.
        :param wim: int, WSI width.
        :param h_: int, the width of the patch to be sampled.
        :param w_: int, the height of the patch to be sampled.
        :param delta_h: int, delta height.
        :param delta_w: int, delta width.
        :param allow_overlap_top: bool, if True, the first row of the top of the largest rectangle containing the
        tumor is translated by -h_/2.
        :return: list of list of [i, j] (index of the patch) where i is the row, j is the column.
        """
        assert (him >= h_) and (wim >= w_), "Inconsistent sizes .... [NOT OK]"

        if delta_w != 1:
            warnings.warn("To sample metastatic patches, we recommend using a delta_w=1. It increases the "
                          "chance to obtain patches with small number of cancerous pixels. You are using "
                          "delta_w=`{}`. However, this will take a long time.".format(delta_w))

        list_rows = []
        for region in list_regions:
            upper_left, lower_right = region
            if allow_overlap_top:
                # if we et h_ to the top left corner, the first row of the sampled patches will not touch the cancerous
                # region. To avoid this, we allow the first row to overlap with half of the cancerous region.
                start_c, start_r = max(upper_left[0] - w_, 0), max(upper_left[1] - int(h_ / 2), 0)
            else:
                start_c, start_r = max(upper_left[0] - w_, 0), max(upper_left[1] - h_, 0)
            end_c, end_r = min(lower_right[0] + w_, wim), min(lower_right[1] + h_, him)

            for r in range(start_r, end_r, delta_h):
                if r + h_ > him:
                    break
                row = []
                for c in range(start_c, end_c, delta_w):
                    if c + w_ > wim:
                        break

                    row.append([r, c])
                list_rows.append(row)

        return list_rows

    @staticmethod
    def find_first_metastatic_column(mtx):
        """
        Find the index of the first column containing metastatic pixels.

        :param mtx: numpy.ndarray 2D matrix. The binary mask of the metastatic region.
        :return: int, or None. The index of the column if there is any, or None.
        """
        if np.sum(mtx) == 0:
            return None
        vec = np.sum(mtx, axis=0)

        return np.nonzero(np.uint8(vec != 0))[0][0]

    @staticmethod
    def padd_region(rectangle, him, wim, h_, w_, allow_overlap_top=False):
        """
        Padd a region by extending its corner a little bit. We do not add any extra information, we just enlarge the
        region (therefore, the extra information is taken from the WSI).

        :param rectangle: list of upper left and lower right coordinates of the rectangle.
        :param him: int, WSI height.
        :param wim: int, WSI width.
        :param h_: int, the width of the patch to be sampled.
        :param w_: int, the height of the patch to be sampled.
        :param allow_overlap_top: bool, if True, the first row of the top of the largest rectangle containing the
        tumor is translated by -h_/2.
        :return: an updated rectangle.
        """
        upper_left = rectangle[0]
        lower_right = rectangle[1]

        if allow_overlap_top:
            # if we et h_ to the top left corner, the first row of the sampled patches will not touch the cancerous
            # region. To avoid this, we allow the first row to overlap with half of the cancerous region.
            start_c, start_r = max(upper_left[0] - w_, 0), max(upper_left[1] - int(h_ / 2), 0)
        else:
            start_c, start_r = max(upper_left[0] - w_, 0), max(upper_left[1] - h_, 0)
        end_c, end_r = min(lower_right[0] + w_, wim), min(lower_right[1] + h_, him)

        return [[start_c, start_r], [end_c, end_r]]

    @staticmethod
    def deal_with_nested_metastatic_egions(list_rectangles):
        """
        In many cases, the annotation is composed of nested polygons. In this case, It is safe to take the largest
        region since when we obtain its binary mask of the cancerous pixels we use all the regions.
        See: self.convert_tumor_coordinates_into_mask().

        Note: rectangle's coordinates are (x, y), where x is in the width of the image.

        :param list_rectangles: list of rectangle coordinates.
        :return: list of (non-nested) rectangles.
        """
        def check_if_rectangle_is_inside_other_rectangles(rectangle, l_rectangles):
            upper_left = rectangle[0]
            lower_right = rectangle[1]
            for rec in l_rectangles:
                u_l = rec[0]
                l_r = rec[1]
                # check if rectangle is inside another rectangle.
                if (upper_left[0] > u_l[0]) and (upper_left[1] > u_l[1]) and (lower_right[0] < l_r[0]) and (
                        lower_right[0] < l_r[0]):
                    return True

            return False

        output = []
        for rectangle in list_rectangles:
            if not check_if_rectangle_is_inside_other_rectangles(rectangle, list_rectangles):
                output.append(rectangle)

        assert len(output) >= 1, "Well we get a problem here. We expected to obtain at least one region, " \
                                 "we found `{}`".format(len(output))
        return output

    @staticmethod
    def get_height_width_rectangle(rectangle):
        """
        Returns the width and the height of a rectangle.
        :param rectangle:
        :return: height, width.
        """
        upper_left = rectangle[0]
        lower_right = rectangle[1]

        h = lower_right[1] - upper_left[1]
        w = lower_right[0] - upper_left[0]

        assert (h > 0) and (w > 0), "Something wrong .... [NOT OK]"

        return h, w

    @staticmethod
    def get_height_width_level_0(slide):
        """
        Get the height and the width of the WSI at level 0.

        :param slide: object of openslide.Openlside(path).
        :return: h, w: int, int, height and width of the WSI at level 0.
        """
        h = slide.properties["openslide.level[0].height"]
        w = slide.properties["openslide.level[0].width"]

        return int(h), int(w)

    @staticmethod
    def get_height_width_level_l(slide, l):
        """
        Get the height and the width of the WSI at level l.

        :param slide: object of openslide.Openlside(path).
        :return: h, w: int, int, height and width of the WSI at level l.
        """
        h = slide.properties["openslide.level[{}].height".format(l)]
        w = slide.properties["openslide.level[{}].width".format(l)]

        return int(h), int(w)

    @staticmethod
    def read_wsi_2_rgb(path, level):
        """
        Read a WSI from disc. Return the image.

        :param path: str, path to the wsi.
        :param level: int, level of resolution.
        :return: numpy array of size (h, w, 3) that is the image of level self.level.
        """
        slide = openslide.OpenSlide(path)
        img_rgb = np.array(slide.read_region(
            (0, 0), level, slide.level_dimensions[level]).convert("RGB"))
        return img_rgb

    @staticmethod
    def get_tissue_mask(img_rgb, rgb_min):
        """
        Apply Otsu threshold to get the only the tissue mask.

        :param img_rgb: (h, w, 3) RGB image (numpy format).
        :param rgb_min: int, minimal value for rgb. Used for thresholding.
        :return:
        """
        img_hsv = rgb2hsv(img_rgb)

        background_r = img_rgb[:, :, 0] > threshold_otsu(img_rgb[:, :, 0])
        background_g = img_rgb[:, :, 1] > threshold_otsu(img_rgb[:, :, 1])
        background_b = img_rgb[:, :, 2] > threshold_otsu(img_rgb[:, :, 2])

        tissue_rgb = np.logical_not(background_r & background_g & background_b)

        tissue_s = img_hsv[:, :, 1] > threshold_otsu(img_hsv[:, :, 1])

        min_r = img_rgb[:, :, 0] > rgb_min
        min_g = img_rgb[:, :, 1] > rgb_min
        min_b = img_rgb[:, :, 2] > rgb_min

        tissue_mask = tissue_s & tissue_rgb & min_r & min_g & min_b

        return tissue_mask

    @staticmethod
    def check_if_tissue_mask_has_enough_tissue(tissue_mask, thres):
        """
        Check if a tissue mask has enough pixel on.

        :param tissue_mask: numpy.ndarray matrix of bool.
        :param thres: int, the
        :return: bool, float.
            bool: True, if there are more tissue pixels that the required minimum. Else, False.
            float, in [0, 1], percentage of pixels that are tissue.
        """
        assert tissue_mask.ndim == 2, "Inconsistent input. Expect a 2D matrix."

        h, w = tissue_mask.shape
        total = h * w
        tissue_perentage = np.uint8(tissue_mask).sum() / float(total)
        if tissue_perentage >= thres:
            return True, tissue_perentage
        else:
            return False, tissue_perentage

    @staticmethod
    def check_if_img_thresholdable(img_rgb):
        """
        Check if the image is multi-color at each channel in RGB and HSV space.

        :param img_rgb: numpy.ndarray, (h, w, 3) RGB image.
        :return:
        """
        def min_max_same(plan):
            return plan.min() == plan.max()

        assert img_rgb.ndim == 3, "Inconsistent shape."

        # check RGB
        error_rgb = any([min_max_same(img_rgb[:, :, i]) for i in range(3)])

        # check HSV
        img_hsv = rgb2hsv(img_rgb)
        error_hsv = min_max_same(img_hsv[:, :, 1])

        return not (error_hsv or error_rgb)


    @staticmethod
    def check_if_patch_has_enough_metastatic_pixels(mask, p0, p1):
        """
        Check if a patch has enough cancerous pixels. p0 <= p <= p1.

        :param mask: numpy.ndarray of type bool. Mask where 1 indicates metastatic pixel.
        :param p0: float in [0, 1], the minimum percentage of cancerous pixels required within a patch.
        :param p1: float in ]0, 1], the maximum percentage of cancerous pixels required within a patch.
        :return: bool, and float.
            bool: True, if there are enough metastatic pixels. Else, False.
            float: [0, 1], percentage of metastatic pixels.
        """
        assert mask.ndim == 2, "Inconsistent input. Expect a 2D matrix."

        h, w = mask.shape
        total = h * w
        percentage_cancer = np.uint8(mask).sum() / float(total)

        if p0 <= percentage_cancer <= p1:
            return True, percentage_cancer
        else:
            return False, percentage_cancer

    @staticmethod
    def check_if_patch_has_enough_metastatic_pixels_first_pass(mask, p0):
        """
        Check if a patch has enough cancerous pixels. p0 <= p .

        :param mask: numpy.ndarray of type bool. Mask where 1 indicates metastatic pixel.
        :param p0: float in [0, 1], the minimum percentage of cancerous pixels required within a patch.
        :return: bool, and float.
            bool: True, if there are enough metastatic pixels. Else, False.
            float: [0, 1], percentage of metastatic pixels.
        """
        assert mask.ndim == 2, "Inconsistent input. Expect a 2D matrix."

        h, w = mask.shape
        total = h * w
        percentage_cancer = np.uint8(mask).sum() / float(total)

        if percentage_cancer >= p0:
            return True, percentage_cancer
        else:
            return False, percentage_cancer

    @staticmethod
    def get_annotation(pathxml):
        """
        Load the xml annotation of the tumors. Used only for WSI with tumors.

        Output:
            Tumor coordinates (x, y) within the WSI:
            y: represents the height.
            x: represents the width.

        :param pathxml: str, path to the xml file (CAMELYON16).
        :return: dict
        """
        root = ET.parse(pathxml).getroot()

        annotations_tumor = root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')

        annotations_positive = annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        out_dict = {}
        out_dict["positive"] = []
        out_dict["negative"] = []

        for annotation in annotations_positive:
            x = list(map(lambda x_: float(x_.get('X')),
                         annotation.findall('./Coordinates/Coordinate')))
            y = list(map(lambda y_: float(y_.get('Y')),
                         annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([x, y]).astype(int).transpose()
            name = annotation.attrib["Name"]
            out_dict["positive"].append({"name": name, 'vertices': np.array(vertices)})

        for annotation in annotations_negative:
            x = list(map(lambda x_: float(x_.get('X')),
                         annotation.findall('./Coordinates/Coordinate')))
            y = list(map(lambda y_: float(y_.get('Y')),
                         annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([x, y]).astype(int).transpose()
            name = annotation.attrib['Name']
            out_dict['negative'].append({'name': name, 'vertices': np.array(vertices)})

        return out_dict

    @staticmethod
    def convert_tumor_coordinates_into_mask_points_in_poly(x_c, y_c, img_rgb, anno_dict):
        """
        Given the coordinates of a patch, build a mask image where all tumor pixels are set to 1 while normal pixels
        are set to 0.

        Warning: This is VERY-VERY-VERY SLOW DUE TO THE HOW SLOW IS THE FUNCTION skimage.measure.points_in_poly()


        :param x_c: int, the x-coordinate of the upper left corner of the patch within the WSI.
        :param y_c: int, the y-coordinate of the upper left corner of the patch within the WSI.
        :param img_rgb: numpy.ndarray, RGB mage of size (h, w, 3).
        :param anno_dict: dict, the output of self.get_annotation.
        :return: bool numpy.ndarray, a matrix with size (h, w) where pixels with True value are cancerous.
        """
        h, w, _ = img_rgb.shape
        output_mask = np.zeros((h, w), dtype=bool)

        x = np.arange(0, w) + x_c
        y = np.arange(0, h) + y_c
        xv, yv = np.meshgrid(x, y)
        grid = np.array([xv.flatten(), yv.flatten()]).T

        t0 = dt.datetime.now()
        for tumor in anno_dict["positive"]:
            vertices = tumor["vertices"]
            tmp_mask = points_in_poly(grid, vertices)
            output_mask = np.logical_or(output_mask, tmp_mask.reshape((h, w)))
        print("Checking p in poly took {}".format(dt.datetime.now() - t0))

        return output_mask

    @staticmethod
    def convert_tumor_coordinates_into_mask(x_c, y_c, img_rgb, anno_dict):
        """
        Given the coordinates of a patch (or a region), build a mask image where all tumor pixels are set to 1 while
        normal pixels are set to 0.

        NOTE: THIS IS WAY FASTER THAN SELF.convert_tumor_coordinates_into_mask_points_in_poly(),  LIKE NOT COMPARISON!


        :param x_c: int, the x-coordinate of the upper left corner of the patch (region) within the WSI.
        :param y_c: int, the y-coordinate of the upper left corner of the patch (region) within the WSI.
        :param img_rgb: numpy.ndarray, RGB mage of size (h, w, 3).
        :param anno_dict: dict, the output of self.get_annotation.
        :return: bool numpy.ndarray, a matrix with size (h, w) where pixels with True value are cancerous.
        """
        h, w, _ = img_rgb.shape
        output_mask = np.zeros((h, w), dtype=bool)

        for tumor in anno_dict["positive"]:
            vertices = tumor["vertices"].copy()  # [X, Y], use a copy to avoid modifying anno_dict.
            # Re-reference the vertices within the current
            vertices[:, 0] = vertices[:, 0] - x_c
            vertices[:, 1] = vertices[:, 1] - y_c

            vertices = vertices.tolist()

            vertices = [tuple(l) for l in vertices]
            img = Image.new('1', (w, h), 0)
            ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
            tmp_mask = np.array(img)
            output_mask = np.logical_or(output_mask, tmp_mask)

        return output_mask


def find_possibly_acceptable_metastatic_patches(list_patches, slide, wsi_fact, step, annotation):
    """
    Given a list of coordinates of patches, find all the patches that may be accepted.
    We first find the first accepted patch using a stride of 1. Then, we take all the following patches with stride
    delta_w (independently if they fit the constraints of not).

    Use a pseudo-dichotomic binary search.

    :param list_patches: list of [i, j] (patches coordinates).
    :param slide: instance of openslide.Opneslide().
    :param wsi_fact: instance of WSI_Factory().
    :param step: int, dichotomoy step.
    :param annotation: dict, output of WSI_Factory.get_annotation()
    :return: list of possibly acceptable patches to be processed later.
    """
    patches_to_process = []
    first_found = False
    position = 0
    last_position_not_found = 0
    last_position_found = None
    i_lf, j_lf = None, None
    stop = False
    k = 0
    if step >= len(list_patches):
        step = int(len(list_patches) / 2)

    def check_if_OK(i, j):
        """
        Check if a patch is acceptable as a metastases.

        :return: bool. True/False.
        """
        t0 = dt.datetime.now()
        patch = slide.read_region((j, i), 0, (wsi_fact.w_, wsi_fact.h_)).convert('RGB')
        # print("openslide.read_region() took {} ".format(dt.datetime.now() - t0))
        t0 = dt.datetime.now()
        patch_array = np.array(patch)
        # print("np.array(patch) took {} ".format(dt.datetime.now() - t0))

        # 2.2 TEST 1: Compute the tissue mask. If there is enough tissue mask, proceed, else, abort.

        # Avoid the case where the patch has only once color (in this case, the patch is entirely white):
        # https://github.com/scikit-image/scikit-image/issues/1856

        if not wsi_fact.check_if_img_thresholdable(patch_array):
            return False

        tissue_mask = wsi_fact.get_tissue_mask(patch_array, wsi_fact.tissue_mask_min)
        checked_tissue, tissue_perentage = wsi_fact.check_if_tissue_mask_has_enough_tissue(
            tissue_mask, wsi_fact.tissue_mask_min)
        if not checked_tissue:
            return False

        # 2.3 TEST 2: Compute the percentage of cancerous pixels. If there is enough, proceed, else, abort.
        cancerous_mask = wsi_fact.convert_tumor_coordinates_into_mask(j, i, patch_array, annotation)
        checked_metast, percentage_cancer = wsi_fact.check_if_patch_has_enough_metastatic_pixels(
            cancerous_mask, wsi_fact.p0, wsi_fact.p1)
        if not checked_metast:
            return False

        return True

    while not stop:
        i, j = list_patches[position]
        k += 1

        if first_found:  # get all the patches with stride delta_w independently of the constraints.
            if j - previous_j == wsi_fact.delta_w:
                patches_to_process.append([i, j])
                previous_j = j

            position += 1
            if position >= len(list_patches):
                stop = True
            continue  # do not check the constraints
        else:
            previous_j = j

        if check_if_OK(i, j):
            last_position_found = position
            i_lf, j_lf = i, j

            step = int(step / 2)
            position = last_position_not_found + step

            if step <= 2:  # take this one and stop the search. [not exact result]
                first_found = True
                patches_to_process.append([i, j])
                position = min(position + 1, len(list_patches) - 1)
        else:
            last_position_not_found = position
            position = position + step

            if step <= 2:
                if last_position_found:
                    # Take the last found position and stop the search. [not exact result]
                    first_found = True
                    patches_to_process.append([i_lf, j_lf])
                    position = last_position_found
                else:  # we didn't find any OK patch, therefore, stop.
                    stop = True

        if position >= len(list_patches):
            stop = True

    return patches_to_process


def process_sample_metastatic_patches_one_wsi_pass_one_slow(args):
    """
    Sample all metastatic patches from a WSI with tumor.

    WARNING: VERY SLOW. DO NOT USE IT.

    Note: in this phase, there is no constraint on the maximum percentage of cancerous pixels within the patch. This
    means we take all patches that have p >= p0. Later, in the second phase, we filter the patches to calibrate
    between patches that have few and large percentage of cancerous pixels.

    :param args: object, contains all the necessary information.
    :return:
    """
    path_wsi = args.path_wsi
    path_xml = args.path_xml
    print("Processing file: `{}`".format(path_wsi))

    # Check if the file has not already been processed:
    outd = args.outd
    if not os.path.exists(outd):
        os.makedirs(outd, exist_ok=True)
    csv_name = join(outd, path_wsi.split(os.sep)[-1].split(".")[0] + ".csv")

    if os.path.isfile(csv_name):
        print("File `{}` has already been processed. Output in`{}` already exists .... [OK]".format(path_wsi, csv_name))
        return 0

    wsimager = WSI_Factory(args)

    slide = openslide.OpenSlide(path_wsi)
    annotation = wsimager.get_annotation(path_xml)

    # 1. Get h, w of WSI
    him, wim = wsimager.get_height_width_level_0(slide)

    # 2. Get the metastatic regions.
    metastatic_regions = wsimager.find_largest_rectangle_round_tumor(annotation)
    list_rows = wsimager.indexer_metastatic_regions(metastatic_regions, him, wim, wsimager.h_, wsimager.w_,
                                                    wsimager.delta_h, wsimager.delta_w_inter,
                                                    (wsimager.h_ == wsimager.delta_h))
    total_number_patches = sum([len(t) for t in list_rows])
    print("File: `{}` has `{}` intermediate patches (stride = `{}`) .... [OK]".format(
        path_wsi.split(os.sep)[-1], total_number_patches, wsimager.delta_w_inter))

    # 2. Sample patches.
    r = 0
    k = 0
    list_patches = []
    for row in tqdm.tqdm(list_rows, ncols=80, total=len(list_rows)):
        # Fast-forward to the first OK-patch: Find the first acceptable patch. Once found, make a stride
        # using delta_w.
        patches_to_process = find_possibly_acceptable_metastatic_patches(
            row, slide, wsimager, wsimager.dichotomy_step, annotation)

        for i, j in patches_to_process:
            # 2.1 Get the patch

            patch = slide.read_region((j, i), 0, (wsimager.w_, wsimager.h_)).convert('RGB')
            patch_array = np.array(patch)

            # 2.2 TEST 1: Compute the tissue mask. If there is enough tissue mask, proceed, else, abort.

            # Avoid the case where the patch has only once color (in this case, the patch is entirely white):
            # https://github.com/scikit-image/scikit-image/issues/1856

            if not wsimager.check_if_img_thresholdable(patch_array):
                continue

            tissue_mask = wsimager.get_tissue_mask(patch_array, wsimager.tissue_mask_min)
            checked_tissue, tissue_perentage = wsimager.check_if_tissue_mask_has_enough_tissue(
                tissue_mask, wsimager.tissue_mask_min)
            if not checked_tissue:
                continue

            # 2.3 TEST 2: Compute the percentage of cancerous pixels. If there is enough, proceed, else, abort.
            cancerous_mask = wsimager.convert_tumor_coordinates_into_mask(j, i, patch_array, annotation)
            checked_metast, percentage_cancer = wsimager.check_if_patch_has_enough_metastatic_pixels_first_pass(
                cancerous_mask, wsimager.p0)
            if not checked_metast:
                continue

            # PATCH IS OK. TAKE IT.

            # Add more info: file-image name, file-xml, x, y, tissue percentage, percentage of cancerous pixels.
            list_patches.append([join(*path_wsi.split(os.sep)[-3:]), join(*path_xml.split(os.sep)[-3:]), j, i,
                                 tissue_perentage, percentage_cancer])

            # DEBUG
            if args.debug:
                debug_outd = join("./debug/wsi/patches/", join(*path_wsi.split(os.sep)[-3:]).split(".")[0])
                if not os.path.exists(debug_outd):
                    os.makedirs(debug_outd, exist_ok=True)

                tag = "row_{}_patch_{}_x_{}_y_{}_tissue_{}_metastatic_{}".format(
                    r, k, j, i, tissue_perentage, percentage_cancer)
                patch.save(join(debug_outd, "Patch_" + tag + ".png"), "PNG")
                Image.fromarray(np.uint8(cancerous_mask) * 255).save(join(debug_outd, "mask_" + tag + ".png"), "PNG")
            k += 1
        r += 1

    # Save the patches info in *.csv file.
    with open(csv_name, 'w') as fcsv:
        filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["wsi file name", "xml file name", "x", "y", "Tissue percentage", "Percentage of "
                                                                                              "cancerous pixels"])
        for elem in list_patches:
            filewriter.writerow(elem)

    return 0


def sample_metastatic_patches_one_wsi_pass_one(args):
    """
    Sample all metastatic patches from a WSI with tumor.


    NOTE: VERY FAST!!!

    Note: in this phase, there is no constraint on the maximum percentage of cancerous pixels within the patch. This
    means we take all patches that have p >= p0. Later, in the second phase, we filter the patches to calibrate
    between patches that have few and large percentage of cancerous pixels.

    :param args: object, contains all the necessary information.
    :return:
    """
    path_wsi = args.path_wsi
    path_xml = args.path_xml

    assert os.path.isfile(path_wsi), "File `{}` does not exist ... [NOT OK]".format(path_wsi)
    assert os.path.isfile(path_xml), "File `{}` does not exist ... [NOT OK]".format(path_xml)

    announce_msg("Going to processing file: `{}`".format(path_wsi))

    # Check if the file has not already been processed:
    outd = args.outd
    if not os.path.exists(outd):
        os.makedirs(outd, exist_ok=True)

    csv_name = join(outd, path_wsi.split(os.sep)[-1].split(".")[0] + ".csv")

    if os.path.isfile(csv_name):
        announce_msg(
            "File `{}` has already been processed. Output in`{}` already exists .... [OK]".format(path_wsi, csv_name)
        )
        return 0

    announce_msg("Started processing file: `{}`".format(path_wsi))

    wsimager = WSI_Factory(args)

    slide = openslide.OpenSlide(path_wsi)
    annotation = wsimager.get_annotation(path_xml)

    # 1. Get h, w of WSI
    him, wim = wsimager.get_height_width_level_0(slide)

    # 2. Get the metastatic regions (the largest rectangle around each region).
    metastatic_regions = wsimager.find_largest_rectangle_round_tumor(annotation)
    # Deal with nested regions: Take the largest region among nested regions. (avoid unnecessary computation for
    # speedup.)
    metastatic_regions = wsimager.deal_with_nested_metastatic_egions(metastatic_regions)

    list_patches = []
    cnt_reg = 0
    announce_msg("Processing cancerous regions ... [OK]")
    for m_region in tqdm.tqdm(metastatic_regions, ncols=80, total=len(metastatic_regions)):
        # Grab the entire region!!!! This will be problematic if by chance the region is huge, and there is no enough
        # memory.
        m_region = wsimager.padd_region(m_region, him, wim, wsimager.h_, wsimager.w_, (wsimager.h_ == wsimager.delta_h))
        upper_left = m_region[0]
        lower_right = m_region[1]
        h_rec, w_rec = wsimager.get_height_width_rectangle(m_region)

        cnt_rows = 0
        x1, y1 = upper_left
        x2, y2 = lower_right

        for r in range(0, h_rec, wsimager.delta_h):
            if r + wsimager.h_ > h_rec:
                continue
            # Crop the current row from the WSI.
            # Check which _load_image() function to use depending on the size of the row.
            if (wsimager.delta_h * w_rec) >= 2 ** 29:
                openslide.lowlevel._load_image = _load_image_morethan_2_29
            else:
                openslide.lowlevel._load_image = _load_image_lessthan_2_29

            row = slide.read_region((x1, y1 + r), 0, (w_rec, wsimager.delta_h)).convert("RGB")
            # row = region.crop([0, r, w_rec, r + wsimager.h_])
            row_arr = np.array(row)
            # Check if the cropped row is thesholdable:
            if not wsimager.check_if_img_thresholdable(row_arr):
                continue

            # Compute the tissue mask of the row.
            row_tissue_mask = wsimager.get_tissue_mask(row_arr, wsimager.tissue_mask_min)
            row_metasta = wsimager.convert_tumor_coordinates_into_mask(x1, y1 + r, row_arr, annotation)

            # find the first column containing metastatic pixels.
            f_clm = wsimager.find_first_metastatic_column(row_metasta)

            if f_clm is None:  # the matrix has none metastatic pixels.
                continue

            # Step back by w_.
            f_clm = max(f_clm - wsimager.w_, 0)
            # Now, fid the first acceptable patch: find its column.
            found_first = False
            for cl in range(f_clm, w_rec, wsimager.delta_w_inter):
                if cl + wsimager.w_ > w_rec:
                    continue
                # Check the tissue mass:
                patch_tissue = row_tissue_mask[:, cl: cl + wsimager.w_]
                checked_tissue, tissue_perentage = wsimager.check_if_tissue_mask_has_enough_tissue(
                    patch_tissue, wsimager.tissue_mask_min)
                if not checked_tissue:
                    continue

                # Check if patch has enough cancerous pixels.
                patch_metas = row_metasta[:, cl: cl + wsimager.w_]
                checked_metast, percentage_cancer = wsimager.check_if_patch_has_enough_metastatic_pixels_first_pass(
                    patch_metas, wsimager.p0)
                if not checked_metast:
                    continue

                # This is the first patch.
                f_clm = cl
                found_first = True
                break

            if not found_first:
                continue

            # Now sample all the acceptable patches in this row starting from the column f_clm
            cnt_patches = 0
            for cl in range(f_clm, w_rec, wsimager.delta_w):
                if cl + wsimager.w_ > w_rec:
                    continue
                # Check the tissue mass:
                patch_tissue = row_tissue_mask[:, cl: cl + wsimager.w_]
                checked_tissue, tissue_perentage = wsimager.check_if_tissue_mask_has_enough_tissue(
                    patch_tissue, wsimager.tissue_mask_min)
                if not checked_tissue:
                    continue

                # Check if patch has enough cancerous pixels.
                patch_metas = row_metasta[:, cl: cl + wsimager.w_]
                checked_metast, percentage_cancer = wsimager.check_if_patch_has_enough_metastatic_pixels_first_pass(
                    patch_metas, wsimager.p0)
                if not checked_metast:
                    continue

                # Patch is OK. Store its info, and save it with its metastatic patch on disc.
                # The (x, y) cartesian coordinates of the patch within the WSI.
                x = upper_left[0] + cl
                y = upper_left[1] + r
                tag_patch = "file_{}_reg_{}_row_{}_patch_{}_x_{}_y_{}_tissue_{}_metastatic_{}_w_{}_h_{}".format(
                    join(*path_wsi.split(os.sep)[-3:]).replace(os.sep, "-"), cnt_reg, cnt_rows, cnt_patches, x, y,
                    tissue_perentage, percentage_cancer, wsimager.w_, wsimager.h_)
                list_patches.append([join(*path_wsi.split(os.sep)[-3:]), join(*path_xml.split(os.sep)[-3:]), x, y,
                                     tissue_perentage, percentage_cancer, tag_patch])

                # save on disc the patch and its mask.
                outd_patches = join(args.outd_patches, join(*path_wsi.split(os.sep)[-3:]).split(".")[0])
                if not os.path.exists(outd_patches):
                    os.makedirs(outd_patches, exist_ok=True)

                # crop using PIL.Image.

                patch = row.crop([cl, 0, cl + wsimager.w_, wsimager.h_])
                patch.save(join(outd_patches, "Patch_" + tag_patch + ".png"), "PNG")
                Image.fromarray(np.uint8(patch_metas) * 255).save(join(outd_patches, "mask_" + tag_patch + ".png"),
                                                                  "PNG")

                cnt_patches += 1
                del patch
                del patch_metas
                del patch_tissue

            cnt_rows += 1
            del row
            del row_arr
            del row_metasta
            del row_tissue_mask
        cnt_reg += 1

    # Save the patches info in *.csv file.
    announce_msg("Output *.csv file stored in: `{}`".format(csv_name))
    with open(csv_name, 'w') as fcsv:
        filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(
            ["wsi file name", "xml file name", "x", "y", "Tissue percentage", "Percentage of cancerous pixels",
             "Tag of the patch"])
        for elem in list_patches:
            filewriter.writerow(elem)

    return 0


def process_sample_metastatic_patches_many_wsi_pass_one(itr_imgs, itr_seeds):
    """
    Do iteratively the sampling by calling: sample_metastatic_patches_one_wsi_pass_one.
    :param itr_imgs: list, list of the arguments, each one for one image.
    :param itr_seeds: list, list of int, each is a seed for each image. (for reproducibility reasons).
    :return:
    """

    assert len(itr_imgs) == len(itr_seeds), "We expect the same number of seeds as the number of images. You provided " \
                                            "{} seeds while there is {} images .... [NOT OK]".format(
        len(itr_seeds), len(itr_imgs))

    for arg, seed in zip(itr_imgs, itr_seeds):
        # ================================
        # Reproducibility: RESET THE SEED.
        # ================================

        # ===========================

        reproducibility.set_seed_to_modules(seed)

        # ===========================

        sample_metastatic_patches_one_wsi_pass_one(arg)


# ============================================
#                 TEST
# ============================================


def test_normal():
    t0 = dt.datetime.now()
    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/camelyon16"
    elif username == "sbelharb":
        baseurl = "/home/sbelharb/workspace/datasets/camelyon16"

    wsi = join(baseurl, "training/normal/normal_001.tif")

    debug_outd = "./debug/wsi/" + wsi.split(os.sep)[-1].split(".")[0]
    if not os.path.exists(debug_outd):
        os.makedirs(debug_outd, exist_ok=True)

    # Size of the metastatic patches.

    h_ = 512
    w_ = 512

    outd_first = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-first-pass"))
    outd_second = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-second-pass"))

    args = {"level": 6,  # level of resolution in the WSI. 0 is the highest resolution [used just for debug]
            "level_approx": 6,  # resolution level where the search for the patches is done.
            "rgb_min": 10,  # minimal value of RGB color. Used in thresholding to estimate the tissue mask.
            "delta_w": 512,  # horizontal stride of the sliding window when sampling patches.
            "delta_h": 512,  # vertical stride of the sliding window when sampling patches.
            "h_": h_,  # height of the patch.
            "w_": w_,  # width of the patch.
            "tissue_mask_min": 0.1,  # minimal percentage of tissue mask in a patch to be accepted.
            "p0": 0.2,  # p0
            "p1": 0.5,  # p1
            "delta_w_inter": 1,  # horizontal stride of the sliding window when sampling patches. Used to approach
            # SLOWLY the border of the tumor. It is better to keep it to 1.
            "dichotomy_step": 100,  # NO LONGER USEFUL. # TODO: REMOVE IT.
            "path_wsi": wsi,  # path to the WSI.
            "path_xml": None,  # path to the xml file of the annotation of the WSI.
            "debug": False,  # USED FOR DEBUG. NO LONGER USEFUL. TODO: REMOVE IT.
            "outd_patches": join(baseurl, join("w-{}xh-{}".format(w_, h_), "normal-patches")),  # path where the
            # patches will be saved.
            "outd": debug_outd,  # path where the *csv files of the first pass will be saved.
            "fold": "./folds/camelyon16/WSI-level/split_0/fold_0",  # use a random split/fold.
            "n": 0.1,  # a percentage (of the patches with p0 <= p <= p1) used to compute the number of patches with p >
            # p1 that we should consider. This number is computed as: N * n, where N is the number of patches
            # with p0 <= p <= p1.
            "n_norm": 100,  # number of normal patches to sample.
            }

    wsimager = WSI_Factory(Dict2Obj(args))

    h, w = wsimager.get_height_width_level_0(openslide.OpenSlide(wsi))
    img_rgb = wsimager.read_wsi_2_rgb(wsi, wsimager.level)
    tissue_mask = wsimager.get_tissue_mask(img_rgb, wsimager.rgb_min)
    Image.fromarray(img_rgb).save(join(debug_outd, "img_rgb.png"), "PNG")
    Image.fromarray(np.uint8(tissue_mask) * 255).save(join(debug_outd, "tissue_mask.png"), "PNG")

    # ============================== TEST SAMPLING
    args["level"] = 0
    args["outd"] = join(args["outd"], join(*wsi.split(os.sep)[-3:-1]))
    sample_n_normal_patches_from_one_wsi(Dict2Obj(args))

    announce_msg("[TEST] Sampling normal patches took: {}".format(dt.datetime.now() - t0))


def test_metastatic():
    # ======================= TEST METASTATIC
    t0 = dt.datetime.now()
    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/camelyon16"
    elif username == "sbelharb":
        baseurl = "/home/sbelharb/workspace/datasets/camelyon16"

    wsi = join(baseurl, "testing/images/test_090.tif")
    xml_ann = join(baseurl, "testing/lesion_annotations/test_090.xml")

    debug_outd = "./debug/wsi/" + wsi.split(os.sep)[-1].split(".")[0]
    if not os.path.exists(debug_outd):
        os.makedirs(debug_outd, exist_ok=True)

    # Size of the metastatic patches.

    h_ = 512
    w_ = 512

    args = {"level": 6,
            "rgb_min": 10,
            "delta_w": 512,
            "delta_h": 512,
            "h_": h_,
            "w_": w_,
            "tissue_mask_min": 0.1,
            "p0": 0.2,
            "p1": 0.5,
            "delta_w_inter": 1,
            "dichotomy_step": 100,
            "path_wsi": wsi,
            "path_xml": xml_ann,
            "debug": False,
            "outd_patches": join(baseurl, join("w={}xh={}".format(w_, h_), "metastatic-patches")),
            "outd": join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-first-pass")),
            "fold": "./folds/camelyon16/WSI-level/split_0/fold_0"
            }

    wsimager = WSI_Factory(Dict2Obj(args))

    h, w = wsimager.get_height_width_level_0(openslide.OpenSlide(wsi))
    img_rgb = wsimager.read_wsi_2_rgb(wsi, wsimager.level)
    tissue_mask = wsimager.get_tissue_mask(img_rgb, wsimager.rgb_min)
    Image.fromarray(img_rgb).save(join(debug_outd, "img_rgb.png"), "PNG")
    Image.fromarray(np.uint8(tissue_mask) * 255).save(join(debug_outd, "tissue_mask.png"), "PNG")

    annotation = wsimager.get_annotation(xml_ann)
    # make serializable
    for key in annotation.keys():
        for i in range(len(annotation[key])):
            annotation[key][i]["vertices"] = annotation[key][i]["vertices"].tolist()
    with open(join(debug_outd, wsi.split(os.sep)[-1].split(".")[0] + ".json"), "w") as f:
        json.dump(annotation, f, indent=1)

    # ============================== TEST SAMPLING
    args["level"] = 0
    args["outd"] = join(args["outd"], join(*wsi.split(os.sep)[-3:-1]))
    sample_metastatic_patches_one_wsi_pass_one(Dict2Obj(args))

    announce_msg("[TEST] Sampling metastatic patches took: {}".format(dt.datetime.now() - t0))


def test():
    """
    Test functions.
    :return:
    """
    # Sample from metastatic WSI.
    # test_metastatic()

    # Sample from normal WSI.
    test_normal()

# ============================================
#                 RUN
# ============================================


def sample_all_metastatic_patches_first_pass_one_dataset(content, outd, nbr_workers, baseurl, args):
    """
    Sample all acceptable metastatic patches in the first pass (all patches with p >= p0).

    :param content: list, list of rows based on the content of *.csv files.
    :param outd: str, absolute path of the output directory.
    :param nbr_workers: int, number of processes.
    :param baseurl: str, absolute directory path to the CAMELYON16 dataset.
    :param args: dict, parameters of the sampler.
    :return:
    """
    list_in = []
    for row in content:
        assert row[1] == "tumor", "Something is wrong. We found a new class `{}`. We expect only the class `tumor` " \
                                  ".... [NOT OK]".format(row[1])
        if row[1] == "normal":
            continue
        c_args = copy.deepcopy(args)
        c_args["path_wsi"] = join(baseurl, row[0])
        c_args["path_xml"] = join(baseurl, row[2])
        c_args["outd"] = join(outd, join(*row[0].split(os.sep)[-3:-1]))

        if not os.path.isdir(c_args["outd"]):
            os.makedirs(c_args["outd"], exist_ok=True)

        list_in.append(Dict2Obj(c_args))

    # Multi-process the WSI.
    # Creates the input of each worker
    input_workers = list(chunks_into_n(list_in, nbr_workers))

    # Generate within the parent process a seed per image (for reproducibility).
    # There is really small change that the two images will have the same seed. Even if it is the case,
    # it is not really a big deal.

    seeds = np.random.randint(0, reproducibility.MAX_SEED + 1, len(list_in)).tolist()
    seed_per_img = list(chunks_into_n(seeds, nbr_workers))

    # Create the workers, start, and join them.
    processes = [
        Process(target=process_sample_metastatic_patches_many_wsi_pass_one,
                args=(input_workers[pp], seed_per_img[pp])) for pp in range(nbr_workers)
    ]
    [p.start() for p in processes]
    [p.join() for p in processes]


def do_first_pass_and_second_pass_metastatic(ind, outd_first, outd_second, nbr_workers, baseurl, args):
    """
    Sample all acceptable metastatic (pass 1) patches from all CAMELYON16. Then, equalize them (pass 2).

    :param ind: str, input directory where the WSI-level splits live.
    :param outd_first: str, output directory where to patch-level split of the first pass will live.
    :param outd_second: str, output directory where to patch-level split of the second pass will live.
    :param nbr_workers: int, number of processes.
    :param baseurl: str, absolute directory path to the CAMELYON16 dataset.
    :param args: dict, parameters of the sampler.
    :return:
    """
    # 1. First pass: Sample all the metastatic patches.
    announce_msg("First pass for the configuration w-{}xh-{}".format(args["w_"], args["h_"]))

    assert float(args["p0"]) > 0, "`p0` must be greater than 0. You provide `p0={}`".format(args["p0"])
    assert float(args["p1"]) <= 1, "`p1` must be <= 1. You provide `p1={}`".format(args["p1"])
    assert 0 < args["n"] <= 1, "`n` must be 0 < n <= 1. You provide `n={}`".format(args["n"])

    classes = ["normal", "tumor"]
    dir_f = args["fold"]
    tr_csv = join(dir_f, "train_s_0_f_0.csv")
    vl_csv = join(dir_f, "valid_s_0_f_0.csv")
    ts_csv = join(dir_f, "test_s_0_f_0.csv")

    assert os.path.isfile(tr_csv), "train file `{}` is missing .... [NOT OK]".format(tr_csv)
    assert os.path.isfile(vl_csv), "valid file `{}` is missing .... [NOT OK]".format(vl_csv)
    assert os.path.isfile(ts_csv), "test file `{}` is missing .... [NOT OK]".format(ts_csv)

    # merge all the csv files. Then, take only the WSI with tumor.
    content = []
    for csvfile in [tr_csv, vl_csv, ts_csv]:
        with open(csvfile, 'r') as fx:
            reader = csv.reader(fx)
            for row in reader:
                assert row[1] in classes, "Something is wrong. We found a new class `{}` that does not belong to the " \
                                          "predefined classes `{}`".format(row[1], classes)
                if row[1] == "normal":
                    continue

                content.append(row)

    print("There are `{}` WSI with tumor .... [OK]".format(len(content)))

    sample_all_metastatic_patches_first_pass_one_dataset(content, outd_first, nbr_workers, baseurl, args)

    # 2. Second pass: Calibrate the number of patches with respect to the percentage, p, of metastatic pixels
    # within the patch. We do not want many patches with high p (because the task of localization will loose its
    # meaning since a large portion of the patch is metastatic). The results of this pass will be
    # stored in the directory `outd_second`.
    # How to proceed?
    # A. All the patches with p0<= p <= p1 are accepted. Let us say that there are N patches that fall in this category.
    # We refer to this type of patches by `category A`.
    # B. Among all the patches with p > p1, we take only (N * n), where `n` is some predefined percentage. We refer
    # to this type of patches by `category B`.
    # The above steps (A, B) are performed on the ENTIRE metastatic patches of the current set.
    # This type of calibration is performed on each set: train and test separately. (since we do not know if both
    # sets have the same distribution of un-calibration between `category A` and `category B`).

    announce_msg("Second pass for the configuration w-{}xh-{}".format(args["w_"], args["h_"]))

    all_csv_test = glob.glob(join(args["outd"], "testing", "*", "*.csv"))
    all_csv_train = glob.glob(join(args["outd"], "training", "*", "*.csv"))

    def calibrate_set(list_csv, name):
        """
        Calibrate the distribution of patches using their percentage of metastatic pixels as a criterion.
        :param list_csv: list, list of paths to *.csvf files that contain the patches sampled from the first pass.
        :param name: str, name of the set.
        :return:
        """
        # Pass 1 over the *.csv files: Compute N (the total number of patches with p0 <= p <= p1).
        # *.csv files has a header as follows:
        # wsi file name,	xml file name,	x,	y,	Tissue percentage,	Percentage of cancerous pixels,	Tag of the patch
        N_A = 0  # The total number of patches with p0 <= p <= p1.
        N_B = 0  # The total number of patches with p > p1.

        p0 = float(args["p0"])
        p1 = float(args["p1"])

        list_ps = []

        for fcsv in tqdm.tqdm(list_csv, ncols=80, total=len(list_csv)):
            with open(fcsv, 'r') as fx:
                reader = csv.reader(fx)
                next(reader, None)  # Skip the header
                for row in reader:
                    p = float(row[5])
                    list_ps.append(p)
                    if p0 <= p <= p1:
                        N_A += 1
                    else:
                        N_B += 1

        msg = "Before calibration: The configuration w-{}xh-{} in {} contains a total number of patches of: " \
              "{} where {} ({:.2f}%) are in " \
              "{} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                args["w_"], args["h_"], name, N_A + N_B, N_A, (100 * N_A/float(N_A + N_B)), p0, p1, N_B,
                (100 * N_B/float(N_A + N_B)), p1)

        # Plot the histogram of the p.
        title = "Before calibrating {}: patch with (w, h) = ({}, {})." \
                "\n{:.2f}% where {} <= p <= {}; {:.2f}% where p > {}".format(
                    name, args["w_"], args["h_"], (100 * N_A/float(N_A + N_B)), p0, p1, (100 * N_B/float(N_A + N_B)),
                    p1)
        fig = draw_hist_probability_of_list(list_ps, 10, [0, 1], title)
        outd_fig = join(*list_csv[1].split(os.sep)[:-2])

        assert os.path.isdir(outd_fig), "Something wrong. The path `{}` is supposed to exist, but we couldn't find it." \
                                        " .... [NOT OK]".format(outd_fig)

        name_fig = "BEFORE-CALIBRATION-{}-patch-w-{}-h{}.png".format(name, args["w_"], args["h_"]).replace(" ", "-")

        fig.savefig(join(outd_fig, name_fig), dpi=100, bbox_inches='tight')

        # Write stats in text file
        with open(join(outd_fig, "stats-{}-w-{}-h-{}.txt".format(name, args["w_"], args["h_"])), "w") as ftxt:
            ftxt.write(name + ":\n" + msg)

        # Pass 2 over the *.csv files: Take only N * n of the patches in `category B`.
        list_ps = np.array(list_ps)

        # Among the patches with p > p1, take only (N * n) patches. To do that, we sample uniformly without
        # repetition from the bins of the histogram of the frequency of the percentages.

        # =============================================================
        # Sample UNIFORMLY (WITHOUT REPETITION) FROM BINS > P1 AND < 1.
        # =============================================================
        # Compute the number of of samples to take.
        n = int(N_A * args["n"])
        bins = np.arange(p1, 1, args["delta_bin"])
        selector = np.zeros((list_ps.size,), dtype=bool)

        # Check the worse case, where the number of samples with p > p1 are less than (n * samples with p <= p1).
        list_B = np.where(list_ps > p1)[0]

        if n > list_B.size:
            msgx = "You asked to take {} patches from patches with p > {}, but we have only {}. We recommend " \
                   "you to use a smalled `n` in the provided arguments. Exiting .... [NOT OK]".format(
                    n, p1, list_B.size)
            warnings.warn(msgx)
            sys.exit(1)

        for sx in range(n):
            # Find me the next sample.
            found_it = False
            while not found_it:
                # 1. Randomly select the bin.
                l_bound = bins[np.random.randint(0, bins.size, 1)[0]]
                upper_bound = min(l_bound + args["delta_bin"], 1.)
                list_ind = np.where(np.logical_and(l_bound <= list_ps, list_ps <= upper_bound))[0]
                # Check if there is any sample that verifies the condition
                if list_ind.size == 0:
                    continue

                # 2. Randomly select a sample within the bin. Try many times if the sample has already bin selected.
                for binner in range(args["max_trials_inside_bin"]):
                    # Sample UNIFORMLY FROM THE FOUND SAMPLES!!!!!!
                    position = np.random.choice(list_ind, 1)

                    # Check if we already have selected this sample
                    if selector[position]:
                        continue
                    else:
                        selector[position] = True
                        found_it = True

                # Now that you are out of the loop, tow possible options:
                # 1. Either you found it. Cool, you get out from the infinite loop.
                # 2. You did not find it, in this case, give up on that poor bin, and start all over by selecting a new
                # random bin. There is a risque to fall into an infinite loop if the number of samples with p > p1 is
                # less than the number of samples p0 <= p. But, this situation is impractical. We exit the program if
                # it is the case.

        # ============================================
        #                  END
        # ============================================

        new_list_ps = []
        cnt = 0
        N_B_new = 0
        for fcsv in tqdm.tqdm(list_csv, ncols=80, total=len(list_csv)):
            with open(fcsv, 'r') as fx:
                # Create a new *.csv file that contains only the metastatic patches that we will use.
                second_outd = join(outd_second, join(*fcsv.split(os.sep)[-3:-1]))
                # create the directory
                if not os.path.exists(second_outd):
                    os.makedirs(second_outd)
                new_csv_path = join(second_outd, fcsv.split(os.sep)[-1])
                with open(new_csv_path, 'w') as fsecond:
                    reader = csv.reader(fx)
                    filewriter = csv.writer(fsecond, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    header = next(reader)
                    filewriter.writerow(header)
                    for row in reader:
                        p = float(row[5])
                        if (p0 <= p <= p1) or selector[cnt]:
                            new_list_ps.append(p)
                            filewriter.writerow(row)
                        if selector[cnt]:
                            N_B_new += 1

                        cnt += 1

        msg = "After calibration: The configuration w-{}xh-{} in {} contains a total number of patches of: {} where " \
              "{} ({:.2f}%) are in {} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                 args["w_"], args["h_"], name, N_A + N_B_new, N_A, (100 * N_A / float(N_A + N_B_new)), p0, p1, N_B_new,
                 (100 * N_B_new / float(N_A + N_B_new)), p1)
        # Write stats in text file
        with open(join(outd_fig, "stats-{}-w-{}-h-{}.txt".format(name, args["w_"], args["h_"])), "a") as ftxt:
            ftxt.write("\n" + msg)
        # Plot the histogram of the p.
        title = "After calibrating {}: patch with (w, h) = ({}, {})." \
                "\n {:.2f}% where {} <= p <= {}; {:.2f}% where p > {}".format(
                    name, args["w_"], args["h_"], (100 * N_A / float(N_A + N_B_new)), p0, p1,
                    (100 * N_B_new / float(N_A + N_B_new)), p1)
        fig = draw_hist_probability_of_list(new_list_ps, 10, [0, 1], title)

        name_fig = "AFTER-CALIBRATION-{}-patch-w-{}-h{}.png".format(
            name, args["w_"], args["h_"]).replace(" ", "-")

        fig.savefig(join(outd_fig, name_fig), dpi=100, bbox_inches='tight')

        # Print stats.
        msg_before = "\t Before calibration: The configuration w-{}xh-{} contains a total number of patches of: " \
                     "{} where {} ({:.2f}%) are in " \
                     "{} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                        args["w_"], args["h_"], N_A + N_B, N_A, (100 * N_A / float(N_A + N_B)), p0, p1, N_B,
                        (100 * N_B / float(N_A + N_B)), p1)
        msg_after = "\t After calibration: The configuration w-{}xh-{} contains a total number of patches of: " \
                    "{} where {} ({:.2f}%) are in " \
                    "{} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                        args["w_"], args["h_"], N_A + N_B_new, N_A, (100 * N_A / float(N_A + N_B_new)), p0, p1, N_B_new,
                        (100 * N_B_new / float(N_A + N_B_new)), p1)

        announce_msg(name + "\n" + msg_before + "\n" + msg_after)

        return {"N_A": N_A,
                "N_B": N_B,
                "N_B_new": N_B_new}

    # Calibrate train set
    stat_train = calibrate_set(all_csv_train, "train-set")

    # Calibrate test set
    stat_test = calibrate_set(all_csv_test, "test-set")

    # Compute total stats
    N_A = stat_train["N_A"] + stat_test["N_A"]
    N_B = stat_train["N_B"] + stat_test["N_B"]
    N_B_new = stat_train["N_B_new"] + stat_test["N_B_new"]
    p0 = float(args["p0"])
    p1 = float(args["p1"])
    # Print stats.
    msg_before = "\t Before calibration: The configuration w-{}xh-{} contains a total number of patches of: " \
                 "{} where {} ({:.2f}%) are in " \
                 "{} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                    args["w_"], args["h_"], N_A + N_B, N_A, (100 * N_A / float(N_A + N_B)), p0, p1, N_B,
                    (100 * N_B / float(N_A + N_B)), p1)
    msg_after = "\t After calibration: The configuration w-{}xh-{} contains a total number of patches of: " \
                "{} where {} ({:.2f}%) are in " \
                "{} <= p <= {}, and {} ({:.2f}%) are in p > {}".format(
                    args["w_"], args["h_"], N_A + N_B_new, N_A, (100 * N_A / float(N_A + N_B_new)), p0, p1, N_B_new,
                    (100 * N_B_new / float(N_A + N_B_new)), p1)

    msg = "Total data:" + "\n" + msg_before + "\n" + msg_after
    announce_msg(msg)
    with open(join(args["outd"], "stats-{}-w-{}-h-{}.txt".format("train-and-test", args["w_"], args["h_"])),
              "w") as ftxt:
        ftxt.write( msg)


def sample_n_normal_patches_from_one_wsi(args):
    """
    Sample RANDOMLY `n` normal patches from a normal WSI.
    The sampling is performed in `level 0`, however, the search is done in another level where the binarization
    allows to separate the tissue from the background. Example: level 6.

    Note: Sampling is based on randomness. If `n` is large, it may take a while to finish. Sampling is done without
    repetition.

    :param args: object, contains arguments for sampling.
    :return: leftover: int, how many patches left to sample. Sometimes, a WSI does not have enough tissue,
    and it is not possible to sample the `n` required patches. Therefore, we pass the leftover to the next WSI to
    sample from it (if possible).
    """

    path_wsi = args.path_wsi

    assert os.path.isfile(path_wsi), "File `{}` does not exist ... [NOT OK]".format(path_wsi)

    announce_msg("Going to process file: `{}`".format(path_wsi))

    # Check if the file has not already been processed:
    outd = args.outd_third
    if not os.path.exists(outd):
        os.makedirs(outd, exist_ok=True)

    csv_name = join(outd, path_wsi.split(os.sep)[-1].split(".")[0] + ".csv")

    if os.path.isfile(csv_name):
        announce_msg(
            "File `{}` has already been processed. Output in`{}` already exists .... [OK]".format(path_wsi, csv_name)
        )
        return 0

    announce_msg("Started process file: `{}`".format(path_wsi))

    wsimager = WSI_Factory(args)

    slide = openslide.OpenSlide(path_wsi)

    # 1. Get h, w of WSI at level 0
    him, wim = wsimager.get_height_width_level_0(slide)
    him_l, wim_l = wsimager.get_height_width_level_l(slide, args.level_approx)
    ratio_him_him_l = him / float(him_l)
    ratio_wim_wim_l = wim / float(wim_l)
    w_l = int(wsimager.w_ / ratio_wim_wim_l)
    h_l = int(wsimager.h_ / ratio_him_him_l)

    img_level_approx = slide.read_region(
        (0, 0), args.level_approx, slide.level_dimensions[args.level_approx]).convert("RGB")
    img_level_approx_arr = np.array(img_level_approx)
    # We need to binarize the entire image at this level.
    tissue_mask_vele_approx = wsimager.get_tissue_mask(img_level_approx_arr, wsimager.tissue_mask_min)

    list_patches = []
    announce_msg("Processing normal WSI ... [OK]")
    list_r_c = []

    failed = False

    for i in tqdm.tqdm(range(args.n_norm), ncols=80, total=args.n_norm):
        # Try to sample a normal patch.
        found_it = False
        trials = 0
        while not found_it and trials <= args.max_trials_sample_normal_patches_inside_wsi:
            trials += 1
            # sample position
            r, c = wsimager.random_indexer_grid_wsi_patch_level(him_l, wim_l, h_l, w_l,
                                                                int(wsimager.delta_h / ratio_him_him_l),
                                                                int(wsimager.delta_w / ratio_wim_wim_l))
            # check if we have already sample this position:
            if (r, c) in list_r_c:
                continue
            else:
                list_r_c.append((r, c))

            # It seems not necessary to check if the patch itself has a lot of white to reject it.

            # Check the tissue mass:
            patch_tissue_mask = tissue_mask_vele_approx[r:r + h_l, c:c + w_l]
            checked_tissue, tissue_perentage = wsimager.check_if_tissue_mask_has_enough_tissue(
                patch_tissue_mask, wsimager.tissue_mask_min)
            if not checked_tissue:
                continue

            # Patch at level approximate is OK. Now, double-check the patch at level 0.
            # Map the cooridinates from level-approx to level 0.
            r = int(r * ratio_him_him_l)
            c = int(c * ratio_wim_wim_l)

            if ((r + wsimager.h_) > him) or ((c + wsimager.w_) > wim):
                continue

            patch = slide.read_region((c, r), 0, (wsimager.w_, wsimager.h_)).convert("RGB")
            patch_arr = np.array(patch)

            # Check if the patch row is thesholdable:
            if not wsimager.check_if_img_thresholdable(patch_arr):
                continue

            # Check the tissue mass:
            patch_tissue_mask = wsimager.get_tissue_mask(patch_arr, wsimager.tissue_mask_min)
            checked_tissue, tissue_perentage = wsimager.check_if_tissue_mask_has_enough_tissue(
                patch_tissue_mask, wsimager.tissue_mask_min)
            if not checked_tissue:
                continue

            # Patch is OK. Take it.
            found_it = True

            # Save the patch
            tag_patch = "file_{}_patch_{}_x_{}_y_{}_tissue_{}_w_{}_h_{}".format(
                join(*path_wsi.split(os.sep)[-3:]).replace(os.sep, "-"), i, c, r,
                tissue_perentage, wsimager.w_, wsimager.h_)
            list_patches.append([join(*path_wsi.split(os.sep)[-3:]), "", c, r, tissue_perentage, 0, tag_patch])

            # save on disc the patch and its mask.
            # outd_patches = join(args.outd_patches, join(*path_wsi.split(os.sep)[-3:]).split(".")[0])
            if not os.path.exists(args.outd_patches):
                os.makedirs(args.outd_patches, exist_ok=True)

            # crop using PIL.Image.
            patch.save(join(args.outd_patches, "Patch_" + tag_patch + ".png"), "PNG")

            del patch
            del patch_arr
            del patch_tissue_mask

        if trials > args.max_trials_sample_normal_patches_inside_wsi:
            failed = True
            print("We failed to find a patch in {} .... [NOT OK]".format(path_wsi))

    if failed:
        print("We are supposed to sample {} normal patches from {} but we were able to sample only {} ... The WSI "
              "seems contain only small region with tissues. Sorry about that .... [NOT OK]".format(
               args.n_norm, path_wsi, len(list_patches)))
    # Save the patches info in *.csv file.
    announce_msg("Output *.csv file stored in: `{}`".format(csv_name))
    with open(csv_name, 'w') as fcsv:
        filewriter = csv.writer(fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(
            ["wsi file name", "xml file name", "x", "y", "Tissue percentage", "Percentage of cancerous pixels",
             "Tag of the patch"])
        for elem in list_patches:
            filewriter.writerow(elem)

    leftover = args.n_norm - len(list_patches)
    return leftover


def process_sample_normal_patches_many_wsi(itr_imgs, itr_seeds):
    """
    Do iteratively the sampling by calling: sample_n_normal_patches_from_one_wsi.

    :param itr_imgs: list, list of the arguments, each one for one image.
    :param itr_seeds: list, list of int, each is a seed for each image. (for reproducibility reasons).
    :return:
    """

    assert len(itr_imgs) == len(itr_seeds), "We expect the same number of seeds as the number of images. You provided " \
                                            "{} seeds while there is {} images .... [NOT OK]".format(
        len(itr_seeds), len(itr_imgs))

    leftover = 0
    for arg, seed in zip(itr_imgs, itr_seeds):
        # ================================
        # Reproducibility: RESET THE SEED.
        # ================================

        # ===========================

        reproducibility.set_seed_to_modules(seed)

        # ===========================
        arg.n_norm += leftover
        leftover = sample_n_normal_patches_from_one_wsi(arg)
    if leftover != 0:
        warnings.warn("Some of the WSI were small, and we were not able to sample all the required normal patches. "
                      "There are {} patches left to sample .... [NOT OK]".format(leftover))


def sample_one_set_third_pass(args, split_i, fold_i, csvfile, name_set, nbr_workers):
    """
    Sample a set.

    :param args: dict, contain the sampling configuration.
    :param split_i: int, number of the split.
    :param fold_i: int, number of the fold.
    :param csvfile: str, path to the *.csv file.
    :param name_set: str, name of the set (used to name the output folder).
    :param nbr_workers: int, number of workers for multiprocessing.
    :return:
    """
    classes = ["normal", "tumor"]

    # separate normal from tumor WSIs.
    list_tumor = []
    list_normal = []

    with open(csvfile, 'r') as fx:
        reader = csv.reader(fx)
        for row in reader:
            assert row[1] in classes, "Something is wrong. We found a new class `{}` that does not belong to the " \
                                      "predefined classes `{}`".format(row[1], classes)

            if row[1] == "normal":
                list_normal.append(row[0])
            elif row[1] == "tumor":
                list_tumor.append(row[0])
            else:
                raise ValueError("Class `{}` is unknown. Expected `{}` .... [NOT OK]".format(row[0], classes))

    outd_csv = join(args["outd_third"], "split_{}".format(split_i), "fold_{}".format(fold_i), name_set)
    if not os.path.exists(outd_csv):
        os.makedirs(outd_csv, exist_ok=True)

    # Nothing to do for the metastatic patches. Just copy them from the second pass to the third pass.
    nbr_metastatic_patches = 0  # count the number of metastatic patches.
    for f in list_tumor:
        name_f = f.split(os.sep)[-1].split(".")[0]
        in_path = join(args["outd_second"], f.replace(".tif", ".csv"))
        output_path = join(outd_csv, name_f + ".csv")

#        assert not os.path.isfile(output_path), "The file `{}` already exists .... [NOT OK]".format(output_path)
        shutil.copy(in_path, output_path)
        with open(output_path, 'r') as fx:
            reader = csv.reader(fx)
            next(reader, None)
            for _ in reader:
                nbr_metastatic_patches += 1

    announce_msg("We found {} metastatic patch in split-{}-fold-{}-set-{} .... [OK]".format(
        nbr_metastatic_patches, split_i, fold_i, name_set))

    # Now, sample randomly `nbr_metastatic_patches` normal patch.
    n_per_wsi = nbr_metastatic_patches // len(list_normal)
    left_over = nbr_metastatic_patches % len(list_normal)

    args_sampling = []
    for wsi in list_normal:
        tmp_args = copy.deepcopy(args)
        if left_over != 0:
            tmp_args["n_norm"] = n_per_wsi + left_over
            left_over = 0
        else:
            tmp_args["n_norm"] = n_per_wsi

        tmp_args["outd_patches"] = join(
            tmp_args["outd_patches"], "split_{}".format(split_i), "fold_{}".format(fold_i), name_set)
        tmp_args["path_wsi"] = join(tmp_args["baseurl"], wsi)
        tmp_args["outd_third"] = outd_csv

        args_sampling.append(Dict2Obj(tmp_args))

    # Now, sample normal patches using multiprocessing.
    # multi-process the WSI.
    # creates the input of each worker
    input_workers = list(chunks_into_n(args_sampling, nbr_workers))

    # Generate within the parent process a seed per image (for reproducibility).
    # There is really small change that the two images will have the same seed. Even if it is the case,
    # it is not really a big deal. There is also very small chance that the same image with two folds will have the
    # same seed.

    seeds = np.random.randint(0, reproducibility.MAX_SEED + 1, len(args_sampling)).tolist()
    seed_per_img = list(chunks_into_n(seeds, nbr_workers))

    # Create the workers, start, and join them.
    processes = [
        Process(target=process_sample_normal_patches_many_wsi,
                args=(input_workers[pp], seed_per_img[pp])) for pp in range(nbr_workers)
    ]
    [p.start() for p in processes]
    [p.join() for p in processes]

    # Now, prepare the splits for the experimenter: They do not need to know anything about all this. All what they
    # need is a list of patches and their labels.
    # Format of *.csv final: path to the patch, path to the mask, class (normal, tumor).

    # Normal output patches folder.
    outd_patches = join(args["outd_patches"], "split_{}".format(split_i), "fold_{}".format(fold_i), name_set)
    name_dataset_fd = args["baseurl"].split(os.sep)[-1]
    normal_parent_folder = join(name_dataset_fd, os.path.relpath(outd_patches, args["baseurl"]))

    list_all_csv = glob.glob(join(outd_csv, "*.csv"))
    outd_fourth_pass = join(args["outd_fourth"], "split_{}".format(split_i), "fold_{}".format(fold_i))

    if not os.path.exists(outd_fourth_pass):
        os.makedirs(outd_fourth_pass, exist_ok=True)

    path_csv_final = join(outd_fourth_pass, "{}_s_{}_f_{}.csv".format(name_set, split_i, fold_i))
    with open(path_csv_final, 'w') as fcsvfinale:
        writer = csv.writer(fcsvfinale, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for fcsv in list_all_csv:
            with open(fcsv, 'r') as fin:
                reader = csv.reader(fin)
                next(reader, None)
                for row in reader:
                    # check the class of the patch by looking to the percentage of metastatic pixels.
                    percentage_metast = float(row[5])
                    path_patch = ""
                    path_mask = ""
                    cl = None
                    if percentage_metast == 0:  # normal patch
                        cl = "normal"
                        path_patch = join(normal_parent_folder, "Patch_{}.png".format(row[6]))
                    else:  # tumor patch
                        cl = "tumor"
                        folder_name = row[0].split(".")[0]
                        metastatic_parent_folder = join(name_dataset_fd,
                                                        join(args["relative_path_metastatic_patches"],
                                                             folder_name))
                        path_mask = join(metastatic_parent_folder, "mask_{}.png".format(row[6]))
                        path_patch = join(metastatic_parent_folder, "Patch_{}.png".format(row[6]))

                    assert cl is not None, "Something wrong. We expect the patch class to be in [normal, " \
                                           "tumor]. Something wrong with the percentage .... [NOT OK]"
                    writer.writerow([path_patch, path_mask, cl])

    return path_csv_final


def do_splits_folds(args, nbr_workers):
    """
    Sample for all the splits/folds normal and metastatic patches.

    Note: Metastatic patches are already sampled and calibrated.

    :param args: dict, sampling parameters.
    :param nbr_workers: int, number of workers.
    :return:
    """
    splits_dir = args["splits_dir"]
    list_splits_fd = [join(splits_dir, x) for x in os.listdir(splits_dir) if x.startswith("split_")]

    # Test is done only once, then reused.
    test_is_done = False

    for sp in list_splits_fd:
        list_folds = [join(sp, x) for x in os.listdir(sp) if x.startswith("fold_")]
        sp_i = sp.split("_")[-1]
        for fold in list_folds:
            fd_i = fold.split("_")[-1]

            # train:
            csv_train = join(fold, "train_s_{}_f_{}.csv".format(sp_i, fd_i))

            assert os.path.isfile(csv_train), "The file {} does not exist .... [NOT OK]".format(csv_train)

            sample_one_set_third_pass(copy.deepcopy(args), sp_i, fd_i, csv_train, "train", nbr_workers)

            # valid:
            csv_valid = join(fold, "valid_s_{}_f_{}.csv".format(sp_i, fd_i))

            assert os.path.isfile(csv_valid), "The file {} does not exist .... [NOT OK]".format(csv_valid)

            sample_one_set_third_pass(copy.deepcopy(args), sp_i, fd_i, csv_valid, "valid", nbr_workers)

            # test
            if not test_is_done:  # if it is not done: Do it.
                csv_test = join(fold, "test_s_{}_f_{}.csv".format(sp_i, fd_i))

                assert os.path.isfile(csv_test), "The file {} does not exist .... [NOT OK]".format(csv_test)

                path_csv_final_test = sample_one_set_third_pass(
                    copy.deepcopy(args), sp_i, fd_i, csv_test, "test", nbr_workers)

                test_is_done = True
            else:  # if it is done, then copy the csv files fro where it was done, to the current split/fold.
                assert os.path.isfile(path_csv_final_test), "Test file `{}` does not exist. What happened .... [NOT " \
                                                            "OK]".format(path_csv_final_test)

                outd_fourth_pass = join(args["outd_fourth"], "split_{}".format(sp_i), "fold_{}".format(fd_i))
                if not os.path.exists(outd_fourth_pass):
                    os.makedirs(outd_fourth_pass, exist_ok=True)
                current_path_csv = join(outd_fourth_pass, "{}_s_{}_f_{}.csv".format("test", sp_i, fd_i))
                if path_csv_final_test != current_path_csv:
                    shutil.copyfile(path_csv_final_test, current_path_csv)


def sample_all_metastatic_patches_and_calibrate_them(ind, outd_first, outd_second, nbr_workers, baseurl, args):
    """
    Sample metastatic patches from from CAMELYON16. (performed only once).

    :param ind: str, input directory path where the WSI-level splits live.
    :param outd_first: str, output directory path where to patch-level split of the first pass will live.
    :param outd_second: str, output directory path where to patch-level split of the second pass will live.
    :param nbr_workers: int, number of processes.
    :param baseurl: str, absolute directory path to the CAMELYON16 dataset.
    :param args: dict, contains the parameters of the sampling.
    :return:
    """
    # Sample and equalize metastatic patches.
    do_first_pass_and_second_pass_metastatic(ind, outd_first, outd_second, nbr_workers, baseurl, args)


def run():
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    reproducibility.set_seed()

    # ===========================

    ind = "./folds/camelyon16/WSI-level"
    nbr_workers = int(sys.argv[1])

    username = getpass.getuser()
    if username == "brian":
        baseurl = "/media/brian/Seagate Backup Plus Drive/datasets/camelyon16"
    elif username == "sbelharb":
        baseurl = "/home/sbelharb/workspace/datasets/camelyon16"
    else:
        raise ValueError("Your username seems to be unknown, huh! you must be trying to run this code in a host that "
                         "we do not know about it. You can either add your username and the corresponding path to the "
                         "CAMELYON16 dataset, or, delete this section that checks for the username and set your own "
                         "path.")

    def do_one_size(current_w, current_h):
        """
        Do all the sampling: Pass 1, 2, 3, 4 for a specific size.

        :param current_w: int, width of the pacth.
        :param current_h: int, height of the patch.
        :return:
        """
        # ===============
        # Reproducibility
        # ===============

        # ===========================

        reproducibility.set_seed()

        # ===========================

        t0 = dt.datetime.now()
        # Size of the patches.

        h_ = current_h
        w_ = current_w
        delta_w = w_  # no overlap.
        delta_h = h_  # no overlap.

        outd_first = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-first-pass"))
        outd_second = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-second-pass"))
        outd_third = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-third-pass"))
        outd_fourth = join("./folds/camelyon16/", join("w-{}xh-{}".format(w_, h_), "patches-level-fourth-pass"))

        args = {"level": 0,  # level of resolution in the WSI. 0 is the highest resolution [Juts for debug]
                "level_approx": 6,  # resolution level where the search for the patches is done.
                "rgb_min": 10,  # minimal value of RGB color. Used in thresholding to estimate the tissue mask.
                "delta_w": delta_w,  # horizontal stride of the sliding window when sampling patches.
                "delta_h": delta_h,  # vertical stride of the sliding window when sampling patches.
                "h_": h_,  # height of the patch.
                "w_": w_,  # width of the patch.
                "tissue_mask_min": 0.1,  # minimal percentage of tissue mask in a patch to be accepted.
                "p0": 0.2,  # p0
                "p1": 0.5,  # p1
                "delta_w_inter": 1,  # horizontal stride of the sliding window when sampling patches. Used to approach
                # SLOWLY the border of the tumor. It is better to keep it to 1.
                "dichotomy_step": 100,  # NO LONGER USEFUL. # TODO: REMOVE IT.
                "path_wsi": None,  # path to the WSI.
                "path_xml": None,  # path to the xml file of the annotation of the WSI.
                "debug": False,  # USED FOR DEBUG. NO LONGER USEFUL. TODO: REMOVE IT.
                "outd_patches": join(baseurl, join("w-{}xh-{}".format(w_, h_), "metastatic-patches")),  # path where the
                # patches will be saved.
                "outd": outd_first,  # path where the *.csv files of the first pass will be saved.
                "outd_second": outd_second,  # path to directory where *.csv files of the second pass will be stored.
                "outd_third": outd_third,  # path to directory where *.csv files of the third pass will be stored.
                "outd_fourth": outd_fourth,  # path to directory where *.csv files of the fourth (final) pass are
                # stored.
                "fold": "./folds/camelyon16/WSI-level/split_0/fold_0",  # use a random split/fold.
                "n": 0.01,  # a percentage (of the patches with p0 <= p <= p1) used to compute the number of patches
                # with p > p1 that we should consider. This number is computed as: N * n, where N is the number of
                # patches with p0 <= p <= p1.
                "n_norm": None,  # number of normal patches to sample.
                "splits_dir": "folds/camelyon16/WSI-level",  # directory where the splits are.
                "baseurl": baseurl,  # the base path to CAMELYON16 dataset.
                "delta_bin": 0.05,  # delta used to create bins for sampling (calibrate metastatic patches).
                "max_trials_inside_bin": 100,  # maximum number of times to try sampling a sample from a bin if the
                # sampling fails because we sampled all the samples inside that bin.
                "max_trials_sample_normal_patches_inside_wsi": 1000,  # maximum number to try to sample a normal patch
                # within a WSI before giving up and moving to the next patch. Some WSI contain very small region of
                # tissue, and it makes it difficult to sample in the case of non-overlapping patches.
                }

        sample_all_metastatic_patches_and_calibrate_them(ind, outd_first, outd_second, nbr_workers, baseurl, args)

        # Do the splits/folds
        # absolute path where the normal patches will be saved.
        args["outd_patches"] = join(baseurl, join("w-{}xh-{}".format(w_, h_), "normal-patches"))
        # relative path where the metastatic patches have been saved.
        args["relative_path_metastatic_patches"] = join("w-{}xh-{}".format(w_, h_), "metastatic-patches")
        do_splits_folds(args, nbr_workers)

        announce_msg("Running of case (w={}, h={}) took: {}".format(w_, h_, dt.datetime.now() - t0))

    patches_sizes = [[512, 512], [768, 768], [1024, 1024]]

    for c_w, c_h in patches_sizes:
        do_one_size(c_w, c_h)


if __name__ == "__main__":
    # ===============
    # Reproducibility*
    # ===============

    reproducibility.set_seed()
    # test()
    run()
