"""
Customized transforms.

Reference: https://github.com/NVIDIA/semantic-segmentation/blob/
5cdce2c7b349b4ae740d363eb7d934a4473dbc04/transforms/transforms.py
"""

"""
Standard Transform
"""

import random


import numpy as np

from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral

from PIL import Image, ImageEnhance

import torch
import torchvision.transforms as torch_tr

import torchvision.transforms.functional as TF

PROB_THRESHOLD = 0.5  # probability threshold.

try:
    import accimage
except ImportError:
    accimage = None


__all__ = [
    "Compose",
    "RandomGaussianBlur",
    "RandomBilateralBlur",
    "ColorJitter",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomAffine"
]


class _BasicTransform(object):
    """
    Defines our basic transform.
    It accept two inputs when it is called:
    - image.
    - mask: the corresponding mask (depends on the transformation).
    """
    def __call__(self, img, mask1=None, mask2=None):
        """

        :param img: PIL.Image.Image. the image.
        :param mask1: PIL.Image.Image the corresponding mask. Can be None
        depending on the transformation.
        :param mask2: PIL.Image.Image the corresponding mask. Can be None
        depending on the transformation. Sometimes, we need 2 masks.
        :return:
        """
        raise NotImplementedError


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to
        compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask1=None, mask2=None):
        for t in self.transforms:
            img, mask1, mask2 = t(img, mask1, mask2)
        return img, mask1, mask2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposeSingleInput(Compose):
    """
    Compose class but accepts only one input (image).
    """
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomGaussianBlur(_BasicTransform):
    """
    Apply Gaussian Blur
    """
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, mask1=None, mask2=None):
        if random.random() < self.p:
            return img, mask1, mask2

        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8)), mask1, mask2


class RandomBilateralBlur(_BasicTransform):
    """
    Apply Bilateral Filtering

    """

    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, mask1=None, mask2=None):
        if random.random() < self.p:
            return img, mask1, mask2

        sigma = random.uniform(0.05, 0.75)
        blurred_img = denoise_bilateral(np.array(img),
                                        sigma_spatial=sigma,
                                        multichannel=True
                                        )
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8)), mask1, mask2


class RandomBrightness(_BasicTransform):

    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, mask1=None, mask2=None):
        if random.random() < self.p:
            return img, mask1, mask2

        v = random.uniform(0.1, 1.9)
        return ImageEnhance.Brightness(img).enhance(v), mask1, mask2


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def random_adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if random.random() < PROB_THRESHOLD:
        return img

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def random_adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if random.random() < PROB_THRESHOLD:
        return img

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def random_adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if random.random() < PROB_THRESHOLD:
        return img

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def random_adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if random.random() < PROB_THRESHOLD:
        return img

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


class ColorJitter(_BasicTransform):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness),
                                                  1 + brightness)
            transforms.append(
                torch_tr.Lambda(
                    lambda img: random_adjust_brightness(img,
                                                         brightness_factor)
                )
            )

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast),
                                                1 + contrast)
            transforms.append(
                torch_tr.Lambda(
                    lambda img: random_adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation),
                                                  1 + saturation)
            transforms.append(
                torch_tr.Lambda(
                    lambda img: random_adjust_saturation(img,
                                                         saturation_factor)
                )
            )

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                torch_tr.Lambda(lambda img: random_adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        # Use Compose that operates only on the image.
        transform = ComposeSingleInput(transforms)

        return transform

    def __call__(self, img, mask1=None, mask2=None):
        """
        Args:
            img (PIL Image): Input image.
            mask1 (PIL image): the corresponding mask or None.
            mask2 (PIL image): the corresponding mask or None.

        Returns:
            PIL Image: Color jittered image.
        """
        if random.random() < PROB_THRESHOLD:
            return img, mask1, mask2

        transform = self.get_params(self.brightness,
                                    self.contrast,
                                    self.saturation,
                                    self.hue
                                    )
        return transform(img), mask1, mask2


class RandomHorizontalFlip(_BasicTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, mask1=None, mask2=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            mask1 (PIL Image): Corresponding mask (not mandatory).
            mask2 (PIL image): the corresponding mask or None.

        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped mask (same flip) if the mask exists.
            otherwise None.
            PIL Image: Randomly flipped mask (same flip) if the mask exists.
            otherwise None.

        """
        if random.random() < self.p:
            outmask1 = mask1 if mask1 is None else TF.hflip(mask1)
            outmask2 = mask2 if mask2 is None else TF.hflip(mask2)
            return TF.hflip(img), outmask1, outmask2

        return img, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(_BasicTransform):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, mask1=None, mask2=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            mask1 (PIL Image): Corresponding mask (not mandatory).
            mask2 (PIL Image): Corresponding mask (not mandatory).


        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped mask (same flip) if the mask exists.
            otherwise None.
            PIL Image: Randomly flipped mask (same flip) if the mask exists.
            otherwise None.
        """
        if random.random() < self.p:
            outmask1 = mask1 if mask1 is None else TF.vflip(mask1)
            outmask2 = mask2 if mask2 is None else TF.vflip(mask2)
            return TF.vflip(img), outmask1, outmask2

        return img, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAffine(_BasicTransform):
    """
    Affine transformation.
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None,
                 resample=False, fillcolor=0):
        raise NotImplementedError

    def __call__(self, img, mask1=None, mask2=None):
        raise NotImplementedError