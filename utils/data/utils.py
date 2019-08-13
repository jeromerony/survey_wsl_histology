import torch

from PIL import Image
from random import choice, sample, shuffle
from torch.utils.data.sampler import Sampler
from torchvision.transforms import functional as F


class RandomDiscreteRotation:
    def __init__(self, degrees: list):
        self.degrees = degrees

    def __call__(self, img):
        angle = choice(self.degrees)

        return img.rotate(angle)

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={})'.format(self.degrees)


class ExpandedRandomSampler(Sampler):
    """Iterate multiple times over the same dataset instead of once.
    Args:
        length (int): initial length of the dataset to sample from
        multiplier (float): desired multiplier for the length of the dataset
    """

    def __init__(self, length, multiplier):
        self.length = length
        self.indices = [i for i in range(length)]
        self.total = round(self.length * multiplier)

    def __iter__(self):
        return (self.indices[i % self.length] for i in torch.randperm(self.total))

    def __len__(self):
        return self.total


def load_data(samples, resize=None, min_resize=None):
    images = {}
    for image_path in samples:
        image = Image.open(image_path)
        if resize is not None:
            image = image.resize(resize, resample=Image.LANCZOS)
        elif min_resize:
            image = F.resize(image, min_resize, interpolation=Image.LANCZOS)
        images[image_path] = image.copy()
    return images


def check_overlap(train, valid, test):
    train_set = set([s[0] for s in train])
    valid_set = set([s[0] for s in valid])
    test_set = set([s[0] for s in test])

    train_valid = train_set.intersection(valid_set)
    train_test = train_set.intersection(test_set)
    valid_test = valid_set.intersection(test_set)
    if bool(train_valid):
        raise ValueError('Train and valid sets are overlapping: {}'.format(train_valid))
    if bool(train_test):
        raise ValueError('Train and test sets are overlapping: {}'.format(train_test))
    if bool(valid_test):
        raise ValueError('Valid and test sets are overlapping: {}'.format(valid_test))


def fraction_dataset(samples, num_classes, fraction):
    selected_samples = []

    for i in range(num_classes):
        class_samples = [s for s in samples if s[-1] == i]
        num_samples = int(len(class_samples) * fraction)
        selected_samples.extend(sample(class_samples, num_samples))

    shuffle(selected_samples)
    return selected_samples
