import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple
import numbers
from collections.abc import Sequence

from torch import Tensor
import torch
import munch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler

PROB_THRESHOLD = 0.5  # probability threshold.

"Credit: https://github.com/clovaai/wsolevaluation/blob/master/data_loaders.py"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.functional import _functional as dlibf
from dlib.configure import constants

from dlib.datasets.wsol_data_core import get_image_ids
from dlib.datasets.wsol_data_core import get_class_labels
from dlib.datasets.wsol_data_core import get_cams_paths
from dlib.datasets.wsol_data_core import get_mask_paths
from dlib.datasets.wsol_data_core import get_mask

from dlib.datasets.wsol_data_core import IMAGE_MEAN_VALUE
from dlib.datasets.wsol_data_core import IMAGE_STD_VALUE


_SPLITS = (constants.TRAINSET, constants.PXVALIDSET, constants.CLVALIDSET,
           constants.TESTSET)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = join(metadata_root, 'image_sizes.txt')
    metadata.localization = join(metadata_root, 'localization.txt')
    return metadata


class WSOLImageLabelDataset(Dataset):
    def __init__(self,
                 data_root,
                 metadata_root,
                 transform,
                 proxy,
                 resize_size,
                 crop_size,
                 mask_root='',
                 load_tr_masks=False,
                 num_sample_per_class=0,
                 root_data_cams=''):

        self.load_tr_masks = load_tr_masks
        self.mask_root = mask_root
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self.index_id: dict = {
            id_: idx for id_, idx in zip(self.image_ids,
                                         range(len(self.image_ids)))
        }

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        self.cams_paths = None
        if os.path.isdir(root_data_cams):
            self.cams_paths = get_cams_paths(root_data_cams=root_data_cams,
                                             image_ids=self.image_ids)

        self.resize_size = resize_size
        self.crop_size = crop_size

        self._adjust_samples_per_class()

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(join(self.data_root, image_id))
        image = image.convert('RGB')  # H, W, 3
        raw_img = image.copy()
        mask = None
        if self.load_tr_masks:
            mask = get_mask(mask_root=self.mask_root,
                            mask_paths=self.mask_paths[image_id],
                            ignore_path=self.ignore_paths[image_id],
                            f_resize_length=self.resize_size)  # h, w
            mask: torch.Tensor = torch.from_numpy(mask).long()
            mask = mask.unsqueeze(0)  # 1, H, W

        std_cam = None
        if self.cams_paths is not None:
            std_cam_path = self.cams_paths[image_id]
            # h', w'
            std_cam: torch.Tensor = torch.load(f=std_cam_path,
                                               map_location=torch.device('cpu'))
            assert std_cam.ndim == 2
            std_cam = std_cam.unsqueeze(0)  # 1, h', w'

        image, raw_img, std_cam, mask = self.transform(
            image, raw_img, std_cam, mask)

        raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
        raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.
        mask = mask.long() if mask is not None else mask  # 1, h, w. val: {0, 1}

        std_cam = 0 if std_cam is None else std_cam
        mask = 0 if mask is None else mask

        return image, image_label, image_id, raw_img, std_cam, mask

    def __len__(self):
        return len(self.image_ids)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize),
                        isinstance(t, RandomCrop),
                        isinstance(t, RandomHorizontalFlip),
                        isinstance(t, RandomVerticalFlip),
                        isinstance(t, transforms.ToTensor),
                        isinstance(t, transforms.Normalize),
                        isinstance(t, transforms.ColorJitter)
                        ]
                       )

    def chec_if_random(self, transf):
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self, img, raw_img, std_cam, mask):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip,
                              RandomCrop, Resize)):
                img, raw_img, std_cam, mask = t(img, raw_img, std_cam, mask)
            else:
                img = t(img)

        return img, raw_img, std_cam, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img,std_cam, mask):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam, mask):
        if random.random() < self.p:

            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.hflip(std_cam)

            mask_ = mask
            if mask_ is not None:
                mask_ = TF.hflip(mask_)

            return TF.hflip(img), TF.hflip(raw_img), std_cam_, mask_

        return img, raw_img, std_cam, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam, mask):
        if random.random() < self.p:

            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.vflip(std_cam)

            mask_ = mask
            if mask_ is not None:
                mask_ = TF.vflip(mask_)

            return TF.vflip(img), TF.vflip(raw_img), std_cam_, mask_

        return img, raw_img, std_cam, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self, img, raw_img, std_cam, mask):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = self.forward(std_cam)
            std_cam_ = TF.crop(std_cam_, i, j, h, w)

        mask_ = mask
        if mask is not None:
            mask_ = self.forward(mask_)
            mask_ = TF.crop(mask_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(
            raw_img_, i, j, h, w), std_cam_, mask_

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, raw_img, std_cam, mask):
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resize(std_cam_, self.size, self.interpolation)

        mask_ = mask
        if mask_ is not None:
            mask_ = TF.resize(mask_, self.size, self.interpolation)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), std_cam_, mask_

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_data_loader(data_roots,
                    metadata_root,
                    batch_size,
                    workers,
                    resize_size,
                    crop_size,
                    proxy_training_set,
                    load_tr_masks=False,
                    mask_root='',
                    num_val_sample_per_class=0,
                    std_cams_folder=None,
                    get_splits_eval=None,
                    distributed_eval=False
                    ):
    train_sampler = None

    def get_eval_tranforms():
        return Compose([
            Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE)
        ])

    if isinstance(get_splits_eval, list):
        assert len(get_splits_eval) > 0
        eval_datasets = {
            split: WSOLImageLabelDataset(
                    data_root=data_roots[split],
                    metadata_root=join(metadata_root, split),
                    transform=get_eval_tranforms(),
                    proxy=False,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    load_tr_masks=False,
                    mask_root='',
                    num_sample_per_class=0,
                    root_data_cams=''
                )
            for split in get_splits_eval
        }
        if distributed_eval:
            loaders = {
                split: DataLoader(
                    eval_datasets[split],
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=DistributedSampler(dataset=eval_datasets[split],
                                               shuffle=False),
                    num_workers=workers)
                for split in get_splits_eval
            }
        else:
            loaders = {
                split: DataLoader(
                    eval_datasets[split],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers)
                for split in get_splits_eval
            }

        return loaders, train_sampler

    # todo: check transformations in
    #  https://github.com/gatsby2016/Augmentation-PyTorch-Transforms
    dataset_transforms = {
        constants.TRAINSET: Compose([
            Resize((resize_size, resize_size)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                   saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE)
        ]),
        constants.PXVALIDSET: get_eval_tranforms(),
        constants.CLVALIDSET: get_eval_tranforms(),
        constants.TESTSET: get_eval_tranforms()
    }

    datasets = {
        split: WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == constants.TRAINSET,
                resize_size=resize_size,
                crop_size=crop_size,
                load_tr_masks=load_tr_masks if split == constants.TRAINSET
                else False,
                mask_root=mask_root if split == constants.TRAINSET else '',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == constants.PXVALIDSET else 0),
                root_data_cams=std_cams_folder[split]
            )
        for split in _SPLITS
    }

    samplers = {
        split: DistributedSampler(dataset=datasets[split],
                                  shuffle=split == constants.TRAINSET)
        for split in _SPLITS
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            sampler=samplers[split],
            num_workers=workers)
        for split in _SPLITS
    }

    if constants.TRAINSET in _SPLITS:
        train_sampler = samplers[constants.TRAINSET]

    return loaders, train_sampler
