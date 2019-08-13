from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from random import random, randint

from ..localization.utils import check_files
from ..utils import load_data

class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, mask_transform, augment=False, patch_size=None, rotate=False,
                 preload=False, resize=None, min_resize=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.resize = resize
        self.min_resize = min_resize
        self.augment = augment
        self.patch_size = patch_size
        self.rotate = rotate

        self.samples = check_files(data_path, files)
        self.n = len(self.samples)

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, _, _ in self.samples],
                                    resize=resize, min_resize=min_resize)
            self.masks = load_data([mask_path for _, mask_path, _ in self.samples if mask_path != ''],
                                   resize=resize, min_resize=min_resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]

        if self.preloaded:
            image = self.images[image_path].convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
            if self.resize is not None:
                image = image.resize(self.resize, resample=Image.LANCZOS)
            elif self.min_resize is not None:
                image = F.resize(image, self.min_resize, interpolation=Image.LANCZOS)
        image_size = image.size # to generate the mask if there is no file

        if mask_path == '':
            mask = Image.new('L', image_size)
        else:
            if self.preloaded:
                mask = self.masks[mask_path].convert('L')
            else:
                mask = Image.open(mask_path).convert('L')
                if self.resize is not None:
                    mask = mask.resize(self.resize, resample=Image.LANCZOS)
                elif self.min_resize is not None:
                    mask = F.resize(mask, self.min_resize, interpolation=Image.LANCZOS)

        if self.augment:

            # extract patch
            if self.patch_size is not None:
                left = randint(0, image_size[0] - self.patch_size)
                up = randint(0, image_size[1] - self.patch_size)
                image = image.crop(box=(left, up, left + self.patch_size, up + self.patch_size))
                mask = mask.crop(box=(left, up, left + self.patch_size, up + self.patch_size))

            # rotate
            if self.rotate:
                angle = randint(0, 3) * 90
                image = image.rotate(angle)
                mask = mask.rotate(angle)

            # flip
            if random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask, label