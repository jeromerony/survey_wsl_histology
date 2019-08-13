from PIL import Image
from torch.utils.data import Dataset

from .utils import check_files
from utils.data.utils import load_data

class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, preload=False, resize=None, with_mask=False):
        self.transform = transform
        self.resize = resize
        self.with_mask = with_mask

        self.samples = check_files(data_path, files)
        self.n = len(self.samples)

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, _, _ in self.samples], resize=resize)
            if with_mask:
                self.masks = load_data([mask_path for _, mask_path, _ in self.samples if mask_path != ''],
                                       resize=resize)
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
        image_size = image.size # to generate the mask if there is no file

        if self.transform is not None:
            image = self.transform(image)

        if self.with_mask:
            if mask_path == '':
                mask = Image.new('L', image_size)
            else:
                if self.preloaded:
                    mask = self.masks[mask_path].convert('L')
                else:
                    mask = Image.open(mask_path).convert('L')
                    if self.resize is not None:
                        mask = mask.resize(self.resize, resample=Image.LANCZOS)

            if self.transform is not None:
                mask = self.transform(mask)

            return image, mask, label

        return image, label