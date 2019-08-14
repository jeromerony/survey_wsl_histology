from PIL import Image
from torch.utils.data import Dataset

from utils.data.classification.utils import check_files
from utils.data.utils import load_data


class PatchDataset(Dataset):
    def __init__(self, data_path, files, transform, patch_size, extractor, preload=False, resize=None):
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.transform = transform

        self.samples = check_files(data_path, files)
        self.n = len(self.samples)

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, label in self.samples], resize=resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded')

        img = Image.open(self.samples[0][0])
        size = img.size if resize is None else resize
        inverted_size = (size[1], size[0])  # invert terms: PIL returns image size as (width, height)
        self.extractor = extractor(inverted_size, self.patch_size)

    def __len__(self):
        return len(self.samples) * len(self.extractor)

    def __getitem__(self, index):
        image_path, label = self.samples[index // len(self.extractor)]
        patch_index = index % len(self.extractor)

        if self.preloaded:
            image = self.images[image_path]
        else:
            image = Image.open(image_path)

        # extract patch
        patch = self.extractor(image, patch_index)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, label

