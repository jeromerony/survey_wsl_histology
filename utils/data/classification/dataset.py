from PIL import Image
from torch.utils.data import Dataset

from .utils import check_files
from utils.data.utils import load_data


class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, preload=False, resize=None):
        self.transform = transform
        self.resize = resize

        self.samples = check_files(data_path, files)
        self.n = len(self.samples)

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, label in self.samples], resize=resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        if self.preloaded:
            image = self.images[image_path].convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
            if self.resize is not None:
                image = image.resize(self.resize, resample=Image.LANCZOS)

        if self.transform is not None:
            image = self.transform(image)

        return image, label