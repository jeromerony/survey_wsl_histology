from sacred import Ingredient
from torchvision import transforms
from torch.utils.data import DataLoader

from .patch_extraction import RandomProperRotation
from utils.data.utils import RandomDiscreteRotation, ExpandedRandomSampler, check_overlap, fraction_dataset

dataset_ingredient = Ingredient('dataset')

test_transform = transforms.ToTensor()


@dataset_ingredient.config
def config():
    data_path = 'data'
    split = 0
    fold = 0
    preload = True
    batch_size = 64
    shuffle = True
    num_workers = 8
    drop_last = True
    pin_memory = True

    fraction = 1


@dataset_ingredient.named_config
def bach():
    name = 'bach'
    folds_dir = 'folds/bach'
    patch_size = 512


@dataset_ingredient.named_config
def breakhis():
    data_path = 'data/BreakHis'
    name = 'breakhis'
    folds_dir = 'folds/breakhis'
    fold = 1
    zoom = 40
    patch_size = 448
    sampler_mul = 4


@dataset_ingredient.capture
def load_bach(folds_dir, split, fold, data_path, patch_size, preload, batch_size,
               shuffle, num_workers, drop_last, pin_memory, fraction):
    from .bach.dataset import PatchDataset
    from .dataset import PhotoDataset
    from .bach.utils import get_files, decode_classes

    files = get_files(folds_dir, split, fold)
    train_files, valid_files, test_files = [decode_classes(f) for f in files]
    check_overlap(train_files, valid_files, test_files)

    if fraction < 1:
        train_files = fraction_dataset(train_files, 4, fraction)

    train_loader = DataLoader(
        PatchDataset(
            data_path=data_path,
            files=train_files,
            transform=transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            patch_size=patch_size,
            extractor=RandomProperRotation,
            preload=preload
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path=data_path, files=valid_files, transform=test_transform, preload=preload),
        batch_size=1, num_workers=1,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        PhotoDataset(data_path=data_path, files=test_files, transform=test_transform, preload=preload),
        batch_size=1, num_workers=1,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


@dataset_ingredient.capture
def load_breakhis(folds_dir, split, fold, zoom, data_path, preload, patch_size, batch_size, shuffle,
                  sampler_mul, num_workers, drop_last, pin_memory, fraction):
    from .dataset import PhotoDataset
    from .breakhis.utils import get_files, decode_classes

    files = get_files(folds_dir, split, fold, zoom)
    train_files, valid_files, test_files = [decode_classes(f) for f in files]
    check_overlap(train_files, valid_files, test_files)

    if fraction < 1:
        train_files = fraction_dataset(train_files, 2, fraction)

    train_loader = DataLoader(
        PhotoDataset(
            data_path=data_path,
            files=train_files,
            transform=transforms.Compose([
                transforms.RandomCrop((patch_size, patch_size)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.RandomHorizontalFlip(),
                RandomDiscreteRotation([i * 90 for i in range(4)]),
                transforms.ToTensor(),
            ]),
            preload=preload
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=ExpandedRandomSampler(len(train_files), sampler_mul),
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path=data_path, files=valid_files, transform=test_transform, preload=preload),
        batch_size=1, num_workers=1,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        PhotoDataset(data_path=data_path, files=test_files, transform=test_transform, preload=preload),
        batch_size=1, num_workers=1,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


_dataset_loaders = {
    'bach': load_bach,
    'breakhis': load_breakhis,
}


@dataset_ingredient.capture
def load_dataset(name):
    return _dataset_loaders[name]()
