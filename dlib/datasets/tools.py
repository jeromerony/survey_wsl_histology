import sys
import os
from os.path import dirname, abspath, join
import csv

import yaml

from torchvision import transforms
from torch.utils.data import DataLoader

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets import transforms as extended_transforms
from dlib.datasets.loader import Dataset
from dlib.datasets.loader import default_collate
from dlib.datasets.loader import _init_fn

from dlib.configure import constants

from dlib.utils.reproducibility import set_default_seed


def get_train_transforms_img(args):
    """
    Get the transformation to perform over the images for the train samples.
    All the transformation must perform on PIL.Image.Image and returns a
    PIL.Image.Image object.

    :param args: object. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: a torchvision.transforms.Compose() object.
    """

    if args.dataset == "bach-part-a-2018":
        # TODO: check values of jittering: https://arxiv.org/pdf/1806.07064.pdf
        return extended_transforms.Compose([
            extended_transforms.ColorJitter(0.5, 0.5, 0.5, 0.05),
            extended_transforms.RandomHorizontalFlip(),
            extended_transforms.RandomVerticalFlip()
        ])
    elif args.dataset == "fgnet":
        return extended_transforms.Compose([
            # extended_transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == "afad-lite":
        return extended_transforms.Compose([
            # extended_transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == "afad-full":
        return extended_transforms.Compose([
            # extended_transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == "historical-color-image-decade":
        return extended_transforms.Compose([
            # extended_transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            extended_transforms.RandomHorizontalFlip(),
            extended_transforms.RandomVerticalFlip()  # objects are not important.
        ])
    elif args.dataset == constants.GLAS:
        return extended_transforms.Compose([
            # extended_transforms.ColorJitter(brightness=0.5, contrast=0.5,
            # saturation=0.5, hue=0.05),
            extended_transforms.RandomHorizontalFlip(),
            extended_transforms.RandomVerticalFlip()
        ])
    elif args.dataset == constants.CAM16:
        return extended_transforms.Compose([
            extended_transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.05
            ),
            extended_transforms.RandomHorizontalFlip(),
            extended_transforms.RandomVerticalFlip()
        ])
    elif args.dataset == constants.CUB:
        return extended_transforms.Compose([
            # extended_transforms.RandomGaussianBlur(),
            # extended_transforms.ColorJitter(0.25, 0.25, 0.25, 0.0),
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == constants.OXF:
        return extended_transforms.Compose([
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == constants.CSCAPES:
        return extended_transforms.Compose([
            extended_transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            extended_transforms.RandomGaussianBlur(),
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset in ['cifar-10', 'cifar-100', 'svhn']:
        return extended_transforms.Compose([
            extended_transforms.RandomAffine(0, translate=(1 / 16, 1 / 16)),
            # affine translation by at most 2 pixels.
            extended_transforms.RandomHorizontalFlip()
        ])
    elif args.dataset in ['mnist']:
        return None
    else:
        raise ValueError("Dataset {} unsupported. Exiting .... "
                         "[NOT OK]".format(args.dataset))


def get_transforms_tensor(args):
    """
     Return tensor transforms.
    :param args: object. Contains the configuration of the exp that has
    been read from the yaml file.
    :return: a torchvision.transforms.Compose() object.
    """
    if args.dataset == "bach-part-a-2018":
        # TODO: check values of jittering: https://arxiv.org/pdf/1806.07064.pdf
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == 'fgnet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == 'afad-lite':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == 'afad-full':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == 'historical-color-image-decade':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == constants.CUB:
        # Normalization.
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/
        # blob/28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py
        # [102.9801 / 255., 115.9465 / 255., 122.7717 / 255.],
        #                                  [1., 1., 1.]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == constants.OXF:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset in ['cifar-10', 'cifar-100', 'svhn']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset in ['mnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],
                                 [0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == constants.GLAS:
        # resnet pretrained:
        # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c5976
        # 15dc0ad8f6173/imagenet/main.py#L197-L198
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]), constants.RANGE_TANH
    elif args.dataset == constants.CAM16:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]), constants.RANGE_TANH
    elif args.dataset == constants.CSCAPES:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]), constants.RANGE_TANH
    else:
        raise ValueError("Dataset {} unsupported. Exiting .... "
                         "[NOT OK]".format(args.dataset))


def get_rootpath_2_dataset(args):
    """
    Returns the root path to the dataset depending on the server.
    :param args: object. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: baseurl, a str. The root path to the dataset independently from
    the host.
    """
    datasetname = args.dataset
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets".format(os.environ["NEWHOME"])
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = "{}/datasets".format(os.environ["SBHOME"])
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = "{}/datasets".format(os.environ["DATASETSH"])

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = "{}/datasets".format(os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this cate my happen if someone calls this function outside a job.
            baseurl = "{}/datasets".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are enable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "So, we recommend you add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    if datasetname == "bach-part-a-2018":
        baseurl = join(baseurl, "ICIAR-2018-BACH-Challenge")
    elif datasetname == "fgnet":
        baseurl = join(baseurl, "FGNET")
    elif datasetname == "afad-lite":
        baseurl = join(baseurl, "tarball-lite")
    elif datasetname == "afad-full":
        baseurl = join(baseurl, "tarball")
    elif datasetname == constants.CUB:
        baseurl = join(baseurl, "Caltech-UCSD-Birds-200-2011")
    elif datasetname == constants.OXF:
        baseurl = join(baseurl, 'Oxford-flowers-102')
    elif datasetname == constants.CAM16:  # camelyon16
        pass  # relative path starts from the root: camelyon16/......
    elif datasetname == 'historical-color-image-decade':
        baseurl = join(baseurl, 'HistoricalColor-ECCV2012')
    elif datasetname == 'cifar-10':
        baseurl = join(baseurl, 'cifar-10')
    elif datasetname == 'cifar-100':
        baseurl = join(baseurl, 'cifar-100')
    elif datasetname == 'svhn':
        baseurl = join(baseurl, 'svhn')
    elif datasetname == 'mnist':
        baseurl = join(baseurl, 'mnist')
    elif datasetname == constants.GLAS:
        baseurl = join(baseurl,
                       "GlaS-2015/Warwick QU Dataset (Released 2016_07_08)")
    elif datasetname == constants.CSCAPES:
        baseurl = join(baseurl, "cityscapes")

    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def drop_normal_samples(l_samples):
    """
    Remove normal samples from the list of samples.

    When to call this?
    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.

    :param l_samples: list of samples resulting from csv_loader().
    :return: l_samples without any normal sample.
    """
    return [el for el in l_samples if el[3] == 'tumor']


def csv_loader(fname, rootpath, drop_normal=False):
    """
    Read a *.csv file. Each line contains:
     0. id_: str
     1. img: str
     2. mask: str or '' or None
     3. label: str
     4. tag: int in {0, 1}

     Example: 50162.0, test/img_50162_label_frog.jpeg, , frog, 0

    :param fname: Path to the *.csv file.
    :param rootpath: The root path to the folders of the images.
    :return: List of elements.
    :param drop_normal: bool. if true, normal samples are dropped.
    Each element is the path to an image: image path, mask path [optional],
    class name.
    """
    with open(fname, 'r') as f:
        out = [
            [row[0],
             join(rootpath, row[1]),
             join(rootpath, row[2]) if row[2] else None,
             row[3],
             int(row[4])
             ]
            for row in csv.reader(f)
        ]

    if drop_normal:
        out = drop_normal_samples(out)

    return out


def csv_writer(data, fname):
    """
    Write a list of rows into a file.
    """
    msg = "'data' must be a list. found {}".format(type(data))
    assert isinstance(data, list), msg

    with open(fname, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            filewriter.writerow(row)


def get_csv_files(args):
    """
    Get the csv files.
    :return:
    """
    relative_fold_path = join(
        root_dir,
        args.fold_folder,
        args.dataset,
        "split_" + str(args.split),
        "fold_" + str(args.fold)
    )
    if isinstance(args.name_classes, str):  # path
        path_classes = join(relative_fold_path, args.name_classes)
        assert os.path.isfile(path_classes), "File {} does not exist .... " \
                                             "[NOT OK]".format(path_classes)
        with open(path_classes, "r") as fin:
            args.name_classes = yaml.load(fin)
    csvfiles = []
    for subp in ["train_s_", "valid_s_", "test_s_"]:
        csvfiles.append(
            join(
                relative_fold_path, "{}{}_f_{}.csv".format(
                    subp, args.split, args.fold))
        )

    train_csv, valid_csv, test_csv = csvfiles

    for fcsv in csvfiles:
        if not os.path.isfile(fcsv):
            raise ValueError("{} does not exist.".format(fcsv))

    return train_csv, valid_csv, test_csv


def get_trainset(args,
                 train_samples,
                 transform_tensor,
                 train_transform_img,
                 SELFLEARNDD
                 ):
    """
    Get the trainset.
    :return:
    """
    set_default_seed()

    pxl_sup = args.pxl_sup
    if pxl_sup == constants.VOID:
        pxl_sup = constants.ORACLE

    trainset = Dataset(
        data=train_samples,
        transform_tensor=transform_tensor,
        dataset_name=args.dataset,
        name_classes=args.name_classes,
        mode=constants.DS_TRAIN,
        multi_label_flag=args.multi_label_flag,
        transform_img=train_transform_img,
        crop_size=args.crop_size,
        pxl_sup=pxl_sup,
        folder_self_l=SELFLEARNDD
    )
    set_default_seed()

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_init_fn,
        collate_fn=default_collate
    )

    set_default_seed()

    return trainset, train_loader


def get_validationset(args,
                      valid_samples,
                      transform_tensor,
                      batch_size=None
                      ):
    """
    Get the validation set
    :param batch_size: int or None. batch size. if None, the value defined in
    `args.valid_batch_size` will be used.
    :return:
    """
    set_default_seed()

    validset = Dataset(
        data=valid_samples,
        transform_tensor=transform_tensor,
        dataset_name=args.dataset,
        name_classes=args.name_classes,
        mode=constants.DS_EVAL,
        multi_label_flag=args.multi_label_flag,
        transform_img=None,
        crop_size=None,
        pxl_sup=constants.ORACLE,
        folder_self_l=None
    )
    set_default_seed()

    valid_loader = DataLoader(
        validset,
        batch_size=args.valid_batch_size if batch_size is None else batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        worker_init_fn=_init_fn
    )
    set_default_seed()

    return validset, valid_loader