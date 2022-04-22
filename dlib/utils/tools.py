import sys
from os.path import dirname, abspath, join, basename, normpath
import os
import subprocess
import glob
import shutil
import subprocess
import datetime as dt
import math
from collections.abc import Iterable

import torch
import yaml
from sklearn.metrics import auc
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger

from dlib.utils.shared import fmsg
from dlib.configure import constants
from dlib.utils.shared import is_cc


def get_cpu_device():
    """
    Return CPU device.
    :return:
    """
    return torch.device("cpu")


def log_device(args):
    assert torch.cuda.is_available()

    txt = subprocess.run(
        ['nvidia-smi', '--list-gpus'],
        stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    try:
        cudaids = args.cudaid.split(',')
        tag = 'CUDA devices: \n'
        for cid in cudaids:
            tag += 'ID: {} - {} \n'.format(cid, txt[int(cid)])
    except IndexError:
        tag = 'CUDA devices: lost.'

    DLLogger.log(message=tag)


def chunks_into_n(l: Iterable, n: int) -> Iterable:
    """
    Split iterable l into n chunks (iterables) with the same size.

    :param l: iterable.
    :param n: number of chunks.
    :return: iterable of length n.
    """
    chunksize = int(math.ceil(len(l) / n))
    return (l[i * chunksize:i * chunksize + chunksize] for i in range(n))


def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of
     the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def count_nb_params(model):
    """
    Count the number of parameters within a model.

    :param model: nn.Module or None.
    :return: int, number of learnable parameters.
    """
    if model is None:
        return 0
    else:
        return sum([p.numel() for p in model.parameters()])


def create_folders_for_exp(exp_folder, name):
    """
    Create a set of folder for the current exp.
    :param exp_folder: str, the path to the current exp.
    :param name: str, name of the dataset (train, validation, test)
    :return: object, where each attribute is a folder.
    There is the following attributes:
        . folder: the name of the folder that will contain everything about
        this dataset.
        . prediction: for the image prediction.
    """
    l_dirs = dict()

    l_dirs["folder"] = join(exp_folder, name)
    l_dirs["prediction"] = join(exp_folder, "{}/prediction".format(name))

    for k in l_dirs:
        if not os.path.exists(l_dirs[k]):
            os.makedirs(l_dirs[k])

    return Dict2Obj(l_dirs)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    """Copy code to the exp folder for reproducibility.
    Input:
        dest: path to the destination folder (the exp folder).
        compress: bool. if true, we compress the destination folder and
        delete it.
        verbose: bool. if true, we show what is going on.
    """
    # extensions to copy.
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def log_args(args_dict):
    DLLogger.log(fmsg("Configuration"))
    # todo


def save_model(model, args, outfd):
    model.eval()
    cpu_device = get_cpu_device()
    model.to(cpu_device)
    torch.save(model.state_dict(), join(outfd, "best_model.pt"))

    if args.task == constants.STD_CL:
        tag = "{}-{}-{}".format(
            args.dataset, args.model['encoder_name'], args.spatial_pooling)
        path = join(outfd, tag)
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(model.encoder.state_dict(), join(path, 'encoder.pt'))
        torch.save(model.classification_head.state_dict(),
                   join(path, 'head.pt'))
        DLLogger.log(message="Stored classifier. TAG: {}".format(tag))


def save_config(config_dict, outfd):
    with open(join(outfd, 'config.yaml'), 'w') as fout:
        yaml.dump(config_dict, fout)


def get_best_epoch(fyaml):
    with open(fyaml, 'r') as f:
        config = yaml.load(f)
        return config['best_epoch']


def compute_auc(vec: np.ndarray, nbr_p: int):
    """
    Compute the area under a curve.
    :param vec: vector contains values in [0, 100.].
    :param nbr_p: int. number of points in the x-axis. it is expected to be
    the same as the number of values in `vec`.
    :return: float in [0, 100]. percentage of the area from the perfect area.
    """
    if vec.size == 1:
        return float(vec[0])
    else:
        area_under_c = auc(x=np.array(list(range(vec.size))), y=vec)
        area_under_c /= (100. * (nbr_p - 1))
        area_under_c *= 100.  # (%)
        return area_under_c


# WSOL

def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

def t2n(t):
    return t.detach().cpu().numpy().astype(float)


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def get_tag(args: object, checkpoint_type: str = None) -> str:
    if checkpoint_type is None:
        checkpoint_type = args.eval_checkpoint_type

    if args.task == constants.SEG:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]
        tag = "{}-{}-{}-{}-{}-{}-cp_{}".format(
            args.dataset,
            args.fold,
            args.model['arch'],
            args.model['encoder_name'],
            args.method,
            args.spatial_pooling,
            checkpoint_type)
    elif args.task == constants.NEGEV:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]
        tag = "{}-{}-{}-{}-{}-{}-cp_{}".format(
            args.dataset,
            args.fold,
            args.model['arch'],
            args.model['encoder_name'],
            args.method,
            args.spatial_pooling,
            checkpoint_type)
    elif args.dataset == constants.BREAKHIS:
        tag = "{}-{}-{}-{}-{}-{}-cp_{}".format(
            args.dataset,
            args.fold,
            args.magnification,
            args.model['encoder_name'],
            args.method,
            args.spatial_pooling,
            checkpoint_type)
    else:
        tag = "{}-{}-{}-{}-{}-cp_{}".format(
            args.dataset,
            args.fold,
            args.model['encoder_name'],
            args.method,
            args.spatial_pooling,
            checkpoint_type)

    return tag


def bye(args):
    DLLogger.log(fmsg("End time: {}".format(args.tend)))
    DLLogger.log(fmsg("Total time: {}".format(args.tend - args.t0)))

    with open(join(root_dir, 'LOG.txt'), 'a') as f:
        if args.task == constants.F_CL:
            m = "{}: \t " \
                "Dataset: {} \t " \
                "Method: {} \t " \
                "Spatial pooling: {} \t " \
                "Encoder: {} \t " \
                "Check point: {} \t " \
                "SL: {} \t " \
                "CRF: {} \t " \
                "... Passed in [{}]. \n".format(
                    dt.datetime.now(),
                    args.dataset,
                    args.method,
                    args.spatial_pooling,
                    args.model['encoder_name'],
                    args.eval_checkpoint_type,
                    args.sl_fc,
                    args.crf_fc,
                    args.tend - args.t0
                )

        elif args.task == constants.NEGEV:
            m = "{}: \t " \
                "Dataset: {} \t " \
                "Method: {} \t " \
                "Spatial pooling: {} \t " \
                "Encoder: {} \t " \
                "Check point: {} \t " \
                "SL: {} \t " \
                "CRF: {} \t " \
                "JCRF: {} \t " \
                "Size: {} \t " \
                "Neg-Samples: {} \t " \
                "... Passed in [{}]. \n".format(
                    dt.datetime.now(),
                    args.dataset,
                    args.method,
                    args.spatial_pooling,
                    args.model['encoder_name'],
                    args.eval_checkpoint_type,
                    args.sl_ng,
                    args.crf_ng,
                    args.jcrf_ng,
                    args.max_sizepos_ng,
                    args.neg_samples_ng,
                    args.tend - args.t0
                )
        else:
            m = "{}: \t " \
                "Dataset: {} \t " \
                "Method: {} \t " \
                "Spatial pooling: {} \t " \
                "Encoder: {} \t " \
                "Check point: {} \t " \
                "... Passed in [{}]. \n".format(
                    dt.datetime.now(),
                    args.dataset,
                    args.method,
                    args.spatial_pooling,
                    args.model['encoder_name'],
                    args.eval_checkpoint_type,
                    args.tend - args.t0
                )
        f.write(m)

    with open(join(args.outd, 'passed.txt'), 'w') as fout:
        fout.write('Passed.')

    DLLogger.log(fmsg('bye.'))

    # clean cc
    if is_cc():
        scratch_exp_fd = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER,
                              args.subpath)
        os.makedirs(scratch_exp_fd, exist_ok=True)
        scratch_tmp = dirname(normpath(scratch_exp_fd))  # parent
        _tag = basename(normpath(args.outd))
        cmdx = [
            "cd {} ".format(args.outd),
            "cd .. ",
            "tar -cf {}.tar.gz {}".format(_tag, _tag),
            'cp {}.tar.gz {}'.format(_tag, scratch_tmp),
            'cd {}'.format(scratch_tmp),
            'tar -xf {}.tar.gz -C {} --strip-components=1'.format(
                _tag, basename(normpath(scratch_exp_fd))),
            "rm {}.tar.gz".format(_tag)
        ]
        cmdx = " && ".join(cmdx)
        print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
        subprocess.run(cmdx, shell=True, check=True)


if __name__ == '__main__':
    print(root_dir)
