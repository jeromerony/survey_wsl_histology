import datetime as dt
from os.path import join

import torch.cuda

from dlib.process.parseit import parse_input

from dlib.utils.shared import fmsg

from dlib.utils.tools import log_device
from dlib.utils.tools import log_args
from dlib.utils.tools import save_model
from dlib.utils.tools import get_best_epoch


from dlib.process.instantiators import get_model
from dlib.process.instantiators import get_evaluer


from dlib.process.prologues import prologue_init
from dlib.process.prologues import make_set_folds

from dlib.routines.fast_eval import acc_cam_seed_std_cl

from dlib.datasets.tools import get_transforms_tensor
from dlib.datasets.tools import get_rootpath_2_dataset
from dlib.datasets.tools import csv_loader
from dlib.datasets.tools import get_csv_files
from dlib.datasets.tools import get_validationset

from dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger


DIAG_STD_SEED = False

args, args_dict, input_args = parse_input(eval=True)


if __name__ == "__main__":
    init_time = dt.datetime.now()

    OUTD, OUTD_TLB, SELFLEARNEDD = prologue_init(args, input_args, args_dict)
    OUTD_TR, OUTD_VL, OUTD_TS = make_set_folds(OUTD)

    args.abs_fd_exp = OUTD

    log_backends = [
        ArbJSONStreamBackend(Verbosity.VERBOSE, join(OUTD, "log-eval.json")),
        ArbTextStreamBackend(Verbosity.VERBOSE, join(OUTD, "log-eval.txt")),
    ]

    if args.verbose:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
    DLLogger.init_arb(backends=log_backends)

    log_args(args_dict)

    log_device(args)
    torch.cuda.set_device(0)

    # data
    DLLogger.log(
        fmsg("Dataset: {} - SPLIT: {} - FOLD: {}".format(
            args.dataset, args.split, args.fold)))

    transform_tensor, img_range = get_transforms_tensor(args)
    args.img_range = img_range
    train_csv, valid_csv, test_csv = get_csv_files(args)
    rootpath = get_rootpath_2_dataset(args)
    train_samples = csv_loader(train_csv, rootpath, drop_normal=False)
    valid_samples = csv_loader(valid_csv, rootpath, drop_normal=False)
    test_samples = csv_loader(test_csv, rootpath, drop_normal=False)

    validset, valid_loader = get_validationset(
        args, valid_samples, transform_tensor, batch_size=None)

    model = get_model(args, eval=True)
    model.cuda()

    if DIAG_STD_SEED and args.task == constants.STD_CL:
        subset = constants.TRAINSET
        print(fmsg('DIAG-STD-SEED: {}'.format(subset)))

        _, setloader = get_validationset(
            args, train_samples, transform_tensor, batch_size=None)

        _fdout = join(OUTD, 'DIAG-STD-SEED', subset)
        acc_cam_seed_std_cl(args, setloader, model, device, subset,
                            _fdout)

        import sys
        sys.exit()

    samples = [train_samples, valid_samples, test_samples][::-1]
    subsets = [constants.TRAINSET, constants.VALIDSET, constants.TESTSET][::-1]
    fdsout = [OUTD_TR, OUTD_VL, OUTD_TS][::-1]
    best_epoch = get_best_epoch(join(OUTD, 'config.yaml'))

    for sub, s, fdout in zip(subsets, samples, fdsout):
        evaluer = get_evaluer(args, device, model, sub, fdout,
                              store_per_sample=False,
                              store_fig_cams=(sub == constants.TESTSET)
                              )
        _, setloader = get_validationset(args, s, transform_tensor,
                                         batch_size=None)
        evaluer.run(setloader, best_epoch)
        evaluer.store_perm_meters()
        evaluer.compress_fdout()
        DLLogger.flush()

    save_model(model, args, OUTD)

    DLLogger.log("Total time: {}".format(dt.datetime.now() - init_time))
    DLLogger.log(fmsg("End eval. Bye."))

