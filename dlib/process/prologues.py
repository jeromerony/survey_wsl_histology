import os
import sys
from os.path import join, dirname, expanduser, abspath

import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.tools import create_folders_for_exp
from dlib.utils.tools import copy_code

from dlib.utils.shared import wrap_command_line

from dlib.configure import constants


__all__ = [
    'prologue_init',
    'make_set_folds'
]


def prologue_init(args, input_args, args_dict):
    """
    Prepare writing folders.
    :return:
    """
    # Write in scratch instead of /project
    output = dict()
    subset_target = 'train'
    placement_scr, parent, exp_name = None, None, None
    placement_node = None

    eval = not (args.fd_exp in [None, ''])

    tag = [('id', args.exp_id),
           ('tsk', args.task),
           ('ds', args.dataset),
           ('sd', args.MYSEED),
           ('ecd', args.model['encoder_name']),
           ('st', args.split),
           ('fl', args.fold),
           ('epx', args.max_epochs),
           ('bsz', args.batch_size),
           ('lr', args.optimizer['opt__lr']),
           ('scale_in', args.model['scale_in']),
           ('pxl_sup', args.pxl_sup)
           ]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    if args.task == constants.F_CL:
        # todo: add hyper-params.
        tag2 = []

        if args.sl_fc:
            tag2.append(("sl_fc", 'yes'))

        if args.crf_fc:
            tag2.append(("crf_fc", 'yes'))

        if args.entropy_fc:
            tag2.append(("entropy_fc", 'yes'))

        if args.partuncertentro_lc:
            tag2.append(("partuncertentro_lc", 'yes'))

        if args.partcert_lc:
            tag2.append(("partcert_lc", 'yes'))
            if args.partcert_lc_elb:
                tag2.append(('partcert_lc_elb', 'yes'))
            if args.partcert_lc_logit:
                tag2.append(('partcert_lc_logit', 'yes'))

        if args.min_sizeneg_lc:
            tag2.append(("min_sizeneg_lc", 'yes'))

        if args.max_sizepos_lc:
            tag2.append(("max_sizepos_lc", 'yes'))

        if args.max_sizepos_fc:
            tag2.append(("max_sizepos_fc", 'yes'))

        if tag2:
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

    parent_lv = "exps"
    if args.debug_subfolder != '':
        parent_lv = join(parent_lv, args.debug_subfolder)

    if not eval:
        OUTD = join(root_dir,
                    parent_lv,
                    tag
                    )
    else:
        OUTD = join(root_dir, args.fd_exp)

    OUTD = expanduser(OUTD)
    SELFLEARNEDD = join(OUTD, 'self_learned')

    lfolders = [OUTD, SELFLEARNEDD]

    for fdxx in lfolders:
        if not os.path.exists(fdxx):
            os.makedirs(fdxx)



    OUTD_TLB = join(OUTD, "tlb")

    if not eval:
        if not os.path.exists(join(OUTD, "code/")):
            os.makedirs(join(OUTD, "code/"))

        with open(join(OUTD, "code/", input_args.yaml), 'w') as fyaml:
            args_dict['fd_exp'] = join(parent_lv, tag)
            yaml.dump(args_dict, fyaml)

        str_cmd = "time python " + " ".join(sys.argv)
        str_cmd = wrap_command_line(str_cmd)
        with open(join(OUTD, "code/cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

        copy_code(join(OUTD, "code/"), compress=True, verbose=False)

    return OUTD, OUTD_TLB, SELFLEARNEDD


def make_set_folds(outd):

    OUTD_TR = join(outd, "train")
    OUTD_VL = join(outd, "valid")
    OUTD_TS = join(outd, "test")

    for f in [OUTD_TR, OUTD_VL, OUTD_TS]:
        if not os.path.isdir(f):
            os.makedirs(f)

    return OUTD_TR, OUTD_VL, OUTD_TS
