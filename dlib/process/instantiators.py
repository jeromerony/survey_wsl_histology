import warnings
import sys
import os
from os.path import dirname, abspath, join, basename
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.learning import lr_scheduler as my_lr_scheduler

from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import count_nb_params
from dlib.configure import constants
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag
from dlib.utils.shared import format_dict_2_str

import dlib
from dlib import create_model

from dlib.losses.elb import ELB
from dlib import losses


import dlib.dllogger as DLLogger


__all__ = [
    'get_loss',
    'get_pretrainde_classifier',
    'get_model',
    'get_optimizer'
]


def get_negev_loss(args, masterloss):
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag

    assert args.dataset in [constants.CAMELYON512, constants.GLAS]

    if not args.model['freeze_cl']:
        masterloss.add(losses.ClLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.crf_ng:
        masterloss.add(losses.ConRanFieldNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.crf_ng_lambda,
            sigma_rgb=args.crf_ng_sigma_rgb,
            sigma_xy=args.crf_ng_sigma_xy,
            scale_factor=args.crf_ng_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.crf_ng_start_ep,
            end_epoch=args.crf_ng_end_ep,
        ))

    if args.jcrf_ng:
        ljcrf = losses.JointConRanFieldNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.jcrf_ng_lambda,
            sigma_rgb=args.jcrf_ng_sigma_rgb,
            scale_factor=args.jcrf_ng_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.jcrf_ng_start_ep,
            end_epoch=args.jcrf_ng_end_ep
        )
        ljcrf.set_it(pair_mode=args.jcrf_ng_pair_mode, n=args.jcrf_ng_n,
                     dataset_name=args.dataset)
        masterloss.add(ljcrf)

    if args.max_sizepos_ng:
        size_loss = losses.MaxSizePositiveNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.max_sizepos_ng_lambda,
            elb=deepcopy(elb),
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.max_sizepos_ng_start_ep,
            end_epoch=args.max_sizepos_ng_end_ep
        )
        size_loss.set_it(apply_negative_samples=not constants.DS_HAS_NEG_SAM[
            args.dataset], negative_c=constants.DS_NEG_CL[args.dataset])
        masterloss.add(size_loss)

    if args.neg_samples_ng:
        lnegs = losses.NegativeSamplesNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.neg_samples_ng_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.neg_samples_ng_start_ep,
            end_epoch=args.neg_samples_ng_end_ep
        )
        lnegs.set_it(negative_c=constants.DS_NEG_CL[args.dataset])
        masterloss.add(lnegs)

    if args.sl_ng:
        sl_loss = losses.SelfLearningNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.sl_ng_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.sl_ng_start_ep,
            end_epoch=args.sl_ng_end_ep,
            seg_ignore_idx=args.seg_ignore_idx
        )
        sl_loss.set_it(apply_negative_samples=not constants.DS_HAS_NEG_SAM[
            args.dataset], negative_c=constants.DS_NEG_CL[args.dataset])

        masterloss.add(sl_loss)

    return masterloss


def get_encoder_d_c(encoder_name):
    if encoder_name in [constants.VGG16]:
        vgg_encoders = dlib.encoders.vgg_encoders
        encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        decoder_channels = (256, 128, 64)
    else:
        encoder_depth = 5
        decoder_channels = (256, 128, 64, 32, 16)

    return encoder_depth, decoder_channels


def get_loss(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    # image classification loss
    if args.task == constants.STD_CL:
        if args.method == constants.METHOD_SPG:
            cl_loss = losses.SpgLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag)
            cl_loss.spg_threshold_1h = args.spg_threshold_1h
            cl_loss.spg_threshold_1l = args.spg_threshold_1l
            cl_loss.spg_threshold_2h = args.spg_threshold_2h
            cl_loss.spg_threshold_2l = args.spg_threshold_2l
            cl_loss.spg_threshold_3h = args.spg_threshold_3h
            cl_loss.spg_threshold_3l = args.spg_threshold_3l
            cl_loss.hyper_p_set = True
            masterloss.add(cl_loss)

        elif args.method == constants.METHOD_ACOL:
            masterloss.add(losses.AcolLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        elif args.method == constants.METHOD_CUTMIX:
            masterloss.add(losses.CutMixLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        elif args.method == constants.METHOD_MAXMIN:
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)
            loss = losses.MaxMinLoss(
                    cuda_id=args.c_cudaid,
                    elb=elb,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
            loss.set_dataset_name(dataset_name=args.dataset)
            loss.set_lambda_neg(lambda_neg=args.minmax_lambda_neg)
            loss.set_lambda_size(lambda_size=args.minmax_lambda_size)
            masterloss.add(loss)

        else:
            masterloss.add(losses.ClLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

    elif args.task == constants.SEG:
        masterloss.add(losses.SegLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elif args.task == constants.NEGEV:
        masterloss = get_negev_loss(args, masterloss)
    # fcams
    elif args.task == constants.F_CL:

        if not args.model['freeze_cl']:
            masterloss.add(losses.ClLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                  mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

        if args.im_rec:
            masterloss.add(
                losses.ImgReconstruction(
                    cuda_id=args.c_cudaid,
                    lambda_=args.im_rec_lambda,
                    elb=deepcopy(elb) if args.sr_elb else nn.Identity(),
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
            )

        if args.crf_fc:
            masterloss.add(losses.ConRanFieldFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.crf_lambda,
                sigma_rgb=args.crf_sigma_rgb, sigma_xy=args.crf_sigma_xy,
                scale_factor=args.crf_scale,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.crf_start_ep, end_epoch=args.crf_end_ep,
            ))

        if args.entropy_fc:
            masterloss.add(losses.EntropyFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.entropy_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        if args.max_sizepos_fc:
            masterloss.add(losses.MaxSizePositiveFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.max_sizepos_fc_lambda,
                elb=deepcopy(elb), support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.max_sizepos_fc_start_ep,
                end_epoch=args.max_sizepos_fc_end_ep
            ))

        if args.sl_fc:
            sl_fcam = losses.SelfLearningFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.sl_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sl_start_ep, end_epoch=args.sl_end_ep,
                seg_ignore_idx=args.seg_ignore_idx
            )

            masterloss.add(sl_fcam)

        assert len(masterloss.n_holder) > 1
    else:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message="Train loss: {}".format(masterloss))
    return masterloss


def get_aux_params(args):
    """
    Prepare the head params.
    :param args:
    :return:
    """
    assert args.spatial_pooling in constants.SPATIAL_POOLINGS
    return {
        "pooling_head": args.spatial_pooling,
        "classes": args.num_classes,
        "modalities": args.wc_modalities,
        "kmax": args.wc_kmax,
        "kmin": args.wc_kmin,
        "alpha": args.wc_alpha,
        "dropout": args.wc_dropout,
        "support_background": args.model['support_background'],
        "r": args.lse_r,
        "mid_channels": args.mil_mid_channels,
        "gated": args.mil_gated
    }


def get_pretrainde_classifier(args):
    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL]

    if args.method in spec_mth:
        if args.method == constants.METHOD_ACOL:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                acol_drop_threshold=args.acol_drop_threshold,
                large_feature_map=args.acol_large_feature_map,
                scale_in=p.scale_in
            )
        elif args.method == constants.METHOD_SPG:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                large_feature_map=args.spg_large_feature_map,
                scale_in=p.scale_in
            )
        elif args.method == constants.METHOD_ADL:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                adl_drop_rate=args.adl_drop_rate,
                adl_drop_threshold=args.adl_drop_threshold,
                large_feature_map=args.adl_large_feature_map,
                scale_in=p.scale_in
            )
        else:
            raise ValueError
    else:

        aux_params = get_aux_params(args)
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            method='',
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=p.in_channels,
            encoder_depth=encoder_depth,
            scale_in=p.scale_in,
            aux_params=aux_params
        )

    DLLogger.log("PRETRAINED CLASSIFIER `{}` was created. "
                 "Nbr.params: {}".format(model, count_nb_params(model)))
    log = "Arch: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch, p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.img_range
                    )
    DLLogger.log(log)

    path_cl = args.model['folder_pre_trained_cl']
    assert path_cl not in [None, 'None', '']

    msg = "You have asked to set the classifier " \
          " from {} .... [OK]".format(path_cl)
    warnings.warn(msg)
    DLLogger.log(msg)

    if args.task == constants.NEGEV:
        cl_cp = args.negev_ptretrained_cl_cp
        std_cl_args = deepcopy(args)
        std_cl_args.task = constants.STD_CL
        tag = get_tag(std_cl_args, checkpoint_type=cl_cp)

    else:
        tag = get_tag(args)

    if path_cl.endswith(os.sep):
        source_tag = basename(path_cl[:-1])
    else:
        source_tag = basename(path_cl)

    assert tag == source_tag, f'{tag}, {source_tag}'

    if args.method in spec_mth:
        weights = torch.load(join(path_cl, 'model.pt'),
                             map_location=get_cpu_device())
        model.load_state_dict(weights, strict=True)
    else:
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                               map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                              map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

    # if args.model['freeze_cl']:
    #     assert args.task == constants.F_CL
    #     assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']
    #
    #     model.freeze_classifier()
    #     model.assert_cl_is_frozen()

    model.eval()
    return model


def get_model(args, eval=False, eval_path_weights=''):

    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL]
    method = ''
    support_background = args.model['support_background'],

    if args.task == constants.STD_CL:
        aux_params = None
        if args.method in spec_mth:

            if args.method == constants.METHOD_ACOL:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    acol_drop_threshold=args.acol_drop_threshold,
                    large_feature_map=args.acol_large_feature_map,
                    scale_in=p.scale_in
                )
            elif args.method == constants.METHOD_SPG:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    large_feature_map=args.spg_large_feature_map,
                    scale_in=p.scale_in
                )
            elif args.method == constants.METHOD_ADL:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    adl_drop_rate=args.adl_drop_rate,
                    adl_drop_threshold=args.adl_drop_threshold,
                    large_feature_map=args.adl_large_feature_map,
                    scale_in=p.scale_in
                )
            else:
                raise ValueError
        elif args.method == constants.METHOD_MAXMIN:
            aux_params = get_aux_params(args)
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                aux_params=aux_params,
                w=args.maxmin_w,
                dataset_name=args.dataset
            )
        else:
            aux_params = get_aux_params(args)
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                aux_params=aux_params
            )

    elif args.task == constants.F_CL:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.NEGEV:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.SEG:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]
        assert args.seg_mode == constants.BINARY_MODE
        assert classes == 2

        aux_params = None

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            classes=classes)
    else:
        raise NotImplementedError

    DLLogger.log("`{}` was created. Nbr.params: {}".format(
        model,  count_nb_params(model)))
    log = "Arch: {}\n" \
          "task: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "im_rec: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch, args.task, p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.im_rec, args.img_range
                    )
    DLLogger.log(log)
    DLLogger.log(model.get_info_nbr_params())

    path_file = args.model['path_pre_trained']
    if path_file not in [None, 'None']:
        msg = "You have asked to load a specific pre-trained " \
              "model from {} .... [OK]".format(path_file)
        warnings.warn(msg)
        DLLogger.log(msg)
        pre_tr_state = torch.load(path_file, map_location=get_cpu_device())
        model.load_state_dict(pre_tr_state, strict=args.model['strict'])

    path_cl = args.model['folder_pre_trained_cl']
    if path_cl not in [None, 'None', '']:
        assert args.task in [constants.F_CL, constants.NEGEV]

        msg = "You have asked to set the classifier's weights " \
              " from {} .... [OK]".format(path_cl)
        warnings.warn(msg)
        DLLogger.log(msg)

        if args.task == constants.NEGEV:
            cl_cp = args.negev_ptretrained_cl_cp
            std_cl_args = deepcopy(args)
            std_cl_args.task = constants.STD_CL
            tag = get_tag(std_cl_args, checkpoint_type=cl_cp)

        else:
            tag = get_tag(args)

        if path_cl.endswith(os.sep):
            source_tag = basename(path_cl[:-1])
        else:
            source_tag = basename(path_cl)

        assert tag == source_tag, f'{tag}, {source_tag}'

        if args.method in spec_mth:
            weights = torch.load(join(path_cl, 'model.pt'),
                                 map_location=get_cpu_device())
            model.load_state_dict(weights, strict=True)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                   map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                  map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

    if args.model['freeze_cl'] and not eval:
        assert args.task in [constants.F_CL, constants.NEGEV]

        assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']

        model.freeze_classifier()
        model.assert_cl_is_frozen()

    if eval:
        if os.path.isdir(eval_path_weights):
            path = eval_path_weights
        else:
            assert os.path.isdir(args.outd)
            tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
            path = join(args.outd, tag)
        cpu_device = get_cpu_device()

        if args.task == constants.STD_CL:
            if args.method in spec_mth:
                weights = torch.load(join(path, 'model.pt'),
                                     map_location=get_cpu_device())
                model.load_state_dict(weights, strict=True)
            else:
                weights = torch.load(join(path, 'encoder.pt'),
                                     map_location=cpu_device)
                model.encoder.super_load_state_dict(weights, strict=True)

                weights = torch.load(join(path, 'classification_head.pt'),
                                     map_location=cpu_device)
                model.classification_head.load_state_dict(weights, strict=True)

        elif args.task == constants.F_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
            if model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=cpu_device)
                model.reconstruction_head.load_state_dict(weights, strict=True)

        elif args.task == constants.NEGEV:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=cpu_device)
            model.classification_head.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)

        elif args.task == constants.SEG:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
        else:
            raise NotImplementedError

        msg = "EVAL-mode. Reset model weights to: {}".format(path)
        warnings.warn(msg)
        DLLogger.log(msg)

    return model


def standardize_otpmizers_params(optm_dict):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'optn[?]__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    msg = "'optm_dict' must be of type dict. found {}.".format(type(optm_dict))
    assert isinstance(optm_dict, dict), msg
    new_optm_dict = deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith('opt'):
            msg = "'{}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing....".format(k)
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def _get_model_params_for_opt(args, model):
    hparams = deepcopy(args.optimizer)
    hparams = standardize_otpmizers_params(hparams)
    hparams = Dict2Obj(hparams)

    if args.task in [constants.F_CL, constants.SEG]:
        return [
            {'params': model.parameters(), 'lr': hparams.lr}
        ]

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL]

    sp_method = (args.task == constants.STD_CL) and (args.method in spec_mth)

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    architecture = args.model['encoder_name']
    assert architecture in constants.BACKBONES

    if not sp_method:
        _FEATURE_PARAM_LAYER_PATTERNS = {
            'vgg': ['encoder.features.'],  # features
            'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
            'inception': ['encoder.Mixed', 'encoder.Conv2d_1',
                          'encoder.Conv2d_2',
                          'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
        }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(arch):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if arch.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(arch))

    for name, parameter in model.named_parameters():

        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_features.append(parameter)
            elif architecture == constants.RESNET50:
                param_classifiers.append(parameter)
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_classifiers.append(parameter)
            elif architecture == constants.RESNET50:
                param_features.append(parameter)

    return [
            {'params': param_features, 'lr': hparams.lr},
            {'params': param_classifiers,
             'lr': hparams.lr * hparams.lr_classifier_ratio}
    ]


def get_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    hparams = deepcopy(args.optimizer)
    hparams = standardize_otpmizers_params(hparams)
    hparams = Dict2Obj(hparams)

    op_col = {}

    params = _get_model_params_for_opt(args, model)

    if hparams.name_optimizer == "sgd":
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['momentum'] = hparams.momentum
        op_col['dampening'] = hparams.dampening
        op_col['weight_decay'] = hparams.weight_decay
        op_col['nesterov'] = hparams.nesterov

    elif hparams.name_optimizer == "adam":
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['beta1'] = hparams.beta1
        op_col['beta2'] = hparams.beta2
        op_col['weight_decay'] = hparams.weight_decay
        op_col['amsgrad'] = hparams.amsgrad
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "[NOT OK]".format(args.optimizer["name"]))

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == "step":
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "cosine":
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['T_max'] = hparams.T_max
            op_col['eta_min'] = hparams.eta_min
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mystep":
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=hparams.step_size,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch,
                min_lr=hparams.min_lr)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mycosine":
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer,
                coef=hparams.coef,
                max_epochs=hparams.max_epochs,
                min_lr=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['coef'] = hparams.coef
            op_col['max_epochs'] = hparams.max_epochs
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "multistep":
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['milestones'] = hparams.milestones
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                hparams.name_lr_scheduler))
    else:
        lrate_scheduler = None

    DLLogger.log("Optimizer:\n{}".format(format_dict_2_str(op_col)))

    return optimizer, lrate_scheduler
