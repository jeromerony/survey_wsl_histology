import os
import sys
from os.path import dirname, abspath, join
from copy import deepcopy
import pickle as pkl
import subprocess
import math


from texttable import Texttable
import numpy as np
import torch
from tqdm import tqdm as tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.meter import AverageValueMeter
from dlib.configure import constants
from dlib.utils.reproducibility import set_seed
from dlib.utils.shared import frmt_dict_mtr_str
from dlib.utils.tools import compute_auc

from dlib.metrics import average as avg_metrics

from dlib.cams import BasicAccSeedsMeter
from dlib.cams import AccSeedsmeter
from dlib.cams import AccSeeds
from dlib.cams.selflearning import GetPseudoMaskSLFCAMS


import dlib.dllogger as DLLogger


__all__ = [
    'get_cl_mtr_meter',
    'get_seg_mtr_meter',
    'get_losses_meter',
    'get_selection_metrics',
    'TrainEpoch',
    'ValidEpoch',
    'FastSeedCamEvalStdCl'
]


def get_cl_mtr_meter(metrics):
    return {
            metric.__name__: AverageValueMeter() for metric in
            metrics
        }


def get_seg_mtr_meter(metrics):
    return {
            metric.__name__: AverageValueMeter() for metric in
            metrics
        }


def get_losses_meter(n_holder):
    return {
            namel: AverageValueMeter() for namel in n_holder
        }


def get_selection_metrics(task, pxl_sup):
    if task == constants.STD_CL:
        return ['mean_accu_out']

    if (task == constants.F_CL) and (
            pxl_sup in [constants.VOID, constants.SELF_LEARNED]):
        return ['mean_accu_out']

    if (task == constants.F_CL) and (pxl_sup == constants.ORACLE):
        return ['mean_accu_out', avg_metrics.MeanIoU().__name__]

    if task == constants.SEG:
        return [avg_metrics.MeanIoU().__name__]

    raise NotImplementedError


class _Epoch:

    def __init__(self,
                 model,
                 loss,
                 cl_mtrc,
                 seg_mtrc,
                 task,
                 pxl_sup,
                 support_background,
                 stage_name,
                 subset,
                 select_metrics,
                 fdout,
                 inference,
                 visualiazer,
                 multi_label_flag,
                 store_per_sample,
                 store_fig_cams,
                 device='cpu',
                 verbose=True
                 ):
        self.model = model
        self.loss = loss
        self.cl_mtrc = cl_mtrc
        self.seg_mtrc = seg_mtrc
        self.task = task
        self.pxl_sup = pxl_sup
        self.support_background = support_background
        self.stage_name = stage_name
        self.subset = subset
        self.fdout = fdout
        self.inference = inference
        self.visualiazer = visualiazer
        self.multi_label_flag = multi_label_flag
        self.store_per_sample = store_per_sample
        self.store_fig_cams = store_fig_cams
        self.verbose = verbose
        self.device = device

        self.sl_mask_builder: GetPseudoMaskSLFCAMS = None

        self.cl_mtr_meter = None
        self.seg_mtr_meter = None
        self.losses_meter = None

        self.perm_cl_mtr_meter = None
        self.perm_seg_mtr_meter = None
        self.perm_losses_meter = None
        self.create_permanent_meters()

        self.seed = int(os.environ["MYSEED"])
        self.counter = 0
        self.epoch = 0
        self.dec_prec_mtric = 6
        self.dec_prec_loss = 3

        self.best_model = None
        self.best_metrics_val = None
        self.select_metrics = select_metrics
        self.best_epoch = None

    def set_sl_mask_builder(self, sl_mask_builder):
        self.sl_mask_builder = sl_mask_builder

    @property
    def fd_inference(self):
        return join(self.fdout, 'inference') if self.fdout else None

    def create_permanent_meters(self):
        self.perm_cl_mtr_meter = get_cl_mtr_meter(
            self.cl_mtrc.metrics
        ) if self.cl_mtrc is not None else None

        self.perm_seg_mtr_meter = get_seg_mtr_meter(
            self.seg_mtrc.metrics
        ) if self.seg_mtrc is not None else None

        self.perm_losses_meter = get_losses_meter(
            self.loss.n_holder
        ) if self.loss is not None else None

    def update_perm_meters(self):
        if self.perm_cl_mtr_meter is not None:
            for k in self.perm_cl_mtr_meter.keys():
                self.perm_cl_mtr_meter[k].add(self.cl_mtr_meter[k].avg)

        if self.perm_seg_mtr_meter is not None:
            for k in self.perm_seg_mtr_meter.keys():
                self.perm_seg_mtr_meter[k].add(self.seg_mtr_meter[k].avg)

        if self.perm_losses_meter is not None:
            for k in self.perm_losses_meter.keys():
                self.perm_losses_meter[k].add(self.losses_meter[k].avg)

    def plot_perm_meter(self, perm_meter: dict, tag):
        nrows = 4
        ks = list(perm_meter.keys())
        ncols = math.ceil(len(ks) / float(nrows))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = perm_meter[ks[t]].values
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=5)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].grid(True)
                axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                t += 1

        fig.suptitle(tag, fontsize=7)
        plt.tight_layout()

        self.check_fdout()
        fig.savefig(join(self.fdout, '{}.png'.format(tag)),
                    bbox_inches='tight', dpi=300)

    def plot_all_perm_meters(self):
        if self.perm_cl_mtr_meter is not None:
            self.plot_perm_meter(self.perm_cl_mtr_meter, 'cl_metrics')

        if self.perm_seg_mtr_meter is not None:
            self.plot_perm_meter(self.perm_seg_mtr_meter, 'seg_metrics')

        if self.perm_losses_meter is not None:
            self.plot_perm_meter(self.perm_losses_meter, 'losses')

    def automatic_plot_perm_meters(self):
        if (self.epoch % 20) == 0:
            self.plot_all_perm_meters()

    def flush_meters(self):
        self.cl_mtr_meter = get_cl_mtr_meter(
            self.cl_mtrc.metrics
        ) if self.cl_mtrc is not None else None

        self.seg_mtr_meter = get_seg_mtr_meter(
            self.seg_mtrc.metrics
        ) if self.seg_mtrc is not None else None

        self.losses_meter = get_losses_meter(
            self.loss.n_holder
        ) if self.loss is not None else None

    def meter_summary(self, meter):
        return {k: meter[k].avg for k in meter.keys()}

    def _to_device(self):
        pass

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y_global, y_mask, true_mask, raw_img):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def store_minibatch_results(self, id_s, loss, y_g_pred, cams_low, cams,
                                mask_pred, y_global, im_recon):
        self.check_fd_inference()

        if cams is not None:
            cam = self.get_cam(cams, y_global, normalize=True)
        else:
            cam = None

        # todo: add function get_low_cam. and store it.
        with open(join(self.fd_inference, "{}-results.pkl".format(id_s[0])),
                  'wb') as fout:
            pkl.dump({
                "id": id_s[0],
                "loss": loss.clone().cpu().detach().item(),
                "loss_l_holder": [l.clone().cpu().detach().item() for l in
                                  self.loss.l_holder],
                "loss_n_holder": [l for l in self.loss.n_holder],
                "y_g_pred": y_g_pred.clone().cpu().detach().numpy() if
                y_g_pred is not None else None,
                "low_cams": None,
                "cam": cam.clone().cpu().detach().squeeze().numpy(
                ) if cam is not None else None,
                "mask_pred": mask_pred.clone().cpu().detach().numpy() if
                mask_pred is not None else None,
                "support_background": self.support_background
            }, fout, protocol=pkl.HIGHEST_PROTOCOL)

    def get_cam(self, cams, y_global, normalize, task=None):
        if task is None:
            task = self.task

        if task in [constants.SEG, constants.STD_CL, constants.F_CL]:
            assert isinstance(cams, torch.Tensor)
            assert cams.ndim == 4
            assert cams.shape[0] == 1
        else:
            raise NotImplementedError

        if self.multi_label_flag:
            raise NotImplementedError

        if task == constants.SEG:
            if normalize:
                cams_n = cams.softmax(dim=1)
            else:
                cams_n = cams
            return cams_n[:, 1, :, :].unsqueeze(1)

        if task == constants.STD_CL:
            assert y_global.numel() == 1
            assert cams.shape[1] > 1

            if normalize:
                cams_n = cams.softmax(dim=1)
            else:
                cams_n = cams
            # ----
            # index = y_global.view(
            #     cams.shape[0], 1, 1, 1).expand(
            #     cams.shape[0], 1, cams.shape[2], cams.shape[3]).clone()
            # if self.support_background:
            #     index = index + 1
            #
            # return cams_n.gather(dim=1, index=index)
            # -----------------------------------
            index = y_global[0]

            if self.support_background:
                index = index + 1
            return cams_n[0, index, :, :].unsqueeze(0).unsqueeze(0)

        if task == constants.F_CL:
            if normalize:
                cams_n = torch.softmax(cams, dim=1)
            else:
                cams_n = cams

            if self.multi_label_flag:
                # todo
                raise NotImplementedError
            else:
                return cams_n[:, 1, :, :].unsqueeze(1)

        raise NotImplementedError

    def visualize_cams(self, id_s, cams_low, cams_inter, cams, mask_pred,
                       y_global, true_mask, raw_img, im_recon, seeds):
        self.check_fd_inference()

        seed = seeds

        cam = self.get_cam(cams, y_global, normalize=True)
        cam_raw = self.get_cam(cams, y_global, normalize=False)

        mask_pred_inter = None
        if cams_low is not None:
            task = constants.STD_CL
            cam_low = self.get_cam(cams_low, y_global, normalize=True,
                                   task=task)
            cam_low_raw = self.get_cam(cams_low, y_global, normalize=False,
                                       task=task)
            mask_pred_inter = self.predict_mask(cams_inter, y_global,
                                                task=task)

            cam_inter = self.get_cam(cams_inter, y_global, normalize=True,
                                     task=task)
            cam_inter_raw = self.get_cam(cams_inter, y_global, normalize=False,
                                         task=task)
        else:
            cam_low, cam_low_raw = None, None
            task = constants.SEG
            cam_inter = self.get_cam(cams_inter, y_global, normalize=True,
                                     task=task)
            cam_inter_raw = self.get_cam(cams_inter, y_global, normalize=False,
                                         task=task)

        img_pil = Image.fromarray(
            raw_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        self.visualiazer(
            img_pil=img_pil,
            true_mask=true_mask.squeeze().cpu().numpy(),
            mask_pred_inter=mask_pred_inter.squeeze().cpu().numpy(),
            pred_mask=mask_pred.clone().cpu().detach().squeeze().numpy(),
            cam_low=cam_low.clone().cpu().detach().squeeze().numpy() if
            cam_low is not None else None,
            cam_low_raw=cam_low_raw.clone().cpu().detach().squeeze().numpy()
            if cam_low_raw is not None else None,
            cam_inter=cam_inter.clone().cpu().detach().squeeze().numpy(),
            cam_inter_raw=cam_inter_raw.clone().cpu().detach().squeeze().numpy(),
            cam=cam.clone().cpu().detach().squeeze().numpy(),
            cam_raw=cam_raw.clone().cpu().detach().squeeze().numpy(),
            im_recon=im_recon.clone().cpu().detach().squeeze().numpy() if
            im_recon is not None else None,
            seed=seed.clone().cpu().detach().squeeze().numpy() if
            seed is not None else None,
            basefile='{}'.format(id_s[0])
        )

    def on_mini_batch_end(self, id_s, loss, y_g_pred, cams_low, cams_inter,
                          cams,  mask_pred, y_global, true_mask, raw_img,
                          im_recon, seeds):
        if self.inference:
            assert len(id_s) == 1
            if self.store_per_sample:
                self.store_minibatch_results(
                    id_s, loss, y_g_pred, cams_low, cams, mask_pred,
                    y_global, im_recon)
            if self.store_fig_cams:
                self.visualize_cams(
                    id_s, cams_low, cams_inter, cams, mask_pred, y_global,
                    true_mask, raw_img, im_recon, seeds)

    def on_epoch_end(self):

        if self.cl_mtr_meter is not None:
            DLLogger.log("{}[{}](CL/ep: {}): {}".format(
                self.subset,
                self.stage_name,
                self.epoch,
                frmt_dict_mtr_str(self.meter_summary(self.cl_mtr_meter),
                                  dec_prec=self.dec_prec_mtric,
                                  seps=' ' * 2)))

        if self.seg_mtr_meter is not None:
            DLLogger.log("{}[{}](SEG/ep: {}): {}".format(
                self.subset,
                self.stage_name,
                self.epoch,
                frmt_dict_mtr_str(self.meter_summary(self.seg_mtr_meter),
                                  dec_prec=self.dec_prec_mtric,
                                  seps=' ' * 2)))

        if self.losses_meter is not None:
            DLLogger.log("{}[{}](LOSS/ep: {}): {}".format(
                self.subset,
                self.stage_name,
                self.epoch,
                frmt_dict_mtr_str(self.meter_summary(self.losses_meter),
                                  dec_prec=self.dec_prec_loss,
                                  seps=' ' * 2)))

        self.update_perm_meters()

        if self.select_metrics:
            self.select_model()

        self.loss.update_t()

        if hasattr(self, 'lr_scheduler'):
            if self.lr_scheduler:
                self.lr_scheduler.step()

        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.automatic_plot_perm_meters()
        DLLogger.flush()

    def check_fdout(self):
        assert self.fdout
        if not os.path.isdir(self.fdout):
            os.makedirs(self.fdout)

    def check_fd_inference(self):
        self.check_fdout()
        if self.fd_inference and (not os.path.isdir(self.fd_inference)):
            os.makedirs(self.fd_inference)

    def store_perm_meters(self):
        self.check_fdout()

        with open(join(self.fdout, "{}{}-{}-perm-meters.pkl".format(
                'inference-' if self.inference else '', self.subset,
                self.stage_name
        )), 'wb') as fout:
            pkl.dump({
                "perm_cl_mtr_meter": self.perm_cl_mtr_meter,
                "perm_seg_mtr_meter": self.perm_seg_mtr_meter,
                "perm_losses_meter": self.perm_losses_meter
            }, fout, protocol=pkl.HIGHEST_PROTOCOL)

    def store_extra_info(self):
        self.check_fdout()
        with open(join(self.fdout, "{}{}-{}-extra-info.pkl".format(
            'inference-' if self.inference else '', self.subset, self.stage_name
        )), 'wb') as fout:
            pkl.dump({
                "best_epoch": self.best_epoch
            }, fout, protocol=pkl.HIGHEST_PROTOCOL)

    def compress_fdout(self):
        if not self.fdout:
            return 0

        fd = self.fdout.split(os.sep)[-1]
        if fd == '':
            fd = self.fdout.split(os.sep)[-2]
        cmdx = [
            "cd {} ".format(self.fdout),
            "cd .. ",
            "tar -cf {}.tar.gz {} ".format(fd, fd),
            "rm -r {} ".format(fd)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))

    def on_end(self):
        self.store_perm_meters()
        self.store_extra_info()
        self.plot_all_perm_meters()

        self.flush()

    def flush(self):
        if self.select_metrics:
            msg = ', '.join(["({}, {})".format(
                p[0], "{0:.{1}f}".format(
                    p[1], self.dec_prec_mtric)) for p in zip(
                self.select_metrics, self.best_metrics_val)])

            DLLogger.log("BEST MODEL: {}[{}](ep: {}): {}".format(
                self.subset,
                self.stage_name,
                self.best_epoch,
                msg)
            )

        DLLogger.flush()

    def is_best_model(self):
        assert self.select_metrics
        holder = dict()
        for mtr_meter in [self.cl_mtr_meter, self.seg_mtr_meter]:
            if mtr_meter is None:
                continue
            holder.update(
                {k: mtr_meter[k].avg for k in mtr_meter.keys()
                 if k in self.select_metrics}
            )

        metrics_val = sum([0.0] + [holder[k] for k in holder.keys()])
        best_metrics_val = sum(self.best_metrics_val) if \
            self.best_metrics_val is not None else 0.0
        if metrics_val >= best_metrics_val:
            p_now = [holder[k] for k in self.select_metrics]
            p_before = self.best_metrics_val if \
                self.best_metrics_val is not None else \
                [0.0 for _ in self.select_metrics]

            msg = ', '.join(["({}, {})".format(
                p[0], "{0:.{1}f}".format(
                    p[1], self.dec_prec_mtric)) for p in zip(
                self.select_metrics, p_before)])
            msg = "BEFORE: {}, NOW: ".format(msg)
            msg += ', '.join(["({}, {})".format(
                p[0], "{0:.{1}f}".format(
                    p[1], self.dec_prec_mtric)) for p in zip(
                self.select_metrics,  p_now)])

            DLLogger.log("SELECT MODEL: {}[{}](ep: {}): {}".format(
                self.subset,
                self.stage_name,
                self.epoch,
                msg)
            )

            self.best_metrics_val = [holder[k] for k in self.select_metrics]
            self.best_epoch = self.epoch
            return True
        return False

    def select_model(self):
        if (self.best_model is None) or self.is_best_model():
            self.best_model = deepcopy(self.model.state_dict())

    def random(self):
        if self.stage_name == constants.STGS_TR:
            self.counter = self.counter + 1
            self.seed = self.seed + self.counter
            # todo: add set_seed(seed=self.seed, verbose=False)
        else:  # vl, tst
            set_seed(seed=self.seed, verbose=False)

    def update_loss_meter(self, loss):
        if self.losses_meter is None:
            return 0

        for lsv, lsn in zip(self.loss.l_holder, self.loss.n_holder):
            self.losses_meter[lsn].add(lsv.clone().cpu().detach().item())

        self.losses_meter[self.loss.__name__].add(
            loss.clone().cpu().detach().item())

    def update_cl_mtr_meter(self, y_g_pred, y_global, y_g_pred_bdg=None):
        if self.cl_mtr_meter is None:
            return 0

        for metric in self.cl_mtrc.metrics:
            if "bdg" in metric.__name__:
                assert y_g_pred_bdg is not None
                metric_value = metric(y_g_pred_bdg,
                                      y_global).cpu().detach().item()
            else:
                metric_value = metric(y_g_pred,
                                      y_global).cpu().detach().item()

            self.cl_mtr_meter[metric.__name__].add(metric_value)

    def update_seg_mt_meter(self, mask_pred, true_mask):
        if self.seg_mtr_meter is None:
            return 0

        for metric in self.seg_mtrc.metrics:
            metric_value = metric(mask_pred,
                                  true_mask).cpu().detach().item()
            self.seg_mtr_meter[metric.__name__].add(metric_value)

    def fast_get_std_cam(self, cams_inter, y_global):
        assert cams_inter.shape[1] > 1
        assert cams_inter.shape[0] == 1

        index = y_global

        if self.support_background:
            index = index + 1
        return cams_inter[0, index, :, :].unsqueeze(0).unsqueeze(0)

    def build_seeds_from_fcams(self, cams_inter, y_global) -> torch.Tensor:

        assert not self.multi_label_flag  # todo
        assert cams_inter.shape[0] == y_global.shape[0]

        bsz, c, h, w = cams_inter.shape
        seeds = torch.zeros((bsz, h, w), device=self.device,
                            requires_grad=False, dtype=torch.long)

        with torch.no_grad():
            for i in range(bsz):
                y = y_global[i]
                cams = cams_inter[i].unsqueeze(0)
                pulled_cam = self.fast_get_std_cam(cams, y)
                seeds[i] = self.sl_mask_builder(pulled_cam).squeeze(0)

            return seeds

    def unpack_std_cl(self, x, y_global, true_mask):

        cams_low, cams_inter, cl_logits = self.model.forward(x=x)
        # loss:
        loss = self.loss(epoch=self.epoch, cl_logits=cl_logits, glabel=y_global)
        self.update_loss_meter(loss)

        # metrics:
        # 1. cl
        # todo: multi-label.
        y_g_pred = cl_logits.argmax(dim=1, keepdim=False)
        self.update_cl_mtr_meter(y_g_pred, y_global)

        # 2. seg
        mask_pred = self.predict_mask(cams_inter, y_global)
        self.update_seg_mt_meter(mask_pred, true_mask)

        im_recon = None
        seeds = None

        z_out = [loss, y_g_pred, cams_low, cams_inter, cams_inter, mask_pred,
                 im_recon, seeds]

        return z_out

    def unpack_seg(self, x, true_mask):
        masks_logits = self.model.forward(x=x)

        loss = self.loss(epoch=self.epoch, masks_logits=masks_logits,
                         true_mask=true_mask)
        self.update_loss_meter(loss)

        mask_pred = self.predict_mask(masks_logits, None)

        self.update_seg_mt_meter(mask_pred, true_mask)

        im_recon = None
        seeds = None
        y_g_pred, cams_low = None, None

        z_out = [loss, y_g_pred, cams_low, masks_logits, masks_logits,
                 mask_pred, im_recon, seeds]

        return z_out

    def unpacke_f_cl(self, x, y_global, y_mask, true_mask, raw_img):
        out = self.model.forward(x=x, glabel=y_global)
        cams_low, cams_inter, fcams, cl_logits, im_recon = out

        seeds = self.build_seeds_from_fcams(cams_inter, y_global)

        loss = self.loss(
            epoch=self.epoch,
            cams_low=cams_low,
            cams_inter=cams_inter,
            fcams=fcams,
            cl_logits=cl_logits,
            masks_logits=None,
            glabel=y_global,
            true_mask=true_mask,
            raw_img=raw_img,
            x_in=self.model.x_in,
            im_recon=im_recon,
            seeds=seeds
        )

        self.update_loss_meter(loss)

        y_g_pred = cl_logits.argmax(dim=1, keepdim=False)
        self.update_cl_mtr_meter(y_g_pred, y_global)

        mask_pred = self.predict_mask(fcams, y_global)
        self.update_seg_mt_meter(mask_pred, true_mask)

        z_out = [loss, y_g_pred, cams_low, cams_inter, fcams, mask_pred,
                 im_recon, seeds]

        return z_out

    def unpacke_output(self, x, y_global, y_mask, true_mask, raw_img):
        tsk = self.task

        if tsk == constants.STD_CL:
            return self.unpack_std_cl(x=x, y_global=y_global,
                                      true_mask=true_mask)

        elif tsk == constants.SEG:
            return self.unpack_seg(x=x, true_mask=true_mask)

        elif tsk == constants.F_CL:
            return self.unpacke_f_cl(x, y_global, y_mask, true_mask, raw_img)
        else:
            raise NotImplementedError

    def predict_mask(self, tensor, glabel, task=None):
        if isinstance(tensor, torch.Tensor):
            assert tensor.ndim == 4  # bsz, c, h, w. c > 1

        c_task = self.task if task is None else task

        if c_task == constants.SEG:
            assert tensor.shape[1] > 1
            return tensor.argmax(dim=1, keepdim=False).float()

        if self.multi_label_flag:  # todo
            raise NotImplementedError

        if c_task == constants.STD_CL:
            assert tensor.shape[1] > 1
            if self.support_background:
                return ((tensor.argmax(dim=1, keepdim=False) - 1.) ==
                        glabel.view(-1, 1, 1)).float()
            else:
                return (tensor.argmax(dim=1, keepdim=False) ==
                        glabel.view(-1, 1, 1)).float()

        if c_task == constants.F_CL:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 4  # bsz, c, h, w. c > 1
            assert tensor.shape[1] > 1

            assert not self.multi_label_flag

            # todo: deal with multi-label
            if self.support_background:
                return tensor.argmax(dim=1, keepdim=False).float()
            else:
                return (tensor > 0.5).float().squeeze(1)

        raise NotImplementedError

    def predict_multi_mask(self, tensor):
        # show all the classes at once. simply apply argmax to the cams. this
        # will help understand what the zero class predicted in standard
        # cams, are they really background or other classes to show how
        # inconsistent they are.
        # todo
        raise NotImplementedError

    def run(self, dataloader, epoch):
        self.random()

        self.epoch = epoch
        self.on_epoch_start()
        self.flush_meters()

        with tqdm(dataloader,
                  desc="{}[{}]".format(self.subset, self.stage_name),
                  file=sys.stdout,
                  ncols=constants.NCOLS, disable=not self.verbose) as iterator:
            for comp in iterator:
                self.random()
                id_s, data, raw_img, mask, true_mask, label, _, _ = comp
                mask = mask.squeeze(1)
                true_mask = true_mask.squeeze(1)

                x, y_global = data.to(self.device), label.to(self.device)
                y_mask = mask.to(self.device)
                true_mask = true_mask.to(self.device)
                # todo: adjust the mask depending on the support of the
                #  background support.

                z = self.batch_update(x, y_global, y_mask, true_mask, raw_img)

                loss = z[0]
                y_g_pred = z[1]
                cams_low = z[2]
                cams_inter = z[3]
                cams = z[4]
                mask_pred = z[5]
                im_recon = z[6]
                seeds = z[7]

                self.on_mini_batch_end(
                    id_s, loss, y_g_pred, cams_low, cams_inter, cams, mask_pred,
                    y_global, true_mask, raw_img, im_recon, seeds)
                # todo: track meter: average time forward().

        self.on_epoch_end()

        return 0


class TrainEpoch(_Epoch):

    def __init__(self,
                 model,
                 loss,
                 cl_mtrc,
                 seg_mtrc,
                 task,
                 pxl_sup,
                 support_background,
                 subset,
                 fdout,
                 optimizer,
                 lr_scheduler,
                 multi_label_flag,
                 device='cpu',
                 verbose=True
                 ):
        super().__init__(
            model=model,
            loss=loss,
            cl_mtrc=cl_mtrc,
            seg_mtrc=seg_mtrc,
            task=task,
            pxl_sup=pxl_sup,
            support_background=support_background,
            stage_name=constants.STGS_TR,
            subset=subset,
            select_metrics=None,
            fdout=fdout,
            inference=False,
            visualiazer=None,
            multi_label_flag=multi_label_flag,
            store_per_sample=False,
            store_fig_cams=False,
            device=device,
            verbose=verbose
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def on_epoch_start(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.model.train()

    def batch_update(self, x, y_global, y_mask, true_mask, raw_img):
        self.optimizer.zero_grad()

        z = self.unpacke_output(x, y_global, y_mask, true_mask, raw_img)

        loss = z[0]

        loss.backward()
        self.optimizer.step()
        return z


class ValidEpoch(_Epoch):

    def __init__(self,
                 model,
                 loss,
                 cl_mtrc,
                 seg_mtrc,
                 task,
                 pxl_sup,
                 support_background,
                 subset,
                 select_metrics,
                 fdout,
                 inference,
                 visualiazer,
                 multi_label_flag,
                 store_per_sample,
                 store_fig_cams,
                 device='cpu',
                 verbose=True
                 ):
        super().__init__(
            model=model,
            loss=loss,
            cl_mtrc=cl_mtrc,
            seg_mtrc=seg_mtrc,
            task=task,
            pxl_sup=pxl_sup,
            support_background=support_background,
            stage_name=constants.STGS_EV,
            subset=subset,
            select_metrics=select_metrics,
            fdout=fdout,
            inference=inference,
            visualiazer=visualiazer,
            multi_label_flag=multi_label_flag,
            store_per_sample=store_per_sample,
            store_fig_cams=store_fig_cams,
            device=device,
            verbose=verbose
        )

    def on_epoch_start(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

    def batch_update(self, x, y_global, y_mask, true_mask, raw_img):
        with torch.no_grad():
            return self.unpacke_output(x, y_global, y_mask, true_mask, raw_img)


class FastSeedCamEvalStdCl:
    def __init__(self,
                 model,
                 seed_cam_evaluer: AccSeeds,
                 meters: AccSeedsmeter,
                 fdout,
                 multi_label_flag,
                 support_background,
                 device,
                 task,
                 subset
                 ):

        self.model = model
        self.device = device
        self.seed_cam_evaluer = seed_cam_evaluer
        self.meters = meters

        assert task == constants.STD_CL
        self.task = task
        self.subset = subset

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        if not os.path.isdir(fdout):
            os.makedirs(fdout)

        self.fdout = fdout
        self.seed = int(os.environ["MYSEED"])

    def flush_meters(self):
        self.meters.flush()

    def random(self):
        set_seed(seed=self.seed, verbose=False)

    def on_epoch_start(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()

    def get_cam(self, cams, y_global):

        assert self.task == constants.STD_CL
        assert isinstance(cams, torch.Tensor)
        assert cams.ndim == 4
        assert cams.shape[0] == 1

        assert y_global.numel() == 1
        assert cams.shape[1] > 1

        index = y_global[0]

        if self.support_background:
            index = index + 1
        return cams[0, index, :, :].unsqueeze(0).unsqueeze(0)

    def batch_forward(self, x, y_global, true_mask):
        with torch.no_grad():
            cams_low, cams_inter, cl_logits = self.model.forward(x=x)
            cam = self.get_cam(cams=cams_inter, y_global=y_global)
            acc_forg, acc_backg, _ = self.seed_cam_evaluer(cam, true_mask)
            self.meters(acc_forg, acc_backg)

    def plot_meters(self):
        if self.seed_cam_evaluer.max_sn > 5000:
            nrows = 2
            bounce = [0, 1]
            sharex = False
            label_outer = False
        else:
            nrows = 1
            bounce = [0]
            sharex = True
            label_outer = True
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=sharex,
                                 sharey=True, squeeze=False)
        meters = self.meters.meters_cp()

        for z in bounce:
            for i, k in enumerate(meters.keys()):
                meter: BasicAccSeedsMeter = meters[k]
                lim = -1 if z == 0 else 500
                mean = meter.mean.cpu().numpy()[:lim]
                _min = meter.min.cpu().numpy()[:lim]
                _max = meter.max.cpu().numpy()[:lim]
                std = torch.sqrt(meter.emvar).cpu().numpy()[:lim]

                x = self.seed_cam_evaluer.n_seeds[:lim]

                # axes[0, i].plot(x, _min, color='tab:blue', label='Min.',
                #                 alpha=0.2)
                # axes[0, i].plot(x, _max, color='tab:green', label='Max.',
                #                 alpha=0.2)

                # todo: check what's wrong with x ticks when maxns is 40000
                axes[z, i].plot(x, mean, color='tab:orange', label='Avg.')
                axes[z, i].fill_between(x, mean - std, mean + std, alpha=0.2,
                                        color='tab:orange')

                subt = '{}. AUC: {:.3f} %'.format(k,
                                                  compute_auc(mean, mean.size))
                axes[z, i].set_title(subt, fontsize=5)
                axes[z, i].xaxis.set_tick_params(labelsize=4)
                axes[z, i].yaxis.set_tick_params(labelsize=4)
                axes[z, i].grid(True)
                axes[z, i].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[z, i].legend(loc='lower left', fancybox=True, shadow=True,
                                  prop={'size': 5})
                axes[z, i].set_xlabel('Nbr. clicks', fontsize=4)
                axes[z, i].set_ylabel('Clicks accuracy (%)', fontsize=4)
        if label_outer:
            for ax in axes.flat:
                ax.label_outer()

        fig.suptitle('Subset: {}. Task: {}. Kernel size: {}x{}. '
                     'Max_NS: {}'.format(
                      self.subset, self.task, self.seed_cam_evaluer.ksz,
                      self.seed_cam_evaluer.ksz, self.seed_cam_evaluer.max_sn),
                     fontsize=5)
        plt.tight_layout()
        tag = '{}-{}x{}.png'.format(
            self.seed_cam_evaluer.max_sn, self.seed_cam_evaluer.ksz,
            self.seed_cam_evaluer.ksz)
        fig.savefig(join(self.fdout, tag), bbox_inches='tight', dpi=200)

    def store_meter(self):
        tag = '{}-{}x{}'.format(
            self.seed_cam_evaluer.max_sn, self.seed_cam_evaluer.ksz,
            self.seed_cam_evaluer.ksz)
        with open(join(self.fdout, '{}.pkl'.format(tag)), 'wb') as fout:
            pkl.dump(self.sumup_meter(), fout, protocol=pkl.HIGHEST_PROTOCOL)

    def sumup_meter(self):
        out = {'max_sn': self.seed_cam_evaluer.max_sn,
               'ksz': self.seed_cam_evaluer.ksz,
               'n_seeds': self.seed_cam_evaluer.n_seeds,
               'meters': self.meters.meters_cp()
               }
        return out

    def dump_txt_sorted_meter(self):
        meters = self.meters.meters_cp()
        x = self.seed_cam_evaluer.n_seeds
        l_headers = []
        l_meta_headers = []
        l_f_rows = []
        l_rows = []

        for i, k in enumerate(meters.keys()):
            meter: BasicAccSeedsMeter = meters[k]
            mean = meter.mean.cpu().numpy()
            _min = meter.min.cpu().numpy()
            _max = meter.max.cpu().numpy()
            std = torch.sqrt(meter.emvar).cpu().numpy()

            l_meta_headers.extend(['STR', '*', k, '*', 'END'])
            l_headers.extend(['nbr_pxl', 'mean', 'std', 'min', 'max'])
            l_f_rows.extend(['i', 'f', 'f', 'f', 'f'])

            rows = []
            for u, r in enumerate(x):
                rows.append((r, mean[u], std[u], _min[u], _max[u]))

            rows = sorted(rows, key=lambda xx: xx[1], reverse=True)

            if not l_rows:
                l_rows = deepcopy(rows)
            else:
                l_rows = [l_rows[rr] + rows[rr] for rr in range(len(rows))]

        t = Texttable()
        t.set_max_width(500)

        t.set_cols_dtype(l_f_rows)
        l_rows = [tuple(l_meta_headers)] + [tuple(l_headers)] + l_rows
        t.add_rows(l_rows)
        tag = '{}-{}x{}.txt'.format(
            self.seed_cam_evaluer.max_sn, self.seed_cam_evaluer.ksz,
            self.seed_cam_evaluer.ksz)
        path_txt = join(self.fdout, tag)

        with open(path_txt, "w") as fout:
            fout.write('Subset: {}. Task: {}. Kernel size: {}x{}.'
                       'Max_NS: {}\n\n'.format(
                        self.subset, self.task, self.seed_cam_evaluer.ksz,
                        self.seed_cam_evaluer.ksz,
                        self.seed_cam_evaluer.max_sn))

            print(t.draw(), file=fout)

    def on_epoch_end(self):
        self.plot_meters()
        self.store_meter()
        self.dump_txt_sorted_meter()

    def run(self, dataloader):
        self.random()

        self.on_epoch_start()
        self.flush_meters()

        with tqdm(dataloader,
                  desc="DIAG-STD-SEED: {}x{}".format(
                      self.seed_cam_evaluer.ksz, self.seed_cam_evaluer.ksz),
                  file=sys.stdout,
                  ncols=constants.NCOLS, disable=False) as iterator:
            for comp in iterator:
                self.random()
                id_s, data, raw_img, mask, true_mask, label, _, _ = comp

                assert data.shape[0] == 1
                x, y_global = data.to(self.device), label.to(self.device)
                true_mask = true_mask.to(self.device)

                self.batch_forward(x, y_global, true_mask)

        self.on_epoch_end()

        return self.sumup_meter()
