import sys
import os
import time
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

sys.path.append(
    join(root_dir,
         "crf/crfwrapper/bilateralfilter/build/lib.linux-x86_64-3.7")
)

from bilateralfilter import bilateralfilter, bilateralfilter_batch


__all__ = ['DenseCRFLoss']


class DenseCRFLossFunction(Function):
    
    @staticmethod
    @custom_fwd
    def forward(ctx,
                images,
                segmentations,
                sigma_rgb,
                sigma_xy
                ):
        torch.cuda.synchronize()
        ctx.save_for_backward(segmentations)

        device = segmentations.device

        n, k, h, w = segmentations.shape
        ctx.N, ctx.K, ctx.H, ctx.W = n, k, h, w
        ctx.N_FP32 = torch.tensor([n], dtype=torch.float,  device=device)

        # ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)
        # segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        # ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations_np = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations_np.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations_np, AS, ctx.N, ctx.K, ctx.H,
                              ctx.W, sigma_rgb, sigma_xy)

        AS = torch.from_numpy(AS).to(device)
        densecrf_loss = - (segmentations.detach().flatten() * AS).sum().view(1)
        densecrf_loss /= ctx.N_FP32
        densecrf_loss.requires_grad = True
        ctx.AS = AS.contiguous().view(ctx.N, ctx.K, ctx.H, ctx.W)

        return densecrf_loss
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_segmentation = - 2 * grad_output * ctx.AS / ctx.N_FP32
        return None, grad_segmentation, None, None, None


class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        """
        Init. function.
        :param weight: float. It is Lambda for the crf loss.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        :param scale_factor: float. ratio to scale the image and
        segmentation. Helpful to control the computation (speed) / precision.
        """
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations):
        """
        Forward loss.
        Image and segmentation are scaled with the same factor.

        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU. C==3.
        :param segmentations: softmaxed logits. cuda.
        :return: loss score (scalar).
        """
        assert images.ndim == 4
        assert images.shape[1] == 3

        scaled_images = F.interpolate(images,
                                      scale_factor=self.scale_factor,
                                      mode='nearest',
                                      recompute_scale_factor=False
                                      )
        scaled_segs = F.interpolate(segmentations,
                                    scale_factor=self.scale_factor,
                                    mode='bilinear',
                                    recompute_scale_factor=False,
                                    align_corners=False)

        val = self.weight * DenseCRFLossFunction.apply(
            scaled_images,
            scaled_segs,
            self.sigma_rgb,
            self.sigma_xy * self.scale_factor
        )

        return val

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


def test_DenseCRFLoss():
    import time

    from dlib.utils.reproducibility import set_seed
    from dlib.utils.shared import announce_msg

    from torch.profiler import profile, record_function, ProfilerActivity


    seed = 0
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    set_seed(seed=seed)
    n, h, w = 32, 244, 244
    scale_factor = 1.
    nbr_plans = 3
    announce_msg(f'nbr plans imag: {nbr_plans}')
    img = torch.randint(
        low=0, high=256,
        size=(n, nbr_plans, h, w), dtype=torch.float, device=DEVICE,
        requires_grad=False).cpu()
    nbr_cl = 2
    segmentations = torch.rand(size=(n, nbr_cl, h, w), dtype=torch.float,
                               device=DEVICE,
                               requires_grad=True)

    loss = DenseCRFLoss(weight=1e-7,
                        sigma_rgb=15.,
                        sigma_xy=100.,
                        scale_factor=scale_factor
                        ).to(DEVICE)
    announce_msg("testing {}".format(loss))
    set_seed(seed=seed)
    if nbr_cl > 1:
        softmax = nn.Softmax(dim=1)
    else:
        softmax = nn.Sigmoid()

    print(img.sum(), softmax(segmentations).sum())

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with autocast(enabled=False):
        z = loss(images=img, segmentations=softmax(segmentations))
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('time op: {}'.format(elapsed_time_ms))

    print('Time ({} x {} : scale: {}: N: {}): TIME_ABOVE'.format(
        h, w, scale_factor, n))
    tx = time.perf_counter()
    z.backward()
    print('backward {}'.format(time.perf_counter() - tx))
    print('Loss: {} {} (nbr_cl: {})'.format(z, z.dtype, nbr_cl))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    for i in range(3):
        test_DenseCRFLoss()
