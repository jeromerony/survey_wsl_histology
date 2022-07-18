import sys
from os.path import dirname, abspath, join

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import pickle as pkl

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.cams import AccSeeds
from dlib.cams import AccSeedsmeter
from dlib.cams import BasicAccSeedsMeter

from dlib.utils.tools import compute_auc

from dlib.learning.train import FastSeedCamEvalStdCl


def acc_cam_seed_std_cl(args, dataloader, model, device, subset, fdout):
    multi_label_flag = args.multi_label_flag
    support_background = args.model['support_background']

    summary = []

    MAX_NS = 40000

    for KSZ in [1, 3, 5, 7, 9, 11]:

        seed_cam_evaluer = AccSeeds(multi_label_flag, device, MAX_NS, KSZ)
        nbr_samples = len(dataloader.dataset)
        meters = AccSeedsmeter(n_seeds=seed_cam_evaluer.n_seeds.copy(),
                               nbr_samples=nbr_samples, device=device)

        c_fdout = join(fdout, '{}'.format(MAX_NS))

        module = FastSeedCamEvalStdCl(model,
                                      seed_cam_evaluer,
                                      meters,
                                      c_fdout,
                                      multi_label_flag,
                                      support_background,
                                      device,
                                      args.task,
                                      subset)

        summary.append(module.run(dataloader))

    fig, axes = plt.subplots(nrows=len(summary), ncols=2,
                             sharex=True, sharey=True, squeeze=False)
    for kk, elm in enumerate(summary):

        meters = summary[kk]['meters']
        x = summary[kk]['n_seeds']
        ksz = summary[kk]['ksz']
        max_sn = summary[kk]['max_sn']

        for i, k in enumerate(meters.keys()):
            meter: BasicAccSeedsMeter = meters[k]
            mean = meter.mean.cpu().numpy()
            _min = meter.min.cpu().numpy()
            _max = meter.max.cpu().numpy()
            std = torch.sqrt(meter.emvar).cpu().numpy()

            # axes[kk, i].plot(x, _min, color='tab:blue', label='Min.',
            #                  alpha=0.2)
            # axes[kk, i].plot(x, _max, color='tab:green', label='Max.',
            #                  alpha=0.2)

            axes[kk, i].plot(x, mean, color='tab:orange', label='Avg.')
            axes[kk, i].fill_between(x, mean - std, mean + std, alpha=0.2,
                                     color='tab:orange')

            subt = '{}. AUC: {:.3f} %. Kernel size: {}x{}'.format(
                k, compute_auc(mean, mean.size), ksz, ksz)
            axes[kk, i].set_title(subt, fontsize=5)
            axes[kk, i].xaxis.set_tick_params(labelsize=4)
            axes[kk, i].yaxis.set_tick_params(labelsize=4)
            axes[kk, i].grid(True)
            axes[kk, i].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[kk, i].legend(loc='lower left', fancybox=True, shadow=True,
                               prop={'size': 2})
            axes[kk, i].set_xlabel('Nbr. clicks', fontsize=4)
            axes[kk, i].set_ylabel('Clicks accuracy (%)', fontsize=4)

    for ax in axes.flat:
        ax.label_outer()

    fig.suptitle('Subset: {}. Task: {}. '
                 'Max_NS: {}'.format(subset, args.task, max_sn),
                 fontsize=5)
    plt.tight_layout()
    tag = '{}.png'.format(max_sn)
    fig.savefig(join(c_fdout, tag), bbox_inches='tight', dpi=400)
    tag = '{}'.format(max_sn)
    with open(join(c_fdout, '{}.pkl'.format(tag)), 'wb') as fout:
        pkl.dump(summary, fout, protocol=pkl.HIGHEST_PROTOCOL)
