import numpy as np
import os
import time
import torch
import torch.nn.functional as F

from copy import deepcopy
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
from tqdm import tqdm
from sacred import SETTINGS

from utils import state_dict_to_cpu, AverageMeter
from utils.data.segmentation.dataset_loaders import dataset_ingredient, load_dataset
from utils.metrics import Evaluator
from cnn.unet import unet_ingredient, load_unet

# Experiment
from sacred import Experiment

ex = Experiment('unet_training', ingredients=[dataset_ingredient, unet_ingredient])

# Filter backspaces and linefeeds
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def default_config():
    epochs = 30
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    lr_step = 10

    save_dir = os.path.join('results', 'temp')


@ex.capture
def get_optimizer_scheduler(parameters, lr, momentum, weight_decay, lr_step):
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    nesterov=True if momentum else False)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step)
    return optimizer, scheduler


def evaluate(model, loader, device, test=False):
    model.eval()
    all_labels = []
    all_losses = []
    all_seg_preds = []
    all_dices = []
    all_ious = []
    evaluator = Evaluator(2)
    image_evaluator = Evaluator(2)

    pbar = tqdm(loader, ncols=80, desc='Test' if test else 'Validation')
    with torch.no_grad():
        for image, mask, label in pbar:
            image = image.to(device, non_blocking=True)
            segmentation = (mask != 0).squeeze(1)
            t_segmentation = segmentation.to(device, non_blocking=True).long()

            seg_logits = model(image)
            loss = F.cross_entropy(seg_logits, t_segmentation).item()
            seg_probs = torch.softmax(seg_logits, 1)
            seg_preds = seg_logits.argmax(1)
            evaluator.add_batch(t_segmentation, seg_preds)
            image_evaluator.add_batch(t_segmentation, seg_preds)
            dices = image_evaluator.dice()
            ious = image_evaluator.intersection_over_union()
            image_evaluator.reset()

            all_labels.append(label.item())
            all_losses.append(loss)
            all_dices.append(dices.cpu())
            all_ious.append(ious.cpu())
            all_seg_preds.append(seg_preds.squeeze(0).byte().cpu().numpy().astype('bool'))

        all_labels = np.array(all_labels)
        all_losses = np.array(all_losses)
        all_dices = torch.stack(all_dices, 0)
        all_ious = torch.stack(all_ious, 0)

    dices = evaluator.dice()
    ious = evaluator.intersection_over_union()

    metrics = {
        'images_path': loader.dataset.samples,
        'labels': all_labels,
        'losses': all_losses,
        'dice_background_per_image': all_dices[:, 0].numpy(),
        'mean_dice_background': all_dices[:, 0].numpy().mean(),
        'dice_background': dices[0].item(),
        'dice_per_image': all_dices[:, 1].numpy(),
        'mean_dice': all_dices[:, 1].numpy().mean(),
        'dice': dices[1].item(),
        'iou_background_per_image': all_ious[:, 0].numpy(),
        'mean_iou_background': all_ious[:, 0].numpy().mean(),
        'iou_background': ious[0].item(),
        'iou_per_image': all_ious[:, 1].numpy(),
        'mean_iou': all_ious[:, 1].numpy().mean(),
        'iou': ious[1].item(),
    }

    if test and ex.current_run.config['dataset']['split'] == 0 and ex.current_run.config['dataset']['fold'] == 0:
        metrics['seg_preds'] = all_seg_preds

    return metrics


@ex.capture
def get_save_name(save_dir, dataset):
    exp_name = ex.get_experiment_info()['name']
    start_time = ex.current_run.start_time.strftime('%Y-%m-%d_%H-%M-%S')
    name = '{}_{}_{}_{}_{}'.format(exp_name, ex.current_run._id, dataset['name'], 'unet', start_time)
    return os.path.join(save_dir, name)


@ex.automain
def main(epochs, seed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.deterministic = True
    torch.manual_seed(seed)

    train_loader, valid_loader, test_loader = load_dataset()
    model = load_unet()
    model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(parameters=model.parameters())

    train_losses = AverageMeter()
    batch_evaluator = Evaluator(2)
    train_dices_background = AverageMeter()
    train_dices = AverageMeter()

    best_valid_dice = 0
    best_valid_loss = float('inf')
    best_model_dict = deepcopy(model.module.state_dict())

    for epoch in range(epochs):
        model.train()

        train_losses.reset(), train_dices_background.reset(), train_dices.reset()
        loader_length = len(train_loader)

        pbar = tqdm(train_loader, ncols=80, desc='Training')
        start = time.time()

        for i, (images, mask, label) in enumerate(pbar):
            images, mask = images.to(device), mask.squeeze(1).to(device, non_blocking=True)

            seg_logits = model(images)
            if ex.current_run.config['dataset']['name'] == 'caltech_birds':
                class_mask = (mask > 0.5).long()
            else:
                class_mask = (mask != 0).long()
            loss = F.cross_entropy(seg_logits, class_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_evaluator.add_batch(class_mask, seg_logits.argmax(1))
            dices = batch_evaluator.dice()
            dice_background = dices[0].item()
            dice = dices[1].item()
            batch_evaluator.reset()
            loss = loss.item()

            step = epoch + i / loader_length
            ex.log_scalar('training.loss', loss, step)
            ex.log_scalar('training.mean_dice_background', dice_background, step)
            ex.log_scalar('training.mean_dice', dice, step)
            train_losses.append(loss)
            train_dices_background.append(dice_background)
            train_dices.append(dice)

        scheduler.step()
        duration = time.time() - start

        # evaluate on validation set
        valid_metrics = evaluate(model=model, loader=valid_loader, device=device)

        if valid_metrics['losses'].mean() <= best_valid_loss:
            best_valid_dice = valid_metrics['mean_dice']
            best_valid_loss = valid_metrics['losses'].mean()
            best_model_dict = deepcopy(model.module.state_dict())

        ex.log_scalar('validation.loss', np.mean(valid_metrics['losses']), epoch + 1)
        ex.log_scalar('validation.mean_dice', valid_metrics['mean_dice'], epoch + 1)

        print('Epoch {:02d} | Duration: {:.1f}s - per batch ({}): {:.3f}s'.format(epoch, duration, loader_length,
                                                                                  duration / loader_length))
        print(' ' * 8, '| Train loss: {:.4f} dice(b): {:.3f} dice: {:.3f}'.format(train_losses.avg,
                                                                                  train_dices_background.avg,
                                                                                  train_dices.avg))
        print(' ' * 8, '| Valid loss: {:.4f} dice(b): {:.3f} dice: {:.3f}'.format(valid_metrics['losses'].mean(),
                                                                                  valid_metrics['mean_dice_background'],
                                                                                  valid_metrics['mean_dice']))

    # load best model based on validation loss
    model = load_unet()
    model.load_state_dict(best_model_dict)
    model.to(device)

    # evaluate on test set
    test_metrics = evaluate(model=model, loader=test_loader, device=device, test=True)

    ex.log_scalar('test.loss', test_metrics['losses'].mean(), epochs)
    ex.log_scalar('test.mean_dice_background', test_metrics['mean_dice_background'], epochs)
    ex.log_scalar('test.mean_dice', test_metrics['mean_dice'], epochs)

    # save model
    save_name = get_save_name() + '.pickle'
    torch.save(state_dict_to_cpu(best_model_dict), save_name)
    ex.add_artifact(os.path.abspath(save_name))

    # save test metrics
    if len(ex.current_run.observers) > 0:
        dataset = ex.current_run.config['dataset']['name']
        split = ex.current_run.config['dataset']['split']
        fold = ex.current_run.config['dataset']['fold']

        torch.save(
            test_metrics,
            os.path.join(ex.current_run.observers[0].dir, '{}_unet_split-{}_fold-{}.pkl'.format(dataset, split, fold))
        )

    # metrics to info.json
    info_to_save = ['labels', 'losses',
                    'dice_per_image', 'mean_dice', 'dice',
                    'iou_per_image', 'mean_iou', 'iou']
    for k in info_to_save:
        ex.info[k] = test_metrics[k]

    return test_metrics['mean_dice']
