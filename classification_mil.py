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
from utils.metrics import metric_report
from utils.data.classification.dataset_loaders import dataset_ingredient, load_dataset
from mil.models import model_ingredient, load_model

# Experiment
from sacred import Experiment

ex = Experiment('classification_mil', ingredients=[dataset_ingredient, model_ingredient])

# Filter backspaces and linefeeds
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def default_config():
    epochs = 30
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    lr_step = 10

    save_dir = os.path.join('results', 'models')


@ex.capture
def get_optimizer_scheduler(parameters, lr, momentum, weight_decay, lr_step):
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    nesterov=True if momentum else False)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step)
    return optimizer, scheduler


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_logits = []
    all_probabilities = []
    all_predictions = []
    losses = []

    pbar = tqdm(loader, ncols=80, desc='Evaluation')
    with torch.no_grad():
        for image, label in pbar:
            image, t_label = image.to(device), label.to(device, non_blocking=True)

            logits = model(image)
            probs = model.pooling.probabilities(logits=logits)
            pred = model.pooling.predictions(logits=logits)
            loss = model.pooling.loss(logits=logits, labels=t_label)

            all_labels.append(label)
            all_probabilities.append(probs.cpu())
            all_predictions.append(pred.cpu())
            all_logits.append(logits.cpu())
            losses.append(loss.item())

        all_labels = torch.cat(all_labels, 0)
        all_predictions = torch.cat(all_predictions, 0)
        all_logits = torch.cat(all_logits, 0)
        all_probabilities = torch.cat(all_probabilities, 0)

    metrics = metric_report(all_labels, all_probabilities, all_predictions)
    metrics['loss'] = sum(losses) / len(losses)
    metrics['labels'] = all_labels.numpy()
    metrics['logits'] = all_logits.numpy()

    return metrics


@ex.capture
def get_save_name(save_dir, dataset, model):
    exp_name = ex.get_experiment_info()['name']
    start_time = ex.current_run.start_time.strftime('%Y-%m-%d_%H-%M-%S')
    name = '{}_{}_{}_{}_{}'.format(exp_name, ex.current_run._id, dataset['name'], model['pooling'], start_time)
    return os.path.join(save_dir, name)


@ex.automain
def main(epochs, seed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.deterministic = True
    torch.manual_seed(seed)

    train_loader, valid_loader, test_loader = load_dataset()
    model = load_model()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(parameters=model.parameters())

    train_losses = AverageMeter()
    train_accs = AverageMeter()

    best_valid_acc = 0
    best_valid_loss = float('inf')
    best_model_dict = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()

        train_losses.reset(), train_accs.reset()
        loader_length = len(train_loader)

        pbar = tqdm(train_loader, ncols=80, desc='Training')
        start = time.time()

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device, non_blocking=True)

            logits = model(images)
            predictions = model.pooling.predictions(logits=logits)
            loss = model.pooling.loss(logits=logits, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (predictions == labels).float().mean().item()
            loss = loss.item()

            step = epoch + i / loader_length
            ex.log_scalar('training.loss', loss, step)
            ex.log_scalar('training.acc', acc, step)
            train_losses.append(loss)
            train_accs.append(acc)

        scheduler.step()
        end = time.time()
        duration = end - start

        # evaluate on validation set
        valid_metrics = evaluate(model=model, loader=valid_loader, device=device)

        if valid_metrics['loss'] <= best_valid_loss:
            best_valid_acc = valid_metrics['accuracy']
            best_valid_loss = valid_metrics['loss']
            best_model_dict = deepcopy(model.state_dict())

        ex.log_scalar('validation.loss', valid_metrics['loss'], epoch + 1)
        ex.log_scalar('validation.acc', valid_metrics['accuracy'], epoch + 1)

        print('Epoch {:02d} | Duration: {:.1f}s - per batch ({}): {:.3f}s'.format(epoch, duration, loader_length,
                                                                                  duration / loader_length))
        print(' ' * 8, '| Train\tloss: {:.4f}\tacc: {:.3f}'.format(train_losses.avg, train_accs.avg))
        print(' ' * 8, '| Valid\tloss: {:.4f}\tacc: {:.3f}'.format(valid_metrics['loss'], valid_metrics['accuracy']))

    # load best model based on validation loss
    model.load_state_dict(best_model_dict)
    # evaluate on test set
    test_metrics = evaluate(model=model, loader=test_loader, device=device)

    ex.log_scalar('test.loss', test_metrics['loss'], epochs)
    ex.log_scalar('test.acc', test_metrics['accuracy'], epochs)

    # save model
    save_name = get_save_name() + '.pickle'
    torch.save(state_dict_to_cpu(best_model_dict), save_name)
    ex.add_artifact(os.path.abspath(save_name))

    # metrics to info.json
    for k, v in test_metrics.items():
        ex.info[k] = v

    return test_metrics['accuracy']
