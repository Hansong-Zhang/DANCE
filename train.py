# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
from misc.augment import DiffAug
import time
import warnings
from common import define_model

warnings.filterwarnings("ignore")
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

mean_torch = {}
std_torch = {}
for key, val in MEANS.items():
    mean_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)
for key, val in STDS.items():
    std_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)


def main(args, logger, repeat=1):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    _, train_loader, val_loader, nclass = load_data(args)

    best_acc_l = []
    acc_l = []                                              
    for i in range(repeat):
        logger(f"\nRepeat: {i+1}/{repeat}")
        plotter = Plotter(args.save_dir, args.epochs, idx=i)
        model = define_model(args, nclass, logger)

        best_acc, acc = train(args, model, train_loader, val_loader, plotter, logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

    logger(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}')


def train(args, model, train_loader, val_loader, logger=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0


    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    # print(get_time())
    for epoch in range(1, args.epochs+1):
        acc1_tr, acc5_tr, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          aug,
                                          mixup=args.mixup)
        
        if (epoch % args.epoch_print_freq == 0) and (logger is not None):
            logger(
                '(Train) [Epoch {0}/{1}] {2} Top1 {top1:.1f}  Top5 {top5:.1f}  Loss {loss:.3f}'
                .format(epoch, args.epochs, get_time(), top1=acc1_tr, top5=acc5_tr, loss=loss_tr))


        if epoch % args.epoch_eval_interval == 0 or epoch == args.epochs:
            acc1, acc5, loss_val = validate(val_loader, model, criterion)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
            if logger is not None:
                logger('-------Eval Training Epoch [{} / {}] INFO--------'.format(epoch, args.epochs))
                logger(f'Current accuracy (top-1 and 5): {acc1:.1f} {acc5:.1f}')
                logger(f'Best    accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        scheduler.step()

    return best_acc1, acc1


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                aug=None,
                mixup='vanilla'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()

        data_time.update(time.time() - end)

        if aug != None:
            with torch.no_grad():
                input = aug(input)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, top5.avg, losses.avg






