import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import ClassDataLoader, ClassMemDataLoader
from train import define_model, train_epoch, validate
from common import load_resized_data, diffaug


def pretrain(args, logger, device='cuda'):
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)
    nclass = trainset.nclass
    _, aug_rand = diffaug(args)

    criterion = nn.CrossEntropyLoss()

    
    logger(f"Start training {args.pretrain_amount} models for {args.pretrain_epochs} epochs")
    for model_id in range(args.pretrain_amount):

        init_path = os.path.join(args.pretrain_dir, f'premodel{model_id}_init.pth.tar')
        trained_path = os.path.join(args.pretrain_dir, f'premodel{model_id}_trained.pth.tar')

        model = define_model(args, nclass).to(device)
        torch.save(model.state_dict(), init_path)

        model.train()
        optim_net = optim.SGD(model.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optim_net, 
                                                   milestones=[2 * args.pretrain_epochs // 3, 5 * args.pretrain_epochs // 6], 
                                                   gamma=0.2)

        for epoch in range(args.pretrain_epochs):
            top1, _, loss = train_epoch(args,
                                        loader_real,
                                        model,
                                        criterion,
                                        optim_net,
                                        aug=aug_rand,
                                        mixup=args.mixup)
            top1_val, _, _ = validate(val_loader, model, criterion)
            logger("<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} (loss: {:.3f}), Val acc: {:.1f}".format(model_id,
                                                                                                                          epoch,
                                                                                                                          top1,
                                                                                                                          loss,
                                                                                                                          top1_val))
            scheduler.step()

        torch.save(model.state_dict(), trained_path)


if __name__ == '__main__':
    from misc.utils import Logger
    from arguments.arg_pretrain import args
    import torch.backends.cudnn as cudnn


    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.pretrain_dir, exist_ok=True)

    logger = Logger(args.pretrain_dir)
    logger(f"Pretrain models save dir: {args.pretrain_dir}")
    logger(args)
    pretrain(args, logger)