import argparse

from arguments.reproduce_condense import set_arguments

def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ipc_epoch(ipc, factor, nclass=10, bound=-1):
    """Calculating training epochs for ImageNet
    """
    factor = max(factor, 1)
    ipc *= factor**2
    if bound > 0:
        ipc = min(ipc, bound)

    if ipc == 1:
        epoch = 3000
    elif ipc <= 10:
        epoch = 2000
    elif ipc <= 50:
        epoch = 1500
    elif ipc <= 200:
        epoch = 1000
    elif ipc <= 500:
        epoch = 500
    else:
        epoch = 300

    if nclass == 100:
        epoch = int((2 / 3) * epoch)
        epoch = epoch - (epoch % 100)

    return epoch


# def tune_lr_img(args, lr_img):
#     """Tuning lr_img for imagenet 
#     """
#     # Use mse loss for 32x32 img and ConvNet
#     ipc_base = 10
#     if args.dataset == 'imagenet':
#         imsize_base = 224
#     elif args.dataset == 'speech':
#         imsize_base = 64
#     elif args.dataset == 'mnist' or args.dataset == 'fashion':
#         imsize_base = 28
#     else:
#         imsize_base = 32

#     param_ratio = (args.ipc / ipc_base)
#     if args.size > 0:
#         param_ratio *= (args.size / imsize_base)**2

#     lr_img = lr_img * param_ratio
#     return lr_img


def remove_aug(augtype, remove_aug):
    """Remove certain type of augmentation (string)
    """
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


parser = argparse.ArgumentParser(description='')
# Dataset

parser.add_argument('--dmreg', default=1, type=int)

parser.add_argument('-d',
                    '--dataset',
                    default='cifar10',
                    type=str,
                    help='dataset (options: mnist, fashion, svhn, cifar10, cifar100, and imagenet)')


parser.add_argument('--data_dir',
                    default='/data_large/readonly',
                    type=str,
                    help='directory that containing dataset, except imagenet (see data.py)')
parser.add_argument('--imagenet_dir', default='/ssd_data/imagenet/', type=str)

parser.add_argument('--nclass', default=10, type=int, help='number of classes in trianing dataset')
parser.add_argument('--dseed', default=0, type=int, help='seed for class sampling')
parser.add_argument('--size', default=224, type=int, help='spatial size of image')
parser.add_argument('--phase', default=-1, type=int, help='index for multi-processing')
parser.add_argument('--nclass_sub', default=-1, type=int, help='number of classes for each process')
parser.add_argument('-l',
                    '--load_memory',
                    type=str2bool,
                    default=True,
                    help='load training images on the memory')
# Network
parser.add_argument('-n',
                    '--net_type',
                    default='convnet',
                    type=str,
                    help='network type: resnet, resnet_ap, convnet')
parser.add_argument('--norm_type',
                    default='instance',
                    type=str,
                    choices=['batch', 'instance', 'sn', 'none'])
parser.add_argument('--depth', default=10, type=int, help='depth of the network')
parser.add_argument('--width', default=1.0, type=float, help='width of the network')

# Training
parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for training')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=0, type=int, help='random seed for training')


# Mixup
parser.add_argument('--mixup',
                    default='cut',
                    type=str,
                    choices=('vanilla', 'cut'),
                    help='mixup choice for evaluation')

parser.add_argument('--beta', default=1.0, type=float, help='mixup beta distribution')
parser.add_argument('--mix_p', default=1.0, type=float, help='mixup probability')

parser.add_argument('--rrc',
                    type=str2bool,
                    default=True,
                    help='use random resize crop for ImageNet')
parser.add_argument('--dsa',
                    type=str2bool,
                    default=False,
                    help='Use DSA augmentation for evaluation or not')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')


# Logging

parser.add_argument('--epoch_eval_interval',
                    default=500,
                    type=int)
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--verbose',
                    dest='verbose',
                    action='store_true',
                    help='to print the status at every iteration')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--save_ckpt', type=str2bool, default=False)

parser.add_argument('--test', action='store_true', help='for debugging, do not save results')
parser.add_argument('--time', action='store_true', help='measuring time for each step')

# Condense
parser.add_argument('-i', '--ipc', type=int, default=-1, help='number of condensed data per class')
parser.add_argument('-f',
                    '--factor',
                    type=int,
                    default=1,
                    help='multi-formation factor. (1 for IDC-I)')
parser.add_argument('--decode_type',
                    type=str,
                    default='single',
                    choices=['single', 'multi', 'bound'],
                    help='multi-formation type')
parser.add_argument('--init',
                    type=str,
                    default='random',
                    choices=['random', 'noise', 'mix'],
                    help='condensed data initialization type')
parser.add_argument('-a',
                    '--aug_type',
                    type=str,
                    default='color_crop_cutout',
                    help='augmentation strategy for condensation matching objective')

## Optimization
# For small datasets, niter=2000 is enough for the full convergence.
# For faster optimzation, you can early stop the code based on the printed log.
parser.add_argument('--niter', type=int, default=500, help='number of outer iteration')

parser.add_argument('--num_premodel',
                    type=int,
                    default=5,
                    help='number of pretrained models')
parser.add_argument('--iter_calib',
                    type=int,
                    default=1,
                    help='number of the iterations for calibration')


parser.add_argument('--batch_real',
                    type=int,
                    default=128,
                    help='batch size of real training data used for matching')
parser.add_argument(
    '--batch_syn_max',
    type=int,
    default=200,
    help=
    'maximum number of synthetic data used for each matching (ramdom sampling for large synthetic data)'
)
parser.add_argument('--lr_img', type=float, default=5e-3, help='condensed data learning rate')
parser.add_argument('--mom_img', type=float, default=0.5, help='condensed data momentum')
parser.add_argument('--reproduce', action='store_true', help='for reproduce our setting')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=False)
args = parser.parse_args()

if args.reproduce:
    args = set_arguments(args)

""" 
DATA 
"""
args.nch = 3
if args.dataset[:5] == 'cifar':
    args.size = 32
    args.mix_p = 0.5
    args.dsa = True
    if args.dataset == 'cifar10':
        args.nclass = 10
    elif args.dataset == 'cifar100':
        args.nclass = 100

elif args.dataset == 'svhn':
    args.size = 32
    args.nclass = 10
    args.mix_p = 0.5
    args.dsa = True
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'flip')

elif args.dataset[:5] == 'mnist':
    args.nclass = 10
    args.size = 28
    args.nch = 1
    args.mix_p = 0.5
    args.dsa = True
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'flip')

elif args.dataset == 'fashion':
    args.nclass = 10
    args.size = 28
    args.nch = 1
    args.mix_p = 0.5
    args.dsa = True

elif args.dataset == 'tinyimagenet':
    args.nclass = 200
    args.size = 64
    args.nch = 3
    args.mix_p = 0.5
    args.dsa = True

elif args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
    args.nclass = 10
    args.size = 128
    args.nch = 3
    args.mix_p = 0.5
    args.dsa = True

elif args.dataset == 'imagenet':
    if args.net_type == 'convnet':
        args.net_type = 'resnet_ap'
    args.size = 224
    if args.nclass >= 100:
        args.load_memory = False
        print("args.load_memory is setted as False! (see args.argument)")
        # We need to tune lr and weight decay
        args.lr = 0.1
        args.weight_decay = 1e-4
        args.batch_size = max(128, args.batch_size)
        args.batch_real = max(128, args.batch_real)



# Default initialization for multi-formation
if args.factor > 1:
    args.init = 'mix'

"""
Evaluation setting
"""
# Setting evaluation training epochs

if args.dataset == 'imagenet':
    if args.decode_type == 'bound':
        args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass, bound=args.batch_syn_max)
    else:
        args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass)
    args.epoch_print_freq = args.epochs // 100
elif args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
    if args.decode_type == 'bound':
        args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass, bound=args.batch_syn_max)
    else:
        args.epochs = ipc_epoch(args.ipc, args.factor, args.nclass)
    args.epoch_print_freq = args.epochs // 100
else:
    args.epochs = 1500
    args.epoch_print_freq = args.epochs
    args.epoch_eval_interval = 250


# Setting augmentation
if args.mixup == 'cut':
    args.dsa_strategy = remove_aug(args.dsa_strategy, 'cutout')
if args.dsa:
    args.augment = False
    print("DSA strategy: ", args.dsa_strategy)
else:
    args.augment = True



if args.dataset == 'imagenet':
    args.data_name = "{}{}".format(args.dataset, args.nclass)
else:
    args.data_name = "{}".format(args.dataset)


