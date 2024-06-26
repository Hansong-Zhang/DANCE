def set_arguments(args):
    """Specific arguments for reproduce our condensed data
       The metric choice does not matter much.
       But, you should adjust lr_img according to the metric.
    """
    if args.dataset == 'imagenet':
        args.data_name = "{}{}".format(args.dataset, args.nclass)
    else:
        args.data_name = "{}".format(args.dataset)
    args.pretrain_dir = "<>"


    if args.dataset != 'imagenet':
        args.net_type = 'convnet'
        args.depth = 3
        args.niter = 10000
        args.lr_img = 0.01 * args.ipc
        args.pretrain_amount = 20
        args.pretrain_epochs = 60
        if args.dataset[:5] == 'cifar':
            args.data_dir = '<>'
        elif args.dataset == 'svhn':
            args.data_dir = '<>'
            # if args.factor == 1 and args.ipc == 1:
            #     # In this case, evaluation w/o mixup is much more effective
            #     args.mixup = 'vanilla'
            #     args.dsa_strategy = 'color_crop_cutout_scale_rotate'
        elif args.dataset == 'fashion':
            args.data_dir = '<>'
        elif args.dataset == 'mnist':
            args.data_dir = '<>'
        elif args.dataset == 'tinyimagenet':
            args.data_dir = '<>'
        elif args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
            args.lr_img = 0.1 * args.ipc
            args.imagenet_dir = '<>'
            args.net_type = 'convnet'
            args.depth = 5
            args.niter = 10000
            args.pretrain_amount = 5
            args.pretrain_epochs = 80

            if args.nclass == 10:
                args.batch_real = 128
        else:
            raise AssertionError("Not supported dataset!")
    else:
        args.imagenet_dir = '<>'
        args.net_type = 'resnet_ap'
        args.depth = 10
        args.niter = 10000

        # if args.ipc == 10:
        #     args.lr_img = 50.
        # elif args.ipc == 20:
        #     args.lr_img = 100.

        if args.nclass == 10:
            args.batch_real = 128
            
        # elif args.nclass == 100:
        #     args.lr_img = 1.

        if args.factor >= 3 and args.ipc >= 20:
            args.decode_type = 'bound'


    args.exp_name = 'Ipc{}_Fac{}_Lr{}_Npm{}_Ic{}_Bsr{}_Bss{}'.format(args.ipc, 
                                                                args.factor, 
                                                                args.lr_img, 
                                                                args.num_premodel, 
                                                                args.iter_calib,
                                                                args.batch_real,
                                                                args.batch_syn_max)
    # Result folder name
    if args.test:
        args.save_dir = './test_results/'
    else:
        args.save_dir = "<path to save the results>"


    return args
