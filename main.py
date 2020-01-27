""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

# Import my stuff
import cfg
import datasets
import inception_utils
import models
import train_fns
import utils

# The main training file. Config is a dictionary specifying the configuration
# of this training run.


def run(config):

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    # config['resolution'] = cfg.imsize_dict[config['dataset']]
    config['n_classes'] = cfg.nclass_dict[config['dataset']]
    config['G_activation'] = cfg.activation_dict[config['G_nl']]
    config['D_activation'] = cfg.activation_dict[config['D_nl']]
    if config['pretrained'] is None:
        config['pretrained'] = config['dataset'].lower()

    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    if config['dist_url'] == "env://" and config['world_size'] == -1:
        config['world_size'] = int(os.environ["WORLD_SIZE"])

    config['distributed'] = config['world_size'] > 1 or config['multiprocessing_distributed']

    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config['world_size'] = ngpus_per_node * config['world_size']
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config['gpu'], ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):

    device = f'cuda:{gpu}'
    config['gpu'] = gpu
    torch.backends.cudnn.benchmark = True
    if config['distributed']:
        if config['dist_url'] == "env://" and config['rank'] == -1:
            config['rank'] = int(os.environ["RANK"])
        if config['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            if config['rank'] == -1:
                config['rank'] = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
            config['rank'] = config['rank'] * ngpus_per_node + gpu
        dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'],
                                world_size=config['world_size'], rank=config['rank'])
    # Import the model--this line allows us to dynamically select different files.
    model = getattr(models, config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is {}'.format(experiment_name)) if config['rank'] == 0 else None

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay'])) if config['rank'] == 0 else None
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        ema = None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]])) if config['rank'] == 0 else None
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    if config['distributed']:
        if config['cross_replica']:
            print('Converting network to Cross Replica BatchNorm')
            GD = nn.SyncBatchNorm.convert_sync_batchnorm(GD)
        print('Distributing model...') if config['rank'] == 0 else None
        if config['gpu'] is not None:
            torch.cuda.set_device(config['gpu'])
            GD.cuda(config['gpu'])
            config['batch_size'] = int(config['batch_size'] / ngpus_per_node)
            config['num_workers'] = int(config['num_workers'] / ngpus_per_node)
            GD = torch.nn.parallel.DistributedDataParallel(GD, device_ids=[config['gpu']], find_unused_parameters=True)
        else:
            GD.cuda()
            GD = torch.nn.parallel.DistributedDataParallel(GD, find_unused_parameters=True)
    elif config['gpu'] is not None:
        torch.cuda.set_device(config['gpu'])
        GD = GD.cuda(config['gpu'])
    else:
        GD = torch.nn.DataParallel(GD).cuda()

    torch.cuda.empty_cache()

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    if config['rank'] == 0:
        test_metrics_fname = os.path.join(config['logs_root'], f'{experiment_name}_log.jsonl')
        train_metrics_fname = os.path.join(config['logs_root'], experiment_name)
        print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
        test_log = utils.MetricsLogger(test_metrics_fname,
                                       reinitialize=(not config['resume']))
        print('Training Metrics will be saved to {}'.format(train_metrics_fname))
        train_log = utils.MyLogger(train_metrics_fname,
                                   reinitialize=(not config['resume']),
                                   logstyle=config['logstyle'])
        # Write metadata
        utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    loaders = datasets.get_dataloaders(**{**config, 'batch_size': D_batch_size})

    # Prepare inception metrics: FID and IS
    dataset = config['dataset']
    pretrained = config['pretrained']
    resolution = config['resolution']
    fname = f'{dataset}-{resolution}_{pretrained}_inception_moments.npz'
    inception_root_dir = cfg.get_root_dirs(dataset,
                                           resolution=config['resolution'],
                                           data_root=config['data_root'])
    if dataset == 'Hybrid1365':
        inception_root_dir = os.path.join(inception_root_dir, 'Hybrid1365')
    inception_filename = os.path.join(inception_root_dir, fname)
    get_inception_metrics = inception_utils.prepare_inception_metrics(
        inception_filename, config, config['no_fid'])

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                               z_=z_, y_=y_, config=config)

    print('Beginning training at epoch {}...'.format(state_dict['epoch'])) if config['rank'] == 0 else None
    torch.cuda.empty_cache()
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        torch.cuda.empty_cache()
        for i, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            try:
                metrics = train(x, y)
            except IndexError:
                print(f'Skipping itr: {i}')
            if config['rank'] == 0:
                train_log.log(itr=int(state_dict['itr']), **metrics)

                # Every sv_log_interval, log singular values
                if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                    train_log.log(itr=int(state_dict['itr']),
                                  **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                if config['rank'] == 0:
                    print(', '.join(['itr: %d' % state_dict['itr']]
                                    + ['%s : %+4.3f' % (key, metrics[key])
                                       for key in metrics]))
                    # for key in metrics]), end=' ')

            if config['rank'] == 0:
                if not config['multiprocessing_distributed'] or (config['multiprocessing_distributed']
                                                                 and config['rank'] % ngpus_per_node == 0):
                    # Save weights and copies as configured at specified interval
                    if not (state_dict['itr'] % config['save_every']):
                        if config['G_eval_mode']:
                            print('Switchin G to eval mode...')
                            G.eval()
                            if config['ema']:
                                G_ema.eval()
                        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                                  state_dict, config, experiment_name)

                    # Test every specified interval
                    if not (state_dict['itr'] % config['test_every']):
                        if config['G_eval_mode']:
                            print('Switchin G to eval mode...')
                            G.eval()
                        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                                       get_inception_metrics, experiment_name, test_log, use_torch=config['use_torch_FID'])
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # Parse command line and run.
    parser = cfg.prepare_parser()
    config = vars(parser.parse_args())
    run(config)


if __name__ == '__main__':
    main()
