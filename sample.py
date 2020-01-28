''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import json
import os

import numpy as np
import torch
import torchvision
from tqdm import trange

# Import my stuff
import cfg
import models
import inception_utils
import utils


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    orig_exp_name = config['experiment_name']
    for key in ['samples_root', 'results_root']:
        os.makedirs(os.path.join(config[key], orig_exp_name), exist_ok=True)
    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode',
                            'load_weights']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    # config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = cfg.nclass_dict[config['dataset']]
    config['G_activation'] = cfg.activation_dict[config['G_nl']]
    config['D_activation'] = cfg.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'
    config['distributed'] = False
    config['gpu'] = None
    if config['pretrained'] is None:
        config['pretrained'] = config['dataset'].lower()

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = getattr(models, config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is {}'.format(experiment_name))

    G = model.Generator(**config).cuda()
    utils.count_parameters(G)

    # Load weights
    print('Loading weights...')
    # Here is where we deal with the ema--load ema weights or load normal weights
    utils.load_weights(G if not (config['use_ema']) else None, None, state_dict,
                       config['weights_root'], orig_exp_name, config['load_weights'],
                       G if config['ema'] and config['use_ema'] else None,
                       strict=False, load_optim=False)
    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])

    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()
    else:
        print('G is in %s mode...' % ('training' if G.training else 'eval'))

    print(config)
    # Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    if config['accumulate_stats']:
        print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Sample a number of images and save them to an NPZ, for use with TF-Inception
    if config['sample_npz']:
        # Lists to hold images and labels for images
        x, y = [], []
        print('Sampling %d images and saving them to npz...' % config['sample_num_npz'])
        for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
            with torch.no_grad():
                images, labels = sample()
            x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
            y += [labels.cpu().numpy()]
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        y = np.concatenate(y, 0)[:config['sample_num_npz']]
        print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
        npz_filename = '%s/%s/%s_samples.npz' % (config['samples_root'], orig_exp_name, config['load_weights'])
        print('Saving npz to %s...' % npz_filename)
        np.savez(npz_filename, **{'x': x, 'y': y})

    # Prepare sample sheets
    if config['sample_sheets']:
        print('Preparing conditional sample sheets...')
        utils.sample_sheet(G, classes_per_sheet=cfg.classes_per_sheet_dict[config['dataset']],
                           num_classes=config['n_classes'],
                           samples_per_class=10, parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=orig_exp_name,
                           folder_number=config['sample_sheet_folder_num'],
                           z_=z_,)
    # Sample interp sheets
    if config['sample_interps']:
        print('Preparing interp sheets...')
        for fix_z, fix_y in zip([False, False, True], [False, True, False]):
            utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                               num_classes=config['n_classes'],
                               parallel=config['parallel'],
                               samples_root=config['samples_root'],
                               experiment_name=orig_exp_name,
                               folder_number=config['sample_sheet_folder_num'],
                               sheet_number=0,
                               fix_z=fix_z, fix_y=fix_y, device='cuda')
    # Sample random sheet
    if config['sample_random']:
        print('Preparing random sample sheet...')
        images, labels = sample()
        torchvision.utils.save_image(images.float(),
                                     '%s/%s/%srandom_samples.jpg' % (config['samples_root'], orig_exp_name, config['load_weights']),
                                     nrow=int(G_batch_size**0.5),
                                     normalize=True)

    # Get Inception Score and FID
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
    # get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
    # Prepare a simple function get metrics that we use for trunc curves

    def get_metrics():
        sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
        IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'],
                                                     num_splits=10, prints=False, use_torch=config['use_torch_FID'])
        result = {
            'use_ema': config['use_ema'],
            'G_eval_mode': config['G_eval_mode'],
            'z_var': z_.var,
            'num_inception_images': config['num_inception_images'],
            'num_standing_accumulations': config['num_standing_accumulations'],
            'G_batch_size': G_batch_size,
            'itr': state_dict['itr'],
            'IS_mean': IS_mean.item(),
            'IS_std': IS_std.item(),
            'FID': FID.item(),
        }
        # Prepare output string
        outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
        outstring += 'in %s mode, ' % ('eval' if config['G_eval_mode'] else 'training')
        outstring += 'with noise variance %3.3f, ' % z_.var
        outstring += 'over %d images, ' % config['num_inception_images']
        if config['accumulate_stats'] or not config['G_eval_mode']:
            outstring += 'with batch size %d, ' % G_batch_size
        if config['accumulate_stats']:
            outstring += 'using %d standing stat accumulations, ' % config['num_standing_accumulations']
        outstring += 'Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (
            state_dict['itr'], IS_mean, IS_std, FID)
        return result, outstring

    results = []

    name = ''
    name = f'itr_{state_dict["itr"]}' + '_zvar_{}_results.json'
    name = f'{config["load_weights"]}_' + name if config["load_weights"] else name
    results_file_tmpl = os.path.join(config['results_root'], orig_exp_name, name)

    trunc_file = os.path.join(config['samples_root'], orig_exp_name, f'{config["load_weights"]}truncation_curves_{state_dict["itr"]}.txt')
    with open(trunc_file, 'w') as f:
        if config['sample_inception_metrics']:
            print('Calculating Inception metrics...')
            result, msg = get_metrics()
            print(msg)
            f.write(msg + '\n')
            results.append(result)
            with open(results_file_tmpl.format(z_.var.__format__('3.3f')), 'w') as jf:
                json.dump(result, jf)

        # Sample truncation curve stuff. This is basically the same as the inception metrics code
        if config['sample_trunc_curves']:
            start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
            print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
            for var in np.arange(start, end + step, step):
                z_.var = var
                # Optionally comment this out if you want to run with standing stats
                # accumulated at one z variance setting
                if config['accumulate_stats']:
                    utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                                    config['num_standing_accumulations'])
                result, msg = get_metrics()
                print(msg)
                f.write(msg + '\n')
                results.append(result)
                with open(results_file_tmpl.format(z_.var.__format__('3.3f')), 'w') as jf:
                    json.dump(result, jf)
    with open(results_file_tmpl.format('all'), 'w') as jf:
        json.dump(result, jf)


def main():
    # parse command line and run
    parser = cfg.prepare_parser()
    parser = cfg.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
