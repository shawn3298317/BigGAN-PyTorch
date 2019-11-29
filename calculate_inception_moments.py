''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
from argparse import ArgumentParser
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import inception_utils
import utils


def prepare_parser():
    usage = 'Calculate and store inception metrics.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='I128_hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100...'
        'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
    parser.add_argument('--resolution', default=128, type=int)
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data? (default: %(default)s)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    parser.add_argument(
        '--pretrained', type=str, default='imagenet',
    )
    return parser


def run(config):
    # Get loader
    config['gpu'] = None
    config['drop_last'] = False
    config['distributed'] = False
    # loaders = utils.get_data_loaders(**config)
    loaders = datasets.get_dataloaders(**config)
    dataset_name = f'{config["dataset"]}-{config["resolution"]}'

    # Load inception net
    net = inception_utils.load_inception_net(config)
    pool, logits, labels = [], [], []
    device = 'cuda'
    for i, (x, y) in enumerate(tqdm(loaders[0])):
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]

    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    # uncomment to save pool, logits, and labels to disk
    # print('Saving pool, logits, and labels to disk...')
    # np.savez(config['dataset']+'_inception_activations.npz',
    #           {'pool': pool, 'logits': logits, 'labels': labels})
    # Calculate inception metrics and report them
    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    out = f'{config["pretrained"]} model evaluated on {dataset_name} has IS of {IS_mean:5.5f} +/- {IS_std:5.5f}'
    print(out)
    fname = f'{dataset_name}_{config["pretrained"]}_inception_scores.txt'
    with open(fname, 'w') as f:
        f.write(out)

    # Prepare mu and sigma, save to disk. Remove "hdf5" by default
    # (the FID code also knows to strip "hdf5")
    print('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    print('Saving calculated means and covariances to disk...')
    fname = f'{dataset_name}_{config["pretrained"]}_inception_moments.npz'
    output_name = os.path.join(config['data_root'], config['dataset'], fname)
    np.savez(output_name, **{'mu': mu, 'sigma': sigma, 'IS_mean': IS_mean, 'IS_std': IS_std})


def main():
    # parse command line
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
