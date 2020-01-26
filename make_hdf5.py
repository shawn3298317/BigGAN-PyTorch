""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to
    an HDF5 file for improved I/O. """
from argparse import ArgumentParser
import os
from torchvision import transforms

import datasets


def prepare_parser():
    usage = 'Parser for ImageNet HDF5 scripts.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='ImageNet',
        help='Which Dataset to train on, out of ImageNet and Places365')
    parser.add_argument('--resolution', default=128, type=int)
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of dataloader workers (default: %(default)s)')
    parser.add_argument(
        '--chunk_size', type=int, default=500,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--compression', action='store_true', default=False,
        help='Use LZF compression? (default: %(default)s)')
    return parser


def run(config):
    if 'hdf5' in config['dataset']:
        raise ValueError('Reading from an HDF5 file which you will probably be '
                         'about to overwrite! Override this error only if you know '
                         'what you''re doing!')
    # Get image size
    config['image_size'] = config['resolution']

    # Update compression entry
    config['compression'] = 'lzf' if config['compression'] else None  # No compression; can also use 'lzf'

    # Get dataset
    kwargs = {'num_workers': config['num_workers'], 'pin_memory': False, 'drop_last': False, 'distributed': False}

    transform = transforms.Compose([
        datasets.CenterCropLongEdge(),
        transforms.Resize(config['resolution']),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_loader = datasets.get_dataloaders(
        config['dataset'],
        config['data_root'],
        batch_size=config['batch_size'],
        shuffle=False,
        dataset_type='ImageFolder',
        transform=transform,
        **kwargs)[0]

    root = os.path.join(config['data_root'], config['dataset'])
    filename = f'{config["dataset"]}-{config["resolution"]}.hdf5'
    print(f'Saving new hdf5 file to : {os.path.join(root, filename)}')
    datasets.make_hdf5(train_loader, root=root, filename=filename)


def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
