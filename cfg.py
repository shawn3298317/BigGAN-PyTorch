import os
from argparse import ArgumentParser
from collections import defaultdict

import torch.nn as nn


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    # Dataset/Dataloader stuff
    parser.add_argument(
        '--dataset', type=str, default='I128_hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100;'
        'Append "_hdf5" to use the hdf5 version for ISLVRC '
        '(default: %(default)s)')
    parser.add_argument(
        '--resolution', default=128, type=int)
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
        '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data (strongly recommended)? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--world-size', default=-1, type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank', default=-1, type=int,
        help='node rank for distributed training')
    parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help='distributed backend')
    parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training')
    parser.add_argument(
        '--gpu', default=None, type=int,
        help='GPU id to use.')
    parser.add_argument(
        '--dataset_type', type=str, default='ImageHDF5',
    )

    # Model stuff
    parser.add_argument(
        '--model', type=str, default='biggan',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--G_param', type=str, default='SN',
        help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
        ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
        ' or None (default: %(default)s)')
    parser.add_argument(
        '--G_ch', type=int, default=64,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', action='store_true', default=False,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
        '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=128,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--z_var', type=float, default=1.0,
        help='Noise variance: %(default)s)')
    parser.add_argument(
        '--hier', action='store_true', default=False,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--cross_replica', action='store_true', default=False,
        help='Cross_replica batchnorm in G?(default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
        '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D (underscore separated) '
        '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
        'ln [layernorm], gn [groupnorm] (default: %(default)s)')

    # Model init stuff
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
        ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='ortho',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='ortho',
        help='Init style to use for D(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
        '(default: %(default)s)')

    # Optimizer stuff
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')

    # Batch size, parallel, and precision stuff
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--G_batch_size', type=int, default=0,
        help='Batch size to use for G; if 0, same as D (default: %(default)s)')
    parser.add_argument(
        '--num_G_accumulations', type=int, default=1,
        help='Number of passes to accumulate G''s gradients over '
        '(default: %(default)s)')
    parser.add_argument(
        '--num_D_steps', type=int, default=2,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_D_accumulations', type=int, default=1,
        help='Number of passes to accumulate D''s gradients over '
        '(default: %(default)s)')
    parser.add_argument(
        '--split_D', action='store_true', default=False,
        help='Run D twice rather than concatenating inputs? (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=500,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--G_fp16', action='store_true', default=False,
        help='Train with half-precision in G? (default: %(default)s)')
    parser.add_argument(
        '--D_fp16', action='store_true', default=False,
        help='Train with half-precision in D? (default: %(default)s)')
    parser.add_argument(
        '--D_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in D? '
        '(default: %(default)s)')
    parser.add_argument(
        '--G_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in G? '
        '(default: %(default)s)')
    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
        '(default: %(default)s)')
    parser.add_argument('--use_torch_FID', action='store_true', default=False)
    parser.add_argument('--pretrained', default=None, type=str)

    # Bookkeping stuff
    parser.add_argument(
        '--G_eval_mode', action='store_true', default=False,
        help='Run G in eval mode (running/standing stats?) at sample/test time? '
        '(default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_save_copies', type=int, default=2,
        help='How many copies to save (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
        'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--no_fid', action='store_true', default=False,
        help='Calculate IS only, not FID? (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
        '(default: %(default)s)')
    parser.add_argument(
        '--hashname', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
        '(default: %(default)s)')
    parser.add_argument(
        '--base_root', type=str, default='',
        help='Default location to store all weights, samples, data, and logs '
        ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')
    parser.add_argument(
        '--pbar', type=str, default='mine',
        help='Type of progressbar to use; one of "mine" or "tqdm" '
        '(default: %(default)s)')
    parser.add_argument(
        '--name_suffix', type=str, default='',
        help='Suffix for experiment name for loading weights for sampling '
        '(consider "best0") (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
        '(default: %(default)s)')
    parser.add_argument(
        '--config_from_name', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
        '(default: %(default)s)')

    # EMA Stuff
    parser.add_argument(
        '--ema', action='store_true', default=False,
        help='Keep an ema of G''s weights? (default: %(default)s)')
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate (default: %(default)s)')
    parser.add_argument(
        '--use_ema', action='store_true', default=False,
        help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
    parser.add_argument(
        '--ema_start', type=int, default=0,
        help='When to start updating the EMA weights (default: %(default)s)')

    # Numerical precision and SV stuff
    parser.add_argument(
        '--adam_eps', type=float, default=1e-8,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')
    parser.add_argument(
        '--SN_eps', type=float, default=1e-8,
        help='epsilon value to use for Spectral Norm(default: %(default)s)')
    parser.add_argument(
        '--num_G_SVs', type=int, default=1,
        help='Number of SVs to track in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SVs', type=int, default=1,
        help='Number of SVs to track in D (default: %(default)s)')
    parser.add_argument(
        '--num_G_SV_itrs', type=int, default=1,
        help='Number of SV itrs in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SV_itrs', type=int, default=1,
        help='Number of SV itrs in D (default: %(default)s)')

    # Ortho reg stuff
    parser.add_argument(
        '--G_ortho', type=float, default=0.0,  # 1e-4 is default for BigGAN
        help='Modified ortho reg coefficient in G(default: %(default)s)')
    parser.add_argument(
        '--D_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in D (default: %(default)s)')
    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle D and G''s "requires_grad" settings when not training them? '
        ' (default: %(default)s)')

    # Which train function
    parser.add_argument(
        '--which_train_fn', type=str, default='GAN',
        help='How2trainyourbois (default: %(default)s)')

    # Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
        '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')

    # Log stuff
    parser.add_argument(
        '--logstyle', type=str, default='%3.3e',
        help='What style to use when logging training metrics?'
        'One of: %#.#f/ %#.#e (float/exp, text),'
        'pickle (python pickle),'
        'npz (numpy zip),'
        'mat (MATLAB .mat file) (default: %(default)s)')
    parser.add_argument(
        '--log_G_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in G? '
        '(default: %(default)s)')
    parser.add_argument(
        '--log_D_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in D? '
        '(default: %(default)s)')
    parser.add_argument(
        '--sv_log_interval', type=int, default=10,
        help='Iteration interval for logging singular values '
        ' (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False)

    return parser


def add_sample_parser(parser):
    """Arguments for sample.py; not presently used in train.py."""
    parser.add_argument(
        '--sample_npz', action='store_true', default=False,
        help='Sample "sample_num_npz" images and save to npz? '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_num_npz', type=int, default=50000,
        help='Number of images to sample when sampling NPZs '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_sheets', action='store_true', default=False,
        help='Produce class-conditional sample sheets and stick them in '
        'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_interps', action='store_true', default=False,
        help='Produce interpolation sheets and stick them in '
        'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_sheet_folder_num', type=int, default=-1,
        help='Number to use for the folder for these sample sheets '
        '(default: %(default)s)')
    parser.add_argument(
        '--sample_random', action='store_true', default=False,
        help='Produce a single random sheet? (default: %(default)s)')
    parser.add_argument(
        '--sample_trunc_curves', type=str, default='',
        help='Get inception metrics with a range of variances?'
        'To use this, specify a startpoint, step, and endpoint, e.g. '
        '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
        'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
        'not exactly identical to using tf.truncated_normal, but should '
        'have approximately the same effect. (default: %(default)s)')
    parser.add_argument(
        '--sample_inception_metrics', action='store_true', default=False,
        help='Calculate Inception metrics with sample.py? (default: %(default)s)')
    return parser


# Convenience dicts


def get_root_dirs(name, dataset_type='ImageHDF5', resolution=128, data_root='data'):
    root_dirs = {
        'ImageNet': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'ImageNet/train'), {}),
        },
        'Places365': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'Places365/train'), {}),
        },
        'Places365-Challenge': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'Places365/train'), {}),
        },
        'Hybrid1365': {
            'ImageHDF5': defaultdict(lambda: data_root, {}),
            'ImageDataset': defaultdict(lambda: data_root, {}),
        }
    }
    return root_dirs[name][dataset_type][resolution]


nclass_dict = {
    'ImageNet': 1000,
    'Places365': 365,
    'Places365-Challenge': 365,
    'Hybrid1365': 1365,
    'I32': 1000, 'I32_hdf5': 1000,
    'I64': 1000, 'I64_hdf5': 1000,
    'S128': 1, 'S128_hdf5': 1,
    'I128': 1000, 'I128_hdf5': 1000,
    'I256': 1000, 'I256_hdf5': 1000,
    'S256': 1, 'S256_hdf5': 1,
    'P64': 365, 'P64_hdf5': 365,
    'P128': 365, 'P128_hdf5': 365,
    'P256': 365, 'P256_hdf5': 365,
    'B64': 26, 'B64_hdf5': 26,
    'B128': 26, 'B128_hdf5': 26,
    'B256': 26, 'B256_hdf5': 26,
    'P128-Challenge': 365, 'P128-Challenge_hdf5': 365,
    'P256-Challenge': 365, 'P256-Challenge_hdf5': 365,
    'C10': 10, 'C100': 100}

# Number of classes to put per sample sheet
classes_per_sheet_dict = {
    'ImageNet': 16,
    'Places365': 16,
    'Places365-Challenge': 16,
    'I32': 50, 'I32_hdf5': 50,
    'I64': 20, 'I64_hdf5': 20,
    'I128': 20, 'I128_hdf5': 20,
    'I256': 20, 'I256_hdf5': 20,
    'S128': 1, 'S128_hdf5': 1,
    'S256': 1, 'S256_hdf5': 1,
    'P64': 20, 'P64_hdf5': 20,
    'P128': 20, 'P128_hdf5': 20,
    'P256': 10, 'P256_hdf5': 10,
    'B128': 13, 'B128_hdf5': 13,
    'B256': 13, 'B256_hdf5': 13,
    'P128-Challenge': 20, 'P128-Challenge_hdf5': 20,
    'P256-Challenge': 20, 'P256-Challenge_hdf5': 20,
    'C10': 10, 'C100': 100}

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True), }
