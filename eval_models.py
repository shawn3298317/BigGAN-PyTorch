import os

import cfg
import sample

WEIGHTS_ROOT = 'weights'
blacklist = [
    'archive',
    'backup',
]


def get_experiments(root, blacklist):
    return [exp for exp in os.listdir(WEIGHTS_ROOT) if exp not in blacklist]


def get_config(exp_name, load_weights=''):
    print(exp_name)
    parser = cfg.prepare_parser()
    parser = cfg.add_sample_parser(parser)
    config = vars(parser.parse_args([]))

    config['config_from_name'] = True
    config['experiment_name'] = exp_name
    config['weights_root'] = WEIGHTS_ROOT
    config['load_weights'] = load_weights

    config['use_ema'] = True
    config['parallel'] = True
    config['G_eval_mode'] = True
    config['G_batch_size'] = 32
    config['batch_size'] = 32

    config['sample_npz'] = False
    config['sample_random'] = False
    config['sample_sheets'] = False
    config['sample_interps'] = False
    config['sample_inception_metrics'] = True
    config['sample_trunc_curves'] = '0.05_0.05_1.0'
    return config


def run(exp_name, load_weights=''):
    config = get_config(exp_name, load_weights=load_weights)
    sample.run(config)


def test():
    for exp_name in get_experiments(WEIGHTS_ROOT, blacklist):
        run(exp_name)


if __name__ == '__main__':
    test()
