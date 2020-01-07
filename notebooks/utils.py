import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(logfile):
    with open(logfile) as f:
        return [json.loads(x.strip()) for x in f]


def get_logs(log_dir):
    return {exp: load_jsonl(os.path.join(log_dir, exp))
            for exp in os.listdir('logs')
            if exp.endswith('.jsonl')}


def plot_log(name, log):

    itrs = [x['itr'] for x in log]
    IS_scores = [x['IS_mean'] for x in log]
    plt.plot(itrs, IS_scores, label=name)

    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Inception Score', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    plt.show()


def plot_logs(logs):
    for name, log in logs.items():
        name = '_'.join(name.split('_'))
        itrs = [x['itr'] for x in log]
        IS_scores = [x['IS_mean'] for x in log]
        plt.plot(itrs, IS_scores, label=name)

    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Inception Score', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    plt.show()


def smooth_data(data, amount=1.0):
    if not amount > 0.0:
        return data
    data_len = len(data)
    ksize = int(amount * (data_len // 2))
    kernel = np.ones(ksize) / ksize
    return np.convolve(data, kernel, mode='same')


def parse_log(logfile):
    seen = {}
    with open(logfile) as f:
        for x in f:
            itr, val = x.strip().split(': ')
            if itr not in seen:
                seen[itr] = val
        values = seen.values()
    return list(map(float, values))


def _parse_log(logfile):
    with open(logfile) as f:
        values = [x.strip().split(': ')[1] for x in f]
    return list(map(float, values))


def load_logs(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    sv_logs = {f: parse_log(os.path.join(log_dir, f))
               for f in log_files if 'sv0.log' in f}
    loss_logs = {f: parse_log(os.path.join(log_dir, f))
                 for f in log_files if 'loss' in f}
    return {'loss': loss_logs, 'sv': sv_logs}


def plot_loss_logs(logs, smoothing=0.01, figsize=(15, 15)):
    G_loss = logs['G_loss.log']
    D_loss = [x + y for x, y in zip(logs['D_loss_real.log'], logs['D_loss_fake.log'])]
    G_loss = smooth_data(G_loss, amount=smoothing)
    D_loss = smooth_data(D_loss, amount=smoothing)
    plt.figure(figsize=figsize)
    plt.plot(range(len(G_loss)), G_loss, label='G_loss')
    plt.plot(range(len(D_loss)), D_loss, label='D_loss')
    plt.legend(loc='lower right', fontsize='medium')
    plt.xlabel('Iteration', fontsize='x-large')
    plt.ylabel('Losses', fontsize='x-large')
    plt.title('Training History', fontsize='xx-large')
    # plt.gca().set_ylim(top=10, bottom=-10)
    plt.gca().set_ylim(top=10, bottom=-10)
    plt.show()


def plot_sv_logs(logs):
    fig, axs = plt.subplots(2)
    plt.title('Training History', fontsize='xx-large')
    for name, log in logs.items():
        itrs = [i * 10 for i in range(len(log))]
        idx = 0 if name[0] == 'G' else 1
        axs[idx].plot(itrs, log)

    for label, ax in zip([r'G $\sigma_0$', r'D $\sigma_0$'], axs.flat):
        ax.set(ylabel=label)
    plt.xlabel('Iteration', fontsize='x-large')

    plt.show()


def plot_sv_logs(logs, figsize=(15, 15)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
#     plt.title('Training History', fontsize='xx-large')
    for name, log in logs.items():
        itrs = [i * 10 for i in range(len(log))]
        idx = 0 if name[0] == 'G' else 1
        axs[idx].plot(itrs, log)

    for label, ax in zip([r'G $\sigma_0$', r'D $\sigma_0$'], axs.flat):
        ax.set(ylabel=label, xlabel="Iteration")
#     plt.xlabel('Iteration', fontsize='x-large')

    plt.show()


def plot_truncation_curves(trunc_file):
    with open(trunc_file) as f:
        coords = [list(map(float, line.strip().split(' '))) for line in f]
        x, y = list(zip(*coords))
        plt.plot(x, y, '-')
        plt.gca().invert_yaxis()
        plt.xlabel('Inception Score')
        plt.ylabel('FID')


def print_stats(logs, blacklist=[]):
    for name, log in logs.items():
#         if name in blacklist:
#             continue
        print_name = '_'.join(name.split('_')[:16])
        if print_name in blacklist:
            continue
        max_IS_idx = np.argmax([x['IS_mean'] for x in log])
        min_FID_idx = np.argmin([x['FID'] for x in log])
        last_itr = log[-1]['itr']

        max_IS = log[max_IS_idx]['IS_mean']
        min_FID = log[min_FID_idx]['FID']
        max_IS_itr = log[max_IS_idx]['itr']
        min_FID_itr = log[min_FID_idx]['itr']
        print(f'{print_name}\n'
              f'\t current itr: {last_itr}\n'
              f'\t current IS: {log[-1]["IS_mean"]:.3f}\n'
              f'\t current FID: {log[-1]["FID"]:.3f}\n'
              f'\t max IS: {max_IS:.3f} at itr ({max_IS_itr})\n'
              f'\t min FID: {min_FID:.3f} at itr ({min_FID_itr})')


def plot_IS_FID(logs):
    fig, axs = plt.subplots(2, sharex=True)
    for name, log in logs.items():
        name = '_'.join(name.split('_'))
        itrs = [x['itr'] for x in log]
        IS_scores = [x['IS_mean'] for x in log]
        FID = [x['FID'] for x in log]
        axs[0].plot(itrs, IS_scores, label=name)
        axs[1].semilogy(itrs, FID, label=name)

    for label, ax in zip(['Inception Score', 'FID'], axs.flat):
        ax.set(ylabel=label)

    plt.xlabel('Iteration', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='x-small')
    axs[0].set_title('Training History', fontsize='xx-large')
    fig.tight_layout()
    plt.show()
    
