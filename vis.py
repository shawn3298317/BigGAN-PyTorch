import functools
import os
import sys
import imageio

import numpy as np
import torch
import torchvision

from pretorched import visualizers as vutils

import utils
import models


DEVICE = 'cuda'
BATCH_SIZE = 64
WEIGHTS_DIR_256 = 'weights/BigGAN_Places365_256_biggan_seed0_Gch96_Dch96_bs128_nDs2_Glr2.0e-04_Dlr5.0e-05_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gortho1.0e-04_Gshared_hier_ema'
WEIGHTS_DIR = WEIGHTS_DIR_256
SAMPLES_DIR = os.path.join(WEIGHTS_DIR.replace('weights', 'samples'), 'plots')

with open('data/Places365/categories_places365.txt') as f:
    categories = ['-'.join(x.split(' ')[0].split('/')[2:]) for x in f]


def load_model():
    state_dict = torch.load(os.path.join(WEIGHTS_DIR, 'state_dict.pth'))
    config = state_dict['config']
    model = getattr(models, config['model'])
    G = model.Generator(**config).to(DEVICE).eval()
    G.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'G_ema.pth'), map_location=DEVICE))
    return G


def z_interpolation(G, outfile, label=1, num_samples=32, num_midpoints=8,
                    minibatch_size=8, trunc=1.0, use_trunc=True):
    "Intra-class (z only) Latent space interpolation."

    # Choose two coordinates to interpolate between.
    if use_trunc:
        z0 = vutils.truncated_z_sample(num_samples, G.dim_z, device=DEVICE, truncation=trunc)
        z1 = vutils.truncated_z_sample(num_samples, G.dim_z, device=DEVICE, truncation=trunc)
    else:
        z0 = torch.randn(num_samples, G.dim_z).to(DEVICE)
        z1 = torch.randn(num_samples, G.dim_z).to(DEVICE)

    # Interpolate between z0 and z1.
    zs = vutils.interp(z0, z1, num_midpoints, device=DEVICE)
    zs = zs.view(-1, zs.size(-1))

    # Choose a random class for each row of interpolations.
    ys = torch.cat([torch.ones(num_midpoints + 2).long() * label
                    for _ in range(num_samples)]).long().to(DEVICE)

    g = functools.partial(G, embed=True)
    with torch.no_grad():
        samples = utils.elastic_gan(g, zs, ys)
        # Split batches into mini-batches so that it fits in memory.
        # samples = torch.cat([G(z, y, embed=True) for z, y in zip(zs.split(minibatch_size), ys.split(minibatch_size))])
    torchvision.utils.save_image(samples.cpu(), outfile, nrow=num_midpoints + 2, normalize=True)


def z_classwise_interpolation(G, outfile, label=1, num_samples=32, num_midpoints=8, minibatch_size=8):
    dev = next(G.parameters()).device
    x0 = torch.randn(num_samples, G.dim_z).to(dev)
    x1 = torch.randn(num_samples, G.dim_z).to(dev)
    zs = vutils.interp(x0, x1, num_midpoints, device=dev)
    zs = zs.view(-1, zs.size(-1))

    class_a = G.shared(torch.ones(num_samples, device=dev).long() * label)
    # class_b = G.shared(torch.ones(num_samples, device=dev).long() * torch.randint(G.n_classes, (1,), device=dev))
    class_b = G.shared(torch.randint(G.n_classes, (num_samples,), device=dev))
    ys = vutils.interp(class_a, class_b, num_midpoints, device=dev)
    ys = ys.view(-1, ys.size(-1))

    # g = functools.partial(G, embed=True)
    with torch.no_grad():
        # samples = utils.elastic_gan(g, zs, ys)
        # Split batches into mini-batches so that it fits in memory.
        samples = torch.cat([G(z, y).cpu() for z, y in zip(zs.split(minibatch_size), ys.split(minibatch_size))])
    torchvision.utils.save_image(samples.cpu(), outfile, nrow=num_midpoints + 2, normalize=True)


def classwise_interpolation(G, outfile, label=1, num_samples=32, num_midpoints=8, minibatch_size=8):

    # Class-wise interpolation
    # Inter-class (z,y only) Latent space interpolation
    dev = next(G.parameters()).device
    x0 = torch.randn(num_samples, G.dim_z).to(dev)
    zs = vutils.interp(x0, x0, num_midpoints, device=dev)
    zs = zs.view(-1, zs.size(-1))

    class_a = G.shared(torch.ones(num_samples, device=dev).long() * label)
    # class_b = G.shared(torch.ones(num_samples, device=dev).long() * torch.randint(G.n_classes, (1,), device=dev))
    class_b = G.shared(torch.randint(G.n_classes, (num_samples,), device=dev))
    ys = vutils.interp(class_a, class_b, num_midpoints, device=dev)
    ys = ys.view(-1, ys.size(-1))

    g = functools.partial(G, embed=True)
    with torch.no_grad():
        samples = utils.elastic_gan(g, zs, ys).cpu()
        # Split batches into mini-batches so that it fits in memory.
        # samples = torch.cat([G(z, y).cpu() for z, y in zip(zs.split(minibatch_size), ys.split(minibatch_size))])
    torchvision.utils.save_image(samples.cpu(), outfile, nrow=num_midpoints + 2, normalize=True)


def z_truncations(G, outfile, label=1, minibatch_size=4, truncations=[2.0, 1.0, 0.5, 0.04]):
    samples = []
    y = torch.ones((minibatch_size,), device=DEVICE).long() * label
    g = functools.partial(G, embed=True)
    with torch.no_grad():
        for trunc in truncations:
            z = vutils.truncated_z_sample(minibatch_size, G.dim_z, truncation=trunc)
            # out = G(z, y, embed=True).cpu()
            out = utils.elastic_gan(g, z, y).cpu()
            grid = torchvision.utils.make_grid(out, nrow=2, normalize=True).permute(1, 2, 0)
            samples.append(grid.numpy())
    samples = 255 * np.concatenate(samples, 1)
    imageio.imwrite(outfile, samples.astype('uint8'))


def make_z_interpolations(G, num_samples=16, num_midpoints=6, minibatch_size=8):
    outdir_z = os.path.join(SAMPLES_DIR, 'z_interp')
    os.makedirs(outdir_z, exist_ok=True)
    for label, cat in enumerate(categories):
        fname = f'{label}_{cat}_interp.jpg'
        outfile = os.path.join(outdir_z, fname)
        print(f'z interpolation for: {fname}...')
        z_interpolation(G, outfile, label=label,
                        num_samples=num_samples,
                        num_midpoints=num_midpoints,
                        minibatch_size=minibatch_size)


def make_classwise_interpolations(G, num_samples=16, num_midpoints=6, minibatch_size=8):
    outdir_classwise = os.path.join(SAMPLES_DIR, 'classwise_interp')
    os.makedirs(outdir_classwise, exist_ok=True)
    for label, cat in enumerate(categories):
        fname = f'{label}_{cat}_interp.jpg'
        outfile = os.path.join(outdir_classwise, fname)
        print(f'Classwise interpolation for: {fname}...')
        classwise_interpolation(G, outfile,
                                label=label,
                                num_samples=num_samples,
                                num_midpoints=num_midpoints,
                                minibatch_size=minibatch_size)


def make_z_classwise_interpolations(G, num_samples=16, num_midpoints=6, minibatch_size=8):
    outdir_classwise = os.path.join(SAMPLES_DIR, 'z_classwise_interp')
    os.makedirs(outdir_classwise, exist_ok=True)
    for label, cat in enumerate(categories):
        fname = f'{label}_{cat}_interp.jpg'
        outfile = os.path.join(outdir_classwise, fname)
        print(f'Classwise interpolation for: {fname}...')
        z_classwise_interpolation(G, outfile,
                                  label=label,
                                  num_samples=num_samples,
                                  num_midpoints=num_midpoints,
                                  minibatch_size=minibatch_size)


def make_z_truncations(G, minibatch_size=4, truncations=[2.0, 1.0, 0.5, 0.04]):
    outdir_z = os.path.join(SAMPLES_DIR, 'z_truncations')
    os.makedirs(outdir_z, exist_ok=True)
    for label, cat in enumerate(categories):
        fname = f'{label}_{cat}_interp.jpg'
        outfile = os.path.join(outdir_z, fname)
        print(f'z truncations for: {fname}...')
        z_truncations(G, outfile, label=label,
                      truncations=truncations,
                      minibatch_size=minibatch_size)


def make_category_plots(G, minibatch_size=8, num_samples=64, truncs=[0.5, 1.0], z_vars=[0.45]):
    outdir_categories = os.path.join(SAMPLES_DIR, 'categories')
    os.makedirs(outdir_categories, exist_ok=True)
    g = functools.partial(G, embed=True)
    with torch.no_grad():
        trunc_zs = [vutils.truncated_z_sample(num_samples, G.dim_z, device=DEVICE, truncation=t) for t in truncs]
        zs = [utils.prepare_z_y(num_samples, G.dim_z, G.n_classes, device=DEVICE, z_var=z_var)[0] for z_var in z_vars]
        for label, cat in enumerate(categories):
            y = torch.ones((num_samples,), device=DEVICE).long() * label

            for z_var, z in zip(z_vars, zs):
                fname = f'{label}_{cat}_zvar{z_var:.2f}.jpg'
                print(f'Plotting: {fname}...')
                outfile = os.path.join(outdir_categories, fname)
                out = utils.elastic_gan(g, z, y).cpu()
                torchvision.utils.save_image(out, outfile, normalize=True)

            for trunc, trunc_z in zip(truncs, trunc_zs):
                fname = f'{label}_{cat}_trunc{trunc:.2f}.jpg'
                print(f'Plotting: {fname}...')
                outfile = os.path.join(outdir_categories, fname)
                out = utils.elastic_gan(g, trunc_z, y).cpu()
                torchvision.utils.save_image(out, outfile, normalize=True)


def plot_truncations():
    G = load_model()
    make_z_truncations(G)


def plot_categories():
    G = load_model()
    make_category_plots(G)


def plot_z_classwise_interpolations():
    G = load_model()
    make_z_classwise_interpolations(G)


def plot_classwise_interpolations():
    G = load_model()
    make_classwise_interpolations(G)


def plot_all():
    G = load_model()
    make_z_truncations(G)
    make_z_interpolations(G)
    make_classwise_interpolations(G)
