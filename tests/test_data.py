import pytest

import torch

import cfg
import datasets

DATA_ROOT = 'data'


@pytest.mark.parametrize('name, resolution, dataset_type', [
    ('Places365', 64, 'ImageHDF5'),
    ('Places365', 128, 'ImageHDF5'),
    ('ImageNet', 64, 'ImageHDF5'),
    ('ImageNet', 128, 'ImageHDF5'),
    ('Hybrid1365', 64, 'ImageHDF5'),
    ('Hybrid1365', 128, 'ImageHDF5'),
])
def test_get_dataset_hdf5(name, resolution, dataset_type):
    root_dir = cfg.get_root_dirs(name, dataset_type, resolution,
                                 data_root=DATA_ROOT)
    dataset = datasets.get_dataset(name=name,
                                   root_dir=root_dir,
                                   resolution=resolution,
                                   dataset_type=dataset_type)
    img, label = dataset[0]
    assert label == 0
    assert img.shape == torch.Size((3, resolution, resolution))


@pytest.mark.parametrize('name, resolution, dataset_type', [
    ('Places365', 64, 'ImageHDF5'),
    ('Places365', 128, 'ImageHDF5'),
    ('ImageNet', 64, 'ImageHDF5'),
    ('ImageNet', 128, 'ImageHDF5'),
])
def test_get_dataloader_hdf5(name, resolution, dataset_type):
    MAX_ITERS = 10
    BATCH_SIZE = 64
    loader = datasets.get_dataloaders(
        name, data_root=DATA_ROOT, resolution=resolution, dataset_type=dataset_type,
        batch_size=BATCH_SIZE, num_workers=4, shuffle=True, load_in_mem=False, pin_memory=True,
        drop_last=True, distributed=False)[0]

    for i, (x, y) in enumerate(loader):
        if i >= MAX_ITERS:
            break

        assert y.shape == torch.Size((BATCH_SIZE,))
        assert x.shape == torch.Size((BATCH_SIZE, 3, resolution, resolution))
