from . import datasets
from . import transforms
import cfg
from utils import MultiEpochSampler
from torch.utils.data.dataloader import DataLoader

__all__ = ['datasets', 'transforms']


def get_data_loaders(dataset, data_root=None, augment=False, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):

    # Convenience function to centralize all data loaders
    # Append /FILENAME.hdf5 to root if using hdf5
    data_root += '/%s' % cfg.root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = cfg.dset_dict[dataset]
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    image_size = cfg.imsize_dict[dataset]
    # For image folder datasets, name of the file where we store the precomputed
    # image locations to avoid having to walk the dirs every time we load.
    dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset}

    # HDF5 datasets have their own inbuilt transform, no need to train_transform
    if 'hdf5' in dataset:
        train_transform = None
    else:
        if augment:
            print('Data will be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = [transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip()]
            else:
                train_transform = [transforms.RandomCropLongEdge(),
                                   transforms.Resize(image_size),
                                   transforms.RandomHorizontalFlip()]
        else:
            print('Data will not be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = []
            else:
                train_transform = [transforms.CenterCropLongEdge(), transforms.Resize(image_size)]
            # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
        train_transform = transforms.Compose(train_transform + [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
    train_set = which_dataset(root=data_root, transform=train_transform,
                              load_in_mem=load_in_mem, **dataset_kwargs)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders
