import errno
import functools
import os
import pickle
import re
import sys
import tarfile
import warnings

from operator import add

import h5py as h5
import numpy as np
import torch
import torch.hub
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm

import cfg

try:
    getattr(torch.hub, 'HASH_REGEX')
except AttributeError:
    torch.hub.HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def get_dataset(name, root_dir=None, resolution=128, dataset_type='ImageFolder',
                split='train', transform=None, target_transform=None, load_in_mem=False,
                download=False):

    if name == 'Hybrid1365':
        return get_hybrid_dataset(root_dir=root_dir, resolution=resolution,
                                  dataset_type=dataset_type, load_in_mem=load_in_mem)

    if dataset_type == 'ImageFolder':
        # Get torchivision dataset class for desired dataset.
        dataset_func = getattr(torchvision.datasets, name)

        if name in ['CIFAR10', 'CIFAR100']:
            kwargs = {'train': True if split == 'train' else False}
        else:
            kwargs = {'split': split}
        if name == 'CelebA':
            def tf(x):
                return 0 if target_transform is None else target_transform
            kwargs = {**kwargs, 'target_transform': tf}

        if transform is None:
            transform = transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])
        kwargs = {**kwargs,
                  'download': download,
                  'transform': transform}

        # Create dataset class based on config selection.
        dataset = dataset_func(root=root_dir, **kwargs)

    elif dataset_type == 'ImageHDF5':
        if download:
            raise NotImplementedError('Automatic Dataset Download not implemented yet...')

        hdf5_name = '{}-{}.hdf5'.format(name, resolution)
        hdf5_file = os.path.join(root_dir, hdf5_name)
        if not os.path.exists(hdf5_file):
            raise ValueError('Cannot find hdf5 file. You should download it, or create if yourself!')

        dataset = ImageHDF5(hdf5_file, load_in_mem=load_in_mem,
                            target_transform=target_transform)
    return dataset


def get_hybrid_dataset(root_dir=None, resolution=128, dataset_type='ImageHDF5', load_in_mem=False):
    imagenet_root = cfg.get_root_dirs('ImageNet', dataset_type=dataset_type,
                                      resolution=resolution, data_root=root_dir)
    places365_root = cfg.get_root_dirs('Places365', dataset_type=dataset_type,
                                       resolution=resolution, data_root=root_dir)
    imagenet_dataset = get_dataset('ImageNet', resolution=resolution,
                                   dataset_type=dataset_type, load_in_mem=load_in_mem,
                                   root_dir=imagenet_root)
    placess365_dataset = get_dataset('Places365', resolution=resolution,
                                     dataset_type=dataset_type, load_in_mem=load_in_mem,
                                     target_transform=functools.partial(add, 1000),
                                     root_dir=places365_root)
    return torch.utils.data.ConcatDataset((imagenet_dataset, placess365_dataset))


def get_dataloaders(dataset, data_root=None, resolution=128, dataset_type='ImageHDF5',
                    batch_size=64, num_workers=8, shuffle=True, load_in_mem=False,
                    pin_memory=True, drop_last=True, distributed=False, **kwargs):
    root_dir = cfg.get_root_dirs(dataset, dataset_type, resolution,
                                 data_root=data_root)
    dataset = get_dataset(name=dataset,
                          root_dir=root_dir,
                          resolution=resolution,
                          dataset_type=dataset_type,
                          load_in_mem=load_in_mem)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last)
    return [loader]


def get_data_loaders_old(dataset, data_root=None, augment=False, batch_size=64,
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
                train_transform = [RandomCropLongEdge(),
                                   transforms.Resize(image_size),
                                   transforms.RandomHorizontalFlip()]
        else:
            print('Data will not be augmented...')
            if dataset in ['C10', 'C100']:
                train_transform = []
            else:
                train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]
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


ROOT_URL = 'http://ganocracy.csail.mit.edu/data/'
data_urls = {
    'hdf5': {
        'BuildingsHQ-64.hdf5': {
            'url': os.path.join(ROOT_URL, 'BuildingsHQ-64.hdf5'),
            'md5': ','
        },
        'BuildingsHQ-128.hdf5': {
            'url': os.path.join(ROOT_URL, 'BuildingsHQ-128.hdf5'),
            'md5': 'b2c0f7129d6dd117d9dda2a7098870f1'
        },
        'places365-64.hdf5': {
            'url': os.path.join(ROOT_URL, 'BuildingsHQ-64.hdf5'),
            'md5': ','
        },
        'places365-128.hdf5': {
            'url': os.path.join(ROOT_URL, 'BuildingsHQ-128.hdf5'),
            'md5': 'b2c0f7129d6dd117d9dda2a7098870f1'
        }
    },
    'celeba': {
        'tar': os.path.join(ROOT_URL, 'celeba-054b22a6.tar.gz')
    },
    'buildings_hq': {
        'tar': os.path.join(ROOT_URL, 'buildings_hq.tar.gz'),
        'hdf5': {
            '128': os.path.join(ROOT_URL, 'B128.hdf5'),
            '256': os.path.join(ROOT_URL, 'B256.hdf5'),
        },
    },
    'satellite_images': {
        'tar': os.path.join(ROOT_URL, 'satellite_images-79716c2f.tar.gz')
    },
    'imagenet': {
        'tar': os.path.join(ROOT_URL, 'imagenet.tar.gz'),
        'hdf5': {
            '64': os.path.join(ROOT_URL, 'I64.hdf5'),
            '128': os.path.join(ROOT_URL, 'I128.hdf5'),
            '256': os.path.join(ROOT_URL, 'I256.hdf5'),
        },
    },
    'places365': {
        'tar': os.path.join(ROOT_URL, 'places365.tar.gz'),
        'hdf5': {
            '64': os.path.join(ROOT_URL, 'P64.hdf5'),
            '128': os.path.join(ROOT_URL, 'P128.hdf5'),
            '256': os.path.join(ROOT_URL, 'P256.hdf5'),
        }
    }
}


def load_data_from_url(url, root_dir=None, progress=True):
    cached_file = _load_file_from_url(url, root_dir=root_dir, progress=progress)
    # match = torch.hub.HASH_REGEX.search(cached_file)
    # data_dir = cached_file[:match.start()]

    with tarfile.open(cached_file) as tf:
        name = tf.getnames()[0]
    data_dir = os.path.join(root_dir, name)

    if not os.path.exists(data_dir):
        print(f'Extracting:  "{cached_file}" to {data_dir}')
        with tarfile.open(name=cached_file) as tar:
            # Go over each member
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                # Extract member
                tar.extract(member=member, path=root_dir)

        # tf = tarfile.open(cached_file)
        # print(f'Extracting to: {data_dir}')
        # tf.extractall(path=root_dir)
        # print(f'finished extracting to: {root_dir}')
    else:
        print(f'Data found at: {data_dir}')
    return data_dir


def _load_file_from_url(url, root_dir=None, progress=True):
    r"""Loads the dataset file from the given URL.

    If the object is already present in `root_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        data_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr

    # 'https://pytorch.org/docs/stable/_modules/torch/hub.html#load'

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if root_dir is None:
        torch_home = torch.hub._get_torch_home()
        root_dir = os.path.join(torch_home, 'data')

    try:
        os.makedirs(root_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(root_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = torch.hub.HASH_REGEX.search(filename).group(1)
        torch.hub._download_url_to_file(url, cached_file, hash_prefix, progress)
    return cached_file


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in tqdm(sorted(fnames)):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename=None, **kwargs):
        classes, class_to_idx = find_classes(root)

        # Load pre-computed image directory walk
        if index_filename is None:
            index_filename = os.path.join(root, os.path.basename(root) + '.npz')

        if os.path.exists(index_filename):
            print('Loading pre-saved index file {}'.format(index_filename))
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating index file {}'.format(index_filename))
            imgs = make_dataset(root, class_to_idx)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = self.transform(imgs[index][0]), imgs[index][1]
                self.data.append(self.loader(path))
                self.labels.append(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            try:
                img = self.loader(str(path))
            except OSError:
                return self.__getitem__(min(index + 1, len(self)))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of classes: {}\n'.format(len(self.classes))
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SingleImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename='imagenet_imgs.npz', **kwargs):
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved index file {}'.format(index_filename))
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating index file {}'.format(index_filename))
            imgs = []
            fnames = os.listdir(root)
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    imgs.append(item)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all {} images into memory...'.format(self.root))
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = self.transform(imgs[index][0]), imgs[index][1]
                self.data.append(self.loader(path))
                self.labels.append(target)


def hdf5_transform(img):
    return ((torch.from_numpy(img).float() / 255) - 0.5) * 2


class ImageHDF5(data.Dataset):

    def __init__(self, hdf5_file, transform=hdf5_transform, target_transform=None,
                 load_in_mem=False, train=True, download=True):

        self.hdf5_file = hdf5_file
        # Set the transforms here.
        self.transform = transform
        self.target_transform = target_transform
        with h5.File(hdf5_file, 'r') as f:
            self.num_imgs = len(f['labels'])
            self.num_classes = len(np.unique(f['labels'][:]))

        # Load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now.
        if self.load_in_mem:
            print('Loading {} into memory...'.format(hdf5_file))
            with h5.File(hdf5_file, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.hdf5_file, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if target is None:
            target = 0
        return img, int(target)

    def __len__(self):
        return self.num_imgs

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of classes: {}\n'.format(self.num_classes)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    HDF5 File: {}\n'.format(self.hdf5_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def make_hdf5(dataloader, root, filename, chunk_size=500, compression=False):
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        _make_hdf5(dataloader, root, filename,
                   chunk_size=chunk_size,
                   compression=compression)
    else:
        print('HDF5 file {} already exists!'.format(path))
    return path


def _make_hdf5(dataloader, root, filename, chunk_size=500, compression=False):
    # HDF5 supports chunking and compression. You may want to experiment
    # with different chunk sizes to see how it runs on your machines.
    # Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
    # 1 / None                   20/s
    # 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
    # 500 / LZF                                         8/s                   56GB                  23min
    # 1000 / None                78/s
    # 5000 / None                81/s
    # auto:(125,1,16,32) / None                         11/s                  61GB

    print('Starting to load {} into an HDF5 file with chunk size {} and compression {}...'.format(filename, chunk_size, compression))

    # Loop over train loader
    dataset_len = len(dataloader.dataset)
    for i, (x, y) in enumerate(tqdm(dataloader)):
        # Stick x into the range [0, 255] since it's coming from the train loader
        x = (255 * ((x + 1) / 2.0)).byte().numpy()
        # Numpyify y
        y = y.numpy()

        if i == 0:  # If we're on the first batch, prepare the hdf5.
            with h5.File(os.path.join(root, filename), 'w') as f:
                print('Producing dataset of len {}'.format(dataset_len))
                maxshape = (dataset_len, x.shape[-3], x.shape[-2], x.shape[-1])
                chunks = (chunk_size, x.shape[-3], x.shape[-2], x.shape[-1])
                imgs_dset = f.create_dataset('imgs', x.shape, dtype='uint8',
                                             maxshape=maxshape,
                                             chunks=chunks,
                                             compression=compression)
                print('Image chunks chosen as {}'.format(imgs_dset.chunks))
                imgs_dset[...] = x
                maxshape = (dataset_len,)
                chunks = (chunk_size,)
                labels_dset = f.create_dataset('labels', y.shape, dtype='int64',
                                               maxshape=maxshape,
                                               chunks=chunks,
                                               compression=compression)
                print('Label chunks chosen as {}'.format(labels_dset.chunks))
                labels_dset[...] = y

        else:  # Append to the hdf5.
            with h5.File(os.path.join(root, filename), 'a') as f:
                f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
                f['imgs'][-x.shape[0]:] = x
                f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
                f['labels'][-y.shape[0]:] = y


def old_old_get_dataset(name, root_dir=None, resolution=128, dataset_type='ImageFolder',
                        download=True, split='train', transform=None, target_transform=None,
                        load_in_mem=False):
    if name == 'Custom':
        pass

    if dataset_type == 'ImageFolder':
        # Get torchivision dataset class for desired dataset.
        dataset_func = getattr(torchvision.datasets, name)

        if name in ['CIFAR10', 'CIFAR100']:
            kwargs = {'train': True if split == 'train' else False}
        else:
            kwargs = {'split': split}
        if name == 'CelebA':
            def tf(x):
                return 0 if target_transform is None else target_transform
            kwargs = {**kwargs, 'target_transform': tf}

        if transform is None:
            transform = transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])
        kwargs = {**kwargs,
                  'download': download,
                  'transform': transform}

        # Create dataset class based on config selection.
        dataset = dataset_func(root=root_dir, **kwargs)

    elif dataset_type == 'ImageHDF5':

        hdf5_name = '{}-{}.hdf5'.format(name, resolution)
        hdf5_file = os.path.join(root_dir, hdf5_name)
        if not os.path.exists(hdf5_file):
            if download:
                d = data_urls['hdf5'][hdf5_name]
                url, md5 = d['url'], d['md5']
                torchvision.datasets.utils.download_url(url, root_dir, filename=hdf5_name, md5=md5)
            raise ValueError('Cannot find hdf5 file. You need to set=download it, or create if yourself!')

        def target_transform(x):
            return 0 if name == 'CelebA' else None

        dataset = ImageHDF5(hdf5_file, load_in_mem=load_in_mem,
                            target_transform=target_transform)

    return dataset


def get_dataset_old(root_dir, name, resolution, filetype, ):
    if filetype == 'tar':
        url = data_urls[name]['tar']
        data_dir = load_data_from_url(url, root_dir)
        dataset = ImageFolder(root=data_dir,
                              transform=transforms.Compose([
                                  transforms.CenterCropLongEdge(),
                                  transforms.Resize(resolution),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))
                              ]))
    elif filetype == 'hdf5':
        url = data_urls[name]['hdf5'][resolution]
        hdf5_file = load_data_from_url(url, root_dir)
        dataset = ImageHDF5(hdf5_file)
    else:
        raise ValueError('Unreconized filetype: {}'.format(filetype))

    return dataset


def old_get_dataset(name, hdf5=True, size=64, targets=False):
    pass


def imagenet(data_dir, size=64, targets=False):
    pass


def places365(data_dir, size=64, targets=False):
    pass


def ffhq(data_dir, size=64, targets=False):
    pass


class CIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True, validate_seed=0,
                 val_split=0, load_in_mem=True, **kwargs):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.val_split = val_split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.'
                               + ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.labels += entry['labels']
            else:
                self.labels += entry['fine_labels']
            fo.close()

        self.data = np.concatenate(self.data)
        # Randomly select indices for validation
        if self.val_split > 0:
            label_indices = [[] for _ in range(max(self.labels) + 1)]
            for i, l in enumerate(self.labels):
                label_indices[l] += [i]
            label_indices = np.asarray(label_indices)

            # randomly grab 500 elements of each class
            np.random.seed(validate_seed)
            self.val_indices = []
            for l_i in label_indices:
                self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1), replace=False)])

        if self.train == 'validate':
            self.data = self.data[self.val_indices]
            self.labels = list(np.asarray(self.labels)[self.val_indices])

            self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        elif self.train:
            print(np.shape(self.data))
            if self.val_split > 0:
                self.data = np.delete(self.data, self.val_indices, axis=0)
                self.labels = list(np.delete(np.asarray(self.labels), self.val_indices, axis=0))

            self.data = self.data.reshape((int(50e3 * (1. - self.val_split)), 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return torchvision.transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class RandomCropLongEdge(object):
    """Crops the given PIL Image on the long edge with a random start point.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
        return torchvision.transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__
