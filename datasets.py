import os
import pickle
import re
import sys

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
import h5py as h5

try:
    getattr(torch.hub, 'HASH_REGEX')
except AttributeError:
    torch.hub.HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def get_dataset(name, root=None, resolution=128, dataset_type='ImageFolder',
                split='train', transform=None, target_transform=None, load_in_mem=False,
                ):

    if dataset_type == 'ImageFolder':
        if transform is None:
            transform = transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])
        kwargs = {'transform': transform,
                  'target_transform': target_transform,
                  'index_filename': f'{name}_{split}.npz',
                  }

        # Create dataset class based on config selection.
        dataset = ImageFolder(root=root, **kwargs)

    elif dataset_type == 'ImageHDF5':

        hdf5_name = '{}-{}.hdf5'.format(name, resolution)
        hdf5_file = os.path.join(root, hdf5_name)
        if not os.path.exists(hdf5_file):
            raise ValueError('Cannot find hdf5 file. You should download it, or create if yourself!')

        dataset = ImageHDF5(hdf5_file, load_in_mem=load_in_mem,
                            target_transform=target_transform)
    return dataset


def get_dataloaders(dataset, data_root=None, resolution=128, dataset_type='ImageHDF5',
                    batch_size=64, num_workers=8, shuffle=True, load_in_mem=False,
                    pin_memory=True, drop_last=True, distributed=False, **kwargs):
    root = cfg.get_root_dirs(dataset, dataset_type, resolution,
                             data_root=data_root)
    dataset = get_dataset(name=dataset,
                          root=root,
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
