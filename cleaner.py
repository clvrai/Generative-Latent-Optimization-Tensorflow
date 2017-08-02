import os
import os.path as osp
import argparse
import h5py

parser = argparse.ArgumentParser(description='Clean datasets for GLO.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['MNIST', 'SVHN', 'CIFAR10'])


def clean(dir):
    file_path = osp.join(dir, 'data.hdf5')
    if not osp.exists(path):
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        return

    f = h5py.File(file_path, 'r+')
    for key in f.keys():
        try:
            f[key].__delitem__('update')
        except:
            pass

if __name__ == '__main__':
    args = parser.parse_args()
    path = './datasets'
    if not osp.exists(path): os.mkdir(path)

    if 'MNIST' in args.datasets:
        clean('./datasets/mnist')
    if 'SVHN' in args.datasets:
        clean('./datasets/svhn')
    if 'CIFAR10' in args.datasets:
        clean('./datasets/cifar10')
