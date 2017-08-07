from __future__ import print_function
import os
import os.path as osp
import tarfile
import subprocess
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Download dataset for GLO.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['MNIST', 'SVHN', 'CIFAR10'])
parser.add_argument('--dimension', type=int, default=100)
parser.add_argument('--distribution', type=str, default='Uniform', choices=['Uniform', 'Gaussian', 'PCA'])


def pca_feature(X, d):
    X = X/255.
    from sklearn.decomposition import PCA
    X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    pca = PCA(n_components=d)
    return pca.fit_transform(X)


def prepare_h5py(train_image, test_image, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    if args.distribution == 'PCA':
        print('Performing PCA...')
        Y = pca_feature(image, args.dimension)

    import progressbar
    bar = progressbar.ProgressBar(
        maxval=100,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    )
    bar.start()

    f = h5py.File(osp.join(data_dir, 'data.hdf5'), 'w')
    data_id = open(osp.join(data_dir, 'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i % (image.shape[0] / 100) == 0:
            bar.update(i / (image.shape[0] / 100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]

        # sample from a distribution
        if args.distribution == 'Uniform':
            grp['code'] = np.random.random(args.dimension) * 2 - 1  # normal distribution
        elif args.distribution == 'Gaussian':
            grp['code'] = np.random.randn(args.dimension)  # normal distribution
        elif args.distribution == 'PCA':
            grp['code'] = Y[i, :]/np.linalg.norm(Y[i, :], 2)

    bar.finish()
    f.close()
    data_id.close()
    return


def check_file(data_dir):
    if osp.exists(data_dir):
        if osp.isfile(osp.join(data_dir, 'data.hdf5')) and \
           osp.isfile(osp.join(data_dir, 'id.txt')):
            return True
    else:
        os.mkdir(data_dir)
    return False


def download_mnist(download_path):
    data_dir = osp.join(download_path, 'mnist')

    if check_file(data_dir):
        print('MNIST was downloaded.')
        return

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = osp.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(osp.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1)).astype(np.float)

    fd = open(osp.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1)).astype(np.float)

    prepare_h5py(train_image, test_image, data_dir)

    for k in keys:
        cmd = ['rm', '-f', osp.join(data_dir, k[:-3])]
        subprocess.call(cmd)


def download_svhn(download_path):
    data_dir = osp.join(download_path, 'svhn')

    import scipy.io as sio

    # svhn file loader
    def svhn_loader(url, path):
        cmd = ['curl', url, '-o', path]
        subprocess.call(cmd)
        m = sio.loadmat(path)
        return m['X'], m['y']

    if check_file(data_dir):
        print('SVHN was downloaded.')
        return

    data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    train_image, train_label = svhn_loader(data_url, osp.join(data_dir, 'train_32x32.mat'))

    data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    test_image, test_label = svhn_loader(data_url, osp.join(data_dir, 'test_32x32.mat'))

    prepare_h5py(np.transpose(train_image, (3, 0, 1, 2)),
                 np.transpose(test_image, (3, 0, 1, 2)), data_dir)

    cmd = ['rm', '-f', osp.join(data_dir, '*.mat')]
    subprocess.call(cmd)


def download_cifar10(download_path):
    data_dir = osp.join(download_path, 'cifar10')

    # cifar file loader
    def unpickle(file):
        from six.moves import cPickle as pickle

        with open(file, 'rb') as fo:
            try:
                dict = pickle.load(fo)
            except:
                fo.seek(0)
                dict = pickle.load(fo, encoding='latin1')
        return dict

    if check_file(data_dir):
        print('CIFAR was downloaded.')
        return

    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    k = 'cifar-10-python.tar.gz'
    target_path = osp.join(data_dir, k)
    print(target_path)
    cmd = ['curl', data_url, '-o', target_path]
    print('Downloading CIFAR10')
    subprocess.call(cmd)
    tarfile.open(target_path, 'r:gz').extractall(data_dir)

    num_cifar_train = 50000
    num_cifar_test = 10000

    target_path = osp.join(data_dir, 'cifar-10-batches-py')
    train_image = []
    for i in range(5):
        fd = osp.join(target_path, 'data_batch_'+str(i+1))
        dict = unpickle(fd)
        train_image.append(dict['data'])

    train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32*32*3])

    fd = osp.join(target_path, 'test_batch')
    dict = unpickle(fd)
    test_image = np.reshape(dict['data'], [num_cifar_test, 32*32*3])

    prepare_h5py(train_image, test_image, data_dir, [32, 32, 3])

    cmd = ['rm', '-f', osp.join(data_dir, 'cifar-10-python.tar.gz')]
    subprocess.call(cmd)
    cmd = ['rm', '-rf', osp.join(data_dir, 'cifar-10-batches-py')]
    subprocess.call(cmd)

if __name__ == '__main__':
    args = parser.parse_args()
    path = './datasets'
    if not osp.exists(path): os.mkdir(path)

    if 'MNIST' in args.datasets:
        download_mnist('./datasets')
    if 'SVHN' in args.datasets:
        download_svhn('./datasets')
    if 'CIFAR10' in args.datasets:
        download_cifar10('./datasets')
