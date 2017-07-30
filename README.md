# Generative Latent Optimization in Tensorflow

As part of the implementation series of [Joseph Lim's group at USC](http://csail.mit.edu/~lim), our motivation is to accelerate (or sometimes delay) research in the AI community by promoting open-source projects. To this end, we implement state-of-the-art research papers, and publicly share them with concise reports. Please visit our [group github site](https://github.com/gitlimlab) for other projects.

This project is implemented by [Shao-Hua Sun](http://shaohua0116.github.io) and the codes have been reviewed by <!--- --> before being published.

## Descriptions
This project is a [Tensorflow](https://www.tensorflow.org/) implementation of **Generative Latent Optimization (GLO)** proposed in the paper [Optimizing the Latent Space of Generative Networks](https://arxiv.org/abs/1707.05776). The intuition is <!--- -->.

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 1.2.0](https://github.com/tensorflow/tensorflow/tree/r1.2)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)
- [PIL](http://pillow.readthedocs.io/en/3.1.x/installation.html)
- [matplotlib](https://matplotlib.org/)
- [h5py](http://docs.h5py.org/en/latest/)
- [progressbar](http://progressbar-2.readthedocs.io/en/latest/index.html)
- [colorlog](https://github.com/borntyping/python-colorlog)

## Usage

Download datasets with:
```bash
$ python download.py --dataset MNIST SVHN CIFAR10
```
Train models with downloaded datasets:
```bash
$ python trainer.py --dataset MNIST
$ python trainer.py --dataset SVHN
$ python trainer.py --dataset CIFAR10
```
Test models with saved checkpoints:
```bash
$ python evaler.py --dataset MNIST --checkpoint ckpt_dir
$ python evaler.py --dataset SVHN --checkpoint ckpt_dir
$ python evaler.py --dataset CIFAR10 --checkpoint ckpt_dir
```
The *ckpt_dir* should be like: ```train_dir/default-MNIST_lr_0.0001-20170101-194957/model-1001```

Train and test your own datasets:

* Create a directory
```bash
$ mkdir datasets/YOUR_DATASET
```

* Store your data as an h5py file datasets/YOUR_DATASET/data.hy and each data point contains
    * 'image': has shape [h, w, c], where c is the number of channels (grayscale images: 1, color images: 3)
    * 'code': represented as an vector sampled from an uniform distribution or a normal distribution
* Maintain a list datasets/YOUR_DATASET/id.txt listing ids of all data points
* Modify trainer.py including args, data_info, etc.
* Finally, train and test models:
```bash
$ python trainer.py --dataset YOUR_DATASET
$ python evaler.py --dataset YOUR_DATASET
```
## Results

### MNIST

* Generated samples (100th epochs)

<img src="figure/result/mnist/samples.png" height="250"/>

* First 40 epochs

<img src="figure/result/mnist/training.gif" height="250"/>

### SVHN

* Generated samples (100th epochs)

<img src="figure/result/svhn/samples.png" height="250"/>

* First 160 epochs

<img src="figure/result/svhn/training.gif" height="250"/>


### CIFAR-10

* Generated samples (1000th epochs)

<img src="figure/result/cifar10/samples.png" height="250"/>

* First 200 epochs

<img src="figure/result/cifar10/training.gif" height="250"/>

## Training details

### MNIST

* The loss

<img src="figure/result/mnist/loss.png" height="200"/>

### SVHN

* The loss

<img src="figure/result/svhn/loss.png" height="200"/>

### CIFAR-10

* The loss

<img src="figure/result/cifar10/loss.png" height="200"/>

## Related works
* My implementation of [Semi-supervised learning GAN](https://github.com/gitlimlab/SSGAN-Tensorflow)

## Author

Shao-Hua Sun / [@shaohua0116](https://github.com/shaohua0116/) @ [Joseph Lim's research lab](https://github.com/gitlimlab) @ USC
