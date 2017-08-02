import h5py
import numpy as np
import imageio
import glob
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
args = parser.parse_args()

if not args.train_dir or not args.output_file:
    raise ValueError("Please specify train_dir and output_file")

II = []

file_list = sorted(glob.glob(os.path.join(args.train_dir, "*.hdf5")), key=os.path.getmtime)
f = h5py.File(file_list[0], 'r')
img = f['image']
h, w, c = np.asarray(img.shape[1:])
n = int(np.sqrt(img.shape[0]))

for file in file_list:
    f = h5py.File(file, 'r')
    I = np.zeros((n*h, n*w, c))
    for i in range(n):
        for j in range(n):
            I[h * i:h * (i + 1), w * j:w * (j + 1), :] = \
                f['image'][i * n + j, :, :, :]
    II.append(I)

II = np.stack(II)
imageio.mimsave('{}.gif'.format(args.output_file), II, fps=5)
imageio.imwrite('{}.png'.format(args.output_file), I)
