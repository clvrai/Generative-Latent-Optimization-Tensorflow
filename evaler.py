from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np

from util import log

from model import Model
from input_ops import create_input_ops, check_data_id

import tensorflow as tf
import time
import h5py
import imageio
import scipy.misc as sm

class EvalManager(object):

    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):

        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def compute_loss(self, pred, gt):
        return np.sum(np.abs(pred - gt))/np.prod(pred.shape)

    def report(self):
        log.info("Computing scores...")
        total_loss = []

        for id, pred, gt in zip(self._ids, self._predictions, self._groundtruths):
            total_loss.append(self.compute_loss(pred, gt))
        avg_loss = np.average(total_loss)
        log.infov("Average loss : %.4f", avg_loss)


class Evaler(object):
    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) + 1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager()
        x = self.generator()
        img = self.image_grid(x)
        imageio.imwrite('{}.png'.format(self.config.prefix), img)

        try:
            for s in xrange(max_steps):
                step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, loss, step_time)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        evaler.report()
        log.infov("Evaluation complete.")

    def generator(self):
        z = np.random.randn(self.batch_size, self.config.data_info[3])
        row_sums = np.sqrt(np.sum(z ** 2, axis=0))
        z = z / row_sums[np.newaxis, :]
        x_hat = self.session.run(self.model.x_recon, feed_dict={self.model.z: z})
        return x_hat

    def image_grid(self, x, shape=(2048, 2048)):
        n = int(np.sqrt(x.shape[0]))
        h, w, c = self.config.data_info[0], self.config.data_info[1], self.config.data_info[2]
        I = np.zeros((n*h, n*w, c))
        for i in range(n):
            for j in range(n):
                I[h * i:h * (i+1), w * j:w * (j+1), :] = x[i * n + j]
        if c == 1:
            I = I[:, :, 0]
        return sm.imresize(I, shape)

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, loss, all_targets, all_preds, _] = self.session.run(
            [self.global_step, self.model.loss, self.model.x, self.model.x_recon, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, loss, (_end_time - _start_time), batch_chunk, all_preds, all_targets

    def log_step_message(self, step, loss, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss (test): {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--data_id', nargs='*', default=None)
    config = parser.parse_args()

    if config.dataset == 'MNIST':
        import datasets.mnist as dataset
    elif config.dataset == 'SVHN':
        import datasets.svhn as dataset
    elif config.dataset == 'CIFAR10':
        import datasets.cifar10 as dataset
    else:
        raise ValueError(config.dataset)

    config.conv_info = dataset.get_conv_info()
    config.deconv_info = dataset.get_deconv_info()
    dataset_train, dataset_test = dataset.create_default_splits()

    m, l = dataset_train.get_data(dataset_train.ids[0])
    config.data_info = np.concatenate([np.asarray(m.shape), np.asarray(l.shape)])

    evaler = Evaler(config, dataset_test)

    log.warning("dataset: %s", config.dataset)
    evaler.eval_run()

if __name__ == '__main__':
    main()
