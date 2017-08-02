from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import bilinear_deconv2d


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.d_dim = self.config.data_info[3]
        self.deconv_info = self.config.deconv_info
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )

        self.code = tf.placeholder(
            name='code', dtype=tf.float32,
            shape=[self.batch_size, self.d_dim],
        )

        self.is_train = tf.placeholder(
            name='is_train', dtype=tf.bool,
            shape=[],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=True):
        fd = {
            self.image: batch_chunk['image'],  # [B, h, w, c]
            self.code: batch_chunk['code'],  # [B, d]
        }
        fd[self.is_train] = is_training

        return fd

    def build(self, is_train=True):

        deconv_info = self.deconv_info
        d_dim = self.d_dim
        self.num_res_block = 3

        def local_moment_loss(pred, gt):
            with tf.name_scope('local_moment_loss'):

                ksz, kst = 4, 2
                local_patch = tf.ones((ksz, ksz, 1, 1))
                c = pred.get_shape()[-1]

                # Normalize by kernel size
                pr_mean = tf.concat([tf.nn.conv2d(x, local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(pred, c, axis=3)], axis=3)
                pr_var = tf.concat([tf.nn.conv2d(tf.square(x), local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(pred, c, axis=3)], axis=3)
                pr_var = (pr_var - tf.square(pr_mean)/(ksz**2)) / (ksz ** 2)
                pr_mean = pr_mean / (ksz ** 2)

                gt_mean = tf.concat([tf.nn.conv2d(x, local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(gt, c, axis=3)], axis=3)
                gt_var = tf.concat([tf.nn.conv2d(tf.square(x), local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(gt, c, axis=3)], axis=3)
                gt_var = (gt_var - tf.square(gt_mean)/(ksz**2)) / (ksz ** 2)
                gt_mean = gt_mean / (ksz ** 2)

                # scaling by local patch size
                local_mean_loss = tf.reduce_mean(tf.abs(pr_mean - gt_mean))
                local_var_loss = tf.reduce_mean(tf.abs(pr_var - gt_var))
            return local_mean_loss + local_var_loss

        # z -> x
        def g(z, scope='g'):
            with tf.variable_scope(scope) as scope:
                print('\033[93m'+scope.name+'\033[0m')
                _ = tf.reshape(z, [self.batch_size, 1, 1, d_dim])
                _ = bilinear_deconv2d(_, deconv_info[0], is_train, name='deconv1')
                print(scope.name, _)
                _ = bilinear_deconv2d(_, deconv_info[1], is_train, name='deconv2')
                print(scope.name, _)
                _ = bilinear_deconv2d(_, deconv_info[2], is_train, name='deconv3')
                print(scope.name, _)
                _ = bilinear_deconv2d(_, deconv_info[3], is_train, name='deconv4')
                print(scope.name, _)
                _ = bilinear_deconv2d(_, deconv_info[4], is_train, name='deconv5',
                                      batch_norm=False, activation_fn=tf.tanh)
                print(scope.name, _)
                _ = tf.image.resize_images(
                    _, [int(self.image.get_shape()[1]), int(self.image.get_shape()[2])]
                )
            return _

        # Input {{{
        # =========
        self.x, self.z = self.image, self.code
        # }}}

        # Generator {{{
        # =========
        self.x_recon = g(self.z)
        # }}}

        # Build loss {{{
        # =========
        # self.loss = tf.reduce_mean(tf.abs(self.x - self.x_recon))
        self.loss = local_moment_loss(self.x, self.x_recon)
        self.z_grad = tf.gradients(self.loss, self.z)
        # }}}

        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.image("img/reconstructed", self.x_recon, max_outputs=4)
        tf.summary.image("img/real", self.x, max_outputs=4)
        print('\033[93mSuccessfully loaded the model.\033[0m')
