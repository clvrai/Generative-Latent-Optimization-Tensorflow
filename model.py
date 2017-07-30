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
        # if is_training is not None:
        fd[self.is_train] = is_training

        return fd

    def build(self, is_train=True):

        deconv_info = self.deconv_info
        d_dim = self.d_dim
        self.num_res_block = 3

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
        x, z = self.image, self.code
        # }}}

        # Generator {{{
        # =========
        x_recon = g(z)
        self.targets = x
        self.preds = x_recon
        # }}}

        # Build loss {{{
        # =========
        self.loss = tf.reduce_mean(tf.abs(x - x_recon))
        # }}}

        tf.summary.scalar("loss/loss", self.loss)
        tf.summary.image("img/reconstructed", x_recon, max_outputs=4)
        tf.summary.image("img/real", x, max_outputs=4)
        print('\033[93mSuccessfully loaded the model.\033[0m')
