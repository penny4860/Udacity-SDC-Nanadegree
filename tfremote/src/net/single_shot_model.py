# -*- coding: utf-8 -*-

"""Sample Models to train small image patches"""

# Todo : porting
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from .model import _Model


class SingleShotSvhnModel(_Model):
    def build(self):
        X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        Y = tf.placeholder(tf.float32, [None, 7, 7, 5])

        batch_norm_params = {'is_training': self._is_training,
                             'decay': 0.9,
                             'updates_collections': None}

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            # (224, 224)
            net = slim.conv2d(X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # (112, 112)
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            # (56, 56)
            net = slim.conv2d(X, 128, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            # (28, 28)
            net = slim.conv2d(net, 256, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            # (14, 14)
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # (N, 7, 7, 256) => (N, 7*7*256)
            net = slim.flatten(net, scope='flatten7')

            net = slim.fully_connected(net, 7*7*5, activation_fn=None,
                                       normalizer_fn=None, scope='fc0')
            Y_pred = tf.reshape(net, [-1, 7, 7, 5])
        return X, Y, Y_pred

    def cost(self):
        # conditional graph :
        # 1) self.Y[:, 0] 이 0이면 0,
        # 2) self.Y[:, 0] 이 1이면 수식으로 계산,
        # Test : (Xs, Ys) 에서 Ys가 모두 Background 일 때 cost_for_regression == 0 인지 확인하자.
        n_grid_rows = self.Y.shape[1]
        n_grid_cols = self.Y.shape[2]
        print (n_grid_rows, n_grid_cols)
        cost_for_regression = tf.constant(0, tf.float32)
        cost_for_detection = tf.constant(0, tf.float32)
        for i in range(self.batch_size):
            for r in range(n_grid_rows):
                for c in range(n_grid_cols):
                    # Y ground truth 가 object_exist 인 grid 에서만 cost_for_regression 을 더한다.
                    cost_for_regression += tf.cond(tf.equal(self.Y[i, r, c, 0], tf.constant(1, tf.float32)),
                                                   lambda: tf.reduce_sum(tf.square(tf.subtract(self.Y_pred[i, r, c, 1:], self.Y[i, r, c, 1:]))),
                                                   lambda: tf.constant(0, tf.float32))

                    cost_for_detection += tf.cond(tf.equal(self.Y[i, r, c, 0], tf.constant(1, tf.float32)),
                                                   lambda: tf.square(tf.subtract(self.Y_pred[i, r, c, 0], self.Y[i, r, c, 0])),
                                                   lambda: tf.square(tf.subtract(self.Y_pred[i, r, c, 0], self.Y[i, r, c, 0])))


        cost = (cost_for_detection + 10*cost_for_regression) / self.batch_size
        return cost
"""