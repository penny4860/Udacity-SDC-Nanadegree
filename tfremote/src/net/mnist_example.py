# -*- coding: utf-8 -*-

import tensorflow as tf
from src.net.base import _Model

class MnistCnn(_Model):

    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 28, 28, 1])

    def _create_inference_op(self):
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        
        W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
        model = tf.matmul(L3, W4)
        return model

    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))

import tensorflow.contrib.slim as slim
class MnistBn(_Model):
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 28, 28, 1])

    def _create_inference_op(self):
        batch_norm_params = {'is_training': self.is_training,
                             'decay': 0.9,
                             'updates_collections': None}

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            net = slim.conv2d(self.X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.fully_connected(net, 10, activation_fn=None,
                                          normalizer_fn=None, scope='fco')
        
        return net

    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    from src.net.base import train, evaluate

    mnist = input_data.read_data_sets("./mnist/data/", one_hot=False)
    
    train_images = mnist.train.images.reshape(-1, 28, 28, 1)
    valid_images = mnist.validation.images.reshape(-1, 28, 28, 1)
    test_images = mnist.test.images.reshape(-1, 28, 28, 1)
    
    model = MnistCnn()
    train(model, train_images, mnist.train.labels, valid_images, mnist.validation.labels, ckpt='ckpts/cnn')
    evaluate(model, test_images, mnist.test.labels, ckpt='ckpts')
    