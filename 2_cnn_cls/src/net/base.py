# -*- coding: utf-8 -*-

import tensorflow as tf
from abc import ABCMeta, abstractmethod
from sklearn.utils import shuffle


class _Model(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        # Placeholders
        self.X = self._create_input_placeholder()
        self.Y = self._create_output_placeholder()
        self.is_training = self._create_is_train_placeholder()

        # basic operations
        self.inference_op = self._create_inference_op()
        self.loss_op = self._create_loss_op()
        self.accuracy_op = self._create_accuracy_op()

        # summary operations        
        self.train_summary_op = self._create_train_summary_op()
        
    @abstractmethod
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_images')

    @abstractmethod
    def _create_inference_op(self):
        pass

    @abstractmethod
    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))

    def _create_output_placeholder(self):
        return tf.placeholder(tf.int64, [None], name='output_labels')

    def _create_accuracy_op(self):
        is_correct = tf.equal(tf.argmax(self.inference_op, 1), self.Y)
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def _create_is_train_placeholder(self):
        is_training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                  shape=(),
                                                  name='is_training')
        return is_training

    def _create_train_summary_op(self):
        with tf.name_scope('train_summary'):
            summary_loss = tf.summary.scalar('loss', self.loss_op)
            # summary_acc = tf.summary.scalar('accuracy', self.accuracy_op)
            summary_op = tf.summary.merge([summary_loss], name='train_summary')
            return summary_op


def train(model, X_train, y_train, X_val, y_val, batch_size=100, n_epoches=5, ckpt=None):

    def _run_single_epoch(X_train, y_train, batch_size):
        total_cost = 0
        for offset, end in get_batch_index(len(X_train), batch_size):
            _, cost_val, summary_result = sess.run([optimizer, model.loss_op, model.train_summary_op],
                                   feed_dict={model.X: X_train[offset:end],
                                              model.Y: y_train[offset:end],
                                              model.is_training: True})
            total_cost += cost_val
            writer.add_summary(summary_result, sess.run(global_step))
        return total_cost
   
    def _save(sess, ckpt, global_step):
        import os
        directory = os.path.dirname(ckpt)
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        saver = tf.train.Saver()
        saver.save(sess, ckpt, global_step=global_step)
        # saver.save(sess, 'models/cnn')
        # saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)

    def _print_cost(epoch, cost, global_step):
        print('Epoch: {:3d}, Training Step: {:5d}, Avg. cost ={:.3f}'.format(epoch + 1, global_step, cost))
        
    def _write_value_to_writer(value, writer, tag):
        value_obj = tf.Summary.Value(tag=tag, simple_value=value)
        summary_result = tf.Summary(value=[value_obj])
        writer.add_summary(summary_result, sess.run(global_step))

    
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss_op, global_step=global_step)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = get_n_batches(len(X_train), batch_size)
        
        # tensorboard --logdir="./graphs" --port 6006
        writer = tf.summary.FileWriter('./graphs/train', sess.graph)
        writer_val = tf.summary.FileWriter('./graphs/valid', sess.graph)
        
        for epoch in range(n_epoches):
            # 1. shuffle
            X_train, y_train = shuffle(X_train, y_train)
            
            # 2. run training
            cost = _run_single_epoch(X_train, y_train, batch_size)
            _print_cost(epoch, cost / total_batch, sess.run(global_step))
            
            # 3. evaluation accuracy
            train_accuracy = evaluate(model, X_train, y_train, sess, batch_size=batch_size)
            valid_accuracy = evaluate(model, X_val, y_val, sess, batch_size=batch_size)

            # 4. logging
            _write_value_to_writer(train_accuracy, writer, "accuracy")
            _write_value_to_writer(valid_accuracy, writer_val, "accuracy")

            if ckpt:
                _save(sess, ckpt, global_step)
        
        print('Training done')
        writer.close()

def evaluate(model, images, labels, session=None, ckpt=None, batch_size=100):
    """
    ckpt : str
        ckpt directory or ckpt file
    """
    def _evaluate(sess):
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt))

        accuracy_value = 0
        for offset, end in get_batch_index(len(images), batch_size):
            accuracy_value += sess.run(model.accuracy_op,
                                       feed_dict={model.X: images[offset:end],
                                                  model.Y: labels[offset:end],
                                                  model.is_training: False})
            
        accuracy_value = accuracy_value / get_n_batches(len(images), batch_size)
        return accuracy_value

    if session:
        accuracy = _evaluate(session)
    else:
        sess = tf.Session()
        accuracy = _evaluate(sess)
        sess.close()
        
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy


def get_batch_index(num_examples, batch_size=100):
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        if end > num_examples:
            break
        yield (offset, end)

def get_n_batches(num_examples, batch_size=100):
    return int(num_examples / batch_size)

