#!usr/bin/env python3
""" Restore save trained DNN """

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Returns: the networks prediction, accuracy, and loss,
             respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))
        x, *_ = tf.get_collection('x')
        y, *_ = tf.get_collection('y')
        y_pred, *_ = tf.get_collection('y_pred')
        loss, *_ = tf.get_collection('loss')
        accuracy, *_ = tf.get_collection('accuracy')
        net_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        net_loss = sess.run(loss, feed_dict={x: X, y: Y})
        net_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
    return net_pred, net_accuracy, net_loss
