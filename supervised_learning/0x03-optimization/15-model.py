#!/usr/bin/env python3
"""
Complete Model Optimization
"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Returns: the shuffled X and Y matrices
    """
    vector = np.random.permutation(np.arange(X.shape[0]))
    X_shu = X[vector]
    Y_shu = Y[vector]
    return X_shu, Y_shu


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Returns: the Adam optimization operation
    """
    a = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                               beta2=beta2, epsilon=epsilon)
    optimize = a.minimize(loss)
    return optimize


def create_layer(prev, n, activation):
    """
    We have to use this function only in the last layer
    because we dont have to normalize the output
    Returns: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    A = tf.layers.Dense(units=n, name='layer', activation=activation,
                        kernel_initializer=init)
    Y_pred = A(prev)
    return (Y_pred)


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    if activation is None:
        A = create_layer(prev, n, activation)
        return A
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=w_init)
    Z = layers(prev)
    gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n]),
                       name='beta', trainable=True)
    epsilon = tf.constant(1e-8)
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)
    A = activation(Z_norm)
    return A


def forward_prop(x, layers, activations):
    """
    forward propagation
    """
    A = create_batch_norm_layer(x, layers[0], activations[0])
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A


def calculate_accuracy(y, y_pred):
    """
    accuracy of the prediction
    """
    index_y = tf.math.argmax(y, axis=1)
    index_pred = tf.math.argmax(y_pred, axis=1)
    comp = tf.math.equal(index_y, index_pred)
    cast = tf.cast(comp, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)
    return accuracy


def calculate_loss(y, y_pred):
    """
    loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    Data_train is a tuple containing the training inputs and
               training labels, respectively
    Returns: the path where the model was saved
    """
    mini_iter = Data_train[0].shape[0] / batch_size
    if (mini_iter).is_integer() is True:
        mini_iter = int(mini_iter)
    else:
        mini_iter = int(mini_iter) + 1
    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as ses:
        ses.run(init)
        train_feed = {x: Data_train[0], y: Data_train[1]}
        valid_feed = {x: Data_valid[0], y: Data_valid[1]}
        for i in range(epochs + 1):
            T_cost = ses.run(loss, train_feed)
            T_acc = ses.run(accuracy, train_feed)
            V_cost = ses.run(loss, valid_feed)
            V_acc = ses.run(accuracy, valid_feed)
            print("After {} epochs:".format(i))
            print('\tTraining Cost: {}'.format(T_cost))
            print('\tTraining Accuracy: {}'.format(T_acc))
            print('\tValidation Cost: {}'.format(V_cost))
            print('\tValidation Accuracy: {}'.format(V_acc))
            if i < epochs:
                X_shu, Y_shu = shuffle_data(Data_train[0], Data_train[1])
                ses.run(global_step.assign(i))
                a = ses.run(alpha)
                for j in range(mini_iter):
                    ini = j * batch_size
                    fin = (j + 1) * batch_size
                    if fin > Data_train[0].shape[0]:
                        fin = Data_train[0].shape[0]
                    mini_feed = {x: X_shu[ini:fin], y: Y_shu[ini:fin]}
                    ses.run(train_op, feed_dict=mini_feed)
                    if j != 0 and (j + 1) % 100 == 0:
                        Min_cost = ses.run(loss, mini_feed)
                        Min_acc = ses.run(accuracy, mini_feed)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(Min_cost))
                        print("\t\tAccuracy: {}".format(Min_acc))
        save_path = saver.save(ses, save_path)
    return save_path
