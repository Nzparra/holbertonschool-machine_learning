#!/usr/bin/env python3
"""  trains a loaded neural network model using mini-batch gradient descent """

import tensorflow as tf
shuffle = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ x is a placeholder for the input data
        y is a placeholder for the labels
        accuracy is an op to calculate the accuracy of the model
        loss is an op to calculate the cost of the model
        train_op is an op to perform one pass of gradient descent on the model
    """
    with tf.Session() as ss:
        save = tf.train.import_meta_graph(load_path + '.meta')
        save = save.restore(ss, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        acc = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')
        iter_ = X_train.shape[0]/batch_size
        if iter_.isinteger():
            iter_ = int(iter_)
        else:
            iter_ = (int(iter_) + 1)
        train = {x: X_train, y: Y_train}
        valid = {x: X_valid, y: Y_valid}
        for i in range(epochs + 1):
            T_cost = ss.run(loss, feed_dict=train)
            T_acc = ss.run(acc, feed_dict=train)
            V_cost = ss.run(loss, feed_dic=valid)
            V_acc = ss.run(acc, feed_dict=valid)
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(T_cost))
            print('\tTraining Accuracy: {}'.format(T_acc))
            print('\tValidation Cost: {}'.format(V_cost))
            print('\tValidation Accuracy: {}'.format(V_acc))
            if i < epochs:
                X, Y = shuffle(X_train, Y_train)
                for j in range(iter_):
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]
                    temp = {x: X[start:end], y: Y[start:end]}
                    ss.run(train_op, feed_dict=temp)
                    if j != 0 and (j + 1) % 100 == 0:
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(ss.run(loss,
                                                           feed_dict=temp)))
                        print("\t\tAccuracy: {}".format(
                            ss.run(acc, feed_dict=temp)))
        return save.save(ss, save_path)
