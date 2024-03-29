#!/usr/bin/env python3
""" Transfer Knowledge """

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
         Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    entrada = K.Input(shape=(32, 32, 3))
    resize = K.layers.Lambda(lambda image:
                             tf.image.resize(image, (150, 150)))(entrada)
    dense169 = K.applications.DenseNet169(include_top=False,
                                          weights="imagenet",
                                          input_tensor=resize)
    out = dense169(resize)
    pre_model = K.models.Model(inputs=entrada, outputs=out)
    X_p = K.applications.densenet.preprocess_input(X)
    features = pre_model.predict(X_p)
    Y_p = K.utils.to_categorical(y=Y, num_classes=10)
    return (features, Y_p)


if __name__ == "__main__":
    Train, Test = K.datasets.cifar10.load_data()
    (x_train, y_train) = Train
    (x_test, y_test) = Test
    xp_train, yp_train = preprocess_data(x_train, y_train)
    xp_test, yp_test = preprocess_data(x_test, y_test)
    lay_init = K.initializers.he_normal()
    new_input = K.Input(shape=xp_train.shape[1:])
    vector = K.layers.Flatten()(new_input)
    drop1 = K.layers.Dropout(0.3)(vector)
    norm_lay1 = K.layers.BatchNormalization()(drop1)
    FC1 = K.layers.Dense(units=510, activation='relu',
                         kernel_initializer=lay_init)(norm_lay1)
    norm_lay2 = K.layers.BatchNormalization()(FC1)
    out = K.layers.Dense(units=10, activation='softmax',
                         kernel_initializer=lay_init)(norm_lay2)
    model = K.models.Model(inputs=new_input, outputs=out)
    learn_dec = K.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                              factor=0.1, patience=2)
    early = K.callbacks.EarlyStopping(patience=5)
    save = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                       save_best_only=True,
                                       monitor='val_acc',
                                       mode='max')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x=xp_train, y=yp_train, batch_size=32, epochs=15,
              verbose=1, validation_data=(xp_test, yp_test),
              callbacks=[save, early, learn_dec])
