#!/usr/bin/env python3
""" builds an identity block as described"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Returns: the activated output of the identity block
    """
    X = K.Input(shape=(224, 224, 3))
    lay_init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same', strides=(2, 2),
                            kernel_initializer=lay_init)(X)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    X1 = K.layers.Activation('relu')(norm1)
    mxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                    padding="same")(X1)
    pro_block0 = projection_block(mxpool1, [64, 64, 256], 1)
    ide_block1 = identity_block(pro_block0, [64, 64, 256])
    ide_block2 = identity_block(ide_block1, [64, 64, 256])
    pro_block1 = projection_block(ide_block2, [128, 128, 512])
    ide_block3 = identity_block(pro_block1, [128, 128, 512])
    ide_block4 = identity_block(ide_block3, [128, 128, 512])
    ide_block5 = identity_block(ide_block4, [128, 128, 512])
    pro_block2 = projection_block(ide_block5, [256, 256, 1024])
    ide_block6 = identity_block(pro_block2, [256, 256, 1024])
    ide_block7 = identity_block(ide_block6, [256, 256, 1024])
    ide_block8 = identity_block(ide_block7, [256, 256, 1024])
    ide_block9 = identity_block(ide_block8, [256, 256, 1024])
    ide_block10 = identity_block(ide_block9, [256, 256, 1024])
    pro_block3 = projection_block(ide_block10, [512, 512, 2048])
    ide_block11 = identity_block(pro_block3, [512, 512, 2048])
    ide_block12 = identity_block(ide_block11, [512, 512, 2048])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1))(ide_block12)
    FC = K.layers.Dense(units=1000, activation='softmax',
                        kernel_initializer=lay_init)(avgpool)
    return K.models.Model(inputs=X, outputs=FC)
