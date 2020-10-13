#!/usr/bin/env python3
""" returns layer """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
                Returns: placeholders named x and y, respectively
    """
    for layer_size, activation in zip(layer_sizes, activations):
        x = create_layer(x, layer_size, activation)
    return x
