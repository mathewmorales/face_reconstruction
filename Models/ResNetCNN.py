import tensorflow as tf


class ResNetConv2D(tf.keras.Model):
    def __init__(self, depth, filters, kernels, activation, **kwargs):
        super(ResNetConv2D, self).__init__(**kwargs)
        self.my_layers = [[tf.keras.layers.Conv2D(filters=filters,
                                                   kernel_size=kernels,
                                                   strides=1,
                                                   padding='same'),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.Activation(activation)]
                           for _ in range(depth)]

    def call(self, inputs):
        z = inputs
        for block in self.my_layers:
            z = z + block[0](z)
            z = block[1](z)
            z = block[2](z)
        return z + inputs