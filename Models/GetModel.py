import tensorflow as tf
import numpy as np
from Models.ResNetCNN import ResNetConv2D


def load_model(in_shape, path_to_model):
    """Gets the trained model from the file which stores the models weights.

    Keyword Arugments
    in_shape -- The input shape of the model
    path_to_model -- The path to the .hdf5 file which contains the model weights
    """
    model = get_untrained_model(in_shape)
    model.predict(np.expand_dims(np.random.rand(in_shape[0], in_shape[1], in_shape[2]), axis=0))
    model.load_weights(path_to_model)
    return model


def get_untrained_model(in_shape):
    """Returns the uninitialized model object for training"""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', input_shape=in_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        ResNetConv2D(depth=3, filters=32, kernels=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        ResNetConv2D(depth=5, filters=64, kernels=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu')
        ])