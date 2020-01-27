import tensorflow as tf
import os
import numpy as np
from skimage.io import imread


def get_image(path):
    return imread(path) / 255.0


class NoisyImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, corrupt_func, batch_size=32, shuffle=True):
        self.path = path
        self.files = os.listdir(path)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.files))
        self.corrupt = corrupt_func

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indicies = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        batch_files = [self.files[i] for i in indicies]
        X = np.asarray([get_image(os.path.join(self.path, file)) for file in batch_files])
        X_noisy = np.asarray([self.corrupt(x) for x in X])

        return X_noisy, X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)