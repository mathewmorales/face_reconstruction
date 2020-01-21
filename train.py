import tensorflow as tf
import numpy as np
import os
from skimage.io import imread
import time
import matplotlib.pyplot as plt
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

MODELS_ROOT = './store/models/'
TENSORBOARD_ROOT = './store/tensorboard/'
IMAGES_ROOT = '../images/'


def make_noisy(img):
    amount = np.random.random() * 0.05
    noisy_img = np.copy(img)

    num_noise = np.ceil(amount * img.shape[0] * img.shape[1])
    coords = [np.random.randint(0, i - 1, int(num_noise)) for i in img.shape[:-1]]

    noisy_img[coords] = 0
    return noisy_img


def get_image(path):
    return imread(path) / 255.0


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size=32, shuffle=True):
        self.path = path
        self.files = os.listdir(path)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indicies = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        batch_files = [self.files[i] for i in indicies]
        X = [get_image(os.path.join(self.path, file)) for file in batch_files]
        X_noisy = [make_noisy(x) for x in X]

        return np.asarray(X_noisy), np.asarray(X)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_callbacks(model_path, patience, tensorboard_path):
    return [tf.keras.callbacks.ReduceLROnPlateau(patience=patience//3),
            tf.keras.callbacks.EarlyStopping(patience=patience),
            tf.keras.callbacks.ModelCheckpoint(model_path),
            tf.keras.callbacks.TensorBoard(tensorboard_path)]


def get_model(in_shape):
    mobileNet = tf.keras.applications.MobileNetV2(input_shape=in_shape, include_top=False)
    return tf.keras.Sequential([
        mobileNet,
        tf.keras.layers.Conv2DTranspose(filters=320, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')
    ])


def get_tensorboard_path(root):
    return os.path.join(root, time.strftime('run_%Y_%m_%d-%H_%M_%S'))


def get_model_path(root):
    return os.path.join(root, time.strftime('model_%Y_%m_%d-%H_%M_%S.hdf5'))

def save_image(image, name, model):
    f, ax = plt.subplots(1, 3)
    noisy_img = make_noisy(image)
    pred_img = model.predict(np.expand_dims(noisy_img, axis=0))
    ax[0].imshow(image)
    ax[1].imshow(noisy_img)
    ax[2].imshow(pred_img[0])
    plt.savefig(name)


if __name__ == '__main__':
    model_path = get_model_path(MODELS_ROOT)
    tensorboard_path = get_tensorboard_path(TENSORBOARD_ROOT)
    patience = 10
    input_shape = (128, 128, 3)

    callbacks = get_callbacks(model_path=model_path, patience=patience, tensorboard_path=tensorboard_path)
    model = get_model(input_shape)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.1))
    model.summary()

    train_data_generator = DataGenerator(path=os.path.join(IMAGES_ROOT, 'train'))
    val_data_generator = DataGenerator(path=os.path.join(IMAGES_ROOT, 'val'))

    history = model.fit_generator(generator=train_data_generator,
                                  epochs=1000,
                                  validation_data=val_data_generator,
                                  callbacks=callbacks,
                                  verbose=1)
    with open('./history.pickle', 'wb') as f:
        pickle.dump(history, f)
