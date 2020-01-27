import tensorflow as tf
import os
from Generator.DataGenerator import NoisyImageGenerator
from Models.GetModel import get_untrained_model
from Utils.Utilities import get_model_save_path, save_example
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def get_callbacks(save_model_path, reduce_lr_patience, early_stopping_patience):
    """Returns callbacks used for training network"""
    return [tf.keras.callbacks.ReduceLROnPlateau(patience=reduce_lr_patience),
            tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience),
            tf.keras.callbacks.ModelCheckpoint(save_model_path, save_best_only=True, save_weights_only=True)]


if __name__ == '__main__':
    MODELS_ROOT = './store/models/'
    IMAGES_ROOT = './store/images/'
    train_images_path = os.path.join(IMAGES_ROOT, 'train')
    val_images_path = os.path.join(IMAGES_ROOT, 'val')

    model_path = get_model_save_path(MODELS_ROOT)
    early_stopping_patience = 10
    reduce_lr_patience = early_stopping_patience // 3
    input_shape = (128, 128, 3)
    init_lr = 0.1

    callbacks = get_callbacks(model_path, reduce_lr_patience, early_stopping_patience)
    model = get_untrained_model(input_shape)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=init_lr))
    model.summary()

    train_data_generator = NoisyImageGenerator(path=train_images_path)
    val_data_generator = NoisyImageGenerator(path=val_images_path)

    history = model.fit_generator(generator=train_data_generator,
                                  epochs=1000,  # This many epochs won't run because of EarlyStopping callback
                                  validation_data=val_data_generator,
                                  callbacks=callbacks,
                                  verbose=1)

