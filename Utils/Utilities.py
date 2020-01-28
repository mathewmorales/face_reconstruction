import matplotlib.pyplot as plt
import numpy as np
import os
import time


def get_model_save_path(root):
    """Returns a new path for the model to be saved at."""
    return os.path.join(root, time.strftime('model_%Y_%m_%d-%H_%M_%S.hdf5'))


def save_example(images, path, name, trained_model, corrupt_func):
    """Saves an image showing how the model performed on several of the test images

    Keyword Arguments
    images -- An array of images
    path -- The path to folder to save image to
    name -- The name of the file to save
    trained_model -- The trained model to test.
    corrupt_func -- The function to use to corrupt the image
    """
    num_examples = 3
    f, ax = plt.subplots(num_examples, 3)
    for i in range(num_examples):
        noisy_img = corrupt_func(images[i])
        pred_img = trained_model.predict(np.expand_dims(noisy_img, axis=0))
        ax[i, 0].imshow(images[i])
        ax[i, 1].imshow(noisy_img)
        ax[i, 2].imshow(pred_img[0])
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
    plt.savefig(os.path.join(path, name))