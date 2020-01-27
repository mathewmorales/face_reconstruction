import numpy as np


def pepper(img):
    """Corrupts the input image by randomly setting up to 60% of the image to 0.

    Keyword Arguments
    img -- The image to corrupt. Should be (Width, Height, Channels) or (Height, Width, Channels)

    returns -- The noisy image.
    """
    amount = np.random.random() * 0.6
    noisy_img = np.copy(img)

    num_noise = np.ceil(amount * img.shape[0] * img.shape[1])
    coords = [np.random.randint(0, i - 1, int(num_noise)) for i in img.shape[:-1]]

    noisy_img[coords] = 0
    return noisy_img