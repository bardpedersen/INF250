# -*- coding: utf-8 -*-
__author__ = "Bård Pedersen"
__email__ = "Bard.tollef.pedersen@nmbu.no"

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def histogram(image):
    """Returns the image histogram with 256 bins."""
    # Setup
    shape = np.shape(image)
    histo_gram = np.zeros(256)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    for i in range(shape[0]):
        for j in range(shape[1]):
            histogram_index = round(image[i][j])
            histo_gram[histogram_index] += 1

    return histo_gram


def otsu(image):
    """Finds the optimal thresholdvalue of given image using Otsu's method."""
    hist = histogram(image)
    th = 0
    var_max = 0

    for i in range(256):
        background = hist[:i]
        foreground = hist[i:]
        weight_background = sum(background) / sum(hist)
        weight_foreground = sum(foreground) / sum(hist)

        if sum(background) == 0 or sum(foreground) == 0:
            continue

        mean_background = sum(background * range(i)) / sum(background)
        mean_foreground = sum(foreground * range(i, 256)) / sum(foreground)
        var_between = weight_background * weight_foreground * \
                      (mean_background - mean_foreground) * (mean_background - mean_foreground)

        if var_between > var_max:
            var_max = var_between
            th = i

    return th


def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.

    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal thrshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.

    Parameters:
    -----------
    image : np.ndarray
        Image to binarise. If this image is a colour image then the last
        dimension will be the colour value (as RGB values).
    th : numeric
        Threshold value. Uses Otsu's method if this variable is None.

    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
        Image where all pixel values are either 0 or 255.
    """
    # Setup
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    if th is None:
        th = otsu(image)

    binary = image > th
    binarised = (binary + binarised) * 255

    return binarised


if __name__ == "__main__":
    pick = io.imread('gingerbreads.jpg')
    new_pick = threshold(pick)
    plt.imshow(new_pick, cmap=plt.cm.gray)
    plt.savefig('new_pick.jpg')
    plt.show()