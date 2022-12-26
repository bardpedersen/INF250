# -*- coding: utf-8 -*-
import skimage
import matplotlib.pyplot as plt


def edge_operator(img, operator):
    """Returns the result from one of the edge operators, prewitt, sobel,
    canny or laplace

    Parameters:
    -----------
    image : np.ndarray
        Image to detect blobs in. If this image is a colour image then
        the last dimension will be the colour value (as RGB values).
    operator : numeric
    1 = sobel filter
    2 = prewitt filter
    3 = canny filter
    4 = laplace filter

    Returns:
    --------
    filtered : np.ndarray(np.uint)
    result image from the edge operator
    """
    if operator == 1:
        filtered = skimage.filters.sobel(img)
    elif operator == 2:
        filtered = skimage.filters.prewitt(img)
    elif operator == 3:
        filtered = skimage.feature.canny(img, sigma=1)
    elif operator == 4:
        filtered = skimage.filters.laplace(img, ksize=3, mask=None)
    else:
        filtered = None

    return filtered


def sharpen(imag, sharpmask):
    """Performs an image sharpening using Laplace filter or unsharpened mask (USM)
    1 = Laplace
    2 = USM

    Returns: sharpened image
    """
    amount = 2
    if sharpmask == 1:
        gauss = skimage.filters.gaussian(imag)
        sharpened = imag + amount * (imag-gauss)
    elif sharpmask == 2:
        laplac = skimage.filters.laplace(imag)
        sharpened = imag + amount * laplac
    else:
        sharpened = None

    return sharpened


if __name__ == '__main__':
    image = skimage.io.imread('AthenIR.tiff')
    plt.imshow(image)
    plt.title("Original image")
    plt.show()

    edge1 = edge_operator(image, 1)
    plt.imshow(edge1)
    plt.title("sobel image")
    plt.show()

    edge2 = edge_operator(image, 2)
    plt.imshow(edge2)
    plt.title("prewitt image")
    plt.show()

    edge3 = edge_operator(image, 3)
    plt.imshow(edge3)
    plt.title("canny image")
    plt.show()

    edge4 = edge_operator(image, 4)
    plt.imshow(edge4)
    plt.title("laplace image")
    plt.show()


    sharp1 = sharpen(image, 1)
    plt.imshow(sharp1)
    plt.title("unsharpening image")
    plt.show()

    sharp2 = sharpen(image, 2)
    plt.imshow(sharp2)
    plt.title("laplace sharp image")
    plt.show()
