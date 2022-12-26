import matplotlib.pyplot as plt
import skimage
import numpy as np


# %% Part 1
rhino_image = skimage.io.imread('rhino_d.tif')
plt.imshow(rhino_image)
plt.title("rhino_image")
plt.show()

rhinodil = skimage.morphology.dilation(rhino_image, skimage.morphology.square(3))
plt.imshow(rhinodil)
plt.title("rhinodil")
plt.show()
rhinoerode = skimage.morphology.erosion(rhino_image, skimage.morphology.square(3))
plt.imshow(rhinoerode)
plt.title("rhinoerode")
plt.show()

rhino_dil_erode = skimage.morphology.erosion(rhinodil, skimage.morphology.square(3))
plt.imshow(rhino_dil_erode)
plt.title("rhino_dil_erode")
plt.show()
rhino_erode_dil = skimage.morphology.dilation(rhinoerode, skimage.morphology.square(3))
plt.imshow(rhino_erode_dil)
plt.title("rhino_erode_dil")
plt.show()

rhino_close = skimage.morphology.closing(rhino_image, skimage.morphology.square(7))
plt.imshow(rhino_close)
plt.title("rhino_close")
plt.show()
rhino_open = skimage.morphology.opening(rhino_image, skimage.morphology.square(4))
plt.imshow(rhino_open)
plt.title("rhino_open")
plt.show()


rhinodil_max1 = skimage.morphology.dilation(rhino_image, skimage.morphology.disk(1))
plt.imshow(rhinodil_max1)
plt.title("rhinodil_max1")
plt.show()
rhinodil_max3 = skimage.morphology.dilation(rhino_image, skimage.morphology.disk(3))
plt.imshow(rhinodil_max3)
plt.title("rhinodil_max3")
plt.show()
rhinoerode_min1 = skimage.morphology.erosion(rhino_image, skimage.morphology.disk(1))
plt.imshow(rhinoerode_min1)
plt.title("rhinoerode_min1")
plt.show()
rhinoerode_min3 = skimage.morphology.erosion(rhino_image, skimage.morphology.disk(3))
plt.imshow(rhinoerode_min3)
plt.title("rhinoerode_min3")
plt.show()


# %% Part 2
rhino_image = skimage.io.imread('rhino_detail.tif')
rhinoerode_2 = skimage.morphology.erosion(rhino_image, skimage.morphology.square(3))

shape = np.shape(rhino_image)
for i in range(shape[0]):
    for j in range(shape[1]):
        if rhino_image[i][j] and not rhinoerode_2[i][j]:
            rhino_image[i][j] = False
        else:
            rhino_image[i][j] = True

plt.imshow(rhino_image, cmap=plt.cm.gray)
plt.show()


# %% Part 3
fingerprint_image = skimage.io.imread('fingerprint.tif')
plt.imshow(fingerprint_image)
plt.show()
finger_erode = skimage.morphology.erosion(fingerprint_image, skimage.morphology.disk(1))
finger_dilate = skimage.morphology.dilation(finger_erode, skimage.morphology.disk(1))
finger_dilate2 = skimage.morphology.dilation(finger_dilate, skimage.morphology.disk(1))
finger_erode2 = skimage.morphology.erosion(finger_dilate2, skimage.morphology.disk(1))

plt.imshow(finger_erode2, cmap=plt.cm.gray)
plt.show()