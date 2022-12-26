import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

image = "fountain.jpg"
image = io.imread(image)

plt.imshow(image)
plt.show()

filtered_img = gaussian(image, sigma=5, channel_axis=-1)

plt.imshow(filtered_img)
plt.show()

grayscale_image = rgb2gray(filtered_img)

thresh = threshold_otsu(grayscale_image)
binary = grayscale_image > thresh

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(grayscale_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(grayscale_image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Threshold')
ax[2].axis('off')

plt.show()
