import skimage
import numpy as np
import matplotlib.pyplot as plt


img = "UrStoy.tif"
image = skimage.io.imread(img)
plt.imshow(image)
plt.title("Original image")
plt.show()


# Histogram
shape = np.shape(image)
histogram_array = np.zeros(256)
for i in range(shape[0]):
    for j in range(shape[1]):
        histogram_index = round(image[i][j])
        histogram_array[histogram_index] += 1

plt.bar(range(256), histogram_array)
plt.title("Histogram of Original image")
plt.show()


# Histogram equalisation
hist_image = image
hist = np.zeros(256)
hist[0] = histogram_array[0]
for i in range(255):
    hist[i + 1] = hist[i] + histogram_array[i + 1]

for i in range(shape[0]):
    for j in range(shape[1]):
        a = int(image[i, j])
        b = hist[a] * 255 / (shape[0] * shape[1])
        hist_image[i, j] = b

plt.imshow(hist_image)
plt.title("Histogram Equalised image")
plt.show()


# Mean filter
mean = skimage.filters.rank.mean(image, skimage.morphology.disk(5))
plt.imshow(mean)
plt.title("Mean filter image")
plt.show()


# Median filter
median = skimage.filters.rank.median(image, skimage.morphology.disk(5))
plt.imshow(median)
plt.title("Median filter image")
plt.show()


# Gaussian filter
gaussian = skimage.filters.gaussian(image, sigma=3)
plt.imshow(gaussian)
plt.title("Gaussian filter image")
plt.show()


# Denoise
denoise = skimage.restoration.denoise_nl_means(image)
plt.imshow(denoise)
plt.title("Denoise image")
plt.show()


# Residual between the histogram equalised image and the noise corrected image
residual_image = image
for i in range(shape[0]):
    for j in range(shape[1]):
        value_residual = abs(hist_image[i][j] - round((denoise[i][j]) * 255))
        residual_image[i][j] = value_residual

plt.imshow(residual_image)
plt.title("Residual image")
plt.show()
