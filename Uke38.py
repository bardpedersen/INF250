import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.filters import threshold_otsu

# 1) display the image
filename = "airfield.tif"
filename = io.imread(filename)
plt.imshow(filename)
plt.show()

# 2) compute the histogram of the image and plot it
shape = np.shape(filename)
histogram_array = np.zeros(256)
for i in range(shape[0]):
    for j in range(shape[1]):
        histogram_index = round(filename[i][j])
        histogram_array[histogram_index] += 1

plt.bar(range(256), histogram_array)
plt.show()

# 3) Try to make a contrast adjustment and a brightness adjustment by
# carrying out simple point operations. Look at the effects on the image.

# increasing contrast with 50 %
for i in range(shape[0]):
    for j in range(shape[1]):
        filename[i, j] = int(filename[i, j] * 1.5 + 0.5)  # +0.5 to round up to int
        if filename[i, j] > 255:
            filename[i, j] = 255
plt.imshow(filename, cmap='gray')
plt.show()

# increasing brightness with 20 units
for i in range(shape[0]):
    for j in range(shape[1]):
        filename[i, j] = int(filename[i, j] - 20)
        if filename[i, j] > 255:
            filename[i, j] = 255
plt.imshow(filename, cmap='gray')
plt.show()

# 4) Carry out a histogram equalisation.
cumhist = np.zeros(256)
cumhist[0] = histogram_array[0]
for i in range(255):
    cumhist[i+1] = cumhist[i] + histogram_array[i+1]

M = shape[0]
N = shape[1]
for i in range(shape[0]):
    for j in range(shape[1]):
        a = int(filename[i, j])
        b = cumhist[a] * (256 - 1) / (M * N)
        filename[i, j] = b

plt.imshow(filename, 'gray')
plt.show()

# 5) plot the histogram of the modified image
new_histogram_array = np.zeros(256)
for i in range(shape[0]):
    for j in range(shape[1]):
        histogram_index = round(filename[i][j])
        new_histogram_array[histogram_index] += 1

plt.bar(range(256), new_histogram_array)
plt.show()

# 6) Carry out an Otsu threshold on the image
varMax = 0
threshold = 0

for i in range(256):
    Bg = histogram_array[:i]
    Fg = histogram_array[i:]
    wB = sum(Bg)/sum(histogram_array)
    wF = sum(Fg)/sum(histogram_array)

    if sum(Bg) == 0 or sum(Fg) == 0:
        continue

    mB = sum(Bg * range(i))/sum(Bg)
    mF = sum(Fg * range(i, 256))/sum(Fg)

    # Calculate Between Class Variance
    varBetween = wB * wF * (mB - mF) * (mB - mF)

    if varBetween > varMax:
        varMax = varBetween
        threshold = i

print(threshold)

image = "airfield.tif"
image = io.imread(image)
thresh = threshold_otsu(image)
print(thresh)
