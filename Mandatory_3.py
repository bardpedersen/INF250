#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral import *
import numpy as np
import matplotlib.pyplot as plt

# %% Part1 Load image

hyperim = np.load("sandvika.npy")
wavelength = envi.read_envi_header('Visnir.hdr')['wavelength']
ww = [float(i) for i in wavelength]

# %% Part2 Defines band numbers

def color_band(image_band, wavelength_in_function):
    diff_band = []
    for i, value in enumerate(image_band):
        index_tuple = (i, abs(value - wavelength_in_function))
        diff_band.append(index_tuple)
    closest = sorted(diff_band, key=lambda t: t[1])[0]

    return closest[0]


bands = {
    'blue': color_band(ww, 440),
    'green': color_band(ww, 535),
    'red': color_band(ww, 645),
    'NIR': color_band(ww, 800)}

print("bands = ", bands)
band_red = hyperim[:, :, bands['red']]
band_green = hyperim[:, :, bands['green']]
band_blue = hyperim[:, :, bands['blue']]
band_nir = hyperim[:, :, bands['NIR']]

# %% Part3 Display an RGB image

imshow(hyperim, (bands['red'], bands['green'], bands['blue']),
       stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))

plt.show()

# %% Part4 Computes the NDVI

def calculate_NDVI(NIR, Red):
    NDVI = (NIR - Red) / (NIR + Red)
    return NDVI

ndvi_ima = calculate_NDVI(band_nir, band_red)

# %% Part5 Display the NDVI image

plt.imshow(ndvi_ima, vmin=0, vmax=0.7)
plt.show()

# %% Part6 Select a point with vegetation, asphalt and roof

vegetation = np.array(hyperim[300, 300, :].reshape(-1, 1))
asphalt = np.array(hyperim[75, 5, :].reshape(-1, 1))
roof = np.array(hyperim[350, 424, :].reshape(-1, 1))
plt.figure()
plt.plot(ww, vegetation)
plt.plot(ww, asphalt)
plt.plot(ww, roof)
plt.legend(['vegetation', 'asphalt', 'roof'])
plt.show()

# %% Part7 Histogram of all the NDVI
ndvi_ima = np.nan_to_num(ndvi_ima)
ndvi_histogram, bins = np.histogram(ndvi_ima, bins=30)
plt.plot(ndvi_histogram)
plt.show()

# %% Part8 Make a threshold of NDVI

ndvi_thresholded = ndvi_ima[:]
th = 0.6
ndvi_thresholded[ndvi_thresholded < th] = 0.0
ndvi_thresholded[ndvi_thresholded >= th] = 1.0
imshow(ndvi_thresholded)
plt.show()

# %% Part9 Fraction area of the image that has vegetation
grass = 0
not_grass = 0
for i in range(len(ndvi_thresholded)):
    for j in range(len(ndvi_thresholded[0])):
        if ndvi_thresholded[i][j] == 1:
            grass += 1
        elif ndvi_thresholded[i][j] == 0:
            not_grass += 1

fraction = grass / (not_grass + grass)
print(fraction, "%")
"""
Nearly equal with grass and not.
"""

# %% Part10 Carry out a principal component analysis

pc = principal_components(hyperim)
pc_0999 = pc.reduce(fraction=0.999)
img_pc = pc_0999.transform(hyperim)
plt.imshow(img_pc[:, :, 0], vmin=-0.1, vmax=0.15)
plt.show()
plt.imshow(img_pc[:, :, 1], vmin=-0.1, vmax=0.15)
plt.show()
plt.imshow(img_pc[:, :, 2], vmin=-0.1, vmax=0.15)
plt.show()
loadings = pc_0999.eigenvectors
plt.plot(loadings[:, [0, 1, 2]])
plt.legend(['1', '2', '3'])
plt.show()

# %% Part11 Carry out a k-means clustering

# (m, c) = kmeans(img_pc, 2, 5)
# (m, c) = kmeans(img_pc, 3, 5)
# (m, c) = kmeans(img_pc, 4, 5)

(m, c) = kmeans(img_pc[:, :, 0:3], 5, 30)
plt.imshow(m, 'jet')
plt.figure()
for i in range(c.shape[0]):
    plt.plot(c[i])

imshow(m, stretch_all=True)
plt.show()

# %% Part12

"""
NVDI is the most appropriate. It is a standard procedure for vegetation with known performace, 
gives a degree (not just yes/no) answer. It is also the fastest to compute. PCA would be second, 
it can also give degree. Nice here is that it may highlight other interesting aspects.
"""
