import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy

# Get image
image = skimage.io.imread('IMG_2754_nonstop_alltogether.jpg')
image = image[200:3500, 200:5500]
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.show()

# Turn image to binary with yen threshold
grayscale_image = skimage.color.rgb2gray(image)
thresh = skimage.filters.threshold_yen(grayscale_image)
binary_image = grayscale_image > thresh
plt.imshow(binary_image)
plt.title('Binary Image')
plt.show()

# First invert, then fill holes, otherwise all object got removed
binary_invert_image = np.invert(binary_image)
img_fill_holes = scipy.ndimage.binary_fill_holes(binary_invert_image).astype(bool)
plt.imshow(img_fill_holes, cmap=plt.cm.gray)
plt.title('With out holes Image')
plt.show()

# Remove noise
image_erode = skimage.morphology.erosion(img_fill_holes, skimage.morphology.disk(3))
image_dilate = skimage.morphology.dilation(image_erode, skimage.morphology.disk(3))

# Separate Non stop that are close to each other
distance = scipy.ndimage.distance_transform_edt(image_dilate)
local_maxi = skimage.feature.peak_local_max(distance, min_distance=200, indices=False,
                                            footprint=np.ones((3, 3)), labels=image_dilate)
markers = scipy.ndimage.label(local_maxi)[0]
labels = skimage.segmentation.watershed(-distance, markers, mask=image_dilate)

# Create objects from the Non stop
properties = skimage.measure.regionprops(labels)

# Create and plot squares that separate Non stop, broken Non stop and m&m
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

handle = {}

for prop in properties:
    printing = False
    circularity = 4 * np.pi * (prop.area / prop.perimeter ** 2)
    roundnes = 4 * prop.area / (np.pi * prop.axis_major_length**2)
    if roundnes > 0.83 and prop.area > 5000:
        color = 'green'
        printing = True

    elif circularity < 0.83 and prop.area > 5000:
        color = 'blue'
        printing = True

    elif prop.area > 5000:
        color = 'red'
        printing = True
        print(circularity, roundnes)

    if printing == True:
        minr, minc, maxr, maxc = prop.bbox
        handle[color] = ax.add_patch(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor=color, linewidth=2))

ax.set_axis_off()
plt.tight_layout()
plt.title('Image with seperated non-stop and m&m')
plt.legend([handle['green'], handle['blue'], handle['red']], ['Non stop', 'Broken Non stop', 'm&m'])
plt.show()
