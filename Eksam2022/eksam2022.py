# Import librarys
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy

# Get image
coffeebeans_image = skimage.io.imread('coffeebeans.jpg')
plt.imshow(coffeebeans_image)
plt.title('Original Image')
plt.show()


#Brown has a red ratio much higher then shadows/gray therefor i am thresholding based on the red ratio
def remove_based_on_red(image, fraction=0.45):
    shape = np.shape(image)
    temp_image = image
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (int(image[i][j][0])+int(image[i][j][1])+int(image[i][j][2])) == 0:
                continue
            elif (int(image[i][j][0]) / (int(image[i][j][0])+int(image[i][j][1])+int(image[i][j][2]))) < fraction:
                temp_image[i][j][0] = 255
                temp_image[i][j][1] = 255
                temp_image[i][j][2] = 255
    return temp_image


no_background_coffee = remove_based_on_red(coffeebeans_image)

plt.imshow(no_background_coffee)
plt.title('No shadows Image Masked')
plt.show()


def masked_image(image):
    shape = np.shape(image)
    temp_image = skimage.color.rgb2gray(image)
    thres_image = skimage.filters.threshold_otsu(temp_image)
    binary_image = temp_image > thres_image
    for i in range(shape[0]):
        for j in range(shape[1]):
            if binary_image[i][j] == 1:
                image[i][j][0] = 255
                image[i][j][1] = 255
                image[i][j][2] = 255
    return image

coffeebeans_image_1 = skimage.io.imread('coffeebeans.jpg')
masked_coffee = masked_image(coffeebeans_image_1)

plt.imshow(masked_coffee)
plt.title('Image Masked')
plt.show()

no_background_coffee_grey = skimage.color.rgb2gray(no_background_coffee)
no_background_coffee_thres_image = skimage.filters.threshold_otsu(no_background_coffee_grey)
binary_image_no_background_coffee = no_background_coffee_grey > no_background_coffee_thres_image

plt.imshow(binary_image_no_background_coffee, cmap=plt.cm.gray)
plt.title('binary Image Masked')
plt.show()


distance = scipy.ndimage.distance_transform_edt(binary_image_no_background_coffee)
local_maxi = skimage.feature.peak_local_max(distance, min_distance=50, indices=False,
                                            footprint=np.ones((3, 3)), labels=binary_image_no_background_coffee)
markers = scipy.ndimage.label(local_maxi)[0]
labels = skimage.segmentation.watershed(-distance, markers, mask=binary_image_no_background_coffee)

handle = {}
# Create objects from the Non stop
properties = skimage.measure.regionprops(labels)

# Create and plot squares that separate Non stop, broken Non stop and m&m
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(no_background_coffee)

for prop in properties:

    minr, minc, maxr, maxc = prop.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.set_axis_off()
plt.tight_layout()
plt.title('Image with seperated non-stop and m&m')
plt.show()