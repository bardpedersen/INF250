import matplotlib.pyplot as plt
from skimage import io
import numpy as np

# reading image

filename = "fiat.jpg"
filename = io.imread(filename)

lift_red = filename[:, :, 0]
max(lift_red.flatten())
min(lift_red.flatten())
plt.imshow(lift_red, vmin=0, vmax=255)
plt.show()

print(lift_red.min(), lift_red.max(), lift_red.mean())
print(filename.shape)
filename[3, 5] = 255

number_rows, number_cols, spam = filename.shape
row, col = np.ogrid[:number_rows, :number_cols]
cnt_row, cnt_col = number_rows / 2, number_cols / 2
outer_disk_mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 > (number_rows / 3) ** 2)
filename[outer_disk_mask] = 0

plt.imshow(filename)
plt.show()


tiff = "fiat.tiff"
tiff = io.imread(tiff)
print(tiff.shape)
