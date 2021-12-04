
import sys
from utils import *

path = sys.argv[1]
img = loadImage(path)

gradients = compute_gradients(img)

print(img.shape)
print(gradients.shape)

plt.imshow(gradients, cmap='gray')
plt.show()

#angles = compute_gradient_angles(img)

#bins = update_bins(gradients, angles)

# Cells

grid = [[1 for j in range(16)] for i in range(16)]

