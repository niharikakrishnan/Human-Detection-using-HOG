import sys
from utils import *

filename = "hog_feature.txt"
cell_size = 8
step_size = 8
bins = 9
block_size = 16

path = sys.argv[1]
img = loadImage(path)
gradient_magnitude, horizontal_gradient, vertical_gradient = compute_gradients(img)
gradient_angle = compute_gradient_angles(vertical_gradient, horizontal_gradient)
hog_vector = compute_hog_feature(gradient_magnitude, gradient_angle, cell_size, step_size, block_size, bins)
write_hog_feature(filename, hog_vector)
