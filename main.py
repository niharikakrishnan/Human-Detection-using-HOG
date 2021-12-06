# Run the program: python main.py data/train train

import cv2
import os
from utils import *
import sys

path = sys.argv[1]
mode = sys.argv[2]

if mode == "train":
    os.mkdir('results')
    os.mkdir('results/train_features')
    images,filenames = load_images_from_folder(path)

    for i in range(len(images)):
        img = images[i]
        filename = './results/train_features/' + filenames[i][0] + '_features.txt'
        gradient_magnitude, horizontal_gradient, vertical_gradient = compute_gradients(img)
        gradient_angle = compute_gradient_angles(vertical_gradient, horizontal_gradient)
        hog_vector = compute_hog_feature(gradient_magnitude, gradient_angle, cell_size=8, step_size=8, block_size=16, bins=9)
        write_hog_feature(filename, hog_vector)
        file = open(filename,"a")
        file.write(filenames[i][1]) # Writing the label at the end of the feature vector
        file.close()

