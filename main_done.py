# Run the program: python main.py data/train train

import cv2
import os
from utils import *
import sys

path = sys.argv[1]
mode = sys.argv[2]

if mode == "train":
    if not os.path.isdir('results'):
        os.mkdir('results')
        os.mkdir('results/train_features')
        os.mkdir('results/test_features')
    images,filenames = load_images_from_folder(path)

    for i in range(len(images)):
        img = images[i]
        filename = './results/train_features/' + list(filenames.keys())[i] + '_features.txt'
        gradient_magnitude, horizontal_gradient, vertical_gradient = compute_gradients(img)
        gradient_angle = compute_gradient_angles(vertical_gradient, horizontal_gradient)
        hog_vector = compute_hog_feature(gradient_magnitude, gradient_angle, cell_size=8, step_size=8, block_size=16, bins=9)
        write_hog_feature(filename, hog_vector)

_, train_files = load_images_from_folder('data/train')
_, test_files = load_images_from_folder('data/test')

train_feature_path = 'results/train_features/'
test_feature_path = 'results/test_features/'


all_distances = {} # Contains distance and name for every training image for all test images
top_three = {} # Contains top three distances (first, second, third), original labels of nearest neighbours, predicted label, actual label

for test in test_files:
    with open(test_feature_path + test +'_features.txt') as file:
        test_vals = file.readlines()
        test_vals = [line.rstrip() for line in test_vals]
        all_distances[test] = {}  

        for train in train_files:
            with open(train_feature_path + train +'_features.txt') as file:
                train_vals = file.readlines()
                train_vals = [line.rstrip() for line in train_vals]
        
            size = len(train_vals)
            numerator = 0
            denominator = 0
            for i in range(size):
                numerator += min( float(train_vals[i]), float(test_vals[i])) # Computing KNN
                denominator += float(train_vals[i])
                
            distance = numerator / denominator
            all_distances[test][train] = distance

        result = sorted(all_distances[test].items(), key=lambda x: x[1], reverse=True) # [imagename, distance]
        top_three[test] = {}
        top_three[test]['first'] = [result[0][1], result[0][0], train_files[result[0][0]]] # [distance, image, training_label]
        top_three[test]['second'] = [result[1][1], result[1][0], train_files[result[1][0]]]
        top_three[test]['third'] = [result[2][1], result[2][0],train_files[result[2][0]]]

        if train_files[result[0][0]] + train_files[result[1][0]] + train_files[result[2][0]] > 1: # If sum is 2 or 3, that means there's a majority of Pos labels
            top_three[test]['prediction'] = 1
        else:
            top_three[test]['prediction'] = 0
        
        top_three[test]['Actual'] = test_files[test]

#for i in all_distances['person_and_bike_151a.bmp']:
#    print(i, all_distances['person_and_bike_151a.bmp'][i])
#print("**"*20)
#result = all_distances['person_and_bike_151a.bmp']
#result = sorted(result.items(), key=lambda x: x[1], reverse=True)

for i in top_three:
    print(i, top_three[i])

