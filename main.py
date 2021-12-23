# Project: Human Detection using HOG
# Run the program: python main.py "Filepath/Human-Detection-using-HOG" "train"
# Run the program: python main.py "Filepath/Human-Detection-using-HOG" "test"

from utils import *
import sys
import cv2

# Takes in input as argument from the user
path = sys.argv[1]
mode = sys.argv[2]

# If mode is train, positive and negative training images are read and HOG feature vector is calculated
if mode.lower() == "train":
    if not os.path.isdir('results'):
        os.mkdir('results')
        os.mkdir('results/train_features')
        os.mkdir('results/test_features')
    train_images, filenames = load_images_from_folder(path + '/data/train/')

    for i in range(len(train_images)):
        img = train_images[i]
        filename = path + '/results/train_features/' + list(filenames.keys())[i] + '_features.txt'
        get_hog_feature(img, filename)

# If mode is train, test images are loaded, HOG Feature Vector is calculated and the test image is classified as Human or No-Human
if mode.lower() == "test":
    print("Running in Test Mode to generate HOG feature vector for the test.")
    train_images, train_files = load_images_from_folder(path + '/data/train/')
    test_images, test_files = load_images_from_folder(path + '/data/test/')

    print("Loading images from train and test")
    train_feature_path = 'results/train_features/'
    test_feature_path = 'results/test_features/'

    for i in range(len(test_images)):
        img = test_images[i]
        gradient_magnitude, horizontal_gradient, vertical_gradient = compute_gradients(img)
        filename = list(test_files.keys())[i]
        cv2.imwrite(filename, gradient_magnitude)
        filename = path + '/results/test_features/' + list(test_files.keys())[i] + '_features.txt'
        get_hog_feature(img, filename)

    classification = get_nearest_neighbour(train_files, test_files, train_feature_path, test_feature_path)
    for key, values in classification.items():
        print(key, values)
