from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def load_images_from_folder(path):
    images, filenames, labels = [], [], ["Pos", "Neg"]
    for label in labels:
        folder = path + "/" + label
        for filename in os.listdir(folder):
            img = loadImage(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append([filename,label])
    return images,filenames

def loadImage(path):
    img = mpimg.imread(path)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = np.round(0.299 * R + 0.5870 * G + 0.1140 * B)
    #cv2.imwrite(path,imgGray)
    #plt.imshow(imgGray)
    #plt.show()
    return imgGray

def normalize(image):
    normalized_image = np.round(image/(image.max()/255.0))
    return normalized_image

# Padding the image before convolution to maintain the original shape 
def pad(img, pad_size):
    h,w = img.shape
    padded_img = np.zeros((h + 2*pad_size, w + 2*pad_size))
    for i in range(pad_size,h+pad_size):
        for j in range(pad_size,w+pad_size):
            padded_img[i][j] = img[i-pad_size][j-pad_size]
    return padded_img

# Convolution 
def conv(img, filter):
    irows, icols = img.shape # image
    frows, fcols = filter.shape # filter
    rrows, rcols = irows - frows + 1, icols - fcols + 1 # result
    result = np.zeros( (rrows, rcols) )

    for i in range(rrows):
        for j in range(rcols):
            result[i][j] = np.sum( img[i:i+frows, j:j+fcols] * filter)
    return result


# Compute gradients
def compute_gradients(img): 
    Gx = np.array([ 
                    [-1, 0, 1], # Prewitt's operator for Gradients Gx
                    [-1, 0, 1],
                    [-1, 0, 1]
    ], dtype = 'int')

    Gy = np.array([
                    [1, 1, 1], # Prewitt's operator for Gradients Gy
                    [0, 0, 0],
                    [-1, -1, -1]
    ], dtype = 'int')

    pad_size = 1
    grad_x = pad(conv(img, Gx), pad_size)
    grad_y = pad(conv(img, Gy), pad_size)
    x, y = grad_x.shape

    roi_x, roi_y = np.zeros((x-(2*pad_size),y-(2*pad_size))), np.zeros((x-(2*pad_size),y-(2*pad_size)))
    roi_x_rows, roi_x_cols = roi_x.shape
    for i in range(roi_x_rows):  # Ignoring the border pixels from padding from previous operations
        for j in range(roi_x_cols):
            roi_x[i][j] = grad_x[i][j]
            roi_y[i][j] = grad_y[i][j]

    gradient_magnitude = np.sqrt(roi_x*roi_x + roi_y*roi_y) # Computing gradient magnitude
    normalized_gradient_magnitude = pad(normalize(gradient_magnitude), pad_size) #Returning 0 - 255 range
    return normalized_gradient_magnitude, grad_x, grad_y

# Computer Gradient angles
def compute_gradient_angles(y,x):
    rows, cols = y.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if x[i][j] == 0 and y[i][j]==0:
                result[i][j] = 0
            elif x[i][j] != 0:   # to avoid division by zero
                result[i][j] =  np.degrees( np.arctan(y[i][j] / x[i][j]) )

    #gradient_angle = np.arctan2(y, x)
    #gradient_direction = np.rad2deg(gradient_angle)
    return result

def update_bins(gradients, angles, bins):
    bins = [0] * bins
    rows = len(angles)
    cols = len(angles[0])
    #print(rows, cols)
    for r in range(rows):
        for c in range(cols):
            angle = angles[r][c]
            value = gradients[r][c]

            if angle < 0:
                angle += 180
            elif angle >= 180:    # Keep angle between [0,180]
                angle -= 180

            #print(angle)
            fraction = (angle%20)/20    #Fraction of magnitude to be distributed between the bins
            firstBinValue = (1-fraction) * value
            secondBinValue = (fraction) * value

            firstBin = int(angle // 20)      # Find the index of the bin based on input angle
            if firstBin == 8:
                secondBin = 0
            else:
                secondBin = firstBin + 1

            bins[firstBin] += round(firstBinValue,2)
            bins[secondBin] += round(secondBinValue,2)
    return bins

def compute_hog_feature(gradients, angles, cell_size, step_size, block_size, bins):
    #Computing HOG through 8X8 Cells
    gradient_rows, gradient_cols = gradients.shape
    cell_rows = int((gradient_rows - cell_size)/step_size+1)
    cell_cols = int((gradient_cols - cell_size)/step_size+1)
    cell_histogram_list = []
    flag_row = 0

    for i in range(cell_rows):
        flag_col = 0
        for j in range(cell_cols):
            gradient_magnitude_roi = gradients[flag_row : flag_row+cell_size, flag_col : flag_col+cell_size] #Magnitude region of interest
            gradient_angle_roi = angles[flag_row : flag_row+cell_size, flag_col : flag_col+cell_size] #Angle region of interest
            histogram = update_bins(gradient_magnitude_roi, gradient_angle_roi, bins) #9X1
            cell_histogram_list.append(histogram)
            flag_col += step_size
        flag_row += step_size

    #Converting cell-array list to 20 X 12 X 9 ndarray
    cell_histogram_list = np.reshape(cell_histogram_list,(cell_rows, cell_cols, bins))

    #Normalizing Block
    block_step_size = int(block_size/cell_size)
    block_row = int(cell_rows - block_step_size + 1)
    block_col = int(cell_cols - block_step_size + 1)
    block_histogram_list = []
    flag_row = 0

    for i in range(block_row):
        flag_col = 0
        for j in range(block_col):
            block_roi = cell_histogram_list[flag_row : flag_row+block_step_size, flag_col : flag_col+block_step_size] #Block Region of Interest
            normalized_block_list = block_roi / (np.sqrt(np.sum(block_roi ** 2) + 0.00005)) #L2 norm , adding 0.00005 to avoid division by 0
            normalized_block_list = normalized_block_list.flatten().tolist() #36X1 vector
            block_histogram_list += normalized_block_list
            flag_col += 1
        flag_row += 1

    return block_histogram_list

def write_hog_feature(filename, hog_feature):
    print("Saved file to " + filename)
    np.savetxt(filename, hog_feature)
 