from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def loadImage(path):
    img = mpimg.imread(path)

    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.299 * R + 0.5870 * G + 0.1140 * B
    #plt.imshow(imgGray, cmap='gray')
    #plt.show()
    return imgGray


def normalize(image): 
    normalized_image = np.round(image/(image.max()/255.0))
    return normalized_image

# Padding the image before convolution to maintain the original shape 
def pad(img, pad_size=1):
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
                    result[i][j] = np.sum( img[i:i+frows, j:j+fcols] * filter ) 
    
    return result


# Compute gradients
def compute_gradients(img): 
    Gx = np.array([ 
                    [-1, 0, 1], # Prewitt's operator for Gradients Gx
                    [-1, 0, 1],
                    [-1, 0, 1]
    ]) 

    Gy = np.array([
                    [1, 1, 1], # Prewitt's operator for Gradients Gy
                    [0, 0, 0],
                    [-1, -1, -1]
    ])

    grad_x = pad(conv(img, Gx)) 
    grad_y = pad(conv(img, Gy))  
    
    x, y = grad_x.shape
    pad_x, pad_y = np.zeros((x,y)), np.zeros((x,y))
    
    for i in range(0,x):  # Ignoring the border pixels from padding from previous operations
        for j in range(0,y):
                pad_x[i][j] = grad_x[i][j]
                pad_y[i][j] = grad_y[i][j]

    grad = np.sqrt( pad_x*pad_x + pad_y*pad_y ) # Computing gradient magnitude
    normalized_grad = normalize(grad)

    return normalized_grad

# Computer Gradient angles
def compute_gradient_angles(y,x):
    rows, cols = y.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if x[i][j] != 0:   # to avoid division by zero
                result[i][j] =  np.degrees( np.arctan(y[i][j] / x[i][j]) )

    return result

def update_bins(gradients, angles):

    bins = [0] * 9  
    rows = len(angles)
    cols = len(angles[0])

    for r in range(rows):
        for c in range(cols):
            angle = angles[r][c]
            value = gradients[r][c]

            if angle >= 180:    # Keep angle between [0,180]
                angle -= 180

            fraction = (angle%20)/20    #  Fraction of magnitude to be distributed between the bins
            firstBinValue = (1-fraction) * value
            secondBinValue = (fraction) * value

            firstBin = angle // 20      # Find the index of the bin based on input angle

            if firstBin == 8:
                secondBin = 0
            else:
                secondBin = firstBin + 1

            bins[firstBin] += round(firstBinValue,2)
            bins[secondBin] += round(secondBinValue,2)

    return bins