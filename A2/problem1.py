import numpy as np
from scipy.ndimage import convolve


def generate_image():
    """ Generates cocentric simulated image in Figure 1.

    Returns:
        Concentric simulated image with the size (210, 210) with increasing intesity through the center
        as np.array.
    """
    # Initializing the required image array with zeros
    img = np.zeros(shape = (210,210))   

    # Array to iterate through the pixel intensity values
    intensities = np.array([0, 30, 60, 90, 120, 150, 180])   

    start = 0
    end = 210

    # Changing the value of the image pixels by iterating through intensity array
    for i in range(1,7):
        img[start + i*15 : end - (i*15) , start + i*15 : end - (i*15) ] = intensities[i]  
    
    return img

def sobel_edge(img):
    """ Applies sobel edge filter on the image to obtain gradients in x and y directions and gradient map.
    (see lecture 5 slide 30 for filter coefficients)

    Args:
        img: image to be convolved
    Returns:
        Ix derivatives of the source image in x-direction as np.array
        Iy derivatives of the source image in y-direction as np.array
        Ig gradient magnitude map computed by sqrt(Ix^2+Iy^2) for each pixel
    """
    # Initializing Dx and Dy for Sobel edge filters as given in the slides 
    Dx = 0.125 * np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
    Dy = 0.125 * np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

    # Applying convolution operation
    Ix = convolve(img, Dx)
    Iy = convolve(img, Dy)

    # Obtaining gradient map according to its formula
    Ig = np.sqrt((Ix*Ix) + (Iy*Iy))

    return Ix, Iy, Ig

def detect_edges(grad_map, threshold=15):
    """ Applies threshold on the edge map to detect edges.

    Args:
        grad_map: gradient map.
        threshold: threshold to be applied.
    Returns:
        edge_map: thresholded gradient map.
    """

    # Thresholding the gradient map
    edge_map = np.where(grad_map >= threshold, grad_map, 0)
    
    #  In addition to the code, please include your response as a comment to the following questions: 

    #  - Question 1: Which threshold recovers the edge map of the original image when working with the noisy image? 
    #  - Answer: The Threshhold = 8 recovers the edges of the noisy image very nicely!
    
    #  - Question 2: How did you determine this threshold value, and why did you choose it?
    #  - Answer: By experimenting with the different threshold values, the threshold value of aproximate 10 gives good edge 
    #            detection result. 
    #
    #            How?: We used the code down below for experimenting with different treshholds.
    #                  In particular we used/tried combinations of (start, end, step) as follows: (0,200,50), (0, 50, 10), (0,20,2)
    #            Why?: Finding a nice threshhold by automating the process is faster and more efficient than manually typing in
    #                  each number over and over again!
    #            --------------------------------------------------------------------------------
    #            CODE
    #            --------------------------------------------------------------------------------
    #            threshholds = np.arange(start,end, step)
    #            for value in threshholds:
    #                edge_map_noisy = detect_edges(grad_map_noisy, threshold=value)
    #                show_image(edge_map_noisy) 
    #            ---------------------------------------------------------------------------------

    return edge_map

def add_noise(img, mean=0, variance=15):
    """ Applies Gaussian noise on the image.

    Args:
        img: image in np.array
        mean: mean of the noise distribution.
        variance: variance of the noise distribution.
    Returns:
        noisy_image: gaussian noise applied image.
    """
    # Generating noise with the given mean and variance (variance will be converted to standard deviation)
    noise = np.random.normal(mean, np.sqrt(variance), img.shape)  
    
    # Adding gaussian noise to image
    noisy_image = img + noise 

    # Bringing values in the range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image
