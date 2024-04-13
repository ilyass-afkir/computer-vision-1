from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve


def load_img(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """

    # Loading the image with the PILLOW Library and 
    # then creating a numpy array
    img = np.array(Image.open(path)) 
    
    # Bringing all values in range [0,1]
    img = img/255     
    
    return img    
    

def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    # The formula can be found in lecture 4, slide 24 

    # Initializing gaussian filter of required shape
    gf = np.zeros(shape = fsize)  
    W, H = fsize

    # Getting center coordinates (Cx, Cy means center_x, center_y)
    Cx, Cy = W//2, H//2     
    sum = 0
    for x in range(0,W):
        for y in range(0,H):
            # Applying the gaussian formula to all pixels, starting from the center 
            # (by giving an offset with the center coordinates Cx, Cy)
            gf[x][y] = (1/(2 * np.pi * (sigma**2))) * np.exp((-1 * ((x-Cx)**2 + (y-Cy)**2))/(2 * (sigma**2)))     
            sum += gf[x][y]
    
    # Normalising values
    norm_gf = gf/sum   

    return norm_gf


def binomial_2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    W, H = fsize

    # Generate binomial coefficients for each dimension
    # For a filter of size n, we use order n-1
    h_coeff = np.array([binom(H-1, i) for i in range(H)])   
    w_coeff = np.array([binom(W-1, i) for i in range(W)])

    # Normalize binomial coefficients
    h_coeff = h_coeff / np.sum(h_coeff) 
    w_coeff = w_coeff / np.sum(w_coeff) 
    
    # Compute the outer product to create a 2D filter
    bf = np.outer(h_coeff, w_coeff)   
    
    return bf


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    # Apply Gaussian filter
    filtered_img = convolve(img, f, mode="mirror")

    # Downsample the image by factor of 2 (discard every other row and column)
    downsampled_img = filtered_img[::2, ::2]

    # Clip the values to be within valid range (boundary conditions)
    downsampled_img = np.clip(downsampled_img, 0, 1)

    return downsampled_img


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    H, W = img.shape
    
    # Create a temporary variable for the upsampled image
    temp_img = img.copy()

    # Inserting new rows with zero value
    for i in range(1, 2 * H, 2):
        temp_img = np.insert(temp_img, i, 0, axis=0)

    # Inserting new columns with zero value
    for i in range(1, 2 * W, 2):
        temp_img = np.insert(temp_img, i, 0, axis=1)

    # Apply the filter and scaling factor
    upsampled_img = convolve(temp_img, f, mode="mirror") * 4

    # Clipping the values to be within [0, 1]
    upsampled_img = np.clip(upsampled_img, 0, 1)

    return upsampled_img


def gaussian_pyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    gpyramid = [img]
    curr_img = img
    for level in range(1, nlevel):
        # Use deepcopy to avoid modifying the original or previous level images
        curr_img = deepcopy(curr_img)
        # Downsampling image to add to Gaussian pyramid
        curr_img = downsample2(curr_img, f)  
        gpyramid.append(curr_img)

    return gpyramid


def laplacian_pyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    lpyramid = []
    # Reverse the Gaussian pyramid for processing from coarse to fine
    gpyramid = deepcopy(gpyramid[::-1])

    for i in range(len(gpyramid)):
        # For the top level, simply copy the image from the Gaussian pyramid
        if i == 0:
            lpyramid.append(gpyramid[0])
        else:
            # For other levels, subtract the upsampled version of the previous level from the current level
            lpyramid.append(gpyramid[i] - upsample2(gpyramid[i-1], f))

    # Reverse the Laplacian pyramid to get it sorted from fine to coarse
    lpyramid = lpyramid[::-1]

    #------------------------------------------------------------------------------------------------------------#
    # - Question: What is the difference between the top (coarsest) level of Gaussian and Laplacian pyramids?
    # - Answer: The top (coarsest) level of the Gaussian pyramid is a downsampled version of the image,
    #           while the corresponding level in the Laplacian pyramid represents high-frequency components
    #           (details) missed at each level of the Gaussian pyramid.
    #------------------------------------------------------------------------------------------------------------#

    return lpyramid


def create_composite_image(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    # Determine the height of the finest level and the total width for the composite image
    height = pyramid[0].shape[0]
    total_width = sum([img.shape[1] for img in pyramid])

    # Create an empty array for the composite image
    composite_img = np.zeros((height, total_width))

    # Track the current x position in the composite image
    current_x = 0

    for img in pyramid:
        # Normalize each level of the pyramid
        norm_img = img.copy()
        norm_img -= norm_img.min()
        norm_img /= norm_img.max()

        # Determine the width and height of the current image
        H, W = norm_img.shape

        # Place the normalized image into the composite image
        composite_img[:H, current_x:current_x + W] = norm_img

        # Update the x position for the next image
        current_x += W

    return composite_img


def amplify_high_freq(lpyramid, l0_factor=1.5, l1_factor=2.5):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    lpyramid_amp = deepcopy(lpyramid)

    # Amplify the finest level
    lpyramid_amp[0] *= l0_factor
    np.clip(lpyramid_amp[0], 0, 1, out=lpyramid_amp[0])

    # Amplify the second finest level
    if len(lpyramid) > 1:
        lpyramid_amp[1] *= l1_factor
        np.clip(lpyramid_amp[1], 0, 1, out=lpyramid_amp[1])

    #-----------------------------------------------------------------------------------------------------------#
    # After experimenting with different values for l0_factor and l1_factor,
    # l0_factor = 1.5 and l1_factor = 2.5 gives a good sharpened image as result!
    # Since we are building a sharpening application, such an output of the reconstructed image will be desired!
    #-----------------------------------------------------------------------------------------------------------#

    return lpyramid_amp
    
    
def reconstruct_image(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    # Start with the coarsest level
    img_rec = deepcopy(lpyramid[-1])

    # Iterate through the pyramid, upsampling and adding each level
    for i in range(len(lpyramid)-2, -1, -1):
        # Upsample the reconstructed image
        img_rec = upsample2(img_rec, f)
        # Add the current pyramid level to the upsampled image
        img_rec += lpyramid[i]

    # Clip the values to be within [0, 1]
    img_rec = np.clip(img_rec, 0, 1)

    return img_rec
