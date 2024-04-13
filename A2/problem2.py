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
    #
    # You code here     
    #
    img = Image.open(path)
    img_array = np.array(img)
    img_normalized = img_array/255.
    
    return img_normalized

def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    #
    # You code here     
    #
    m = (fsize[0]-1)/2.
    n = (fsize[1]-1)/2.

    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp( -(x**2 + y**2) / (2.*sigma**2) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0 # np.finfo(h.dtype).eps returns the smallest representable positive number such that 1.0 + eps != 1.0 for the data type of h
    return h


def binomial_2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    #
    # You code here
    #
    # 1D binomial filter
    k = np.arange(fsize[0])
    binomial_1d = binom(fsize[0]-1, k)

    # 2D binomial filter by outer product of the 1D filter with itself
    binomial_2d = np.outer(binomial_1d, binomial_1d)

    # Normalization
    binomial_2d /= binomial_2d.sum()

    return binomial_2d


def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    #
    # You code here
    # 
    img_filtered = convolve(img, f)
    img_downsampled = img_filtered[::2, ::2]
    return img_downsampled
    


def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    #
    # You code here
    #
    rows, cols = img.shape
    img_upsampled = np.zeros((rows*2, cols*2))
    img_upsampled[::2, ::2] = img
    img_upsampled_filtered = convolve(img_upsampled, f, mode='mirror')
    img_upsampled_filtered *= 4 # a scale factor of 4 brings the average intensity back up to the original level
    return img_upsampled_filtered


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
    #
    # You code here
    #
    gpyramid = [img]
    for i in range(nlevel - 1):
        img = downsample2(img, f)
        gpyramid.append(img)
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
    #
    # You code here
    #
    lpyramid = []
    for i in range(len(gpyramid) - 1):
        laplacian = gpyramid[i] - upsample2(gpyramid[i + 1], f)
        lpyramid.append(laplacian)
    lpyramid.append(gpyramid[-1]) # The top level of the Laplacian pyramid is the same as the top level of the Gaussian pyramid
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
    #
    # You code here
    #   
    rows, cols = pyramid[0].shape
    composite_image = np.zeros((rows, cols * len(pyramid)))

    for i, img in enumerate(pyramid):
        normalized_img = (img - img.min()) / (img.max() - img.min())
    #    composite_image[:img.shape[0], i * cols:(i + 1) * cols] = normalized_img

        resized_img = np.zeros((rows, cols))
        resized_img[:img.shape[0], :img.shape[1]] = normalized_img
        composite_image[:, i * cols:(i + 1) * cols] = resized_img

    return composite_image


def amplify_high_freq(lpyramid, l0_factor=1, l1_factor=1):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    #
    # You code here
    #  
    amplified_lpyramid = deepcopy(lpyramid)
    amplified_lpyramid[0] *= l0_factor
    amplified_lpyramid[1] *= l1_factor
    return amplified_lpyramid
    
    
def reconstruct_image(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    #
    # You code here
    #     
    img_reconstructed = deepcopy(lpyramid[-1])
    for img in reversed(lpyramid[:-1]):
        img_upsampled = upsample2(img_reconstructed, f)
        img_reconstructed = img + img_upsampled[:img.shape[0], :img.shape[1]]
    return np.clip(img_reconstructed, 0, 1)
