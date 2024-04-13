import numpy as np
from scipy.ndimage import convolve, maximum_filter


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def gauss_2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (h, w) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img: numpy array with the image
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    # Apply Gaussian filter to the image
    smoothed_img = convolve(img, gauss, mode='mirror')

    # Compute first-order derivatives
    img_fx = convolve(smoothed_img, fx, mode='mirror')
    img_fy = convolve(smoothed_img, fy, mode='mirror')
    # Find second-order derivatives
    I_xx = convolve(img_fx, fx.transpose(), mode='mirror')
    I_yy = convolve(img_fy, fy.transpose(), mode='mirror')
    I_xy = convolve(img_fx, fy.transpose(), mode='mirror')
    return I_xx, I_yy, I_xy


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    # Compute the determinant of the Hessian matrix
    det_Hessian = (I_xx * I_yy - I_xy**2)

    # Scale the determinant by sigma^4
    scaled_det_Hessian = det_Hessian / (sigma**4)

    return scaled_det_Hessian


def non_max_suppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """
    # Apply non-maximum suppression to find local maxima
    local_maxima = maximum_filter(criterion, size=5, mode='constant', cval=0) == criterion

    # Discard interest points in a 5-pixel boundary at the image edges
    local_maxima[:5, :] = False
    local_maxima[-5:, :] = False
    local_maxima[:, :5] = False
    local_maxima[:, -5:] = False
    # Thresholding to extract interest points above the specified threshold
    rows, cols = np.where((criterion > threshold) & local_maxima)

    return rows, cols
