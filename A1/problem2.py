import numpy as np
from scipy.ndimage import convolve


def loadbayer(path):
    """ Load data from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array (H,W)
    """
    
    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        bayerdata: Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    r = np.copy(bayerdata)
    g = np.copy(bayerdata)
    b = np.copy(bayerdata)

    r[0::2, 0::2] = 0
    r[1::2] = 0

    g[0::2, 1::2] = 0
    g[1::2, 0::2] = 0

    b[0::2] = 0
    b[1::2, 1::2] = 0

    return r, g, b


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    
    # Using numpy stack with axis = 2 creates a numpy array in the form (H,W,3) 
    img = np.stack([r, g, b], axis=2)

    return img


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        r: red channel as numpy array (H,W)
        g: green channel as numpy array (H,W)
        b: blue channel as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    
    # The source of the used kernels are from the following paper at page 16:
    #
    # Olivier Losson, Ludovic Macaire, Yanqin Yang. Comparison of color demosaicing methods. Advances
    # in Imaging and Electron Physics, 2010, 162, pp.173-265. 10.1016/S1076-5670(10)62005-8. hal-
    # 00683233
    #
    # In summary the green value is modified by considering small amount of blue and red values around blue and red pixels!
   
    G_kernel = 0.25 * np.array([[0, 1, 0],
                                [1, 4, 1],
                                [0, 1, 0]])
    
    RB_kernel = 0.25 * np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]])

    # Mirror handles the edge case to avoid the black pixels of the graph (border) the best!
    R = convolve(r, RB_kernel, mode='mirror')
    G = convolve(g, G_kernel, mode='mirror')
    B = convolve(b, RB_kernel, mode='mirror')

    # The final result is very good, but not perfect!
    img_interpolated = np.stack([R, G, B], axis=2)

    return img_interpolated