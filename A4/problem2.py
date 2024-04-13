from functools import partial
import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve
conv2d = partial(convolve, mode="mirror")


def compute_derivatives(img1, img2):
    """Compute dx, dy and dt derivatives

    Args:
        img1: first image as (H, W) np.array
        img2: second image as (H, W) np.array

    Returns:
        Ix, Iy, It: derivatives of img1 w.r.t. x, y and t as (H, W) np.array
    
    Hint: the provided conv2d function might be useful
    """
    # Sobel filter for spatial gradients
    sobel_x = np.array([[-1, 0, 1]])
    sobel_y = np.array([[-1], [0], [1]])

    # Temporal gradient filter
    temporal_filter = np.array([[-1, 1]])

    # Compute spatial gradients
    Ix = conv2d(img1, sobel_x) + conv2d(img2, sobel_x)
    Iy = conv2d(img1, sobel_y) + conv2d(img2, sobel_y)

    # Compute temporal gradient
    It = conv2d(img2, temporal_filter) - conv2d(img1, temporal_filter)

    return Ix, Iy, It


def compute_motion(Ix, Iy, It, patch_size=15):
    """Computes one iteration of optical flow estimation.

    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t each as (H, W) np.array
        patch_size: specifies the side of the square region R in Eq. (1)
    Returns:
        u: optical flow in x direction as (H, W) np.array
        v: optical flow in y direction as (H, W) np.array

    Hint: the provided conv2d function might be useful
    """
    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)

    # Create averaging filter (patch) for convolution
    kernel = np.ones((patch_size, patch_size)) / (patch_size ** 2)

    # Apply convolution to compute sums over patches
    Ix2 = conv2d(Ix ** 2, kernel)
    Iy2 = conv2d(Iy ** 2, kernel)
    Ixy = conv2d(Ix * Iy, kernel)
    Ixt = conv2d(Ix * It, kernel)
    Iyt = conv2d(Iy * It, kernel)

    # Iterate over each pixel to solve for u and v
    for y in range(Ix.shape[0]):
        for x in range(Ix.shape[1]):
            # Building matrices A (structure tensor) and b to solve A * [u, v].T = b
            A = np.array([[Ix2[y, x], Ixy[y, x]], [Ixy[y, x], Iy2[y, x]]])
            b = -np.array([Ixt[y, x], Iyt[y, x]])

            # Solving for [u, v].T
            if np.linalg.det(A) != 0:
                uv = np.linalg.solve(A, b)
                u[y, x], v[y, x] = uv

    return u, v


def warp(img, u, v):
    """Warping of a given image using provided optical flow.

    Args:
        img: input image as (H, W) np.array
        u, v: optical flow in x and y direction each as (H, W) np.array

    Returns:
        im_warp: warped image as (H, W) np.array
    """
    # Get the height and width of the image
    h, w = img.shape

    # Create a grid of coordinates in the original image
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply the optical flow vectors to the original coordinates
    new_x = x + u
    new_y = y + v

    # Flatten the arrays
    original_coords = np.vstack([y.ravel(), x.ravel()]).T
    warped_coords = np.vstack([new_y.ravel(), new_x.ravel()]).T

    # Warp the image using griddata for interpolation
    im_warp = interpolate.griddata(points=warped_coords, values=img.ravel(), xi=original_coords, method='linear')

    # Reshape to the original image shape
    im_warp = im_warp.reshape(h, w)

    return im_warp
