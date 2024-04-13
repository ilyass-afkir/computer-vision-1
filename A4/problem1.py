import numpy as np
from numpy.linalg import norm


def cost_ssd(patch_l, patch_r):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """
    # Calculate SSD as displayed in the formula
    cost_ssd = np.sum((patch_l - patch_r) ** 2)

    return cost_ssd


def cost_nc(patch_l, patch_r):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    # Convert patches to vectors
    patch_l = patch_l.flatten()
    patch_r = patch_r.flatten()

    # Calculate the mean of the patches
    patch_l_mean = np.mean(patch_l)
    patch_r_mean = np.mean(patch_r)

    # Calculate the numerator of the normalized correlation cost as displayed in the formula
    numerator = np.dot(patch_l - patch_l_mean, patch_r - patch_r_mean)
    
    # Calculate the denominator of the normalized correlation cost function as displayed in the formula
    denominator = norm(patch_l-patch_l_mean) * norm(patch_r-patch_r_mean)

    # Calculate the normalized correlation cost as displayed in the formula
    cost_nc = numerator/denominator

    return cost_nc


def cost_function(patch_l, patch_r, alpha):
    """Compute the cost between two input window patches
    
    Args:
        patch_l: input patch 1 as (m, m) numpy array
        patch_r: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    # Get the dimension m of a patch
    m = patch_l.shape[0]

    # Calculate the cost between two input window patches as displayed in the formula
    cost_val = (1.0 / m ** 2) * cost_ssd(patch_l, patch_r) + alpha * cost_nc(patch_l, patch_r)

    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Add padding to the input image based on the window size
    
    Args:
        input_img: input image as 2-dimensional (H,W) numpy array
        window_size: window size as a scalar value (always and odd number)
        padding_mode: padding scheme, it can be 'symmetric', 'reflect', or 'constant'.
            In the case of 'constant' assume zero padding.
        
    Returns:
        padded_img: padded image as a numpy array of the same type as input_img
    """
    # Calculate the padding width for an odd-sized window. 
    pad_width = window_size // 2

    # Apply padding based on the specified padding_mode
    if padding_mode == 'constant':
        # Zero padding
        padded_img = np.pad(input_img, pad_width, mode='constant', constant_values=0)
    
    elif padding_mode == 'symmetric':
        # Symmetric padding
        padded_img = np.pad(input_img, pad_width, mode='symmetric')
    
    elif padding_mode == 'reflect':
        # Reflect padding
        padded_img = np.pad(input_img, pad_width, mode='reflect')

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map using the window-based matching strategy    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """
    # Initialize the disparity map
    H, W = padded_img_l.shape
    disparity_map = np.zeros((H, W))

    # Padding width for the odd sized window
    pad_width = window_size // 2

    # Iterate over each pixel in the left image
    for y in range(pad_width, H - pad_width):
        for x in range(pad_width, W - pad_width):
            min_cost = float('inf')
            best_disp = 0

            # Iterate over the disparity range
            for d in range(max_disp + 1):
                if x - d < pad_width: 
                    # Ensures that the window in the right image does not go beyond its padded boundary
                    # Skip disparity if it goes beyond the image boundary
                    continue

                # Extract the patches
                patch_l = padded_img_l[y - pad_width:y + pad_width + 1, x - pad_width:x + pad_width + 1]
                patch_r = padded_img_r[y - pad_width:y + pad_width + 1, x - d - pad_width:x - d + pad_width + 1]

                # Compute the cost
                cost = cost_function(patch_l, patch_r, alpha)

                # Find the disparity with the minimum cost
                if cost < min_cost:
                    min_cost = cost
                    best_disp = d

            # Set the disparity value for the current pixel
            disparity_map[y, x] = best_disp

    # Remove the padding from the disparity map
    disparity_map = disparity_map[pad_width:-pad_width, pad_width:-pad_width]

    return disparity_map


def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map
    
    Args:
        disparity_gt: ground truth of disparity map as (H, W) numpy array
        disparity_res: estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    # Calculate the L1 norm of the difference between the two maps
    l1_norm = np.sum(np.abs(disparity_gt - disparity_res))

    # Calculate the number of pixels (N)
    N = disparity_gt.size

    # Compute AEPE
    aepe = l1_norm / N

    return aepe


def optimal_alpha():
    """Return alpha that leads to the smallest EPE (w.r.t. other values)
    Note:
    Remember to check that max_disp = 15, window_size = 11, and padding_mode='symmetric'
    """
    #
    # Once you find the best alpha, you have to fix it
    #
    alpha = np.random.choice([-0.001, -0.01, -0.1, 0.1, 1, 10])
    
    return -0.001


"""
This is a multiple-choice question
"""
def window_based_disparity_matching():
    """Complete the following sentence by choosing the most appropriate answer 
    and return the value as a tuple.
    (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
    
    Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
        1: Using a bigger window size (e.g., 11x11)
        2: Using a smaller window size (e.g., 3x3)
        
    Q2. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
        1: symmetric
        2: reflect
        3: constant

    Q3. The inaccurate disparity estimation on the left image border happens due to [?].
        1: the inappropriate padding scheme
        2: the limitations of the fixed window size
        3: the absence of corresponding pixels
        
    Example or reponse: (1,1,1)
    """
    return (2, 3, 3)
