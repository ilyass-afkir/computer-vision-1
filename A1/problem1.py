import numpy as np
import matplotlib.pyplot as plt


################################################################
#             DO NOT EDIT THIS HELPER FUNCTION                 #
################################################################

def load_image(path):
    return plt.imread(path)

################################################################

def display_image(img):
    """ Show an image with matplotlib

    Args:
        img: Image as numpy array (H,W,3)
    """
    plt.imshow(img, interpolation='none')
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file

    Args:
        img: Image as numpy array (H,W,3)
    """
    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file

    Args:
        path: Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image

    Args:
        img: Image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    return np.fliplr(img)


def display_images(img1, img2):
    """ Display the normal and the mirrored image in one plot

    Args:
        img1: First image to display
        img2: Second image to display
    """
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1, interpolation='none')
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2, interpolation='none')
    plt.show(block=True)