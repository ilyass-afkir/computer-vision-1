import numpy as np
import matplotlib.pyplot as plt


################################################################
#            DO NOT EDIT THESE HELPER FUNCTIONS                #
################################################################

# Plot 2D points
def displaypoints2d(points):
    plt.figure()
    plt.plot(points[0,:],points[1,:], '.b')
    plt.xlabel('Screen X')
    plt.ylabel('Screen Y')


# Plot 3D points
def displaypoints3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:], 'b')
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")

################################################################


def gettranslation(v):
    """ Returns translation matrix T in homogeneous coordinates 
    for translation by v.

    Args:
        v: 3d translation vector

    Returns:
        Translation matrix in homogeneous coordinates
    """

    # The strucutre of the translation matrix in homogeneous coordinates can be seen here: 
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    
    # Create a 4x4 identity matrix 
    T = np.eye(4)

    # Add the translation v in the last column and first 3 rows
    T[:3, 3] = v

    return T


def getyrotation(d):

    """ Returns rotation matrix Ry in homogeneous coordinates for 
    a rotation of d degrees around the y axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """
    
    # The input of np.cos and np.sin need to be in radians!
    rad = np.radians(d)
    
    # The strucutre of the rotation matrix Ry in homogeneous coordinates for 
    # a rotation of d degrees around the y axis can be seen here: 
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    Ry = np.array([[np.cos(rad), 0, np.sin(rad), 0],
                     [0, 1, 0, 0],
                     [-np.sin(rad), 0, np.cos(rad), 0],
                     [0, 0, 0, 1]])

    return Ry


def getxrotation(d):
    """ Returns rotation matrix Rx in homogeneous coordinates for a 
    rotation of d degrees around the x axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """

    # Input of np.cos and np.sin needs to be in radians
    rad = d * np.pi / 180
    
    # Helpful Source: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html

    Rx = np.array([[1, 0, 0, 0],
                     [0, np.cos(rad), -np.sin(rad), 0],
                     [0, np.sin(rad), np.cos(rad), 0],
                     [0, 0, 0, 1]])

    return Rx
    

def getzrotation(d):
    """ Returns rotation matrix Rz in homogeneous coordinates for a 
    rotation of d degrees around the z axis.

    Args:
        d: degrees of the rotation

    Returns:
        Rotation matrix
    """

    # The input of np.cos and np.sin need to be in radians!
    rad = np.radians(d)
    
    # The strucutre of rotation matrix Rz in homogeneous coordinates for a 
    # rotation of d degrees around the z axis can be seen here: 
    # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    Rz = np.array([[np.cos(rad), -np.sin(rad), 0, 0],
                     [np.sin(rad), np.cos(rad), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    return Rz


def getcentralprojection(principal, focal):
    """ Returns the (3 x 4) matrix L that projects homogeneous camera 
    coordinates on homogeneous image coordinates depending on the 
    principal point and focal length.

    Args:
        principal: the principal point, 2d vector
        focal: focal length

    Returns:
        Central projection matrix
    """

    # The structure of the homogeneous central projection matrix L (or the calibration matrix K) can be seen in lecture 2 slide 41:
    L = np.array([[focal, 0, principal[0], 0],
                  [0, focal, principal[1], 0],
                  [0, 0, 1, 0]])
   
    return L


def getfullprojection(T, Rx, Ry, Rz, L):
    """ Returns full projection matrix P and full extrinsic 
    transformation matrix M.

    Args:
        T: translation matrix
        Rx: rotation matrix for rotation around the x-axis
        Ry: rotation matrix for rotation around the y-axis
        Rz: rotation matrix for rotation around the z-axis
        L: central projection matrix

    Returns:
        P: projection matrix
        M: matrix that summarizes extrinsic transformations
    """
    
    # Full extrinsic transformation matrix M can be seen in lecture 2, slide 33
    M = np.matmul(Rz, np.matmul(Rx, np.matmul(Ry,T)))

    # Full projection matrix P can be seen in lecture 2, slide 45
    P = np.matmul(L, M)

    return P, M


def cart2hom(points):
    """ Transforms from cartesian to homogeneous coordinates.

    Args:
        points: a np array of points in cartesian coordinates

    Returns:
        A np array of points in homogeneous coordinates
    """
    # When transforming a point from Cartesian to homogeneous coordinates, 
    # it involves adding a "1" as a new coordinate. See lecture 2, slide 29

    ones = np.ones((points.shape[1]))
    points_hom = np.vstack((points, ones))

    return points_hom


def hom2cart(points):
    """ Transforms from homogeneous to cartesian coordinates.

    Args:
        points: a np array of points in homogenous coordinates

    Returns:
        A np array of points in cartesian coordinates
    """
    # When transitioning a point from homogeneous to Cartesian coordinates,
    # divide each coordinate by the last one and discard the last coordinate.

    cart_points = points[:-1]/points[-1]

    return cart_points


def loadpoints(path):
    """ Load 2d points from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """
    
    points = np.load(path)

    return points


def loadz(path):
    """ Load z-coordinates from file

    Args:
        path: Path of the .npy file
    Returns:
        Data as numpy array
    """

    z = np.load(path)

    return z


def invertprojection(L, P2d, z):
    """
    Invert just the projection L of cartesian image coordinates 
    P2d with z-coordinates z.

    Args:
        L: central projection matrix
        P2d: 2d image coordinates of the projected points
        z: z-components of the homogeneous image coordinates

    Returns:
        3d cartesian camera coordinates of the points 
    """
    # Create homogenous coordinates for P2d using z
    P2d_hom = np.vstack((P2d * z, z))

    # Calculate Pseudo Inverse of central projection matrix L (3x4 Matrix)
    L_inv = np.linalg.pinv(L)

    # Calculate Homogenous camera coordinates
    camerapoints_hom = np.matmul(L_inv, P2d_hom)

    # Convert homogenous camera coordinates to cartesian camera coordinates.
    # We are using a small epsilon value to avoid division by zero (this trick finally made this code WORK!). 
    epsilon = 1e-8  
    camerapoints_cart = camerapoints_hom[:-1] / (camerapoints_hom[-1] + epsilon)

    return camerapoints_cart


def inverttransformation(M, P3d):
    """ Invert just the model transformation in homogeneous 
    coordinates for the 3D points P3d in cartesian coordinates.

    Args:
        M: matrix summarizing the extrinsic transformations
        P3d: 3d points in cartesian coordinates

    Returns:
        3d points after the extrinsic transformations have been reverted
    """
    # Create homogenous coordinates for P3d
    ones = np.ones((P3d.shape[1]))
    P3d_hom = np.vstack((P3d, ones))

    # Calculate the Inverse of extrinsic transformation matrix M (4x4 matrix)
    M_inv = np.linalg.inv(M)

    # Calculate homogenous world coordinates
    worldpoints_hom = np.matmul(M_inv, P3d_hom)

    return worldpoints_hom


def projectpoints(P, X):
    """ Apply full projection matrix P to 3D points X in cartesian coordinates.

    Args:
        P: projection matrix
        X: 3d points in cartesian coordinates

    Returns:
        x: 2d points in cartesian coordinates
    """

    # Create homogenous coordinates for X
    ones = np.ones((X.shape[1]))
    X_hom = np.vstack((X, ones))

    # Calculate homogenous image coordinates
    P2d_hom = np.matmul(P, X_hom)

    # Convert homogenous image coordinates to cartesian image coordinates
    P2d = P2d_hom[:-1]/P2d_hom[-1]

    return P2d


def p3multiplechoice(): 
    '''
    Change the order of the transformations (translation and rotation).
    Check if they are commutative. Make a comment in your code.
    Return 0, 1 or 2:
    0: The transformations do not commute.
    1: Only rotations commute with each other.
    2: All transformations commute.
    '''

    return 0

