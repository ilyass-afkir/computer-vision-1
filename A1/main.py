import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    return plt.imread(path)


#
# Problem 1: Getting to know Python
#
from problem1 import *

def problem1():
    """Example code implementing the steps in Problem 1"""

    img = load_image("data/a1p1.png")
    display_image(img)

    save_as_npy("a1p1.npy", img)

    img1 = load_npy("a1p1.npy")
    display_image(img1)

    img2 = mirror_horizontal(img1)
    display_image(img2)

    display_images(img1, img2)


#
# Problem 2: Bayer Interpolation
#
from problem2 import *

def problem2():
    """Example code implementing the steps in Problem 2
    Note: uses display_image() from Problem 1"""

    data = loadbayer("data/bayerdata.npy")
    r, g, b = separatechannels(data)

    img = assembleimage(r, g, b)
    display_image(img)

    img_interpolated = interpolate(r, g, b)
    display_image(img_interpolated)


#
# Problem 3: Projective Transformation
#
from problem3 import * 

def problem3():
    """Example code implementing the steps in Problem 3"""
    t = np.array([-27.1, -2.9, -3.2])
    principal_point = np.array([8, -10])
    focal_length = 8

    # model transformations
    T = gettranslation(t)
    Rx = getxrotation(-45)
    Ry = getyrotation(110)
    Rz = getzrotation(120)
    print(T)
    print(Rx)
    print(Ry)
    print(Rz)

    K = getcentralprojection(principal_point, focal_length)

    P,M = getfullprojection(T, Rx, Ry, Rz, K)
    print(P)
    print(M)

    points = loadpoints("data/obj2d.npy")
    displaypoints2d(points)

    z = loadz('data/zs.npy')
    Xt = invertprojection(K, points, z)

    Xh = inverttransformation(M, Xt)

    worldpoints = hom2cart(Xh)
    displaypoints3d(worldpoints)

    points2 = projectpoints(P, worldpoints)
    displaypoints2d(points2)

    plt.show()

if __name__ == "__main__":
    problem1()
    problem2()
    #problem3()
