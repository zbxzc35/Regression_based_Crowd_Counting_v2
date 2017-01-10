import cv2
from scipy import ndimage
import numpy as np
import tiah.tools as tools
from dscore.system import LinearDS
from skimage.feature import greycoprops, greycomatrix



def get_size_L(fg, weight, contour_set):
    size = len(contour_set)
    L = [ 0 for x in range(size)]

    for i in range(size):
        contour = contour_set[i]
        for pt in contour:
            coordinate = pt[0]
            L[i] += np.sqrt(weight[coordinate[0]])  # x? y?
    return L



def get_size_S(fg, weight, contour_set):

    size = len(contour_set)
    S = [ 0 for x in range(size)]

    for i in range(size):
        contour = contour_set[i]
        for x in range(fg.shape[1]):  # x
            for y in range(fg.shape[0]):  # y
                if fg[y,x]:
                    ret = cv2.pointPolygonTest(contour, (x,y), False)
                # ret>0 inside ret<0 : outside ret=0 :edge
                    if ret >= -2:
                        S[i] += weight[x]
    return S

def get_size_S_v2(fg, weight, segmentation_set):

    size = len(segmentation_set)
    S = [ 0 for x in range(size)]

    for i in range(size):
        s = segmentation_set[i]

        for y in range(fg.shape[1]):  # x
            for x in range(fg.shape[0]):  # y

                if (s[0]<x<s[1]) and (s[2]<y<s[3]):
                    if fg[y,x] == 1:
                        S[i] += weight[x]
    return S


def get_shape_P(fg, weight, contour_set):
    """

    :param fg: single
    :param perspective:
    :param contour: single
    :return:
    """
    P = []

    for contour in contour_set:


        histogram = [0 for x in range(4)]

        for i in range(len(contour) - 1):
            pre = contour[i][0]
            curr = contour[i + 1][0]

            d = tools.polar_coordiates(pre, curr, True)
            assert (d > -1), str(i) + ' d should be not negative.'
            d %= 180
            x = pre[0]
            y = pre[1]

            if d < 45:
                histogram[0] += np.sqrt(fg[y, x])
            elif d < 90:
                histogram[1] += np.sqrt(fg[y, x])
            elif d < 135:
                histogram[2] += np.sqrt(fg[y, x])
            else:
                histogram[3] += np.sqrt(fg[y, x])

        P.append(histogram)
    return P