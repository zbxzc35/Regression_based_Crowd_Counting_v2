import cv2
from scipy import ndimage
import numpy as np
import tiah.tools as tools
from dscore.system import LinearDS
from skimage.feature import greycoprops, greycomatrix

def run_SURF_v4(dp_color, perspective, segmentation_set):
    """
    v4 : if keypoint is inside of given segment, accept it.
    Finds SURF keypoints in a whole frame.
    :param dp_color: original image
    :param perspective: weight map which contains a weight value of each pixel.
    :param contour_set: contour list
    :param segmentation_set:  rectangles fitted to contours
    :return:
    """
    score = [0 for x in range(len(segmentation_set))]  # initialize sum of weight of each segmentation
    counts = [0 for x in range(len(segmentation_set))]
    surf = cv2.SURF(200)  # create SURF object
    surf.upright = True

    tmp_dp = ndimage.filters.gaussian_filter(dp_color, 2)
    kp_set = surf.detect(tmp_dp, None)  # estimate only keypoints.

    for kp in kp_set:
        x = int(kp.pt[0])
        y = int(kp.pt[1])

        for i in range(len(segmentation_set)):
            # find the nearest contour from selected keypoint.
            s = segmentation_set[i]  # x1 x2 y1 y2

            # TODO maybe single keypoint could affects multiple segmentation

            if (s[0] < x < s[1]) and (s[2] < y < s[3]):
                score[i] += np.sqrt(perspective[x])
                counts[i] += 1

    return np.array(score)


def run_FAST_v4(dp_color, perspective, segmentation_set):
    """
    with max suppression is better than without max suppression

    v4 : if keypoint is inside of given segment, accept it.

    iterates for kp ( for contour)
    :param img:
    :return:
    """

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector(threshold=60, nonmaxSuppression=True)

    # find and draw the keypoints
    kp_set = fast.detect(dp_color, None)

    score = [0 for x in range(len(segmentation_set))]
    counts = [0 for x in range(len(segmentation_set))]

    for kp in kp_set:
        x = int(kp.pt[0])
        y = int(kp.pt[1])

        checked = False
        for i in range(len(segmentation_set)):
            # find the nearest contour from selected keypoint.
            s = segmentation_set[i]  # x1 x2 y1 y2

            if (s[0] < x < s[1]) and (s[2] < y < s[3]):
                score[i] += np.sqrt(perspective[x])
                counts[i] += 1

    return np.array(score)


def compute_orientation_matrix(dx, dy):
    a = []
    ldx = dx.reshape(dx.size)
    ldy = dy.reshape(dy.size)
    for i in range(len(ldx)):
        d = tools.polar_coordiates(ldx[i], ldy[i], False)
        a.append(d)
    return np.array(a).reshape(dx.shape)


def get_canny_edges(dp_color, weight, segmentation_set, version):
    """
    edge matrix contains binary value, if edge then 255 else 0.
    same size as given image, and it's size is same as given image.

    :param dp_color:
    :param perspective:
    :return:
    """

    from scipy import ndimage
    gray = cv2.cvtColor(dp_color, cv2.COLOR_BGR2GRAY)

    # dir ver 1,2,3,4
    # edges = cv2.Canny(gray, 20, 200)

    #dir ver 5
    edges = cv2.Canny(gray, 140, 250)

    dx = tools.sobel(gray, 0)
    dy = tools.sobel(gray, 1)
    dxy = compute_orientation_matrix(dx, dy)
    dxy %= 180

    histogram_set = []
    for s in segmentation_set:
        histogram = [0.0 for x in range(6)]

        for y in range(s[2], s[3] + 1):
            for x in range(s[0], s[1] + 1):

                if edges[y, x]:
                    d = dxy[y, x]

                    for i in range(6):
                        if d < 30 * (i + 1):
                            histogram[i] += np.sqrt(weight[x])
                            break
        histogram_set.append(histogram)

    return histogram_set


def compute_properties(p):
    contrast = greycoprops(p, 'contrast')
    homogeneity = greycoprops(p, 'homogeneity')
    energy = greycoprops(p, 'energy')
    dissimilarity = greycoprops(p, 'dissimilarity')

    t = [contrast[0][0], homogeneity[0][0], energy[0][0], dissimilarity[0][0]]
    return np.array(t)

def extract_dynamic_texture(colors, states):
    """
    extract dynamic texture of given frame.
    three frames are given, (i-1)-th, i-th, (i+1)-th, and they are 3ch color image.
    convert them into gray-scale first, then extract dynamic texture.
    It returns feature of i-th frame.


    :param colors: 3ch color image
    :param states: states == step
    :return: float64 image [-1,1]
    """
    grays = []
    for i in range(states):
        gray = cv2.cvtColor(colors[i], cv2.COLOR_RGB2GRAY)
        grays.append(gray)

    size = grays[0].size
    shape = grays[0].shape
    length = len(grays)

    data = np.zeros((size, length))  # size of image, number of images

    for i in range(len(grays)):
        data[:, i] = grays[i].reshape(-1)

    lds = LinearDS(states, False, False)  # number of data
    lds.suboptimalSysID(data)

    for i in range(states):
        a = lds._Chat[:, i].reshape(shape)
        grays.append(a)

    return lds._Chat[:, 1].reshape(shape)

def get_texture_T(dp_colors, segmentation_set):
    """
    returns a feature set for a frame.
    feature for each contour.

        # i, i+1, i+2 frames are given to get i-th feature
        # contour should have (i+1)-th index

    :param dp_gray:
    :param perspective:
    :param segmentation_set: i-th segmentation
    :param path:
    :return:
    """

    textures = []
    dt = extract_dynamic_texture(dp_colors, states=3)

    rescaled_8_dt = tools.rescale_domain(dt, [0, 7])

    for s in segmentation_set:
        p = greycomatrix(rescaled_8_dt[s[2]:s[3] + 1, s[0]:s[1] + 1], [1], [0], 8, True, False)
        t = compute_properties(p)
        textures.append(t)

    return textures





