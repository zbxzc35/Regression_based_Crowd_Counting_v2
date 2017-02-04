import math
from scipy import stats

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2GRAY
from sys import float_info

# color is bgr!!!!
red = (0, 0, 255)  # rect to be discarded
blue = (255, 0, 0)  # new rect
green = (0, 255, 0)  # good rect
white = (255, 255, 255)
magenta = (250, 0, 250)
orange = (0, 105, 255)  # updated rect


def count_all_elements(src):
    if isinstance(src, list):
        n = 0
        for i in range(len(src)):
            n += count_all_elements(src[i])

        return n
    else:
        return len(src)


def int2round(src):
    """
    returns rounded integer recursively
    :param src:
    :return:
    """
    if isinstance(src, float):
        return int(round(src))

    elif isinstance(src, tuple):
        res = []
        for i in range(len(src)):
            res.append(int(round(src[i])))
        return tuple(res)

    elif isinstance(src, list):
        res = []
        for i in range(len(src)):
            res.append(int2round(src[i]))
        return res
    elif isinstance(src, int):
        return src
    if isinstance(src, str):
        return int(src)


def rescale_domain(a, range_):
    aa = a.reshape(a.size)
    oldMin = min(aa)
    oldMax = max(aa)
    newMin = range_[0]
    newMax = range_[1]

    OldRange = (oldMax - oldMin)
    NewRange = (newMax - newMin)

    b = []
    for e in aa:
        n = (((e - oldMin) * NewRange) / OldRange) + newMin
        b.append(n)
    bb = np.array(b)

    return bb.reshape(a.shape)


def plot_image(src, row, labels=None, path=None, fname=None, title=None):
    # plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=0.9)

    # assert(len(src) == len(labels)) , 'src and label size are different!'
    col = len(src) // row
    idx = col * 100 + row * 10
    # print 'tools, plot image src[0]', src[0].shape, src[0].dtype, src[0][0,0] ,src[1].shape,src[1][0,0] , src[1].dtype
    fig = plt.figure(1)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    for n in range(len(src)):
        plt.subplot(col, row, n + 1)
        plt.axis('off')

        if len(src[n].shape) == 3:
            plt.imshow(cvtColor(src[n], COLOR_BGR2RGB))
        else:
            plt.imshow(src[n], cmap='Greys_r')
            # plt.imshow(src[n])

        if labels:
            plt.title(labels[n])

    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/' + fname + '.png', dpi=300)
    plt.clf()
    plt.close()


def sorting_by_col(_A, col):
    """

    :param A: given n by m matrix
    :param col: sorting based on selected column.
    :return:
    """
    A = np.array(_A)
    return A[A[:, col].argsort()]

class graph:
    def __init__(self, X,Y,title=None,axis_label=None, data_label=None, plot_type=None):
        self.X =X
        self.Y =Y
        if title:
            self.title = title
        if axis_label:
            self.axis_label =axis_label
        if data_label:
            self.data_label =data_label
        if plot_type:
            self.plot_type = plot_type



def plot_Graph_v2(graphs,row,path=None, fname=None,):

    col = len(graphs) // row
    if col*row < len(graphs):
        col += 1

    for i in range(len(graphs)):
        graph = graphs[i]
        plt.subplot(313)
        if hasattr(graph,'title'):
            plt.title(graph.title)




def plot_Graph(domains, codomains, row, title=None, path=None, fname=None, isAxis=True, isDot=True, labels=None,
               subtitle=None, legend_loc=4, axis_label=None):
    """
     x1, y1,y2,
     [ X1 ], [  [Y1 Y2 Y3] ] plot Y1~Y3 in single graph.
     [X1, X2], [ [Y1] [ Y2] ] plot X1-Y1 and X2-Y2 in two graphs.
    :param domains: [ x1, x2 , x3 ]
    :param codomains: [ [y1,y2,y3] , [ y1,y2]  ]
    :param labels:  [ t1 ,t2 ,t3 ,t4]
    :param row:
    :param path:
    :param fname:
    :param axis_label: axis_labels for each graph [ [x_label, y_label] [] [] ]
    :return:
    """

    # assert (len(domains) == len(codomains)), 'different dimension between domain and codomain'
    col = len(codomains) // row
    if col*row < len(codomains):
        col += 1


    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    if isDot:
        plot_type = ['bo', 'go', 'ro', 'co', 'mo', 'yo']
    else:
        plot_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-']

    if isAxis:
        axis_flag = 'on'
    else:
        axis_flag = 'off'

    plt.figure(1)
    if title:
        plt.suptitle(title)

    for n in range(len(codomains)):
        legends = []
        print col, row, n
        ax = plt.subplot(col, row, n+1)

        if subtitle:
            plt.title(subtitle[n])

        plt.axis(axis_flag)

        if axis_label:
            plt.xlabel(axis_label[n][0])
            plt.ylabel(axis_label[n][1])

        for k in range(len(codomains[n])):


            if len(domains) == 1:
                # multi-codomain for single domain.
                ax.plot(domains[0], codomains[n][k], plot_type[k % len(colors)],label=labels[k])

            else:
                # each codomain has pair domain.
                if len(codomains[n]) == 1:
                    # one domain for one codomain
                    print 'n: ', n ,' k: ' , k
                    ax.plot(domains[n], codomains[n][k], plot_type[n % len(colors)],label=labels[n])

                else:
                    # one domain for multi-codomain
                    ax.plot(domains[n], codomains[n][k], plot_type[k % len(colors)],label=labels[k])

        ax.legend(loc='best')
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/' + fname + '.png')

    plt.clf()

    plt.close()


def polar_coordiates(pt1, pt2, isPoint):
    """

    :param pt1:
    :param pt2:
    :return: degree value
    """
    if isPoint:
        x = pt2[0] - pt1[0]
        y = pt2[1] - pt1[1]
    else:
        x = pt1
        y = pt2

    pi = np.degrees(np.pi)
    if x > 0 and y >= 0:
        return np.degrees(np.arctan(y / float(x)))
    elif x > 0 and y < 0:
        return np.degrees(np.arctan(y / float(x))) + (pi * 2)
    elif x < 0:
        return np.degrees(np.arctan(y / float(x))) + pi
    elif x == 0 and y > 0:
        return pi / 2
    elif x == 0 and y < 0:
        return pi * 1.5
    else:
        return -1

def gradient(pt1, pt2):
    delta_x = pt1[0] - pt2[0]
    delta_y = pt1[1] - pt2[1]

    m = delta_y / float(delta_x)

    return m


def draw_Gaussian(codomains, row=1, title=None, path=None, fname=None, isAxis=True, labels=None, sub_title=None):
    col = len(codomains) // row
    idx = col * 100 + row * 10
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    if isAxis:
        axis_flag = 'on'
    else:
        axis_flag = 'off'

    plt.figure(1)
    if title:
        plt.suptitle(title)

    for n in range(len(codomains)):
        legends = []
        plt.subplot(idx + n + 1)

        plt.axis(axis_flag)

        count, bins, ignored = plt.hist(codomains[n], len(codomains[n]), normed=True)
        mu = np.mean(codomains[n])
        sigma = math.sqrt(np.var(codomains[n]))
        info = 'mu:' + str(int2round(mu)) + ' sigma:' + str(int2round(sigma))
        if sub_title:
            plt.title(sub_title[n] + '\n' + info)
        else:
            plt.title(info)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
                 color=colors[n])

        if labels:
            legends.append(mpatches.Patch(color=colors[n], label=labels[n]))

        if len(legends) > 0:
            plt.legend(handles=legends, loc=4)
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/' + fname + '.png')
    plt.clf()
    plt.close()


def plot2d_extend(x_axis, x_label, y_axis, y_labels, degree, path, fname):
    """

    :param x_axis:
    :param x_label:
    :param y_axis:
    :param y_labels:
    :param degree: order of linear
    :param path:
    :param fname:
    :return:
    """
    ori = [211, 212]
    color = ['co', 'mo', 'go']

    plt.figure(1)

    min_x = min(x_axis)
    max_x = max(x_axis)
    xp = np.linspace(min_x, max_x, max_x - min_x + 1)

    for i in range(0, len(y_axis)):
        plt.subplot(ori[i])
        plt.plot(x_axis, y_axis[i], color[i])
        model = np.polyfit(x_axis, y_axis[i], degree)
        model = np.poly1d(model)
        plt.plot(xp, model(xp), 'k-')
        plt.xlabel(x_label)
        plt.ylabel(y_labels[i])

    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/' + fname + '.png')

    plt.close(1)


def get_distance_two_points(p1, p2):
    """
    distance between two points
    :param p1:
    :param p2:
    :return:
    """
    inner = pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2)
    return math.sqrt(inner)


def ExpectationLogLikelihood(Z, X, U, V, W):
    """

    :param Z: latent variable (n by k)
    :param X: training data (n by d)
    :param U: bern mean params ( k by d)
    :param V: bern variance (1 by d)
    :param W: weight (1 by k)
    :return:
    """
    similarity = 0
    for n in range(Z.shape[0]):
        for k in range(len(W)):
            domain = Bern_Log_Likelihood_Var(X[n, :], U[k, :], V)
            # domain = pow(math.e, domain)

            try:
                similarity += (Z[n, k] * (math.log(W[k]) + domain))

            except (ValueError, ZeroDivisionError) as e:
                print 'com-log ', e, 'z', Z[n, k], ' w ', W[k], 'likeyhood ', domain
                quit()

    return similarity


def Bern_Log_Likelihood(x, u):
    assert (len(x) == len(u)), 'extrinsic : p(x|u)  data x and model u should have same dimension.'
    likelihood = 0

    for i in range(len(x)):
        if x[i] == 0:  # ln(1-u)
            domain = 1 - u[i]
        else:
            domain = u[i]

        if domain == 0:
            likelihood += math.log(float_info.min)
        elif domain == 1:
            pass
        else:
            likelihood += math.log(domain)

    return likelihood
    # return pow(math.e,likelihood)
    # return bounding(pow(math.e,likelihood))


def Bern_Log_Likelihood_Var(x, u, v):
    """
    don't know why, but 1-float_min = 1.0
    and ln(1) throws zero division error.

    :param x: given data
    :param u: model data
    :param v: var of datas
    :return:
    """

    assert (len(x) == len(u)), 'extrinsic : p(x|u)  data x and model u should have same dimension.'

    likelihood = 0

    for i in range(len(x)):
        if x[i] == 0:  # v * ln(1-u)
            domain = 1 - u[i]
        else:
            domain = u[i]

        if domain == 0:
            likelihood += (v[i] * math.log(float_info.min))
        elif domain == 1:
            pass
        else:
            likelihood += (v[i] * math.log(domain))

    return likelihood


def bounding(val):
    if val > 1:
        quit('exeeeeeeeee one!!!!!!!!!!!!!!!!!!!!!!!!')
    if val == 0:
        return float_info.min
    # elif val==1.0:        return 0.9999
    else:
        return val


def split_into(size, step):
    n = size / step
    q = size % step
    ranges = []
    print size, n, q
    for i in range(n):
        ranges.append((i * step, (i + 1) * step))
    ranges.append((n * step, (n * step) + q))

    print ranges


def rgb2gray_and_binary(raw):
    """
    converts color img to binary image( foreground 1 , background 0)
    :param raw: image list.
    :return:  binary image list
    """

    fg_list = []
    if len(raw[0].shape) == 2:
        for i in range(len(raw)):
            fg_list.append(stats.threshold(raw[i], threshmin=0, threshmax=0, newval=1))
    else:
        for i in range(len(raw)):
            tmp = cvtColor(raw[i], COLOR_RGB2GRAY)
            fg_list.append(stats.threshold(tmp, threshmin=0, threshmax=0, newval=1))
    return fg_list


def Ber_Log_Likelihood_Var_Sum(x, U, v):
    """

    :param x: observed data X
    :param U: shape model, Bern mean model in intrinsic parameter
    :param v: variance parameter in intrinsic parameter
    :return:
    """

    val = 0
    for u in U:
        val += Bern_Log_Likelihood_Var(x, u, v)
    return val


def scatter_test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array([a for a in range(10)])
    y = x.copy()
    z = x.copy()
    ax.scatter(x, y, z)
    plt.show()
    quit()


def sobel(a, axis):
    """
    :param a: given matrix, gray-sacle
    :param axis:  0 for x-axis, 1 for y-axis
    :return:
    """

    assert (len(a.shape) == 2), 'given matrix should be gray-scale.'

    d = np.zeros(a.shape)
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # default x-axis

    if axis == 0:
        kx = np.array(kx)
    else:
        kx = np.array(kx).T

    for y in range(1, a.shape[0] - 1):
        for x in range(1, a.shape[1] - 1):
            aa = a[y - 1:y + 2, x - 1:x + 2]
            d[y, x] = sum(sum(aa * kx))

    return d

def transform_bounding(rect):
    xc = rect[0]
    yc = rect[1]
    w = rect[2]
    h = rect[3]

    ux = xc- (w/2)
    uy = yc-(h/2)
    return (ux,ux+w, uy,uy+h)

def overlapping_area(a, b):
    dx = min(a[1],b[1]) - max(a[0],b[0])
    dy = min(a[3],b[3]) - max(a[2],b[2])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return -1