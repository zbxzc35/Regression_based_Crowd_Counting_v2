import cv2
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from tiah import tools
from scipy import stats
from os import listdir
from platform import system

def split_video(src, video_props, start, end, path, fname):
    fps = video_props[1]
    res = src[start * fps:end * fps]
    write_video(res, video_props, path, fname)


def write_video(src, fps, path, fname):
    """
    :param src:
    :param video_props: fourcc , fps , size
    :param path:
    :param fname:
    :return:
    """

    print 'writing vidoe file ', len(src)
    if src is None:
        print 'tools , write video : no list'
        return None
    elif len(src) == 0:
        print 'tools , write video : empty list'
        return None
    else:
        size = src[0].shape
        fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        writer = cv2.VideoWriter(path + '/' + fname + '.avi', fourcc, fps, (size[1], size[0]))

        if  len(size) == 2:
            for img in src:
                writer.write(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
        else:
            for img in src:
                writer.write(img)

        writer.release()


def write_imgs(src, path, fname):
    if isinstance(src, list) or type(src)==type(np.array(1)):
        n = len(str(len(src)))
        for i in range(len(src)):
            cv2.imwrite(path + "/" + fname + str(i+1).zfill(n) + '.png', src[i])
    else:
        cv2.imwrite(path + "/" + fname+'.jpg', src)

def read_file_list(path,sort=True):
    flist = listdir(path)
    if sort :
        return natsorted(flist)
    else:
        return flist

def read_images(path,flist,color=True):
    images= []
    if color:
        for f in flist:
            images.append(cv2.imread(path+'/'+f, flags=cv2.CV_LOAD_IMAGE_COLOR))
    else :
        for f in flist:
            images.append(cv2.imread(path + '/' + f, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE))
    return images

def read_images_by_path(path , flag_color = True):
    print 'reading images from ',path , flag_color
    file_list = listdir(path)
    file_list = natsorted(file_list)
    img_list = []

    if flag_color:
        for file in file_list:
            img_list.append(cv2.imread(path+'/'+file , flags=cv2.CV_LOAD_IMAGE_COLOR))
    else:
        for file in file_list:
            img_list.append(cv2.imread(path + '/' + file, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE))
    return img_list


def read_video_as_list_by_path(path, fname):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """
    cap = cv2.VideoCapture()
    cap.open(path + '/' + fname)


    if not cap.isOpened():
        print 'tools, read video as list by path : file not exist'
        return None

    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    fourcc = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    f = []
    while 1:
        ret, frame = cap.read()

        if ret is False:
            break
        f.append(frame)

    if len(f) is 0:
        print 'tools, read video as list by path : no frames in video'
        return None

    cap.release()
    return f, tools.int2round(fps)


def get_list_draw_every_ellipse(src, ellipse):
    """
    drawing ellipses for each image.

    :param src: image list
    :param ellipse: ellipse(center,size,orientation) list
    :return: image list
    """
    show_list = []

    idx = 0
    for i in range(len(src)):
        img = src[i].copy()
        if len(ellipse[i]) != 0:

            for j in range(len(ellipse[i])):  # draw all contours and line corresponding to contour.
                cv2.ellipse(img, ellipse[i][j], (0, 255, 0), 1)
                ellipse_center = tools.int2round(ellipse[i][j][0])
                cv2.putText(img, str(idx) , ellipse_center,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,255,255))
                idx +=1
            show_list.append(img)

    return show_list


def display_img(input_list, waitSec=30):
    for img in input_list:
        cv2.imshow('1', img)
        if cv2.waitKey(waitSec) & 0xff == 113:  # 113 refers to 'q'
            break
    cv2.destroyAllWindows()


def get_list_fgmask(src, history=30, nmixtures=10, ratio=0.1):
    """
    background subtraction.
    returns binary image
    """
    print 'masking foreground'
    fgbg = cv2.BackgroundSubtractorMOG(history=history, nmixtures=nmixtures, backgroundRatio=ratio)
    fgmask_list = []
    dp_list = []
    for frame in src:
        fgmask = fgbg.apply(frame)
        dp_list.append(fgmask)
        fgmask_list.append(stats.threshold(fgmask, threshmin=0,threshmax=0, newval=1))

    return fgmask_list,dp_list
