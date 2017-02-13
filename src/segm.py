import cv2
from tiah import ImageHandler as images
from tiah import tools as tools
from tiah import FileManager as files
import numpy as np


def testing(fg1357, dp1357, mask1357, param, rpath):
    filtering(fg1357, dp1357, mask1357, param,rpath)


def filtering(fg1357, dp1357, mask1357, param,rpath):
    minw = param[0]
    minh = param[1]
    miny = param[2]
    print 'min w ,h :', minw, minh

    x_axis = 8
    y_axis = 27
    x,y = 0,0
    cx,cy = 8,27
    ellipse = np.zeros((54,16),dtype=np.uint8)
    for row in range(ellipse.shape[1]):
        for col in range(ellipse.shape[0]):
            ellipse[col,row] = check_ellipse(x_axis,y_axis,cx,cy,row,col)

    filtered_fgdp = []
    for fg, dp, mask in zip(fg1357, dp1357, mask1357):
        y_min = tools.int2round(param[2])

        fg[0: y_min, :] = 0 # remove background building
        mask[0:y_min, :] = 0

        fgdp = fg.copy() # for display

        marking = np.zeros(fgdp.shape)

        for i in range(0, fg.shape[1]-16):
            for j in range(y_min , fg.shape[0]-54):
                roi =  fgdp[j:j+54, i:i+16]
                roi_and = roi*ellipse
                fg_ratio = np.count_nonzero(roi_and)/float(np.count_nonzero(ellipse))

                if fg_ratio  > 0.6:
                    marking[j:j+54, i:i+16] = roi.copy()
                    # tmp_dp = mask.copy()
                    # cv2.rectangle(tmp_dp, (i,j), (i+16,j+54), tools.red, 2)
                    # msg = str(np.count_nonzero(roi_and))+'/'+str((np.count_nonzero(ellipse)))+'='+str(round(fg_ratio,2))
                    # cv2.putText(tmp_dp, str(msg), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tools.green, 2)
                    # cv2.imshow('123',tmp_dp)
                    # cv2.imshow('456',marking*255)
                    # cv2.waitKey(1)

        filtered_fgdp.append(marking*255)

    print rpath
    tpath = files.mkdir(rpath,'filtering_fgmask 0.6')
    images.write_imgs(filtered_fgdp,tpath,'filtered_fg')






def check_ellipse(x_axis,y_axis,cx,cy,x,y):

    term1 = np.square(x-cx) / np.square(x_axis)
    term2 = np.square(y-cy) / np.square(y_axis)
    # print 'x: ' ,x, '  y:', y, ' np.square(x-x_axis):' , np.square(x-x_axis), '  np.square(x_axis): ',np.square(x_axis)

    if term1+term2 <= 1:
        return 1
    else:
        return 0


def manual_case(fg1357, dp1357, mask1357, param):
    segindex = 0
    minw = param[0]
    minh = param[1]
    miny = param[2]
    print 'min w ,h :', minw, minh

    for fg, dp, mask in zip(fg1357, dp1357, mask1357):
        fg[0:param[2], :] = 0
        mask[0:param[2], :] = 0
        rects, countours, not_rects = segmentation_blob(fg, param)
        board = dp.copy()
        # dp[:,:,:]= 0

        fg2 = fg.copy()

        # kernel = np.zeros(tools.int2round((minh, minw)),dtype=np.uint8)
        kernel = np.ones(tools.int2round((minh / 2, minw / 2)), dtype=np.uint8)
        # kernel[:,5:9] = 1

        cv2.drawContours(mask, countours, -1, tools.magenta, 2)

        for s, ns in zip(rects, not_rects):
            cv2.rectangle(mask, (s[0], s[2]), (s[1], s[3]), tools.orange, 2)
            cv2.rectangle(mask, (ns[0], ns[2]), (ns[1], ns[3]), tools.red, 2)

            cv2.putText(mask, str(segindex), (s[0], s[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tools.green, 2)
            segindex += 1
            roi = fg2[s[2]:s[3] + 1, s[0]:s[1] + 1]
            print 'roi shape ', roi.shape
            roi = cv2.erode(roi, kernel, iterations=1)

            fg2[s[2]:s[3] + 1, s[0]:s[1] + 1] = roi

        # cv2.drawContours(dp,countours,-1,tools.green,2)
        # cv2.drawContours(board,countours,-1,tools.green,-1)
        # for s in rects:
        #     cv2.rectangle(dp,(s[0], s[2]), (s[1], s[3]), tools.orange, 1)
        # cv2.imshow('1',np.hstack((dp,mask,board)))
        fg2 *= 255
        # cv2.imshow('1',np.hstack((fg*255,fg2)))
        cv2.imshow('1', mask)
        cv2.waitKey(0)


def segmentation_blob(given_fg, param):
    """
    Find segmenations in given frame.
    :param fg:
    :return: list of rectangle(x1,x2,y1,y2)
    """
    list_rect = []
    list_contours = []
    list_non_rect = []

    minw = param[0]
    minh = param[1]
    miny = param[2]

    # contours, heiarchy = cv2.findContours(given_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, heiarchy = cv2.findContours(given_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # check here

    if len(contours) != 0:  # if contour exists
        for i in range((len(contours))):
            rect = cv2.boundingRect(contours[i])  # find fitted_rectangle to given contour.
            w = rect[2]
            h = rect[3]
            ux = rect[0]
            uy = rect[1]
            # segment should be larger than the minimum size of object in groundtruth.
            # and object should be located in inside of ROI.
            reform = (ux, ux + w, uy, uy + h)
            if len(contours[i]) > 100:  # thresholding n_point of contours

                if w > minw and h > minh and uy + (minh) > miny:
                    list_rect.append(reform)
                    list_contours.append(contours[i])
            elif len(contours[i]) < 30:
                list_non_rect.append(reform)

    return list_rect, list_contours, list_non_rect
