# -*- coding: utf-8 -*-

from tiah import FileManager as files
from tiah import ImageHandler as images
from tiah import tools as tools
import numpy as np
from os import listdir
import cv2
from scipy import stats
from xml.etree.ElementTree import parse
import matplotlib.pyplot as plt
import sys

class Prepare:
    def __init__(self, bpath, rpath):
        self.bpath = bpath
        self.res_path = rpath
        self.param_path = files.mkdir(self.res_path, 'params')

        # self.test_background_subtractor(self.res_path, 'gist2.avi')
        # self.test_background_subtractor(self.res_path, 'gist2_cut.avi')



    def init_pets(self):

        self.FG1357 = 'fg1357.npy'
        self.FG1359 = 'fg1359.npy'
        self.DP1357 = 'dp1357.npy'
        self.DP1359 = 'dp1359.npy'

        self.GT1357 = 'xml_groundtruth1357.npy'
        self.GT1359 = 'xml_groundtruth1359.npy'
        self.WEIGHT = 'weightmatrix'
        self.DATASET = 'F:/DATASET/6.PETS2009/Crowd_PETS09'
        self.xml1357 = 'PETS2009-S1L1-1'
        self.xml1359 = 'PETS2009-S1L1-2'

        self.set_param(self.xml1357, 1357)
        self.set_param(self.xml1359, 1359)



    def set_param(self, xmlname, flag):
        gt_tree, gt_list = self.parse_xml(xmlname)
        np_gt = np.array(gt_list)
        width = min(np_gt[:, 2])
        heigh = min(np_gt[:, 3])
        y = min(np_gt[:, 1])

        if flag==1357:
            self.param1357 = (width, heigh,y)
        else:
            self.param1359 = (width, heigh,y)

    def test_background_subtractor(self):
        path_time_13_57 = files.mkdir(self.DATASET, 'S1/L1/Time_13-57/View_001')

        org_images = images.read_images_by_path(path_time_13_57)

        thresholds = [10, 50, 100, 150, 200, 266, 300]
        histories = [1, 10, 100, 200, 300, 500, 700]
        rates = [0.1, 0.15, 0.45, 0.6, 0.8, 1.0, 10, 30 , 50]
        labels = []
        models = []
        vpath = files.mkdir(self.res_path,'videos')
        for t in thresholds:
            for h in histories:
                models.append(cv2.BackgroundSubtractorMOG2(history=h, varThreshold=t, bShadowDetection=True))
                a = 'h=' + str(h) + '_t=' + str(t)
                labels.append(a)

        for i in range(len(models)):
            m = models[i]
            l = labels[i]
            res_frame = []

            for r in rates:
                print l, ' r= ', r
                tmp = []

                for frame in org_images:
                    tmp.append(m.apply(image=frame, learningRate=r))
                res_frame.append(tmp)

            tmp2 = []
            for frame in org_images:
                tmp2.append(m.apply(frame))


            k = len(res_frame)
            n = len(res_frame[0])
            print 'k= ', k, 'n= ', n
            dp_frame = []

            for j in range(n):
                upper = np.hstack((res_frame[0][j], res_frame[1][j],res_frame[2][j]))
                lower = np.hstack((res_frame[3][j], res_frame[4][j],res_frame[5][j]))
                last = np.hstack((res_frame[6][j], res_frame[7][j],res_frame[8][j]))
                dpitem = np.vstack((upper, lower,last))
                size = dpitem.shape
                dp_frame.append(cv2.resize(dpitem, tools.int2round((size[1] * 0.7, size[0] * 0.7))))
            images.write_video(dp_frame, 30, vpath, l)
            images.write_video(tmp2,30,vpath,'default_'+l)


        quit('asdfasdfadf')


    def prepare(self):
        if files.isExist(self.param_path, self.FG1357):
            fg1357 = np.load(self.param_path + '/' + self.FG1357)
            fg1359 = np.load(self.param_path + '/' + self.FG1359)
            dp1357 = np.load(self.param_path + '/' + self.DP1357)
            dp1359 = np.load(self.param_path + '/' + self.DP1359)

            gt1357 = np.load(self.param_path + '/' + self.GT1357)
            gt1359 = np.load(self.param_path + '/' + self.GT1359)

        else:
            path_time_13_57 = files.mkdir(self.DATASET, 'S1/L1/Time_13-57/View_001')
            path_time_13_59 = files.mkdir(self.DATASET, 'S1/L1/Time_13-59/View_001')
            dilation_flag = 1
            data1357 = self.create_parameters(path_time_13_57,dilation_flag)
            data1359 = self.create_parameters(path_time_13_59,dilation_flag)
            # each data contains [fg set, gray-images, color-images, gray-images of fg set.]

            fg1357 = np.array(data1357[0])
            dp1357 = np.array(data1357[1])
            fg1359 = np.array(data1359[0])
            dp1359 = np.array(data1359[1])

            images.write_video(dp1357[0], 20, self.param_path, 's1l1_1357_dpcolor')
            images.write_video(dp1357[1], 20, self.param_path, 's1l1_1357_dpmask')
            images.write_video(dp1359[0], 20, self.param_path, 's1l1_1359_dpcolor')
            images.write_video(dp1359[1], 20, self.param_path, 's1l1_1359_dpmask')
            np.save(self.param_path + '/' + self.FG1357, fg1357)
            np.save(self.param_path + '/' + self.DP1357, dp1357)
            np.save(self.param_path + '/' + self.FG1359, fg1359)
            np.save(self.param_path + '/' + self.DP1359, dp1359)


            a, gt1357= self.create_count_groundtruth(fg1357[0],self.xml1357,self.GT1357,1357)
            b, gt1359 =self.create_count_groundtruth(fg1359[0],self.xml1359, self.GT1359,1359)


        if files.isExist(self.param_path, self.WEIGHT):
            weight = np.load(self.param_path + '/' + self.WEIGHT)
        else:
            shape = fg1357[0][0].shape
            weight = self.create_weight_matrix(shape)
            np.save(self.param_path + '/' + self.WEIGHT, weight)

        # self.draw_weight_map(weight)


        return fg1357, dp1357, fg1359, dp1359, weight, gt1357,gt1359

    def create_weight_matrix(self, shape):

        gt_tree, gt_list = self.parse_xml(self.xml1357)
        domain = []
        codomain = []
        for gt in gt_list:  # gt [ cx,cy,w,h]
            domain.append(gt[0])
            codomain.append(gt[3])

        height_model = np.poly1d(np.polyfit(domain, codomain, 1))
        W = []
        base = height_model(shape[1])
        for x in range(shape[1]):
            w = height_model(x)
            W.append(base / w)

        return np.array(W)


    def create_parameters(self, selected_path, flag_dilation=0):
        """
        Learns background model from data S0 and creates foreground mask of selected data S1.
        This returns four parameters which are
        foreground mask set, gray images, color images, visualized images of foreground mask.

        :param selected_path:
        :return:
        """

        path_s0 = files.mkdir(self.DATASET, 'S0/Background/View_001/')
        path_s0_dir = listdir(str(path_s0))

        background_train_set = []
        for selected_dir in path_s0_dir[2:3]:
            background_train_set += images.read_images_by_path(path_s0 + '/' + selected_dir, False)

        dp_color = images.read_images_by_path(selected_path)
        dp_gray = images.read_images_by_path(selected_path, False)

        background_model = cv2.BackgroundSubtractorMOG2(len(background_train_set), varThreshold=266,
                                                        bShadowDetection=True)
        for frame in background_train_set:
            background_model.apply(frame, len(background_train_set))  # given frame, learning-rate

        th = 100
        fgmask_set = []
        dp_mask = []
        kernel_size= 1
        iteration = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for frame in dp_gray:
            forward = background_model.apply(frame)  # create foreground mask which is gray-scale(0~255) image.
            tmp = cv2.cvtColor(forward, cv2.COLOR_GRAY2BGR)  # convert to color
            if flag_dilation:
                dp_mask.append(cv2.dilate(tmp,kernel,iterations=iteration))
            else:
                dp_mask.append(tmp)

            # convert gray-scale foreground mask to binary image.
            a = stats.threshold(forward, threshmin=th, threshmax=255, newval=0)
            a = stats.threshold(a, threshmin=0, threshmax=th, newval=1)
            fgmask_set.append(a)

        return (fgmask_set, dp_gray), (dp_color, dp_mask)



    def parse_xml(self,gtname):
        """
        PETS groundtruth는 center x ,center y, width ,height인데 parse them into bound pixels e.g., top_left , bottom_right.
        :param bpath: path for xml file
        :param xml_name:
        :return: [  frame1[ gt1, gt2, gt2...]  , frame2[ gt1, gt2....   ]    ]
        """
        xml_name = gtname
        tree = parse(self.bpath + '/' + xml_name + '.xml')
        note = tree.getroot()

        frames = note.findall('frame')
        gt_tree = []
        gt_list = []
        for f in frames:  # frame node
            obj_list = f.find('objectlist')  # object list of given frame
            obj = obj_list.findall('object')  # each object
            gt_sub = []
            for o in obj:
                b = o.find('box')
                bb = dict(b.items())
                cx = float(bb['xc'])
                cy = float(bb['yc'])
                w = float(bb['w'])
                h = float(bb['h'])

                # half_w = int2round(w/2)
                # half_h = int2round(h/2)
                #
                # ux = cx - half_w
                # uy = cy - half_h
                # gt_sub.append ( int2round(ux,uy, ux+half_w , uy+half_h) )
                gt_sub.append((cx, cy, w, h))
                gt_list.append((cx, cy, w, h))

            gt_tree.append(gt_sub)

        print 'ground truth length is ', len(gt_tree)
        return gt_tree, gt_list

    def draw_weight_map(self, perspective):
        Y = perspective
        X = np.linspace(0, len(Y), len(Y))

        plt.plot(X, Y, 'b-')
        plt.xlabel('row')
        plt.ylabel('weight')
        plt.show()
        plt.title('Weight Distribution')
        quit()

    def create_count_groundtruth(self, fgset, xmlname, fname, flag):

        segmentation_tree = []
        if flag==1357:
            param = self.param1357
        else:
            param = self.param1359

        for fg in fgset:
            list_rect, list_contours = self.segmentation_blob(fg, param)
            segmentation_tree.append(list_rect)

        gt_tree, gt_list = self.parse_xml(xmlname)


        dp_counting_gt_tree = []
        counting_gt_tree = []
        for i in range(len(segmentation_tree)):
            frame_seg = segmentation_tree[i]
            frame_gt = gt_tree[i]
            frame_counting_gt = [[] for x in range(len(frame_seg))]

            for gt in frame_gt:

                pre_ratio = sys.float_info.min
                selected_gt = None
                selected_idx = -1
                for j in range(len(frame_seg)):
                    seg = frame_seg[j]
                    gt_rect_form = tools.transform_bounding(gt)

                    overlapped_area = tools.overlapping_area(gt_rect_form, seg)
                    if overlapped_area != -1:
                        overlapped_ratio = overlapped_area / (gt[2] * gt[3])  # w*h
                        if pre_ratio < overlapped_ratio:
                            selected_gt = tools.int2round(gt_rect_form)
                            pre_ratio = overlapped_ratio
                            selected_idx = j

                if selected_idx != -1:
                    frame_counting_gt[selected_idx].append(selected_gt)

            dp_counting_gt_tree.append(frame_counting_gt)
            tmp = []
            for a in frame_counting_gt:
                tmp.append(len(a))
            counting_gt_tree.append(np.array(tmp))

        np.save(self.param_path + '/' + fname, np.array(counting_gt_tree))
        print fname +' size is ' , len(counting_gt_tree)
        return dp_counting_gt_tree, np.array(counting_gt_tree)

    def segmentation_blob(self, given_fg, param):
        """
        Find segmenations in given frame.
        :param fg:
        :return: list of rectangle(x1,x2,y1,y2)
        """
        list_rect = []
        list_contours = []

        minw = param[0]
        minh = param[1]
        miny = param[2]

        # contours, heiarchy = cv2.findContours(given_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, heiarchy = cv2.findContours(given_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # check here

        if len(contours) != 0:  # if contour exists
            for i in range((len(contours))):
                if len(contours[i]) > 100:  # thresholding n_point of contours
                    rect = cv2.boundingRect(contours[i])  # find fitted_rectangle to given contour.

                    w = rect[2]
                    h = rect[3]
                    ux = rect[0]
                    uy = rect[1]
                    # segment should be larger than the minimum size of object in groundtruth.
                    # and object should be located in inside of ROI.
                    if w > minw and h > minh and uy + h > miny:
                        reform = (ux, ux + w, uy, uy + h)
                        list_rect.append(reform)
                        list_contours.append(contours[i])

        return list_rect, list_contours