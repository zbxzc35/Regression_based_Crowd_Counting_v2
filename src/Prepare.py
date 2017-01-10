# -*- coding: utf-8 -*-

from tiah import FileManager as files
from tiah import ImageHandler as images
import numpy as np
from os import listdir
import cv2
from scipy import stats
from xml.etree.ElementTree import parse
import matplotlib.pyplot as plt

class Prepare:
    def __init__(self, rpath):
        self.res_path = rpath
        self.param_path = files.mkdir(self.res_path, 'params')

        self.FG1357 = 'fg1357.npy'
        self.FG1359 = 'fg1359.npy'
        self.DP1357 ='dp1357.npy'
        self.DP1359 ='dp1359.npy'

        self.FG2GIST ='fg_gist2.npy'
        self.DP2GIST ='dp_gist2.npy'

        self.GT = 'groundtruth.npy'
        self.WEIGHT = 'weightmatrix'
        self.DATASET = 'F:/DATASET/6.PETS2009/Crowd_PETS09'

        gt_tree, gt_list = self.parse_xml()
        np_gt = np.array(gt_list)
        self.width = min(np_gt[:, 2])
        self.height = min(np_gt[:, 3])
        self.y = min(np_gt[:, 1])



    def prepare_gist(self):
        if files.isExist(self.param_path, self.FG2GIST):
            fgset = np.load(self.param_path + '/' + self.FG2GIST)
            dpset = np.load(self.param_path + '/' + self.DP2GIST)
            dp_color = dpset[0]
            dp_mask = dpset[1]

        else:
            fgset, dp_color, dp_mask = self.create_gist_parameters(self.res_path,'gist2.avi')
            # each data contains [fg set, gray-images, color-images, gray-images of fg set.]

            np.save(self.param_path + '/' + self.FG2GIST, fgset)
            np.save(self.param_path + '/' + self.DP2GIST, np.array([dp_color,dp_mask]))



        return fgset,dp_color,dp_mask


    def prepare(self):
        if files.isExist(self.param_path, self.FG1357):
            fg1357 = np.load(self.param_path+'/'+self.FG1357)
            fg1359 = np.load(self.param_path+'/'+self.FG1359)
            dp1357 = np.load(self.param_path+'/'+self.DP1357)
            dp1359 = np.load(self.param_path+'/'+self.DP1359)

        else:
            path_time_13_57 = files.mkdir( self.DATASET, 'S1/L1/Time_13-57/View_001')
            path_time_13_59 = files.mkdir( self.DATASET, 'S1/L1/Time_13-59/View_001')
            data1357 = self.create_parameters(path_time_13_57)
            data1359 = self.create_parameters(path_time_13_59)
            #each data contains [fg set, gray-images, color-images, gray-images of fg set.]

            fg1357 = data1357[0]
            dp1357 = data1357[1]
            fg1359 = data1359[0]
            dp1359 = data1359[1]

            np.save(self.param_path+'/'+self.FG1357,fg1357)
            np.save(self.param_path+'/'+self.DP1357,dp1357)
            np.save(self.param_path+'/'+self.FG1359,fg1359)
            np.save(self.param_path+'/'+self.DP1359,dp1359)


        if files.isExist(self.param_path,self.WEIGHT):
            weight = np.load(self.param_path+'/'+self.WEIGHT)
        else:
            shape = fg1357[0][0].shape
            weight = self.create_weight_matrix(shape)
            np.save(self.param_path+'/'+self.WEIGHT,weight)

        # self.draw_weight_map(weight)

        return fg1357,dp1357,fg1359,dp1359,weight





    def create_weight_matrix(self, shape):

        gt_tree, gt_list = self.parse_xml()
        domain = []
        codomain = []
        for gt in gt_list:  # gt [ cx,cy,w,h]
            domain.append(gt[0])
            codomain.append(gt[3])

        height_model = np.poly1d(np.polyfit(domain,codomain,1))
        W = []
        base = height_model(shape[1])
        for x in range(shape[1]):
            w = height_model(x)
            W.append(base/w)

        return np.array(W)



    def min_width_height(self):

        return self.width, self.height, self.y


    def create_gist_parameters(self,selected_path,video_name):

        org_images = images.read_video_as_list_by_path(selected_path,video_name)
        background_model = cv2.BackgroundSubtractorMOG2(len(org_images), varThreshold=266, bShadowDetection=True)
        for frame in org_images:
            background_model.apply(frame, len(org_images))  # given frame, learning-rate

        th = 150
        fgmask_set = []
        dp_mask = []
        for frame in org_images:
            forward = background_model.apply(frame) # create foreground mask which is gray-scale(0~255) image.
            tmp = cv2.cvtColor(forward, cv2.COLOR_GRAY2BGR) # convert to color
            dp_mask.append(tmp)
            #convert gray-scale foreground mask to binary image.
            a = stats.threshold(forward, threshmin=th, threshmax=255, newval=0)
            a = stats.threshold(a, threshmin=0, threshmax=th, newval=1)
            fgmask_set.append(a)

        return fgmask_set, org_images, dp_mask

    def create_parameters(self, selected_path):
        """
        Learns background model from data S0 and creates foreground mask of selected data S1.
        This returns four parameters which are
        foreground mask set, gray images, color images, visualized images of foreground mask.

        :param selected_path:
        :return:
        """

        path_s0 = files.mkdir(self.DATASET, 'S0/Background/View_001/')
        path_s0_dir = listdir(str(path_s0))

        background_train_set= []
        for selected_dir in path_s0_dir[2:3]:
            background_train_set += images.read_images_by_path(path_s0+'/'+selected_dir, False)


        dp_color = images.read_images_by_path(selected_path)
        dp_gray = images.read_images_by_path(selected_path,False)



        background_model = cv2.BackgroundSubtractorMOG2(len(background_train_set), varThreshold=266, bShadowDetection=True)
        for frame in background_train_set:
            background_model.apply(frame, len(background_train_set))  # given frame, learning-rate

        th = 150
        fgmask_set = []
        dp_mask = []
        for frame in dp_gray:
            forward = background_model.apply(frame) # create foreground mask which is gray-scale(0~255) image.
            tmp = cv2.cvtColor(forward, cv2.COLOR_GRAY2BGR) # convert to color
            dp_mask.append(tmp)
            #convert gray-scale foreground mask to binary image.
            a = stats.threshold(forward, threshmin=th, threshmax=255, newval=0)
            a = stats.threshold(a, threshmin=0, threshmax=th, newval=1)
            fgmask_set.append(a)

        return (fgmask_set,dp_gray), (dp_color, dp_mask)



    def parse_xml(self):
        """
        PETS groundtruth는 center x ,center y, width ,height인데 parse them into bound pixels e.g., top_left , bottom_right.
        :param bpath: path for xml file
        :param xml_name:
        :return: [  frame1[ gt1, gt2, gt2...]  , frame2[ gt1, gt2....   ]    ]
        """
        xml_name = 'PETS2009-S1L1-1'
        tree = parse(self.res_path + '/' + xml_name + '.xml')
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
        np.save(self.param_path + '/'+self.GT, np.array(gt_tree))
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