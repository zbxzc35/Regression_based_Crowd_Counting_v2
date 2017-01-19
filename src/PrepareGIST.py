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

class PrepareGIST:
    def __init__(self, rpath):
        self.res_path = rpath
        self.param_path = files.mkdir(self.res_path, 'params')
        # self.cut_front_gist()

        # self.test_background_subtractor(self.res_path, 'gist2.avi')
        # self.test_background_subtractor(self.res_path, 'gist2_cut.avi')

    def init_gist(self,src):
        self.src = src+'.avi'
        self.FG2GIST = 'fg_'+src+'.npy'
        self.DP2GIST = 'dp_'+src+'.npy'

        # self.FG2GIST = 'fg_gist2_2016_uncut.npy'
        # self.DP2GIST = 'dp_gist2_2016_uncut.npy'

        # self.FG2GIST = 'fg_gist2_2016_cut.npy'
        # self.DP2GIST = 'dp_gist2_2016_cut.npy'

        self.GT = 'groundtruth.npy'
        self.WEIGHT = 'weightmatrix'
        self.DATASET = 'F:/DATASET/6.PETS2009/Crowd_PETS09'



    def cut_front_gist(self):
        _org_images, fps = images.read_video_as_list_by_path(self.res_path, None)
        images.write_video(_org_images[50:],fps,self.res_path,'gist2_cut')



    def test_background_subtractor(self, selected_path, video_name):
        _org_images, fps = images.read_video_as_list_by_path(selected_path, video_name)
        org_images = _org_images

        thresholds = [10, 50, 100, 150, 200, 266, 300]
        histories = [1, 10, 100, 200, 300, 500, 700]
        rates = [0.1, 0.15, 0.45, 0.6, 0.8, 1.0]
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
                dpitem = np.vstack((upper, lower))
                size = dpitem.shape
                dp_frame.append(cv2.resize(dpitem, tools.int2round((size[1] * 0.7, size[0] * 0.7))))
            images.write_video(dp_frame, 30, vpath, l)
            images.write_video(tmp2,30,vpath,'default_'+l)


        quit('asdfasdfadf')

    def prepare_gist(self):
        if files.isExist(self.param_path, self.FG2GIST):
            fgset = np.load(self.param_path + '/' + self.FG2GIST)
            dpset = np.load(self.param_path + '/' + self.DP2GIST)
            dp_color = dpset[0]
            dp_mask = dpset[1]

        else:
            fgset, dp_color, dp_mask = self.create_gist_parameters(self.res_path, self.src)
            # each data contains [fg set, gray-images, color-images, gray-images of fg set.]

            np.save(self.param_path + '/' + self.FG2GIST, fgset)
            np.save(self.param_path + '/' + self.DP2GIST, np.array([dp_color, dp_mask]))

        return fgset, dp_color, dp_mask


    def create_gist_parameters(self, selected_path, video_name):

        org_images, fps = images.read_video_as_list_by_path(selected_path, video_name)
        background_model = cv2.BackgroundSubtractorMOG2(history=300,varThreshold=100,bShadowDetection=True)

        th = 150
        fgmask_set = []
        dp_mask = []
        for frame in org_images:
            forward = background_model.apply(frame)  # create foreground mask which is gray-scale(0~255) image.
            tmp = cv2.cvtColor(forward, cv2.COLOR_GRAY2BGR)  # convert to color
            dp_mask.append(tmp)
            # convert gray-scale foreground mask to binary image.
            a = stats.threshold(forward, threshmin=th, threshmax=255, newval=0)
            a = stats.threshold(a, threshmin=0, threshmax=th, newval=1)
            fgmask_set.append(a)

        return fgmask_set, org_images, dp_mask


    def draw_weight_map(self, perspective):
        Y = perspective
        X = np.linspace(0, len(Y), len(Y))

        plt.plot(X, Y, 'b-')
        plt.xlabel('row')
        plt.ylabel('weight')
        plt.show()
        plt.title('Weight Distribution')
        quit()

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
        np.save(self.param_path + '/' + self.GT, np.array(gt_tree))
        return gt_tree, gt_list






