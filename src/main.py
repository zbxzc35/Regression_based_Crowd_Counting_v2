from tiah import FileManager as files
from tiah import tools as tools
from src.Prepare import Prepare
from src.PrepareGIST import PrepareGIST
from os import getcwd, chdir
import cv2, pyGPs, pickle
# from src.Direct_Feature import run_SURF_v4,run_FAST_v4
import src.Direct_Feature as directs
import src.Indirect_Feature as indirects
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tiah.tools import graph
from src.others import knr
from sklearn.decomposition import PCA
from tiah import ImageHandler as images


class worker:
    def __init__(self):
        self.run_pets_case()
        # self.run_gist_case()

    def run_pets_case(self):
        chdir('..')

        self.feature_version = 4
        self.dir_version = 1 # directory differs parameter



        self.bpath = files.mkdir(getcwd(), 'S1L1')
        self.res_path = files.mkdir(self.bpath, 'res' + str(self.dir_version))
        self.param_path = files.mkdir(self.res_path, 'params')
        self.graph_path = files.mkdir(self.res_path, 'graphs')
        self.model_path = files.mkdir(self.res_path, 'models')
        self.prepare = Prepare(self.bpath, self.res_path)
        self.prepare.init_pets()
        # prepare.test_background_subtractor()
        self.FEATURE1357 = 'featureset1357.npy'
        self.FEATURE1359 = 'featureset1359.npy'
        self.COUNTGT1357 = 'c_groundtruth1357.npy'
        self.COUNTGT1359 = 'c_groundtruth1359.npy'

        a1357, b1357, a1359, b1359, weight, gt1357, gt1359 = self.prepare.prepare()
        param1357 = self.prepare.param1357
        param1359 = self.prepare.param1359
        fg1357 = a1357[0]
        dpcolor1357 = b1357[0]
        dpmask1357 = b1357[1]

        fg1359 = a1359[0]
        dpcolor1359 = b1359[0]
        dpmask1359 = b1359[1]
        # v2: only K
        # v3: only K E T
        # v3: K E T P S S2
        # a= gt1357[1:gt1357.shape[0]-1]
        # b= gt1359[1:gt1359.shape[0]-1]
        # print 'a: ' , a.shape
        # print 'b: ' , b.shape
        # np.save(self.param_path+'/'+self.COUNTGT1357,a)
        # np.save(self.param_path+'/'+self.COUNTGT1359,b)
        # quit()



        self.create_feature_set(fg1357, dpcolor1357, weight, self.feature_version, self.FEATURE1357, param1357, gt1357,
                                self.COUNTGT1357)
        self.create_feature_set(fg1359, dpcolor1359, weight, self.feature_version, self.FEATURE1359, param1359, gt1359,
                                self.COUNTGT1359)
        features1357 = np.load(self.param_path + '/v' + str(self.feature_version) + '_' + self.FEATURE1357)
        features1359 = np.load(self.param_path + '/v' + str(self.feature_version) + '_' + self.FEATURE1359)
        groundtruth1357 = np.load(self.param_path + '/' + self.COUNTGT1357)
        groundtruth1359 = np.load(self.param_path + '/' + self.COUNTGT1359)

        print 'data1357: ', fg1357.shape, dpcolor1357.shape, len(dpmask1357)
        print 'data1359: ', fg1359.shape, dpcolor1359.shape, len(dpmask1359)
        print 'feature 1357: ', features1357.shape, ' 1359: ', features1359.shape
        # K,E,T,P,S,S2
        length = len(dpcolor1359)

        self.dowork(features1357, features1359, dpcolor1359[1:length - 1], fg1359[1:length - 1],
                    groundtruth1357, groundtruth1359)

    def run_gist_case(self):
        chdir('..')
        self.bpath = files.mkdir(getcwd(), 'gist')
        self.res_path = files.mkdir(self.bpath, 'res')
        self.param_path = files.mkdir(self.res_path, 'params')
        self.graph_path = files.mkdir(self.res_path, 'graphs')
        self.model_path = files.mkdir(self.res_path, 'models')
        prepare = PrepareGIST(self.res_path)

        src = 'gist2_2016_cut'
        # src = 'gist2_2015'
        # src = 'gist2_2015'
        prepare.init_gist(src)
        self.FEATURE = 'featureset.npy'
        self.GT = 'groundtruth.npy'

        segmentation_tree = []

        fgset, dp_color, dp_mask = prepare.prepare_gist()
        # images.write_video(dp_mask, 30, self.res_path, src + '_fgmask')
        # self.test_drawing_segmentation(fgset, dp_color, src)

        # images.display_img(dp_mask)
        # images.display_img(dp_color)

    def gt_test(self, prepare, fg1357, dp1357, dpmask1357):

        seg_tree = []
        for fg in fg1357:
            list_rect, list_contours = self.segmentation_blob(fg, prepare.min_width_height())
            seg_tree.append(list_rect)
        dp_count_tree, count_tree = prepare.create_count_groundtruth(seg_tree)

        for frame_count, frame_seg, dpcolor, dpmask in zip(dp_count_tree, seg_tree, dp1357, dpmask1357):
            for i in range(len(frame_seg)):
                s = frame_seg[i]
                cv2.rectangle(dpcolor, (s[0], s[2]), (s[1], s[3]), tools.green, 1)
                cv2.rectangle(dpmask, (s[0], s[2]), (s[1], s[3]), tools.green, 1)

                abc = frame_count[i]
                for gt in abc:
                    cv2.rectangle(dpcolor, (gt[0], gt[2]), (gt[1], gt[3]), tools.red, 1)
                    cv2.rectangle(dpmask, (gt[0], gt[2]), (gt[1], gt[3]), tools.red, 1)

                cv2.imshow('1', np.hstack((dpcolor, dpmask)))
                cv2.waitKey(0)

    def test_drawing_segmentation(self, fgset, dp_color, fname):

        a = []
        for f in fgset:
            frame_rect, frame_contour = self.segmentation_blob(f, [10, 10, 0])
            a.append(frame_rect)

        b = self.draw_segmentation(dp_color, a)

        images.write_video(b, 30, self.res_path, fname + '_segmented')

    def draw_segmentation(self, frame_set, seg_set):

        results = []
        for i in range(len(frame_set)):
            frame = frame_set[i].copy()
            frame_seg = seg_set[i]

            for s in frame_seg:
                cv2.rectangle(frame, (s[0], s[2]), (s[1], s[3]), tools.green, 1)
            results.append(frame)

        return results

    def test3d(self, X, Y, Z):

        plot_type = ['b', 'g', 'r', 'c', 'm', 'y']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, c='b', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def pca_test(self, features, version, label):
        groundtruth = np.load(self.param_path + '/v' + str(version) + '_' + self.GT)

        _trainX = np.concatenate(features[0:features.shape[0]:2])
        _trainY = np.concatenate(groundtruth[0:groundtruth.size:2])
        pca = PCA(n_components=2)
        X = pca.fit_transform(_trainX)
        self.test3d(X[:, 0], X[:, 1], _trainY)

    def dowork(self, features1357, features1359, dpcolors, fgset, groundtruth1357, groundtruth1359):
        labels = ['K', 'E', 'T', 'P', 'S', 'S2']
        for i in range(len(labels)):
            train_feature = features1357[i]
            test_feature = features1359[i]

            # _trainX = np.concatenate(train_feature)
            # _trainY = np.concatenate(groundtruth1357)
            # testX = test_feature
            # testY = groundtruth1359

            # _trainX = np.concatenate(train_feature[i][0:train_feature.shape[0]:2])
            # _trainY = np.concatenate(groundtruth[0:groundtruth.size:2])
            # testX = test_feature[1:test_feature.shape[0]:2]
            # testY = groundtruth[1:groundtruth.size:2]

            _trainX = np.concatenate(train_feature[0:train_feature.shape[0]:2])
            _trainY = np.concatenate(groundtruth1357[0:groundtruth1357.size:2])
            testX = train_feature[1:train_feature.shape[0]:2]
            testY = groundtruth1357[1:groundtruth1357.size:2]

            print '_trainX: ', _trainX.shape, ' _trainY: ', _trainY.shape
            gpr_results, knr_results = self.train_model_and_test(_trainX, _trainY, testX, testY, labels[i], 'all',
                                                                 '1357-1359')
            # each result set contains [pred, sum_pred, gt, gt_sum]

            graph_name ='onlyv1'
            vpath = files.mkdir(self.res_path,'graph_'+graph_name)
            self.plot_gpr(gpr_results[0], gpr_results[1], gpr_results[2], gpr_results[3], labels[i], graph_name, vpath)
            self.plot_knr(knr_results[0], knr_results[1], knr_results[2], knr_results[3], labels[i], graph_name, vpath)
            # self.make_video(dpcolors, fgset, gpr_results, 'gpr_' + labels[i], self.prepare.param1359)
            # self.make_video(dpcolors, fgset, knr_results, 'knr_' + labels[i], self.prepare.param1359)

    def train_model_and_test(self, _trainX, _trainY, testX, testY, label, mname, fname):
        """
        Learns GPR and KNR model from given training set and test on test set.

        Here, training set consists of every odd feature and test set consists of every training set is equal to test set.

        :param features:
        :param version:
        :param label:
        :return:
        """

        trainX, trainY = self.exclude_label(_trainX, _trainY, c=0)

        print '_trainX.shape: ', trainX.shape, ', _trainY.shape: ', trainY.shape

        PYGPR = 'gpr_' + label + '_' + mname
        KNR = 'knr_' + label + '_' + mname

        if files.isExist(self.model_path, PYGPR):
            gprmodel = self.loadf(self.model_path, PYGPR)
            knrmodel = self.loadf(self.model_path, KNR)

        else:
            print 'Learning GPR model'
            gprmodel = pyGPs.GPR()
            gprmodel.getPosterior(trainX, trainY)
            gprmodel.optimize(trainX, trainY)
            self.savef(self.model_path, PYGPR, gprmodel)

            print 'Learning KNR model'
            knrmodel = knr(trainX, trainY)
            self.savef(self.model_path, KNR, knrmodel)

            print 'Learning both GPR and KNR model is DONE.'

        # gpred, gsum_pred, ggt, ggt_sum = self.prediction_gpr_model(gprmodel,testX,[])
        # kpred, ksum_pred, kgt, kgt_sum = self.prediction_gpr_model(knrmodel,testX,[])

        gpr_result = self.prediction_gpr_model(gprmodel, testX, testY)
        knr_result = self.prediction_knr_model(knrmodel, testX, testY)
        return gpr_result, knr_result


        # self.plot_gpr(gprmodel,testX,testY,label,fname)
        # self.plot_knr(knrmodel,testX,testY,label,fname)

    def make_video(self, colordp, fgset, result, fname, param):

        # pred, sum_pred, gt, gt_sum
        Y_pred_frame = result[0]
        print 'Y_pred_frame: ', len(Y_pred_frame), 'fgset: ', len(fgset)
        videopath = files.mkdir(self.res_path, 'prediction_videos')
        imgset = []
        for i in range(len(fgset)):
            rect, cont = self.segmentation_blob(fgset[i], param)
            tmp = colordp[i].copy()

            pred = Y_pred_frame[i]
            # print 'rect size: ' , len(rect), 'frame_pred: ' , len(pred)
            # gt =  groundtruth[i]
            for j in range(len(rect)):
                r = rect[j]
                cv2.rectangle(tmp, (r[0], r[2]), (r[1], r[3]), tools.green, 2)
                msg_pred = '#:' + str(tools.int2round(pred[j]))
                cv2.putText(tmp, msg_pred, (r[0], r[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tools.blue, 2)
                # msg_gt = 'GT: '+str(gt[j])
                # cv2.putText(tmp, msg_gt, (r[0]+10,r[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, tools.red)

            imgset.append(tmp)
        # images.display_img(imgset, 300)
        images.write_video(imgset, 20, videopath, fname)

    def prediction_gpr_model(self, model, testX, testY):
        pred = np.array([])
        sum_pred = []

        gt = np.array([])
        gt_sum = []
        for y in testY:
            gt = np.hstack((gt, y))
            gt_sum.append(sum(y))

        for x in testX:
            ym, ys2, fm, fs2, lp = model.predict(np.array(x))
            ym = ym.reshape(ym.size)
            pred = np.hstack((pred, ym))
            sum_pred.append(sum(ym))

        return pred, sum_pred, gt, gt_sum

    def prediction_knr_model(self, model, testX, testY):
        pred = np.array([])
        sum_pred = []

        gt = np.array([])
        gt_sum = []
        for y in testY:
            gt = np.hstack((gt, y))
            gt_sum.append(sum(y))

        for x in testX:
            ym = model.predict(x)
            pred = np.hstack((pred, ym))
            sum_pred.append(sum(ym))

        return pred, sum_pred, gt, gt_sum

    def get_feature_by_label(self, _X, _Y, c):
        X = []
        Y = []

        for i in range(len(_X)):
            if _Y[i] != c:
                X.append(_X[i])
                Y.append(_Y[i])

        return np.array(X), np.array(Y)

    def plot_knr(self, Y_pred, Y_sum_pred, Y_label, Y_sum_label, label, fname, vpath):
        """

        :param Y_sum_pred: frame_prediction
        :param Y_pred: feature_prediction
        :param Y_sum_label: frame_gt
        :param Y_label: feature_gt
        :param label:
        :param fname:
        :return:
        """

        sxp = np.linspace(0, len(Y_sum_pred), len(Y_sum_pred))
        xp = np.linspace(0, len(Y_pred), len(Y_pred))

        plt.cla()
        plt.clf()
        plt.subplot(211)
        l1, = plt.plot(sxp, Y_sum_pred, 'b^', label='prediction')
        l2, = plt.plot(sxp, Y_sum_label, 'g.', label='groundtruth')
        l3, = plt.plot(sxp, abs(np.array(Y_sum_pred) - np.array(Y_sum_label)), 'r-', label='difference')
        plt.legend(handles=[l1, l2, l3], loc='best')
        plt.ylim([0, 40])
        plt.title('KNR estimation on frame ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')

        plt.subplot(212)
        l4, = plt.plot(xp, Y_pred, 'b^', label='prediction')
        l5, = plt.plot(xp, Y_label, 'g.', label='groundtruth')
        l6, = plt.plot(xp, abs(np.array(Y_pred) - np.array(Y_label)), 'r*', label='difference')
        plt.legend(handles=[l4, l5, l6], loc='best')
        plt.ylim([0, 9])
        plt.title('KNR estimation on feature ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')
        plt.tight_layout()
        plt.savefig(vpath + '/' + fname + '_knr_' + label + '.png')

        # plt.show()

    def plot_gpr(self, Y_pred, Y_sum_pred, Y_label, Y_sum_label, label, fname, vpath):
        """

        :param Y_sum_pred: frame_prediction
        :param Y_pred: feature_prediction
        :param Y_sum_label: frame_gt
        :param Y_label: feature_gt
        :param label:
        :param fname:
        :return:
        """
        print 'feature pred ', len(Y_pred), ' == ', len(Y_label)
        print 'frame pred ', len(Y_sum_pred), ' == ', len(Y_sum_label)

        sxp = np.linspace(0, len(Y_sum_pred), len(Y_sum_pred))
        xp = np.linspace(0, len(Y_pred), len(Y_pred))

        plt.cla()
        plt.clf()
        plt.subplot(211)
        l1, = plt.plot(sxp, Y_sum_pred, 'b^', label='prediction')
        l2, = plt.plot(sxp, Y_sum_label, 'g.', label='groundtruth')
        l3, = plt.plot(sxp, abs(np.array(Y_sum_pred) - np.array(Y_sum_label)), 'r-', label='difference')
        plt.legend(handles=[l1, l2, l3], loc='best')
        plt.ylim([0, 40])
        plt.title('GPR estimation on frame ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')

        plt.subplot(212)
        l4, = plt.plot(xp, Y_pred, 'b^', label='prediction')
        l5, = plt.plot(xp, Y_label, 'g.', label='groundtruth')
        l6, = plt.plot(xp, abs(np.array(Y_pred) - np.array(Y_label)), 'r*', label='difference')
        plt.legend(handles=[l4, l5, l6], loc='best')
        plt.title('GPR estimation on feature ' + label)
        plt.ylim([0, 9])
        plt.xlabel('feature index')
        plt.ylabel('# people')
        plt.tight_layout()
        plt.savefig(vpath + '/' + fname + '_gpr_' + label + '.png')
        # plt.show()

    def create_feature_set(self, fgset, dpcolor, weight, version, fname, param, givengt, gname):
        """
        Extracts features (e.g., K, S, P, E, T) from each image.

        K.shape = n_frames * 2
        S.shape = n_frames * 2
        P.shape = n_frames * 4
        E.shape = n_frames * 6
        T.shape = n_frames * 4

        :param prepare:
        :param fgset:
        :param dpcolor:
        :param weight:
        :param version:
        :return:
        """

        if files.isExist(self.param_path, 'v' + str(version) + '_' + fname):
            return

        print 'making feature set sequence.'

        contours_tree = []
        rectangles_tree = []
        groundtruth_tree = givengt  # self.read_count_groundtruth()

        for f in fgset:
            rect, cont = self.segmentation_blob(f, param)
            contours_tree.append(cont)
            rectangles_tree.append(rect)

        size = len(fgset)
        K = []
        groundtruth = []
        E = []
        T = []
        S = []
        S2 = []
        P = []
        for i in range(1, size - 1):
            print 'extracting at ', i, ', ', round(float(i) / size, 3), '%'
            groundtruth.append(groundtruth_tree[i])

            ks = directs.run_SURF_v4(dpcolor[i], weight, rectangles_tree[i])
            kf = directs.run_FAST_v4(dpcolor[i], weight, rectangles_tree[i])
            K.append(np.vstack((ks, kf)).T)

            e = directs.get_canny_edges(dpcolor[i], weight, rectangles_tree[i])
            E.append(e)

            t = directs.get_texture_T(dpcolor[i - 1:i + 2, :, :], rectangles_tree[i])
            T.append(t)

            l = indirects.get_size_L(fgset[i], weight, contours_tree[i])
            s = indirects.get_size_S(fgset[i], weight, contours_tree[i])
            s2 = indirects.get_size_S_v2(fgset[i], weight, rectangles_tree[i])
            S.append(np.vstack((s, l)).T)
            S2.append(np.vstack((s2, l)).T)

            p = indirects.get_shape_P(fgset[i], weight, contours_tree[i])
            P.append(p)

        K = np.array(K)
        np.save(self.param_path + '/v' + str(version) + '_' + fname, [K, E, T, P, S, S2])
        np.save(self.param_path + '/v' + str(version) + '_' + gname, groundtruth)

    def read_count_groundtruth(self):
        """
        reads text file which contains groundtruth.
        :return:
        """

        lines = files.read_text(self.res_path, 'count_gt')
        res = []
        for line in lines:
            tmp = line.split(',')
            tmp = tools.int2round(tmp)
            res.append(tmp)
        return res

    def exclude_label(self, _X, _Y, c):
        X = []
        Y = []

        for i in range(len(_X)):
            if _Y[i] != c:
                X.append(_X[i])
                Y.append(_Y[i])
        return np.array(X), np.array(Y)

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

    def savef(self, path, fname, given):
        with open(path + '/' + fname, 'w') as f:
            pickle.dump(given, f)

    def loadf(self, path, fname):
        with open(path + '/' + fname) as f:
            return pickle.load(f)


if __name__ == '__main__':
    print "Main Starts", '--------------------------' * 5
    worker()
    print "Main Ends", '--------------------------' * 5
