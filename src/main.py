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
import time
from tiah import tools as tools

##TODO Adaptive thresholding.

class worker:
    def __init__(self):

        P = range(2, 3)
        D = range(2, 3)
        # d1~d6: using pn
        # d7: only E using p3
        # d8: only E using p5
        # d9: only E using p6

        chdir('..')
        for pv, dv in zip(P, D):
            self.run_pets_case(pv, dv)
            # self.run_pets_case(5, 7)
            # self.run_gist_case()

    def run_pets_case(self, pv, dv):

        self.param_version = pv  # 4
        self.feature_version = 5
        # self.feature_version = 4 # for d1~d6
        self.dir_version = dv  # 1  # directory differs parameter

        self.bpath = files.mkdir(getcwd(), 'S1L1')
        self.res_path = files.mkdir(self.bpath, 'res' + str(self.dir_version))
        self.param_path = files.mkdir(self.bpath, 'params_v' + str(self.param_version))
        self.graph_path = files.mkdir(self.res_path, 'graphs')
        self.model_path = files.mkdir(self.res_path, 'models')
        self.prepare = Prepare(self.bpath, self.res_path, self.param_version)
        self.prepare.init_pets()
        # prepare.test_background_subtractor()

        self.COUNTGT1357 = 'c_groundtruth1357.npy'
        self.COUNTGT1359 = 'c_groundtruth1359.npy'

        print 'Current Param Version: ', self.param_version
        print 'Current Directory Version: ', self.dir_version

        a1357, b1357, a1359, b1359, weight, gt1357, gt1359 = self.prepare.prepare()
        param1357 = self.prepare.param1357
        param1359 = self.prepare.param1359
        fg1357 = a1357[0]
        dpcolor1357 = b1357[0]
        dpmask1357 = b1357[1]

        fg1359 = a1359[0]
        dpcolor1359 = b1359[0]
        dpmask1359 = b1359[1]

        # import segm as segm
        # segm.testing(fg1357,dpcolor1357,dpmask1357,param1357,self.res_path)
        # quit()



        # dplist = self.draw_shapes(self.prepare,fg1357,dpcolor1357,dpmask1357)
        # tmp_path = files.mkdir(self.param_path, 'rect_contour')
        # images.write_imgs(dplist,tmp_path,'rc')
        # images.write_video(dplist,8, tmp_path,'rc')
        # return 1

        E,K,P,S,T = 1,1,1,1,1
        feature_version = {'E': E, 'K': K, 'P': P, 'S': S, 'T': T}

        self.create_feature_set(fg1357, dpcolor1357, weight, feature_version, param1357, gt1357,
                                self.COUNTGT1357, '1357')
        self.create_feature_set(fg1359, dpcolor1359, weight, feature_version, param1359, gt1359,
                                self.COUNTGT1359, '1359')

        print 'data1357: ', fg1357.shape, dpcolor1357.shape, len(dpmask1357)
        print 'data1359: ', fg1359.shape, dpcolor1359.shape, len(dpmask1359)
        # K,E,T,P,S,S2
        length = len(dpcolor1359)

        comb1357, labels, groundtruth1357 = self.load_features_by_version(feature_version, '1357')
        comb1359, labels, groundtruth1359 = self.load_features_by_version(feature_version, '1359')

        self.dowork(comb1357, comb1359, labels, dpcolor1359[1:length - 1], fg1359[1:length - 1],
                    groundtruth1357, groundtruth1359)


    def load_features_by_version(self, version, flag):
        feature_path = files.mkdir(self.param_path, flag)
        print 'loading features from ', feature_path
        labels = ['E', 'K', 'T', 'P', 'S']
        E = np.load(feature_path + '/feature_E_v' + str(version['E']) + '.npy')  # np.array(features[0])
        K = np.load(feature_path + '/feature_K_v' + str(version['K']) + '.npy')  # np.array(features[1])
        T = np.load(feature_path + '/feature_T_v' + str(version['T']) + '.npy')  # np.array(features[2])
        P = np.load(feature_path + '/feature_P_v' + str(version['P']) + '.npy')  # np.array(features[3])
        S = np.load(feature_path + '/feature_S_v' + str(version['S']) + '.npy')  # np.array(features[4])
        # S2 = np.load(self.param_path + '/feature_P_v' + str(version['P'])) #np.array(features[5])

        combinations = [E, K, T, P, S]
        if flag == '1357':
            groundtruth = np.load(feature_path + '/' + self.COUNTGT1357)
        else:
            groundtruth = np.load(feature_path + '/' + self.COUNTGT1359)

        return combinations, labels, groundtruth



    def draw_shapes(self, prepare, fg1357, dp1357, dpmask1357):
        """
        draw segmentation and contour ground truth for every segmentation for every frame.
        :param prepare:
        :param fg1357:
        :param dp1357:
        :param dpmask1357:
        :return:
        """

        #####################################
        # DO NOT DELETE
        xml1357 = 'PETS2009-S1L1-1'
        GT1357 = 'xml_groundtruth1357.npy'

        contour_tree = []
        rect_tree = []
        for fg in fg1357:
            list_rect, list_contours = self.segmentation_blob(fg, prepare.param1357)
            contour_tree.append(list_contours)
            rect_tree.append(list_rect)

        dp_count_tree, count_tree = prepare.create_count_groundtruth(fg1357, xml1357, 'adf', 1357)

        dplist = []
        for frame_count, frame_rect, frame_contour, dpcolor, dpmask in zip(count_tree, rect_tree, contour_tree, dp1357,
                                                                           dpmask1357):
            for s, c in zip(frame_rect, frame_count):
                cv2.rectangle(dpcolor, (s[0], s[2]), (s[1], s[3]), tools.orange, 1)
                cv2.rectangle(dpmask, (s[0], s[2]), (s[1], s[3]), tools.orange, 1)
                cv2.putText(dpcolor, str(c), (s[0], s[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tools.blue, 2)
                cv2.putText(dpmask, str(c), (s[0], s[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tools.blue, 2)


            cv2.drawContours(dpcolor, frame_contour, -1, tools.green, 2)
            cv2.drawContours(dpmask, frame_contour, -1, tools.green, 2)
            dplist.append(np.hstack((dpcolor, dpmask)))
            # cv2.imshow('1', np.hstack((dpcolor, dpmask)))
            # cv2.waitKey(0)
        return dplist


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

    def dowork(self, features1357, features1359, flabels, dpcolors, fgset, groundtruth1357, groundtruth1359):

        print '1357 case'
        print 'frame: ', features1357[0].shape, ' Groundntruth: ', groundtruth1357.shape
        print 'feature X: ', np.concatenate(features1357[0]).shape, ' label Y: ', np.concatenate(groundtruth1357).shape

        print '1359 case'
        print 'frame: ', features1359[0].shape, ' Groundntruth: ', groundtruth1359.shape
        print 'feature X: ', np.concatenate(features1359[0]).shape, ' label Y: ', np.concatenate(groundtruth1359).shape


        MAE_frame = []
        MAE_feature = []

        trainY = np.concatenate(groundtruth1357)
        # self.plot_label_distribution(trainY)
        pred_results = {}

        models= []
        print 'Load GPR models'
        for i in range(len(flabels)):
            train_feature = features1357[i]
            trainX = np.concatenate(train_feature)
            gpr_model = self.training_model_by_dataset(trainX, trainY,flabels[i])
            models.append(gpr_model)

        print 'Prediction on each models'
        for i in range(len(flabels)):
            testX = features1359[i]
            pred, featureY, frameY = self.prediction_gpr_model2(models[i],testX,groundtruth1359)
            pred_results[flabels[i]] = (pred, featureY, frameY )



            # MAE_frame.append(np.mean(np.abs(np.array(sum_pred)-np.array(gt_sum))))
            # MAE_feature.append(np.mean(np.abs(np.array(pred)-np.array(gt))))

            # self.plot_knr(knr_results[0], knr_results[1], knr_results[2], knr_results[3], labels[i], graph_name, vpath)
            # self.make_video(dpcolors, fgset, gpr_results, 'gpr_' + labels[i], self.prepare.param1359)
            # self.make_video(dpcolors, fgset, knr_results, 'knr_' + labels[i], self.prepare.param1359)
        print 'MAE per_frame: ', MAE_frame
        print 'MAE per feature: ' , MAE_feature



        # self.plot_gpr(gprmodel,testX,testY,label,fname)
        # self.plot_knr(knrmodel,testX,testY,label,fname)

    def plot_label_distribution(self,trainY):
        print 'np.unqiue groundtruth: ', np.unique(trainY)
        bins = len(np.unique(trainY))
        plt.hist(trainY,bins)
        plt.ylabel('# of datas')
        plt.xlabel('label Y')
        plt.title('Training data distribution')
        plt.show()

    def training_model_by_dataset(self, _trainX, _trainY, flabel):
        trainX, trainY = self.exclude_label(_trainX, _trainY, c=0)

        print '_trainX.shape: ', trainX.shape, ', _trainY.shape: ', trainY.shape

        PYGPR = 'gpr_model' + flabel

        if files.isExist(self.model_path, PYGPR):
            gprmodel = self.loadf(self.model_path, PYGPR)

        else:
            print 'Learning GPR model'
            gprmodel = pyGPs.GPR()
            gprmodel.getPosterior(trainX, trainY)
            gprmodel.optimize(trainX, trainY)
            self.savef(self.model_path, PYGPR, gprmodel)

        return gprmodel

    def prediction_gpr_model2(self, model, testX, testY):

        pred = []

        featureY = np.array([])
        frameY = []
        for y in testY:
            featureY = np.hstack((featureY, y))
            frameY.append(sum(y))

        for x in testX:
            ym, ys2, fm, fs2, lp = model.predict(np.array(x))
            pred.append(ym)

        return pred, featureY, frameY


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
        l3, = plt.plot(sxp, abs(np.array(Y_sum_pred) - np.array(Y_sum_label)), 'r-', label='error')
        plt.legend(handles=[l1, l2, l3], loc='best')
        plt.ylim([0, 40])
        plt.title('GPR estimation on frame ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')

        plt.subplot(212)
        l4, = plt.plot(xp, Y_pred, 'b^', label='prediction')
        l5, = plt.plot(xp, Y_label, 'g.', label='groundtruth')
        l6, = plt.plot(xp, abs(np.array(Y_pred) - np.array(Y_label)), 'r*', label='error')
        plt.legend(handles=[l4, l5, l6], loc='best')
        plt.title('GPR estimation on feature ' + label)
        plt.ylim([0, 9])
        plt.xlabel('feature index')
        plt.ylabel('# people')
        plt.tight_layout()
        plt.savefig(vpath + '/' + fname + '_gpr_' + label + '.png')
        # plt.show()

    def create_feature_set(self, fgset, dpcolor, weight, version, param, givengt, gname, flag):
        """
        Extracts features (e.g., K, S, P, E, T) from each image.

        K.shape = n_frames * 2
        S.shape = n_frames * 2
        P.shape = n_frames * 4
        E.shape = n_frames * 6
        T.shape = n_frames * 4

        :param fgset:
        :param dpcolor:
        :param weight:
        :param version:
        :param param:
        :param givengt:
        :param gname:
        :return:
        """
        print 'making feature set sequence, in ', self.param_path

        feature_path = files.mkdir(self.param_path, flag)

        contours_tree = []
        rectangles_tree = []
        groundtruth_tree = givengt  # self.read_count_groundtruth()

        for f in fgset:
            rect, cont = self.segmentation_blob(f, param)
            contours_tree.append(cont)
            rectangles_tree.append(rect)

        size = len(fgset)

        if not files.isExist(feature_path, gname):
            groundtruth = []
            for i in range(1, size - 1):
                groundtruth.append(groundtruth_tree[i])
            np.save(feature_path + '/' + gname, groundtruth)

        if not files.isExist(feature_path, 'feature_E_v' + str(version['E']) + '.npy'):

            E = []
            for i in range(1, size - 1):
                e = directs.get_canny_edges(dpcolor[i], weight, rectangles_tree[i], self.dir_version)
                E.append(e)
            np.save(feature_path + '/feature_E_v' + str(version['E']), E)

        if not files.isExist(feature_path, 'feature_K_v' + str(version['K']) + '.npy'):
            print 'K not exist'
            K = []
            for i in range(1, size - 1):
                ks = directs.run_SURF_v4(dpcolor[i], weight, rectangles_tree[i])
                kf = directs.run_FAST_v4(dpcolor[i], weight, rectangles_tree[i])
                K.append(np.vstack((ks, kf)).T)
            np.save(feature_path + '/feature_K_v' + str(version['K']), K)
        else:
            print 'K exist'

        if not files.isExist(feature_path, 'feature_T_v' + str(version['T']) + '.npy'):
            T = []
            for i in range(1, size - 1):
                t = directs.get_texture_T(dpcolor[i - 1:i + 2, :, :], rectangles_tree[i])
                T.append(t)
            np.save(feature_path + '/feature_T_v' + str(version['T']), T)

        if not files.isExist(feature_path, 'feature_S_v' + str(version['S']) + '.npy'):
            S = []
            for i in range(1, size - 1):
                l = indirects.get_size_L(fgset[i], weight, contours_tree[i])
                s = indirects.get_size_S(fgset[i], weight, contours_tree[i])
                S.append(np.vstack((s, l)).T)
            np.save(feature_path + '/feature_S_v' + str(version['S']), S)

        if not files.isExist(feature_path, 'feature_P_v' + str(version['P']) + '.npy'):
            P = []
            for i in range(1, size - 1):
                p = indirects.get_shape_P(fgset[i], weight, contours_tree[i])
                P.append(p)
            np.save(feature_path + '/feature_P_v' + str(version['P']), P)


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


worker()
# if __name__ == '__main__':
#     print "Main Starts", '--------------------------' * 5
#     worker()
#     print "Main Ends", '--------------------------' * 5
