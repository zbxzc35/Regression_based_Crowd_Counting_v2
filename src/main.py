
from tiah import FileManager as files
from tiah import tools as tools
from src.Prepare import Prepare
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


    def run_gist_case(self):
        chdir('..')
        self.bpath = files.mkdir(getcwd(), 'gist')
        self.res_path = files.mkdir(self.bpath, 'res')
        self.param_path = files.mkdir(self.res_path,'params')
        self.graph_path = files.mkdir(self.res_path,'graphs')
        self.model_path = files.mkdir(self.res_path,'models')
        prepare = Prepare(self.res_path)
        self.FEATURE ='featureset.npy'
        self.GT = 'groundtruth.npy'

        fgset, dp_color, dp_mask = prepare.prepare_gist()




    def run_pets_case(self):
        chdir('..')
        self.bpath = files.mkdir(getcwd(), 'S1L1')
        self.res_path = files.mkdir(self.bpath, 'res')
        self.param_path = files.mkdir(self.res_path,'params')
        self.graph_path = files.mkdir(self.res_path,'graphs')
        self.model_path = files.mkdir(self.res_path,'models')
        prepare = Prepare(self.res_path)
        self.FEATURE ='featureset.npy'
        self.GT = 'groundtruth.npy'

        a1357, b1357, a1359, b1359, weight = prepare.prepare()
        fg1357 = a1357[0]
        dpcolor1357 = b1357[0]
        dpmask1357 = b1357[1]
        version =4
        # v2: only K
        # v3: only K E T
        # v3: K E T P S S2

        self.create_feature_set(prepare,fg1357,dpcolor1357,weight, version)
        features = np.load(self.param_path+'/v'+ str(version) + '_'+self.FEATURE)
        labels =['K','E','T','P','S','S2']
        #K,E,T,P,S,S2
        K = features[0]
        S = features[4]
        KS = []
        for i in range(len(K)):
            KS.append(np.hstack((K[i],S[i])))
        KS = np.array(KS)
        self.test(KS,version,'KS')
        self.test_trainset_test_same(KS,version,'KS')
        quit('qqqqqqqqqqqqqq')

        for i in range(len(labels)):
            self.test(features[i],version,labels[i])

        for i in range(len(labels)):
            self.test_trainset_test_same(features[i],version,labels[i])

    def test3d(self,X,Y,Z):

        plot_type = ['b', 'g', 'r', 'c', 'm', 'y']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X,Y,Z, c='b', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def pca_test(self,features, version, label):
        groundtruth = np.load(self.param_path+'/v'+ str(version) + '_'+self.GT)

        _trainX = np.concatenate(features[0:features.shape[0]:2])
        _trainY = np.concatenate(groundtruth[0:groundtruth.size:2])
        pca = PCA(n_components=2)
        X = pca.fit_transform(_trainX)
        self.test3d(X[:,0],X[:,1],_trainY)


    def test_trainset_test_same(self,features, version, label):
        """
        Learns GPR and KNR model from given training set and test on test set.

        Here, training set is equal to test set.

        :param features:
        :param version:
        :param label:
        :return:
        """
        groundtruth = np.load(self.param_path + '/v' + str(version) + '_' + self.GT)

        _trainX = np.concatenate(features)
        _trainY = np.concatenate(groundtruth)

        trainX, trainY = self.exclude_label(_trainX, _trainY, c=0)
        testX = features
        testY = groundtruth

        PYGPR = 'gpr_all_'+label
        KNR = 'knr_all_'+label
        if files.isExist(self.model_path, PYGPR):
            gprmodel = self.loadf(self.model_path, PYGPR)
            knrmodel = self.loadf(self.model_path,KNR)

        else:
            print 'Learning GPR model'
            gprmodel = pyGPs.GPR()
            gprmodel.getPosterior(trainX, trainY)
            gprmodel.optimize(trainX, trainY)
            self.savef(self.model_path, PYGPR, gprmodel)

            print 'Learning KNR model'
            knrmodel = knr(trainX,trainY)
            self.savef(self.model_path,KNR,knrmodel)

            print 'Learning both GPR and KNR model is DONE.'

        self.plot_gpr(gprmodel,testX,testY,label,'all_feature')
        self.plot_knr(knrmodel,testX,testY,label,'all_feature')



    def test(self,features, version, label):
        """
        Learns GPR and KNR model from given training set and test on test set.

        Here, training set consists of every odd feature and test set consists of every training set is equal to test set.

        :param features:
        :param version:
        :param label:
        :return:
        """

        groundtruth = np.load(self.param_path+'/v'+ str(version) + '_'+self.GT)

        _trainX = np.concatenate(features[0:features.shape[0]:2])
        _trainY = np.concatenate(groundtruth[0:groundtruth.size:2])
        testX = features[1:features.shape[0]:2]
        testY = groundtruth[1:groundtruth.size:2]

        print 'features.shape: ' ,features.shape, ', groundtruth.shape: ' , groundtruth.shape
        print '_trainX.shape: ' ,_trainX.shape, ', _trainY.shape: ' , _trainY.shape

        trainX, trainY = self.exclude_label(_trainX,_trainY,c=0)

        PYGPR = 'gpr_'+label
        KNR = 'knr_'+label
        if files.isExist(self.model_path, PYGPR):
            gprmodel = self.loadf(self.model_path, PYGPR)
            knrmodel = self.loadf(self.model_path,KNR)

        else:
            print 'Learning GPR model'
            gprmodel = pyGPs.GPR()
            gprmodel.getPosterior(trainX, trainY)
            gprmodel.optimize(trainX, trainY)
            self.savef(self.model_path, PYGPR, gprmodel)

            print 'Learning KNR model'
            knrmodel = knr(trainX,trainY)
            self.savef(self.model_path,KNR,knrmodel)

            print 'Learning both GPR and KNR model is DONE.'


        self.plot_gpr(gprmodel,testX,testY,label,'odd_feature')
        self.plot_knr(knrmodel,testX,testY,label,'odd_feature')


    def get_feature_by_label(self,_X,_Y,c):
        X = []
        Y = []

        for i in range(len(_X)):
            if _Y[i] != c:
                X.append(_X[i])
                Y.append(_Y[i])

        return np.array(X), np.array(Y)

    def visualize_video(self,features, version, label, _fgset, _colordp, param):
        groundtruth = np.load(self.param_path+'/v'+ str(version) + '_'+self.GT)

        _trainX = np.concatenate(features[0:features.shape[0]:2])
        _trainY = np.concatenate(groundtruth[0:groundtruth.size:2])
        testX = features[1:features.shape[0]:2]
        testY = groundtruth[1:groundtruth.size:2]

        np.savetxt(self.res_path+'/feature_'+label+'.txt',np.hstack((_trainX,_trainY.reshape(-1,1))),fmt='%d')
        print 'features.shape: ' ,features.shape, ', groundtruth.shape: ' , groundtruth.shape
        print '_trainX.shape: ' ,_trainX.shape, ', _trainY.shape: ' , _trainY.shape

        trainX, trainY = self.exclude_label(_trainX,_trainY,c=0)

        PYGPR = 'gpr_'+label
        KNR = 'knr_'+label
        if files.isExist(self.res_path, PYGPR):
            gprmodel = self.loadf(self.res_path, PYGPR)
            knrmodel = self.loadf(self.res_path,KNR)

        else:
            print 'Learning GPR model'
            gprmodel = pyGPs.GPR()
            gprmodel.getPosterior(trainX, trainY)
            gprmodel.optimize(trainX, trainY)
            self.savef(self.res_path, PYGPR, gprmodel)

            print 'Learning KNR model'
            knrmodel = knr(trainX,trainY)
            self.savef(self.res_path,KNR,knrmodel)

            print 'Learning both GPR and KNR model is DONE.'


        Y_pred = np.array([])
        Y_sum_pred = []
        Y_pred_frame = []
        for x in testX:
            ym, ys2, fm, fs2, lp = gprmodel.predict(np.array(x))
            Y_pred = np.hstack((Y_pred, ym.reshape(ym.size)))
            ym = ym.reshape(ym.size)
            Y_sum_pred.append(sum(ym))
            Y_pred_frame.append(ym)

        Y_label = []
        Y_sum_label = []

        for y in testY:
            Y_label += y
            Y_sum_label.append(sum(y))

        imgset= []
        fgset = _fgset[1:len(_fgset)-1]
        colordp = _colordp[1:len(_colordp)-1]
        for i in range(len(fgset)):
            rect, cont = self.segmentation_blob(fgset[i],param)
            tmp = colordp[i].copy()

            pred = Y_pred_frame[i]
            gt =  groundtruth[i]
            for j in range(len(rect)):
                r = rect[j]
                cv2.rectangle(tmp,(r[0],r[2]),(r[1],r[3]),tools.green,1)

                msg_pred = 'Pred: '+str(pred[j])
                msg_gt = 'GT: '+str(gt[j])
                cv2.putText(tmp, msg_pred, (r[0],r[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, tools.blue)
                cv2.putText(tmp, msg_gt, (r[0]+10,r[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, tools.red)

            imgset.append(tmp)
        images.display_img(imgset,300)




    def plot_knr(self, model, testX, testY, label, fname):
        """
        Predicts on test set using the given model.

        :param model:
        :param testX: n_frames * (m_features * d_feature_dimension)
        :param testY: n_frames * m_features
        :param label:
        :return:
        """

        Y_pred = np.array([])
        Y_sum_pred = []
        for x in testX:
            ym  = model.predict(x)
            Y_pred = np.hstack((Y_pred, ym))
            Y_sum_pred.append(sum(ym))

        Y_label = []
        Y_sum_label = []
        for y in testY:
            Y_label += y
            Y_sum_label.append(sum(y))

        Y_sum_label = np.array(Y_sum_label)
        Y_sum_pred = np.array(Y_sum_pred)

        Y_sum_label = np.array(Y_sum_label)
        Y_sum_pred = np.array(Y_sum_pred)

        sxp = np.linspace(0, len(Y_sum_pred), len(Y_sum_pred))
        xp = np.linspace(0, len(Y_pred), len(Y_pred))

        plt.cla()
        plt.clf()
        plt.subplot(211)
        l1, = plt.plot(sxp, Y_sum_pred, 'b^', label='prediction')
        l2, = plt.plot(sxp, Y_sum_label, 'g.', label='groundtruth')
        l3, = plt.plot(sxp, abs(Y_sum_pred - Y_sum_label), 'r-', label='difference')
        plt.legend(handles=[l1, l2, l3], loc='best')
        plt.ylim([0,40])
        plt.title('KNR estimation on frame ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')

        plt.subplot(212)
        l4, = plt.plot(xp, Y_pred, 'b^', label='prediction')
        l5, = plt.plot(xp, Y_label, 'g.', label='groundtruth')
        l6, = plt.plot(xp, abs(Y_pred - Y_label), 'r*', label='difference')
        plt.legend(handles=[l4, l5, l6], loc='best')
        plt.ylim([0, 9])
        plt.title('KNR estimation on feature '+ label)
        plt.xlabel('frame index')
        plt.ylabel('# people')
        plt.tight_layout()
        plt.savefig(self.graph_path + '/' + fname +'_knr_'+label+ '.png')

        # plt.show()

    def plot_gpr(self, model, testX, testY, label,fname):
        """
        Predicts on test set using the given model.

        :param model:
        :param testX: n_frames * (m_features * d_feature_dimension)
        :param testY: n_frames * m_features
        :param label:
        :return:
        """

        Y_pred = np.array([])
        Y_sum_pred = []
        for x in testX:
            ym, ys2, fm, fs2, lp = model.predict(np.array(x))
            Y_pred = np.hstack((Y_pred, ym.reshape(ym.size)))
            ym = ym.reshape(ym.size)
            Y_sum_pred.append(sum(ym))

        Y_label = []
        Y_sum_label = []
        for y in testY:
            Y_label += y
            Y_sum_label.append(sum(y))

        Y_sum_label = np.array(Y_sum_label)
        Y_sum_pred = np.array(Y_sum_pred)

        Y_sum_label = np.array(Y_sum_label)
        Y_sum_pred = np.array(Y_sum_pred)

        sxp = np.linspace(0, len(Y_sum_pred), len(Y_sum_pred))
        xp = np.linspace(0, len(Y_pred), len(Y_pred))

        plt.cla()
        plt.clf()
        plt.subplot(211)
        l1, = plt.plot(sxp, Y_sum_pred, 'b^', label='prediction')
        l2, = plt.plot(sxp, Y_sum_label, 'g.', label='groundtruth')
        l3, = plt.plot(sxp, abs(Y_sum_pred - Y_sum_label), 'r-', label='difference')
        plt.legend(handles=[l1, l2, l3], loc='best')
        plt.ylim([0, 40])
        plt.title('GPR estimation on frame ' + label)
        plt.xlabel('frame index')
        plt.ylabel('# people')

        plt.subplot(212)
        l4, = plt.plot(xp, Y_pred, 'b^', label='prediction')
        l5, = plt.plot(xp, Y_label, 'g.', label='groundtruth')
        l6, = plt.plot(xp, abs(Y_pred - Y_label), 'r*', label='difference')
        plt.legend(handles=[l4, l5, l6], loc='best')
        plt.title('GPR estimation on feature '+ label)
        plt.ylim([0, 9])
        plt.xlabel('feature index')
        plt.ylabel('# people')
        plt.tight_layout()
        plt.savefig(self.graph_path + '/' + fname + '_gpr_' + label + '.png')
        # plt.show()

    def create_feature_set(self,prepare, fgset, dpcolor, weight, version):
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

        if files.isExist(self.param_path,'v'+ str(version) + '_'+self.FEATURE):
            return

        print 'making feature set sequence.'

        contours_tree= []
        rectangles_tree = []
        param = prepare.min_width_height()
        groundtruth_tree = self.read_count_groundtruth()


        for f in fgset:
            rect, cont = self.segmentation_blob(f,param)
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

            ks = directs.run_SURF_v4(dpcolor[i],weight,rectangles_tree[i])
            kf = directs.run_FAST_v4(dpcolor[i],weight,rectangles_tree[i])
            K.append( np.vstack((ks,kf)).T)

            e = directs.get_canny_edges(dpcolor[i],weight,rectangles_tree[i])
            E.append(e)

            t = directs.get_texture_T(dpcolor[i - 1:i + 2, :, :],rectangles_tree[i])
            T.append(t)

            l = indirects.get_size_L(fgset[i],weight,contours_tree[i])
            s = indirects.get_size_S(fgset[i],weight,contours_tree[i])
            s2 = indirects.get_size_S_v2(fgset[i],weight,rectangles_tree[i])
            S.append(np.vstack((s,l)).T)
            S2.append(np.vstack((s2,l)).T)

            p = indirects.get_shape_P(fgset[i],weight,contours_tree[i])
            P.append(p)

        K = np.array(K)
        np.save(self.param_path+'/v'+ str(version) + '_'+self.FEATURE, [K,E,T,P,S,S2])
        np.save(self.param_path+'/v'+ str(version) + '_'+self.GT, groundtruth )

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

    def exclude_label(self, _X,_Y, c):
        X = []
        Y = []

        for i in range(len(_X)):
            if _Y[i] != c:
                X.append(_X[i])
                Y.append(_Y[i])
        return np.array(X), np.array(Y)

    def segmentation_blob(self, given_frame, param):
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
        contours, heiarchy = cv2.findContours(given_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # check here

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


if __name__=='__main__':
    print "Main Starts", '--------------------------' * 5
    worker()
    print "Main Ends", '--------------------------' * 5
