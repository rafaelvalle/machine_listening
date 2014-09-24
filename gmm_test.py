import pylab as plt
import matplotlib as mpl
import numpy as np
import itertools

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from sklearn.metrics import confusion_matrix

from data_mining import *
from pre_processing import standardize, normalize

"""
#example of usage
features = ['median', 'mean', 'std',\
                        'ps_median', 'ps_mean', 'ps_std', \
                        'centroid', 'spread', 'skewness', 'kurtosis', 'slope', \
                        'mfcc', 'mae']

#features = ['median', 'mean']

results = {}

location_id = 30
door_count_placement_view_pair = ('1000029', '0')
start = 9
end =  28
start_time = str2date('2013-05-'+ "%02d" % start+'-00-00-00')
end_time = str2date('2013-05-'+ "%02d" % end+'-00-00-00')
start_time = pytz.timezone("America/Los_Angeles").localize(start_time, is_dst=None)
end_time = pytz.timezone("America/Los_Angeles").localize(end_time, is_dst=None)

plotEE = True   #Plot PD
plotPF = False  #Prediction and Fit
plotCM = False  #Confusion Matrix

"""

#############
#  METHODS  #
#############
def make_ellipses(gmm, figure):
  """Given a GMM it adds an error ellipse to the figure
    gmm: gmm classifier <sklearn.mixture.GMM>
    figure: figure from <plt.subplot>
  """
  for n, color in enumerate('rgb'):
    v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(figure.bbox)
    ell.set_alpha(0.5)
    figure.add_artist(ell)

def run(location_id, door_count_placement_view_pair, start_time, end_time, features, n_components=16, pre_processing='', BALANCE_DATA=False):
  """ Fits data to one GMM and plots confusion matrix, prediction and error ellipses
  location_id: location_id of installation, eg '55' <int> or <str>
  door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
  start_time: for prunning. time object with hour and minute time <time>
  end_time: for prunning. time object with hour and minute time <time>
  features: list with label (keys) of features used, [<str>,<str>...]
  n_components: number of mixture components 
  pre_processing: pre-processing to be applied; accepts 'standardize' and 'gaussianize' with default values <str>
  BALANCE_DATA: hack for trying to balance the data <bool>
  """

  global plotEE, plotPF, plotCM

  n_folds = 4

  print '\npre-processing step'
  standardizer = None
  if 'standardize' in pre_processing:
        standardizer = Standardizer(copy=copy, with_mean=True, with_std=True)
  if 'gaussianize' in pre_processing:
    gaussianizer = robjects.r('Gaussianize')

  train_X, train_Y = collectData(train_location_id, train_placement_view_pair, train_start_time, train_end_time, features, adjusted=True, pre_processor=standardizer)

  if standardizer is not None:
    train_X = standardizer.transform(train_X, copy=None)
  if gaussianizer is not None:
    from rpy2.robjects.numpy2ri import numpy2ri
    ro.conversion.py2ri = numpy2ri
    rtrain_X = gaussianizer(train_X)
    train_X = np.array(rtrain_X)
    gaussianizer.mean = train_X.mean(axis=1)
    gaussianizer.std = train_X.mean(axis=1)


  #add ones column
  print 'adding constant to train_X'
  if len(train_X.shape) > 1:
    ones_array = np.ones((train_X.shape[0],1))
    train_X = np.append(train_X, ones_array, 1)
  else:
    train_X= np.dstack((train_X, np.ones(len(train_X))))
    if len(train_X.shape) == 3:
      train_X= train_X[0]

  if pre_processing == 'normalize':
    print 'normalizing data'
    normalizer = normalize(train_X)

    #use sqrt of target to reduce hypothesis space 
    truth[truth<0] = 0
    truth = np.sqrt(truth).astype(int)
    truth_str = map(str, truth)

    #hack to better balance the data
    if BALANCE_DATA:
        bins = np.bincount(truth)
        avg = bins.mean()
        dev = bins.std()
        tol = 10
        tosmall = np.where(bins < avg - tol)[0]
        tobig = np.where(bins > avg + tol)[0]

        for item in tosmall:
            data = np.delete(data, np.where(truth == item)[0], axis=0)
            truth = np.delete(truth, np.where(truth == item)[0])

        for item in tobig:
            data = np.delete(data, np.where(truth == item)[0], axis=0)
            truth = np.delete(truth, np.where(truth == item)[0])

    #unbalanced targets affects, can't use StratifiedKFold, and, more important, GMM!, which assumes equal probability to all classes
    print 'Sample size is', len(truth)
    folds = KFold(len(truth), n_folds=n_folds) #shuffle=True, random_state=4
    #to only take the first fold
    #train_index, test_index = next(iter(folds)) 

    #for plotting
    idx = 1

    for train_index, test_index in folds:
        X_train = data[train_index]
        y_train = truth[train_index]
        X_test = data[test_index]
        y_test = truth[test_index]

        # Try GMMs using different types of covariances.
        classifiers = dict((covar_type, GMM(n_components=n_components,
                            covariance_type=covar_type, params='wmc', init_params='wmc', n_iter=10000))
                             for covar_type in ['spherical', 'diag', 'tied', 'full'])

        n_classifiers = len(classifiers)

        if plotEE:
            plt.figure(idx, figsize=(3 * n_classifiers / 2, 6))
            plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)

        for index, (name, classifier) in enumerate(classifiers.iteritems()):
            #np.array([X_train[y_train == i].mean(axis=0) for i in xrange(n_classes)])
            #start classifier with known means
            classifier.means_ = np.array([X_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
            classifier.fit(X_train)


            y_train_pred = classifier.predict(X_train)
            y_train_pred = y_train_pred.astype(int)

            yresid = y_train - y_train_pred;
            SSresid = np.sum(yresid**2)
            SStotal = (len(y_train)-1) * np.var(y_train)
            train_accuracy = 1 - SSresid/SStotal #rsq isntead of np.mean(y_train_pred.ravel() == y_train.ravel()) * 100

            y_test_pred = classifier.predict(X_test)
            y_test_pred = y_test_pred.astype(int)

            yresid = y_test - y_test_pred;
            SSresid = np.sum(yresid**2)
            SStotal = (len(y_test)-1) * np.var(y_test)
            test_accuracy = 1 - SSresid/SStotal #rsq instead of np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
            
            """
            if features not in results:
                results[features] = {}
                results[features][name] = (train_accuracy, test_accuracy)
            """

            if plotEE:
                plt.figure(idx, figsize=(3 * n_classifiers / 2, 6))
                plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)
                fig1 = plt.subplot(2, n_classifiers / 2, index + 1)

                make_ellipses(classifier, fig1)

                for n, color in enumerate('rgb'):
                    sample = data[truth == n]
                    plt.scatter(sample[:, 0], sample[:, 1], 0.8, color=color, label=truth_str[n])

                for n, color in enumerate('rgb'):
                    sample = X_test[y_test == n]
                    plt.plot(sample[:, 0], sample[:, 1], 'x', color=color)

                plt.text(0.05, 0.9, 'Train accuracy: %.2f' % train_accuracy, transform=fig1.transAxes)

                plt.text(0.05, 0.8, 'Test accuracy: %.2f' % test_accuracy, transform=fig1.transAxes)

                plt.xticks(())
                plt.yticks(())
                plt.title(name)

            if plotPF:
                #plot Ground Truth and Prediction
                x_train = np.array(range(len(y_train)))
                x_test = np.array(range(len(y_test)))

                plt.figure(idx*n_folds + index)
                fig2, ax2 = plt.subplots(2, 1, 1, figsize=(3 * n_classifiers / 2, 6))
                #ax2 = plt.subplot(2, 2, index + 1)

                ax2[0].plot(x_train, y_train, label='Train Ground Truth')
                ax2[0].plot(x_train, y_train_pred, label='Train Prediction')
                ax2[1].plot(x_test, y_test, label='Test Ground Truth')
                ax2[1].plot(x_test, y_test_pred, label='Test Prediction')

                """
                ax2[0].plot(x_train, y_train, label='Train Ground Truth', linestyle='none', marker='o')
                ax2[0].plot(x_train, y_train_pred, label='Train Prediction', linestyle='none', marker='o')
                ax2[1].plot(x_test, y_test, label='Test Ground Truth', linestyle='none', marker='o')
                ax2[1].plot(x_test, y_test_pred, label='Test Prediction', linestyle='none', marker='o')
                """
                ax2[0].legend(loc='upper right', prop=dict(size=12), numpoints=1)
                ax2[0].set_title(str(features))
                ax2[0].set_xlabel('time')
                ax2[0].set_ylabel('occupancy')
                ax2[0].grid()

                ax2[1].legend(loc='upper right', prop=dict(size=12), numpoints=1)
                ax2[1].set_title(str(features))
                ax2[1].set_xlabel('time')
                ax2[1].set_ylabel('occupancy')
                ax2[1].grid()

                print 'y_train \n', y_train
                print 'y_train_pred \n',y_train_pred
                print 'y_test \n',y_test
                print 'y_test_pred \n',y_test_pred

            if plotCM:
                # Plot confusion matrices in a separate window
                cm = confusion_matrix(y_train, y_train_pred)

                plt.matshow(cm)
                plt.title('Confusion matrix')
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.show()

        #plt.figure(idx)
        #plt.legend(loc='lower right', prop=dict(size=12))

        idx += 1

    if plotEE or plotPF or plotCM:
        plt.show()
