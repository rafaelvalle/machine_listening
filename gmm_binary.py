import matplotlib as mpl
import pylab as plt
import numpy as np
from scipy import linalg
import itertools

from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler as Standardizer

from data_mining import *
from machine_learning_tools import compute_residuals_and_rsquared
from pre_processing import standardize, normalize

try:
  import rpy2.robjects as robjects
  robjects.r('library(LambertW)')
except Exception, e:
  print 'error trying to import rpy2.robjects and loadling the LambertW library'
  print e

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

cut = 90
plot_density = True
plot_ground_truth_and_prediction = True
plot_confusion_matrix = True

results = {}

#############
#  METHODS  #
#############
def make_ellipses(gmm, ax, n_std=3):
  color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
  for i, (mean, covar, color) in enumerate(zip(gmm.means_, gmm._get_covars(), color_iter)):
    try:
      #eigen_values, eigen_vectors = linalg.eigh(gmm._get_covars()[n][:2, :2])
      eigen_values, eigen_vectors = linalg.eigh(covar)
      indices = eigen_values.argsort()
      #u = eigen_vectors[indices[0]] / linalg.norm(eigen_vectors[indices[0]])
      #angle = np.degrees(np.arctan2(u[1], u[0]))
      angle = np.degres(np.arctan2(eigenvectors[1]/eigenvectors[0]))

      width, height = eigen_values[0], eigen_values[1]
      ell = mpl.patches.Ellipse(mean, width, height, 180 + angle, color=color)

      ell.set_clip_box(ax.bbox)
      ell.set_alpha(0.25)
      ax.add_artist(ell)
    except Exception, e:
      print 'error in make_ellipse', e

def draw_nstd_ellipses_classifier_and_means(gmm, means, ax, feature_idx, n_std=3):
  color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
  for i, (mean_pair, covar, color) in enumerate(zip(means, gmm._get_covars(), color_iter)):
    try:
      eigen_values, eigen_vectors = linalg.eigh(covar)
      eigen_values = np.sqrt(eigen_values)
      u = eigen_vectors[feature_idx] / linalg.norm(eigen_vectors[feature_idx])

      angle = np.degrees(np.arctan2(u[feature_idx+1], u[feature_idx]))
      for k in xrange(1, n_std):
        width, height = eigen_values[feature_idx]*k*2, eigen_values[feature_idx+1]*k*2
        ell = mpl.patches.Ellipse(mean_pair, width, height, 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.25)
        ax.add_artist(ell)
    except Exception, e:
      print 'error in make_ellipse', e

def compute_and_plot_bic_scores(location_id, door_count_placement_view_pair, start_time, end_time, features, pre_processing=''):
  global plot_density, plot_ground_truth_and_prediction, plot_confusion_matrix

  print 'pre-processing step'
  standardizer, normalizer, gaussianizer = None, None, None
  if 'standardize' in pre_processing:
     standardizer = Standardizer(copy=True, with_mean=True, with_std=True)
  if 'gaussianize' in pre_processing:
    gaussianizer = robjects.r('Gaussianize')

  print 'data mining step\n'
  data, _ = collectData(location_id, door_count_placement_view_pair, start_time, end_time, features, adjusted=False, pre_processor=standardizer)

  if standardizer is not None:
    data = standardizer.transform(data, copy=True)
  if gaussianizer is not None:
    print 'gaussianize'
    from rpy2.robjects.numpy2ri import numpy2ri
    robjects.conversion.py2ri = numpy2ri
    #this is slow and hacky, but takes care of NaN and infs(due to std ~ 0?)
    if data.ndim == 2 and data.shape[1] > 1 :
      data_transposed = data.T
      for idx in range(len(data_transposed)):
        try:
          rdata = gaussianizer(data_transposed[idx])
          gaussianized_data = np.array(rdata)
          gaussianized_data = gaussianized_data.reshape((len(gaussianized_data),))
          data[:,idx] = gaussianized_data
        except Exception, e:
          print 'gaussianization on idx %d failed' % idx 
          print e
    else:
      try:
        rdata = gaussianizer(data)
        data = np.array(rdata)
      except Exception, e:
        print 'gaussianization failed'
        print e

    gaussianizer.mean = data.mean(axis=1)
    gaussianizer.std = data.mean(axis=1)

  if pre_processing == 'normalize':
    print 'normalizing data'
    normalizer = normalize(data)

  lowest_bic = np.infty
  bic = []

  n_components_range = range(1, 10)

  cv_types = ['spherical', 'tied', 'diag', 'full']
  for cv_type in cv_types:
      for n_components in n_components_range:
          # Fit a mixture of gaussians with EM
          gmm = GMM(n_components=n_components, covariance_type=cv_type)
          gmm.fit(data)
          bic.append(gmm.bic(data))
          if bic[-1] < lowest_bic:
              lowest_bic = bic[-1]
              best_gmm = gmm

  bic = np.array(bic)
  color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
  clf = best_gmm
  bars = []

  # Plot the BIC scores
  spl = plt.subplot(2, 1, 1)
  for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
      xpos = np.array(n_components_range) + .2 * (i - 2)
      bars.append(plt.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], \
                         width=.2, color=color))

  plt.xticks(n_components_range)
  plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
  plt.title('BIC score per model')
  xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
      .2 * np.floor(bic.argmin() / len(n_components_range))
  plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
  spl.set_xlabel('Number of components')
  spl.legend([b[0] for b in bars], cv_types)

  # Plot the winner
  splot = plt.subplot(2, 1, 2)
  Y_ = clf.predict(data)

  for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_, color_iter)):
    eigv, eigw = np.linalg.eigh(covar)
    if not np.any(Y_ == i):
        continue

    #for count in range(0, data.shape[1] - data.shape[1]%2, 2):
    count = 0
    plt.scatter(data[Y_ == i, count], data[Y_ == i, count+1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(eigw[0][1], eigw[0][0])
    angle = 180 * angle / np.pi  # convert to degrees

    eigv *= 1024

    ell = mpl.patches.Ellipse(mean, eigv[0], eigv[1], 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

  #plt.xlim(-10, 10)
  #plt.ylim(-3, 6)
  #plt.xticks(())
  #plt.yticks(())
  plt.title('Selected GMM: full model, showing MFCC 0 and 1.')
  plt.subplots_adjust(hspace=.45, bottom=.08)
  plt.show()

  bic_filename= 'gmm_bic_score_features_%s_%s_cut_at_%d.png' % (str(features), pre_processing, cut)
  print 'saving bic scores to %s' % bic_filename
  plt.savefig(bic_filename, bbox_inches='tight')

def train_and_cluster_gmm(location_id, door_count_placement_view_pair, start_time, end_time, features, pre_processing='', BALANCE_DATA=False):
  global plot_density, plot_ground_truth_and_prediction, plot_confusion_matrix

  n_folds = 2

  print 'pre-processing step'
  standardizer, normalizer, gaussianizer = None, None, None

  if 'standardize' in pre_processing:
    standardizer = Standardizer(copy=True, with_mean=True, with_std=True)
  if 'gaussianize' in pre_processing:
    gaussianizer = robjects.r('Gaussianize')

  print 'data mining step'
  data, truth = collectData(location_id, door_count_placement_view_pair, start_time, end_time, features, adjusted=False, pre_processor=standardizer)

  if standardizer is not None:
    print 'standardizer'
    data = standardizer.transform(data, copy=True)

  if gaussianizer is not None:
    print 'gaussianize'
    from rpy2.robjects.numpy2ri import numpy2ri
    robjects.conversion.py2ri = numpy2ri
    #this is slow and hacky!
    if data.ndim == 2 and data.shape[1] > 1 :
      data_transposed = data.T
      for idx in range(len(data_transposed)):
        try:
          rdata = gaussianizer(data_transposed[idx])
          gaussianized_data = np.array(rdata)
          gaussianized_data = gaussianized_data.reshape((len(gaussianized_data),))
          data[:,idx] = gaussianized_data
        except Exception, e:
          print 'gaussianization on idx %d failed' % idx
          print e
    else:
      try:
        rdata = gaussianizer(data)
        data = np.array(rdata)
      except Exception, e:
        print 'gaussianization failed'
        print e

      gaussianizer.mean = data.mean(axis=1)
      gaussianizer.std = data.mean(axis=1)

  if pre_processing == 'normalize':
    print 'normalizing data'
    normalizer = normalize(data)

  #truth as 2 classes
  truth[truth <= cut] = 0
  truth[truth > cut] = 1
  truth_str = map(str, truth)

  #unbalanced targets affects, can't use StratifiedKFold, and, more important, GMM!, which assumes equal probability to all components
  print 'Sample size is', len(truth)
  folds = KFold(len(truth), n_folds=n_folds) #shuffle=True, random_state=4

  #for plotting
  fold_idx = 1

  for train_index, test_index in folds:
    X_train = data[train_index]
    y_train = truth[train_index]
    X_test = data[test_index]
    y_test = truth[test_index]
    n_components= len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=n_components,
              covariance_type=covar_type, params='wmc', init_params='wmc', n_iter=20))
               for covar_type in ['spherical', 'diag', 'tied', 'full'])

    n_classifiers = len(classifiers)

    for index, (covar_label, classifier) in enumerate(classifiers.iteritems()):
      #start classifier with known means
      classifier.means_ = np.array([X_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
      classifier.fit(X_train)

      y_train_pred = classifier.predict(X_train)
      y_train_pred = y_train_pred.astype(int)

      _, _, train_rsquared = compute_residuals_and_rsquared(y_train_pred, y_train)

      y_test_pred = classifier.predict(X_test)
      y_test_pred = y_test_pred.astype(int)

      _, _, test_rsquared = compute_residuals_and_rsquared(y_test_pred, y_test)

      if plot_ground_truth_and_prediction:
        x_train = np.array(range(len(y_train)))
        x_test = np.array(range(len(y_test)))

        fig_prediction, ax_prediction = plt.subplots(2, 1, figsize=(3 * n_classifiers / 2, 6))

        ax_prediction[0].step(x_train, y_train, label='Train Ground Truth')
        ax_prediction[0].step(x_train, y_train_pred, label='Train Prediction')

        ax_prediction[0].legend(loc='upper right', prop=dict(size=12), numpoints=1)
        ax_prediction[0].set_title('fold %d with %s and covariance %s' % (fold_idx, str(features), covar_label))
        #ax_prediction[0].set_xlabel('time')
        ax_prediction[0].set_ylabel('occupancy')
        ax_prediction[0].set_ylim([-1, np.max(truth) + 1])
        ax_prediction[0].yaxis.set_ticks(np.unique(y_train_pred))

        plt.yticks(np.unique(truth))
        ax_prediction[1].step(x_test, y_test, label='Test Ground Truth')
        ax_prediction[1].step(x_test, y_test_pred, label='Test Prediction')

        ax_prediction[1].legend(loc='upper right', prop=dict(size=12), numpoints=1)
        #ax_prediction[1].set_title(str(features) + ' with ' + covar_label)
        ax_prediction[1].set_xlabel('time')
        ax_prediction[1].set_ylabel('occupancy')
        ax_prediction[1].set_ylim([-1, np.max(truth) + 1])
        ax_prediction[1].yaxis.set_ticks(np.unique(y_test_pred))
        
        prediction_filename = 'gmm_prediction_%s_covariance_fold_%d_features_%s_%s_cut_at_%d.png' % (covar_label, fold_idx, str(features), pre_processing, cut)
        print 'saving prediction to %s' % prediction_filename 
        fig_prediction.savefig(prediction_filename, bbox_inches='tight', dpi=fig_prediction.dpi)
        plt.close("all")

      if plot_confusion_matrix:
        cm = confusion_matrix(y_train, y_train_pred)

        plt.matshow(cm)
        plt.title('Confusion matrix on fold %d with %s covariance' % (fold_idx, covar_label))
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        confusion_matrix_filename = 'confusion_matrix_%s_covariance_fold_%d_features_%s_%s_cut_at_%d.png' % (covar_label, fold_idx, str(features), pre_processing, cut)
        print 'saving confusion matrix to %s' % confusion_matrix_filename
        plt.savefig(confusion_matrix_filename)
        plt.close("all")

      if plot_density:
        plt.figure(fold_idx, figsize=(4 * data.shape[1] / 2, 8))
        plt.subplots_adjust(bottom=.01, top=0.95, hspace=.4, wspace=.1, left=.01, right=.99)

        for feature_idx in xrange(0, data.shape[1] - data.shape[1] % 2, 2):
          width = data.shape[1] / 10 
          if width == 0:
            width = 1

          fig_classification = plt.subplot(5, width, feature_idx/2 + 1)
          means_couple = classifier.means_[:,feature_idx:feature_idx+2]

          draw_nstd_ellipses_classifier_and_means(classifier, means_couple, fig_classification, feature_idx)

          #plot train with dots
          for n, color in enumerate('rg'):
            sample = data[truth == n]
            plt.scatter(sample[:, feature_idx], sample[:, feature_idx+1], 0.8, color=color, label=truth_str[n])

          #plot test with  es
          for n, color in enumerate('rg'):
            sample = X_test[y_test == n]
            plt.plot(sample[:, feature_idx], sample[:, feature_idx+1], 'x', color=color)

          y_train_pred = classifier.predict(X_train)
          train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
          plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy, transform=fig_classification.transAxes)

          y_test_pred = classifier.predict(X_test)
          test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
          plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy, transform=fig_classification.transAxes)

          #plt.xticks(())
          #plt.yticks(())
          plt.title('covariance %s with features %d and %d ' % (covar_label, feature_idx, feature_idx+1))

        classification_filename = 'gmm_classification_%s_covariance_fold_%d_features_%s_%s_cut_at_%d.png' % (covar_label, fold_idx, str(features), pre_processing, cut)
        print 'saving classification to %s' % classification_filename
        plt.savefig(classification_filename, bbox_inches='tight')
        plt.close("all")

    fold_idx += 1
