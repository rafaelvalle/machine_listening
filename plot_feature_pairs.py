"""
PLOT AUDIUO FEATURE PAIRS
"""

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

def plot_features(location_id, door_count_placement_view_pair, start_time, end_time, features, pre_processing=''):
  """Plots features in pairs by incremental index, e.g. (0,1), (2,3)...
  ARGS
    location_id: location_id of installation, eg '55' <int> or <str>
    door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
    start_time: for prunning. time object with hour and minute time <time>
    end_time: for prunning. time object with hour and minute time <time>
    features: list with label (keys) of features used, [<str>,<str>...]
    pre_processing: pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' with default values <str>
  """
  print 'pre-processing step'
  standardizer, normalizer, gaussianizer = None, None, None
  dict_of_features = {}

  for feature in features :
    for processing in ['standardize']:
      pre_processing = [processing]
      if 'standardize' in pre_processing:
        standardizer = Standardizer(copy=True, with_mean=True, with_std=True)
      if 'gaussianize' in pre_processing:
        gaussianizer = robjects.r('Gaussianize')

      print 'data mining step'
      data, _ = collectData(location_id, door_count_placement_view_pair, start_time, end_time, [feature], adjusted=False, pre_processor=standardizer)

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

      if 'normalize' in pre_processing:
        print 'normalizing data'
        normalizer = normalize(data)

      dict_of_features[feature+'_'+pre_processing[0]] = data

  plt.figure(1, figsize=(4 * data.shape[1] / 2, 8))
  plt.subplots_adjust(bottom=.05, top=0.95, hspace=.4, wspace=.1, left=.05, right=.95)
  colors = ['r', 'g', 'b', 'c', 'm']

  for feature_idx in xrange(0, data.shape[1] - data.shape[1] % 2, 2):
    width = data.shape[1] / 10
    if width == 0:
      width = 1

    fig_classification = plt.subplot(5, width, feature_idx/2 + 1)

    #plot train with dots
    color_idx = 0
    for feature_label, values in dict_of_features.items():
      plt.scatter(values[:, feature_idx], values[:, feature_idx+1], 0.8, color=colors[color_idx], label=feature_label)
      color_idx = (color_idx + 1) % len(colors)
    plt.title('features %d and %d ' % (feature_idx, feature_idx+1))
    legend = plt.legend(loc=4)
    legend.get_frame().set_alpha(0.5)
