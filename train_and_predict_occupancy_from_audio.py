"""
TEST AND PREDICT OCCUPANCY FROM AUDIO
"""

import csv
import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 4})
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn import linear_model, svm
from sklearn.cross_validation import KFold, StratifiedKFold
from scipy.stats.stats import pearsonr
from pyearth import Earth as mras
from sklearn.preprocessing import StandardScaler as Standardizer
#WATCH OUT: pyearth as it gives <ValueError: Wrong number of columns in X> if you change re-fit a different dimension feature
#vector without instantiating a new pyearth class

try:
  import rpy2.robjects as robjects
  robjects.r('library(LambertW)')
except Exception, e:
  print 'error trying to import rpy2.robjects and loadling the LambertW library'
  print e

from pre_processing import standardize, normalize
from machine_learning_tools import predict, compute_rmse, compute_residuals_and_rsquared, regressionScikit, regressionStats, train_and_predict
from data_mining import *
from globales import *

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

############
#  PYTHON  #
############
plt.ion()

#############
#  METHODS  #
#############

def train_model_and_predict(train_data, test_data, location_id, door_count_placement_view_pair, train_period, test_period, features, kernel, pre_processing='', occupancy_range_pairs=None):
  """ Trains a GLM to predict occupancy given audio features. Plots ground truth, prediction and difference, saves statistical
      summary and saves the predicted values to a .csv file
  ARGS
    train_data: audio features (n_observations, n_features) <numpy array>
    test_data: audio features (n_observations, n_features) <numpy array>
    location_id: location_id of installation, eg '55' <int> or <str>
    door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
    train_period: (start_time, end_time) (<datetime>,<datetime>)
    test_period: (start_time, end_time) (<datetime>,<datetime>)
    features: list with label (keys) of features used, [<str>,<str>...]
    kernel: regression model or kernel with link from sklearn, statsmodels or pyearth regression models
    pre_processing: pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' with default values <str>
    occupancy_range_pairs: list of ranges to split the data for training independent regressions. , e.g. [(0,10), (10,30)]
  """
  if occupancy_range_pairs is None:
    occupancy_range_pairs = [(-np.infty, np.infty)]

  train_start = train_period[0]
  train_end = train_period[1]
  prediction_start = test_period[0]
  prediction_end = test_period[1]

  if all([isinstance(location_id, list), isinstance(door_count_placement_view_pair, list)]):
    train_location_id = location_id[0]
    train_placement_view_pair = door_count_placement_view_pair[0]
    test_location_id = location_id[1]
    test_placement_view_pair = door_count_placement_view_pair[1]
  else:
    train_location_id = location_id
    train_placement_view_pair = door_count_placement_view_pair
    test_location_id = location_id
    test_placement_view_pair = door_count_placement_view_pair

  train_X = train_data[0]
  train_Y = train_data[1]
  test_X = test_data[0]
  test_Y = test_data[1]

  #hopefully there's a better way to do this!
  kernel_module = kernel.__module__[:kernel.__module__.index('.')]

  models = []
  for idx in range(len(occupancy_range_pairs)):
    current_lower_bound = occupancy_range_pairs[idx][0]
    current_upper_bound = occupancy_range_pairs[idx][1]
    mask = ((train_Y > current_lower_bound) & (train_Y < current_upper_bound))
    train_X_set = train_X[mask]
    train_Y_set = train_Y[mask]

    print '\ntraining step'
    if kernel_module == 'sklearn' or kernel_module == 'pyearth':
      res = regressionScikit(train_X_set, train_Y_set, kernel)
      kernstr = str(kernel.__class__)
      kernstr = kernstr[kernstr.rindex('.'):]
    elif kernel_module == 'statsmodels':
      res = regressionStats(train_X_set, train_Y_set, kernel)
      kernstr = str(kernel.__class__)
      kernstr = kernstr[kernstr.rindex('.'):]
      linkstr = str(kernel.link.__class__)
      linkstr = linkstr[linkstr.rindex('.'):]
      kernstr += linkstr
    else:
      print 'could not find kernel module', kernel_module
      return None

    models.append(res)

    daystr = 'test [%s, %s) predict [%s, %s)' % (train_start, train_end, prediction_start,prediction_end)
    filename = namestructure(test_location_id, test_placement_view_pair, daystr, kernstr)
    features_str = str(features)
    filename += features_str
    filename.replace("', '", '_')

    if len(filename) > 192:
      filename = filename[0:192]
      filename += '_etc'

    if pre_processing is not None and pre_processing != '':
      filename += '_'+pre_processing

  predictions = []
  for idx in range(len(occupancy_range_pairs)):
    current_model = models[idx]
    current_lower_bound = occupancy_range_pairs[idx][0]
    current_upper_bound = occupancy_range_pairs[idx][1]
    mask = ((test_Y > current_lower_bound) & (test_Y < current_upper_bound))
    test_X_set = test_X[mask]
    test_Y_set = test_Y[mask]
    yfit = predict(res,test_X_set)

    SSresid, SStotal, rsq = compute_residuals_and_rsquared(yfit, test_Y_set)

    difference = abs(test_Y_set- yfit)

    errstr = 'Mean Square Error %.3f\nR-squared %.3f' % (np.mean(difference**2), rsq)

    plt.ion()
    count = len(test_Y_set)

    #generate time axis
    x_time = np.array(range(count))
    x_time = x_time / 4.0  + 8

    fullpath = RESULTS_FOLDER_GLOBAL+filename+statsext

    #######################
    #  test AND PREDICT  #
    #######################
    print '\nstatistical summary'
    if kernel_module == 'sklearn' or kernel_module == 'pyearth':
      sk_score = res.score(test_X_set, test_Y_set)
      try:
        print res.summary()
      except:
        print 'model has no res.summary()'
      try:
        print res.trace()
      except:
        print 'model has no res.trace()'

      #save statistical summary to .stats file
      with open(fullpath, 'w') as stats_file:
        print 'saving scikit-learn summary to \n %s' % fullpath
        stats_file.write('Using features %s\n' % features)
        stats_file.write(str(kernel)+'\n')
        stats_file.write(str(sk_score)+'\n')
        stats_file.write(errstr)
    elif kernel_module == 'statsmodels':
      sm_summary = res.summary()
      print sm_summary
      #perform F-Test
      A = np.identity(len(res.params))
      A = A[1:,:]
      f_test = res.f_test(A)
      print 'F_Test', f_test
      print 'Akaike Information Criterion', res.aic

      with open(fullpath, 'w') as stats_file:
        print 'saving statsmodels summary to %s' % fullpath
        stats_file.write('Using features %s' % features)
        stats_file.write(str(sm_summary)+'\n')
        stats_file.write(str(f_test)+'\n')
        stats_file.write('Akaike Information Criterion np.nan_to_num(res.aic) %d \n' % np.nan_to_num(res.aic))
        stats_file.write(errstr)

    textstr = daystr + '\n' + kernstr + '\n' + errstr

    ##############
    #  PLOTTING  #
    ##############
    fig, ax = plt.subplots(1, figsize=(8,2), dpi=300)

    ax.plot(x_time, test_Y_set, label='Ground Truth')
    ax.plot(x_time, yfit, label='Prediction')
    ax.plot(x_time, difference, label='Difference', ls='dashed')

    leg = ax.legend(loc='upper right', prop={'size':4})
    leg.get_frame().set_alpha(0.5)

    plot_header = str(features)
    plot_title = ''
    for i in range(len(features_str)/64):
      plot_title += features_str[64*i:64*(i+1)]+'\n'
    try:
      plot_title + features_str[64*(i+1):-1]
    except Exception, e:
      print 'plot title error', e

    ax.set_title(plot_title)
    ax.set_xlabel('time')
    ax.set_ylabel('occupancy')
    ax.grid()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', size=4, bbox=props)

    savepath = RESULTS_FOLDER_GLOBAL+filename
    savepath = savepath.replace("', '", '_')
    print 'Saving stats summary, plot and prediction to', savepath, 'each with their respective file extension .stats, .png, .csv'
    #truncate savepath
    if len(savepath) > 256:
      savepath = savepath[0:255]
      savepath += '_etc'

    fig.savefig(savepath+figext, bbox_inches='tight', dpi=fig.dpi)

    #####################
    #  SAVE PREDICTION  #
    #####################
    with open( savepath+".csv", "wb") as output_file:
      csvwriter = csv.writer(output_file, delimiter=',')
      for key, value in zip(x_time, yfit):
        csvwriter.writerow([key, value])
