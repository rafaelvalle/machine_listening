import matplotlib, shelve, sys
import numpy as np
import pylab as plt
import cPickle as pkl

from sklearn.mixture import GMM
from sklearn.metrics import confusion_matrix
from sklearn.utils.extmath import logsumexp
from data_mining import *
from machine_learning_tools import build_transition_matrix, compute_viterbi_encoding

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def get_params_from_gmm(gmm):
  """Returns in the form of a dictionary the params of a gmm object
  ARGS
    gmm: GMM object from sklearn <sklearn.mixture.GMM>
  RETURN
    gmm_params: dictionary with means, covars and weights of a GMM. <dictionary>
  """
  gmm_params = {}
  gmm_params['means'] = gmm.means_
  gmm_params['covars'] = gmm.covars_
  gmm_params['weights'] = gmm.weights_
  return gmm_params

def train_GMM(n_components, cv_type, feats, n_init=1, n_iter=500, thresh=0.01):
  """Returns model and parameters estimated with the expectation-maximization algorithm.
  ARGS
    n_components: number of mixture components
    cv_type: covariance type
    feats: list of n_features-dimensional data points. row corresponds to a data point <numerical array>
    n_iter: number of EM iterations
    thresh: convergence threshold (Log-Likelihood)
  RETURN
    gmm_params: dictionary with means, covars and weights of a GMM. <dictionary>
    gmm: GMM object from sklearn <sklearn.mixture.GMM>
  """
  gmm = GMM(n_components=n_components, covariance_type=cv_type, n_init=n_init, n_iter=n_iter, thresh=thresh)
  print 'training gmm'
  try:
    gmm.fit(feats)
  except Exception, e:
    print 'train_GMM:',e
    return None, None

  gmm_params = get_params_from_gmm(gmm)

  return gmm_params, gmm

def create_binned_data_dictionary(data, data_labels, unique_labels):
  """Bins data according to their labels
  ARGS
    data: list of n_features-dimensional data points. row corresponds to a data point <numerical array>
    data_labels: label of each data point <number array>
    unique_labels: iterable representing label set <list> or <set>...
  RETURN
    binned_data: dictionary where label is key and label's datapoints is value <dictionary>
  """
  # ... create binned data dictionary { bin label : features from label } ...
  binned_data = {}
  for bin_label in unique_labels:
    # expects numpy array
    if data[data_labels == bin_label].size:
      binned_data[str(bin_label)] = reshape_data(data[data_labels == bin_label])
  return binned_data

def adapt_gmms_and_save(ubm, binned_feats, savepath='adapted_gmms'):
  """
  ARGS
    ubm: Universal Background Model GMM <sklearn.mixture.GMM>
    binned_feats: dictionary where label is key and label's datapoints is value <dictionary>
    savepath: path to save adapted GMMs
  """
  ubm_params = get_params_from_gmm(ubm)
  n_components = ubm.n_components
  cv_type = ubm.covariance_type
  adapted_gmms = adapt_UBM_to_occupancy_bins(n_components, cv_type, binned_feats, ubm_params)
  with open(savepath,'wb') as fp:
    pkl.dump(adapted_gmms, fp, protocol=-1)

def adapt_UBM_to_occupancy_bins(n_components, cv_type, occupancy_bins_feats, ubm_params, include_params=False):
  """
  ARGS
    n_components: number of mixture components
    cv_type: covariance type
    occupancy_bins_feats: list of n_features-dimensional data points. row corresponds to a data point <numerical array>
    ubm_params: dictionary with means, covars and weights of a GMM. <dictionary>
  RETURN
    gmms_adapted_and_params: dictionary where label is key and label's datapoints is value <dictionary>
  """
  # ... Adapt UBM ...
  gmms_adapted_and_params = {}

  # ... create dictionary with adapted GMM and params for each occupancy bin { binned label : (gmm object, { param: value }) }...
  for occupancy_bin_label, features in occupancy_bins_feats.items():
      print 'adapting occupancy_bin_label', occupancy_bin_label
      adapted_gmm, adapted_params  = adapt_UBM(n_components, cv_type, ubm_params, features)
      if include_params:
        gmms_adapted_and_params[occupancy_bin_label] = (adapted_gmm, adapted_params)
      else:
        gmms_adapted_and_params[occupancy_bin_label] = adapted_gmm
  return gmms_adapted_and_params

def adapt_UBM(n_components, cv_type, ubm_params, data):
  """
  ARGS
    n_components: number of mixture components
    cv_type: covariance type
    ubm_params: dictionary with means, covars and weights of a GMM. <dictionary>
    data: list of n_features-dimensional data points. row corresponds to a data point <numerical array>
  RETURN
    adapted_gmm: GMM adapted from the UBM <sklearn.mixture.GMM>
    adapted_means: means of the adapted GMM <number array>
  """
  adapted_gmm = GMM(n_components=n_components, covariance_type=cv_type, n_iter=1)

  # ... Start GMM with UBM parameters ... 
  adapted_gmm.means_ = ubm_params['means']
  adapted_gmm.covars_ = ubm_params['covars']
  adapted_gmm.weights_ = ubm_params['weights']
  adapted_gmm.fit(data)

  #  ... Adapt params ( mean only for now ) ... 
  new_means = adapted_gmm.means_
  new_covars = adapted_gmm.covars_
  new_weights = adapted_gmm.weights_
  T = data.shape[0]
  adapted_means = adapt_means(ubm_params['means'],\
                              ubm_params['covars'],\
                              ubm_params['weights'],\
                              new_means, new_weights, T)

  #  ... Update means of adapted GMM ...
  adapted_gmm.means_ = adapted_means

  return adapted_gmm, adapted_means

def adapt_means(ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T, relevance_factor=16):
  """Adapts means from an Universal Background Model
  described in "Speaker Verification Using Adapted Gaussain Mixture Models, Reynolds, A. et al"
  ARGS
    ubm_means: means of the ubm <number array>
    ubm_covars: covariances of the ubm <number array> 
    ubm_weights: weights of the ubm <number array>
    new_means: means adapted from the ubm <number array>
    new_weights: weights adapted from the ubm <number array>
    T: scaling factor, number of samples
    relevance_factor: factor for scaling the adapted means
  RETURN
    return_means: adapted means <number array>
  """
  n_i = new_weights*T
  alpha_i = n_i/(n_i+relevance_factor)
  new_means[np.isnan(new_means)] = 0.0
  return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T

  # Normalization from Ekaterina's pycast code
  #return_means = (np.sqrt(ubm_weights)*(1/np.sqrt(ubm_covars.T))*return_means.T).T
  return return_means

def adapt_parameters(ubm_means, ubm_covars, ubm_weights, new_means, new_covars, new_weights, T, relevance_factor=16):
  #TODO
  return None

def reshape_data(data, n_coeffs=None, n_observations=None):
  """Unrolls and reshapes a 2d numpy array of arrays in the form [[[]], [[]]] to become [[],[]]
  This is necessary for MFFC and FFT in the recent implementation
  ARGS
    data: rolled array <array>
    n_coeffs: number of coefficients or frequency bins, columns <int>
    n_observations: number of observations, rows <int>
  RETURN
    data_reshaped: reshaped data (n_observations, n_coeffs) <numpy 2d array>
  RAISE
    an Exception if some problem is raised
  """
  if n_coeffs is None or n_observations is None:
    if len(data.shape) == 1:
      n_observations, n_coeffs = len(data[0]), len(data)
    else:
      n_observations, n_coeffs = data.shape
  data_reshaped = []
  for coeff_num in range(n_coeffs):
    if len(data.shape) == 1:
      data_reshaped.append(data[coeff_num])
    elif any((isinstance(data[:,coeff_num][0], list), isinstance(data[:,coeff_num][0], np.ndarray))):
      #array of arrays
      data_reshaped.append(data[:,coeff_num][0])
      for time_bin in range(1, n_observations):
        data_reshaped[coeff_num] = np.concatenate((data_reshaped[coeff_num], data[:,coeff_num][time_bin]))
    else:
      #array
      data_reshaped.append(data[:,coeff_num])
  data_reshaped = np.array(data_reshaped).T
  return data_reshaped

def window_data(data, window_size, offset=8):
  """windows data without unrolling it
  This is necessary for MFFC and FFT in the recent implementation
  ARGS
    data: rolled array <array>
    window_size: window size <int>
    offset: offset for window start <int>
  RETURN
    windowed_features: windowed_data <numpy array>
  RAISE
    an Exception if some problem is raised
  """
  windowed_features = []
  rows, cols = data.shape
  for row in xrange(rows):
    timebin = []
    if row < rows -1:
      for col in xrange(cols):
        timebin.append(np.concatenate((data[row][col][-window_size:], data[row+1][col][offset:window_size+offset])))
    else:
      for col in xrange(cols):
        timebin.append(data[row][col][-window_size:])
    windowed_features.append(timebin)
  return np.array(windowed_features)

def compute_class_interpolation_given_log_likelihoods(log_likelihoods):
  """Interpolates (linear) between given models by using their log likelihoods and numerical labels.
  Suppose you have 2 GMMs trained on [1,10], [11,20] people data respectively.
  Given LL(X|model) = [-4,-4], this method returns a predicted_values value that is a combination of the weighted bins of each model.
  bin 0 = [1,10], bin 1 =[11,20]
  |-4   -4
  |
  |_____________
    0    1
  In this case, the method estimates the predicted bin value to be 1.5,
  which estimates occupancy to be 10.5 people, the median of (1,2...,20)

  ARGS
    log_likelihoods: row as observation, column as log-likelihood of class. <2d numpy array>
    adaptations are required if distance between model labels is not 1. 
  RETURN
  computed_classes: predicted labels after interpolation <number array>
  """
  # compute inverse of log likelihood to obtain marginal probability of model
  probabilities = np.array(np.power(np.e, log_likelihoods), ndmin=2)
  
  # normalize probabilities
  row_sums = probabilities.sum(axis=1)
  probabilities = probabilities / row_sums[:, np.newaxis]

  # get idx of classes with highest probability and set their index to be the starting class index
  # does not work for cases where distances between class bins are not the same e.g. {1, 3, 4} 
  idx_max_probabilities = np.argmax(probabilities, axis=1)
  computed_classes = np.copy(idx_max_probabilities).astype(float)

  #iterate through observations
  for idx in range(probabilities.shape[0]):
    # get label_indices list and erase highest probability from label_indices list
    observation_probabilities = probabilities[idx]
    idx_max_probability_of_obs = idx_max_probabilities[idx]
    index_of_classes = range(len(observation_probabilities))
    del index_of_classes[idx_max_probability_of_obs]

    # compute effects of other probabilities onto winning class value/bin
    for kdx in index_of_classes:
      computed_classes[idx] += observation_probabilities[kdx] * (kdx - idx_max_probability_of_obs)

  return computed_classes

def time_combine_likelihoods_with_logsumexp(log_likelihoods, bin_start_list):
  """Combines log-likelihoods of features (cols) over time (rows) to reach n temporal bins
  ARGS
    log_likelihoods: matrix
    bin_start_list: has the starting index of each bin <int>
  RETURN
    time_binned_log_likelihoods: time binned log-likelihoods summed with logsumexp <tb_LLnumpy array>
  """
  if len(bin_start_list) == 1:
    return logsumexp(log_likelihoods)
  n_bins = len(bin_start_list)
  n_columns = log_likelihoods.shape[1]
  time_binned_log_likelihoods = np.empty((n_bins,n_columns))
  for idx in range(1, len(bin_start_list)-1):
    bin_start = bin_start_list[idx-1]
    bin_end = bin_start_list[idx]
    time_binned_log_likelihoods[idx] = logsumexp(log_likelihoods[bin_start:bin_end])
  idx += 1
  time_binned_log_likelihoods[idx] = logsumexp(log_likelihoods[idx:])  
  return time_binned_log_likelihoods


def compute_log_class_interpolation_given_log_likelihoods(log_likelihoods):
  """MUST BE WRITTEN"""

def predict_from_trained_gmms_by_bin(trained_gmms, all_data, all_labels):
  """Given data, labels and GMMs, computes the log-likelihods and saves them to a dict 
  ARGS
    trained_gmms: dictionary where key is GMM label and value is GMM
    all_data: audio features (n_observations, n_features) <numpy array>
    all_labels: label of each data point <number array>
  RETURN
    predicted_labels: dictionary where key is data label and values is log-likelihood of data
  """

  # ... divide labels into occupancy bins ...
  unique_labels = np.unique(all_labels)
  n_labels = len(unique_labels)

  # ... create train and test data dictionary { binned label : features from label } ...
  test_feats = {}
  for bin_label in unique_labels:
    test_feats[bin_label] = reshape_data(all_data[all_labels == bin_label])

  #compute log-likelihoods
  log_likelihoods_per_label = {}
  for label in unique_labels:
    for gmm_label in sorted(trained_gmms.keys(), key=int):
      trained_gmm = trained_gmms[gmm_label]
      gmm_log_likelihoods = np.array(trained_gmm.score(test_feats[label]), ndmin=2).T
      if label not in log_likelihoods_per_label:
        log_likelihoods_per_label[label] = gmm_log_likelihoods
      else:
        log_likelihoods_per_label[label] = np.append(log_likelihoods_per_label[label], gmm_log_likelihoods, axis=1)

  predicted_labels = {}
  for label, log_likelihoods in log_likelihoods_per_label.items():
    predicted_labels[label] = compute_class_interpolation_given_log_likelihoods(log_likelihoods)

  return predicted_labels

def compute_log_likelihoods_from_trained_gmms(trained_gmms, all_data):
  """Given GMMs and data, computes the log-likelihoods
  ARGS
    trained_gmms: dictionary where key is GMM label and value is GMM
    all_data: audio features (n_observations, n_features) <numpy array>
  RETURN
    predicted_labels: dictionary where key is GMM label and values is log-likelihood of data
  """
  log_likelihoods = None
  for gmm_label in sorted(trained_gmms.keys(), key=int):
    trained_gmm = trained_gmms[gmm_label]
    gmm_log_likelihoods = np.array(trained_gmm.score(all_data), ndmin=2).T
    if log_likelihoods is None:
      log_likelihoods = gmm_log_likelihoods
    else:
      log_likelihoods = np.append(log_likelihoods, gmm_log_likelihoods, axis=1)
  return log_likelihoods

def predict_from_trained_gmms(trained_gmms, all_data):
  """Given GMMs and data, computes the log-likelihoods
  ARGS
    trained_gmms: dictionary where key is GMM label and value is GMM
    all_data: audio features (n_observations, n_features) <numpy array>
  RETURN
    predicted_labels: dictionary where key is GMM label and values is log-likelihood of data
  """

  #compute log-likelihoods
  log_likelihoods = None

  for gmm_label in sorted(trained_gmms.keys(), key=int):
    trained_gmm = trained_gmms[gmm_label]
    gmm_log_likelihoods = np.array(trained_gmm.score(all_data), ndmin=2).T
    if log_likelihoods is None:
      log_likelihoods = gmm_log_likelihoods
    else:
      log_likelihoods = np.append(log_likelihoods, gmm_log_likelihoods, axis=1)
  predicted_labels = compute_class_interpolation_given_log_likelihoods(log_likelihoods)

  return predicted_labels

def adapt_predicted_values_to_label_count(predicted_values, n_labels):
  """Rounds predicted values to bin labels
  ARGS
    predicted_values <numpy array>
    n_labels: number of unique labels
  RETURN
    adapted_predicted_values: list with adapted predicted values
  """
  predicted_values_count = len(predicted_values)
  rest = predicted_values_count % n_labels
  if rest != 0:
    predicted_values = predicted_values[:-rest]
  adapted_predicted_values = []
  step_size = predicted_values_count/n_labels
  label_indices = np.arange(0, predicted_values_count - step_size, step_size )
  for idx in label_indices:
    adapted_predicted_values.append(predicted_values[idx:idx+step_size].mean())
  adapted_predicted_values = np.array(adapted_predicted_values)
  return adapted_predicted_values

def plot_predicted_values_per_bin_bar_graph(predicted_labels):
  """Bar plot of correct and wrong predicted values per bin.
  ARGS
    predicted_values: list where idx represents bin label and values predicted labels <numpy array>
  """
  labels = predicted_labels.keys()
  N = len(labels)
  ind = np.arange(N)  # the x locations for the groups
  width = 0.35       # the width of the bars
  correct = []
  wrong = []
  for idx in ind:
    label = labels[idx]
    correct.append(len(predicted_labels[label].astype(int)[predicted_labels[label].astype(int)==label]))
    wrong.append(len(predicted_labels[label]) - correct[idx])
  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, correct, width, color='g')
  rects2 = ax.bar(ind+width, wrong, width, color='R')
  ax.set_ylabel('Number of observations')
  ax.set_title('Correct and Wrong observations per bin')
  ax.set_xticks(ind+width)
  ax.set_xticklabels( labels )
  ax.legend( (rects1[0], rects2[0]), ('Correct', 'Wrong') )

def generate_balanced_folds(labels, n_folds=1):
  """
  MUST REWRITE
  """
  if isinstance(n_folds, int):
    unique_labels = np.unique(labels)
    folds_semi_balanced = [[] for x in xrange(n_folds)]
    for label in unique_labels:
      label_indices = np.where(labels==label)[0].tolist()
      step_size = len(label_indices)/n_folds
      for idx in range(n_folds - 1):
        folds_semi_balanced[idx] += label_indices[idx*step_size:(idx+1)*step_size]
      folds_semi_balanced[n_folds - 1] += label_indices[(n_folds-1)*step_size:]
    #hackintosh
    folds_semi_balanced = [(folds_semi_balanced[0], folds_semi_balanced[1])]
    #for idx in range(len(folds_semi_balanced)):
      #len_half_fold = len(folds_semi_balanced[idx])/2
      #np.random.shuffle(folds_semi_balanced[idx])
      #folds_semi_balanced[idx] = (folds_semi_balanced[idx][:len_half_fold], folds_semi_balanced[idx][len_half_fold:])
    return folds_semi_balanced
  else:
    raise Exception("Number of folds %s must be of type int and bigger than 1" % str(n_folds))

def plot_confusion_matrix(y, y_predict, title=None, normalize=True):
  """Plots a confusion matrix given labels and predicted labels
  ARGS
    y: ground truth labels <int array>
    y_predict: predicted labels <int array>
  """
  cm = confusion_matrix(y, y_predict)
  if normalize:
    cm = (cm+0.0)/cm.max()

  plt.matshow(cm)
  if title:
    plt.title(title)
  else:
    plt.title('Confusion matrix')
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

def estimate_best_bic_score(all_data, all_labels, n_components_list, cv_types = ['diag'], folds = None, savepath='ubm_adapted_gmm_data'):
  """Estimates the best bic score of UBM adapted GMMs per class bin and saves them to file with the following structure:
    gmms_data['bics'] =  bics (dict with all BIC values of each class bin)
    gmms_data['bics_min'] =  bics_min (dict with smallest BIC per class bin)
    gmms_data['best_gmms'] =  best_gmms (dict with the best GMM objects)
    ARGS
      all_data: audio features (n_observations, n_features) <numpy array>
      all_labels: list with labels of datapoints <number array>
      n_components_list: list with the number of mixture componets to try <number array>
      cv_type: list with covariance types. accepts 'spherical', 'tied', 'diag', 'full' <string array>
      folds: train/test indices to split data in train and test sets
      savepath: filepath to save data <str>    
  """

  # ... divide labels into occupancy bins ...
  unique_labels = np.unique(all_labels)
  n_labels = len(unique_labels)

  if folds is None:
    label_indices = range(len(all_labels))
    folds = [(label_indices,label_indices)]

  for train_index, test_index in folds:
    X_train = all_data[train_index]
    y_train = all_labels[train_index]
    X_test = all_data[test_index]
    y_test = all_labels[test_index]

    # ... create train and test data dictionary { binned label : features from label } ...  
    train_feats = create_binned_data_dictionary(X_train, y_train, unique_labels)
    test_feats = create_binned_data_dictionary(X_test, y_test, unique_labels)

    # ... arrays for storing best GMMs and BIC scores for each GMM bin ...
    bics = {}
    bics_min = {}
    best_gmms = {}
    gmms_data= {}
    # ... iterate through covariance types and number of components ...
    for cv_type in cv_types:
      for n_components in n_components_list:
        print 'evaluating BIC score for %s covariance and %d n_components' % (cv_type, n_components)

        # ... Train the Universal Background Model with all data ...
        ubm_params, _ = train_GMM(n_components, cv_type, all_data)

        # ... Adapt the Universal Background Model each occupancy bins ...
        gmms_adapted_and_params = adapt_UBM_to_occupancy_bins(n_components, cv_type, train_feats, ubm_params)

        # ... compute BIC scores for each GMM bin ...
        for bin_label, gmm in gmms_adapted_and_params.items():
          bic_score = gmm.bic(test_feats[bin_label])
          if bin_label not in bics:
            bics[bin_label] = []
            bics[bin_label].append(bic_score)
            bics_min[bin_label] = bic_score
            best_gmms[bin_label] = gmm
          else:
            bics[bin_label].append(bic_score)
            if bic_score < bics_min[bin_label]:
              print 'bic is now ', bic_score, ' was ', bics_min[bin_label]
              best_gmms[bin_label] = gmm
              bics_min[bin_label] = bic_score

        #save gmms and related data to disk
        gmms_data['bics'] =  bics
        gmms_data['bics_min'] =  bics_min
        gmms_data['best_gmms'] =  best_gmms
        print 'saving to', savepath
        with open(savepath,'wb') as fp:
          pkl.dump(gmms_data, fp, protocol=-1)
        print 'best GMMs so far:', best_gmms
  return (bics, bics_min, best_gmms)

def couple_gmms_to_ubm(gmm_data_filepath, ubm_key='ubm_gmm'):
  """Couples GMMs into a single GMM to create an UBM by coupling features and scaling weights. 
  Saves the UBM to the same file 
  MUST DEBUG SCALING
  ARGS
    gmm_data_filepath: filepath of shelve saved dictionary where the GMMs are saved with key 'best_gmm' <str>
    ubm_key: key used to save the UBM to the file <str>
  """
  gmm_data = shelve.open(gmm_data_filepath, flag='w')
  ubm_weights = None
  ubm_means = None
  ubm_covars = None
  ubm_cv_type = None
  for label in sorted(gmm_data.keys(), key=int):
    gmm = gmm_data[label]
    print 'adding label', label
    if ubm_covars is None:
      ubm_weights = gmm['best_gmm'].weights_
      ubm_means = gmm['best_gmm'].means_
      ubm_covars = gmm['best_gmm'].covars_
      ubm_cv_type = gmm['best_gmm'].covariance_type
    else:
      ubm_weights = np.append(ubm_weights, gmm['best_gmm'].weights_, axis=0)
      ubm_means = np.append(ubm_means, gmm['best_gmm'].means_, axis=0)
      ubm_covars = np.append(ubm_covars, gmm['best_gmm'].covars_, axis=0)
  n_components = ubm_means.shape[0]
  
  #scale weights to one
  ubm_weights = ubm_weights/ubm_weights.sum()

  ubm_gmm = GMM(n_components=n_components, covariance_type=ubm_cv_type)
  ubm_gmm.weights_ = ubm_weights 
  ubm_gmm.means_ = ubm_means
  ubm_gmm.covars_ = ubm_covars
  ubm_gmm.converged_ = True
  try:
    gmm_data[ubm_key] = ubm_gmm
    gmm_data.close()
  except Exception, e:
    print e

def train_and_save_gmm_with_best_bic(all_data, label, n_components_list, cv_types = ['diag'], n_init = 1, n_iter=1000, folds = None, save_path='ubm_data'):
  """Given data and params, saves the GMM with the best BIC score using shelve and label as key
    ARGS
      all_data: audio features from data class (n_observations, n_features) <numpy array>
      label: label that represents data class <str>
      n_components_list: list with the number of mixture componets to try <number array>
      cv_type: list with covariance types. accepts 'spherical', 'tied', 'diag', 'full' <string array>
      folds: train/test indices to split data in train and test sets
      savepath: filepath to save data <str> 
  """
  n_frames = all_data.shape[0]
  bics = []
  bic_min = np.infty
  best_gmm = None
  gmm_data= {}
  
  indices = range(n_frames)
  np.random.shuffle(indices)
  train_last_idx = int(n_frames * 0.7)

  X_train = all_data[indices[:train_last_idx]]
  X_test = all_data[indices[train_last_idx:]]
  for cv_type in cv_types:
    for n_components in n_components_list:
      print 'evaluating BIC score for %s covariance and %d n_components' % (cv_type, n_components)
      _, gmm = train_GMM(n_components, cv_type, X_train, n_init=n_init, n_iter=n_iter, thresh=0.01)
      if gmm is not None:
        bic_score = gmm.bic(X_test)
        bics.append(bic_score)
        if bic_score < bic_min:
          print 'BIC is now ', bic_score, ' was ', bic_min
          best_gmm = gmm
          bic_min = bic_score
        gmm_data['bics'] =  bics
        gmm_data['bic_min'] =  bic_min
        gmm_data['best_gmm'] =  best_gmm
        file = shelve.open(save_path, flag='c')
        try:
          file[str(label)] = gmm_data
          file.close()
        except Exception, e:
          print e
        print best_gmm
      else:
        print 'breaking from for loop'
        break
  return best_gmm
