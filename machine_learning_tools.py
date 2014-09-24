"""
MACHINE LEARNING TOOLS
"""
import itertools
import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import statsmodels.api as sm
from sklearn import linear_model
from pyearth import Earth as mras

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

class TransitionMatrix:
  """Class for holding Transition Probabilities of ordered events"""

  def __init__(self, event_domain, ordered_events):
    """
      ARGS
        event_domain: set with domain of events (unique elements) <list>
        ordered_events: ordered list with events <list>
    """
    self.transition_matrix, self.event_counter = TransitionMatrix.build_transition_matrix_dict_from_ordered_events(event_domain, ordered_events)
    self.last_event = ordered_events[-1]

  @staticmethod
  def init_transition_matrix_dict(event_domain, start_value = 1):
    """Initialized a transition matrix dictionary to hold count of transitions
      ARGS
        event_domain:set with domain of events (unique elements) <list>
        start_value: start count for each transition <int>
      RETURN
        transition_matrix_dict: {(event,next_event):count}, e.g. {('a','b'):2}, or {(1,2):3}
    """
    transition_matrix_dict = {}
    permutations = [permutation for permutation in itertools.product(event_domain, repeat=2)]
    for permutation in permutations:
      transition_matrix_dict[permutation] = start_value
    return transition_matrix_dict

  @staticmethod
  def build_transition_matrix_dict_from_ordered_events(event_domain, ordered_events):
    """
      ARGS
        event_domain: set with domain of events (unique elements) <list>
        ordered_events: ordered list with events <list>
      RETURN
        transition_matrix_dict: with updated counts {(event,next_event):count}, e.g. {('a','b'):2}, or {(1,2):3}
    """
    counter = {}
    transition_matrix_dict = TransitionMatrix.init_transition_matrix_dict(event_domain, start_value = 1)

    for event in event_domain:
      counter[event] = len(event_domain)

    for idx in range(1, len(ordered_events)):
      current_event = ordered_events[idx-1]
      next_event = ordered_events[idx]
      transition_matrix_dict[current_event, next_event] += 1
      counter[current_event] += 1

    return transition_matrix_dict, counter

  def get_probability(self, current_event, next_event):
    """Returns probability of transition.
      ARGS
        current_event: valid object used as key
        next_event:valid object used as key
      RETURN
        probability of transition <float>
      RAISE:
        KeyError if transition does not exit. returns 0
    """
    try:
      return float(self.transition_matrix[(current_event, next_event)]) / self.event_counter[current_event]
    except KeyError, e:
      print e,'returning 0'
      return 0

  def update_transition_matrix_dict(self, ordered_events, include_last=True):
    """Updates the counts in the transition matrix dictionary
      ARGS
        ordered_events: ordered list with events <list>
        include_last: include transition from last stored event to current first event in ordered_events<bool>
    """
    if include_last:
      ordered_events = [self.last_event] + ordered_events
    event_counter = collections.Counter(ordered_events)

    for event, count in event_counter.items():
      if event not in self.event_counter:
        self.event_counter[event] = 0
      self.event_counter[event] += count

    for idx in range(1, len(ordered_events)):
      current_event = ordered_events[idx-1]
      next_event = ordered_events[idx]
      self.transition_matrix[current_event, next_event] += 1


def plot_learning_curve(data, labels, fold_max, kernel, title=''):
  """ Plots the learning curve. Also plots sorted label vales and predictions respective predictions
    ARGS
      data: audio features (n_observations, n_features) <numpy array>
      labels: labels (dependent variables) for the data set. <number array>
      fold_max: use up to fold-th of the data <int>
      kernel: regression model or kernel with link from sklearn, statsmodels or pyearth regression models
  """
  data_length = len(labels)
  indices = range(data_length)
  train_mse_values = []
  train_rsquared_values = []
  test_mse_values= []
  test_rsquared_values = []

  if fold_max < 1:
    raise Exception("Fold %d is smaller than 1" % fold_max)

  fold_sizes = range(fold_max, 0, -1)
  fig, ax_rows = plt.subplots(fold_max+1, 2, figsize=(16, 12))
  fig.subplots_adjust(hspace = 0.3, wspace = 0.2)
  fig.suptitle(title, fontsize = 16.0)

  for idx in range(len(fold_sizes)):
    fold = fold_sizes[idx]
    np.random.shuffle(indices)
    shuffle_indices = indices[:data_length/fold]
    n_datapoints = len(shuffle_indices)
    split_index = int(n_datapoints*0.7)

    train_data = data[shuffle_indices[:split_index]]
    test_data = data[shuffle_indices[split_index:]]

    train_labels = labels[shuffle_indices[:split_index]]
    test_labels = labels[shuffle_indices[split_index:]]

    res, train_fit = train_and_predict(kernel, train_data, train_labels)
    test_fit = predict(res, test_data)

    train_mse, _, train_rsquared = compute_residuals_and_rsquared(train_fit, train_labels)
    test_mse, _, test_rsquared = compute_residuals_and_rsquared(test_fit, test_labels)

    train_mse /= len(train_labels)
    test_mse /= len(test_labels)

    train_mse_values.append(train_mse)
    test_mse_values.append(test_mse)
    train_rsquared_values.append(train_rsquared)
    test_rsquared_values.append(test_rsquared)

    train_sorted_indices = np.argsort(train_labels)
    test_sorted_indices = np.argsort(test_labels)

    x_ticks_train = range(len(train_labels))
    x_ticks_test = range(len(test_labels))
    #plot labels, and prediction
    ax_rows[idx][0].set_xlabel('Ticks')
    ax_rows[idx][0].set_ylabel('Values')
    ax_rows[idx][0].plot(x_ticks_train, train_labels[train_sorted_indices], label='train labels')
    ax_rows[idx][0].plot(x_ticks_train, train_fit[train_sorted_indices], label='train predictions')
    ax_rows[idx][0].plot(x_ticks_test, test_labels[test_sorted_indices], label='test labels')
    ax_rows[idx][0].plot(x_ticks_test, test_fit[test_sorted_indices], label='test predictions')
    ax_rows[idx][0].legend(loc='lower right', prop={'size':8})

    #plot coefficient values
    n_coefficients = len(kernel.coef_)
    ax_rows[idx][1].set_xlabel('Coefficients')
    ax_rows[idx][1].set_ylabel('Values')
    ax_rows[idx][1].plot(range(n_coefficients+1), np.append(kernel.intercept_ , kernel.coef_), '.', color = 'steelblue', alpha = 0.5, label='coefficients')

  idx += 1
  n_samples = data_length/np.array(fold_sizes)
  #plot MSE values
  ax_rows[idx][0].set_xlabel('Sample size')
  ax_rows[idx][0].set_ylabel('MSE')
  ax_rows[idx][0].plot(n_samples, train_mse_values, 'o', color='blue', label='train mse')
  ax_rows[idx][0].plot(n_samples, test_mse_values, 'o', color='red', label='test mse')
  ax_rows[idx][0].legend(loc='lower right', prop={'size':8})

  #plot r_squared
  ax_rows[idx][1].set_xlabel('Sample size')
  ax_rows[idx][1].set_ylabel('R^2')
  ax_rows[idx][1].plot(n_samples, train_rsquared_values, 'o', color='blue', label='train r^2')
  ax_rows[idx][1].plot(n_samples, test_rsquared_values, 'o', color='red', label='test r^2')
  ax_rows[idx][1].legend(loc='lower right', prop={'size':8})

  if save:
    fig.savefig(str(title)+'.png', bbox_inches='tight')

  return fig, ax_rows

def train_and_predict(kernel, data, labels):
  """Train a regression model, predict and print statistical summary of model.
  ARGS
    kernel: regression model or kernel with link from sklearn, statsmodels or pyearth regression models
    data: independent variables for the train set. <number array>
    labels: dependent variables for the train set. <number array>
  RETURN
    res: trained model
    yfit: predicted values
  """
  #hopefully there's a better way to do this!
  kernel_module = kernel.__module__[:kernel.__module__.index('.')]

  #choose model wrapper and train model
  if kernel_module == 'sklearn' or kernel_module == 'pyearth':
    res = regressionScikit(data, labels, kernel)
    kernstr = str(kernel.__class__)
    kernstr = kernstr[kernstr.rindex('.'):]
  elif kernel_module == 'statsmodels':
    res = regressionStats(data, labels, kernel)
    kernstr = str(kernel.__class__)
    kernstr = kernstr[kernstr.rindex('.'):]
    linkstr = str(kernel.link.__class__)
    linkstr = linkstr[linkstr.rindex('.'):]
    kernstr += linkstr
  else:
    raise Exception("Could not find compatible kernel module %s" % str(kernel_module))

  yfit = predict(res, data)

  if kernel_module == 'sklearn' or kernel_module == 'pyearth':
    try:
      sk_score = res.score(data, labels)
      print 'sk_score', sk_score
    except Exception, e:
      print e
    try:
      print 'coefficients', res.coef_
      print 'intercept', res.intercept_
      print res.summary()
    except:
      print 'model has no res.summary()'
    try:
      print res.trace()
    except:
      print 'model has no res.trace()'
  elif kernel_module == 'statsmodels':
    print res.summary()
    #perform F-Test
    if len(res.params) > 1:
      A = np.identity(len(res.params))
      A = A[1:,:]
      f_test = res.f_test(A)
      print 'F_Test', f_test
    print 'Akaike Information Criterion %d' % res.aic
  return res, yfit

def train_and_predic_with_cross_validations(kernel, train_data, test_data, ground_truth, test_ground_truth=None):
  """Train a regression model, predict and print statistical summary of model.
  ARGS
    kernel: regression model or kernel with link from sklearn, statsmodels or pyearth regression models
    train_data: independent variables for the train set. <number array>
    test_data: independent variables for the test set. <number array>
    ground_truth: dependent variables for the train set. <number array>
    test_ground_truth(optional): dependent variables for the train set. <number array>
  RETURN
    res: trained model
    yfit: predicted values
  """
  #hopefully there's a better way to do this!
  kernel_module = kernel.__module__[:kernel.__module__.index('.')]

  #choose model wrapper
  if kernel_module == 'sklearn' or kernel_module == 'pyearth':
    res = regressionScikit
    kernstr = str(kernel.__class__)
    kernstr = kernstr[kernstr.rindex('.'):]
  elif kernel_module == 'statsmodels':
    res = regressionStats
    kernstr = str(kernel.__class__)
    kernstr = kernstr[kernstr.rindex('.'):]
    linkstr = str(kernel.link.__class__)
    linkstr = linkstr[linkstr.rindex('.'):]
    kernstr += linkstr
  else:
    raise Exception("Could not find compatible kernel module %s" % str(kernel_module))

  #train model
  res = res(train_data, ground_truth, kernel)
  yfit = predict(res, test_data)

  if kernel_module == 'sklearn' or kernel_module == 'pyearth':
    if test_ground_truth is not None:
      try:
        sk_score = res.score(test_data, test_ground_truth)
        print 'sk_score', sk_score
      except Exception, e:
        print e
    try:
      print 'coefficients', res.coef_
      print 'intercept', res.intercept_
      print res.summary()
    except:
      print 'model has no res.summary()'
    try:
      print res.trace()
    except:
      print 'model has no res.trace()'
  elif kernel_module == 'statsmodels':
    print res.summary()
    #perform F-Test
    if len(res.params) > 1:
      A = np.identity(len(res.params))
      A = A[1:,:]
      f_test = res.f_test(A)
      print 'F_Test', f_test
    print 'Akaike Information Criterion %d' % res.aic
  return res, yfit

def predict(model, x):
  """Predicts the dependent variable, given a GLM model from sklearn, statsmodels or pyearth.
  ARGS
    model: GLM model from sklearn, statsmodels or pyearth.
    x: {array-like, sparse matrix}, shape = (n_samples, n_features)
  RETURN
    yfit : numpy array with predicted values 
  """
  yfit = model.predict(x)

  if not isinstance(yfit, np.ndarray):
    yfit = np.array(yfit)

  return yfit

def compute_rmse(yfit, y):
  """Computes the root mean squared error"""
  return np.sqrt(np.mean((yfit-y)**2))

def compute_residuals_and_rsquared(yfit, y, n_coef=1):
  """Computes residuals and R-Squared from dependent variable (y) and prediction (yfit)
  ARGS
    y: numpy array with dependent variable
    yfit: numpy array with predicted values
  RETURNS: tuoke with SSresid, SStotal, rsq
    SSresid: residual sum of squares
    SStotal: total sum of squares (proportional to sample variance) <float>
    rsq: R-squared <float>
    n_coef: number of coefficients used. <int>
  """
  yresid = y - yfit;
  SSresid = np.sum(yresid**2)
  SStotal = np.sum((y - np.mean(y))**2)
  rsq = 1 - (SSresid * (len(y)-1))/(SStotal * (len(y) - n_coef))
  return SSresid, SStotal, rsq

def regressionStats(X, Y, kernel=None):
  """Wrapper for applying regression using statsmodels. 
  If no model is provided, choose Gaussian with a log link by default
  ARGS
    X: independent variables. {array-like, sparse matrix}, shape = (n_samples, n_features)
    Y: dependent variables <number array>
  RETURN
    res: trained model
  """
  print 'Statsmodel : Regression with kernel', kernel

  if kernel:
    kernel = sm.GLM(Y, X, family=kernel)
  else:
    kernel = sm.GLM(Y, X, family=sm.families.Gaussian(sm.families.links.log))

  res = kernel.fit()
  return res

def regressionScikit(X,Y,kernel=None):
  """Wrapper for applying regression using sklearn. 
  If no model is provided, choose OLS with fit intercept True by default
  ARGS
    X: independent variables. {array-like, sparse matrix}, shape = (n_samples, n_features)
    Y: dependent variables <number array>
  RETURN
    res: trained model
  """
  print 'Scikit : Regression with kernel', kernel
  if not kernel:
    kernel = linear_model.LinearRegression()

  kernel.fit(X,Y)
  return kernel

def poly(x, degree):
  """Generate orthonormal (orthogonal and normalized) polynomial basis functions from a vector.
  ARGS
    x : numerical data <numpy array>
    degree : degree of polynomial <int>
  RETURN
    Z : orthonormal polynomial basis functions <numpy array>
  """
  xbar = np.mean(x)
  X = np.power.outer(x - x.mean(), np.arange(0, degree + 1))
  Q, R = la.qr(X)
  diagind = np.subtract.outer(np.arange(R.shape[0]), np.arange(R.shape[1])) == 0
  z = R * diagind
  Qz = np.dot(Q, z)
  norm2 = (Qz**2).sum(axis = 0)
  Z = Qz / np.sqrt(norm2)
  Z = Z[:, 1:]
  return Z

def generate_polynomials(data, degrees):
  """Creates a dictionary of orthonormal polynomial basis functions from a vector.
    ARGS
      x : numerical data <numpy array>
      degrees : list with degrees of polynomial <int>
    RETURN
      Z : dictionary with orthonormal polynomial basis functions {degree:basis_functions} <dictionary>  
  """
  if isinstance(degrees, int):
    degrees = [degrees]
  if not isinstance(degrees,list):
    raise Exception("degrees must be an int or a list, got %s" % degrees)
  
  polys = {}
  
  for degree in degrees:
    polys[degree] = np.empty((data.shape[0], data.shape[1] * degree))
    for i in range(data.shape[1]):
      for k in range(degree):
        polys[degree][:,i*k + k] = np.pow(data[:,i], k)
        
  return polys

def generate_orthonormal_polynomials(data, degrees):
  """Creates a dictionary of orthonormal (orthogonal and normalized) polynomial basis functions from a vector.
  ARGS
    x : numerical data <numpy array>
    degrees : list with degrees of polynomial <int>
  RETURN
    Z : dictionary with orthonormal polynomial basis functions {degree:basis_functions} <dictionary>  
  """
  if isinstance(degrees, int):
    degrees = [degrees]
  if not isinstance(degrees,list):
    raise Exception("degrees must be an int or a list, got %s" % degrees)
  
  polys = {}
  
  for degree in degrees:
    polys[degree] = np.empty((data.shape[0], data.shape[1] * degree))
    for i in range(data.shape[1]):
      polys[degree][:,i*degree:(i+1)*degree] = poly(data[:,i], degree)
      
  return polys

def build_transition_matrix(ordered_events, max_label):
  """Builds a transition matrix from a sequence of events
  ARGS
    ordered_events: events must be encoded to numbers. <number arra>
  RETURN
    transition_matrix: matrix with transition probabilities <numpy 2d array>
  """

  label_index_dict = {}
  unique_labels = range(0, max_label)
  n_unique_labels = len(unique_labels)

  for idx_unique_label in range(n_unique_labels):
    label_index_dict[unique_labels[idx_unique_label]] = idx_unique_label

  transition_matrix = np.zeros((n_unique_labels, n_unique_labels))
  for event_idx in range(1, len(ordered_events)):
    current_event = ordered_events[event_idx - 1]
    next_event = ordered_events[event_idx]
    transition_matrix[label_index_dict[current_event], label_index_dict[next_event]] += 1
  return transition_matrix

def compute_viterbi_encoding(init_logs, trans_logs, emis_logs):
  """
  Returns the best path and the viterbi trellis of a sequence given params
  """

  n_states, n_observations = emis_logs.shape[0], emis_logs.shape[1]

  # Allocate dynamic programming table for Viterbi
  trellis = np.empty((n_states, n_observations))
  trellis.fill(-np.infty)

  # base case
  for k in range(n_states):
    trellis[k:, 0] = init_logs[k] + emis_logs[k,0]

  # columns 1 ... N-1
  for obs_idx in range(1, n_observations):
    for k in range(n_states):
      val = -np.infty
      for j in range(n_states):
        if trellis[j, obs_idx-1] + trans_logs[j, k] > val:
          val = trellis[j, obs_idx-1] + trans_logs[j, k]
      trellis[k,obs_idx] = emis_logs[k,obs_idx] + val

  return trellis.argmax(axis=0), trellis
