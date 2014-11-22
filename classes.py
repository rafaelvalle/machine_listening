"""
CLASSES
K-Means:
  Wrapper class for initializing and storing K-Means estimator

Statistics:
  Provides incremental algorithms for:
    mean, variance, standard deviation
    computeds delta, growth

FeatureSpace:
  Stores a collection of TimeSeries objects and information about location id,
      door count placement view pair, start time and timestep

TimeSeries:
  Stores a sequence of Sample objects as a time series, their statistical information
   and audio properties
  Includes methods for plotting TimeSeries data

Samples:
  Stores audio features, FFT information and start time timestamp of an audio file
  Includes methods for plotting Samples data
"""

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pre_processing import remove_inf_and_nan

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

#############
#  CLASSES  #
#############
class K_Means:
  """Container for K-Means estimators"""
  def __init__(self):
    self.estimators = {}

  def add_estimator(self, init_methods, n_clusters, n_init=10):
    """Adds K-means based estimators given parameters
      ARGS
        init_methods: Method for initialization (from sklearn)
          'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
          'random': choose k observations (rows) at random from data for the initial centroids
          PCA: PCA object from sklearn
        n_clusters: number os clusters <int>
        n_init: number of time the k-means algorithm will be run with different centroid seeds <int>
    """
    if isinstance(init_methods, str) or isinstance(init_methods,PCA):
      init_methods = [init_methods]
    for init_method in init_methods:
      if type(init_method) == PCA:
        self.estimators['pca'] = KMeans(init=init_method.components_, n_clusters=n_clusters, n_init=n_init)
      else:
        if init_method == 'k-means++' or init_method=='random':
          self.estimators[init_method] = KMeans(init=init_method, n_clusters=n_clusters, n_init=n_init)
        else:
          raise Exception("add_estimator: initialization method %s not supported" % init_method)

class Statistics:
  """ class for storing and incrementally computing statistical information of data""" 
  def __init__(self):
    self.obs = 0.
    self.var = None
    self.std = None
    self.mean = None

  @staticmethod
  def compute_growth(data, order):
    """ Computes the n-th order rate of growth of the feature data in the time_series
    ARGS
      order: number <int>
    RETURN
      n-th rate of growth of time series data, <number array>
    """
    result = []
    for idx in range(1, len(data)):
      try:
        result.append(data[idx]/data[idx-1])
      except Exception, e:
        print 'compute_growth:', e
    return np.array(result)

  @staticmethod
  def compute_difference(data, order):
    """ Computes the n-th delta of the feature data in the time_series
    ARGS
      order: number <int>
    RETURN
      n-th delta of time series data, <number array>
    """
    return np.diff(data, order)

  def update_mean(self, data):
    """given new data, updates the mean of the time series
      ARGS
        data: <number array> 
    """ 
    if self.mean is not None:
      if len(data.shape) == 2:
        self.mean += (data.mean(axis=1) - self.mean)/self.obs
      else:
        self.mean += (data - self.mean)/self.obs
    else:
      if len(data.shape) == 2:
        self.mean = data.mean(axis=1)
      else:
        self.mean = data

  def update_variance(self, data, prevmean, currmean):
    """given new data, updates the variance of the existing time series
      ARGS
        data: <number array> 
    """ 
    if self.var is not None:
      if len(data.shape) == 2:
        self.var += (data.mean(axis=1) - prevmean) * (data.mean(axis=1) - currmean)
      else:
        self.var += (data - prevmean) * (data - currmean)
    else:
      if len(data.shape) == 2:
        self.var = np.zeros(data.shape[0])
      else:
        self.var = 0

  def update(self, data, average = False):
    """given data, incrementally updates the statistical information of the TimeSeries related to the Statistics
    ARGS
      data: <number array>
      average: if true, store of average of data <bool>
    """
    #if average store as single value or average, else store all values
    if average:
      if isinstance(data, list):
        data = np.array(data)
      if isinstance(data, np.ndarray):
        for datapoint in data:
          prevmean = self.mean
          self.obs += 1
          self.update_mean(datapoint)
          self.update_variance(datapoint, prevmean, self.mean)
      else:
        prevmean = self.mean
        self.obs += 1
        self.update_mean(data)
        self.update_variance(data, prevmean, self.mean)
    else:
        prevmean = self.mean
        self.obs += 1
        self.update_mean(data)
        self.update_variance(data, prevmean, self.mean)

  def get_mean(self):
    """returns the standard deviation of the object that instantiated this Statistics"""
    return self.mean

  def get_std(self):
    """returns the standard deviation of the object that instantiated this Statistics"""
    #unbiased standart deviation
    if self.obs > 1:
      return np.sqrt(self.var/(self.obs-1))
    else:
      return 0

  def get_variance(self):
    """returns variance of the object that instantiated this Statistics"""
    return self.var

class FeatureSpace:
  """Class to hold a collection of TimeSeries objects
    INIT ARGS:
      time_series: dictionary with {feature_name:time_series} key:value pairs
      location_id: location_id of installation, eg '55'
      door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0')
      start_time: in format YYYY-MM-DD, <datetime>
      timestep: length in minutes of each audio file
    """
  def __init__(self, location_id, door_count_placement_view_pair, start_time, timestep=timedelta(minutes=15), time_series=None):
    if time_series is None:
      self.time_series = {}
    self.location_id = location_id
    self.door_count_placement_view_pair = door_count_placement_view_pair
    self.start_time = start_time
    self.timestep = timestep

  def __str__(self):
    return self.descr

  def update(self, sample, replace=False):
    """Given a sample object, updates or replaces feature data and its statistical properties
    ARGS:
      feature: feature label <str>
        this feature label should be included in the list_of_features variable in globales.py
      data: number array with feature data <number array>
      replace: replace all existing time_series data
    """
    for feature_label, data in sample.features.items():
      if replace:
        try:
          del self.time_series[feature_label]
        except Exception, e:
          print 'update: unable to delete key when replacing', e
      if feature_label not in self.time_series:
        ts = TimeSeries(feature_label, sample.sr, sample.n_fft, sample.hop_length, sample.win_length, sample.timestamp, sample.label)
        self.time_series[feature_label] = ts
      #check that sampling rate and fft params are the same
      if any(((self.time_series[feature_label].sr != sample.sr), (self.time_series[feature_label].n_fft != sample.n_fft), \
        (self.time_series[feature_label].hop_length != sample.hop_length), (self.time_series[feature_label].win_length != sample.win_length))):
        print "update: unmatching paremeters between existing and new data, printing new, old pairs"
        print "sr (%d, %d) n_fft (%d, %d) hop_length (%d, %d) win_length (%d, %d)" % (self.time_series[feature_label].sr, sample.sr, self.time_series[feature_label].n_fft, sample.n_fft, self.time_series[feature_label].hop_length, sample.hop_length, self.time_series[feature_label].win_length, sample.win_length)


      # store data array, otherwise store average
      if feature_label.split('_')[0] == 'mfcc' or feature_label.split('_')[0] == 'fft':
        self.time_series[feature_label].update(data, average=False)
      else:
        self.time_series[feature_label].update(data, average=True)

  def list_features(self):
    """Returns a list with the keys of features in the time_series dictionary"""
    return self.time_series.keys()

  def get_feature(self, feature):
    """returns the required feature from the time_series dictionary"""
    return self.time_series[feature].getData()

class TimeSeries:
  """Container to hold a sequence of Sample objects and store statistical information about them,
  by using the Statistics object. It should be accessed through the class FeatureSpace.
  INIT ARGS:
    descr: feature name <str>
    sr: Sampling rate <int>
    n_fft: FFT Size <int>
    hop_length: hop size in samples <int>
    win_length: length of window in samples <int>
  """

  def __init__(self, descr, sr, n_fft, hop_length, win_length, timestamp, sample_label='', data = None):
    if data is None:
      self.data = []
    self.descr = descr
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.timestamp = timestamp
    self.sample_label = sample_label
    self.statistics = Statistics()

  def getData(self, as_numpy=True):
    """Returns the time_series data
    ARGS
      as_numpy: if true, returns numpy array <bool>
    RETURN
      time_series data <number array>
    """
    if as_numpy:
      try:
        return np.array(self.data)
      except Exception, e:
        raise Exception("getData: Maybe not compatible with numpy >= %s  as of Sep 25 2014, %s" % (np.version.version, e))
    else:
      return self.data

  def get_difference(self, order=1):
    """ Computes the n-th delta of the feature data in the time_series
    ARGS
      order: number <int>
    RETURN
      n-th delta of time series data, <number array>
    """
    return Statistics.compute_difference(self.getData(), order)

  def get_growth(self, order=1):
    """ Computes the n-th order rate of growth of the feature data in the time_series
    ARGS
      order: number <int>
    RETURN
      n-th rate of growth of time series data, <number array>
    """
    return Statistics.compute_growth(self.getData(), order)

  def update(self, data, average=False):
    """given data, appends it to the time series and computes the new statistics
    ARGS
      data : <number array>
      average: take average of data <bool>
    """
    #if average store as single value or average, else store all values
    if average:
      if isinstance(data, list):
        data = np.array(data)
      if isinstance(data, np.ndarray):
        self.data.append(data.mean())
      else:
        self.data.append(data)
    else:
      self.data.append(data)

    self.statistics.update(data, average)

  def visualize(self):
    """ plots the data in the TimeSeries"""
    secs_per_frame = float(self.hop) / self.sr
    fig, ax = plt.subplots(1)

    y = self.getData()
    y = y.ravel()
    x = np.arange(0, len(y), 1.0)
    x *= secs_per_frame
    plt.plot(x, y, label=self.descr)

    ax.set_title('Mean %s over time' % self.descr)
    ax.legend(loc='upper left', numpoints=1)
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('mean')
    ax.grid()
    plt.show()

class Sample:
  """ Container for audio features, FFT information and start time timestamp of an audio file
    INIT ARGS:
      sr: sampling rate <int>
      n_fft: FFT size <int>
      hop_length : hop length <int>
      win_size: window size <int>
      timestamp: start time timestamp <datetime optional>
      label: label for sample <str optional>
      features: dictionary with feature name, value pairs. e.g: {'mean':0.75} <dictionary optional>
    """
  def __init__(self, sr, n_fft, hop_length, win_length, timestamp=None, label = '', features = None):
    if features is None:
      self.features = {}
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.timestamp = timestamp
    self.label = label

  def __str__(self):
    return self.label

  def set_label(self, label):
    """Setter for class variable label"""
    self.label = label

  def get_label(self):
    """Getter for class variable label"""
    return self.label

  def add_feature(self, feature, data, remove_inf_nan_bool = True):
    """adds audio features to self.features
    ARGS:
      feature: feature label <str>
        this feature label should be included in the list_of_features variable in globales.py
      data: number array with feature data <number array>
      remove_inf_nan_bool: set infs and nans to zero <boolean>
    """
    if remove_inf_nan_bool :
      data = remove_inf_and_nan(data)
    if feature in self.features:
      np.append(self.features[feature], data)
    else:
      self.features[feature] = data

  def get_feature(self, feature):
    """ returns the specified feature
    ARGS
      feature: feature label <str>
        this feature label should be included in the list_of_features variable in globales.py
    RETURN
      requested feature <number array>
    """
    return self.features[feature]

  def visualize(self, features):
    """ plots the requested features
    ARGS
      features: feature labels <str array>
    """
    secs_per_frame = float(self.hop_length) / self.sr
    fig, ax = plt.subplots(1)

    if not isinstance(features, list) and not isinstance(features, np.ndarray):
      features = [features]

    for feature in features:
      y = self.get_feature(feature)
      if isinstance(y, np.ndarray):
        y = y.ravel()
        x = np.arange(0, len(y), 1.0)
        x *= secs_per_frame
        plt.plot(x, y, label=feature)
      else:
        plt.plot(0, y, label=feature, marker='o')

    ax.set_title('%s Features' % self.get_label())
    ax.legend(loc='upper left', numpoints=1)
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('value')
    ax.grid()
    plt.show()
