"""
PRE-PROCESSING
General tools for pre-processing data in general
"""
from numpy import isnan, isinf, ndarray, copy
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

def remove_inf_and_nan(data, new_value=0.):
  """Wrapper for setting infs and nans to zero
  ARGS
    data: <numpy array> or <number>
    new_value: value to substitue for <number>
  RETURN
    data: processed data <numpy array> or <number>
  """
  #set nan and infs to zero (default)
  if isinstance(data, ndarray):
    data[isnan(data)] = new_value
    data[isinf(data)] = new_value
  else:
    if isnan(data) or isinf(data):
      data = new_value
  return data

def standardize(data, copy=False, with_mean=True, with_std=True):
  """Scales data in place: subtract data by the mean and divide it by the standard deviation
  http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
  ARGS
    data: array-like or CSR matrix with shape [n_samples, n_features]
    copy" if False, try to avoid a copy and do inplace scaling instead.
    with_mean: if True, center the data before scaling <bool>
    with_std: If True, scale the data to unit variance (or equivalently, unit standard deviation) <bool>
  RETURN
    standardizer: class with learned parameters <sklearn.preprocessing.StandardScaler>
    """
  standardizer = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std).fit(data)
  standardizer.transform(data)
  return standardizer

def normalize(data, copy=False, norm='l2'):
  """Normalize samples individually to unit norm
  ARGS
    data: array-like with shape [n_samples, n_features]
    norm: norm to use to normalize each non zero sample. 'l1' or 'l2'
  RETURN
    normalizer: Normalizer object
  http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
  """
  normalizer = Normalizer(copy=copy, norm=norm).fit(data)
  normalizer.transform(data)
  return normalizer

def pre_process(data, pre_processing):
  """Wrapper for pre_processing data
    ARGS
      data:
      pre_processing:
    RETURN
      data: processed data <number array>
      standardizer: StandardScaler object with learned params
      normalizer: Normalizer object with learned params
      gaussianizer: Gaussianizer object with learned params
  """
  print 'pre-processing step'
  standardizer, normalizer, gaussianizer = None, None, None

  if isinstance(pre_processing, str):
    pre_processing = [pre_processing]

  for idx in range(len(pre_processing)):
    process = pre_processing[idx]
    print process
    if process == 'standardize':
      print 'standardizing data'
      standardizer = StandardScaler(copy=True, with_mean=True, with_std=True)
      standardizer.fit(data)
      data = standardizer.transform(data, copy=True)
    elif process == 'gaussianize':
      print 'gaussianize data'
      gaussianizer = robjects.r('Gaussianize')
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

    elif process == 'normalize':
      print 'normalizing data'
      normalizer = normalize(data)

  return data, standardizer, normalizer, gaussianizer
