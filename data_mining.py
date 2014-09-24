"""
DATA MINING
Methods for audio feature extraction based on id, start_date and end_date of from eyestalks data
"""

from ntpath import basename
import os
from copy import copy
from numpy import array as np_array
import pytz

from classes import *
from wrappers import *
from ground_truth import get_zone_count_estimates
from tools import get_file_paths, parse_audio_filename, load_data_using_shelve, save_data_using_shelve, str2date, date2str, createDay
from globales import *
from pre_processing import standardize, normalize

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

#############
#  METHODS  #
#############
def extract(placement_id, analysis_date, features, feature_space, audios_folder=AUDIO_FOLDER_GLOBAL, replace=False, pre_processing=None):
  print 'Analysing', features, 'with', pre_processing, 'pre processing from', audios_folder

  day = timedelta(days = 1)

  curr_date = analysis_date.date()

  #next day might have previous day evening data 
  for i in range(2):
    date_str = date2str(curr_date + i * day, "%Y-%m-%d")
    fullpath = os.path.join(audios_folder, str(placement_id), date_str)
    print 'Audio folder with placement id and date is', fullpath
    #sorted is very imported such that features are correctly aligned!
    filepaths = sorted(get_file_paths(fullpath, '.mp3'))
    for i in range(len(filepaths)):
      filepath = filepaths[i]
      inst_id, dt = parse_audio_filename(basename(filepath))
      if dt >= analysis_date and dt < analysis_date + day:
        #set the start time to be the datetime of the first analyzed audio file
        if feature_space.start_time is None:
          print 'Feature Space start time is', dt
          feature_space.start_time = dt

        print 'Adding %s to analysis ' % filepath

        filename = os.path.basename(filepath)
        #only works if filename structure is locationid_es-YYY-MM-DD-HH-MM-SS_1_*
        left_index = filename.index('_es')+4
        right_index = filename.index('_1_')
        full_time_str = filename[left_index:right_index]
        full_time = str2date(full_time_str)
        full_time.replace(second=0)
        features_sample = extract_features(filepath, features, full_time, pre_processing)
        feature_space.update(features_sample, replace=replace)
        #updateFeatureSpace(features_sample, feature_space, replace=replace)

        #in case we chose to replace a feature
        replace = False


def extract_features(filepath, features, timestamp=None, pre_processing=None):
  """Given an audio filepatth and a list with features, returns a Sample object with audio features,
  FFT information and start time timestamp
  ARGS
    filepath: full filepath of audio file <str>
    features: list with featres to be extracted
    timestamp: start time timestamp <datetime optional>
    pre_processing: pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' <str>
  RETURN
    Sample object
  """
  # AUDIO FILENAMES ARE OFTEN IN LOCAL TIME!
  pre_processes = ''
  if pre_processing :
    try:
      if isinstance(pre_processing, str):
        pre_processes = '_'+pre_processing
      else:
        pre_processes = '_' + '_'.join(pre_processing)
    except Exception, e:
      print "extract_features: something wrong with '_'.join(pre_processing)"
      print e

  if not isinstance(features, list) and not isinstance(features, np.ndarray):
    features = [features]

  #time-domain
  sig, sr = load_signal(filepath, sr=None, pre_processing=pre_processing)

  #full wave rectification
  sig_abs = abs(sig)

  #n_fft, hop_length, win_length and n_mfcc are set as global vars in globales.py
  features_sample = Sample(sr, n_fft, hop_length, win_length, timestamp=timestamp)

  #get bin count and frequency resolution, create arrays with bin frequencies
  nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
  fbins = binres * np_array(range(0, nbins+1))

  #features
  if 'median' in features:
    features_sample.add_feature('median'+pre_processes, np.median(sig_abs))
  if 'mean' in features:
    features_sample.add_feature('mean'+pre_processes, np.mean(sig_abs))
  if 'std' in features:
    features_sample.add_feature('std'+pre_processes, np.std(sig_abs))

  #frequency-domain
  spectrum = get_fft(sig, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mags = np.abs(spectrum) #slightly faster than spec * spec.conj()

  #ps is power spectrum
  ps = mags**2

  #features
  if 'ps_median' in features:
    features_sample.add_feature('ps_median'+pre_processes, np.median(ps))
  if 'ps_mean' in features:
    features_sample.add_feature('ps_mean'+pre_processes, np.mean(ps))
  if 'ps_std' in features:
    features_sample.add_feature('ps_std'+pre_processes, np.std(ps))
  if 'centroid' in features:
    centroids = get_centroid(mags, n_fft, hop_length, sr, fbins)
    features_sample.add_feature('centroid'+pre_processes, centroids)
  if 'spread' in features:
    if 'centroid' not in features_sample.features:
      centroids = get_centroid(mags, n_fft, hop_length, sr, fbins)
      features_sample.add_feature('centroid'+pre_processes, centroids)
    spreads = get_spread(centroids, mags, n_fft, hop_length, sr, fbins)
    features_sample.add_feature('spread'+pre_processes, spreads)
  if 'skewness' in features:
    if 'centroid' not in features_sample.features:
      centroids = get_centroid(mags, n_fft, hop_length, sr, fbins)
      features_sample.add_feature('centroid'+pre_processes, centroids)
    skewness = get_skewness(centroids, mags, n_fft, hop_length, sr, fbins)
    features_sample.add_feature('skewness'+pre_processes, skewness)
  if 'kurtosis' in features:
    if 'centroid' not in features_sample.features:
      centroids = get_centroid(mags, n_fft, hop_length, sr, fbins)
      features_sample.add_feature('centroid'+pre_processes, centroids)
    kurtosix = get_kurtosis(centroids, mags, n_fft, hop_length, sr, fbins)
    features_sample.add_feature('kurtosis'+pre_processes,  kurtosix)
  if 'slope' in features:
    slopes = get_slope(mags, n_fft, hop_length, sr, fbins)
    features_sample.add_feature('slope'+pre_processes, slopes)
  if 'mfcc'  in features:
    mfcc = get_mfcc(sig, n_fft, hop_length, sr, n_mfcc=n_mfcc)
    features_sample.add_feature('mfcc'+pre_processes, mfcc)
  if 'mfcc_1st_delta' in features:
    if 'mfcc' not in features:
      mfcc = get_mfcc(sig, n_fft, hop_length, sr, n_mfcc=n_mfcc)
    mfcc_1st_delta = get_feature_delta(mfcc)
    features_sample.add_feature('mfcc_1st_delta'+pre_processes, mfcc_1st_delta)
  if 'mfcc_2nd_delta' in features:
    if 'mfcc' not in features:
      mfcc = get_mfcc(sig, n_fft, hop_length, sr, n_mfcc=n_mfcc)
    mfcc_2nd_delta = get_feature_delta(mfcc, order=2)
    features_sample.add_feature('mfcc_2nd_delta'+pre_processes, mfcc_2nd_delta)
  if 'mfcc_avg'  in features:
    #add average of each coefficient
    mfcc = get_mfcc(sig, n_fft, hop_length, sr, n_mfcc=n_mfcc)
    mfcc = mfcc.mean(axis=1)
    features_sample.add_feature('mfcc_avg'+pre_processes, mfcc)
  if 'fft' in features:
    features_sample.add_feature('fft'+pre_processes, mags)
  if 'fft_avg' in features:
    #add average of each frequency bin
    mags_avg = mags.mean(axis=1)
    features_sample.add_feature('fft_avg'+pre_processes, mags_avg)
  if 'enrate' in features:
    enrate = get_enrate(sig,sr)
    features_sample.add_feature('enrate'+pre_processes, enrate)

  return features_sample

def extract_and_save_features(location_id, door_count_placement_view_pair, start_date, end_date, features, savepath = None, update=False, audios_folder=None, replace=False, pre_processing=None):
  """Extracts and saves audio features from audio files
    ARGS
      location_id: location_id of installation, eg '55' <int> or <str>
      door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
      start_time: a datetime.time object for time lower bound
      end_time: a datetime.time object for time upper bound
      features: list with label (keys) of features used, [<str>,<str>...]
      savepath: folder path to save analysis file. if None, uses default from globales.py <str>
      update: update existing data (append) <boolean>
      audios_folder: folder where subfolder of placement_id with audio files exists <str>
      replace: replace existing data <boolean>
      pre_processing: pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' with default values <str>
  """
  oneday = timedelta(days = 1)
  dates = []
  dates.append(start_date)

  if not os.path.isdir(audios_folder):
    raise Exception("Folder %s does not exist" % audios_folder)

  while dates[-1] < end_date:
    dates.append(dates[-1] + oneday)

  print 'Extract and save features'
  print 'location_id', location_id
  print 'Update?', update
  print 'Replace?', replace
  print 'Pre-processing', pre_processing
  print 'Audios Folder', audios_folder

  for day in dates:
    datetime_string = date2str(day, format="%Y-%m-%d")
    print '\nDay is %s' % (datetime_string)

    save_filename = analfilename(location_id, door_count_placement_view_pair, day)
    if savepath is None:
      path = os.path.join(ANALYSIS_FOLDER_GLOBAL,str(location_id),save_filename)
    else:
      path = os.path.join(savepath, save_filename)

    if update or replace:
      #ensure file is writeble
      os.chmod(path, 0666)
      analysis = getData(location_id, door_count_placement_view_pair, day)
    else:
      analysis = FeatureSpace(location_id, door_count_placement_view_pair, day)
    extract(door_count_placement_view_pair[0], day, features, analysis, audios_folder, replace=replace, pre_processing=pre_processing)

    key = str(location_id)+'-'+str(door_count_placement_view_pair[0])+'-'+str(door_count_placement_view_pair[1])+'-'+datetime_string

    print 'Saving analysis to path %s and key %s ' % (path, key)
    save_data_using_shelve(analysis, path, key, flag='c')

def collectData(location_id, door_count_placement_view_pair, start_time, end_time, features, adjusted = False):
  """Collects audio features from analysis files and occupancy from csv files
    ARGS
      location_id: location_id of installation, eg '55' <int> or <str>
      door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
      start_time: a datetime.time object for time lower bound
      end_time: a datetime.time object for time upper bound
      features: list with label (keys) of features used, [<str>,<str>...]
      adjusted: get adjusted LOI (datetime, raw_loi, adjusted) <boolean>
  """
  bigdata = None
  bigtruth = None

  day = timedelta(days = 1)
  while start_time < end_time:
    data = getData(location_id, door_count_placement_view_pair, start_time)

    if data is None:
      'New analysis'
      analysis = FeatureSpace(location_id, door_count_placement_view_pair, start_time)
      data = extract(door_count_placement_view_pair[0], start_time, features, analysis)

    Y = get_zone_count_estimates(location_id, door_count_placement_view_pair, start_time, start_time+day, adjusted = adjusted)

    X = combineData(data, features, start_time)

    if len(X.shape) == 3:
      X = X[0]

    #SUPERHACK to fix different lengths between GT and AUDIO
    diff = len(X) - len(Y)

    if diff > 0:
      print 'X', len(X), 'Y', len(Y)
      print 'collectData: SUPERHACK Removing audio features timesteps', start_time
      X = X[diff:]
    elif diff < 0:
      print 'Y', len(Y), 'X', len(X)
      print 'collectData: SUPERHACK Removing ground truth timesteps', start_time
      Y = Y[diff:]

    if bigdata is None:
      bigdata = X
    else:
      bigdata = np.append(bigdata, X, axis=0)

    if bigtruth is None:
      bigtruth = Y
    else:
      bigtruth = np.append(bigtruth, Y)

    start_time += day

  if bigdata is None:
    print 'bigdata is None, exiting'
    exit()

  return bigdata, bigtruth

def combineData(data, features, start_time = None):
  X = None

  print '  Adding', features

  if isinstance(features, str):
    features = [features]

  for feature in features:
    item_number, feature_type = None, None
    #parse feature arguments
    feature_split = feature.split('-')
    if len(feature_split) > 1:
      for item in feature_split:
        try:
          #DO NOT CHANGE THE ORDER!
          if item in list_of_features:
            feature = item
          elif item[0] == 'p':
            feature_type = item
          elif item.isdigit():
            item_number = int(item)
          else:
            print 'could not find feature, aborting!'
            return
        except:
          print 'exception'

    #get features
    if feature == 'day':
      print 'combine day as feature - NOT IMPLEMENTED'
    elif feature == 'mae': #day in 4 slices
      if start_time is not None:
        start_mae = int(start_time.strftime('%H'))
        end_mae = data.time_series[features[0]].getData().shape[0]
        X = updateX(X, EARLY_MORNING_BINARY[start_mae:start_mae+end_mae]) #00, 06
        X = updateX(X, np.roll(EARLY_MORNING_BINARY, 24)[start_mae:start_mae+end_mae]) #06,12
        X = updateX(X, np.roll(EARLY_MORNING_BINARY, 48)[start_mae:start_mae+end_mae]) #12,18
        feature_data = np.roll(EARLY_MORNING_BINARY, 72)[start_mae:start_mae+end_mae] #18,24
      else:
        print 'Requested mae but start_time was not provided!'
    else:
      feature_data = data.time_series[feature].getData()
      #if feature.split('_')[0] != 'mfcc':
      #better yet
      if len(feature_data.shape) == 1:
        #reshaping to 2 dimensions and transforming row into column 
        feature_data = feature_data.reshape(len(feature_data),1)

    # get item of number
    if item_number is not None :
      #get nth column from feature, for example MFCC
      #if feature.split('_')[0] == 'mfcc':
      #better yet
      if len(feature_data.shape) == 2:
        try:
          feature_data = feature_data[item_number]
        except:
          print 'could not get the requested mfcc or fft frequency bin'
      else:
        try:
          feature_data = feature_data[item_number]
        except:
          print 'could not get the requested index'

    #get feature type
    if feature_type is not None:
      if feature_type[0] == 'p': #raise feature to the n-th power
        try:
          power = int(feature_type[1])
          feature_data = np.power(feature_data, power)
        except:
          print 'could not get int from power number'
      elif feature_type == 'diff':
        feature_data = data.time_series[feature].get_difference()
        feature_data = np.insert(feature_data,0,0) #add zero to beggining to keep vector size equal
      elif feature_type == 'growth':
        feature_data = data.time_series[feature].get_growth()
        feature_data = np.insert(feature_data,0,0) #add zero to beggining keep to vector size equal

    #update feature vector
    X = updateX(X, feature_data)
  return X

def updateX(X, feature_data):
  if X is None:
    if feature_data.ndim == 1:
      X = np.array(feature_data, ndmin=2)
    else:
      X = feature_data
  else:
    try:
      X = np.append(X, feature_data, axis=1)
    except Exception, e:
      print str(e)
  return X

def getData(location_id, door_count_placement_view_pair, start_date, mode='r'):
  """Collects audio features from analysis file
    ARGS
      location_id: location_id of installation, eg '55' <int> or <str>
      door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0') (<str>, <str>)
      start_date: a datetime.time object
  """
  filename = analfilename(location_id, door_count_placement_view_pair, start_date)
  fullpath = os.path.join(ANALYSIS_FOLDER_GLOBAL,str(location_id),filename)
  key = analkey(location_id, door_count_placement_view_pair, start_date)

  print 'getDATA fullpath', fullpath
  print 'getDATA Key', key

  if mode == 'w':
    #ensure that file will be writable
    os.os.chmod(fullpath, 0775)
  return load_data_using_shelve(fullpath, key, flag=mode)

def collect_train_and_test_data(location_id, door_count_placement_view_pair, trainPer, testPer, features, timezone = None, pre_processing=''):

  standardizer, normalizer, gaussianizer = None, None, None
  trainStart = trainPer[0]
  trainEnd = trainPer[1]
  predStart = testPer[0]
  predEnd = testPer[1]

  if all([isinstance(location_id, list), isinstance(door_count_placement_view_pair, list), isinstance(timezone,list)]):
    train_location_id = location_id[0]
    train_placement_view_pair = door_count_placement_view_pair[0]
    train_timezone = timezone[0]
    test_location_id = location_id[1]
    test_placement_view_pair = door_count_placement_view_pair[1]
    test_timezone = timezone[1]
  else:
    train_location_id = location_id
    train_placement_view_pair = door_count_placement_view_pair
    train_timezone = timezone
    test_location_id = location_id
    test_placement_view_pair = door_count_placement_view_pair
    test_timezone = timezone

  print train_placement_view_pair, test_placement_view_pair

  train_start_time = createDay(trainStart, train_timezone)
  train_end_time = createDay(trainEnd, train_timezone)

  print train_location_id, train_placement_view_pair, train_start_time, train_end_time
  print'\nGetting train data'
  print 'pre-processing step'
  standardizer = None
  if 'standardize' in pre_processing:
    standardizer = Standardizer(copy=copy, with_mean=True, with_std=True)
  if 'gaussianize' in pre_processing:
    gaussianizer = robjects.r('Gaussianize')

  train_X, train_Y = collectData(train_location_id, train_placement_view_pair, train_start_time, train_end_time, features, adjusted=True)

  if standardizer is not None:
    train_X = standardizer.transform(train_X, copy=None)
  if gaussianizer is not None:
    from rpy2.robjects.numpy2ri import numpy2ri
    robjects.conversion.py2ri = numpy2ri
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
    train_X = np.dstack((train_X, np.ones(len(train_X))))
    if len(train_X.shape) == 3:
      train_X = train_X[0]

  if pre_processing == 'normalize':
    print 'normalizing data'
    normalizer = normalize(train_X)

  print'\nGetting train data'
  if trainPer == testPer:
    test_X = train_X
    test_Y = train_Y
  else:
    test_start_time = createDay(predStart, test_timezone)
    test_end_time = createDay(predEnd, test_timezone)
    print test_location_id, test_placement_view_pair, test_start_time, test_end_time
    test_X, test_Y = collectData(test_location_id, test_placement_view_pair, test_start_time, test_end_time, features, adjusted=True)

    #pre process data
    if pre_processing == 'standardize':
      test_X = standardizer.transform(test_X, copy=None)
    if pre_processing == 'normalize':
      test_X = normalizer.transform(test_X, copy=None)
    if pre_processing == 'gaussianize':
      text_X = (text_X - gaussianizer.mean)/gaussianizer.std

    #add ones column
    print 'adding constant to test_X'
    if len(test_X.shape) > 1:
      ones_array = np.ones((test_X.shape[0],1))
      test_X = np.append(test_X, ones_array, 1)
    else:
      test_X = np.dstack((test_X, np.ones(len(test_X))))
      if len(test_X.shape) == 3:
        test_X = test_X[0]

  return ((train_X, train_Y), (test_X, test_Y))
