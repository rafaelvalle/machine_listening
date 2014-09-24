"""
TOOLS
General tools for diverse purposes
"""
import os, shelve, csv
import os
import pytz
from datetime import datetime

import numpy as np
from scipy.stats.mstats import gmean

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def read_csv(path, flag='rb', delimiter=','):
  """Wrapper to read a csv file and return a nested list
  ARGS
    path: filepath <str>
    flag: mode to open the file <str>
    delimiter: csv delimiter
  RETURN
    data: array object with csv data <array>
  """
  data = []
  with open(path, flag) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=delimiter)
    for row in csvreader:
      data.append(row)
  return data

def round_to_multiple(number, multiple):
  """Rounds a number to the closest multiple of another number""" 
  number = np.array(number)
  return number+multiple/2 - (number+multiple/2) % multiple

def save_data_using_shelve(data, filepath, key, flag='c'):
  """
  ARGS
    data: data to be used <any>
    filepath: fullpath of file to save <str>
    key: key to store data <check python's list of valid keys>
    flag: open file mode <str>
  """
  #to ensure folder will be created
  create_path(filepath)
  file = shelve.open(filepath, flag=flag)
  file[key] = data
  file.close()

def load_data_using_shelve(filepath, key, flag='r'):
  """
  ARGS
    filepath: fullpath of file to save <str>
    key: key to store data <check python's list of valid keys>
    flag: open file mode <str>
  RETURN
    data: data loaded from shelve
  """
  file = shelve.open(filepath, flag=flag)
  if not file:
    return None
  data = file[key]
  file.close()
  return data

def create_path(filepath):
  """creates a path in case it does not exist"""
  if not os.path.exists(os.path.dirname(filepath)):
    os.makedirs(os.path.dirname(filepath))

def get_file_paths(path, filetypes = None):
  """ Recurssively search a folder for files of a specific extension and returns an array with their fullpath 
  ARGS
    path: folder path to be looked up. <str>
    filetypes: file extension(file ends with). <str>, e.g. '.mp3'
  RETURN
    filepaths: array with fullpath of all files that match the extension criteria
  """
  filepaths = []
  for root, dirnames, filenames in os.walk(path):
    for file in filenames:
      if filetypes:
        if file.endswith(filetypes):
          filepaths.append(os.path.join(root, file))
      else:
        filepaths.append(os.path.join(root, file))
  return filepaths

def parse_audio_filename(filename):
  """given a filename formatted using standards, parses its installation information and datetime"""
  return ( filename[0:10], datetime.strptime(filename[11:30], "%Y-%m-%d-%H-%M-%S"))

def create_audio_filename(placement_id, dt, sfx='00_1_90000', ext='.mp3'):
  """builds a filename formatted using standards
  ARGS
    placement_id: placement id
    dt: datetime object
    sfx: standard suffix <str>
    ext: audio file type <str>
    RETURN
      audio filename formatted <str>
  """
  date_str = dt.strftime(format="%Y-%m-%d")
  return placement_id+'-'+date_str+'-'+sfx+ext

def date2str(date, format="%Y-%m-%d-%H-%M-%S"):
  """ wrapper to convert a datetime object to string"""
  return date.strftime(format)

def str2date(date_str, format="%Y-%m-%d-%H-%M-%S"):
  """ wrapper to convert a string to datetime"""
  return datetime.strptime(date_str, format)

def createDay(daystr, timezone=None, format="%Y-%m-%d-%H-%M-%S"):
  """Returns a datetime object given a string date representation, a timezone obj and a time format"""
  daytime = str2date(daystr, format)
  if timezone is not None:
    return timezone.localize(daytime, is_dst=None)
  else:
    return pytz.utc.localize(daytime, is_dst=None)

###################
#  MATH AND ALGO  #
###################
def geometric_mean(frame):
    return 10 ** (sum(log10(frame)) / float(len(frame)))

def arithmetic_mean(frame):
    return float(sum(frame)) / float(len(frame))

def find_subsequence(seq, subseq):
  """http://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray/20689091#20689091"""
  target = np.dot(subseq, subseq)
  candidates = np.where(np.correlate(seq, subseq, mode='valid') == target)[0]
  # some of the candidates entries may be false positives, double check
  check = candidates[:, np.newaxis] + np.arange(len(subseq))
  mask = np.all((np.take(seq, check) == subseq), axis=-1)
  return candidates[mask]

def debug(debugging, message):
  if debugging:
    print message
