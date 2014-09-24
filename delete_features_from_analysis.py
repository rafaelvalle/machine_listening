#! /usr/bin/env python
"""Deletes feature(s) from an analysis (.anal) file
ARGS
  folder_or_filepath: folder or filepath of .anal file(s) to be modified <str>
  features: feature(s) feature or list of features to be deleted <str>
"""
import sys

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def main(args):
  if len(sys.argv) == 3:
    import shelve
    import os
    from tools import get_file_paths
    import cPickle as pkl
    import numpy as np
    path = sys.argv[1]
    features = sys.argv[2].split()

    filepaths = []

    if os.path.isfile(path):
      filepaths.append(path)
    else:
      filepaths = sorted(get_file_paths(path, filetypes='.anal'))

    for filepath in filepaths:
      file = shelve.open(filepath, flag='w')
      
      if not file:
        raise Exception("File not found")

      for key, value in file.items():
        for feature in features:
          del value.time_series[feature]
      try:
        file[key] = value
      except Exception, e:
        print e

      file.close()
      print '\n'
  else:
    print "syntax is delete_features_from_analysis.py <folder_or_filepath> <'feature_1 feature_2'>"

if __name__ == "__main__":
  main(sys.argv[1:])
