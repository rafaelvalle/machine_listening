#! /usr/bin/env python
"""Loads an analysis file and saves the specified feature to another file using cPickle"""

import sys

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def main(args):
  if len(sys.argv) >= 2:
    import shelve
    import os
    from tools import getFilePaths
    import cPickle as pkl
    import numpy as np
    path = sys.argv[1]

    save_folder = ''
    if len(sys.argv) >= 3:
      features = sys.argv[2].split()

    if len(sys.argv) >= 4:
      save_folder = sys.argv[3]

    filepaths = []

    if os.path.isfile(path):
      filepaths.append(path)
    else:
      filepaths = sorted(getFilePaths(path, filetypes='.anal'))

    for filepath in filepaths:
      file = shelve.open(filepath, flag='r')
      if not file:
        return 'File not found'

      for key, value in file.items():
        for feature in features:
          data = []
          feature_data = value.time_series[feature].getData()
          n_bins,n_coeffs = feature_data.shape

          for i in range(n_coeffs):
            data.append([])

          for time_bin in range(n_bins):
            for coeff_num in range(n_coeffs):
              data[coeff_num] += feature_data[time_bin][coeff_num].tolist()

          data = np.array(data)
          output_filepath = key+'-'+feature+'.pkl'
          print 'saving %s shaped data with %s to %s' % (str(data.shape), feature, output_filepath)
          with open(save_folder+output_filepath,'wb') as fp:
            pkl.dump(data, fp, protocol=-1)
      file.close()
      print '\n'
  else:
    print "syntax is save_features_to_individual_pickles.py <folder_or_filepath> <'feature_1 feature_2'> <save_folder optional>"

if __name__ == "__main__":
  main(sys.argv[1:])
