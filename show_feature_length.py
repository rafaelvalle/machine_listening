#! /usr/bin/env python
import sys

"""Outputs the description and shape of features (TimeSeries) saved in a analysis file (.anal)"""
def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def main(args):
  if len(sys.argv) == 2:
    import shelve
    import os
    from tools import get_file_paths

    path = sys.argv[1]
    filepaths = []

    if os.path.isfile(path):
      filepaths.append(path)
    else:
      filepaths = sorted(get_file_paths(path, filetypes='.anal'))

    for filepath in filepaths:
      file = shelve.open(filepath, flag='r')
      if not file:
        return 'File not found'
      for key, value in file.items():
        print 'key', key
        ts = value.time_series
        for feature_name, feature in sorted(ts.items()):
          print 'description %s' % feature.descr
          print 'shape\t', feature.getData().shape, 'key', feature_name, '\n'
      file.close()
      print '\n'
  else:
    print 'syntax is python show_feature_length.py folder_or_filepath'

if __name__ == "__main__":
  main(sys.argv[1:])
