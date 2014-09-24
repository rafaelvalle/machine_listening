#! /usr/bin/env python
"""Saves data from an analysis file (.anal) generated with machine listening to a .mat file"""
import sys

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def main(args):
  if len(sys.argv) > 1:
    from os.path import splitext, split
    from tools import load_data_using_shelve
    from scipy.io import savemat, whosmat

    path = sys.argv[1]
    folder, filename = split(path)
    #ignoring H,M,S from filename
    keyid = splitext(filename)[0]

    print 'Using keyid %s' % keyid
    output_file = keyid+'.mat'

    print 'Loading %s' % path
    analysis = load_data_using_shelve(path, keyid, flag='r')

    print 'Saving as %s' % output_file
    analysis_dict = {}

    for feature, data in analysis.time_series.items():
      print 'Adding', feature
      analysis_dict[feature] = data.getData()

    savemat(output_file, analysis_dict, appendmat=True)
    print whosmat(output_file)
  else:
    print 'syntax is : python py2mat.py inputfile'

if __name__ == "__main__":
  main(sys.argv[1:])
