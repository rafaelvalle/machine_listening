#! /usr/bin/env python
"""Sets the start_time and timestep of a FeatureSpace object"""
import sys

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
  
def main(args):
  if len(args) == 3:
    import shelve
    import os
    from datetime import datetime, timedelta
    import pytz
    from tools import get_file_paths

    path = args[0]
    datetime_str = args[1]
    dt = datetime.strptime(datetime_str, "%Y-%m-%d-%H-%M-%S")
    dt = dt.replace(tzinfo=pytz.utc)
    timestep = timedelta(minutes=int(args[2]))
    filepaths = []

    if os.path.isfile(path):
      filepaths.append(path)
    else:
      filepaths = sorted(get_file_paths(path, filetypes='.anal'))

    for filepath in filepaths:
      file = shelve.open(filepath, flag='w', writeback=False)
      for key, feature_space in file.items():
        print 'key', key
        if fs.descr:
          print 'update fs.descr to new format', feature_space.descr
          try:
            fs.location_id = fs.descr[0]
            fs.door_count_placement_view_pair = fs.descr[1]
          except Exception, e:
            print 'set_feature_start_time_and_timestep', e
        del fs.descr 
        feature_space.start_time = dt
        feature_space.timestep = timestep

        del file[key]
        file[key] = feature_space
      file.close()
      print '\n'
  else:
    print 'syntax is ./set_featurespace_start_time_and_timestep.py folder_or_filepath YYYY-MM-DD-HH-MM-SS timedelta_in_minutes'

if __name__ == "__main__":
  main(sys.argv[1:])
