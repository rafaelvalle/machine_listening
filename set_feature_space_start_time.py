#! /usr/bin/env python
import sys

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
        print 'fs.descr', feature_space.descr
        new_descr = (feature_space.descr[0], dt, timestep)
        feature_space.start_time = dt
        feature_space.timestep = timestep

        del file[key]
        file[key] = feature_space
      file.close()
      print '\n'
  else:
    print 'syntax is python show_feature_length.py folder_or_filepath YYYY-MM-DD-HH-MM-SS timedelta_in_minutes'

if __name__ == "__main__":
  main(sys.argv[1:])
