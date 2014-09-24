#! /usr/bin/env python
""" Extracts audio features from files in an audio folder
  Variable features in the body of this method contains the features to be extracted. 
    These values must exist in the extract_features method in data_mining.py, otherwise the behavior is undefined.
  ARGS
    location_id: as in the deployed devices list <int>
    placement_name: as in the deployed devices list <int>
    view_id: as in the deployed devices list. should not be applicable to audio and value should be 0<int>
    start_day: YYYY-MM-DD. Folders should be named accordingly <str>
    end_day: YYYY-MM-DD. Folders should be named accordingly <str>
    audios_folder(optional): folder where audio files are stored. default from globales.py is used. 
      ATTENTION! folder structure must be as follows:
       audios_folder\placement_name\YYYY-MM-DD
    update(optional): Loads a previously saved file and updates its features. Default is False. <bool 1 or 0>
    replace(optional): Loads a previously saved file and replaces its features. Default is False. <bool 1 or 0>
    pre_processing(optional): pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' <str>
"""

import sys

def main(args):
  if len(sys.argv) >= 6:
    from data_mining import extract_and_save_features
    from datetime import datetime

    #Full list of defined extractions. DO NOT MODIFY! Comment this and write your own list with features.
    """
    features = ['median', 'mean', 'std',\
                'ps_median', 'ps_mean', 'ps_std', \
                'centroid', 'spread', 'skewness', 'kurtosis', 'slope', \
                'mfcc_avg', 'mfcc', 'mfcc_1st_delta', 'mfcc_2nd_delta', \
                'fft', 'fft_avg','enrate']
    """
    features = ['enrate']

    location_id = sys.argv[1]
    placement_id = sys.argv[2]
    view_id = sys.argv[3]
    start_str = sys.argv[4]
    end_str = sys.argv[5]

    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    audios_folder = None
    update = False
    replace = False
    pre_processing = None

    if len(sys.argv) >= 7:
      audios_folder = sys.argv[6]
    if len(sys.argv) >= 8:
      update = bool(int(sys.argv[7]))
    if len(sys.argv) >= 9:
      replace = bool(int(sys.argv[8]))
    if len(sys.argv) >= 10:
      pre_processing = sys.argv[9].split()

    extract_and_save_features(location_id, (placement_id, view_id), start_date, end_date, features, update=update, audios_folder=audios_folder, replace=replace, pre_processing=pre_processing)
  else:
    print 'Syntax is : ./extract_audio_features.py <location_id> <placement_name> <view_id> <start_day(YYYY-MM-DD)> <end_day(YYYY-MM-DD)> <audios_folder optional> <update optional> <replace optional> <pre_processing optional>'
    print "example: ./extract_audio_features.py 23 1000022 0 2013-04-01 2013-04-01 /mnt/data/audio/ 1 1 'regularization'"

if __name__ == "__main__":
  main(sys.argv[1:])

