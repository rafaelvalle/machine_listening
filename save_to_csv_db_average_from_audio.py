""" Saves to .csv file the db average or per minute db average of an audio file
This is dependent on external libraries such as ffmpeg.
"""

import os, csv
import numpy as np
import scipy
import librosa
from datetime import timedelta, datetime

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def save_db_average_to_csv(audio_path, save_path, start_time, end_time):
  """Saves to a .csv file the SPL (0.00001) dB average of an audio file
  ARGS
    audio_path: path of audio file <str>
    save_path: path of save file <str>
    currrent_timezone: timezone of the audio file <pytz.timezone>, e.g. pytz.timezone('America/Los_Angeles')
      this is required since audio files are most of the time stored in local time, whereas .csv files are in UTC
  """
  file_paths = sorted(get_filepaths(audio_path, filetypes='.mp3'))

  folder_name = os.path.basename(os.path.normpath(audio_path))
  output_csv = folder_name+'-db-average.csv'
  full_savepath = save_path+output_csv
  print 'saving to %s' % full_savepath

  with open(full_savepath, "wb") as output_file:
    csvwriter = csv.writer(output_file, delimiter=',')
    for idx in range(len(file_paths)):
      filepath = file_paths[idx]
      filename = os.path.splitext(os.path.basename(filepath))[0]

      start_time_str = filename.split('_')[1][3:]
      curr_time = datetime.strptime(start_time_str, "%Y-%m-%d-%H-%M-%S")
      if curr_time >= start_time and curr_time < end_time:
        signal, sr = librosa.load(filepath, None)
        signal_abs = abs(signal)

        average_db = 20 * scipy.log10(signal_abs.mean()/0.00001)

        #save mean db per minute as csv file with format <datetime, db value>
        csvwriter.writerow([curr_time, average_db])

def save_per_minute_db_average_to_csv(audio_path, save_path, start_time, end_time):
  """Saves to a .csv file the non-full(0.00001) per minute dB average of an audio file
  ARGS
    audio_path: path of audio file <str>
    save_path: path of save file <str>
    currrent_timezone: timezone of the audio file <pytz.timezone>, e.g. pytz.timezone('America/Los_Angeles')
      this is required since audio files are most of the time stored in local time, whereas .csv files are in UTC
  """
  file_paths = get_filepaths(audio_path, filetypes='.mp3')
  for idx in range(len(file_paths)):
    filepath = file_paths[idx]
    filename = os.path.splitext(os.path.basename(filepath))[0]

    start_time_str = filename.split('_')[1][3:]
    curr_time = datetime.strptime(start_time_str, "%Y-%m-%d-%H-%M-%S")
    curr_time = curr_time.replace(second=00, microsecond=00)

    output_csv = filename+'-onsets.csv'
    if curr_time >= start_time and curr_time < end_time:
      signal, sr = librosa.load(filepath, None)
      signal_abs = abs(signal)

      per_minute_average = []
      samples_in_a_minute = sr * 60
      chunk_number = 1
      while(samples_in_a_minute*(chunk_number-1) < len(signal_abs)):
        start_index = samples_in_a_minute*(chunk_number-1)
        end_index = samples_in_a_minute*chunk_number
        if end_index >= len(signal_abs):
          end_index = -1
        signal_abs_slice = signal_abs[start_index:end_index]
        per_minute_average.append(signal_abs_slice.mean())
        chunk_number += 1

      per_minute_average = np.array(per_minute_average)
      per_minute_average_db = 20 * scipy.log10(per_minute_average/0.00001)
      signal_db = 20 * scipy.log10(signal_abs)
      timestamps = []

      for count in range(len(per_minute_average_db)):
        timestamps.append(curr_time + count*timedelta(minutes=1))

      full_savepath = save_path+"catamini-per_minute_db_average-"+filename+".csv"
      print 'saving to %s' % full_savepath

      #save mean db per minute as csv file with format <datetime, db value>
      with open(full_savepath, "wb") as output_file:
        csvwriter = csv.writer(output_file, delimiter=',')
        for key, value in zip(sorted(timestamps), per_minute_average_db):
            csvwriter.writerow([key, value])

def get_filepaths(path, filetypes = None):
  filepaths = []
  for root, dirnames, filenames in os.walk(path):
    for file in filenames:
      if filetypes:
        if file.endswith(filetypes):
          filepaths.append(os.path.join(root, file))
      else:
        filepaths.append(os.path.join(root, file))
  return filepaths