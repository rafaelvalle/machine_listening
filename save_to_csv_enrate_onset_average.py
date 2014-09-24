""" TO BE IMPLEMENTED 
Saves to a .csv file the enrate (energy rate) or onset average or per minute average of an audio file
"""

import os, pytz
import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import librosa
from datetime import timedelta, datetime

from tools import get_filepaths

def save_onset_average_to_csv(audio_path, save_path, timezone='America/Los_Angeles'):
    file_paths = get_filepaths(audio_path, filetypes='.mp3')

    for idx in range(len(file_paths)):
        filepath = file_paths[idx]
        filename = os.path.splitext(os.path.basename(filepath))[0]
        current_timezone = pytz.timezone(timezone)
        start_time_str = filename.split('_')[1][3:]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d-%H-%M-%S")
        start_time = start_time.replace(second=00, microsecond=00)
        output_csv = save_path+filename+'-onsets.csv'

        signal, sr = librosa.load(filepath, None)

        print 'extracting onset_frames from %s' % filepath
        onset_frames = librosa.onset.onset_detect(y=signal, sr=sr, hop_length=64)
       
        print 'extracting onset_times from %s' % filepath
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=64, n_fft=2048)

        onsets_per_second = np.bincount(onset_times.astype(int))
        onsets_per_minute_average = []

        for minute in range(1, len(onsets_per_second)/60):
            start_index = (minute-1) * 60
            end_index = minute * 60
            onsets_per_second_slice = onsets_per_second[start_index:end_index]
            onsets_per_minute_average.append(onsets_per_second_slice.mean())

        try:
            #do the last slice
            start_index = (minute+1) * 60
            end_index = -1
            onsets_per_second_slice = onsets_per_second[start_index:end_index]
            onsets_per_minute_average.append(onsets_per_second_slice.mean())
        except Exception, e:
            print e

        timestamps = []

        for count in range(len(onsets_per_minute_average)):
            timestamps.append(start_time + timedelta(minutes=1*count))

        full_savepath = save_path+"catamini-per_minute_onset_average-"+filename+".csv"
        print 'Saving output to ', full_savepath

        #save mean db per minute as csv file with format <datetime, db value>
        with open(full_savepath, "wb") as output_file:
          csvwriter = csv.writer(output_file, delimiter=',')
          for key, value in zip(sorted(timestamps), onsets_per_minute_average):
            csvwriter.writerow([key, value])

    print 'done!'
