"""
GLOBALES
Class to carry global variables used in the machine_listening repo
"""
from numpy import zeros, ones, concatenate, vstack
from os.path import expanduser

#############
#  FOLDERS  #
#############
#must and should end with a /
#folder to read audio files for feature extraction from. These the files should be stored in folders named after their placement id.
AUDIO_FOLDER_GLOBAL = '/Media/Retail/'
#folder to read analysis (.anal) files from. These files should be stored in folders named after their placement id.
ANALYSIS_FOLDER_GLOBAL = expanduser("~")+'/GoogleDrive/AudioAnalytics/features/'
#folder to save result plots and statistical summary.
RESULTS_FOLDER_GLOBAL = expanduser("~")+'/GoogleDrive/AudioAnalytics/results/'

##############
#  FILE EXT  #
##############
#standard file extensions used
gtext = '.csv'
analext = '.anal'
figext = '.png'
statsext = '.stats'
wifiext = '.wifi'
separator = '-'
zone_to_placement_table_json = 'zone_to_placement_table.json'

####################################
#  METHODS FOR BUILDING FILENAMES  #
####################################
wififilename = lambda location_id,door_count_placement_view_pair,dt : str(location_id)+\
    separator+str(door_count_placement_view_pair[0])+\
    separator+str(door_count_placement_view_pair[1])+\
    separator+dt.strftime(format="%Y-%m-%d")\
    +'-wifi'+gtext

gtfilename = lambda location_id,door_count_placement_view_pair,dt : str(location_id)+\
    separator+str(door_count_placement_view_pair[0])+\
    separator+str(door_count_placement_view_pair[1])+\
    separator+dt.strftime(format="%Y-%m-%d")\
    +gtext

gtstructure = lambda dt, occupancy, adjusted_occupancy : ' '.join([str(dt), str(occupancy), str(adjusted_occupancy)])

analfilename = lambda location_id,door_count_placement_view_pair,dt : str(location_id)+\
    separator+str(door_count_placement_view_pair[0])+\
    separator+str(door_count_placement_view_pair[1])+\
    separator+dt.strftime(format="%Y-%m-%d")\
    +analext

analkey = lambda location_id,door_count_placement_view_pair,dt : str(location_id)+\
    separator+str(door_count_placement_view_pair[0])+\
    separator+str(door_count_placement_view_pair[1])+\
    separator+dt.strftime(format="%Y-%m-%d")

namestructure = lambda location_id, door_count_placement_view_pair, fold, kernel: str(location_id)+\
    separator+str(door_count_placement_view_pair[0])+\
    separator+str(door_count_placement_view_pair[1])+\
    separator+fold+\
    separator+kernel

################
#  REGRESSION  #
################
EARLY_MORNING_BINARY = vstack((concatenate((ones(24, dtype=int), zeros(72, dtype=int)))))

#################
#  AUDIO & FFT  #
#################
n_mfcc = 20
n_fft = 8192 # FFT WINDOW SIZE
hop_length =  4096 # HOP SIZE
win_length = 4096 # WINDOW LENGTH
list_of_features = ['median', 'mean', 'std', \
            'ps_median', 'ps_mean', 'ps_std', \
            'centroid', 'spread', 'skewness', 'kurtosis', 'slope', \
            'mfcc_avg', 'mfcc', 'mfcc_1st_delta', 'mfcc_2nd_delta', \
            'mae', 'fft_avg', 'fft', 'enrate']

###################
#  FOR DEBUGGING  #
###################
DEBUG = 0

def setDebug(val):
    DEBUG = val
