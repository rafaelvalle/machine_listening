"""
GROUND TRUTH
Methods for parsing ground truth from a .csv file
"""

from datetime import datetime, timedelta
from globales import ANALYSIS_FOLDER_GLOBAL, gtext, DEBUG, setDebug, gtfilename
from tools import date2str, read_csv
from pytz import utc
from numpy import array as np_array


def convert_time_and_count_to_tuple(data, adjusted=False):
  """In-place conversion of csv row to tuple with datetime and occupancy"""
  for idx in range(len(data)):
    data[idx] = get_datetime_and_zone_count(data[idx], adjusted)

def get_datetime_from_csv_row(data, datetime_format="%Y-%m-%d %H:%M:%S"):
  """Given a csv row where the dstetime is the first column, 
  creates a datetime object from a string representing a date and time and a corresponding format string.
  ARGS
    data: data row from a .csv file <csv row>
  RETURN
    <datetime object>
  """
  return datetime.strptime(data[0], datetime_format)

def get_datetime_and_zone_count(data, adjusted=False):
  """Get datetime and zone count from data row extracted from a .csv file.
  ARGS
    data: data row from a .csv file <csv row>
    adjusted: to select between raw data or adusted <bool>. if adjusted is chosen but not available, returns raw.
  RETURN
    (datetime, zone_count)
  """
  return ( get_datetime_from_csv_row(data), get_zone_count(data, adjusted) )

def get_zone_count(data, adjusted=False):
  """Get zone count from data extracted from as .csv file.
  ARGS
    data: data row from a .csv file. e.g. YYYY-MM-DD HH:MM:SS, raw_zone_count, adjusted_zone_count <csv row>
    adjusted: to select between raw data or adusted <bool>. if adjusted is chosen but not available, returns raw.
  RETURN
    zone count <float>
  """
  if adjusted:
    try:
      return float(data[2])
    except Exception, e:
      print 'get_zone_count: adjusted value not available, returning raw'
  return float(data[1])

def get_zone_count_estimates(location_id, door_count_placement_view_pair, start_date, end_date, adjusted=False):
  """Iterates through .csv files to return a list of (datetime, zone_count)
  ARGS
    location_id: location_id of installation, eg '55'
    door_count_placement_view_pair: placement and view id pair, e.g. ('3333230','0')
    start_date: in format YYYY-MM-DD, <datetime>
    end_date: in format YYYY-MM-DD. range is exclusive '<'. <datetime>
    adjusted: to select between raw data or adusted <bool>. if adjusted is chosen but not available, returns raw.
  RETURN
    array with (datetime, zone_count) tuples
  """
  datetime_zone_count_pairs = []
  day = timedelta(days = 1)

  curr_day = start_date

  while curr_day < end_date:
    date_str = date2str(curr_day, "%Y-%m-%d")
    fullpath = ANALYSIS_FOLDER_GLOBAL+str(location_id)+'/'+gtfilename(location_id,door_count_placement_view_pair,curr_day)
    if DEBUG:
      print 'get_zone_count_estimates: reading file:', fullpath
    data = read_csv(fullpath)
    for idx in range(len(data)):
      ts = utc.localize(get_datetime_from_csv_row(data[idx]), is_dst=None).astimezone(utc)
      if ts >= start_date and ts < end_date:
        datetime_zone_count_pairs.append(get_zone_count(data[idx], adjusted))
    curr_day += day
  datetime_zone_count_pairs = np_array(datetime_zone_count_pairs)
  return datetime_zone_count_pairs
