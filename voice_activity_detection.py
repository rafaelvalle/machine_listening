#adapted from Shriphani Palakodety code
from numpy.fft import *
from numpy import log10, sqrt
from scipy.stats.mstats import gmean

import math
from audiotools import *
from tools import find_subsequence, geometric_mean, arithmetic_mean

MLD_FRAME_DURATION = 30 #frame length in milliseconds for milanovic, lukac and domazetovic
MLD_SAMPLES_PER_FRAME = lambda sr : int(sr * (MLD_FRAME_DURATION / 1000.0))

MH_FRAME_DURATION = 10 #frame length in milliseconds for Moattar & Homayounpour
MH_SAMPLES_PER_FRAME = lambda sr : int(sr * (MH_FRAME_DURATION / 1000.0))

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def voice_activity_detection(samples, average_intensity, instances, samples_per_frame):
  """
  Voice activity detection based on the Moattar Homayounpour algorithm
  ARGS
    samples: signal in 16 bit (int16)
    average_intensity: former average_intensity set by the user (we supply an updated value)
    instances: number of times this VAD was run was previously
  """
  #globals init thresholds
  energy_prim_thresh = 40
  freq_prim_thresh = 185
  sfm_prim_thresh = 5

  #compute the intensity
  intensity = get_sample_intensity(samples)

  #frame attribute arrays
  frame_energies = []  #holds the energy value for each frame
  frame_max_frequencies = []  #holds the dominant frequency for each frame
  frame_SFMs = []  #holds the spectral flatness measure for every frame
  frame_voiced = []  #tells us if a frame contains silence or speech

  #attributes for the entire sampled signal
  min_energy = 0
  min_dominant_freq = 0
  min_sfm = 0

  #check for the 30 frame mark
  thirty_frame_mark = False
  n_frames = samples.shape[0]/samples_per_frame
  for frame_number in range(n_frames-1):
    frame_start = frame_number * samples_per_frame
    frame_end = frame_start + samples_per_frame

    # marks if 30 frames have been sampled
    if frame_number == 30:
      thirty_frame_mark = True

    frame = samples[frame_start:frame_end]

    #compute frame energy
    frame_energy = get_sample_energy(frame)
    freq_domain_real, freq_domain_imag = real_imaginary_freq_domain(frame)

    freq_magnitudes = get_freq_domain_magnitudes(freq_domain_real, freq_domain_imag)
    dominant_freq = get_dominant_freq(freq_domain_real, freq_domain_imag)
    frame_SFM = get_sfm(freq_magnitudes)

    #now, append these attributes to the frame attribute arrays created previously
    frame_energies.append(frame_energy)
    frame_max_frequencies.append(dominant_freq)
    frame_SFMs.append(frame_SFM)

    #the first 30 frames are used to set min-energy, min-frequency and min-SFM
    if not thirty_frame_mark and not frame_number:
        min_energy = frame_energy
        min_dominant_freq = dominant_freq
        min_sfm = frame_SFM
    elif not thirty_frame_mark:
        min_energy = min(min_energy, frame_energy)
        min_dominant_freq = min(dominant_freq, min_dominant_freq)
        min_sfm = min(frame_SFM, min_sfm)

    #once we compute the min values, we compute the thresholds for each of the frame attributes
    energy_thresh = energy_prim_thresh * log10(min_energy)
    dominant_freq_thresh = freq_prim_thresh
    sfm_thresh = sfm_prim_thresh

    counter = 0

    if (frame_energy - min_energy) > energy_thresh:
        counter += 1
    if (dominant_freq - min_dominant_freq) > dominant_freq_thresh:
        counter += 1
    if (frame_SFM - min_sfm) > sfm_thresh:
        counter += 1

    if counter > 1:  #this means that the current frame is not silent.
        frame_voiced.append(1)
    else:
        frame_voiced.append(0)
        min_energy = ((frame_voiced.count(0) * min_energy) + frame_energy)/(frame_voiced.count(0) + 1)

    #now update the energy threshold
    energy_thresh = energy_prim_thresh * log10(min_energy)

  #once the frame attributes are obtained, a final check is performed to determine speech.
  #at least 5 consecutive frames are needed for speech.

  instances += 1  #a new instance has been processed
  old_average_intensity = average_intensity
  average_intensity = ((old_average_intensity * (instances-1)) + intensity) / float(instances)  #update average intensity

  return frame_voiced
