"""
AUDIO TOOLS
  General tools used in machine listening
  CONVERSION
    ftom               frequency to midi note
    hz2mel             hz to mel
    mel2hz             mel to hz
  PRE-PROCESSSING
    pre_emphasis        apply pre-emphasis filter
    de_mean             subtract signal mean from signal (0 center signal)
    amplitude_regularization      apply signal^0.7
  FFT
    get_bin_count_and_frequency_resolution      return number of frequency bins and frequency resolution 
  AUDIO HANDLING
    load_signal        load a signal, resample it accordingly and apply pre-processing 
"""

import scipy
from numpy.fft import *
from numpy import log, exp, mean, sqrt,log10, sum, argmax
from numpy import array as np_array 
from librosa import load as load_audio

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

################
#  CONVERSION  #
################

def ftom(f):
  """ convert frequency to midi note 
  ARGS 
    f: frequency(ies) in Hz > 0, <number> or numpy number array
  RETURN
    m: MIDI note with A 440 as reference, <float>
  """
  if isinstance(f, list):
    f = np_array(f)
  m = 17.3123405046 * log(.12231220585 * f) 
  return m


def hz2mel(f):
  """ Convert a number or numpy numerical array of frequencies in Hz into mel
  ARGS: Frequency or array of frequencies,  <number> or numpy array of numbers
  RETURN: Mel frequency(ies), <number> or numpy array of numbers
  """
  if isinstance(f, list):
    f = np_array(f)  
  return 1127.01048 * log(f/700.0 +1)

def mel2hz(m):
  """ Convert a number or numpy numerical array of frequency in mel into Hz
  ARGS: Mel Frequency or array of mel frequencies,  <number> or numpy array of numbers
  RETURN: frequency(ies) in Hz, <number> or numpy array of numbers
  """
  if isinstance(m, list):
    m = np_array(m)
  return (exp(m / 1127.01048) - 1) * 700


####################
#  PRE-PROCESSING  #
####################

def pre_emphasis(signal, p):
  """ Apply pre-emphasis filter
  ARGS 
    signal: signal amplitudes, should be in the range [-1.0,1.0], array of numbers
  RETURN 
    Filtered signal, array of numbers
  """
  if isinstance(signal, list):
    signal = np_array(signal)
  elif isinstance(signal, (int, long, float, complex)):
    raise Exception("Invalid arg")
  return scipy.signal.lfilter([1., -p], 1, signal)

def de_mean(signal, axis=-1):
  """ De-mean signal(zero center)
  ARGS: 
    signal: signal alitudeii, shouldin the range [-1.0,1.0], <number> or array of numbers
    axis: axis to apply mean, <int>
  RETURN: 
    Zero centered signal, <number> or numpy array of numbers
  """
  if isinstance(signal, list):
    signal = np_array(signal)
  elif isinstance(signal, (int, long, float, complex)):
    raise Exception("Invalid arg")
  return signal - mean(signal,axis)

def convert_signal_to_bit_integer(signal, bits, signed=True):
  """
  ARGS
    signal: signal amplitudes, should be in the range [-1.0, 1.0], numpy array of numbers
    bits: bit-depth value, <int>
  RETURN
    signal_scaled_to_n_bits: signal cast to n bits <int array>  
  """  
  if isinstance(signal, list):
    signal = np_array(signal)
  elif isinstance(signal, (int, long, float, complex)):
    raise Exception("Invalid arg")
    
  # convert amplitude [-1.0, 1.0] to N-bit samples
  half_n_bits = 2**(bits-1)
  if signed:
    signal_scaled_to_n_bits = signal * half_n_bits
  else:
    signal_scaled_to_n_bits = (signal + 1) * half_n_bits
  signal_scaled_to_n_bits = signal_scaled_to_n_bits.astype(int)  
  return signal_scaled_to_n_bits

def amplitude_regularization(signal, bits=16, factor=0.7):
  """
  ARGS: 
    signal: signal amplitudes, should be in the range [-1.0, 1.0], numpy array of numbers
    bits: bit-depth value, <int>
    factor: 0.7 by default, as suggested by Gerald Friedland @ ICSI
  RETURN: 
    regularized: amplitude regularized signal, <number> or numpy array of numbers
  """
  if isinstance(signal, list):
    signal = np_array(signal)
  elif isinstance(signal, (int, long, float, complex)):
    raise Exception("Invalid arg")
    
  # convert amplitude [-1.0, 1.0] to N-bit samples
  half_n_bits = 2**(bits-1)
  signal_scaled_to_n_bits = (signal + 1) * half_n_bits

  # regularize
  regularized = signal_scaled_to_n_bits ** factor

  # scale back to [-1.0,1.0]
  regularized -=  half_n_bits
  regularized /= half_n_bits
  return regularized


#########
#  FFT  #
#########
def get_bin_count_and_frequency_resolution(n_fft, sr):
  """Given FFT size and Sampling Rate, returns the number of frequency bins and frequency resolution
  ARGS
    n_fft: FFT Size <int>
    sr: Sampling rate
  RETURN
    nbins: number of frequency bins <int>
    freq_resolution: frequency resolution in Hz <float>
  """
  n_bins = n_fft/2
  freq_resolution = sr*0.5/n_bins
  return (n_bins, freq_resolution)

def real_imaginary_freq_domain(samples):
  """
  Apply fft on the samples and return the real and imaginary
  parts in separate 
  """
  freq_domain = fft(samples)
  freq_domain_real = np_array([abs(x.real) for x in freq_domain])
  freq_domain_imag = np_array([abs(x.imag) for x in freq_domain])

  return freq_domain_real, freq_domain_imag

def get_dominant_freq(real_freq_domain_part, imag_freq_domain_part):
  """Returns the dominant frequency"""
  max_real_idx = argmax(real_freq_domain_part)
  max_imag_idx = argmax(imag_freq_domain_part)

  dominant_freq = 0

  if (real_freq_domain_part[max_imag_idx] > imag_freq_domain_part[max_imag_idx]):
    dominant_freq = abs(fftfreq(len(real_freq_domain_part), d=(1.0/44100.0))[max_real_idx])
  else:
    dominant_freq = abs(fftfreq(len(imag_freq_domain_part), d=(1.0/44100.0))[max_imag_idx])

  return dominant_freq

def get_freq_domain_magnitudes(real_part, imaginary_part):
  """Magnitudes of the real-imag frequencies"""
  return sqrt(real_part**2 + imaginary_part**2)

def get_sfm(frequencies):
  """long-term spectral flatness measure"""
  return 10 * log10(stats.mstats.gmean(frequencies) / mean(frequencies))

#########
#  DSP  #
#########
def get_sample_energy(samples):
  """
  ARGS:
    samples: samples of a signal
  """
  if isinstance(samples, list) or isinstance(samples, tuple):
    samples = np_array(samples)
  return sum(samples**2)

def get_sample_intensity(samples):
  if isinstance(samples, list) or isinstance(samples, tuple):
    samples = np_array(samples)
  return 20.8 * log10(sqrt(sum(samples**2)/float(len(samples))))

def load_signal(filepath, sr=None, pre_processing=None):
  """Given a filepath, loads a signal and applies the desired pre-processing to it
  ARGS
    filepath: full filepath of audio to be loaded, <str>
    sr: sampling rate of audio file. If None, the file's sampling rate is used; If different from the file's sampling rate, resampling is applied.<int>"
    pre_processing: pre-processing to be applied; accepts 'regularization' and 'pre_emphasis' with default values <str>
  RETURN
    signal: loaded signal with requested pre-processing, <float array>
  """
  signal, sr = load_audio(filepath, sr)
  if pre_processing is not None:
    if 'pre_emphasis' in pre_processing:
      signal = pre_emphasis(signal, 0.95)
    if 'regularization' in pre_processing:
      signal = amplitude_regularization(signal)
  return (signal, sr)
