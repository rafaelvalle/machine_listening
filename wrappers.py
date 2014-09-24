import librosa
import numpy as np
import scipy
from audiotools import *

def decompose_save(filepath, kernel_size=(5,17), n_fft = 4096, hop_length = 1024):
  """
  Performs Harmonic/Percussive Source Separation on an audio file by applying median filters and saves each filtered file and
  a mix of them as an audio file.
  ARGS
    filepath: fullpath of audio file <str>
    kernel_size: tuple sized of (harmonic, percussive) filters (<int>,<int>)
    n_fft: FFT size <int>
    hop_length : hop length <int>

  """
  signal, sr = load_signal(filepath)
  D = librosa.stft(signal, n_fft, hop_length)
  H, P = librosa.decompose.hpss(D, kernel_size=(5,17))

  signal_harm = librosa.istft(H)
  signal_perc = librosa.istft(P)
  signal_mix = librosa.istft(D)

  librosa.output.write_wav(filepath[:-4]+"-harm.wav", signal_harm, sr)
  librosa.output.write_wav(filepath[:-4]+"-perc.wav", signal_perc, sr)
  librosa.output.write_wav(filepath[:-4]+"-mix.wav", signal_mix, sr)

def decompose_into_harmonic_and_percussive(filepath, kernel_size=(7,15), n_fft = 4096, hop_length = 1024):
  """
  Performs Harmonic/Percussive Source Separation on an audio file by applying median filters and returns each filtered version 
  as an audio signal
  ARGS
    filepath: fullpath of audio file <str>
    kernel_size: tuple sized of (harmonic, percussive) filters (<int>,<int>)
    n_fft: FFT size <int>
    hop_length : hop length <int>
  """
  signal, sr = load_signal(filepath)
  D = librosa.stft(signal, n_fft, hop_length)
  H, P = librosa.decompose.hpss(D, kernel_size=(7,15))
  signal_harm = librosa.istft(H)
  signal_perc = librosa.istft(P)
  return signal_harm, signal_perc

def get_percussive(filepath, kernel_size=(7,15), n_fft = 4096, hop_length = 1024):
  """
  Performs Harmonic/Percussive Source Separation on an audio file by applying median filters and returns the percussive version 
  as an audio signal
  ARGS
    filepath: fullpath of audio file <str>
    kernel_size: tuple sized of (harmonic, percussive) filters (<int>,<int>)
    n_fft: FFT size <int>
    hop_length : hop length <int>
  RETURN
    signal_perc: percussion enhanced audio signal <float numpy array>
  """
  signal, sr = load_signal(filepath)
  D = librosa.stft(signal, n_fft, hop_length)
  _, P = librosa.decompose.hpss(D, kernel_size=(7,15))
  signal_perc = librosa.istft(P)
  return signal_perc

def get_harmonic(filepath, kernel_size=(7,15), n_fft = 4096, hop_length = 1024):
  """
  Performs Harmonic/Percussive Source Separation on an audio file by applying median filters and returns the harmoic version 
  as an audio signal
  ARGS
    filepath: fullpath of audio file <str>
    kernel_size: tuple sized of (harmonic, percussive) filters (<int>,<int>)
    n_fft: FFT size <int>
    hop_length : hop length <int>
  RETURN
    harmonic_perc: harmonic enhanced audio signal <float numpy array>
  """
  signal, sr = load_signal(filepath)
  D = librosa.stft(signal, n_fft, hop_length)
  H, _ = librosa.decompose.hpss(D, kernel_size=(7,15))
  signal_harm = librosa.istft(H)
  return signal_harm

def get_chromagram(filepath, n_fft = 4096, hop_length = 1024):
  """
  Returns the chromagram of an audio file
  ARGS
    filepath: fullpath of audio file <str>
    n_fft: FFT size <int>
    hop_length : hop length <int>
  RETURN
    C: chromagram of audio signal <float numpy array>
  """
  signal, sr = load_signal(filepath)
  D = librosa.stft(signal, n_fft, hop_length)
  C = librosa.feature.chromagram(S=D)
  return C

def get_enrate(signal = None, sr = None, filepath = None, downsample=100):
  """Computes the enrate as described in
    ARGS
      signal: audio signal <number array>
      sr: sampling rate <int>
      filepath: fullpath of audio file <str>
      downsample: sampling rate to downsample signal <int>
    RETURN
      enrate: proportional to speaking rate <float>
  """
  if signal == None:
    signal, sr = load_signal(filepath)

  # FFT data
  n_fft = downsample
  hop_length = int(0.8*downsample)

  # Half-wave rectify the signal waveform
  signal[signal<0] = 0

  # Low-pass filter
  numtaps=2
  cutoff=16.0
  nyq=sr/2.0

  transfer_function = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff/nyq, nyq=nyq)
  signal = scipy.signal.lfilter(transfer_function, 1.0, signal)

  # Downsample to 100hz
  signal = librosa.resample(signal, sr, downsample)

  # Hamming window 1-2 seconds with > 75% overlap
  fft_window = scipy.signal.hamming(downsample, sym=False)

  # FFT, ignore values above 16 hz
  magnitudes = np.abs(librosa.stft(signal, n_fft, hop_length, window=fft_window))

  bin_count, freq_res = get_bin_count_and_frequency_resolution(n_fft, downsample)
  lowest_fbin_idx = int(1/freq_res)
  highest_fbin_idx = int(16/freq_res)

  # Compute the spectral moment ( index weight each power spectral value and sum )
  enrate = np.sum(magnitudes[lowest_fbin_idx:highest_fbin_idx].T * np.array(range(lowest_fbin_idx, highest_fbin_idx)))
  return enrate

def get_magnitudes(filepath, n_fft = 4096, hop_length = 1024):
  """Returns the spectrum magnitudes of an audio file
  ARGS
    filepath: fullpath of audio file <str>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
  RETURNS
    mags: spectrum magnitudes <numpy number array>
  """
  signal, sr = load_signal(filepath, sr=44100)
  mags = np.abs(librosa.stft(signal, n_fft, hop_length))
  return mags

def get_fft(signal = None, filepath = None, n_fft = 4096, hop_length = 512, win_length=2048):
  """Returns the spectrum magnitudes and phases of an audio file
  ARGS
    filepath: fullpath of audio file <str>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
  RETURNS
    mags: a complex-valued matrix <numpy number array>
  """

  if signal == None:
    signal, sr = signal, sr = load_signal(filepath)
  return librosa.stft(signal, n_fft, hop_length)


def get_mfcc(signal, n_fft = 4096, hop_length = 1024, sr=44100, n_mfcc=20, logscaled=True):
  """Computes the mel-frequency cepstral coefficients of a signal
    ARGS
      signal: audio signal <number array>
      n_fft: FFT size <int>
      hop_length : hop length <int>    
      sr: sampling rate <int>
      n_mfcc: number of MFC coefficients <int>
      logscaled: log-scale the magnitudes of the spectrogram <bool>
    RETURN
      mfcc: mel-frequency cepstral coefficients  <number numpy array>
  """ 
  S = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
  if logscaled:
    log_S = librosa.logamplitude(S)
  mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
  return mfcc

def get_feature_delta(feature, order=1):
  """Computes the n-th order delta of a singal. This is not the same as the Dn = Sn+1 - Sn
  ARGS
    feature: 1d or 2d numerical array <number array>
    order: order of delta <int>
  RETURN
    delta 1d or 2d numerical array <number array>
  """

  return librosa.feature.delta(feature, order=order)

def get_centroid(mags = None, n_fft = 1024, hop_length = 512, sr = None, fbins=None):
  """Computes the centroid of a spectrum. Equivalent to the first statistical moment of the spectrum 
  ARGS
    mags: spectrum magnitudes <number array>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
    sr: sampling rate <int>
    fbins: array with frequency of each frequency bin(optional) <number array>
  RETURN
    centroid: spectrum centroid <float>
  """
  if mags is None:
    print 'CENTROID : mags is None'
    return -1

  if fbins is None:
    print 'has not fbin'
    nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
    fbins = binres * np.array(range(0, nbins+1))

  centroid = fbins.dot(mags)/mags.sum(axis=0)
  return centroid

def get_spread(centroids = None, mags = None, n_fft = 1024, hop_length = 512,  sr = None, fbins=None):
  """Computes the spread of a spectrum. Equivalent to the second statistical moment of the spectrum 
  SHOULD BE VECTORIZED
  ARGS
    centroids: centroid(s) <number array>
    mags: spectrum magnitudes <number array>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
    sr: sampling rate <int>
    fbins: array with frequency of each frequency bin(optional) <number array>
  RETURN
    spread: spectrum spread <float>
  """
  if mags is None:
    raise Exception("get_spread: mags is None")

  spread = []
  probs = None

  if fbins is None:
    nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
    fbins = binres * np.array(range(0, nbins+1))

  if centroids is None:
    centroids = get_centroids(mags, n_fft, hop_length)

  for i in range(0,len(centroids)):
    probs = mags[:,i]/sum(mags[:,i])
    spread.append(np.sqrt(np.mean(probs * (abs(fbins - centroids[i])**2))))
  spread = np.array(spread)  
  return spread

def get_skewness(centroids = None, mags = None, n_fft = 1024, hop_length = 512,  sr = None, fbins=None):
  """Computes the skewness of a spectrum. Equivalent to the third statistical moment of the spectrum 
  SHOULD BE VECTORIZED
  ARGS
    centroids: centroid(s) <number array>
    mags: spectrum magnitudes <number array>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
    sr: sampling rate <int>
    fbins: array with frequency of each frequency bin(optional) <number array>
  RETURN
    skewness: spectrum skewness <float>
  """
  if mags is None:
    raise Exception("get_skewness: mags is None")

  skewness = []
  probs = None

  if fbins is None:
    nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
    fbins = binres * np.array(range(0, nbins+1))

  if centroids is None:
    centroids = get_centroids(mags, n_fft, hop_length, sr)

  stds = get_spread(centroids, mags, n_fft, hop_length, sr)

  for i in range(0,len(centroids)):
    probs = mags[:,i]/sum(mags[:,i])
    skewness.append(np.mean((probs * (abs(fbins - centroids[i])**3))/stds[i]**3))
  skewness = np.array(skewness)
  return skewness

def get_kurtosis(centroids = None, mags = None, n_fft = 1024, hop_length = 512,  sr = None, fbins=None):
  """Computes the kurtosis of a spectrum. Equivalent to the fourth statistical moment of the spectrum 
  SHOULD BE VECTORIZED
  ARGS
    centroids: centroid(s) <number array>
    mags: spectrum magnitudes <number array>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
    sr: sampling rate <int>
    fbins: array with frequency of each frequency bin(optional) <number array>
  RETURN
    kurtosis: spectrum kurtosis <float>
  """
  if mags is None:
    raise Exception("get_kurtosis: mags is None")

  kurtosis = []
  probs = None

  if fbins is None:
    nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
    fbins = binres * np.array(range(0, nbins+1))

  if centroids is None:
    centroids = get_centroids(mags, n_fft, hop_length, sr)

  stds = get_spread(centroids, mags, n_fft, hop_length, sr)

  for i in range(0,len(centroids)):
    probs = mags[:,i]/sum(mags[:,i])
    kurtosis.append(np.mean((probs * (abs(fbins - centroids[i])**4))/stds[i]**4))
  kurtosis = np.array(kurtosis)
  return kurtosis

def get_slope(mags = None, n_fft = 1024, hop_length = 512, sr = None, fbins = None):
  """Computes the slope of each timeframe of a spectrum.
  ARGS
    mags: spectrum magnitudes <number array>
    n_fft: FFT size <int>
    hop_length : hop length <int>    
    sr: sampling rate <int>
    fbins: array with frequency of each frequency bin(optional) <number array>
  RETURN
    slope: slope of time frames <float>
  """
  if mags is None:
    raise Exception("SLOPE: mags is None")

  slopes = []
  slope = None

  if fbins is None:
    nbins, binres = get_bin_count_and_frequency_resolution(n_fft,sr)
    fbins = binres * np.array(range(0, nbins+1))

  for i in range(0,mags.shape[1]):
    slope, intercept,_,_,_ = scipy.stats.linregress(fbins,mags[:,i])
    slopes.append((slope, intercept))
  slopes = np.array(slopes)
  return slopes
