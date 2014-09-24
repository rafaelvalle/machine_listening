#! /usr/bin/env python
"""
Plots the choosen audio features
"""

import sys

def main(args):
  if len(sys.argv) == 3:
    import shelve

    filepath = sys.argv[1]
    file = shelve.open(filepath, flag='r')
    if not file:
      return 'File not found'

    import matplotlib
    import matplotlib.pyplot as plt
    plt.ion()
    matplotlib.rcParams.update({'font.size': 4})

    features = sys.argv[2].split()

    for key, value in file.items():
      for feature in features:
        data = value.time_series[feature].getData().T
        if feature == 'mfcc' or feature =='fft':
          print "mfcc and fft plotting is not supported, only average"
          continue
        if feature.split('_')[0] == 'mfcc_avg' or feature.split('_')[0] == 'fft_avg':
          num_of_coefficients = data.shape[0]
          for idx in range(num_of_coefficients):
            fig, ax = plt.subplots(1, figsize=(1,0.6), dpi=300)
            x = range(len(data[idx]))
            ax.step(x, data[idx], label=feature+str(idx))
            leg = ax.legend(loc=1, ncol=3, prop={'size':4})
            leg.get_frame().set_alpha(0.5)
        else:
          x = range(len(data))
          ax.step(x, data, label=feature)
      ax.set_title(value.descr)

    file.close()

    ax.set_xlabel('time')
    ax.set_ylabel('feature value')
    ax.grid()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1)

    raw_input("Press enter to quit")
  else:
    print 'syntax is ./plot_audio_features.py filepath feature'


if __name__ == "__main__":
  main(sys.argv[1:])
