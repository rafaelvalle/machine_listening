#! /usr/bin/env python
"""
PLOT AUDIO FEATURE AND LOI OCCUPANY ON THE SAME GRAPH
"""

import sys

def main(args):
	if len(args) == 4:
		import shelve
		from tools import read_csv

		loi_path = args[0]
		feature_filepath = args[1]
		features = args[2].split()

		file = shelve.open(feature_filepath, flag='r')
		if not file:
			return 'File not found'

		import matplotlib
		import matplotlib.pyplot as plt
		plt.ion()
		matplotlib.rcParams.update({'font.size': 4})

		fig, ax = plt.subplots(1, figsize=(4,3), dpi=200)

		for key, value in file.items():
			for feature in features:
				data = value.time_series[feature].getData()
				x = range(len(data))
				ax.step(x, data, label=feature)
			ax.set_title(value.descr)
		file.close()

		leg = ax.legend(loc='upper right', prop={'size':4})
		leg.get_frame().set_alpha(0.5)

		ax.set_xlabel('time')
		ax.set_ylabel('feature value')
		ax.grid()

		from ground_truth import get_zone_count

		file = read_csv(loi_path)
		occupancy = [get_zone_count(data, True) for data in file]

		fig, ax = plt.subplots(1, figsize=(4,3), dpi=200)
		ax.step(range(len(occupancy)), occupancy,label='LOI Occupancy')

		leg = ax.legend(loc='upper right', prop={'size':4})
		leg.get_frame().set_alpha(0.5)

		ax.set_xlabel('time')
		ax.set_ylabel('occupancy')
		ax.grid()
		raw_input("Press enter to quit")
	else:
		print 'syntax is ./plot_feature_and_loi.py loi.csv audio_features.anal feature'

if __name__ == "__main__":
	main(sys.argv[1:])
