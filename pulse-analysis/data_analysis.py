# A python script containing data analysis functions #
# The functions should be slightly more complicated than simple plots with labels

# Contents:
#	1)	polyfit_scatter(file_name,title,xlabel,ylabel,x,y,poly_order)
#	2) 	histogram_peak(data,bins)
#	3) 	mass_histogram_peak(file_name,bin_start,bin_end,bin_interval)

# Imports libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

# A simple scatter plot with a linear fit
def polyfit_scatter(file_name,title,xlabel,ylabel,x,y,poly_order):

	# Calculates the polynomial coefficients
	fit_poly = np.polyfit(x,y,poly_order)


	# Obtains the fitted y values from the polynomial coefficients
	y_fit = np.zeros(len(x))

	for i in range(0,len(fit_poly)):

		y_fit += fit_poly[i]*np.power(x,len(fit_poly)-i-1)

	#Plots the results
	plt.scatter(x,y,c='c',marker='x',label='Data')
	plt.plot(x,y_fit,c='m',label='Fit')

	#Adds labels
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	#Saves result
	plt.savefig(file_name)

#Calculates and returns the peak bin no of a histogram
def histogram_peak(data,bins):

	# Converts data set into histograms
	hist_data = np.array(np.histogram(data,bins))

	# Gets position of peak (assumes center of bin with largest value and assumes peak is not the last bin as more data would be needed)
	peak_position = int(np.argmax(hist_data[0]))
	peak_value = hist_data[1][peak_position] + (hist_data[1][peak_position + 1] - hist_data[1][peak_position])/2

	return peak_value


# Returns a set of peaks for a collection of histograms
def mass_histogram_peak(filename,bin_start,bin_end,bin_interval):

	# Defines initial variables
	histogram_data = np.array(pd.read_csv(filename))
	peak_data = np.zeros(len(histogram_data[:,0]))
	bins = np.arange(bin_start,bin_end,bin_interval)

	# Calculates histogram peak for each histogram
	for i in range(len(peak_data)):
		peak_data[i] = histogram_peak(histogram_data[i,:],bins)

	print(peak_data)
	#Saves results to file
	pd.DataFrame(peak_data).to_csv('peaks-'+filename,header=None,index=None)
	return peak_data

mass_histogram_peak(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))
