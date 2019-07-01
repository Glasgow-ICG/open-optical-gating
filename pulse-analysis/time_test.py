# A python script to test the accuracy of various timing functions #
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


# Tests runs a function for N itterations and caluclates the average time #
def time_function(function,parameters,N):

	# Tests if N is a natural number
	if N < 1:
		print('N must be greater than 0')
		return -1 

	#Defines initial variables
	N = int(N)
	time_avg = np.zeros(len(parameters))

	#Test runs over all parameters
	for i in range(len(parameters)):

		#Gets the initial time
		time_init = time.time()

		#Runs the function for N itterations
		for k in range(N):
			function(parameters[i])
		
		#Gets the final time
		time_fin = time.time()

		#Calculates the average time
		time_avg[i] = (time_fin - time_init)/N	

	return time_avg

# Calculates the percentage error of a measured value compared to a true value
def percentage_error(measured_value,true_value):

	# Avoids dividing by zero
	if true_value != 0:
		return abs(measured_value - true_value)/true_value

	else:
		return -1

# Obtains the distribution of a function for single set of parameters
def distribution_function(function,parameters,n):

	#Defines initial variables
	time_distribution = np.zeros(n)

	# Repeats the function n times each time noting the the time of operation and storing the result
	for i in range(n):
		
		# Gets the initial time
		time_init = time.time()

		#Performs the operation
		function(parameters)

		# Gets the final time and calculates the time difference
		time_final = time.time()
		time_distribution[i] = time_final - time_init
	
	return time_distribution
	



#np.savetxt(file_name,array,delimiter=',')

	
# Performs historgram of the time.sleep for each over a range of wait times (wt)
def mass_time_distribution(wt_start,wt_end,wt_interval,N):

	# Creates array of wait times
	wt_ary = np.arange(wt_start,wt_end,wt_interval)
	hist_array = np.zeros((len(wt_ary),N))

	# Loops through wait times calculating the actual sleep time of each
	for i in range(len(wt_ary)):

		# Gets a histogram  of the time_sleep function for varies wait times
		hist_array[i,:] = distribution_function(time.sleep,wt_ary[i],N)

	# Saves the histogram data
	pd.DataFrame(hist_array).to_csv('histdata-'+str(wt_start)+'-'+str(wt_end)+'-'+str(wt_interval)+'-'+str(N)+'.csv',header=None,index=None)
			
	# Returns the array
	print(hist_array)
	return hist_array


mass_time_distribution(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]))
