# A python script to obtain the time stamps for various frame rates for the picam

# Imports libraries
import os
import numpy as np
import pandas as pd


# Function to obtain the timestamps for a given frame rate
def get_time_stamps(frame_rate,no_of_entries):

	# Calculates the time needed to obtain the correct number of entries (time is in ms)
	time = 1000*no_of_entries/frame_rate

	# Launches the camera recording the frame rate and saving the output to a temporary file
	os.system('raspivid -o temp_vid.h264 -w 300 -h 300 -t '+str(time)+' -fps '+str(frame_rate)+' -pts temp_data.csv -fl -n')

	# Reads the data into an array
	time_stamp_ary = np.ndarray.flatten(np.array(pd.read_csv('temp_data.csv')))

	# Clears the temporary files
	os.system('rm temp_vid.h264 temp_data.csv')

	# Returns the array of time stamps
	return time_stamp_ary

# The main function that loops over frame rates
def main(frame_rate_start,frame_rate_end,no_of_entries):

	# Defines initial variables
	frame_rate_ary = np.arange(frame_rate_start,frame_rate_end,1)
	time_stamp_mtx = np.zeros((len(frame_rate_ary),no_of_entries))

	# Calculates minimum processing time for the user
	processing_time = np.sum(1/frame_rate_ary)*no_of_entries/60
	print('Minimum processing time: %.0f minutes.' % processing_time)

	# Loops through frame rates obtaining the time stamp data
	for i in range(len(frame_rate_ary)):

		temp_ary = get_time_stamps(frame_rate_ary[i],no_of_entries)
		time_stamp_mtx[i,:len(temp_ary)] = temp_ary


	# Saves time stamp data to file
	pd.DataFrame(time_stamp_mtx).to_csv('time-stamp-'+str(frame_rate_start)+'-'+str(frame_rate_end)+'-'+str(no_of_entries)+'.csv',header=None,index=None)	

main(10,50,100)
	
