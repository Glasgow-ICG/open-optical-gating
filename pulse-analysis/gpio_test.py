# A script to test the delay between sending a GPIO true signal and receiving the signal

#Imports libraries
import RPi.GPIO as gpio
import time
import numpy as np
import sys
import pandas as pd

# Initialises the board and pins
def init(pin_send,pin_receive):

	# Sets up the board
	gpio.setmode(gpio.BOARD)

	# Sets up the pins
	gpio.setup(pin_send,gpio.OUT)
	gpio.setup(pin_receive,gpio.IN)





# Detemines the time delay for the GPIO signal
def GPIO_delay(pin_send,pin_receive):

	# Ininitial time of send signal
	time_init = time.time()

	# Sends a signal through pin_send
	gpio.output(pin_send,True)

	# Waits for the signal and notes time of arival
	while (gpio.input(pin_receive) == False):
		pass

	# Gets the final time
	time_fin = time.time()

	# Resets the output to false
	gpio.output(pin_send,False)


	return time_fin - time_init


# Initialises board and performs time test
def GPIO_test(pin_send,pin_receive,N):

	# Initialises pins and board
	init(pin_send,pin_receive)

	# Defines an array to hold the GPIO delay times
	time_ary = np.zeros(N)

	# Perforsm the GPIO_delay test N times
	for i in range(N):

		time_ary[i] = GPIO_delay(pin_send,pin_receive)

	return time_ary



	
# Performs the main test and saves results
def main(pin_send,pin_receive,N):

	# Gets the array of time delays
	time_ary = GPIO_test(pin_send,pin_receive,N)

	
	#Saves the result to a file
	filename ='GPIOtest-'+str(pin_send)+'-'+str(pin_receive)+'-'+str(N)+'.csv'
	pd.DataFrame(time_ary).to_csv(filename,index=None,header=None)


main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])) 
