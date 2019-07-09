# A simple script to test the triggering of the QIClick flourescence camera

#Imports python libraries
import RPi.GPIO as gpio
import time

# Initialises system
def init():
	
	# Sets up the board
	gpio.setmode(gpio.BOARD)

	pins = (8,	# Trigger
		10,	# SYNC-A
		12)	# SYNC-B

	# Initialises trigger, sync-a and sync-b pins respectively
	gpio.setup(pins[0],gpio.OUT)
	gpio.setup(pins[1],gpio.IN,pull_up_down=gpio.PUD_DOWN)
	gpio.setup(pins[2],gpio.IN)

	return pins


# Defines the  pulse triggering function
def trigger_pulse(pin,duration):

	gpio.output(pin,1)
	time.sleep(duration)
	gpio.output(pin,0)

	return 0


# Exectutes the program
if __name__ == '__main__':

	# Initialises the program
	pins = init()
	timeout = 1 


	# Tests inital readout signals
	if gpio.input(pins[1]):
		print('SYNC-A signal present on start up.')
	if gpio.input(pins[2]):
		print('SYNC-B signal present on start up.')

	# Triggers image capture (using the edge detection -use the trigger_pulse function for pulse detection)
	gpio.output(pins[0],1)
	print('Trigger fired, check QI software for a successful image capture.')


	# Waits for SYNC-B signal
	time_init = time.mktime(time.gmtime())
	exit_status_B  = 0
	while gpio.input(pins[2]) == False:
		time_fin = time.mktime(time.gmtime())
		if time_fin - time_init > timeout:
			exit_status_B = 1
			break
	if exit_status_B == 0:
		print('SYNC-B signal detected.')
	elif exit_status_B == 1:
		print('SYNC-B signal timed out.')
	else:
		print('Program has encountered an unknown error at SYNC-B.')



	#Waits for SYNC-A signal
	time_init = time.mktime(time.gmtime())
	exit_status_A = 0
	while gpio.input(pins[1]) == False:
		time_fin = time.mktime(time.gmtime())
		if time_fin - time_init > timeout:
			exit_status_A = 1
			break
	if exit_status_A == 0:
		print('SYNC-A signal detected.')
	elif exit_status_A == 1 and exit_status_B == 1:
		print('SYNC-A signal timed out. The signal could have been missed when waiting for the SYNC-B signal.')
	elif exit_status_A == 1 and exit_status_B == 0:
		print('SYNC-A signal timed out.')
	else:
		print('Program has encountered an unkown error at SYNC-A.')


	# Cleans up GPIO pins and ensures trigger is turned off
	gpio.output(pins[0],0)
	gpio.cleanup()
