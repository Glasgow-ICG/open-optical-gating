# Python imports
import fastpins as fp
import time 



#Function that initialises various controlls (pins for triggering laser and fluorescence camera along with the USB for controlling the Newport stages)
def init_controls(laser_trigger_pin, fluorescence_camera_pins):

# Function inputs:
#       laser_trigger_pin = the GPIO pin number connected to fire the laser
#       fluorescence_camera_pins = an array of 3 pins used to for the fluoresence camera
#                                                                       (trigger, SYNC-A, SYNC-B)
#       usb_information = a list containing the information used to set up the usb for controlling the Newport stages
#                                                       (USB address (str),timeout (flt), baud rate (int), byte size (int), parity (char), stop bits (int), xonxoff (bool))


	# Initialises fastpins module
	try:
		fp.init()
	except Exception as inst:
		print('Error setting up fastpins module.')
		print(inst)
		return 1

	# Sets up laser trigger pin
	try:
		fp.setpin(laser_trigger_pin, 1, 0) #PUD resistor needs to be specified but will be ignored in setup
	except Exception as inst:
		print('Error setting up laser pin.')
		print(inst)
		return 2

	# Sets up fluorescence camera pins
	try:
		fp.setpin(fluorescence_camera_pins[0],1,0)      #Trigger
		fp.setpin(fluorescence_camera_pins[1],0,0)      #SYNC-A
		fp.setpin(fluorescence_camera_pins[2],0,0)      #SYNC-B
	except Exception as inst:
		print('Error setting up fluorescence camera pins.')
		print(inst)
		return 3

	return 0

#Triggers both the laser and fluorescence camera (assumes edge trigger mode by default)
def trigger_fluorescence_image_capture(delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=True, duration=100):

	# Function inputs:
	#               delay = delay time (in microseconds) before the image is captured
	#               laser_trigger_pin = the pin number (int) of the laser trigger
	#               fluorescence_camera_pins = an int array containg the triggering, SYNC-A and SYNC-B pin numbers for the fluorescence camera
	#
	# Optional inputs:
	#               edge_trigger:
	#                       True = the fluorescence camera captures the image once detecting the start of an increased signal
	#                       False = the fluorescence camera captures for the duration of the signal pulse (pulse mode)
	#               duration = (only applies to pulse mode [edge_trigger=False]) the duration (in microseconds) of the pulse

	# Captures an image in edge mode
	if edge_trigger:

		fp.edge(delay, laser_trigger_pin, fluorescence_camera_pins[0], fluorescence_camera_pins[2])

	# Captures in trigger mode
	else:

		fp.pulse(delay, duration, laser_trigger_pin, fluorescence_camera_pins[0])

if __name__ == '__main__':

	laser_trigger_pin = 22
	fluorescence_camera_pins = (8,10,12)
	delay = 400000
	duration = 20000


	if init_controls(laser_trigger_pin,fluorescence_camera_pins) == 0:

		# Tests laser and camera
		for i in range(10):

			trigger_fluorescence_image_capture(delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=False, duration=duration)
