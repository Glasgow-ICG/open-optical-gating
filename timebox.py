#Python imports
import picamera
import numpy as np
import io
from picamera import array
import time
import os
import sys
import serial
import re

# Local imports
import j_py_sad_correlation as jps
import fastpins as fp
import getPeriod as gp
import realTimeSync as rts
import helper as hlp
import stage_control_functions as scf

class YUVLumaAnalysis(array.PiYUVAnalysis):

	#Custom class to convert and analyse Y (luma) channel of each YUV frame.
	#Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analze that is overidden here.


	def __init__(self, camera, brightfield_framerate, laser_trigger_pin, fluorescence_camera_pins, usb_serial, plane_address, encoding, terminator, increment, negative_limit, positive_limit, current_position, ref_frames = None, frame_num = 0,  size=None):


		# Function inputs:
		#	camera = the raspberry picam PiCamera object
		#	laser_trigger_pin = the pin number (int) of the laser trigger
		#	fluorescence_camera_pins = an array (int) of fluorescence camera pin numbers containg (trigger,SYNC-A, SYNC-B)
		#	usb_serial = the usb serial object for controlling the movement stages
		#	plane_address = the address of the stage that moves the zebrafish through the light sheet
		#	encoding = the encoding used to control the Newport stages (usually utf-8)
		#	terminator = the character set used to terminate a command sent to the Newport stages
		#	increment = the required increment to move the stage by after each image capture (float)
		#	negative_limit = the smallest z value (float) of the edge of the zebrafish heart (selected by the user)
		#	positive_limit = the largest z value (float) of the edge of the zebrafish heart (selected by the user)
		#	current_position = the current z value of the stage.

		# Optional inputs:
		#	ref_frames = a set of reference frames containg a whole period for the zebrafish
		# 	frame_num = the current frame number
		# 	size = ??? <- Need to find out what this does


		# To-do:
		#	- Find out what the size parameter does
		# 	- Make sure that period is acquired at the start
		#		- Could do in analyse function only on first instance

		super(YUVLumaAnalysis, self).__init__(camera)
		self.frame_num = frame_num
		self.width, self.height = camera.resolution
		self.framerate = brightfield_framerate

		# Defines laser, fluorescence camera and usb serial information
		self.laser_trigger_pin = laser_trigger_pin
		self.fluorescence_camera_pins = fluorescence_camera_pins
		self.usb_serial = usb_serial
		self.plane_address = plane_address
		self.encoding = encoding
		self.terminator = terminator
		self.increment = increment
		self.negative_limit = negative_limit
		self.positive_limit = positive_limit
		self.current_position = current_position

		# Defines the arrays for sad and frameSummaryHistory (which contains period, timestamp and argmin(sad))
		self.temp_ary_length = 680
		self.ref_frame_length = 20
		self.sad = np.zeros(self.temp_ary_length)
		self.frameSummaryHistory = np.empty((self.ref_frame_length,3))
		self.dtype = 'uint8'

		# Sets the get_period variable to prep the analyse function for acquiring a period
		self.get_period_status = 2

		# Initialises reference frames if not specified
		if not ref_frames:

			self.ref_frames = np.empty((self.ref_frame_length, self.width, self.height), dtype=self.dtype)
		else:
			self.ref_frames = ref_frames


	def analyze(self, frame):

		# To-do:
		#	- Stage captures over heart z values (rather than until software limit)
		#	- Allow time for stage to move and camera to capture?
		# 	- Act on status of stage (stage_result)

		frame = frame[:,:,0] # Select just Y values from frame

		#method to analyse each frame as they are captured by the camera. Must be fast since it is running within
		#the encoder's callback, and so must return before the next frame is produced.

		# Ensures stage is always within user defined limits
		if self.current_position <= self.positive_limit and self.current_position >= self.negative_limit:

			# Captures a set of reference frames for obtaining a reference period
			if self.get_period_status == 2:

				# Obtains a minimum amount of reference frames
				if self.frame_num < self.ref_frame_length:

					# Adds current frame to reference
					self.ref_frames[self.frame_num,:,:] = frame

					# Increases frame number
					self.frame_num += 1

				# Once a suitible reference size has been obtained, gets a period and the user selects the phase
				else:

					# Obtains a reference period
					self.ref_frames, self.settings = get_period(self.ref_frames,{})
					(self.settings).update({'framerate':self.framerate})

					# User selects the period
					self.settings, self.get_period_status = select_period(self.ref_frames,self.settings)

			# Clears ref_frames and resets frame number to reselect period
			elif self.get_period_status == 1:

				# Resets frame number
				self.frame_num = 0
				self.ref_frames = np.empty((self.ref_frame_length, self.width, self.height), dtype=self.dtype)
				self.get_period_status = 2


			# Once period has been selected, analyses brightfield data for phase triggering
			else:

				# Clears framerateSummaryHistory if it exceeds the reference frame length
				if self.frame_num >= self.ref_frame_length:

					self.frameSummaryHistory = np.empty((self.ref_frame_length,3))
					self.frame_num = 0

				else:
					# Gets the phase and sad of the current frame (settings ignored until format is known)
					self.frameSummaryHistory[self.frame_num,1], self.sad_ = jps.compareFrame(frame, self.ref_frames)

					# Gets the current timestamp
					self.frameSummaryHistory[self.frame_num,0] = time.localtime()

					# Gets the argmin of SAD and adds to frameSummaryHistory array
					self.frameSummaryHistory[self.frame_num,2] = np.argmin(self.sad)

					self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame

					# Gets the trigger response
					trigger_response =  rts.predictTrigger(self.frameSummaryHistory, self.settings, fitBackToBarrier=True, log=False, output="seconds")
					# frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
					# phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
					# targetSyncPhase should be in [0,2pi]

					# Captures the image (in edge mode) and then moves the stage if triggered
					if trigger_response > 0:

						trigger_response *= 1e6 # Converts to microseconds
						trigger_fluorescence_image_capture(trigger_response,self.laser_trigger_pin, self.fluorescence_trigger_pins)

						#time.sleep(1) # Allows time for successful image capture
						stage_result, self.current_position = scf.move_stage(self.usb_serial, self.plane_address,self.increment, self.encoding, self.terminator)

						# Do something with the stage result:
						#	0 = Continue as normal
						#	1 or 2 = Pause capture




# Function that initialises various controlls (pins for triggering laser and fluorescence camera along with the USB for controlling the Newport stages)
def init_controls(laser_trigger_pin, fluorescence_camera_pins, usb_information):

	# Function inputs:
	#	laser_trigger_pin = the GPIO pin number connected to fire the laser
	#	fluorescence_camera_pins = an array of 3 pins used to for the fluoresence camera
	#									(trigger, SYNC-A, SYNC-B)
	#	usb_information = a list containing the information used to set up the usb for controlling the Newport stages
	#							(USB address (str),timeout (flt), baud rate (int), byte size (int), parity (char), stop bits (int), xonxoff (bool))


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
		fp.setpin(fluorescence_camera_pins[0],1,0) 	#Trigger
		fp.setpin(fluorescence_camera_pins[1],0,0)	#SYNC-A
		fp.setpin(fluorescence_camera_pins[2],0,0)	#SYNC-B
	except Exception as inst:
		print('Error setting up fluorescence camera pins.')
		print(inst)
		return 3

	# Sets up USB for Newport stages
	try:
		ser = serial.Serial(usb_information[0],
						timeout=usb_information[1],
						baudrate=usb_information[2],
						bytesize=usb_information[3],
						parity=usb_information[4],
						stopbits=usb_information[5],
						xonxoff=usb_information[6])
	except Exception as inst:
		print('Error setting up usb.')
		print(inst)
		return 4

	# Serial object is the only new object
	return ser



# Triggers both the laser and fluorescence camera (assumes edge trigger mode by default)
def trigger_fluorescence_image_capture(delay, laser_trigger_pin, fluorescence_camera_pins, edge_trigger=True, duration=100):

	# Function inputs:
	#		delay = delay time (in microseconds) before the image is captured
	#		laser_trigger_pin = the pin number (int) of the laser trigger
	#		fluorescence_camera_pins = an int array containg the triggering, SYNC-A and SYNC-B pin numbers for the fluorescence camera
	#
	# Optional inputs:
	#		edge_trigger:
	#			True = the fluorescence camera captures the image once detecting the start of an increased signal
	#			False = the fluorescence camera captures for the duration of the signal pulse (pulse mode)
	#		duration = (only applies to pulse mode [edge_trigger=False]) the duration (in microseconds) of the pulse

	# Captures an image in edge mode
	if edge_trigger:

		fp.edge(delay, laser_trigger_pin, fluorescence_camera_pins[0], fluorescence_camera_pins[2])

	# Captures in trigger mode
	else:

		fp.pulse(delay, duration, laser_trigger_pin, fluorescence_camera_pins[0])




# Gets the period from sample set
def get_period(brightfield_sequence, settings):

	# Function inputs
	#		brightfield_sequence = (numpy array) a 3D array of the brightfiled picam data
	#		settings = the settings

	# If the settings are empty creates settings
	if not settings:
		    settings = hlp.initialiseSettings()

	# Calculates period from getPeriod.py
	brightfield_period, settings = gp.doEstablishPeriodProcessingForFrame(brightfield_sequence, settings)

	return brightfield_period, settings


# Selects the period from a set of reference frames
def select_period(brightfield_period_frames, settings):

	# Function inputs:
	#	brightfield_period_frames = a 3D array consisting of evenly spaced frames containing exactly one period
	#	settings = the settings dictionary (for more information see the helper.py file

	# Defines initial variables
	period_length_in_frames = brightfield_period_frames.shape[0]

	# For now it is a simple command line interface (which is not helpful at all)
	frame = int(input('Please select a frame between 0 and '+str(period_length_in_frames - 1)+'\nOr enter -1 to select a new period.\n'))

	# Checks if user wants to select a new period. Users can use their creative side by selecting any negative number.
	if frame < 0:

		return settings, 1

	# Converts frame number to period
	period = 2*np.pi*frame/period_length_in_frames

	# Updates settings dictionary with reference data
	settings.update({'referenceFrame':frame})
	settings.update({'referencePeriod':period_length_in_frames})
	settings.update({'targetSyncPhase':period})

	return settings, 0

# Main script to capture a zebrafish heart

if __name__ == '__main__':

	# Defines initial variables
	laser_trigger_pin = 22
	fluorescence_camera_pins = 8,10,12 # Trigger, SYNC-A, SYNC-B
	usb_information = ('/dev/ttyUSB0',0.1,57600,8,'N',1,True)	#USB address, timeout, baud rate, data bits, parity, Xon/Xoff

	encoding = 'utf-8'
	terminator = chr(13)+chr(10)
	increment = 0.1
	plane_address = 1

	brightfield_resolution = 256
	brightfield_framerate = 20

	analyse_time = 100

	# Sets up pins and usb
	usb_serial = init_controls(laser_trigger_pin, fluorescence_camera_pins, usb_information)

	# Checks if usb_serial has recieved an error code
	if isinstance(usb_serial, int):
		print('Error code '+str(usb_serial))
		sys.exit()

	# Sets up stage to recieve input
	neg_limit, pos_limit, current_position = scf.set_user_stage_limits(usb_serial,plane_address,encoding,terminator)

	# Sets up brightfield camera and YUVLumaAnalysis object
	camera = picamera.PiCamera()
	camera.resolution = (brightfield_resolution,brightfield_resolution)
	camera.framerate = brightfield_framerate
	analyse_camera = YUVLumaAnalysis(camera, brightfield_framerate, laser_trigger_pin, fluorescence_camera_pins, usb_serial, plane_address, encoding, terminator, increment, neg_limit, pos_limit, current_position)

	# Starts analysing brightfield data
	camera.start_recording(analyse_camera, format = 'yuv')
	camera.wait_recording(analyse_time)
	camera.stop_recording()
