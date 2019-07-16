#Python imports
import picamera
import numpy as np
import matplotlib.pyplot as plt
import io
from picamera import array
import time
import os

# Local imports
import j_py_sad_correlation as jps
import fastpins as fp
import getPeriod as gp
import realTimeSync.py as rts

class YUVLumaAnalysis(array.PiYUVAnalysis):

	#Custom class to convert and analyse Y (luma) channel of each YUV frame.
	#Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analze that is overidden here.


	def __init__(self, camera, laser_trigger_pin, fluorescence_camera_pins, usb_serial, plane_address, encoding, terminator, increment, ref_frames = None, frame_num = 0, make_refs = True, size=None):


		# Function inputs:
		#	camera = the raspberry picam PiCamera object
		#	laser_trigger_pin = the pin number (int) of the laser trigger
		#	fluorescence_camera_pins = an array (int) of fluorescence camera pin numbers containg (trigger,SYNC-A, SYNC-B)
		#	usb_serial = the usb serial object for controlling the movement stages
		#	plane_address = the address of the stage that moves the zebrafish through the light sheet
		#	encoding = the encoding used to control the Newport stages (usually utf-8)
		#	terminator = the character set used to terminate a command sent to the Newport stages
		#	increment = the required increment to move the stage by after each image capture
		
		# Optional inputs:
		#	ref_frames = a set of reference frames containg a whole period for the zebrafish
		# 	frame_num = the current frame number
		#	make_refs = whether a period needs to be obtained
		# 	size = ??? <- Need to find out what this does


		# To-do:
		#	Find out what the size parameter does

		super(YUVLumaAnalysis, self).__init__(camera)
		self.frame_num = frame_num
		self.make_refs = make_refs
		self.width, self.height = camera.resolution
		self.timer = 0 #measure how long each frame takes to capture incase this is necesary for simulator

		# Defines laser, fluorescence camera and usb serial information
		self.laser_trigger_pin = laser_trigger_pin
		self.fluorescence_camera_pins = fluorescence_camera_pins
		self.usb_serial = usb_serial
		self.plane_address = plane_address
		self.encoding = encoding
		self.terminator = terminator
		self.increment = increment

		# Defines the arrays for sad and frameSummaryHistory (which contains period, timestamp and argmin(sad))
		self.temp_ary_length = 680
		self.ref_frame_length = 20
		self.sad = np.zeros(self.temp_ary_length)
		self.frameSummaryHistory = np.empty((self.ref_frame_length,3))


		# Gets period refence frames
		if make_refs:

			# Obtains a period of reference frames (if specified)
			self.ref_frames, self.settings = get_period(ref_frames,settings)
				# sequence is 3D stack in form of ref_frames[t,x,y]

		else:
			self.ref_frames = ref_frames

		# User selects period
		self.settings = select_period(self.ref_frames, self.settings)		

	def analyze(self, frame):


		start = time.time()
		frame = frame[:,:,0] # Select just Y values from frame

		#method to analyse each frame as they are captured by the camera. Must be fast since it is running within
		#the encoder's callback, and so must return before the next frame is produced.

		# Clears framerateSummaryHistory if it exceeds the reference frame length
		if self.frame_num >= self.ref_frame_length:
			self.frameSummaryHistory = np.empty((self.ref_frame_length,3))
			self.frame_num = 0

		# For the first set of frames (if make_refs is True), assigns current frame as a reference frame
		if self.frame_num < self.ref_frame_length and self.make_refs:
			self.ref_frames[self.frame_num,:,:] = frame

			self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame
			end = time.time()
			self.timer += (end - start)

		else:
			# Gets the phase and sad of the current frame (settings ignored until format is known)
			self.frameSummaryHistory[self.frame_num,1], self.sad_ = jps.compareFrame(frame, self.ref_frames)

			# Gets the current timestamp
			self.frameSummaryHistory[self.frame_num,0] = time.localtime()

			# Gets the argmin of SAD and adds to frameSummaryHistory array
			self.frameSummaryHistory[self.frame_num,2] = np.argmin(self.sad)

			self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame
			end = time.time()
			self.timer += (end - start)

			# Gets the trigger response
			trigger_response =  rts.predictTrigger(self.frameSummaryHistory, self.settings, fitBackToBarrier=True, log=False, output="seconds"))
			# frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
			# phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
			# targetSyncPhase should be in [0,2pi]
			
			# Captures the image (in edge mode) and then moves the stage if triggered
			if trigger_response > 0:

				trigger_response *= 1e6 # Converts to microseconds
				trigger_fluorescence_image_capture(trigger_response,self.laser_trigger_pin, self.fluorescence_trigger_pins)	

				#time.sleep(1) # Allows time for successful image capture
				stage_result = move_stage(self.usb_serial, self.plane_address,self.increment self.encoding, self.terminator)




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
	except:
		print('Error setting up fastpins module.')
		return 1

	# Sets up laser trigger pin
	try:
		fp.setpin(laser_trigger_pin, 1, 0) #PUD resistor needs to be specified but will be ignored in setup
	except:
		print('Error setting up laser pin.')
		return 2

	# Sets up fluorescence camera pins
	try:
		fp.setpin(fluorescence_camera_pins[0],1,0) 	#Trigger
		fp.setpin(fluorescence_camera_pins[1],0,0)	#SYNC-A
		fp.setpin(fluorescence_camera_pins[2],0,0)	#SYNC-B
	except:
		print('Error setting up fluorescence camera pins.')
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
	except:
		print('Error setting up usb.')
		return 4

	# Serial object is the only new object
	return ser


# A function that ensures the stages are set to the correct mode
def set_stages_to_recieve_input(ser,address,encoding,terminator):
	
	# Function inputs:
	#	ser = the serial object for the usb (already set up in initialisation function)
	#	address = the address number (int) of the stage to ensure is in the correct state
	#	encoding = the encoding needed for the stage controllers (usually utf-8)
	#	terminator = the terminating characters sent at the end of each command


	# To-do:
	#	- Checks address is correct
	#	- Perform error checks rather than printing responses
	#	- Set up stage parameters (increment, acceleration, velocity etc)

	# Information:
	#	The easiest way to do this is to reset each controller on start up so the initial state is known.
	#	Currently, all initial parameters are left but if they need to be changed should be done so in this function.

	#Command list
	reset = 'RS'
	ready = 'OR'
	config = 'PW'
	request_info = 'VE'

	# Checks address is correct
	command = str(address)+str(request_info)+str(terminator)
	ser.write(command.encode(encoding))
	response = (ser.readline())
	if not response:
		print('No information recieved when requesting version information.\nCommand sent: '+command)
		return 1

	# Resets the controllers
	command = str(address)+str(reset)+str(terminator)
	ser.write(command.encode(encoding))
	response = (ser.readline())
	print(response.decode(encoding))


	# Enters config stage (currently does nothing but here for easy of adding configuration"
	command = str(address)+str(config)+str(0)+str(terminator)
	ser.write(command.encode(encoding))
	response = (ser.readline())
	print(response.decode(encoding))
	
	# Leaves config state
	command = str(address)+str(config)+str(1)+str(terminator)
	ser.write(command.encode(encoding))
	response = (ser.readline())
	print(response.decode(encoding))
	
	# Readies controller
	command = str(address)+str(ready)+str(terminator)
	ser.write(command.encode(encoding))
	response = (ser.readline())
	print(response.decode(encoding))
	
	return 0


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



# Moves stage by an increment to move the zebrafish through the plane
def move_stage(ser, address, increment, encoding, terminate):

	# Function inputs
	#		ser = the serial object return by the init_controls function
	#		address = the address of the stage to be moved
	#		increment = the increment to move the stage by (float)
	#		encoding = the type of encoding to be used to send/recieve commands (usually 'utf-8') (str)
	#		terminate = the termination character set (str)

	# To-dos:
	#		- Function should check errors errors sent from stage
	#		- Function should check what mode controller is in and change to ready mode (if not in ready mode)
	#		- Function should return current position and error in current position

	# Gets the current stage position
	command = str(address) + 'TP?' + str(terminate)
	ser.write(command.encode(encoding))
	response = (ser.readline()).decode(encoding)

	# Tests if response present and if so extracts position information and tests move validity
	if not response:

		print('Error. No response obtained from stage '+str(address)+'.\nCheck stage is switched on and responsive.')

		return 1

	else:
		current_position = float(response.remove(str(address)+'TP')) #Extracts position from response

		# Gets negative software limit
		command = str(address)+'SL?'+str(terminate)
		ser.write(command.encode(encoding))
		response = (ser.readline()).decode(encoding)
		negative_software_limit = float(response.remove(str(address)+'SL'))

		#Gets positive software limit
		command = str(address)+'SR?'+str(terminate)
		ser.write(command.encode(encoding))
		response = (ser.readline()).decode(encoding)
		positive_software_limit = float(response.remove(str(address)+'SR'))

		# Checks if movement request is within software limit
		if (current_position + float(increment)) > negative_software_limit and (current_position + float(increment)) < positive_software_limit :

			command = str(address)+'PR'+str(increment)+str(terminate)
			ser.write(command.encode(encoding))

			return 0

		else:

			print('Error. Cannot move stage by increment '+str(increment)+ ' as would be outside software limits of '+str(negative_software_limit) + ' - ' + str(positive_software_limit) )
			return 2


# Gets the period from sample set
def get_period(brightfield_sequence, settings):

	# Function inputs
	#		brightfield_sequence = (numpy array) a 3D array of the brightfiled picam data
	#		settings = the settings

	# If the settings are empty creates settings
	if not settings:
		    settings = hlp.initialiseSettings(drift=[-5,-2],
		                            framerate=80,
		                            referencePeriod=42.410156325856882,
		                            barrierFrame=4.9949327669365964,
		                            predictionLatency=0.01,
		                            referenceFrame=20.994110773833857)

	# Calculates period from getPeriod.py
	brightfield_period, settings = gp.doEstablishPeriodProcessingForFrame(brightfiled_sequence, settings)

	return brightfield_period, settings


# Selects the period from a set of reference frames
def select_period(brightfield_period_frames, settings):

	# Function inputs:
	#	brightfield_period_frames = a 3D array consisting of evenly spaced frames containing exactly one period
	#	settings = the settings dictionary (for more information see the helper.py file

	# Defines initial variables
	period_length_in_frames = brightfield_period_frames.shape[0]

	# For now it is a simple command line interface (which is not helpful at all)
	frame = int(input('Please select a frame between 0 and '+str(period_length_in_frames - 1)))

	# Converts frame number to period
	period = 2*np.pi*frame/period_length_in_frames

	# Updates settings dictionary with reference data
	settings.update({'referenceFrame':frame})
	settings.update({'referencePeriod':period_length_in_frames})
	settings.update({'targetSyncPhase':period})

	return settings

# Main script to capture a zebrafish heart

if __name__ == '__main__':

	# Defines initial variables
	laser_trigger_pin = 22
	fluorescence_camera_pins = 8,10,12 # Trigger, SYNC-A, SYNC-B
	usb_information = ('/dev/ttyUSB0',0.1,57600,8,None,1,True)	#USB address, timeout, baud rate, data bits, parity, Xon/Xoff

	econding = 'utf-8'
	terminator = chr(13)+chr(10)
	increment = 1

	brightfield_resolution = 256
	brightfield_framerate = 20

	analyse_time = 100

	# Sets up pins and usb
	usb_serial = init_controls(laser_trigger_pin, fulorescence_camera_pins, usb_information)

	# Sets up stage to recieve input
	set_stages_to_recieve_input(usb_serial,plane_address,encoding,terminator)

	# Sets up brightfield camera and YUVLumaAnalysis object
	camera = picamera.PiCamera()
	camera.resolution = (brightfield_resolution,brightfield_resolution)
	camera.framerate = brightfield_framerate
	analyse_camera = YUVLumaAnalysis(camera,laser_trigger_pin, fluorescence_camera_pins, usb_serial, plane_address, encoding, terminator, increment)
	
	# Starts analysing brightfield data
	start = time.time()
	camera.start_recording(output, format = 'yuv')
	camera.wait_recording(analyse_time)
	camera.stop_recording()
	end = time.time()
