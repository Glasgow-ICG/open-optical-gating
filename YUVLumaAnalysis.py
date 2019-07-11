# Python imports
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

class YUVLumaAnalysis(array.PiYUVAnalysis):

	#Custom class to convert and analyse Y (luma) channel of each YUV frame.
	#Extends the picamera.array.PiYUVAnalysis class, which has a stub method called analze that is overidden here.


	def __init__(self, camera, ref_frames = None, frame_num = 1, make_refs = True, size=None):

		#initialises relevant variables as class attributes. This is the best way i could think of to pass
		#any external variables needed for analysis to this class (is there a better way? Not sure in Python)
		#camera = object representing PiCam,
		#frame = optiononal input to tell the object which frame in a sequence we are on

		super(YUVLumaAnalysis, self).__init__(camera)
		self.frame_num = frame_num
		self.make_refs = make_refs
		self.width, self.height = camera.resolution
		self.timer = 0 #measure how long each frame takes to capture in case this is necesary for simulator

		# Defines the arrays for the timestamp, phase and sad
		self.temp_ary_length = 680
		self.ref_frame_length = 20
		self.sad = np.zeros((self.temp_ary_length,self.ref_frame_length))
		self.phase = np.zeros(self.temp_ary_length)
		self.timestamp = np.zeros(self.temp_ary_length)


		if make_refs:

			# Obtains a period of reference frames (if specified)
			self.ref_frames = jps.doEstablishPeriodProcessingForFrame(ref_frames,settings)
				# sequence is 3D stack in form of ref_frames[t,x,y]
# - Determine all settings parameters and the ability to update them
		else:
			self.ref_frames = ref_frames


	def analyze(self, frame):

		start = time.time()
		frame = frame[:,:,0] # Select just Y values from frame

		#method to analyse each frame as they are captured by the camera. Must be fast since it is running within
		#the encoder's callback, and so must return before the next frame is produced.

		# For the first set of frames (if make_refs is True), assigns current frame as a reference frame
		if self.frame_num<self.ref_frame_length and self.make_refs:
			self.ref_frames[self.frame_num,:,:] = frame

		else:
			# Gets the phase and sad of the current frame (settings ignored until format is known)
			self.phase, self.sad[self.frame_num,:],_ = jps.compareFrame(frame, self.ref_frames)

			# Gets the current timestamp
			self.timestamp[self.frame_num] = time.localtime()


		self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame
		end = time.time()
		self.timer += (end - start)

# Function that initialises various controlls (pins for triggering laser and fluorescence camera along with the USB for controlling the Newport stages)
def init_controls(laser_trigger_pin, fluorescence_camera_pins, usb_information):

	# Function inputs:
	#	laser_trigger_pin = the GPIO pin number connected to fire the laser
	#	fluorescence_camera_pins = an array of 3 pins used to for the fluoresence camera
	#									(trigger, SYNC-A, SYNC-B)
	#	usb_information = a list containing the information used to set up the usb for controlling the Newport stages
	#							(USB address (str),timeout (flt), baud rate (int), byte size (int), parity (char), stop bits (int), xonxoff (bool))

	# To-do:
	#	- Check pins to see if set up correctly
	#	- Check set stages in correct mode
	#	- Set up stage parameters (increment, acceleration, velocity etc)
	#	- Return information regarding set up (all successful/ all failed)

	
	# Initialises fastpins module
	fp.init()

	# Sets up laser trigger pin
	fp.setpin(laser_trigger_pin, 1, 0) #PUD resistor needs to be specified but will be ignored in setup

	# Sets up fluorescence camera pins
	fp.setpin(fluorescence_camera_pins[0],1,0) #Trigger
	fp.setpin(fluorescence_camera_pins[1],0,0)	#SYNC-A
	fp.setpin(fluorescence_camera_pins[2],0,0)	#SYNC-B

	# Sets up USB for Newport stages
	ser = serial.Serial(usb_information[0],
						timeout=usb_information[1],
						baudrate=usb_information[2],
						bytesize=usb_information[3],
						parity=usb_information[4],
						stopbits=usb_information[5],
						xonxoff=usb_information[6])

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

		fp.edge(delay, laser_trigger_pin, fluorescence_camera_pins[0], fluorescence_camera_pins[1])

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
def get_period(sequenceName):

	os.system('python3 getPeriod.py' + sequenceName)

# Synchronises the capture of a full zebrafish heart.
def caputre_full_heart(resolution,framerate):

	# Sets up picamera and YUVLumaAnalysis object
	camera = picamera.PiCamera()
	camera.resolution = (resolution)
	camera.framerate = framerate

	# Caputres reference data (for now just uses sample data)
	get_period('sample_period')

	brightfield_stream = YUVLumaAnalysis(camera)

	# predictTrigger(frameSummaryHistory, settings, fitBackToBarrier=True, log=False, output="seconds"))
	# frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
	# phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
	# targetSyncPhase should be in [0,2pi]


for i in range(20):
	camera = picamera.PiCamera()
	res = (i+8)*32
	camera.resolution = (res,res)
	camera.framerate = framerate

	output = YUVLumaAnalysis(camera)

	start = time.time()
	camera.start_recording(output, format = 'yuv')
	camera.wait_recording(10)
	camera.stop_recording()
	end = time.time()
