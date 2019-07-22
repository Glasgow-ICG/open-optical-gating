#Python imports
import picamera
import numpy as np
import io
from picamera import array
import time
import os
import sys
import serial
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

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


	def __init__(self, camera=None, usb_serial=None, brightfield_framerate=80, laser_trigger_pin=22 , fluorescence_camera_pins=(8,10,12), plane_address=1, encoding='utf-8', terminator=chr(13)+chr(10), increment=0.0005, negative_limit=0, positive_limit=0.075, current_position=0, frame_buffer_length=80, ref_frames = None, frame_num = 0,  size=None):


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

		super(YUVLumaAnalysis, self).__init__(camera)
		self.frame_num = frame_num
		if camera is not None:
			self.width, self.height = camera.resolution
		else:
			self.width, self.height = (256,256)

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
		self.frame_buffer_length = frame_buffer_length
#		self.sad = np.zeros(self.temp_ary_length)
		self.frameSummaryHistory = np.empty((self.frame_buffer_length,3))
		self.dtype = 'uint8'

		# Variable for emulator
		self.live = True

		# Initialises reference frames if not specified
		if ref_frames is None:

			self.ref_frames = np.empty((self.frame_buffer_length, self.height, self.width), dtype=self.dtype)
			self.get_period_status = 2

		else:
			self.ref_frames = ref_frames
			self.get_period_status = 0
			self.settings = hlp.initialiseSettings(framerate=self.framerate, referencePeriod=ref_frames.shape[0])
			self.settings = rts.deduceBarrierFrameArray(self.settings)
			print(self.settings)
		


	def analyze(self, frame):

		frame = frame[:,:,0]
		#method to analyse each frame as they are captured by the camera. Must be fast since it is running within
		#the encoder's callback, and so must return before the next frame is produced.

		# Ensures stage is always within user defined limits
		if self.current_position <= self.positive_limit and self.current_position >= self.negative_limit:

			# Captures a set of reference frames for obtaining a reference period
			if self.get_period_status == 2:

				# Obtains a minimum amount of reference frames
				if self.frame_num < self.frame_buffer_length:

					# Adds current frame to reference
					self.ref_frames[self.frame_num,:,:] = frame
					
					# Increases frame number
					self.frame_num += 1

				# Once a suitible reference size has been obtained, gets a period and the user selects the phase
				else:

					# Obtains a reference period
					self.ref_frames, self.settings = get_period(self.ref_frames,{}, framerate=self.framerate)
					(self.settings).update({'framerate':self.framerate})

					# User selects the period
					self.settings, self.get_period_status = select_period(self.ref_frames,self.settings)

					# If user is happy with period
					if self.get_period_status == 0:

						self.frame_num = 0
						self.ref_frames = np.empty((self.frame_buffer_length, self.height, self.width), dtype=self.dtype)
						print(self.settings)
						self.initial_process_time = time.time()

			# Clears ref_frames and resets frame number to reselect period
			elif self.get_period_status == 1:

				# Resets frame number
				self.frame_num = 0
				self.ref_frames = np.empty((self.frame_buffer_length, self.height, self.width), dtype=self.dtype)
				self.get_period_status = 2


			# Once period has been selected, analyses brightfield data for phase triggering
			else:

				# Clears framerateSummaryHistory if it exceeds the reference frame length
				if self.frame_num >= self.frame_buffer_length:
					self.frameSummaryHistory = np.empty((self.frame_buffer_length,3))
					self.frame_num = 0

				else:
					# Gets the phase and sad of the current frame 
					pp, self.sad, self.settings = rts.compareFrame(frame, self.ref_frames, settings = self.settings)
					pp = ((pp-self.settings['numExtraRefFrames'])/self.settings['referencePeriod'])*(2*np.pi)#convert phase to 2pi base

					# Gets the current timestamp
					tt = (time.time() - self.initial_process_time)*1000 # Converts time into milliseconds

					# Arrhythmia
					if self.frame_num != 0:
						t0 = self.frameSummaryHistory[0,0]
						t1 = self.frameSummaryHistory[self.frame_num -1, 0]
						p1 = self.pp_old
						expected = p1 + ((t1-t0)/(self.settings['referencePeriod']/self.settings['framerate']) - (t1-t0)//(self.settings['referencePeriod']/self.settings['framerate']))-((tt-t0)/(self.settings['referencePeriod']/self.settings['framerate']) - (tt-t0)//(self.settings['referencePeriod']/self.settings['framerate']))
						residual = pp - expected

						# Fix for wrapping
						if residual > np.pi:
							residual = residual - 2*np.pi
						elif residual < -np.pi:
							residual = residual + 2*np.pi
						residual = abs(residual)
					else:
						residual = 0

					# Cumulative Phase
					if self.frame_num != 0:
						wrapped = False
						deltaPhase = pp-self.pp_old
						while deltaPhase<-np.pi:
							wrapped = True
							deltaPhase+= 2*np.pi
						self.frameSummaryHistory[self.frame_num, 1] = self.frameSummaryHistory[self.frame_num -1,1] + deltaPhase
					else:
						self.frameSummaryHistory[self.frame_num, 1] = pp 
						

					# Gets the argmin of SAD and adds to frameSummaryHistory array
					self.frameSummaryHistory[self.frame_num,2] = np.argmin(self.sad)

					# Doesn't predict if haven't done on whole period
					if self.frame_num > self.settings['referencePeriod']:

						# Gets the trigger response
						trigger_response =  rts.predictTrigger(self.frameSummaryHistory, self.settings, fitBackToBarrier=True, log=False, output="seconds")
						# frameSummaryHistory is an nx3 array of [timestamp, phase, argmin(SAD)]
						# phase (i.e. frameSummaryHistory[:,1]) should be cumulative 2Pi phase
						# targetSyncPhase should be in [0,2pi]

						# Captures the image  and then moves the stage if triggered
						if trigger_response > 0 and self.live:

							trigger_response *= 1e6 # Converts to microseconds
							print('Triggering in ' + str(trigger_response) +'us')
							trigger_fluorescence_image_capture(trigger_response,self.laser_trigger_pin, self.fluorescence_camera_pins, edge_trigger=False, duration=20000)

							stage_result = scf.move_stage(self.usb_serial, self.plane_address,self.increment, self.encoding, self.terminator)

							# Do something with the stage result:
							#	0 = Continue as normal
							#	1 or 2 = Pause capture

						# Returns the trigger response, phase and timestamp for emulated data
						else:
							return trigger_response,pp,tt

					self.pp_old = float(pp)
					self.frame_num += 1    #keeps track of which frame we are on since analyze is called for each frame


	# Function to emulate live data
	def emulate(self, video_file):

		# Defines initial variables and objects
		emulated_data_set = cv2.VideoCapture(video_file)
		timestamp = []
		phase = []
		process_time = []
		self.width, self.height = (int(emulated_data_set.get(3)), int(emulated_data_set.get(4)))
		self.ref_frames = np.empty((self.frame_buffer_length, self.height, self.width), dtype=self.dtype)
		self.live = False

		for i in tqdm(range(300)):

			ret, frame = emulated_data_set.read()
			frame = np.array(frame)

			if self.get_period_status == 0 and self.frame_num > self.settings['referencePeriod']:
				time_init = time.time()
				trigger_response, pp, tt = self.analyze(frame)
				time_fin = time.time()	

				process_time.append(time_fin - time_init)
				timestamp.append(tt)
				phase.append(pp)
			else:
				self.analyze(frame)

		plt.subplot(2,1,1)
		plt.title('Phase vs time (ms)')
		plt.plot(phase,timestamp)

		plt.subplot(2,1,2)
		plt.title('Histogram of frame processing time.')
		plt.hist(process_time)

		plt.show()

		print(phase)
		print(timestamp)
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
def get_period(brightfield_sequence, settings, framerate=80, minFramesForFit=5, maxRecievedFramesForFit=80):

	# Function inputs
	#		brightfield_sequence = (numpy array) a 3D array of the brightfiled picam data
	#		settings = the settings

	# If the settings are empty creates settings
	if not settings:
		settings = hlp.initialiseSettings(framerate=framerate, referencePeriod=brightfield_sequence.shape[0], minFramesForFit=minFramesForFit)

	# Calculates period from getPeriod.py
	brightfield_period, settings = gp.doEstablishPeriodProcessingForFrame(brightfield_sequence, settings)
	settings = rts.deduceBarrierFrameArray(settings)

	# Checks if period_data dir exists and if not creates dir
	if os.path.isdir('period_data') != True:
		os.mkdir('period_data')

	# Saves the period
	if isinstance(brightfield_period, int) == False:

		for i in range(brightfield_period.shape[0]):
			
			cv2.imwrite(('period_data/'+str(i)+'.tiff'), brightfield_period[i,:,:])

	return brightfield_period, settings


# Selects the period from a set of reference frames
def select_period(brightfield_period_frames, settings, framerate=80):

	# Function inputs:
	#	brightfield_period_frames = a 3D array consisting of evenly spaced frames containing exactly one period
	#	settings = the settings dictionary (for more information see the helper.py file)

	# Optional inputs:
	#	framerate = the framerate of the brightfield picam (float or int)

	# Defines initial variables
	period_length_in_frames = brightfield_period_frames.shape[0]

	# For now it is a simple command line interface (which is not helpful at all)
	frame = int(input('Please select a frame between 0 and '+str(period_length_in_frames - 1)+'\nOr enter -1 to select a new period.\n'))

	# Checks if user wants to select a new period. Users can use their creative side by selecting any negative number.
	if frame < 0:

		return settings, 1
	
	settings = hlp.updateSettings(settings,referenceFrame=frame)

	return settings, 0

# Main script to capture a zebrafish heart

if __name__ == '__main__':

	# Defines initial variables
	laser_trigger_pin = 22
	fluorescence_camera_pins = 8,10,12 # Trigger, SYNC-A, SYNC-B
	usb_information = ('/dev/ttyUSB0',0.1,57600,8,'N',1,True)	#USB address, timeout, baud rate, data bits, parity, Xon/Xoff

	brightfield_resolution = 256 
	brightfield_framerate = 40 

	analyse_time = 10000 

	# Enters emulate mode for testing
	emulate_mode = True
	emulate_data_set = 'sample_data.h264'

	if not emulate_mode:

		# Sets up basic picam
		camera = picamera.PiCamera()
		camera.framerate = brightfield_framerate
		camera.resolution = (brightfield_resolution,brightfield_resolution)

		# Starts preview
		camera.start_preview(fullscreen=False, window = (900,20,640,480))

		# Sets up pins and usb
		usb_serial = init_controls(laser_trigger_pin, fluorescence_camera_pins, usb_information)

		# Checks if usb_serial has recieved an error code
		if isinstance(usb_serial, int):
			print('Error code '+str(usb_serial))
			sys.exit()

		# Sets up stage to recieve input
		#neg_limit, pos_limit, current_position = scf.set_user_stage_limits(usb_serial,plane_address,encoding,terminator)

		input('Press any key once the heart is in position.')
		# Generate fake reference frame set
		# dummy_reference_frames = np.random.randint(0,high=128, size=(10,brightfield_resolution, brightfield_resolution),dtype=np.uint8)

		# Sets up YUVLumaAnalysis object
		analyse_camera = YUVLumaAnalysis(camera=camera, brightfield_framerate=brightfield_framerate, usb_serial=usb_serial)

		# Starts analysing brightfield data
		camera.start_recording(analyse_camera, format = 'yuv')
		camera.wait_recording(analyse_time)
		camera.stop_recording()

		# Ends preview
		camera.stop_preview()

	else:
		
		analyse_camera = YUVLumaAnalysis()
		analyse_camera.emulate(emulate_data_set)
