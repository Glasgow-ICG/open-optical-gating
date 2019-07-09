# Python imports
import picamera
import numpy as np
import j_py_sad_correlation as jps
import matplotlib.pyplot as plt
import io
from picamera import array
import time
import os


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


# Empty trigger function that just prints the wait time <- Will be replaced with real trigger function
def trigger_fluorescence_image_capture(wait_time):

	return wait_time

# Empty move function that moves the object through the light sheet
def move(increment):

	print('Moved by %0.1f' % increment)

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
