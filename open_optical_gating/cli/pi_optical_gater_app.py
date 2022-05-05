"""Extension of CLI Open Optical Gating System for using a Raspberry Pi and PiCam NoIR"""

# Python imports
import sys
import json
import time

# Module imports
import numpy as np
from loguru import logger

# Raspberry Pi-specific imports
import picamera
from picamera.array import PiYUVArray

# Fastpins module
import fastpins as fp

# Local imports
from . import optical_gater_server as server
from . import pixelarray as pa

class PiAnalysisWrapper(picamera.array.PiYUVAnalysis):
    # This wrapper exists because we need something to be a subclass of PiYUVAnalysis,
    # whereas the main PiOpticalGater class needs to be a subclass of OpticalGater
    # This is just a very lightweight stub, and all the camera configuration etc is
    # still dealt with by PiOpticalGater
    def __init__(self, camera, size=None, gater=None):
        super().__init__(camera, size=size)
        self.gater = gater
        self.array = None
        self.counter = 0

    def analyze(self, array):
        # Callback from picamera.array.PiYUVAnalysis
        if (self.gater is not None):
            self.array = array[:, :, 0] # Y channel
            if not self.array.all() == 0:
                self.pixelarray = pa.PixelArray(
                    self.array,  
                    metadata={
                        "timestamp": time.time() - self.gater.start_time
                    },  # relative to start_time to sanitise
                )
                # Call through to the analysis method of PiOpticalGater
                self.gater.analyze_pixelarray(self.pixelarray)
 
class PiOpticalGater(server.OpticalGater):
    """Extends the optical gater server for the Raspberry Pi."""

    def __init__(
        self,
        settings=None,
        ref_frames=None,
        ref_frame_period=None,
        automatic_target_frame_selection=True,
    ):
        """Function inputs:
        settings      dict  Parameters affecting operation (see optical_gating_data/json_format_description.md)
        """
        # Initialise parent
        super(PiOpticalGater, self).__init__(
            settings=settings,
            ref_frames=ref_frames,
            ref_frame_period=ref_frame_period,
        )
        # By default we will take a guess at a good target frame (True)
        # rather than ask user for their preferred initial target frame (False)
        self.automatic_target_frame_selection = automatic_target_frame_selection
        
    def user_pick_target_frame(self):
        """Prompts the user to select the target frame from a one-period set of reference frames"""
        # Find the number of reference frames and send this to the main flask process
        ref_sequence_length = str(len(self.ref_seq_manager.ref_frames) - 1)
        self.refActivateQueue.put(ref_sequence_length)
        
        # Wait for a reference frame selection from the flask process
        # Or until the user presses 'Abort' (which writes to the stopQueue)
        while True:
            if not self.refSelectQueue.empty():
                choice = int(self.refSelectQueue.get())
                print("The user has selected frame {0}".format(choice))
                return choice, None
            elif not self.stopQueue.empty():
                self.stop = True
                return

    def setup_pins(self):
        """Initialise Raspberry Pi hardware I/O interfaces."""     
        # Initialise Pins for Triggering
        logger.debug("Initialising triggering hardware...")
        # Initialises fastpins module
        try:
            logger.debug("Initialising fastpins...")
            fp.init()
        except Exception as inst:
            logger.critical("Error setting up fastpins module. {0}", inst)
        # Sets up trigger pin (for laser or BNC trigger)
        laser_trigger_pin = self.settings["trigger"]["laser_trigger_pin"]
        if laser_trigger_pin is not None:
            try:
                logger.debug("Initialising BNC trigger pin... (Laser/Trigger Box)")
                # PUD resistor needs to be specified but will be ignored in setup
                fp.setpin(laser_trigger_pin, 1, 0)
            except Exception as inst:
                logger.critical("Error setting up laser pin. {0}", inst)
        # Sets up fluorescence camera pins
        fluorescence_camera_pins = self.settings["trigger"]["fluorescence_camera_pins"]
        if fluorescence_camera_pins is not None:
            try:
                logger.debug("Initialising fluorescence camera pins...")
                fp.setpin(fluorescence_camera_pins["trigger"], 1, 0)
                fp.setpin(fluorescence_camera_pins["SYNC-A"], 0, 0)
                fp.setpin(fluorescence_camera_pins["SYNC-B"], 0, 0)
            except Exception as inst:
                logger.critical("Error setting up fluorescence camera pins. {0}", inst)

    def frame_rate_calculator(self, windowSize = 10):
        """Compute average live framerate over timestamp window."""
        self.timeWindow.append(self.currentTimeStamp)
        # Calculate the framerate once enough timestamps have been recorded (windowSize)
        if len(self.timeWindow) > windowSize:
                
            # Delete the first timestamp
            del self.timeWindow[0]
            
            # Compute the window averaged framerate
            timeDifferences = [self.timeWindow[i] - self.timeWindow[i - 1] for i in range(1, windowSize)]
            self.framerate = np.round(1/np.mean(timeDifferences), 0)
            
    def setup_camera(self):
        """Setup logging and the attributes of the Pi Camera."""
        # Setup logging
        logger.remove()
        logger.add("user_log_folder/oog_{time}.log", level="INFO", format = "{time:YYYY-MM-DD | HH:mm:ss:SSSSS} | {level} | {module}:{name}:{function}:{line} --- {message}")
        logger.enable("open_optical_gating")
        
        # Setup attributes required for framerate calculation
        self.previousRealTime = None
        self.timeWindow = []
        self.framerate = 0
        
        # Initialise the Pi Camera object
        print("Initialising pi camera...")
        self.camera = picamera.PiCamera()
        print("Picamera initialised...")
        
        # Set the properties of the Pi Camera according to the brightfield settings
        self.camera.framerate = self.settings["brightfield"]["brightfield_framerate"]
        self.camera.resolution = (
                self.settings["brightfield"]["brightfield_resolution"], 
                self.settings["brightfield"]["brightfield_resolution"]
                )
        self.camera.awb_mode = self.settings["brightfield"]["awb_mode"]
        self.camera.exposure_mode = self.settings["brightfield"]["exposure_mode"]
        self.camera.shutter_speed = self.settings["brightfield"]["shutter_speed_us"] 
        self.camera.image_denoise = self.settings["brightfield"]["image_denoise"]
        self.camera.contrast = self.settings["brightfield"]["contrast"]
        
        # store the array analysis object for later output processing
        self.analysis = PiAnalysisWrapper(self.camera, gater=self)
        
    def start_sync(self, eventQueue, stopQueue, refActivateQueue, refSelectQueue):
        """
        The function called by the web-server 'Start' button.
        Initialises the Pi Camera, and runs until the time_limit or trigger_limit is reached.
        Inputs:
            eventQueue = multiprocessing Queue object to hold current time, framerate, trigger number, and state
            stopQueue = multiprocessing Queue object. If it is not empty, the sync will halt.
        """
        # Assigning the recieved queues as attributes to be used by other object functions
        self.stopQueue = stopQueue
        self.refActivateQueue = refActivateQueue
        self.refSelectQueue = refSelectQueue
        
        # Run the setup_camera function
        self.setup_camera()
        
        # Log the start time and initiate camera acquisition
        self.start_time = time.time()
        self.camera.start_recording(self.analysis, format="yuv")
        
        # Continue running while the timit_limit and trigger_limit have not been reached
        # and the 'Stop' button has not been pressed by the user
        while (
                time.time() - self.start_time < self.settings["general"]["time_limit_seconds"]
                and self.trigger_num_total <= self.settings["general"]["trigger_limit"]
                and stopQueue.empty()
                ): 
                        
            # The while loop polls the object attributes continuously
            # Attributes will only update as fast as the camera framerate, so care needs to be taken
            # such that the while loop doesn't repeat-record values. 
            # The current time attribute is compared to the most recently polled one.
            # If they are different we record the new information. This could probably be streamlined...
            if (
                self.previousRealTime == None
                or self.previousRealTime != self.currentTimeStamp
                ):
                self.previousRealTime = self.currentTimeStamp
                self.frame_rate_calculator() 
                
                # Provide a console time output to compare to web-server outputs when debugging
                sys.stdout.write("\r Time = {0}".format(np.round(self.currentTimeStamp, 2)))
                
                # Place the relevant data in the multiprocessing eventQueue
                eventQueue.put([self.currentTimeStamp, self.framerate, self.trigger_num_total, self.state])
                
            self.camera.wait_recording(0.001)  
            
        print("\nPI OPTICAL GATER SHUTTING DOWN...")
        
        # Set stop attribute to true to halt parent class loop
        self.stop = True
        # Place an item in stopQueue to halt all concurrent processes
        stopQueue.put(True)
        
        # Stop recording and kill the camera object
        self.camera.stop_recording()
        self.camera.close()
        
        print("PI OPTICAL GATER SHUT DOWN COMPLETELY...")

    def trigger_fluorescence_image_capture(self, trigger_time_s):
        """
        Triggers both the laser and fluorescence camera (assumes edge trigger mode by default) at the specified future time.
        IMPORTANT: this function call is a blocking call, i.e. it will not return until the specified delay has elapsed
        and the trigger has been sent. This is probably acceptable for the RPi implementation, but we should be aware
        that this means everything will hang until the trigger is sent. It also means that camera frames may be dropped
        in the meantime. If all is going well, though, the delay should only be a for a couple of frames.

        Function inputs:
            trigger_time_s = time (in seconds) at which the trigger should be sent
        """
        logger.debug(
            "Sending RPi camera trigger at {0:.6f}s".format(trigger_time_s)
        )
        trigger_mode = self.settings["trigger"]["fluorescence_trigger_mode"]
        if trigger_mode == "edge":
            # The fluorescence camera captures an image when it detects a rising edge on the trigger pin
            fp.edge(
                (trigger_time_s - time.time()) * 1e6,
                self.settings["trigger"]["laser_trigger_pin"],
                self.settings["trigger"]["fluorescence_camera_pins"]["trigger"],
                self.settings["trigger"]["fluorescence_camera_pins"]["SYNC-B"],
            )
        elif trigger_mode == "expose":
            # The fluorescence camera exposes an image for the duration that the trigger pin is high
            fp.pulse(
                (trigger_time_s - time.time()) * 1e6,
                self.settings["trigger"]["fluorescence_exposure_us"],
                self.settings["trigger"]["laser_trigger_pin"],
                self.settings["trigger"]["fluorescence_camera_pins"]["trigger"],
            )
        else:
            logger.critical(
                "Ignoring unknown trigger mode {0}".format(trigger_mode)
            )
