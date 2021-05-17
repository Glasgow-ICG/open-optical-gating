"""Extension of CLI Open Optical Gating System for using a Raspberry Pi and PiCam NoIR"""

# Python imports
import sys
import json
import time

# Module imports
import numpy as np
from loguru import logger

# Raspberry Pi-specific imports
# Picamera
import picamera
from picamera.array import PiYUVArray

# Fastpins module
import fastpins as fp

# Local imports
from . import optical_gater_server as server
from . import pixelarray as pa
from . import file_optical_gater


class PiAnalysisWrapper(picamera.array.PiYUVAnalysis):
    # This wrapper exists because we need something to be a subclass of PiYUVAnalysis,
    # whereas the main PiOpticalGater class needs to be a subclass of OpticalGater
    # This is just a very lightweight stub, and all the camera configuration etc is
    # still dealt with by PiOpticalGater
    def __init__(self, camera, size=None, gater=None):
        super().__init__(camera, size=size)
        self.gater = gater

    def analyze(self, array):
        # Callback from picamera.array.PiYUVAnalysis
        if self.gater is not None:
            pixelarray = pa.PixelArray(
                array[:, :, 0],  # Y channel
                metadata={
                    "timestamp": time.time() - self.gater.start_time
                },  # relative to start_time to sanitise
            )
            # Call through to the analysis method of PiOpticalGater
            self.gater.analyze_pixelarray(pixelarray)


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

        # Set-up the Raspberry Pi hardware
        self.setup_pi()

        # By default we will take a guess at a good target frame (True)
        # rather than ask user for their preferred initial target frame (False)
        self.automatic_target_frame_selection = automatic_target_frame_selection

    def setup_pi(self):
        """Initialise Raspberry Pi hardware I/O interfaces."""
        # Initialise PiCamera
        logger.success("Initialising camera...")
        # Camera settings from settings.json
        camera = picamera.PiCamera()
        camera.framerate = self.settings["brightfield_framerate"]
        camera.resolution = (
            self.settings["brightfield_resolution"],
            self.settings["brightfield_resolution"],
        )
        camera.awb_mode = self.settings["awb_mode"]
        camera.exposure_mode = self.settings["exposure_mode"]
        camera.shutter_speed = self.settings["shutter_speed_us"]  # us
        camera.image_denoise = self.settings["image_denoise"]
        # Store key variables for later
        self.height, self.width = camera.resolution
        self.framerate = camera.framerate
        self.camera = camera

        # store the array analysis object for later output processing
        self.analysis = PiAnalysisWrapper(camera, gater=self)

        # Initialise Pins for Triggering
        logger.success("Initialising triggering hardware...")
        # Initialises fastpins module
        try:
            logger.debug("Initialising fastpins...")
            fp.init()
        except Exception as inst:
            logger.critical("Error setting up fastpins module. {0}", inst)
        # Sets up trigger pin (for laser or BNC trigger)
        laser_trigger_pin = self.settings["laser_trigger_pin"]
        if laser_trigger_pin is not None:
            try:
                logger.debug("Initialising BNC trigger pin... (Laser/Trigger Box)")
                # PUD resistor needs to be specified but will be ignored in setup
                fp.setpin(laser_trigger_pin, 1, 0)
            except Exception as inst:
                logger.critical("Error setting up laser pin. {0}", inst)
        # Sets up fluorescence camera pins
        fluorescence_camera_pins = self.settings["fluorescence_camera_pins"]
        if fluorescence_camera_pins is not None:
            try:
                logger.debug("Initialising fluorescence camera pins...")
                fp.setpin(fluorescence_camera_pins["trigger"], 1, 0)
                fp.setpin(fluorescence_camera_pins["SYNC-A"], 0, 0)
                fp.setpin(fluorescence_camera_pins["SYNC-B"], 0, 0)
            except Exception as inst:
                logger.critical("Error setting up fluorescence camera pins. {0}", inst)

        # Initialise frame iterator and time tracker
        self.next_frame_index = 0
        self.start_time = time.time()  # we use this to sanitise our timestamps
        self.last_frame_wallclock_time = None

    def run_and_analyze_until_stopped(self):
        self.camera.start_recording(self.analysis, format="yuv")
        while not self.stop:
            self.camera.wait_recording(0.001)  # s
        self.camera.stop_recording()

     def trigger_fluorescence_image_capture(self, trigger_time_s):
        """Triggers both the laser and fluorescence camera (assumes edge trigger mode by default) at the specified future time.
        IMPORTANT: this function call is a blocking call, i.e. it will not return until the specified delay has elapsed
        and the trigger has been sent. This is probably acceptable for the RPi implementation, but we should be aware
        that this means everything will hang until the trigger is sent. It also means that camera frames may be dropped
        in the meantime. If all is going well, though, the delay should only be a for a couple of frames.

        Function inputs:
                    trigger_time_s = time (in seconds) at which the trigger should be sent
        """

        logger.success(
            "Sending RPi camera trigger at {0:.6f}s".format(trigger_time_s)
        )
        trigger_mode = self.settings["fluorescence_trigger_mode"]
        if trigger_mode == "edge":
            # The fluorescence camera captures an image when it detects a rising edge on the trigger pin
            fp.edge(
                (trigger_time_s - time.time()) * 1e6,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"]["trigger"],
                self.settings["fluorescence_camera_pins"]["SYNC-B"],
            )
        elif trigger_mode == "expose":
            # The fluorescence camera exposes an image for the duration that the trigger pin is high
            fp.pulse(
                (trigger_time_s - time.time()) * 1e6,
                self.settings["fluorescence_exposure_us"],
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"]["trigger"],
            )
        else:
            logger.critical(
                "Ignoring unknown trigger mode {0}".format(trigger_mode)
            )


def run(args, desc):
    """
    Run the Raspberry Pi optical gater, based on a settings.json file

    Params:   raw_args   list    Caller should normally pass sys.argv here
              desc       str     Description to provide as command line help description
    """
    settings = file_optical_gater.load_settings(args, desc)

    logger.success("Initialising gater...")
    analyser = PiOpticalGater(settings=settings, automatic_target_frame_selection=False)

    logger.success("Running server...")
    analyser.run_server()

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_accuracy()
    analyser.plot_running()


if __name__ == "__main__":
    run(sys.argv[1:], "Run optical gater on image data contained in tiff file")
