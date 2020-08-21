"""Main CLI Open Optical Gating System"""

# Python imports
import sys
import json
import time

# Module imports
from loguru import logger

# Raspberry Pi-specific imports
# Picamera
import picamera
from picamera.array import PiYUVAnalysis

# Fastpins module
import fastpins as fp

# Local imports
import open_optical_gating.cli.optical_gater_server as server

logger.remove()
logger.add(sys.stderr, level="SUCCESS")
logger.enable("open_optical_gating")


class PiOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a picam stream.
    """

    def __init__(
        self, camera=None, settings=None, ref_frames=None, ref_frame_period=None
    ):
        """Function inputs:
            camera - the raspberry picam PiCamera object
            settings - a dictionary of settings (see default_settings.json)
        """

        # initialise parent
        super(PiOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period,
        )
        self.setup_camera(camera)
        self.init_hardware()

    def setup_camera(self, camera):
        """Initialise and apply camera-related settings."""
        logger.success("Configuring RPi camera...")
        self.frame_num = self.settings["frame_num"]
        self.width, self.height = camera.resolution
        self.framerate = camera.framerate
        self.camera = camera

    def init_hardware(self):
        """ Function that initialises various hardware I/O interfaces (pins for triggering laser and fluorescence camera) """
        logger.success("Initialising triggering hardware...")

        # Initialises fastpins module
        try:
            logger.debug("Initialising fastpins...")
            fp.init()
        except Exception as inst:
            logger.critical("Error setting up fastpins module. {0}", inst)

        # Sets up laser trigger pin
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
                self.trigger_mode = self.settings["fluorescence_trigger_mode"]
                self.duration_us = self.settings["fluorescence_exposure_us"]
            except Exception as inst:
                logger.critical("Error setting up fluorescence camera pins. {0}", inst)

    def trigger_fluorescence_image_capture(self, delay_us):
        """ Triggers both the laser and fluorescence camera (assumes edge trigger mode by default) at the specified future time.
            IMPORTANT: this function call is a blocking call, i.e. it will not return until the specified delay has elapsed
            and the trigger has been sent. This is probably acceptable for the RPi implementation, but we should be aware
            that this means everything will hang until the trigger is sent. It also means that camera frames may be dropped
            in the meantime. If all is going well, though, the delay should only be a for a couple of frames.

            Function inputs:
         		delay_us = delay time (in microseconds) before the image is captured
        
            Relevant class variables:
              # TODO: JT writes: these should probably be in a RPi-specific settings dictionary,
              # and the different dictionary entries should be documented somewhere suitable
          		laser_trigger_pin         the GPIO pin number (int) of the laser trigger
          		fluorescence_camera_pins  a dict containg the "trigger", "SYNC-A" and "SYNC-B" pin numbers for the fluorescence camera
          		edge_trigger              [see inline comments below at point of use]
          		duration_us               the duration (in microseconds) of the pulse (only applies to pulse mode [edge_trigger=False])
        """

        logger.success(
            "Sending RPi camera trigger after delay of {0:.1f}us".format(delay_us)
        )
        if self.trigger_mode == "edge":
            # The fluorescence camera captures an image when it detects a rising edge on the trigger pin
            fp.edge(
                delay_us,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"]["trigger"],
                self.settings["fluorescence_camera_pins"]["SYNC-B"],
            )
        elif self.trigger_mode != "expose":
            # The fluorescence camera exposes an image for the duration that the trigger pin is high
            fp.pulse(
                delay_us,
                self.duration_us,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"]["trigger"],
            )
        else:
            logger.critical(
                "Ignoring unknown trigger mode {0}".format(self.trigger_mode)
            )


def run(settings):
    """Run the software using a live PiCam."""

    # Initialise PiCamera
    logger.success("Initialising camera...")
    # Camera settings
    camera = picamera.PiCamera()
    camera.framerate = settings["brightfield_framerate"]
    camera.resolution = (
        settings["brightfield_resolution"],
        settings["brightfield_resolution"],
    )
    camera.awb_mode = settings["awb_mode"]
    camera.exposure_mode = settings["exposure_mode"]
    camera.shutter_speed = settings["shutter_speed_us"]  # us
    camera.image_denoise = settings["image_denoise"]

    # Initialise Gater
    logger.success("Initialising gater...")
    analyser = PiOpticalGater(camera=camera, settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state != "sync":
        camera.start_recording(analyser, format="yuv")
        while not analyser.stop:
            camera.wait_recording(0.001)  # s
        camera.stop_recording()
        # logger.info("Requesting user input...")
        # analyser.user_select_period(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["reference_period"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Running...")
    camera.start_recording(analyser, format="yuv")
    time_started = time.time()  # so we can time-out
    time_running = 0
    while not analyser.stop and time_running < settings["analyse_time_s"]:
        camera.wait_recording(0.001)  # s
        time_running = time.time() - time_started
    camera.stop_recording()

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_accuracy()
    analyser.plot_running()

    logger.success("Fin.")


if __name__ == "__main__":
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Performs a live data capture
    run(settings)
