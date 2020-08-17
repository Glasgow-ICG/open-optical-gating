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

# TODO there are one or two places where settings should be updated, e.g. user-bound limits
# TODO create a time-stamped copy of the settings file after this
# TODO create a time-stamped log somewhere


class PiOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a picam stream.
    """

    def __init__(
        self, source=None, settings=None, ref_frames=None, ref_frame_period=None
    ):
        """Function inputs:
            source - the raspberry picam PiCamera object
            settings - a dictionary of settings (see default_settings.json)
        """

        # initialise parent
        super(PiOpticalGater, self).__init__(
            source=source,
            settings=settings,
            ref_frames=ref_frames,
            ref_frame_period=ref_frame_period,
        )
        self.emulate_frame = (
            -1
        )  # we start at -1 to avoid an extra variable in next_frame()

    def setup_camera(self, camera):
        """Initialise and apply camera-related settings."""
        self.frame_num = self.settings["frame_num"]
        self.width, self.height = camera.resolution
        self.framerate = camera.framerate
        self.camera = camera

    def init_hardware(self):
        """Function that initialises various hardware I/O interfaces (pins for triggering laser and fluorescence camera)
        Function inputs:
            laser_trigger_pin = the GPIO pin number connected to fire the laser
            fluorescence_camera_pins = an array of 3 pins used to interact with the fluoresence camera
                                            (trigger, SYNC-A, SYNC-B)
        When this function return, self.hardware_state will be one of:
            0 - if no failure
            1 - if fastpins fails
            2 - if laser pin fails
            3 - if camera pins fail
        """
        # TODO update this docstring
        self.hardware_state = None  # default - carry on as if everything else worked

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
        # TODO: JT writes: why not make these a [sub-]dictionary, so we don't have to keep track of what [0], [1], and [2] represent?
        fluorescence_camera_pins = self.settings[
            "fluorescence_camera_pins"
        ]  # Trigger, SYNC-A, SYNC-B
        if fluorescence_camera_pins is not None:
            try:
                logger.debug("Initialising fluorescence camera pins...")
                fp.setpin(fluorescence_camera_pins[0], 1, 0)  # Trigger
                fp.setpin(fluorescence_camera_pins[1], 0, 0)  # SYNC-A
                fp.setpin(fluorescence_camera_pins[2], 0, 0)  # SYNC-B
                self.duration = 1e3  # TODO relate to settings
                self.edge_trigger = True  #  TODO relate to settings     # TODO: JT writes: flagging this again as definitely belonging in settings!
            except Exception as inst:
                logger.critical("Error setting up fluorescence camera pins. {0}", inst)

    def trigger_fluorescence_image_capture(self, delay):
        """Triggers both the laser and fluorescence camera (assumes edge trigger mode by default) at a future time

        # Function inputs:
        # 		delay = delay time (in microseconds) before the image is captured
        # 		laser_trigger_pin = the pin number (int) of the laser trigger
        # 		fluorescence_camera_pins = an int array containg the triggering, SYNC-A and SYNC-B pin numbers for the fluorescence camera
        #
        # Optional inputs:    # TODO: JT writes: this is not an optional input, it is a class variable (which should really be a 'settings' element)
        # 		edge_trigger:
        # 			True = the fluorescence camera captures the image once detecting the start of an increased signal
        # 			False = the fluorescence camera captures for the duration of the signal pulse (pulse mode)
        # 		duration = (only applies to pulse mode [edge_trigger=False]) the duration (in microseconds) of the pulse
        # TODO: ABD add some logging here
        """

        # TODO: JT writes: needs better comment to explain what these modes are. What is "trigger mode" vs "edge mode"?
        # TODO: JT writes: really important to clarify that these calls to fp are *blocking* calls, i.e. they only return after the trigger has been sent
        # Captures an image in edge mode
        if self.edge_trigger:
            fp.edge(
                delay,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"][0],
                self.settings["fluorescence_camera_pins"][2],
            )
        # Captures in trigger mode
        else:
            fp.pulse(
                delay,
                self.duration,
                self.settings["laser_trigger_pin"],
                self.settings["fluorescence_camera_pins"][0],
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
    camera.shutter_speed = settings["shutter_speed"]  # us
    camera.image_denoise = settings["image_denoise"]

    # Initialise Gater
    logger.success("Initialising gater...")
    analyser = PiOpticalGater(source=camera, settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state > 0:
        camera.start_recording(analyser, format="yuv")
        while not analyser.stop:
            camera.wait_recording(0.001)  # s
        camera.stop_recording()
        logger.info("Requesting user input...")
        analyser.state = analyser.select_period(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["referencePeriod"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Running...")
    camera.start_recording(analyser, format="yuv")
    time_started = time.time()  # so we can time-out
    time_running = 0
    while not analyser.stop and time_running < settings["analyse_time"]:
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
