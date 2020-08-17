"""Extension of CLI Open Optical Gating System for Emulation"""

# Python imports
import sys
import json

# Module imports
from loguru import logger
from skimage import io

# Local imports
import open_optical_gating.cli.cli as cli


class OpticalGater(cli.OpticalGater):
    """Custom class to extend optical gater for emulating from a data file.
    """

    def __init__(
        self, file=None, settings=None, ref_frames=None, ref_frame_period=None
    ):
        """Function inputs:
            camera - the raspberry picam PiCamera object
            settings - a dictionary of settings (see default_settings.json)
        """

        # initialise parent
        super(OpticalGater, self).__init__(
            camera=file,
            settings=settings,
            ref_frames=ref_frames,
            ref_frame_period=ref_frame_period,
        )
        self.emulate_frame = (
            -1
        )  # we start at -1 to avoid an extra variable in next_frame()

    def load_data(self, filename):
        """Apply data source-related settings."""
        # Data-source settings
        logger.success("Loading data instead...")
        self.frame_num = self.settings["frame_num"]
        self.data = io.imread(filename)
        self.width, self.height = self.data[0].shape
        self.framerate = self.settings["brightfield_framerate"]

    def init_hardware(self):
        """As this is an emulator, this function just outputs a log and returns a success."""
        logger.success("Hardware would be initialised now.")
        return 0  # default return

    def next_frame(self):
        """This function gets the next frame from the data source, which can be passed to analyze()"""
        self.emulate_frame = self.emulate_frame + 1
        if self.emulate_frame + 1 == self.data.shape[0]:
            ## if this is our last frame we set the stop flag for the user/app to know
            self.stop = True
        return self.data[self.emulate_frame, :, :]

    def trigger_fluorescence_image_capture(self, delay):
        """As this is an emulator, this function just outputs a log that a trigger would have been sent."""
        logger.success("A fluorescence image would be triggered now.")


def run(settings):
    """Emulated data capture for a set of sample brightfield frames."""
    logger.success("Initialising gater...")
    analyser = OpticalGater(file=settings["path"], settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state > 0:
        while not analyser.stop:
            analyser.analyze(analyser.next_frame())
        logger.info("Requesting user input...")
        analyser.state = analyser.select_period(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["referencePeriod"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Emulating...")
    while not analyser.stop:
        analyser.analyze(analyser.next_frame())

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

    # Performs an emulated data capture
    run(settings)
