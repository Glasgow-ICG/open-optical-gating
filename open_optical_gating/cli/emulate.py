"""Extension of CLI Open Optical Gating System for Emulation"""

# Python imports
import sys
import json

# Module imports
from loguru import logger
from skimage import io

# Local imports
import open_optical_gating.cli.optical_gater_server as server


class FileOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a data file.
    """

    def __init__(
        self, source=None, settings=None, ref_frames=None, ref_frame_period=None
    ):
        """Function inputs:
            source - the raspberry picam PiCamera object
            settings - a dictionary of settings (see default_settings.json)
        """

        # initialise parent
        super(FileOpticalGater, self).__init__(
            source=source,
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

    def next_frame(self):
        """This function gets the next frame from the data source, which can be passed to analyze()"""
        self.emulate_frame = self.emulate_frame + 1
        if self.emulate_frame + 1 == self.data.shape[0]:
            ## if this is our last frame we set the stop flag for the user/app to know
            self.stop = True
        return self.data[self.emulate_frame, :, :]


def run(settings):
    """Emulated data capture for a set of sample brightfield frames."""
    logger.success("Initialising gater...")
    analyser = FileOpticalGater(source=settings["path"], settings=settings,)

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
