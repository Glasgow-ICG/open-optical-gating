"""Extension of CLI Open Optical Gating System for Emulation"""

# Python imports
import sys
import json
import time

# Module imports
from loguru import logger
from skimage import io

# Local imports
import open_optical_gating.cli.pixelarray as pa
import open_optical_gating.cli.optical_gater_server as server


class FileOpticalGater(server.OpticalGater):
    """Extends the optical gater server for a data file.
    """

    def __init__(
        self,
        file_source_path=None,
        settings=None,
        ref_frames=None,
        ref_frame_period=None,
    ):
        """Function inputs:
            file_source_path   str   Path to file we will read as our input
            settings           dict  Parameters affecting operation (see default_settings.json)
        """

        # initialise parent
        super(FileOpticalGater, self).__init__(
            settings=settings, ref_frames=ref_frames, ref_frame_period=ref_frame_period,
        )
        self.load_data(file_source_path)
        self.next_frame_index = 0

    def load_data(self, filename):
        """Apply data source-related settings."""
        # Data-source settings
        logger.success("Loading image data...")
        self.frame_num = self.settings["frame_num"]
        self.data = io.imread(filename)
        self.height, self.width = self.data[0].shape
        self.framerate = self.settings["brightfield_framerate"]
        self.start_time = time.time()  # we use this to sanitise our timestamps

    def next_frame(self, force_framerate=False):
        """This function gets the next frame from the data source, which can be passed to analyze()"""
        # Force framerate to match the brightfield_framerate in the settings
        # This gives accurate timings and plots
        if force_framerate and (
            len(self.frame_history) > 0
        ):  # only do once we have frames
            while (
                time.time()  # now in seconds
                - self.start_time  # when we started in seconds; used to sanitise
                - self.frame_history[-1].metadata["timestamp"]  # last timestamp
            ) < (1 / self.settings["brightfield_framerate"]):
                time.sleep(1e-5)
                continue
        if self.next_frame_index == self.data.shape[0] - 1:
            ## if this is our last frame we set the stop flag for the user/app to know
            self.stop = True
        next = pa.PixelArray(
            self.data[self.next_frame_index, :, :],
            metadata={
                "timestamp": time.time() - self.start_time
            },  # relative to start_time to sanitise
        )
        self.next_frame_index += 1
        return next


def run(settings):
    """Emulated data capture for a set of sample brightfield frames."""
    logger.success("Initialising gater...")
    analyser = FileOpticalGater(file_source_path=settings["path"], settings=settings,)

    logger.success("Determining reference period...")
    while analyser.state != "sync":
        while not analyser.stop:
            analyser.analyze_pixelarray(analyser.next_frame(force_framerate=True))
        logger.info("Requesting user input...")
        analyser.user_select_period(10)
    logger.success(
        "Period determined ({0} frames long) and user has selected frame {1} as target.",
        analyser.pog_settings["reference_period"],
        analyser.pog_settings["referenceFrame"],
    )

    logger.success("Emulating...")
    while not analyser.stop:
        analyser.analyze_pixelarray(analyser.next_frame(force_framerate=True))

    logger.success("Plotting summaries...")
    analyser.plot_triggers()
    analyser.plot_prediction()
    analyser.plot_accuracy()
    analyser.plot_running()

    logger.success("Fin.")


if __name__ == "__main__":
    t = time.time()
    # Reads data from settings json file
    if len(sys.argv) > 1:
        settings_file = sys.argv[1]
    else:
        settings_file = "settings.json"

    with open(settings_file) as data_file:
        settings = json.load(data_file)

    # Performs an emulated data capture
    run(settings)
    print(time.time() - t)
